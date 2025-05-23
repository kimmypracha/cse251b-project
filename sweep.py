import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import tqdm
import wandb

from trajectory_dataset import TrajectoryDatasetTrain
from model.linear import LinearRegressionModel
from model.mlp    import MLP
from model.cnn    import CNN
from model.lstm   import LSTM

# pick device
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Silicon GPU")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA GPU")
else:
    device = torch.device('cpu')

def load_dataset(batch_size, scale=7.0):
    print("Loading data")
    train_npz = np.load('./data/train.npz')
    train_npz = train_npz["data"]
    print("Done")
    N = len(train_npz)
    val_size = int(0.1 * N)
    train_data = train_npz[:-val_size]
    val_data   = train_npz[-val_size:]

    train_ds = TrajectoryDatasetTrain(train_data, scale=scale, augment=True)
    val_ds   = TrajectoryDatasetTrain(val_data,   scale=scale, augment=False)

    collate = lambda batch: Batch.from_data_list(batch)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader
model_name = None 
def train():
    # initialize W&B run
    wandb.init()
    config = wandb.config

    # --- select model ---
    if model_name == "linear_regression":
        model = LinearRegressionModel().to(device)
    elif model_name == "mlp":
        model = MLP(input_features=50*50*6,
                    output_features=60*2,
                    num_layers=config.num_layers
                   ).to(device)
    elif model_name == "cnn":
        model = CNN(input_channels=6,
                    output_channels=2,
                    num_conv_blocks=config.num_conv_blocks,
                    num_fc_blocks=config.num_fc_blocks,
                    max_pooling=config.max_pooling
                   ).to(device)
    elif model_name == "lstm":
        model = LSTM(input_dim=6,
                     output_dim=60*2,
                     hidden_dim=config.hidden_dim,
                     num_layers=config.num_layers
                    ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # --- optimizer & scheduler from sweep config ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # total steps = epochs * steps_per_epoch
    train_loader, val_loader = load_dataset(config.batch_size)
    total_steps = config.num_epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        cycle_momentum=False
    )

    criterion = nn.MSELoss()
    best_val = float('inf')
    patience = 0

    # --- training loop driven by config.num_epochs ---
    for epoch in tqdm.tqdm(range(config.num_epochs), desc="Epoch"):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            y    = batch.y.view(batch.num_graphs, 60, 2)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        # validation
        model.eval()
        val_loss = val_mae = val_mse = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred  = model(batch)
                y_true = batch.y.view(batch.num_graphs, 60, 2)

                val_loss += criterion(pred, y_true).item()

                # unnormalize for MAE/MSE
                pred = pred * batch.scale.view(-1,1,1) + batch.origin.unsqueeze(1)
                y    = y_true * batch.scale.view(-1,1,1) + batch.origin.unsqueeze(1)
                val_mae += nn.L1Loss()(pred, y).item()
                val_mse += nn.MSELoss()(pred, y).item()

        # averages
        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        val_mae    /= len(val_loader)
        val_mse    /= len(val_loader)

        scheduler.step()

        # log to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_mae":    val_mae,
            "val_mse":    val_mse,
            "epoch":      epoch,
            "lr":         optimizer.param_groups[0]['lr']
        })

        tqdm.tqdm.write(
            f"Epoch {epoch:03d} | lr {optimizer.param_groups[0]['lr']:.2e} | "
            f"train MSE {train_loss:.4f} | val MSE {val_loss:.4f} | "
            f"val MAE {val_mae:.4f} | val MSE (unnorm) {val_mse:.4f}"
        )

        # early stopping on normalized val_loss
        if val_loss < best_val - 1e-3:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience += 1
            if patience >= 10:
                print("Early stopping.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="trajectory_prediction")
    # parser.add_argument("--entity",  type=str, default="your_wandb_entity")
    parser.add_argument("--model",   type=str, default="linear_regression",
                        choices=["linear_regression", "mlp", "cnn", "lstm"])
    args = parser.parse_args()

    # Define a generic sweep; we'll override per-model below.
    base_sweep = {
        'method': 'random',
        'metric': {'goal': 'minimize', 'name': 'val_loss'},
        'parameters': {
            'batch_size':    {'values': [32, 64, 128]},
            'learning_rate': {'min': 1e-5, 'max': 1e-2},
            'weight_decay':  {'min': 1e-5, 'max': 1e-2},
            'num_epochs':    {'values': [50, 100]},
        }
    }
    model_name = args.model
    # model-specific additions
    if args.model == "mlp":
        base_sweep['parameters'].update({
            'num_layers': {'values': [2, 3, 4]},
        })
    if args.model == "cnn":
        base_sweep['parameters'].update({
            'num_conv_blocks': {'values': [2, 3]},
            'num_fc_blocks':   {'values': [1, 2]},
            'max_pooling':     {'values': [True, False]},
        })
    elif args.model == "lstm":
        base_sweep['parameters'].update({
            'hidden_dim': {'values': [64, 128, 256]},
            'num_layers': {'values': [1, 2, 3]},
        })

    sweep_id = wandb.sweep(
        sweep   = { **base_sweep},
        project = args.project,
        # entity  = args.entity
    )
    wandb.agent(sweep_id, function=train, count=1)
    # train_loader, val_loader = load_dataset(32)

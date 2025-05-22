import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

import tqdm
import wandb
import argparse
from trajectory_dataset import TrajectoryDatasetTrain, TrajectoryDatasetTest
from model.linear import LinearRegressionModel
from model.mlp import MLP
from model.cnn import CNN
from model.lstm import LSTM
# Set device for training speedup
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Silicon GPU")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA GPU")
else:
    device = torch.device('cpu')

model_name = None
args = None
    
def load_dataset():
    train_npz = np.load('./data/train.npz')
    train_data = train_npz['data']
    test_npz  = np.load('./data/test_input.npz')
    test_data  = test_npz['data']

    X_train = train_data[..., :50, :]
    Y_train = train_data[:, 0, 50:, :2]

    torch.manual_seed(251)
    np.random.seed(42)

    scale = 7.0

    N = len(train_data)
    val_size = int(0.1 * N)
    train_size = N - val_size

    train_dataset = TrajectoryDatasetTrain(train_data[:train_size], scale=scale, augment=True)
    val_dataset = TrajectoryDatasetTrain(train_data[train_size:], scale=scale, augment=False)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: Batch.from_data_list(x))
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: Batch.from_data_list(x))

    return train_dataloader, val_dataloader

    



def train():

    if model_name == "linear_regression":
        model = LinearRegressionModel().to(device)
    elif model_name == "mlp":
        model = MLP(input_dim=50 * 50 * 2, output_dim=60 * 2, num_conv_blocks=2).to(device)
    elif model_name == "ccn":
        model = CNN(input_dim=50 * 50 * 2, output_dim=60 * 2, num_conv_blocks=2).to(device)
    elif model_name == "lstm":
        model = LSTM(input_dim=50 * 50 * 2, output_dim=60 * 2, num_conv_blocks=2).to(device)
    

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.25) # You can try different schedulers
    
    early_stopping_patience = 10
    best_val_loss = float('inf')
    no_improvement = 0
    criterion = nn.MSELoss()

    train_dataloader, val_dataloader = load_dataset()

    EPOCHS = 100
    for epoch in tqdm.tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
        # ---- Training ----
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            batch = batch.to(device)
            pred = model(batch)
        
            y = batch.y.view(batch.num_graphs, 60, 2)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
        
        # ---- Validation ----
        model.eval()
        val_loss = 0
        val_mae = 0
        val_mse = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(device)
                pred = model(batch)
                y = batch.y.view(batch.num_graphs, 60, 2)
                val_loss += criterion(pred, y).item()

                # show MAE and MSE with unnormalized data
                pred = pred * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                y = y * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                val_mae += nn.L1Loss()(pred, y).item()
                val_mse += nn.MSELoss()(pred, y).item()
        
        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        val_mae /= len(val_dataloader)
        val_mse /= len(val_dataloader)
        scheduler.step()
        # scheduler.step(val_loss)
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_mse": val_mse,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr'],
        })

        tqdm.tqdm.write(f"Epoch {epoch:03d} | Learning rate {optimizer.param_groups[0]['lr']:.6f} | train normalized MSE {train_loss:8.4f} | val normalized MSE {val_loss:8.4f}, | val MAE {val_mae:8.4f} | val MSE {val_mse:8.4f}")
        if val_loss < best_val_loss - 1e-3:
            best_val_loss = val_loss
            no_improvement = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                print("Early stop!")
                break


if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--project", type=str, default="linear_regression")
    args.add_argument("--entity", type=str, default="your_wandb_entity")
    args.add_argument("--model", type=str, default="linear_regression")

    args = args.parse_args()

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "lr":               {"min": 1e-5, "max": 1e-2},
            "optimizer":        {"values": ["adam", "sgd", "adamw", "rmsprop"]},
            "epochs":           {"values": [10, 20, 30]},
        },
    }

    sweep_id = wandb.sweep(
        sweep   = sweep_config,
        project = args.project,
        entity  = args.entity
    )
    if args.model == "linear_regression":
        wandb.init(
            project = args.project,
            entity  = args.entity,
            config  = {
                "lr":               1e-3,
                "optimizer":        "adam",
                "epochs":           100,
            }
        )
    elif args.model == "mlp":
        wandb.init(
            project = args.project,
            entity  = args.entity,
            config  = {
                "lr":               1e-3,
                "batch_size":       32,
                "num_conv_blocks":  2,
                "optimizer":        "adam",
                "epochs":           100,
            }
        )
    elif args.model == "cnn" : 
        wandb.init(
            project = args.project,
            entity  = args.entity,
            config  = {
                "lr":               1e-3,
                "batch_size":       32,
                "num_conv_blocks":  2,
                "optimizer":        "adam",
                "epochs":           100,
            }
        )
    elif args.model == "lstm" : 
        wandb.init(
            project = args.project,
            entity  = args.entity,
            config  = {
                "lr":               1e-3,
                "batch_size":       32,
                "num_conv_blocks":  2,
                "optimizer":        "adam",
                "epochs":           100,
            }
        )
    # Launch agents — each one runs sweep_train()
    wandb.agent(sweep_id, function=train)
        

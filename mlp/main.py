# mlp.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

INPUT_FEATURES  = 6 * 50 * 50
OUTPUT_FEATURES = 2 * 60
DATA_DIR        = "../data"

class MLP(pl.LightningModule):
    def __init__(self, 
                 lr: float = 1e-3,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_FEATURES, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024,           512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512,            256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, OUTPUT_FEATURES)
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def make_dataloaders(batch_size: int):
    # load files
    train_file = np.load(f"{DATA_DIR}/train.npz")["data"]
    #  shape: (N, T, 50, 6)  → split into x,y
    N = len(train_file)
    split = int(0.8 * N)
    x = train_file[..., :50, :].reshape(N, -1)
    y = train_file[:, 0, 50:, :2]  # (N,60,2)

    x_train, y_train = x[:split], y[:split].reshape(split, -1)
    x_val,   y_val   = x[split:], y[split:].reshape(N - split, -1)

    train_ds = TensorDataset(torch.from_numpy(x_train).float(),
                             torch.from_numpy(y_train).float())
    val_ds   = TensorDataset(torch.from_numpy(x_val).float(),
                             torch.from_numpy(y_val).float())

    return (
      DataLoader(train_ds, batch_size=batch_size, shuffle=True),
      DataLoader(val_ds,   batch_size=batch_size)
    )


if __name__ == "__main__":
    import argparse
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=1)
    args = parser.parse_args()

    wandb_logger = WandbLogger(project="mlp")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min"
    )

    model = MLP()
    train_loader, val_loader = make_dataloaders(args.batch_size)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    trainer.fit(model, train_loader, val_loader)
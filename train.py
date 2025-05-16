import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np


class MLP(pl.LightningModule):
    def __init__(self, input_features, output_features, lr=1e-3):
        super().__init__()
        # save hyperparameters to self.hparams automatically
        self.save_hyperparameters()

        # model architecture
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 512),          nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256),           nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, output_features)
        )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        print("t:y_hat's shape", y_hat.shape)
        print("t:y's shape", y.shape)

        loss = self.criterion(y_hat, y)

        # log to both progress bar and wandb
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        print("v:y_hat's shape", y_hat.shape)
        print("v:y's shape", y.shape)
        loss = self.criterion(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
def make_dataloaders(x_train, y_train, x_val, y_val, batch_size, input_features, output_features):
    train_ds = TensorDataset(
        torch.FloatTensor(x_train).view(-1, input_features),
        torch.FloatTensor(y_train).view(-1, output_features)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(x_val).view(-1, input_features),
        torch.FloatTensor(y_val).view(-1, output_features)
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size)
    )

if __name__ == "__main__":
    import argparse
    import wandb

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--project", type=str, default="ml_experiments")
    args = parser.parse_args()
    input_features = 50 * 50 * 6  # = 5000
    output_features = 60 * 2
    # initialise wandb run
    wandb_logger = WandbLogger(
        project=args.project,
        config=vars(args)
    )
    train_file = np.load('data/train.npz')
    train_data = train_file['data']
    print("train_data's shape", train_data.shape)
    test_file = np.load('data/test_input.npz')
    test_data = test_file['data']
    print("test_data's shape", test_data.shape)
    # replace these with your own data splits
    train_len = int(0.8 * len(train_data))
    val_len = len(train_data) - train_len

    x_train, y_train = train_data[:train_len, :, :50, :], train_data[:train_len, 0, 50:, :2]
    x_val,   y_val   = train_data[train_len:, :, :50, :], train_data[train_len:, 0, 50:, :2]

    train_loader, val_loader = make_dataloaders(
        x_train, y_train, x_val, y_val, args.batch_size, 
        input_features=input_features, output_features=output_features
    )

    model = MLP(
        input_features=input_features,
        output_features=output_features,
        lr=args.lr
    )

    # callbacks for checkpointing & early stopping
    ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[ckpt, early_stop],
        max_epochs=args.max_epochs,
    )

    trainer.fit(model, train_loader, val_loader)
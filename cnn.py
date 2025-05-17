import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

class CNN(pl.LightningModule):
    def __init__(self, input_features, output_features, lr=1e-3, num_conv_blocks=2, optimizer="adam"):
        super().__init__()
        # save hyperparameters to self.hparams automatically
        self.save_hyperparameters()

        # model architecture
        # self.model = nn.Sequential(
        #     nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Flatten(),
        #     nn.Linear(32 * 12 * 12, output_features)
        # )
        # repeat the convolutional block num_conv_blocks times
        conv_blocks = []
        in_channels = 6
        out_channels = 16
        kernel_size = 3
        stride = 1
        padding = 1
        # calculate the output size after each convolutional block
        out_size = 50
        self.optimizer_name = optimizer
        for _ in range(num_conv_blocks):
            conv_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            conv_blocks.append(nn.ReLU())
            conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels *= 2
            out_size = (out_size - kernel_size + 2 * padding) // stride + 1
            out_size = (out_size - 2) // 2 + 1

        self.model = nn.Sequential(
            *conv_blocks,
            nn.Flatten(),
            nn.Linear(in_channels * out_size * out_size, output_features)
        )


        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        # log to both progress bar and wandb
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.optimizer_name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        elif self.optimizer_name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, ... )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        # }

    # def validation_epoch_end(self, outputs):
    #     lr = self.trainer.optimizers[0].param_groups[0]["lr"]
    #     self.log("lr", lr, prog_bar=True)

def make_dataloaders_cnn(x_train, y_train, x_val, y_val, batch_size, input_features, output_features):
    train_ds = TensorDataset(
        torch.FloatTensor(x_train).view(-1, 6, 50, 50),
        torch.FloatTensor(y_train).view(-1, output_features)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(x_val).view(-1, 6, 50, 50),
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_conv_blocks", type=int, default=2)
    parser.add_argument("--project", type=str, default="ml_experiments")
    parser.add_argument("--optimizer", type=str, default="adam")
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

    train_loader, val_loader = make_dataloaders_cnn(
        x_train, y_train, x_val, y_val, args.batch_size, 
        input_features=input_features, output_features=output_features
    )
    model = CNN(
        input_features=input_features,
        output_features=output_features,
        lr=args.lr,
        num_conv_blocks=args.num_conv_blocks,
        optimizer=args.optimizer
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
        max_epochs=args.epochs,
    )

    trainer.fit(model, train_loader, val_loader)
# cnn_sweep.py

import argparse
import wandb
import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from main import CNN, make_dataloaders_cnn

DEFAULT_PROJECT = "cnn_experiments"

def sweep_train():
    # Called once per agent/trial
    wandb.init()
    cfg = wandb.config

    # --- load & split data ---
    data = np.load("../data/train.npz")["data"]
    split = int(0.8 * len(data))
    x_train = data[:split, :, :50, :]
    y_train = data[:split, 0, 50:, :2]
    x_val   = data[split:, :, :50, :]
    y_val   = data[split:, 0, 50:, :2]

    # --- dataloaders ---
    train_loader, val_loader = make_dataloaders_cnn(
        x_train, y_train,
        x_val,   y_val,
        batch_size    = cfg.batch_size,
        input_features = 6 * 50 * 50,
        output_features= 2 * 60
    )

    # --- model ---
    model = CNN(
        input_features = 6 * 50 * 50,
        output_features= 2 * 60,
        lr             = cfg.lr,
        num_conv_blocks = cfg.num_conv_blocks,
        optimizer      = cfg.optimizer
    )

    # --- logging & callbacks ---
    wandb_logger = WandbLogger(project=DEFAULT_PROJECT, config=cfg)
    ckpt = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    early = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = pl.Trainer(
        logger      = wandb_logger,
        callbacks   = [ckpt, early],
        max_epochs  = cfg.epochs,
    )
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--entity",  type=str, default=None)
    args = parser.parse_args()

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "lr":               {"min": 1e-5, "max": 1e-2},
            "batch_size":       {"values": [16, 32, 64, 128]},
            "num_conv_blocks":  {"values": [2, 3]},
            "optimizer":        {"values": ["adam", "sgd", "adamw", "rmsprop"]},
            "epochs":           {"values": [10, 20, 30]},
        },
    }

    sweep_id = wandb.sweep(
        sweep   = sweep_config,
        project = args.project,
        entity  = args.entity
    )
    # Launch agents — each one runs sweep_train()
    wandb.agent(sweep_id, function=sweep_train)

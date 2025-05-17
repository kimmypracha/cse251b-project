# mlp_sweep.py

import argparse
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from main import MLP, make_dataloaders

SLEEPER_PROJECT = "mlp_experiments"

def sweep_train():
    # each agent run
    wandb.init()
    cfg = wandb.config

    train_loader, val_loader = make_dataloaders(cfg.batch_size)

    model = MLP(lr=cfg.lr)
    wandb_logger = WandbLogger(project=SLEEPER_PROJECT, config=cfg)

    ckpt = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    early = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[ckpt, early],
        max_epochs=cfg.max_epochs
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",    type=str,   default=SLEEPER_PROJECT)
    parser.add_argument("--entity",     type=str,   default=None)
    args = parser.parse_args()

    sweep_config = {
      "method": "bayes",
      "metric": {"name": "val_loss", "goal": "minimize"},
      "parameters": {
        "lr":         {"min": 1e-5, "max": 1e-2},
        "batch_size": {"values": [16, 32, 64, 128]},
        "max_epochs": {"values": [10, 20, 30]},
      }
    }

    sweep_id = wandb.sweep(
      sweep       = sweep_config,
      project     = args.project,
      entity      = args.entity
    )
    wandb.agent(sweep_id, function=sweep_train)

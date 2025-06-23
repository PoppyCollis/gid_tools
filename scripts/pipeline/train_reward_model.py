#!/usr/bin/env python
# File: scripts/pipeline/train_reward_model.py

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
import logging

from gid_tools.helpers.utils import build_reward_dataset
from gid_tools.helpers.plots import plot_reward_mlp_training_loss
from gid_tools.reward_model.reward_mlp import RewardMLP
from gid_tools.helpers.utils import load_config

# === logger setup ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(ch)


def main():
    # parse args
    default_cfg = Path(__file__).resolve().parent / "config.ini"
    parser = argparse.ArgumentParser(
        description="Train a reward model based on features and rewards data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help=f"Path to config file (default: {default_cfg.name})"
    )
    args = parser.parse_args()

    # load config
    cfg = load_config(args.config)
    logger.info(f"Loaded config: {cfg['meta']['config_name']}")

    # get training hyperparams
    train_cfg = cfg["train_reward_model"]
    num_epochs = train_cfg.getint("num_epochs")
    batch_size = train_cfg.getint("batch_size")
    lr         = train_cfg.getfloat("lr")

    # paths
    pipeline_dir  = Path(__file__).resolve().parent
    features_dir  = pipeline_dir / cfg["features"]["directory"]
    rewards_file  = pipeline_dir / cfg["rewards"]["file"]
    root_dir      = pipeline_dir.parents[2]
    output_model  = root_dir / "checkpoints" / "reward_mlp.pth"
    output_model.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Features dir: {features_dir}")
    logger.info(f"Rewards file: {rewards_file}")
    logger.info(f"Output model path: {output_model}")

    # build dataset
    try:
        dataset = build_reward_dataset(features_dir, rewards_file)
    except Exception as e:
        logger.error(f"Failed to build reward dataset: {e}")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model, criterion, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = dataset.tensors[0].shape[1]
    model = RewardMLP(input_dim=input_dim).to(device)
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # training
    avg_losses = []
    std_losses = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        avg = float(np.mean(batch_losses))
        std = float(np.std(batch_losses))
        avg_losses.append(avg)
        std_losses.append(std)
        logger.info(f"Epoch {epoch}/{num_epochs} — Mean MSE: {avg:.4f} ± {std:.4f}")

    # plot training loss
    plot_reward_mlp_training_loss(avg_losses, std_losses)

    # final evaluation
    model.eval()
    with torch.no_grad():
        preds_all = model(dataset.tensors[0].to(device))
        final_mse = criterion(preds_all, dataset.tensors[1].to(device)).item()
    logger.info(f"Final training MSE: {final_mse:.4f}")

    # save model
    torch.save(model.state_dict(), output_model)
    logger.info(f"Saved trained model to {output_model}")


if __name__ == "__main__":
    main()

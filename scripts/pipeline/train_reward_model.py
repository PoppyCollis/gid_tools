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
    features_root  = pipeline_dir / cfg["features"]["directory"]
    
    # splits under features/
    features_train_dir = features_root / "train"
    features_test_dir  = features_root / "test"
    rewards_base       = cfg["rewards"]["file"]
    rewards_train_path = pipeline_dir / f"train_{rewards_base}"
    rewards_test_path  = pipeline_dir / f"test_{rewards_base}"
    root_dir           = pipeline_dir.parents[2]
    output_model_path  = root_dir / "checkpoints" / "reward_mlp.pth"
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # build datasets
    try:
        train_dataset = build_reward_dataset(features_train_dir, rewards_train_path)
    except Exception as e:
        logger.error(f"Failed to build train dataset: {e}")
        return
    try:
        test_dataset = build_reward_dataset(features_test_dir, rewards_test_path)
    except Exception as e:
        logger.error(f"Failed to build test dataset: {e}")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model, criterion, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = train_dataset.tensors[0].shape[1]
    model = RewardMLP(input_dim=input_dim).to(device)
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # training
    avg_losses = []
    std_losses = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
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
    
    plot_reward_mlp_training_loss(batch_losses, np.zeros(len(batch_losses)))

    # plot training loss
    plot_reward_mlp_training_loss(avg_losses, std_losses)
    
    # final evaluation
    model.eval()
    with torch.no_grad():
        train_preds = model(train_dataset.tensors[0].to(device))
        train_mse   = criterion(train_preds, train_dataset.tensors[1].to(device)).item()
    logger.info(f"Final train MSE: {train_mse:.4f}")

    # 10) evaluation on test set
    with torch.no_grad():
        test_feats = test_dataset.tensors[0].to(device)
        test_labels= test_dataset.tensors[1].to(device)
        test_preds = model(test_feats)
        test_mse   = criterion(test_preds, test_labels).item()
    logger.info(f"Test MSE: {test_mse:.4f}")

    # 11) save model
    torch.save(model.state_dict(), output_model_path)
    logger.info(f"Saved trained model to {output_model_path}")


if __name__ == "__main__":
    main()

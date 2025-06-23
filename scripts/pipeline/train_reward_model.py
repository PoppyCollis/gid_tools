import argparse
from pathlib import Path
import configparser
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam

from gid_tools.helpers.utils import build_reward_dataset
from gid_tools.reward_model.reward_mlp import RewardMLP
from gid_tools.helpers.plots import plot_reward_mlp_training_loss

def main():
    # Determine default config path (same directory as this script)
    default_cfg = Path(__file__).resolve().parent / 'config.ini'

    p = argparse.ArgumentParser(
        description="Train a reward model based on features and rewards data."
    )
    p.add_argument(
        '--config',
        type=Path,
        default=default_cfg,
        help=f"Path to config.ini (default: {default_cfg})"
    )
    args = p.parse_args()

    # Read configuration
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    num_epochs = cfg.getint('train_reward_model', 'num_epochs')
    batch_size = cfg.getint('train_reward_model', 'batch_size')
    lr = cfg.getfloat('train_reward_model', 'lr')

    # Directory setup
    base_dir = Path(__file__).resolve().parent
    root_dir = base_dir.parents[2]
    features_dir = base_dir / "features"
    rewards_file = base_dir / "rewards.json"
    output_model = root_dir / "checkpoints" / "reward_mlp.pth"

    # Build the reward dataset
    try:
        dataset = build_reward_dataset(features_dir, rewards_file)
    except Exception as e:
        print(f"Failed to build reward dataset: {e}")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = dataset.tensors[0].shape[1]
    model = RewardMLP(input_dim=input_dim).to(device)
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # Training loop
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

        # Epoch statistics
        avg_loss = np.mean(batch_losses)
        std_loss = np.std(batch_losses)
        avg_losses.append(avg_loss)
        std_losses.append(std_loss)

        print(f"Epoch {epoch}/{num_epochs} - Mean MSE: {avg_loss:.4f} ± {std_loss:.4f}")

    # Plot training loss
    plot_reward_mlp_training_loss(avg_losses, std_losses)

    # Final evaluation on entire set
    model.eval()
    with torch.no_grad():
        all_preds = model(dataset.tensors[0].to(device))
        final_mse = criterion(all_preds, dataset.tensors[1].to(device)).item()
    print(f"Final training MSE: {final_mse:.4f}")

    # Save model
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_model)
    print(f"Saved trained model to {output_model}")

if __name__ == "__main__":
    main()
"""
Step 4. Train reward model given dataset D = {ϕ(x), y}
"""
# scripts/pipeline/train_reward_model.py

# scripts/pipeline/train_reward_model.py
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam

from gid_tools.helpers.utils import build_reward_dataset
from gid_tools.reward_model.reward_mlp import RewardMLP


def main():
    base_dir = Path(__file__).resolve().parent
    root_dir = Path(__file__).resolve().parents[2]
    features_dir = base_dir / "features"
    rewards_file = base_dir / "rewards.json"
    output_model = root_dir / "checkpoints" / "reward_mlp.pth"

    # Build the reward dataset
    try:
        dataset = build_reward_dataset(features_dir, rewards_file)
    except Exception as e:
        print(f"Failed to build reward dataset: {e}")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = dataset.tensors[0].shape[1]
    model = RewardMLP(input_dim=input_dim).to(device)
    criterion = MSELoss()
    # optimizer = Adam(model.parameters(), lr=1e-3)
    optimizer = Adam(model.parameters(), lr=0.1)


    # Training loop
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{num_epochs} - Training MSE: {avg_loss:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        all_preds = model(dataset.tensors[0].to(device))
        final_mse = criterion(all_preds, dataset.tensors[1].to(device)).item()
    print(f"Final training MSE: {final_mse:.4f}")

    # Save model weights
    torch.save(model.state_dict(), output_model)
    print(f"Saved trained model to {output_model}")


if __name__ == "__main__":
    main()

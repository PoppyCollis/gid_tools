"""
Step 3. Extract features from generated samples using the U-Net middle-block hook
and compute reward predictions via the LinearRewardModel.
"""
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from gid_tools.reward_model.linear_reward_model import LinearRewardModel
from gid_tools.viz.utils import download_checkpoint


def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Directory paths
    root_dir = Path(__file__).resolve().parents[2]
    samples_dir = Path(__file__).resolve().parent / "samples"
    features_dir = Path(__file__).resolve().parent / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Ensure checkpoint is available
    ckpt_path = download_checkpoint(root_dir)

    # Initialize model
    model = LinearRewardModel(unet_ckpt_path=ckpt_path, device=device)
    model.eval()

    # Preprocessing pipeline: convert to grayscale, tensor, normalize to [-1,1]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Iterate over samples
    for img_file in sorted(samples_dir.glob("*.png")):
        # Load and preprocess image
        img = Image.open(img_file).convert("L")
        x = transform(img).unsqueeze(0).to(device)  # [1,1,H,W]

        # Dummy timestep tensor (zeroed since embedding layers are zeroed)
        t = torch.zeros((1,), dtype=torch.long, device=device)

        with torch.no_grad():
            # Forward pass to populate model._mid_feats
            _ = model.unet(x, t)
            mid_feats = model._mid_feats  # [1, C_mid, H_mid, W_mid]

            # Global average pool
            pooled = mid_feats.mean(dim=(2, 3))  # [1, C_mid]
            # L2 normalize
            pooled_norm = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-6)

            # Compute reward prediction
            reward = model.reward_head(pooled_norm)

        # Convert to numpy and save
        feature_vec = pooled_norm.squeeze(0).cpu().numpy()
        reward_val = reward.cpu().item()

        out_path = features_dir / f"{img_file.stem}_features.npz"
        np.savez(out_path, features=feature_vec, reward=reward_val)

        print(f"Saved features and reward for {img_file.name} to {out_path}")

    print("Done extracting features.")


if __name__ == "__main__":
    main()

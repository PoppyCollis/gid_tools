#!/usr/bin/env python
# File: scripts/pipeline/feature_extraction.py

"""
Step 3. Extract features from generated samples using the U-Net middle-block hook
and compute reward predictions via the modular RewardModel (extractor + MLP head).
"""

import argparse
from pathlib import Path
import torch
import logging

from gid_tools.reward_model.feature_extractor import UnetFeatureExtractor
from gid_tools.reward_model.reward_mlp import RewardMLP
from gid_tools.reward_model.reward_model import RewardModel
from gid_tools.helpers.utils import download_checkpoint, load_config

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
        description="Extract U-Net features and compute rewards from samples.pt"
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

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Paths from config
    pipeline_dir = Path(__file__).resolve().parent
    sample_dir   = pipeline_dir / cfg["samples"]["directory"]
    tensor_path  = sample_dir / cfg["samples"]["tensor_file"]
    features_dir = pipeline_dir / cfg["features"]["directory"]
    features_dir.mkdir(parents=True, exist_ok=True)

    # Download / load UNet checkpoint
    project_root = Path(__file__).resolve().parents[2]
    ckpt_path = download_checkpoint(project_root)

    # --- Modular model setup ---
    extractor = UnetFeatureExtractor(
        unet_ckpt_path=ckpt_path,
        ch=128,    # base channel size; if you need to parameterize it, add to [feature_extraction]
        in_ch=1,
        device=device
    )
    head = RewardMLP(input_dim=2*128).to(device)
    model = RewardModel(feature_extractor=extractor, reward_head=head).to(device)
    model.eval()

    # Load samples tensor
    if not tensor_path.exists():
        logger.error(f"Tensor file not found: {tensor_path}")
        return
    batch = torch.load(tensor_path, map_location=device)  # [B, C, H, W]

    # Dummy timestep for U-Net extractor API
    t = torch.zeros((1,), dtype=torch.long, device=device)

    # Process each image
    for idx, img in enumerate(batch):
        x = img.unsqueeze(0).to(device)  # [1, C, H, W]
        with torch.no_grad():
            feats = extractor.extract(x, t=t)    # [1, D]
            reward = head(feats).squeeze(-1)     # scalar

        out = {
            "features": feats.squeeze(0).cpu(),
            "reward":   reward.cpu(),
        }
        out_path = features_dir / f"sample_{idx}_features.pt"
        torch.save(out, out_path)

    logger.info("All features extracted and saved.")


if __name__ == "__main__":
    main()

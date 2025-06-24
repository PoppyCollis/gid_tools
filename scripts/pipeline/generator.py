#!/usr/bin/env python
"""
Step 1. Load in the diffusion model, generate images and save them to output folder
"""

import torch
import argparse
from pathlib import Path

from gid_tools.diffusion_model.unet import UNet
from gid_tools.diffusion_model.diffusion import DiffusionModel
from gid_tools.helpers.utils import save_samples, download_checkpoint, load_config

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def main():
    # Parse command-line args
    default_cfg = Path(__file__).resolve().parent / "config.ini"
    parser = argparse.ArgumentParser(
        description="Generate samples using the diffusion model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help=f"Path to config file (default: {default_cfg.name})"
    )
    args = parser.parse_args()

    # Load and parse config
    cfg = load_config(args.config)  # returns a ConfigParser already .read() for you
    gen_cfg = cfg["generator"]
    batch_size = gen_cfg.getint("batch_size")

    # Ensure checkpoint is present
    project_root = Path(__file__).resolve().parents[2]
    ckpt_path = download_checkpoint(project_root)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize model + diffusion process
    model = UNet(ch=128, in_ch=1).to(device)
    diffusion = DiffusionModel(T=1000, model=model, device=device)

    # Load weights
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded pretrained weights into UNet.")

    # Sample
    samples = diffusion.sampling(
        n_samples=batch_size,
        image_channels=1,
        img_size=(32, 32),
        use_tqdm=True
    )

    # Save outputs
    out_dir = Path(__file__).resolve().parent / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) PNGs
    save_samples(samples, out_dir, prefix="sample")
    logger.info(f"Saved {batch_size} PNG samples to {out_dir}")

    # 2) Raw tensor batch
    tensor_path = out_dir / "train_samples.pt"
    torch.save(samples.cpu(), tensor_path)
    logger.info(f"Saved raw tensor batch to {tensor_path}")


if __name__ == "__main__":
    main()

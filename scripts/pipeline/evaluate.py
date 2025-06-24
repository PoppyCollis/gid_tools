#!/usr/bin/env python
"""
Step 2. Get reward feedback for images in the output folder
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from gid_tools.envs.feedback import ToolRewardEnv, pixel_area_tensor
from gid_tools.helpers.utils import load_config


def process_split(split: str, samples_dir: Path, tensor_file: str,
                  method: str, threshold: float, output_path: Path):
    """
    Load a tensor batch, compute rewards, and write to JSON.
    """
    tensor_path = samples_dir / tensor_file
    if not tensor_path.exists():
        logger.error(f"[{split}] Tensor file not found: {tensor_path}")
        return

    logger.info(f"[{split}] Loading tensor batch from {tensor_path}")
    batch = torch.load(tensor_path)  # [B, C, H, W]

    env = ToolRewardEnv(default_method=None)
    env.register_reward("pixel_area", pixel_area_tensor)

    rewards = []
    for idx, img in enumerate(batch):
        r = env.compute(img, method=method, threshold=threshold)
        rewards.append(float(r))
        if idx % 1000 == 0:
            logger.debug(f"[{split}] Sample {idx}: {method} → {r}")

    rewards_dict = {str(i): r for i, r in enumerate(rewards)}

    # Write out
    with open(output_path, "w") as f:
        json.dump(rewards_dict, f, indent=2)
    logger.info(f"[{split}] Wrote rewards to {output_path}")


# === logger setup ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(ch)


def main():
    # Parse args
    default_cfg = Path(__file__).resolve().parent / "config.ini"
    parser = argparse.ArgumentParser(
        description="Compute rewards for a batch of samples"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help=f"Path to config file (default: {default_cfg.name})"
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    logger.info(f"Loaded config: {cfg['meta']['config_name']}")

    # Paths from [samples] and [rewards]
    pipeline_dir = Path(__file__).resolve().parent
    
    # Samples section
    sample_cfg  = cfg["samples"]
    samples_dir = pipeline_dir / sample_cfg["directory"]
    train_tensor = sample_cfg["train_tensor_file"]
    test_tensor  = sample_cfg["test_tensor_file"]
    
    # Evaluate options
    eval_cfg   = cfg["evaluate"]
    method     = eval_cfg.get("method", "pixel_area")
    threshold  = eval_cfg.getfloat("threshold", 0.0)
    
    # Rewards section
    rewards_cfg = cfg["rewards"]
    base_name   = rewards_cfg.get("file", "rewards.json")
    # derive per-split output names
    train_out = pipeline_dir / f"train_{base_name}"
    test_out  = pipeline_dir / f"test_{base_name}"
    
     # Process each split
    process_split("train", samples_dir, train_tensor, method, threshold, train_out)
    process_split("test",  samples_dir, test_tensor,  method, threshold, test_out)

if __name__ == "__main__":
    main()

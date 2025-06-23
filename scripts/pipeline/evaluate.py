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
    samples_dir  = pipeline_dir / cfg["samples"]["directory"]
    tensor_path  = samples_dir / cfg["samples"]["tensor_file"]

    rewards_cfg  = cfg["rewards"]
    output_path  = pipeline_dir / rewards_cfg.get("file", "rewards.json")

    # Script-specific options
    eval_cfg     = cfg["evaluate"]
    method       = eval_cfg.get("method", "pixel_area")
    threshold    = eval_cfg.getfloat("threshold", 0.0)

    # Validate inputs
    if not tensor_path.exists():
        logger.error(f"No tensor file found at {tensor_path}")
        raise FileNotFoundError(f"No tensor file found at {tensor_path}")

    # Load batch: [B, C, H, W]
    batch = torch.load(tensor_path)

    # Setup reward environment
    env = ToolRewardEnv(default_method=None)
    env.register_reward("pixel_area", pixel_area_tensor)

    # Compute rewards
    rewards = []
    for idx, img in enumerate(batch):
        r = env.compute(img, method=method, threshold=threshold)
        rewards.append(float(r))
        logger.debug(f"Sample {idx}: {method} → {r}")

    # Map indices → rewards
    rewards_dict = {str(i): r for i, r in enumerate(rewards)}

    # Write out
    with open(output_path, "w") as f:
        json.dump(rewards_dict, f, indent=2)
    logger.info(f"Wrote rewards to {output_path}")


if __name__ == "__main__":
    main()

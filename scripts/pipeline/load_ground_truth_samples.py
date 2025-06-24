#!/usr/bin/env python
"""
Load ground truth tool dataset from JSONL, 
split into train/test, and save each as a raw torch tensor batch.
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from gid_tools.helpers.custom_tool_dataset import DiffusionToolDataset
from gid_tools.helpers.utils import load_config, save_split


def main():
    # ------------------------
    # 1) Parse command‐line args
    # ------------------------
    default_cfg = Path(__file__).resolve().parent / "config_ground_truth.ini"
    parser = argparse.ArgumentParser(
        description="Build train/test tensor batches from a subset of the JSONL ground-truth dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="Which config file to load (default: config_ground_truth.ini)"
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=0.10,
        help="Fraction of the full dataset to keep for train+test (default: 0.10)"
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.10,
        help="Fraction of the subset to hold out as test (default: 0.10 of the subset)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size when assembling tensors (default: 64)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting (default: 42)"
    )
    args = parser.parse_args()
    
    # ------------------------
    # 2) Load config
    # ------------------------
    cfg = load_config(args.config)
    out_dir_name         = cfg["samples"]["directory"]  # e.g. ground_truth_samples
    train_fname          = cfg["samples"]["train_tensor_file"]
    test_fname           = cfg["samples"]["test_tensor_file"]
    pipeline_dir         = Path(__file__).resolve().parent
    

    # ------------------------
    # 3) Paths
    # ------------------------
    ROOT_DIR        = Path(__file__).resolve().parents[2]
    DATASET_PATH    = ROOT_DIR / "gid_tools" / "datasets" / "tools_dataset_classes_reduced.jsonl"
    OUTPUT_DIR      = pipeline_dir / out_dir_name
    SAMPLES_DIR     = pipeline_dir / "samples"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    train_path              = OUTPUT_DIR / train_fname
    test_path               = OUTPUT_DIR / test_fname
    test_path_generated     = SAMPLES_DIR / test_fname

    # Skip if both splits exist
    if train_path.exists() and test_path.exists():
        print("Found existing train & test batches; skipping.")
        return

    # ------------------------
    # 4) Load full dataset
    # ------------------------
    print("Loading full ground-truth dataset...")
    dataset = DiffusionToolDataset(str(DATASET_PATH))
    N = len(dataset)

     # ------------------------
    # 5) First split: subset of size subset_k
    # ------------------------
    subset_k = int(N * args.subset_fraction)
    subset_k = max(1, subset_k)  # at least one
    print(f"Selecting random subset of {subset_k}/{N} samples...")
    subset_ds, _ = random_split(
        dataset,
        [subset_k, N - subset_k],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # ------------------------
    # 6) Second split: train/test from subset
    # ------------------------
    test_k  = int(subset_k * args.test_fraction)
    train_k = subset_k - test_k
    test_k  = max(1, test_k)
    train_k = max(1, train_k)
    print(f"  → From subset, splitting into {train_k} train / {test_k} test")

    train_ds, test_ds = random_split(
        subset_ds,
        [train_k, test_k],
        generator=torch.Generator().manual_seed(args.seed + 1)
    )

    # ------------------------
    # 6) DataLoaders
    # ------------------------
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # ------------------------
    # 7) Execute saves
    # ------------------------
    print("Saving train split...")
    save_split(train_loader, train_path)
    print("Saving test split...")
    save_split(test_loader, test_path)
    print("Copying test split to standard samples directory...")
    save_split(test_loader, test_path_generated)
    print("Done.")

if __name__ == "__main__":
    main()

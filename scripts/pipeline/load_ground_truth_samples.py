"""
Load ground truth tool dataset from JSONL and save as a raw torch tensor batch.
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from gid_tools.helpers.custom_tool_dataset import DiffusionToolDataset


def main():
    # Define paths
    ROOT_DIR = Path(__file__).resolve().parents[2]
    DATASET_PATH = ROOT_DIR / "gid_tools" / "datasets" / "tools_dataset_classes_reduced.jsonl"
    OUTPUT_DIR = Path(__file__).resolve().parent / "ground_truth_samples"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tensor_path = OUTPUT_DIR / "samples.pt"
    
    # Skip if already exists
    if tensor_path.exists():
        print(f"Found existing tensor batch at {tensor_path}, skipping generation.")
        return

    fraction_of_dataset = 0.1
    # Load dataset
    dataset = DiffusionToolDataset(str(DATASET_PATH))
    
    

    N = len(dataset)
    k = int(N * fraction_of_dataset)
    small_ds, _ = random_split(dataset, [k, N - k], generator=torch.Generator().manual_seed(42))

    dataloader = DataLoader(small_ds, batch_size=64, shuffle=False)
    
    print("Loaded ground-truth dataset.")

    # Collect all batches
    all_batches = []
    for batch in tqdm(dataloader, desc="Collecting tensor batches"):
        all_batches.append(batch)

    # Stack and save as a single tensor
    all_samples = torch.cat(all_batches, dim=0)  # Shape: [N, 1, 32, 32]
    tensor_path = OUTPUT_DIR / "samples.pt"
    torch.save(all_samples.cpu(), tensor_path)

    print(f"Saved {all_samples.shape[0]} samples as raw tensor batch to {tensor_path}")


if __name__ == "__main__":
    main()

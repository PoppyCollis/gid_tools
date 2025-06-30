# gid_tools/envs/classifier/load_dataset.py

import configparser
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from gid_tools.helpers.custom_tool_dataset import CNNToolDataset

def load_data(config_path: str = None):
    cfg = configparser.ConfigParser()
    cfg.read(config_path or Path(__file__).with_name("config_cnn.ini"))

    filename    = cfg["dataset"]["jsonl_filename"]
    test_frac   = float(cfg["dataset"]["test_fraction"])
    batch_size  = int(cfg["loader"]["batch_size"])
    num_workers = int(cfg["loader"]["num_workers"])
    pin_memory  = cfg["loader"].getboolean("pin_memory")

    # resolve full path: 
    config_dir   = Path(__file__).resolve().parent              
    project_root = config_dir.parents[2]                        
    jsonl_path   = (project_root / "datasets" / filename).resolve()

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {jsonl_path}")

    dataset = CNNToolDataset(str(jsonl_path))
    N        = len(dataset)
    test_k   = max(1, int(N * test_frac))
    train_k  = N - test_k

    train_ds, test_ds = random_split(
        dataset,
        [train_k, test_k],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader

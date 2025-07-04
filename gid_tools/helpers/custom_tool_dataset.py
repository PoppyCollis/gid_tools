import configparser
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from gid_tools.helpers.plots import make_bracket_image, render_T_image, render_V_image


class DiffusionToolDataset(Dataset):
    def __init__(self, jsonl_path):
        with open(jsonl_path) as f:
            self.records = [json.loads(line) for line in f]
        self.transform = transforms.Compose([
            transforms.ToTensor(),                   # → [0,1]
            transforms.Resize((32,32)),              # → 32×32
            transforms.Lambda(lambda x: x*2 - 1)      # → [-1,1]
        ])
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        L = np.array(r['lengths']); T = np.array(r['thicknesses'])
        A = np.array(r['angles']); phi = r['phi']; fill = r['fill']
        lab = r.get('label','')
        if lab == "T":
            pil = render_T_image(L, T, A, phi)
        elif lab == "V":
            pil = render_V_image(L, T, A, phi)
        else:
            pil = make_bracket_image(L, T, A, phi, fill=fill)
        return self.transform(pil)

class CNNToolDataset(Dataset):
    """
    A dataset that returns (image_tensor, label_index) for CNN training.
    Mirrors DiffusionToolDataset’s rendering logic (pixel→image) but also extracts labels.
    """
    def __init__(self, jsonl_path: str):
        # Load raw records
        with open(jsonl_path) as f:
            self.records = [json.loads(line) for line in f]
        # Same transform as DiffusionToolDataset: Tensor, Resize→32×32, scale to [-1,1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda x: x*2 - 1),
        ])
        # Map your text labels to integers
        self.label_map = {
            "straight": 0,
            "L": 1,
            "Z": 2,
            "V": 3,
            "T": 4
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        L, T_, A, phi, fill = (
            np.array(r["lengths"]),
            np.array(r["thicknesses"]),
            np.array(r["angles"]),
            r["phi"],
            r["fill"],
        )

        lab = r.get("label", "")
        if lab == "T":
            pil = render_T_image(L, T_, A, phi)
        elif lab == "V":
            pil = render_V_image(L, T_, A, phi)
        else:
            pil = make_bracket_image(L, T_, A, phi, fill=fill)

        img = self.transform(pil)
        # Default to “other” class = len(label_map) if unseen
        label_idx = self.label_map.get(lab, len(self.label_map))
        return img, label_idx
    
    

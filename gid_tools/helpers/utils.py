# gidtools/viz/utils.py
"""
Utility functions for saving and visualizing outputs from diffusion models.
"""
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Union
import torch
from torchvision import transforms
import subprocess
import sys
import json
from torch.utils.data import TensorDataset


def save_samples(
    samples, 
    output_dir: Union[Path, str],
    prefix: str = "sample",
    scale_to_uint8: bool = True
):
    """
    Save a batch of image tensors or numpy arrays as PNG files.

    Args:
        samples (Iterable[torch.Tensor] or Iterable[np.ndarray]):
            List or tensor of image samples. Each sample should be shape [C, H, W] or [H, W].
        output_dir (Path or str):
            Directory where images will be saved.
        prefix (str):
            Filename prefix for each image (default: "sample").
        scale_to_uint8 (bool):
            If True, scales values from [-1,1] to [0,255] before converting to uint8.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, img in enumerate(samples):
        # Convert torch.Tensor to numpy if needed
        if hasattr(img, "cpu"):
            img = img.cpu().numpy()

        # Move channel dimension if needed
        # Accept shapes [C, H, W] or [H, W, C]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        # If single channel dimension remains (H, W, 1), squeeze it
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]

        # Scale and convert to uint8
        if scale_to_uint8:
            img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)

        # Create PIL image
        pil_img = Image.fromarray(img)
        filename = f"{prefix}_{idx}.png"
        pil_img.save(output_path / filename)


def load_image_as_tensor(image_path: Path) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Grayscale(),  # ensure single channel
        transforms.ToTensor(),   # scales to [0,1]
        transforms.Lambda(lambda x: x * 2 - 1)  # rescale to [-1, 1]
    ])
    img = Image.open(image_path).convert('L')  # convert to grayscale
    return transform(img)


def download_checkpoint(root_dir: Path, script_name: str = "download_model_weights.py") -> Path:
    """
    Ensure the diffusion model checkpoint is available, downloading it if necessary.
    Returns the path to the checkpoint file.
    """
    ckpt_dir = root_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = "diffusion_ckpt.pth"
    ckpt_path = ckpt_dir / ckpt_file
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}. Downloading...")
        download_script = root_dir / "scripts" / script_name
        if not download_script.exists():
            raise FileNotFoundError(f"Download script not found: {download_script}")
        subprocess.run([sys.executable, str(download_script)], check=True)
    return ckpt_path



def build_reward_dataset(
    features_dir: Path,
    rewards_path: Path,
) -> TensorDataset:
    """
    Load extracted feature files from a directory and corresponding rewards,
    returning a TensorDataset.

    Rewards file should be a JSON dict mapping string indices ("0", "1", ...) to scalar rewards.
    Feature files in features_dir should be sorted so that their order corresponds to these indices.
    Feature files may contain tensors directly or be dicts containing tensors under key 'features'.

    Args:
        features_dir (Path): Directory containing .pt files, e.g. sample_0_features.pt, sample_1_features.pt, ...
        rewards_path (Path): Path to the .json file mapping "0"->reward0, "1"->reward1, etc.

    Returns:
        TensorDataset: A dataset of (features, reward) pairs as torch tensors.
    """
    # Validate inputs
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    if not features_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory for features, got: {features_dir}")
    if not rewards_path.is_file():
        raise FileNotFoundError(f"Rewards file not found: {rewards_path}")

    # Load rewards mapping: keys are string indices
    with open(rewards_path, "r") as f:
        rewards_dict = json.load(f)

    # Collect feature files
    feature_files = sorted([p for p in features_dir.glob("*.pt") if p.is_file()])
    if not feature_files:
        raise FileNotFoundError(f"No .pt feature files found in {features_dir}")
    if len(feature_files) != len(rewards_dict):
        raise ValueError(
            f"Number of feature files ({len(feature_files)}) does not match number of rewards ({len(rewards_dict)})"
        )

    features_list = []
    rewards_list = []

    for idx, feat_file in enumerate(feature_files):
        key = str(idx)
        if key not in rewards_dict:
            raise KeyError(f"Missing reward for index {key}")

        # Load object from file
        try:
            data = torch.load(feat_file)
        except Exception as e:
            raise IOError(f"Error loading {feat_file}: {e}")

        # Extract tensor
        if torch.is_tensor(data):
            arr = data
        elif isinstance(data, dict):
            # Prefer 'features' key
            if 'features' in data and torch.is_tensor(data['features']):
                arr = data['features']
            else:
                # Fallback: first tensor value
                tensor_vals = [v for v in data.values() if torch.is_tensor(v)]
                if not tensor_vals:
                    raise ValueError(f"No tensor found in dict from {feat_file}")
                arr = tensor_vals[0]
        else:
            raise ValueError(f"Unsupported data type in {feat_file}: {type(data)}")

        features_list.append(arr.float())
        rewards_list.append(float(rewards_dict[key]))

    # Stack into tensors
    X = torch.stack(features_list, dim=0)  # shape [N, D]
    y = torch.tensor(rewards_list, dtype=torch.float32)  # shape [N]

    return TensorDataset(X, y)

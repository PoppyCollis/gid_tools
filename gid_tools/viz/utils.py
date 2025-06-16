# gidtools/viz/utils.py
"""
Utility functions for saving and visualizing outputs from diffusion models.
"""
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Union

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

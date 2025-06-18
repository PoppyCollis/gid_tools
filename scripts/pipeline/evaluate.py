"""
Step 2. Get reward feedback for generated images in the output folder
"""

from pathlib import Path
import json
import torch
from gid_tools.envs.feedback import ToolRewardEnv, pixel_area_tensor
from gid_tools.helpers.utils import load_image_as_tensor


def main():
    # Setup
    script_dir = Path(__file__).resolve().parent
    sample_dir = script_dir / "samples"
    tensor_path = sample_dir / "samples.pt"
    if not tensor_path.exists():
        raise FileNotFoundError(f"No tensor file found at {tensor_path}")

    # load the raw [B, C, H, W] batch
    batch = torch.load(tensor_path)
    
    # set up your env
    env = ToolRewardEnv(default_method=None)
    env.register_reward('pixel_area', pixel_area_tensor)

    # compute rewards
    rewards = [env.compute(img, method='pixel_area') for img in batch]
    #logger.info(f"Pixel-area rewards: {rewards}")
    
    # Build filename→reward mapping using the index as the “filename”
    rewards_dict = {
        str(i): float(r)
        for i, r in enumerate(rewards)
    }
    
    # rewards is a dict[str, float]
    output_path = Path(__file__).resolve().parent / "rewards.json"
    with open(output_path, "w") as f:
        json.dump(rewards_dict, f, indent=2)

if __name__ == "__main__":
    main()

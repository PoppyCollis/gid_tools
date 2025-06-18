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
    
    # rewards is a dict[str, float]
    output_path = script_dir / "rewards.json"
    with open(output_path, "w") as f:
        json.dump(rewards, f, indent=2)


    # script_dir = Path(__file__).resolve().parent
    # samples_dir = script_dir / "samples"
    # output_path = script_dir / "rewards.json"

    # # Initialize reward environment
    # env = ToolRewardEnv(default_method=None)
    # env.register_reward('pixel_area', pixel_area_tensor)

    # # Evaluate rewards
    # rewards = {}
    # for fname in sorted(samples_dir.glob("sample_*.png")):
    #     img_tensor = load_image_as_tensor(fname)
    #     reward = env.compute(img_tensor, method='pixel_area')
    #     rewards[fname.name] = reward
    #     #print(f"{fname.name}: {reward}")

    # # Save to JSON
    # with open(output_path, "w") as f:
    #     json.dump(rewards, f, indent=2)
    # print(f"Saved rewards to {output_path}")


if __name__ == "__main__":
    main()

"""
Step 2. Get reward feedback for generated images in the output folder
"""

from pathlib import Path
import json
from gid_tools.envs.feedback import ToolRewardEnv, pixel_area_tensor
from gid_tools.helpers.utils import load_image_as_tensor


def main():
    # Setup
    script_dir = Path(__file__).resolve().parent
    samples_dir = script_dir / "samples"
    output_path = script_dir / "rewards.json"

    # Initialize reward environment
    env = ToolRewardEnv(default_method=None)
    env.register_reward('pixel_area', pixel_area_tensor)

    # Evaluate rewards
    rewards = {}
    for fname in sorted(samples_dir.glob("sample_*.png")):
        img_tensor = load_image_as_tensor(fname)
        reward = env.compute(img_tensor, method='pixel_area')
        rewards[fname.name] = reward
        #print(f"{fname.name}: {reward}")

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(rewards, f, indent=2)
    print(f"Saved rewards to {output_path}")


if __name__ == "__main__":
    main()

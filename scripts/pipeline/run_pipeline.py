"""
Master script to execute the full reward-model pipeline:
1. generator.py          → generates samples
2. evaluate.py           → computes rewards.json
3. feature_extraction.py → builds .npz feature files in features/
4. train_reward_model.py → trains and saves the MLP head
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path, config_path: Path):
    print(f"\n>>> Running {script_path.name} with config {config_path.name}...")
    result = subprocess.run(
        [sys.executable, str(script_path), "--config", str(config_path)],
        check=False
    )
    if result.returncode != 0:
        print(f"[Error] {script_path.name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    # Locate project and pipeline directories
    project_root = Path(__file__).resolve().parents[3]
    pipeline_dir = project_root / "scripts" / "pipeline"
    config_path = pipeline_dir / "config.ini"

    if not config_path.exists():
        print(f"[Error] Config not found: {config_path}")
        sys.exit(1)

    # Ordered list of pipeline steps
    steps = [
        pipeline_dir / "generator.py",
        pipeline_dir / "evaluate.py",
        pipeline_dir / "feature_extraction.py",
        pipeline_dir / "train_reward_model.py",
    ]

    # Run each step
    for script in steps:
        if not script.exists():
            print(f"[Error] Pipeline script not found: {script}")
            sys.exit(1)
        run_script(script, config_path)

    print("\n All pipeline steps completed successfully.")


if __name__ == "__main__":
    main()

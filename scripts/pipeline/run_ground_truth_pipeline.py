#!/usr/bin/env python
"""
Master script to execute the full reward‐model pipeline
using ground‐truth samples.
1. ground_truth_dataset_samples.py → builds samples.pt in ground_truth_samples/
2. evaluate.py                    → computes rewards.json
3. feature_extraction.py          → builds feature files
4. train_reward_model.py          → trains & saves the MLP head
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path: Path, config_path: Path):
    print(f"\n>>> Running {script_path.name} with config {config_path.name}…")
    result = subprocess.run(
        [sys.executable, str(script_path), "--config", str(config_path)],
        check=False
    )
    if result.returncode != 0:
        print(f"[Error] {script_path.name} failed (exit {result.returncode})")
        sys.exit(result.returncode)

def main():
    # We only need the pipeline directory
    pipeline_dir = Path(__file__).resolve().parent

    # config_ground_truth.ini lives right alongside the scripts
    config_path = pipeline_dir / "config_ground_truth.ini"
    if not config_path.exists():
        print(f"[Error] ground-truth config not found: {config_path}")
        sys.exit(1)

    # Ordered pipeline steps
    steps = [
        pipeline_dir / "load_ground_truth_samples.py",
        pipeline_dir / "evaluate.py",
        pipeline_dir / "feature_extraction.py",
        pipeline_dir / "train_reward_model.py",
    ]

    for step in steps:
        if not step.exists():
            print(f"[Error] pipeline script missing: {step}")
            sys.exit(1)
        run_script(step, config_path)

    print("\n All ground-truth pipeline steps completed successfully.")

if __name__ == "__main__":
    main()

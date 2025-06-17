# run_pipeline.py
"""
Master script to execute the full reward-model pipeline:
1. generator.py        → generates samples
2. evaluate.py         → computes rewards.json
3. feature_extraction.py → builds .npz feature files in features/
4. train_reward_model.py → trains and saves the MLP head

"""
import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path):
    print(f"\n>>> Running {script_path.name}...")
    result = subprocess.run([sys.executable, str(script_path)], check=True)
    if result.returncode != 0:
        print(f"Script {script_path.name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    pipeline_dir = project_root / "scripts" / "pipeline"

    # Ordered list of pipeline steps
    steps = [
        pipeline_dir / "generator.py",
        pipeline_dir / "evaluate.py",
        pipeline_dir / "feature_extraction.py",
        pipeline_dir / "train_reward_model.py",
    ]

    for script in steps:
        if not script.exists():
            print(f"Error: pipeline script not found: {script}")
            sys.exit(1)
        run_script(script)

    print("\n All pipeline steps completed successfully.")


if __name__ == "__main__":
    main()

"""Run from repository root: python scripts/train_offline_fb.py --config ..."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import yaml
    from bfm.fb.trainer import train
    parser = argparse.ArgumentParser(description="Vanilla reward-free offline Forward-Backward training")
    parser.add_argument("--config", default="configs/fb/vanilla_offline.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--training-steps", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for key in ("dataset_manifest", "output_dir", "training_steps", "device"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    print(f"Saved {train(cfg, resume=args.resume)}")


if __name__ == "__main__":
    main()

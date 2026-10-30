"""Train visual FB from already-converted Zarr; never open raw recordings."""
import argparse
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import yaml
    from bfm.data.training import validate_training
    from bfm.fb.trainer import train
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zarr', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/fb/vision_offline.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--feature-batch-size', type=int)
    parser.add_argument('--reference-count', type=int)
    args = parser.parse_args()
    manifest = validate_training(args.zarr)
    cfg = yaml.safe_load(args.config.read_text())
    cfg['dataset_manifest'] = str(manifest)
    cfg['output_dir'] = str(ROOT/'runs/fb/zarr'/uuid.uuid4().hex)
    cfg['device'] = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    for argument, key in [('steps', 'training_steps'), ('batch_size', 'batch_size'), ('reference_count', 'reference_count')]:
        if getattr(args, argument) is not None:
            cfg[key] = getattr(args, argument)
    if args.feature_batch_size is not None:
        cfg.setdefault('vision', {})['feature_batch_size'] = args.feature_batch_size
    backbone = cfg.get('vision', {}).get('backbone_id')
    if backbone and (ROOT/backbone).is_dir():
        cfg['vision']['backbone_id'] = str((ROOT/backbone).resolve())
    print(f'Training {manifest} on {cfg["device"]}; output {cfg["output_dir"]}', flush=True)
    print(f'Saved {train(cfg)}', flush=True)


if __name__ == '__main__':
    main()

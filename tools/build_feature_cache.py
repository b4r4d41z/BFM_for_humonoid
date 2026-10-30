"""Build disposable frozen DINO features without changing canonical RGB Zarr."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import yaml
    from bfm.data.feature_cache import build_cache
    from bfm.data.sharpa import SharpaDataset
    from bfm.fb.observation import ObservationEncoder
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zarr', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'data/feature_cache')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/fb/vision_offline.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int, default=16, help='Observation batch and DINO forward chunk size')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get('observation_profile') != 'vision_proprio_v1':
        parser.error('Cache requires vision_proprio_v1')
    manifest = args.zarr/'manifest.json' if args.zarr.is_dir() else args.zarr
    data = SharpaDataset(manifest, observation_profile='vision_proprio_v1', outcome=cfg.get('outcome', 'all'))
    vision = dict(cfg.get('vision', {}), feature_batch_size=args.batch_size)
    if (ROOT/vision.get('backbone_id', '')).is_dir() and vision.get('backbone_id'):
        vision['backbone_id'] = str((ROOT/vision['backbone_id']).resolve())
    torch.use_deterministic_algorithms(True)
    encoder = ObservationEncoder(vision).to(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(build_cache(data, encoder, args.output, batch_size=args.batch_size))


if __name__ == '__main__':
    main()

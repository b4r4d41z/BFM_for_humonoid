"""Convert one v3 episode and run a small visual FB integration check, not evaluation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--collection-commit', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--max-ages', type=float, nargs=5, default=[.15]*5)
    parser.add_argument('--mock-vision', action='store_true', help='Explicit software-only encoder, no pretrained-model claim')
    parser.add_argument('--backbone-id', default='facebook/dinov2-small', help='HF model ID or local pretrained artifact directory')
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    import numpy as np
    import zarr
    from bfm.data.mcap import convert
    from bfm.data.sharpa import SharpaDataset, validate
    from bfm.data.visual_replay import VisualReplayBuffer
    from bfm.fb.trainer import train
    from bfm.sharpa_runtime import SharpaRunner
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    store = output/'episode.zarr'
    convert(args.episode, store, max_age_s=args.max_ages, collection_commit=args.collection_commit)
    root = zarr.open_group(str(store), mode='r')
    report = {'episode': str(Path(args.episode).resolve()), 'validation': validate(root), 'success': root.attrs['success'],
              'sync_statistics': root.attrs['sync_statistics_before_filter'],
              'camera_skew_ms': {k: root.attrs['camera_skew_ms_'+k] for k in ('p50', 'p95', 'max')},
              'raw_command_checks': root.attrs['raw_command_checks'], 'images': {}}
    for camera in ('head', 'left_wrist', 'right_wrist'):
        image = root[f'obs/images/{camera}'][0]
        report['images'][camera] = {'shape': list(root[f'obs/images/{camera}'].shape), 'dtype': str(image.dtype),
                                    'first_frame_min': int(image.min()), 'first_frame_max': int(image.max())}
    manifest = output/'manifest.json'
    manifest.write_text(json.dumps({'schema_version': '3.0.0', 'dataset_id': root.attrs['source_episode_id'],
        'episodes': [{'path': 'episode.zarr', 'split': 'train'}]}, indent=2))
    vision = ({'backbone': 'mock', 'backbone_id': 'test_only', 'revision': 'mock_v1', 'image_size': 28,
               'feature_dim': 12, 'projection_dim': 16, 'embedding_dim': 32}
              if args.mock_vision else {'backbone_id': args.backbone_id, 'revision': args.revision})
    checkpoint = train({'dataset_schema': 'sharpa_v3', 'observation_profile': 'vision_proprio_v1',
        'vision': vision, 'dataset_manifest': str(manifest), 'output_dir': str(output/'run'),
        'device': args.device, 'batch_size': 4, 'training_steps': args.steps, 'logging_interval': 1,
        'checkpoint_interval': args.steps, 'reference_count': 8,
        'model': {'hidden_dim': 32, 'latent_dim': 8}})
    replay = VisualReplayBuffer(SharpaDataset(manifest, observation_profile='vision_proprio_v1'))
    runner = SharpaRunner(checkpoint, device=args.device, observation_profile='vision_proprio_v1')
    action = runner.act(replay.observations([0]))
    assert action.shape == (26,) and np.isfinite(action).all()
    report.update(checkpoint=str(checkpoint), model_config=runner.payload['model_config'], steps=args.steps,
                  runtime_action_finite=True, mock_vision=args.mock_vision,
                  last_metrics=json.loads((output/'run/metrics.jsonl').read_text().splitlines()[-1]))
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

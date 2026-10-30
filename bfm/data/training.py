"""Validate visual training data using Zarr alone; model assets are external."""
from pathlib import Path
import json
import torch
import zarr
from .sharpa import SharpaDataset, validate
from .visual_replay import VisualReplayBuffer


def validate_training(source):
    path = Path(source).resolve()
    manifest = path/'manifest.json' if path.is_dir() else path
    if not manifest.is_file():
        raise ValueError(f'Missing converted-data manifest: {manifest}; run convert_recordings.py first')
    spec = json.loads(manifest.read_text())
    for entry in spec['episodes']:
        validate(zarr.open_group(str((manifest.parent/entry['path']).resolve()), mode='r'))
    data = SharpaDataset(manifest, observation_profile='vision_proprio_v1')
    replay = VisualReplayBuffer(data)
    batch = replay.sample_transitions(2)
    if batch['action'].shape != (2, 26) or batch['observation']['proprio'].shape != (2, 116):
        raise ValueError('Visual training dimensions mismatch')
    if not torch.isfinite(batch['duration']).all() or (batch['duration'] <= 0).any():
        raise ValueError('Invalid transition duration')
    refs, _ = replay.reference_states(2, 17)
    if not torch.isfinite(refs).all():
        raise ValueError('Invalid reference states')
    return manifest

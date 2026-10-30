"""Disposable float32 frozen-backbone features, separate from canonical Zarr."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from bfm.fb import observation
from .sharpa import validate


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def feature_identity(encoder):
    # Transformers configs can contain integer mapping keys; compare the exact
    # JSON representation that survives process restarts.
    identity = json.loads(json.dumps(encoder.identity()))
    # Execution batching and trainable layers do not define a frozen feature.
    for key in ('feature_batch_size', 'projection_dim', 'embedding_dim', 'fusion'):
        identity.pop(key)
    return {'version': 1, 'vision': identity, 'dtype': 'float32', 'selection': 'last_hidden_state_cls',
            'implementation_sha256': file_digest(observation.__file__),
            'cache_implementation_sha256': file_digest(__file__),
            'matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32}


def check_encoder(encoder):
    if (not encoder.cfg.frozen or any(p.requires_grad for p in encoder.backbone.parameters())
            or observation.weights_digest(encoder.backbone) != encoder.cfg.weights_sha256):
        raise ValueError('Feature cache requires the identified frozen backbone weights')


def endpoints(root):
    """One initial obs plus every next_obs per segment; never deduplicate gaps."""
    segments = root['segment_id'][:]
    sources, current, following = [], [], []
    for row, segment in enumerate(segments):
        if row == 0 or segment != segments[row-1]:
            sources.append(('obs', row))
        current.append(len(sources)-1)
        sources.append(('next_obs', row))
        following.append(len(sources)-1)
    return sources, {'obs': np.asarray(current), 'next_obs': np.asarray(following)}


def specification(dataset, encoder, index):
    root = dataset.roots[index]
    sources, _ = endpoints(root)
    return {'source_episode_id': dataset.ids[index],
            'source_fingerprint': dataset.episode_fingerprints[index],
            'feature_identity': feature_identity(encoder),
            'shape': [len(sources), len(encoder.cfg.cameras), encoder.cfg.feature_dim]}


def open_entry(directory, spec):
    metadata = json.loads((directory/'entry.json').read_text())
    if metadata.get('specification') != spec:
        raise ValueError(f'Incompatible feature cache entry: {directory}')
    path = directory/'features.npy'
    if file_digest(path) != metadata.get('sha256'):
        raise ValueError(f'Corrupt feature cache checksum: {directory}')
    features = np.load(path, mmap_mode='r', allow_pickle=False)
    if list(features.shape) != spec['shape'] or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError(f'Invalid feature cache tensor: {directory}')
    return features


def build_cache(dataset, encoder, destination, *, batch_size=16):
    """Publish complete entries and then the dataset index, each atomically.

    An interrupted build leaves the previous index intact. Complete content-keyed
    episodes can be reused when the manifest grows; incomplete staging dirs cannot.
    """
    if dataset.observation_profile != 'vision_proprio_v1' or batch_size < 1:
        raise ValueError('Visual dataset and positive cache batch_size required')
    check_encoder(encoder)
    destination = Path(destination).resolve()
    # Never allow a derived store inside a canonical episode.
    if any(destination.is_relative_to(Path(root.store.path).resolve()) for root in dataset.roots):
        raise ValueError('Feature cache must be outside canonical Zarr stores')
    (destination/'episodes').mkdir(parents=True, exist_ok=True)
    encoder.eval()
    entries = []
    for index, root in enumerate(dataset.roots):
        spec = specification(dataset, encoder, index)
        key = digest(spec)
        directory = destination/'episodes'/key
        if directory.exists():
            open_entry(directory, spec)
            state = 'reused'
        else:
            # This is the contract justifying endpoint reuse, including pixels.
            validate(root)
            sources, _ = endpoints(root)
            with tempfile.TemporaryDirectory(prefix='.building-', dir=destination/'episodes') as temp:
                staged = Path(temp)/key
                staged.mkdir()
                values = np.lib.format.open_memmap(staged/'features.npy', mode='w+', dtype=np.float32, shape=tuple(spec['shape']))
                for start in range(0, len(sources), batch_size):
                    rows = sources[start:start+batch_size]
                    inputs = {'proprio': torch.empty(len(rows), encoder.cfg.proprio_dim)}
                    for camera in encoder.cfg.cameras:
                        inputs[camera] = np.stack([root[f'{prefix}/images/{camera}'][row] for prefix, row in rows])
                    # One authoritative preprocessing and frozen extraction path.
                    features = encoder.features(inputs)
                    if any(f.dtype != torch.float32 for f in features.values()):
                        raise ValueError('Cache extraction requires float32 backbone output')
                    values[start:start+len(rows)] = np.stack(
                        [features[c].detach().cpu().numpy() for c in encoder.cfg.cameras], axis=1)
                values.flush()
                del values
                (staged/'entry.json').write_text(json.dumps({'specification': spec,
                    'sha256': file_digest(staged/'features.npy')}, sort_keys=True)+'\n')
                open_entry(staged, spec)
                # Never overwrite a completed entry, including a concurrent build.
                if directory.exists():
                    open_entry(directory, spec)
                else:
                    os.rename(staged, directory)
            state = 'built'
        entries.append(key)
        print(f'{state} {index+1}/{len(dataset.roots)} {dataset.ids[index]}', flush=True)
    metadata = {'version': 1, 'dataset_fingerprint': dataset.fingerprint,
                'feature_identity': feature_identity(encoder), 'entries': entries}
    with tempfile.NamedTemporaryFile(mode='w', dir=destination, prefix='.index-', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(metadata, stream, sort_keys=True)
        stream.write('\n')
    try:
        os.replace(temporary, destination/'index.json')
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class FeatureCache:
    def __init__(self, directory, dataset, encoder):
        check_encoder(encoder)
        directory = Path(directory)
        metadata = json.loads((directory/'index.json').read_text())
        expected = {'version': 1, 'dataset_fingerprint': dataset.fingerprint,
                    'feature_identity': feature_identity(encoder),
                    'entries': [digest(specification(dataset, encoder, i)) for i in range(len(dataset.roots))]}
        if metadata != expected:
            raise ValueError('Incompatible feature cache dataset/vision identity; rebuild the cache')
        self.cameras = tuple(encoder.cfg.cameras)
        self.image_rows = dataset.image_rows
        self.features, self.maps = [], []
        for i, key in enumerate(expected['entries']):
            self.features.append(open_entry(directory/'episodes'/key, specification(dataset, encoder, i)))
            self.maps.append(endpoints(dataset.roots[i])[1])

    def observations(self, indices, prefix):
        rows = [self.features[episode][self.maps[episode][prefix][row]]
                for episode, row in (self.image_rows[i] for i in indices)]
        values = torch.from_numpy(np.stack(rows))
        return {f'features/{camera}': values[:, i] for i, camera in enumerate(self.cameras)}

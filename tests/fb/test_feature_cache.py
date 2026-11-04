"""Persistent cache tests; the local pretrained DINO test never accesses network."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import zarr

from test_sharpa import source, make
from test_vision import visual_manifest, mock_config
from bfm.data.feature_cache import FeatureCache, build_cache, endpoints
from bfm.data.sharpa import SharpaDataset
from bfm.data.visual_replay import VisualReplayBuffer
from bfm.fb.observation import ObservationEncoder, MockBackbone
from bfm.fb.trainer import train
from bfm.fb.checkpoint import load_payload, load_model


def dataset(manifest):
    return SharpaDataset(manifest, observation_profile='vision_proprio_v1')


def test_cache_equivalence_dedup_reopen_and_incremental(visual_manifest, tmp_path, monkeypatch):
    data = dataset(visual_manifest)
    encoder = ObservationEncoder(mock_config())
    seen = []
    hook = encoder.backbone.register_forward_hook(lambda m, inputs, output: seen.append(len(inputs[0])))
    path = build_cache(data, encoder, tmp_path/'cache', batch_size=3)
    hook.remove()
    assert sum(seen) == 10*3  # 9 edges -> 10 endpoints, three named cameras.
    cache = FeatureCache(path, data, encoder)
    assert cache.features[0].shape == (10, 3, 12)
    replay = VisualReplayBuffer(data)
    for prefix in ('obs', 'next_obs'):
        obs = replay.observations([0, 3, 8], prefix)
        direct = encoder.features(obs)
        cached = cache.observations([0, 3, 8], prefix)
        assert list(cached) == [f'features/{c}' for c in encoder.cfg.cameras]
        for c in encoder.cfg.cameras:
            torch.testing.assert_close(cached[f'features/{c}'], direct[c], rtol=1e-6, atol=1e-6)
    # Reopen through the actual reader in an independent Python process.
    code = """import sys
from bfm.data.feature_cache import FeatureCache
from bfm.data.sharpa import SharpaDataset
from bfm.fb.observation import ObservationEncoder
import json
c=FeatureCache(sys.argv[1], SharpaDataset(sys.argv[2], observation_profile='vision_proprio_v1'), ObservationEncoder(json.loads(sys.argv[3])))
assert c.observations([0], 'obs')['features/head'].shape == (1,12)
"""
    subprocess.run([sys.executable, '-c', code, str(path), str(visual_manifest), json.dumps(mock_config())], check=True)
    def forbidden(*args, **kwargs): raise AssertionError('Existing entries must not rerun DINO')
    monkeypatch.setattr(encoder, 'features', forbidden)
    build_cache(data, encoder, path)
    # A changed manifest needs a new index but can reuse identical episode files.
    spec = json.loads(visual_manifest.read_text()); spec['dataset_id'] = 'new_manifest'
    visual_manifest.write_text(json.dumps(spec))
    changed = dataset(visual_manifest)
    with pytest.raises(ValueError, match='Incompatible'): FeatureCache(path, changed, encoder)
    build_cache(changed, encoder, path)
    assert len(list((path/'episodes').glob('*/features.npy'))) == 1


def test_append_only_builds_new_episode_and_failure_preserves_index(visual_manifest, tmp_path, monkeypatch):
    import shutil
    encoder = ObservationEncoder(mock_config())
    path = build_cache(dataset(visual_manifest), encoder, tmp_path/'cache')
    original_index = (path/'index.json').read_bytes()
    shutil.copytree(visual_manifest.parent/'episode.zarr', visual_manifest.parent/'second.zarr')
    root = zarr.open_group(str(visual_manifest.parent/'second.zarr'), mode='a')
    meta = root.attrs['source_metadata']; meta['bfm']['episode']['episode_id'] = 'second'
    root.attrs.update(source_episode_id='second', source_metadata=meta)
    spec = json.loads(visual_manifest.read_text())
    spec['episodes'].append({'path':'second.zarr', 'split':'train'})
    visual_manifest.write_text(json.dumps(spec))
    data = dataset(visual_manifest)
    def interrupted(*args): raise RuntimeError('interrupted')
    with monkeypatch.context() as patch:
        patch.setattr(encoder, 'features', interrupted)
        with pytest.raises(RuntimeError): build_cache(data, encoder, path)
    assert (path/'index.json').read_bytes() == original_index
    seen = []
    hook = encoder.backbone.register_forward_hook(lambda m, inputs, output: seen.append(len(inputs[0])))
    build_cache(data, encoder, path)
    hook.remove()
    assert sum(seen) == 30 and len(FeatureCache(path, data, encoder).features) == 2


@pytest.mark.parametrize('fault', ['image_size', 'cameras', 'revision', 'feature_dim', 'rgb', 'checksum', 'shape'])
def test_stale_and_corrupt_features_rejected(visual_manifest, tmp_path, fault):
    data = dataset(visual_manifest)
    encoder = ObservationEncoder(mock_config())
    path = build_cache(data, encoder, tmp_path/'cache')
    if fault == 'rgb':
        root = zarr.open_group(str(visual_manifest.parent/'episode.zarr'), mode='a')
        root['obs/images/head'][0] = 255
        data = dataset(visual_manifest)  # Includes compressed RGB content identity.
    elif fault in ('checksum', 'shape'):
        file = next((path/'episodes').glob('*/features.npy'))
        np.save(file, np.zeros((1, 3, 12), dtype=np.float32))
        if fault == 'shape':
            from bfm.data.feature_cache import file_digest
            meta = json.loads(file.with_name('entry.json').read_text()); meta['sha256'] = file_digest(file)
            file.with_name('entry.json').write_text(json.dumps(meta))
    else:
        cfg = copy.deepcopy(mock_config())
        if fault == 'image_size': cfg['image_size'] = 28
        if fault == 'cameras': cfg['cameras'] = list(reversed(encoder.cfg.cameras))
        if fault == 'revision': cfg['revision'] = 'different'
        if fault == 'feature_dim': cfg['feature_dim'] = 18
        encoder = ObservationEncoder(cfg)
    with pytest.raises(ValueError, match='cache'): FeatureCache(path, data, encoder)


def test_segment_endpoints_and_atomic_failure(visual_manifest, tmp_path, monkeypatch):
    data = dataset(visual_manifest)
    root = zarr.open_group(str(visual_manifest.parent/'episode.zarr'), mode='a')
    # Pure endpoint mapping regression: an equal pixel across a gap is not deduped.
    root['segment_id'][4:] = 1
    sources, maps = endpoints(root)
    assert len(sources) == 11 and maps['next_obs'][3] != maps['obs'][4]
    root['segment_id'][:] = 0
    encoder = ObservationEncoder(mock_config())
    def broken(*args): raise RuntimeError('interrupted extraction')
    monkeypatch.setattr(encoder, 'features', broken)
    with pytest.raises(RuntimeError, match='interrupted'):
        build_cache(data, encoder, tmp_path/'failed')
    assert not (tmp_path/'failed/index.json').exists()
    assert not list((tmp_path/'failed/episodes').iterdir())
    with pytest.raises(ValueError, match='outside canonical'):
        build_cache(data, encoder, visual_manifest.parent/'episode.zarr/cache')


def test_cached_trainer_no_rgb_or_backbone_and_identical_resume(visual_manifest, tmp_path, monkeypatch):
    data = dataset(visual_manifest)
    path = build_cache(data, ObservationEncoder(mock_config()), tmp_path/'cache')
    cfg = {'dataset_schema':'sharpa_v3', 'observation_profile':'vision_proprio_v1', 'vision':mock_config(),
           'dataset_manifest':str(visual_manifest), 'batch_size':2, 'training_steps':2,
           'reference_count':2, 'checkpoint_interval':2, 'model':{'hidden_dim':16, 'latent_dim':4}}
    online = train({**cfg, 'output_dir':str(tmp_path/'online')})
    getitem = zarr.Array.__getitem__
    def no_rgb(array, key):
        if '/images/' in array.path: raise AssertionError('Cached training decompressed RGB')
        return getitem(array, key)
    def no_dino(*args): raise AssertionError('Cached training ran DINO')
    with monkeypatch.context() as patch:
        patch.setattr(zarr.Array, '__getitem__', no_rgb)
        patch.setattr(MockBackbone, 'forward', no_dino)
        cached = train({**cfg, 'feature_cache':str(path), 'output_dir':str(tmp_path/'cached')})
        a, b = load_payload(online), load_payload(cached)
        for k in a['model']: torch.testing.assert_close(a['model'][k], b['model'][k], rtol=1e-5, atol=1e-6)
        train({**cfg, 'feature_cache':str(path), 'output_dir':str(tmp_path/'cached'), 'training_steps':3}, resume=cached)
    model, payload = load_model(cached)
    assert payload['step'] == 3
    batch = VisualReplayBuffer(data).observations([0])
    assert model.act(batch, model.sample_z(1)).shape == (1,26)


def test_real_local_dino_cached_equals_direct(visual_manifest, tmp_path):
    artifact = Path(__file__).resolve().parents[2]/'models/dinov2-small'
    weights = artifact/'model.safetensors'
    if not weights.exists() or weights.stat().st_size < 1_000_000:
        pytest.skip('Materialize local pretrained DINO Git LFS weights for real equivalence test')
    encoder = ObservationEncoder({'backbone_id':str(artifact), 'image_size':518, 'feature_batch_size':1})
    data = dataset(visual_manifest)
    path = build_cache(data, encoder, tmp_path/'real-cache', batch_size=1)
    encoder.cfg.feature_batch_size = 2  # Execution batching may change on reuse.
    cached = FeatureCache(path, data, encoder).observations([0, 8], 'next_obs')
    direct = encoder.features(VisualReplayBuffer(data).observations([0, 8], 'next_obs'))
    assert encoder.cfg.feature_dim == 384
    for c in encoder.cfg.cameras:
        torch.testing.assert_close(cached[f'features/{c}'], direct[c], rtol=1e-5, atol=1e-5)
    assert not encoder.backbone.training and all(not p.requires_grad for p in encoder.backbone.parameters())

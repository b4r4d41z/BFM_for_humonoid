"""Offline software checks; mock visual features are not pretrained-model results."""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
import zarr
from test_sharpa import source, make
from bfm.contracts.sharpa import LOW, HIGH, JOINTS, GROUPS, SOURCE_COMMIT
from bfm.contracts.vision import CAMERAS, PROPRIO_DIM, PROPRIO_JOINTS, proprioception
from bfm.data.mcap import ACTION, TOPICS, HELD, RAW_ARM, ARM_INDICES, build, convert
from bfm.data.sharpa import write, validate, SharpaDataset
from bfm.data.visual_replay import VisualReplayBuffer
from bfm.fb.observation import VisionConfig, preprocess_rgb
from bfm.fb.agent import FBAgent, TrainConfig
from bfm.fb.model import ModelConfig
from bfm.fb.trainer import train
from bfm.fb.checkpoint import load_model, load_payload
from bfm.sharpa_runtime import SharpaRunner, VisualObservationBuffer


def mock_config():
    return {'backbone': 'mock', 'backbone_id': 'test_only', 'revision': 'mock_v1',
            'feature_dim': 12, 'image_size': 14, 'projection_dim': 8, 'embedding_dim': 16}


@pytest.fixture
def visual_manifest(source, tmp_path):
    records, metadata = copy.deepcopy(source)
    metadata['success'] = False
    # Distinct views and changing measured positions exercise all input paths.
    for i in range(10):
        joints = records[TOPICS[1]][i][1]
        joints.position = (np.arange(len(joints.name))*.01 + i*.001).tolist()
        joints.velocity = (np.arange(len(joints.name))*.005).tolist()
        for c, topic in enumerate(TOPICS[2:]):
            records[topic][i][1].data = bytes([i+10*c, 20+c, 30+c, 40+c, 50+c, 60+c, 0, 0]*2)
    arrays, attrs = make((records, metadata))
    write(tmp_path/'episode.zarr', arrays, attrs)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'schema_version': '3.0.0', 'dataset_id': 'visual_fixture',
        'episodes': [{'path': 'episode.zarr', 'split': 'train'}]}))
    return manifest


def test_v3_rejects_old_metadata_and_bounds_provenance(source):
    records, metadata = copy.deepcopy(source)
    metadata['bfm']['contract_version'] = 2
    with pytest.raises(ValueError, match='collection-v2'): make((records, metadata))
    metadata['bfm']['contract_version'] = 3
    metadata['bfm']['arm_target_processing']['bounds_source_sha256'] = 'bad'
    with pytest.raises(ValueError, match='provenance'): make((records, metadata))
    records, metadata = copy.deepcopy(source)
    records[HELD[1]][0][1].left_hand_position = []
    records[HELD[1]][0][1].right_hand_position = [50.]
    with pytest.raises(ValueError, match='six finite'): make((records, metadata))


@pytest.mark.parametrize('boundary', [LOW, HIGH])
def test_clamped_raw_and_exact_canonical_bounds(source, boundary):
    records, metadata = copy.deepcopy(source)
    for i, (t, action) in enumerate(records[ACTION]):
        for group, sl in GROUPS.items():
            setattr(action, group+'_target', boundary[sl])
        records[HELD[0]][i][1].position = np.array(boundary)[ARM_INDICES].tolist()
        records[HELD[1]][i][1].left_hand_position = boundary[7:13]
        records[HELD[1]][i][1].right_hand_position = boundary[20:26]
        raw = np.array(boundary)[ARM_INDICES] + (-.2 if boundary is LOW else .2)
        records[RAW_ARM][i][1].position = raw.tolist()
    arrays, attrs = make((records, metadata))
    assert np.array_equal(arrays['action'], np.tile(boundary, (9, 1)))
    assert attrs['raw_command_checks']['saturated_ticks'] == 10
    # The asymmetric second joints retain distinct left/right bounds.
    assert LOW[1] == -.349066 and LOW[14] == -2.0944
    assert HIGH[1] == 2.0944 and HIGH[14] == .349066
    records[ACTION][0][1].left_arm_target = list(records[ACTION][0][1].left_arm_target)
    records[ACTION][0][1].left_arm_target[3] = -1.7
    with pytest.raises(ValueError, match='outside frozen'): make((records, metadata))


def test_raw_provenance_header_and_transform_must_match(source):
    records, metadata = copy.deepcopy(source)
    # Unique source/held identities prohibit borrowing a different tick's value.
    for i in range(10):
        records[RAW_ARM][i][1].header.stamp.nanosec = i
        records[HELD[0]][i][1].header.stamp.nanosec = i
    assert make((records, metadata))[1]['raw_command_checks']['count'] == 10
    records[RAW_ARM][0][1].position[3] = -1.668971
    with pytest.raises(ValueError, match=r'clamp\(raw\)'): make((records, metadata))
    records[ACTION][0][1].left_arm_target[3] = -1.5708
    records[HELD[0]][0][1].position[3] = -1.5708
    assert make((records, metadata))[1]['raw_command_checks']['saturated_ticks'] == 1
    records[RAW_ARM][0][1].header.stamp.sec += 1
    with pytest.raises(ValueError, match='provenance'): make((records, metadata))


def test_camera_skew_and_failed_outcome(source):
    records, metadata = copy.deepcopy(source)
    metadata['success'] = False
    for topic, delta in zip(TOPICS[2:], [0, -3_000_000, -7_000_000]):
        records[topic] = [(t+delta, msg) for t, msg in records[topic]]
    arrays, attrs = make((records, metadata))
    assert attrs['success'] is False and not arrays['terminated'].any()
    assert [attrs['camera_skew_ms_'+k] for k in ('p50', 'p95', 'max')] == [7., 7., 7.]
    assert (arrays['obs/source_record_ns'] <= arrays['record_ns'][:, None]).all()
    assert (arrays['next_obs/source_record_ns'] <= arrays['next_record_ns'][:, None]).all()
    with pytest.raises(ValueError, match='No valid causal'):
        build(records, metadata, max_age_s=[.05]*5, collection_commit=SOURCE_COMMIT, max_camera_skew_s=.005)


def test_task56_no_image_reads(visual_manifest, monkeypatch):
    original = zarr.Array.__getitem__
    def checked(array, key):
        if '/images/' in array.path:
            raise AssertionError('task56 must not decompress image pixels')
        return original(array, key)
    monkeypatch.setattr(zarr.Array, '__getitem__', checked)
    assert SharpaDataset(visual_manifest).episodes[0]['observation'].shape == (9, 56)


def test_visual_named_measured_inputs_and_no_privileged_leak(visual_manifest):
    data = SharpaDataset(visual_manifest, observation_profile='vision_proprio_v1', outcome='failed')
    replay = VisualReplayBuffer(data)
    batch = replay.sample_transitions(4)
    observation = batch['observation']
    assert list(observation) == ['proprio', *CAMERAS]
    assert observation['proprio'].shape == (4, PROPRIO_DIM) and PROPRIO_DIM == 116
    assert all(observation[c].dtype == torch.uint8 and observation[c].shape == (4, 2, 2, 3) for c in CAMERAS)
    assert not torch.equal(observation['head'], observation['left_wrist'])
    assert 'evaluation_state' not in batch
    root = zarr.open_group(str(visual_manifest.parent/'episode.zarr'), mode='a')
    before = replay.observations([0])
    root['obs/state'][:, 34:40] = 987.  # privileged current object q/dq
    root['next_obs/state'][:, 34:40] = 987.
    data2 = SharpaDataset(visual_manifest, observation_profile='vision_proprio_v1')
    after = VisualReplayBuffer(data2).observations([0])
    for key in before: assert torch.equal(before[key], after[key])
    names = list(reversed(PROPRIO_JOINTS))
    values = np.arange(58)
    p = proprioception(names, values, -values)
    assert np.array_equal(p[:58], values[::-1]) and np.array_equal(p[58:], -values[::-1])
    with pytest.raises(ValueError, match='missing'):
        proprioception(names[:-1], values[:-1], values[:-1])
    with pytest.raises(ValueError, match='No train'):
        SharpaDataset(visual_manifest, outcome='successful')
    root.attrs['joint_units'] = {'position': 'deg', 'velocity': 'rad/s', 'effort': 'N*m'}
    with pytest.raises(ValueError, match='joint units'):
        SharpaDataset(visual_manifest, observation_profile='vision_proprio_v1')


def test_preprocess_rgb_deterministic_letterbox():
    cfg = VisionConfig(**mock_config())
    red = torch.zeros(2, 4, 8, 3, dtype=torch.uint8); red[..., 0] = 255
    a, b = preprocess_rgb(red, cfg), preprocess_rgb(red.numpy(), cfg)
    assert torch.equal(a, b) and a.shape == (2, 3, 14, 14)
    assert torch.equal(a[:, :, 0], torch.zeros_like(a[:, :, 0]))  # normalized mean padding
    assert a[0, 0, 7, 7] > 2 and a[0, 1, 7, 7] < -2  # RGB, not BGR
    with pytest.raises(ValueError, match='uint8'): preprocess_rgb(red.float(), cfg)


def test_fb_visual_gradients_frozen_backbone_and_camera_identity(visual_manifest):
    data = SharpaDataset(visual_manifest, observation_profile='vision_proprio_v1')
    batch = VisualReplayBuffer(data).sample_transitions(4)
    agent = FBAgent(ModelConfig(state_dim=16, action_dim=26, latent_dim=4, hidden_dim=16,
        observation_profile='vision_proprio_v1', vision=mock_config()), TrainConfig())
    encoder = agent.model.observation_encoder
    before = {n: p.clone() for n,p in encoder.fusion.named_parameters()}
    metrics = agent.update(batch)
    assert all(np.isfinite(list(metrics.values())))
    assert any(not torch.equal(before[n], p) for n,p in encoder.fusion.named_parameters())
    assert all(p.grad is None and not p.requires_grad for p in encoder.backbone.parameters())
    encoder.train(); assert not encoder.backbone.training
    with torch.no_grad():
        original = agent.model._states(batch['observation'])
        swapped = dict(batch['observation'])
        swapped['head'], swapped['left_wrist'] = swapped['left_wrist'], swapped['head']
        assert not torch.allclose(original, agent.model._states(swapped))
        assert agent.model.act(batch['observation'], agent.model.sample_z(4)).shape == (4, 26)
    assert any(not torch.equal(p, t) for p,t in zip(encoder.fusion.parameters(), agent.target_fusion.parameters()))
    bad = dict(batch['observation']); bad['head'] = bad['head'][:1]
    with pytest.raises(ValueError, match='batch size'): agent.model._states(bad)


def test_visual_train_resume_checkpoint_and_runtime(visual_manifest, tmp_path):
    cfg = {'dataset_schema': 'sharpa_v3', 'observation_profile': 'vision_proprio_v1',
        'vision': mock_config(), 'dataset_manifest': str(visual_manifest), 'output_dir': str(tmp_path/'run'),
        'batch_size': 4, 'reference_count': 4, 'training_steps': 3, 'logging_interval': 1, 'checkpoint_interval': 3,
        'model': {'state_dim': 16, 'action_dim': 26, 'latent_dim': 4, 'hidden_dim': 16}}
    checkpoint = train(cfg)
    model, payload = load_model(checkpoint, expected_profile='vision_proprio_v1')
    runner = SharpaRunner(checkpoint, observation_profile='vision_proprio_v1')
    replay = VisualReplayBuffer(SharpaDataset(visual_manifest, observation_profile='vision_proprio_v1'))
    observation = replay.observations([0])
    action = runner.act(observation)
    assert action.shape == (26,) and np.isfinite(action).all()
    assert (action >= LOW).all() and (action <= HIGH).all()
    assert model.cfg.vision == payload['model_config']['vision']
    with torch.no_grad():
        assert torch.equal(model._states(observation), runner.model._states(observation))
    with pytest.raises(ValueError, match='profile'): SharpaRunner(checkpoint)
    with pytest.raises(ValueError, match='visual configuration'):
        load_model(checkpoint, expected_vision={**model.cfg.vision, 'image_size': 28})
    train({**cfg, 'training_steps': 4}, resume=checkpoint)
    # Compare resumed update with uninterrupted training, including lazy RGB RNG.
    uninterrupted = train({**cfg, 'training_steps': 4, 'output_dir': str(tmp_path/'continuous')})
    a, b = load_payload(checkpoint), load_payload(uninterrupted)
    assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
    payload['model_config']['vision']['cameras'].reverse()
    tampered = tmp_path/'tampered.pt'; torch.save(payload, tampered)
    with pytest.raises(ValueError, match='identity'): load_model(tampered)
    payload = load_payload(checkpoint); payload['contract']['version'] = 2
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match='contract'): load_model(tampered)


def test_visual_runtime_causality_freshness_and_named_joints():
    buffer = VisualObservationBuffer(.15, CAMERAS, .03)
    assert buffer.snapshot(1.) is None
    joints = NS(name=list(PROPRIO_JOINTS), position=list(range(58)), velocity=[0.]*58)
    buffer.receive('joints', joints, .99)
    for i,c in enumerate(CAMERAS): buffer.receive(c, np.full((4,8,3), i, dtype=np.uint8), .98+i*.001)
    observation = buffer.snapshot(1.)
    assert observation['proprio'].shape == (1,116)
    assert observation['head'].dtype == np.uint8
    with pytest.raises(RuntimeError, match='stale'): buffer.snapshot(1.2)
    with pytest.raises(RuntimeError, match='Future'): buffer.snapshot(.9)
    buffer.receive('head', observation['head'][0], 1.04)
    with pytest.raises(RuntimeError, match='skew'): buffer.snapshot(1.05)


def test_dinov2_local_api_frozen_weights_and_offline_restore(tmp_path, monkeypatch):
    # Actual transformers API, tiny RANDOM local artifact, not pretrained validation.
    pytest.importorskip('transformers')
    from transformers import Dinov2Config, Dinov2Model
    from bfm.fb.observation import ObservationEncoder, weights_digest
    artifact = tmp_path/'local_dino_fixture'
    Dinov2Model(Dinov2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
                            intermediate_size=32, image_size=28, patch_size=14)).save_pretrained(artifact)
    config = {'backbone_id': str(artifact), 'image_size': 28, 'embedding_dim': 16,
              'projection_dim': 8, 'feature_batch_size': 2}
    encoder = ObservationEncoder(config)
    assert encoder.cfg.feature_dim == 16
    def no_download(*args, **kwargs):
        raise AssertionError('Checkpoint restore must not access pretrained source')
    monkeypatch.setattr(Dinov2Model, 'from_pretrained', no_download)
    saved_config = asdict(encoder.cfg)
    config_before_restore = copy.deepcopy(saved_config)
    restored = ObservationEncoder(saved_config, initialize_pretrained=False)
    assert saved_config == config_before_restore == asdict(restored.cfg)
    restored.load_state_dict(encoder.state_dict())
    obs = {'proprio': torch.zeros(2,116), **{c: torch.full((2,8,12,3), 100, dtype=torch.uint8) for c in CAMERAS}}
    a = encoder(obs, obs['proprio']); b = restored(obs, obs['proprio'])
    assert torch.equal(a, b) and torch.isfinite(a).all()
    a.square().sum().backward()
    assert all(not p.requires_grad and p.grad is None for p in encoder.backbone.parameters())
    assert any(p.grad is not None for p in encoder.fusion.parameters())
    assert weights_digest(restored.backbone) == encoder.cfg.weights_sha256


def test_camera_ablation_keeps_explicit_names(visual_manifest):
    data = SharpaDataset(visual_manifest, observation_profile='vision_proprio_v1')
    batch = VisualReplayBuffer(data, cameras=('head',)).sample_transitions(2)
    agent = FBAgent(ModelConfig(state_dim=16, action_dim=26, latent_dim=4, hidden_dim=16,
        observation_profile='vision_proprio_v1', vision={**mock_config(), 'cameras': ['head']}), TrainConfig())
    assert set(batch['observation']) == {'head', 'proprio'}
    assert np.isfinite(agent.update(batch)['fb_loss'])


def test_real_v3_episode_if_available(tmp_path):
    tracked_sample = Path(__file__).resolve().parents[2] / 'data/current_sample/record_20261029_091718_285773_658bd774'
    episode = Path(os.getenv('BFM_REAL_EPISODE', str(tracked_sample)))
    if not (episode/'bag/bag_0.mcap').is_file():
        pytest.skip('Real collection-v3 episode is not available in this environment')
    with (episode/'bag/bag_0.mcap').open('rb') as stream:
        if stream.read(8) != b'\x89MCAP0\r\n':
            pytest.skip('Real binary MCAP unavailable; materialize the Git LFS object first')
    metadata = json.loads((episode/'episode.json').read_text())
    commit = os.environ.get('BFM_REAL_COLLECTION_COMMIT') or metadata.get('recording_code', {}).get('commit_sha')
    assert commit, 'Set BFM_REAL_COLLECTION_COMMIT to the actual recording checkout SHA'
    output = tmp_path/'real.zarr'
    convert(episode, output, max_age_s=[.15]*5, collection_commit=commit)
    root = zarr.open_group(str(output), mode='r')
    assert validate(root)['transitions'] > 0
    if metadata['bfm'].get('policy_clock', {}).get('domain') == 'isaac_simulation':
        assert root.attrs['policy_timing']['domain'] == 'action_header_simulation_ns'
        assert np.allclose(root['next_timestamp'][:]-root['timestamp'][:], metadata['bfm']['policy_dt'], atol=.02)
    assert root.attrs['raw_command_checks']['saturated_ticks'] > 0
    for camera in CAMERAS:
        assert root[f'obs/images/{camera}'][0].dtype == np.uint8
        assert root[f'obs/images/{camera}'][0].max() > 0
    manifest = tmp_path/'real.json'
    manifest.write_text(json.dumps({'schema_version': '3.0.0', 'dataset_id': 'real-smoke',
        'episodes': [{'path': 'real.zarr', 'split': 'train'}]}))
    replay = VisualReplayBuffer(SharpaDataset(manifest, observation_profile='vision_proprio_v1'))
    agent = FBAgent(ModelConfig(state_dim=16, action_dim=26, latent_dim=4, hidden_dim=16,
        observation_profile='vision_proprio_v1', vision=mock_config()), TrainConfig())
    assert all(np.isfinite(list(agent.update(replay.sample_transitions(4)).values())))

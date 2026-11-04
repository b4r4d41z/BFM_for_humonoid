"""Runtime-only precision and profiling; CUDA tests skip explicitly on CPU hosts."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import yaml

from test_runtime_tasks import trained, visual_manifest, source
from test_evaluation import evaluator
from bfm.data.sharpa import SharpaDataset
from bfm.data.visual_replay import VisualReplayBuffer
from bfm.fb.runtime import VisualRuntime, InferenceTimings, validate_ros_config


@pytest.mark.parametrize('precision,profile', [('int8',False),('FP32',False),('fp32',1),('fp32','false')])
def test_invalid_inference_config(trained, precision, profile):
    with pytest.raises(ValueError, match='Inference requires'):
        VisualRuntime(trained[2], precision=precision, profile=profile)


def test_fp32_matches_original_and_profiles(trained, monkeypatch):
    runtime, manifest, path = trained
    obs = VisualReplayBuffer(SharpaDataset(manifest, observation_profile='vision_proprio_v1')).observations([0])
    z = runtime.model.sample_z(1)
    with torch.no_grad(): expected = runtime.model.act(runtime.observation(obs), z, deterministic=True)[0].cpu()
    original_identity = runtime.model.observation_encoder.identity()
    def forbidden(*args): pytest.fail('Unprofiled CPU path called CUDA synchronization')
    monkeypatch.setattr(torch.cuda, 'synchronize', forbidden)
    torch.testing.assert_close(runtime.act(obs, z), expected, rtol=0, atol=0)
    assert runtime.last_profile == {}
    with torch.autocast('cpu', dtype=torch.bfloat16):
        torch.testing.assert_close(runtime.act(obs,z), expected, rtol=0, atol=0)
    restored = VisualRuntime(path, profile=True)
    torch.testing.assert_close(restored.act(obs, z), expected, rtol=0, atol=0)
    fields = {'input_latency_s', 'preprocessing_latency_s', 'vision_backbone_latency_s',
              'feature_handling_latency_s', 'fusion_latency_s', 'policy_head_latency_s'}
    assert set(restored.last_profile) == fields | {'total_policy_latency_s'}
    assert all(v >= 0 for v in restored.last_profile.values())
    assert sum(restored.last_profile[k] for k in fields) <= restored.last_profile['total_policy_latency_s']
    assert restored.model.observation_encoder.identity() == original_identity
    assert all(p.dtype == torch.float32 for p in restored.model.parameters())


def test_precision_config_and_cpu_rejection(trained):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/runtime/isaac_ros2.yaml').read_text())
    for precision in ('fp32','bf16','fp16'):
        cfg['inference_precision'] = precision
        validate_ros_config(cfg)
    cfg['profile_inference'] = 'yes'
    with pytest.raises(ValueError): validate_ros_config(cfg)
    for precision in ('bf16', 'fp16'):
        with pytest.raises(ValueError, match='requires CUDA'): VisualRuntime(trained[2], 'cpu', precision=precision)


def test_timing_synchronizes_only_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda d: calls.append(d))
    device = torch.device('cuda:0')
    with InferenceTimings(device, False).stage('test'): pass
    assert not calls
    timer = InferenceTimings(device, True)
    with timer.stage('test'): pass
    assert calls == [device, device] and timer.values['test'] >= 0


def test_profile_logged_before_stale_rejection(tmp_path):
    runtime = NS(precision='fp32', last_profile={})
    evaluation, _ = evaluator(tmp_path, warmup=0, runtime=runtime)
    def act(obs):
        runtime.last_profile = {'vision_backbone_latency_s':.2}
        evaluation.buffer.samples['head'] = (.1, evaluation.buffer.samples['head'][1])
        return torch.zeros(26)
    runtime.act = act
    with pytest.raises(RuntimeError, match='stale'): evaluation.tick(lambda a: pytest.fail('Published stale data'))
    row = json.loads((tmp_path/'run/ticks.jsonl').read_text())
    assert row['vision_backbone_latency_s'] == .2 and row['inference_precision'] == 'fp32'
    assert not row['published']


@pytest.mark.parametrize('precision', ['bf16', 'fp16'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for inference autocast comparison')
def test_cuda_precision_with_real_dino_api(trained, tmp_path, precision):
    if precision == 'bf16' and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip('Native CUDA BF16 unavailable')
    # Restore an actual DINO architecture locally; no network or pretrained download.
    from transformers import Dinov2Config, Dinov2Model
    from bfm.fb.observation import weights_digest
    runtime, manifest, _ = trained
    payload = runtime.payload
    cfg = Dinov2Config(hidden_size=12, num_hidden_layers=1, num_attention_heads=3,
                       intermediate_size=24, image_size=14, patch_size=14)
    backbone = Dinov2Model(cfg)
    vision = dict(payload['model_config']['vision'])
    import transformers
    vision.update(backbone='dinov2', backbone_config=cfg.to_dict(), weights_sha256=weights_digest(backbone),
                  implementation_version=transformers.__version__, image_size=14)
    payload['model_config']['vision'] = vision
    payload['observation_identity'] = {'observation_profile':'vision_proprio_v1', **vision}
    payload['model'] = {k:v for k,v in payload['model'].items() if not k.startswith('observation_encoder.backbone.')}
    payload['model'].update({'observation_encoder.backbone.'+k:v for k,v in backbone.state_dict().items()})
    path = tmp_path/'cuda-fixture.pt';torch.save(payload, path)
    fp32 = VisualRuntime(path, 'cuda');mixed = VisualRuntime(path, 'cuda', precision=precision, profile=True)
    obs = VisualReplayBuffer(SharpaDataset(manifest, observation_profile='vision_proprio_v1')).observations([0])
    z = fp32.model.sample_z(1)
    outputs = {}
    def capture(name):
        return lambda module, args, output: outputs.update({name: output.dtype})
    backbone_linear = next(m for m in mixed.model.observation_encoder.backbone.modules() if isinstance(m, torch.nn.Linear))
    hooks = [backbone_linear.register_forward_hook(capture('backbone')),
             mixed.model.observation_encoder.fusion.register_forward_hook(capture('fusion')),
             mixed.model.actor.register_forward_hook(capture('actor'))]
    try:
        a, b = fp32.act(obs,z), mixed.act(obs,z)
    finally:
        for hook in hooks: hook.remove()
    assert outputs == {'backbone': torch.bfloat16 if precision == 'bf16' else torch.float16,
                       'fusion':torch.float32, 'actor':torch.float32}
    assert all(p.dtype == torch.float32 for p in mixed.model.parameters())
    assert b.shape == (26,) and torch.isfinite(b).all() and (b.abs()<=1).all()
    # .01 normalized = <=.016 rad for these arm bounds and .5 logical hand
    # units (half a command unit). A fixture guard, not a trained-policy certificate.
    difference = float((a-b).abs().max())
    assert difference <= .01, f'{precision} normalized action error: {difference}'
    assert mixed.last_profile['vision_backbone_latency_s'] > 0


@pytest.mark.parametrize('flags,precision,profile', [(['--precision','bf16','--profile-inference'],'bf16',True),
                                                     (['--precision','fp32','--no-profile-inference'],'fp32',False)])
def test_ros_cli_precision_overrides(monkeypatch, flags, precision, profile):
    import importlib.util
    import sys
    import bfm.fb.runtime as module
    spec = importlib.util.spec_from_file_location('run_policy_ros', Path(__file__).resolve().parents[2]/'tools/run_policy_ros.py')
    tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
    def parsed(config):
        assert config['inference_precision'] == precision and config['profile_inference'] is profile
        raise RuntimeError('validated CLI overrides before ROS import')
    monkeypatch.setattr(module, 'validate_ros_config', parsed)
    monkeypatch.setattr(sys, 'argv', ['run_policy_ros.py', *flags])
    with pytest.raises(RuntimeError, match='validated CLI'): tool.main()

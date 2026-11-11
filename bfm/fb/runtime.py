"""Frozen visual checkpoint inference with a persistent, explicit task latent."""
from contextlib import contextmanager
import hashlib
import time
from pathlib import Path
import torch
from bfm.contracts.vision import CAMERAS, PROPRIO_DIM
from .checkpoint import load_model


def checkpoint_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_inference_config(precision, profile):
    if precision not in ('fp32', 'bf16', 'fp16') or type(profile) is not bool:
        raise ValueError('Inference requires precision fp32/bf16/fp16 and boolean profile_inference')


class InferenceTimings:
    """Synchronized wall times including CPU dispatch; inactive in production."""
    def __init__(self, device, enabled):
        self.device, self.enabled, self.values = device, enabled, {}

    def now(self):
        if self.enabled and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        start = self.now()
        try:
            yield
        finally:
            self.values[name] = self.now()-start


class VisualRuntime:
    def __init__(self, checkpoint, device='cpu', *, precision='fp32', profile=False):
        validate_inference_config(precision, profile)
        if precision != 'fp32':
            if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
                raise ValueError('Mixed runtime precision requires CUDA')
            with torch.cuda.device(device):
                if precision == 'bf16' and not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError('Native CUDA BF16 is unavailable; use fp16 or fp32')
        self.precision, self.profile, self.last_profile = precision, profile, {}
        self.model, self.payload = load_model(checkpoint, device, expected_profile='vision_proprio_v1')
        if self.model.cfg.action_dim != 26 or self.model.cfg.vision['cameras'] != list(CAMERAS):
            raise ValueError('Runtime requires canonical 26D actions and three-camera order')
        self.checkpoint_sha256 = checkpoint_digest(checkpoint)
        self.z = None

    def require_v4(self):
        from bfm.contracts.sharpa import SharpaV4Contract, execution_configuration
        if self.payload.get('format')!='vanilla_fb_sharpa_v4' or self.payload.get('contract')!=SharpaV4Contract().to_dict():
            raise ValueError('v4 runtime requires v4-trained checkpoint; no legacy relabeling')
        execution_configuration(self.payload['execution'])

    def task(self, value):
        z = torch.as_tensor(value, dtype=torch.float32, device=self.model.device)
        if z.shape == (self.model.cfg.latent_dim,):
            z = z[None]
        if z.shape != (1, self.model.cfg.latent_dim):
            raise ValueError('Task z must be [latent_dim] or [1,latent_dim]')
        return self.model.project_z(z).detach()

    def load_task(self, path, *, require_bound=False):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if require_bound and not isinstance(saved, dict):
            raise ValueError('Evaluation requires checkpoint-bound task metadata')
        if isinstance(saved, dict):
            if saved.get('checkpoint_sha256') != self.checkpoint_sha256:
                raise ValueError('Task z belongs to a different checkpoint')
            saved = saved['z']
        self.z = self.task(saved)
        return self.z

    def observation(self, value):
        if not isinstance(value, dict) or set(value) != {'proprio', *CAMERAS}:
            raise ValueError('Expected only proprio and the three named cameras')
        result = {}
        q = torch.as_tensor(value['proprio'], device=self.model.device)
        if q.shape == (PROPRIO_DIM,):
            q = q[None]
        if q.shape != (1, PROPRIO_DIM) or not torch.isfinite(q).all():
            raise ValueError('Runtime proprio must be finite [116] or [1,116]')
        result['proprio'] = q.float()
        for camera in CAMERAS:
            image = torch.as_tensor(value[camera], device=self.model.device)
            if image.ndim == 3:
                image = image[None]
            if image.dtype != torch.uint8 or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3 or min(image.shape[1:3]) < 1:
                raise ValueError(f'{camera} must be RGB uint8 [H,W,3] or [1,H,W,3]')
            result[camera] = image
        return result

    @torch.no_grad()
    def act(self, observation, z=None):
        from .observation import preprocess_rgb
        timings = InferenceTimings(self.model.device, self.profile)
        self.last_profile = timings.values
        start = timings.now()
        # Runtime-only precision boundary. Keep preprocessing, normalization and
        # the small policy layers in FP32; autocast only the expensive backbone.
        with torch.autocast(self.model.device.type, enabled=False):
            with timings.stage('input_latency_s'):
                task = self.z if z is None else self.task(z)
                if task is None:
                    raise ValueError('Load or supply task z before inference')
                obs = self.observation(observation)
            encoder = self.model.observation_encoder
            cfg = encoder.cfg
            with timings.stage('preprocessing_latency_s'):
                pixels = torch.cat([preprocess_rgb(obs[c], cfg, self.model.device) for c in cfg.cameras])
            with timings.stage('vision_backbone_latency_s'):
                dtype = torch.bfloat16 if self.precision == 'bf16' else torch.float16
                with torch.autocast(self.model.device.type, dtype=dtype, enabled=self.precision != 'fp32'):
                    chunks = []
                    for chunk in pixels.split(cfg.feature_batch_size):
                        output = encoder.backbone(chunk)
                        chunks.append(output if cfg.backbone == 'mock' else output.last_hidden_state[:, 0])
            with timings.stage('feature_handling_latency_s'):
                features = torch.cat(chunks).float()
                if features.shape != (len(cfg.cameras), cfg.feature_dim) or not torch.isfinite(features).all():
                    raise ValueError('Invalid frozen backbone features')
                features = dict(zip(cfg.cameras, features.split(1)))
                proprio = self.model.normalizer(obs['proprio'])
            with timings.stage('fusion_latency_s'):
                state = encoder.fusion(proprio, features)
            with timings.stage('policy_head_latency_s'):
                action = self.model.act_encoded(state, task, deterministic=True)
            if action.shape != (1,26) or not torch.isfinite(action).all() or (action.abs()>1).any():
                raise ValueError('Invalid normalized canonical action')
            result = action[0].cpu()  # Completes GPU work before returning to safety checks.
        if self.profile:
            self.last_profile['total_policy_latency_s'] = timings.now()-start
        return result


def validate_ros_config(config):
    from bfm.contracts.sharpa import SharpaContract, SharpaV4Contract
    v4=config.get('control_contract_version',3)==4
    if config.get('control_contract_version',3) not in (3,4):raise ValueError('Unknown runtime contract')
    if (config['cameras'] != list(CAMERAS) or config['proprio_order'] != 'vision_proprio_v1'
            or config['action_dim'] != 26 or config['normalized_bounds'] != [-1,1]
            or config['physical_bounds'] != 'SharpaContract'
            or config['policy_hz'] != (15 if v4 else 1/SharpaContract().policy_dt)):
        raise ValueError('ROS configuration differs from checkpoint observation/action contract')
    validate_inference_config(config.get('inference_precision', 'fp32'), config.get('profile_inference', False))
    if v4 and (not config.get('use_sim_time') or config['output_topics']!={'action':'/bfm/action'} or config.get('execution_status_topic')!='/bfm/execution_status'):
        raise ValueError('Invalid v4 simulation interface')
    topics=[config['joint_topic'],*config['camera_topics'].values(),*config['output_topics'].values()]
    if set(config['camera_topics'])!=set(CAMERAS) or set(config['output_topics'])!=({'action'} if v4 else {'arm','hand','action'}) or len(set(topics))!=len(topics) or any(not isinstance(t,str) or not t.startswith('/') for t in topics):
        raise ValueError('Invalid or duplicate ROS topics')

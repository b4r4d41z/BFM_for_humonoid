"""Frozen visual checkpoint inference with a persistent, explicit task latent."""
import hashlib
from pathlib import Path
import torch
from bfm.contracts.vision import CAMERAS, PROPRIO_DIM
from .checkpoint import load_model


def checkpoint_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class VisualRuntime:
    def __init__(self, checkpoint, device='cpu'):
        self.model, self.payload = load_model(checkpoint, device, expected_profile='vision_proprio_v1')
        if self.model.cfg.action_dim != 26 or self.model.cfg.vision['cameras'] != list(CAMERAS):
            raise ValueError('Runtime requires canonical 26D actions and three-camera order')
        self.checkpoint_sha256 = checkpoint_digest(checkpoint)
        self.z = None

    def task(self, value):
        z = torch.as_tensor(value, dtype=torch.float32, device=self.model.device)
        if z.shape == (self.model.cfg.latent_dim,):
            z = z[None]
        if z.shape != (1, self.model.cfg.latent_dim):
            raise ValueError('Task z must be [latent_dim] or [1,latent_dim]')
        return self.model.project_z(z).detach()

    def load_task(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
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
        task = self.z if z is None else self.task(z)
        if task is None:
            raise ValueError('Load or supply task z before inference')
        action = self.model.act(self.observation(observation), task, deterministic=True)
        if action.shape != (1,26) or not torch.isfinite(action).all() or (action.abs()>1).any():
            raise ValueError('Invalid normalized canonical action')
        return action[0].cpu()


def validate_ros_config(config):
    from bfm.contracts.sharpa import SharpaContract
    if (config['cameras'] != list(CAMERAS) or config['proprio_order'] != 'vision_proprio_v1'
            or config['action_dim'] != 26 or config['normalized_bounds'] != [-1,1]
            or config['physical_bounds'] != 'SharpaContract'
            or config['policy_hz'] != 1/SharpaContract().policy_dt):
        raise ValueError('ROS configuration differs from checkpoint observation/action contract')
    topics=[config['joint_topic'],*config['camera_topics'].values(),*config['output_topics'].values()]
    if set(config['camera_topics'])!=set(CAMERAS) or set(config['output_topics'])!={'arm','hand','action'} or len(set(topics))!=len(topics) or any(not isinstance(t,str) or not t.startswith('/') for t in topics):
        raise ValueError('Invalid or duplicate ROS topics')

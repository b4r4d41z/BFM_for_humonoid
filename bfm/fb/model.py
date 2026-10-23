from dataclasses import dataclass
import math

import torch
from torch import nn

from bfm.data.normalization import StateNormalizer
from .networks import LatentActor, mlp


@dataclass
class ModelConfig:
    state_dim: int = 28
    action_dim: int = 8
    latent_dim: int = 32
    hidden_dim: int = 256
    hidden_layers: int = 2
    actor_std: float = 0.2
    noise_clip: float = 0.3

    def __post_init__(self):
        if any(x < 1 for x in (self.state_dim, self.action_dim, self.latent_dim, self.hidden_dim, self.hidden_layers)):
            raise ValueError("Network dimensions/layers must be positive")
        if not (0 <= self.actor_std < float('inf') and 0 <= self.noise_clip < float('inf')):
            raise ValueError("Invalid policy noise")


class FBModel(nn.Module):
    """All public state inputs are raw physical state; normalization is internal."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.normalizer = StateNormalizer(cfg.state_dim)
        self.backward = mlp(cfg.state_dim, cfg.latent_dim, cfg.hidden_dim, cfg.hidden_layers)
        self.forwards = nn.ModuleList([
            mlp(cfg.state_dim + cfg.action_dim + cfg.latent_dim, cfg.latent_dim, cfg.hidden_dim, cfg.hidden_layers)
            for _ in range(2)
        ])
        self.actor = LatentActor(cfg.state_dim, cfg.action_dim, cfg.latent_dim, cfg.hidden_dim, cfg.hidden_layers)

    @property
    def device(self):
        return self.normalizer.mean.device

    def _states(self, states):
        x = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        if x.ndim != 2 or x.shape[-1] != self.cfg.state_dim or not torch.isfinite(x).all():
            raise ValueError(f"States must be finite [N,{self.cfg.state_dim}]")
        return self.normalizer(x)

    def backward_map(self, states):
        return self.backward(self._states(states))

    def forward_map(self, states, actions, z):
        x = self._states(states)
        actions, z = actions.to(self.device), z.to(self.device)
        if actions.shape != (len(x), self.cfg.action_dim) or z.shape != (len(x), self.cfg.latent_dim):
            raise ValueError("Forward action/latent batch shape mismatch")
        inp = torch.cat((x, actions, z), -1)
        return torch.stack([f(inp) for f in self.forwards])

    def project_z(self, z):
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if z.ndim != 2 or z.shape[-1] != self.cfg.latent_dim or not torch.isfinite(z).all():
            raise ValueError("Latents must be finite [N,latent_dim]")
        norm = z.norm(dim=-1, keepdim=True)
        if (norm < 1e-8).any():
            raise ValueError("Zero reward/embedding produced no latent direction; choose a supported task/reference set")
        return math.sqrt(self.cfg.latent_dim) * z / norm

    def sample_z(self, batch_size):
        return self.project_z(torch.randn(batch_size, self.cfg.latent_dim, device=self.device))

    def act(self, states, z, deterministic=True):
        x = self._states(states)
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if z.shape != (len(x), self.cfg.latent_dim) or not torch.isfinite(z).all():
            raise ValueError("Actor needs one finite latent per state")
        return self.actor(x, z, std=0. if deterministic else self.cfg.actor_std, noise_clip=self.cfg.noise_clip)

    def goal_inference(self, goal_states):
        return self.project_z(self.backward_map(goal_states))

    def reward_inference(self, states, rewards, batch_size=4096):
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        if rewards.shape != (len(states),) or not len(states) or not torch.isfinite(rewards).all():
            raise ValueError("Rewards must be finite [N] aligned with reference states")
        # Sum and mean have identical projected directions. No inverse covariance
        # is used: the backward representation is trained with orthogonality loss.
        z = torch.zeros(1, self.cfg.latent_dim, device=self.device)
        for start in range(0, len(states), batch_size):
            b = self.backward_map(states[start:start + batch_size])
            z = z + (b * rewards[start:start + batch_size, None]).sum(0, keepdim=True) / len(states)
        return self.project_z(z)

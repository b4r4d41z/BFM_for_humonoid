from dataclasses import dataclass, field, asdict
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
    observation_profile: str = 'task56'
    vision: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.observation_profile not in ('task56', 'vision_proprio_v1'):
            raise ValueError('Unknown observation_profile')
        if self.observation_profile == 'task56' and self.vision:
            raise ValueError('task56 cannot have visual configuration')
        if any(x < 1 for x in (self.state_dim, self.action_dim, self.latent_dim, self.hidden_dim, self.hidden_layers)):
            raise ValueError("Network dimensions/layers must be positive")
        if not (0 <= self.actor_std < float('inf') and 0 <= self.noise_clip < float('inf')):
            raise ValueError("Invalid policy noise")


class FBModel(nn.Module):
    """All public state inputs are raw physical state; normalization is internal."""
    def __init__(self, cfg: ModelConfig, *, initialize_pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.observation_encoder = None
        if cfg.observation_profile == 'vision_proprio_v1':
            from .observation import ObservationEncoder
            from bfm.contracts.vision import PROPRIO_DIM
            self.observation_encoder = ObservationEncoder(cfg.vision, initialize_pretrained=initialize_pretrained)
            cfg.vision = asdict(self.observation_encoder.cfg)
            if cfg.state_dim != self.observation_encoder.cfg.embedding_dim:
                raise ValueError('Visual state_dim must equal observation embedding dimension')
            self.normalizer = StateNormalizer(PROPRIO_DIM)
        else:
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
        if self.observation_encoder is not None:
            return self.encode_observation(states)
        x = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        if x.ndim != 2 or x.shape[-1] != self.cfg.state_dim or not torch.isfinite(x).all():
            raise ValueError(f"States must be finite [N,{self.cfg.state_dim}]")
        return self.normalizer(x)

    def encode_observation(self, observation, *, fusion=None):
        proprio, features = self.prepare_observation(observation)
        return (self.observation_encoder.fusion if fusion is None else fusion)(proprio, features)

    def prepare_observation(self, observation):
        cameras = self.observation_encoder.cfg.cameras
        cached = isinstance(observation, dict) and set(observation) == {'proprio', *[f'features/{c}' for c in cameras]}
        if not isinstance(observation, dict) or (not cached and set(observation) != {'proprio', *cameras}):
            raise ValueError('Visual input requires only measured proprio and configured named cameras')
        from bfm.contracts.vision import PROPRIO_DIM
        proprio = torch.as_tensor(observation['proprio'], dtype=torch.float32, device=self.device)
        if proprio.ndim != 2 or proprio.shape[1] != PROPRIO_DIM or not torch.isfinite(proprio).all():
            raise ValueError('Visual proprioception must be finite [N,116]')
        if cached:
            features = {}
            for camera in cameras:
                value = torch.as_tensor(observation[f'features/{camera}'], device=self.device)
                if value.dtype != torch.float32 or value.shape != (len(proprio), self.observation_encoder.cfg.feature_dim) or not torch.isfinite(value).all():
                    raise ValueError('Cached DINO features must be finite float32 [N,feature_dim]')
                features[camera] = value.detach()
        else:
            features = self.observation_encoder.features(observation)
        return self.normalizer(proprio), features

    def backward_map(self, states):
        return self.backward(self._states(states))

    def forward_map(self, states, actions, z):
        x = self._states(states)
        return self.forward_encoded(x, actions, z)

    def forward_encoded(self, x, actions, z):
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
        return self.act_encoded(x, z, deterministic)

    def act_encoded(self, x, z, deterministic=True):
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if z.shape != (len(x), self.cfg.latent_dim) or not torch.isfinite(z).all():
            raise ValueError("Actor needs one finite latent per state")
        return self.actor(x, z, std=0. if deterministic else self.cfg.actor_std, noise_clip=self.cfg.noise_clip)

    def goal_inference(self, goal_states):
        return self.project_z(self.backward_map(goal_states))

    def reward_inference(self, states, rewards, batch_size=4096):
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        count = len(states['proprio']) if isinstance(states, dict) else len(states)
        if rewards.shape != (count,) or not count or not torch.isfinite(rewards).all():
            raise ValueError("Rewards must be finite [N] aligned with reference states")
        # Sum and mean have identical projected directions. No inverse covariance
        # is used: the backward representation is trained with orthogonality loss.
        z = torch.zeros(1, self.cfg.latent_dim, device=self.device)
        for start in range(0, count, batch_size):
            batch = ({k: v[start:start+batch_size] for k,v in states.items()} if isinstance(states, dict)
                     else states[start:start+batch_size])
            b = self.backward_map(batch)
            z = z + (b * rewards[start:start + batch_size, None]).sum(0, keepdim=True) / count
        return self.project_z(z)

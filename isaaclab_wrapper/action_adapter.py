from __future__ import annotations

import torch


class ActionAdapter:
    def __init__(
        self,
        expected_action_dim: int | None = None,
        env_device: str = "cuda:0",
        action_scale: float = 1.0,
        clip_actions: bool = True,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        debug: bool = False,
    ):
        self.expected_action_dim = expected_action_dim
        self.env_device = torch.device(env_device)
        self.action_scale = float(action_scale)
        self.clip_actions = bool(clip_actions)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.debug = debug

    def __call__(self, model_action: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(model_action):
            raise TypeError(f"model_action must be torch.Tensor, got {type(model_action).__name__}")

        env_action = model_action.to(device=self.env_device, dtype=torch.float32)

        if env_action.ndim == 1:
            env_action = env_action.unsqueeze(0)
        elif env_action.ndim > 2:
            env_action = env_action.reshape(env_action.shape[0], -1)

        if env_action.ndim != 2:
            raise ValueError(f"env_action must be 2D [num_envs, action_dim], got {tuple(env_action.shape)}")

        if self.expected_action_dim is not None and int(env_action.shape[-1]) != int(self.expected_action_dim):
            raise ValueError(
                f"Action dim mismatch: expected_action_dim={self.expected_action_dim}, got={int(env_action.shape[-1])}"
            )

        # Identity mapping for now.
        # TODO: support joint re-ordering and robot-specific arm/hand channel mapping.
        env_action = env_action * self.action_scale
        # TODO: support per-joint scaling based on robot limits if required.

        if self.clip_actions:
            env_action = torch.clamp(env_action, min=self.clip_min, max=self.clip_max)

        if self.debug:
            print(f"[ActionAdapter] model_action_shape={tuple(model_action.shape)} env_action_shape={tuple(env_action.shape)}")

        return env_action

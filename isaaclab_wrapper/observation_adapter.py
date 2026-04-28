from __future__ import annotations

from typing import Any

import torch


class ObservationAdapter:
    def __init__(self, expected_obs_dim: int | None = None, device: str = "cuda:0", debug: bool = False):
        self.expected_obs_dim = expected_obs_dim
        self.device = torch.device(device)
        self.debug = debug

    def _extract_obs_tensor(self, raw_obs: Any) -> torch.Tensor:
        if isinstance(raw_obs, torch.Tensor):
            return raw_obs

        if isinstance(raw_obs, dict):
            # Common IsaacLab observation keys for policy/state tensors.
            for key in ("policy", "obs", "state"):
                if key in raw_obs:
                    value = raw_obs[key]
                    if isinstance(value, torch.Tensor):
                        return value
                    if isinstance(value, dict):
                        for nested_key in ("full", "state", "policy"):
                            if nested_key in value and isinstance(value[nested_key], torch.Tensor):
                                return value[nested_key]

            # Fallback: first tensor found by DFS.
            stack: list[Any] = [raw_obs]
            while stack:
                item = stack.pop()
                if isinstance(item, torch.Tensor):
                    return item
                if isinstance(item, dict):
                    stack.extend(item.values())
                elif isinstance(item, (list, tuple)):
                    stack.extend(item)

        raise TypeError(
            "Unsupported observation format. Expected tensor or dict containing tensor keys like "
            "['policy', 'obs', 'state']"
        )

    def __call__(self, raw_obs: Any) -> torch.Tensor:
        obs = self._extract_obs_tensor(raw_obs)
        obs = obs.to(device=self.device, dtype=torch.float32)

        # Ensure [num_envs, obs_dim].
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        elif obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)

        if obs.ndim != 2:
            raise ValueError(f"Model observation must be 2D [num_envs, obs_dim], got {tuple(obs.shape)}")

        if self.expected_obs_dim is not None and int(obs.shape[-1]) != int(self.expected_obs_dim):
            raise ValueError(
                f"Observation dim mismatch: expected_obs_dim={self.expected_obs_dim}, got={int(obs.shape[-1])}"
            )

        if self.debug:
            print(f"[ObservationAdapter] raw_type={type(raw_obs).__name__} model_obs_shape={tuple(obs.shape)}")

        # TODO: add explicit multi-camera/image extraction + preprocessing path for future vision policies.
        return obs

from __future__ import annotations

from typing import Any

import torch


class ObservationAdapter:
    def __init__(
        self,
        expected_obs_dim: int | None = None,
        device: str = "cuda:0",
        debug: bool = False,
        auto_adjust_dim: bool = True,
    ):
        self.expected_obs_dim = expected_obs_dim
        self.device = torch.device(device)
        self.debug = debug
        self.auto_adjust_dim = auto_adjust_dim

    def _extract_obs_tensor(self, raw_obs: Any) -> torch.Tensor:
        candidates: list[torch.Tensor] = []

        def _collect_tensor(x: Any) -> None:
            if isinstance(x, torch.Tensor):
                candidates.append(x)

        if isinstance(raw_obs, torch.Tensor):
            return raw_obs

        if isinstance(raw_obs, dict):
            # Common IsaacLab observation keys for policy/state tensors.
            for key in ("policy", "obs", "state"):
                if key in raw_obs:
                    value = raw_obs[key]
                    if isinstance(value, torch.Tensor):
                        candidates.append(value)
                    if isinstance(value, dict):
                        for nested_key in ("full", "state", "policy"):
                            if nested_key in value and isinstance(value[nested_key], torch.Tensor):
                                candidates.append(value[nested_key])

            # Fallback: first tensor found by DFS.
            stack: list[Any] = [raw_obs]
            while stack:
                item = stack.pop()
                if isinstance(item, torch.Tensor):
                    candidates.append(item)
                if isinstance(item, dict):
                    stack.extend(item.values())
                elif isinstance(item, (list, tuple)):
                    stack.extend(item)

        _collect_tensor(raw_obs)
        if candidates:
            # Prefer exact dim match when expected_obs_dim is known.
            if self.expected_obs_dim is not None:
                for tensor in candidates:
                    if tensor.ndim >= 1 and int(tensor.shape[-1]) == int(self.expected_obs_dim):
                        return tensor
            return candidates[0]

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
            got_dim = int(obs.shape[-1])
            want_dim = int(self.expected_obs_dim)

            if self.auto_adjust_dim and got_dim > want_dim:
                if self.debug:
                    print(
                        f"[ObservationAdapter] auto-trim obs dim from {got_dim} to {want_dim} "
                        "(keeping first dims)"
                    )
                obs = obs[:, :want_dim]
            elif self.auto_adjust_dim and got_dim < want_dim:
                if self.debug:
                    print(
                        f"[ObservationAdapter] auto-pad obs dim from {got_dim} to {want_dim} "
                        "(zero padding)"
                    )
                pad = torch.zeros((obs.shape[0], want_dim - got_dim), dtype=obs.dtype, device=obs.device)
                obs = torch.cat([obs, pad], dim=-1)
            else:
                raise ValueError(
                    f"Observation dim mismatch: expected_obs_dim={self.expected_obs_dim}, got={int(obs.shape[-1])}"
                )

        if self.debug:
            print(f"[ObservationAdapter] raw_type={type(raw_obs).__name__} model_obs_shape={tuple(obs.shape)}")
            if isinstance(raw_obs, dict):
                print(f"[ObservationAdapter] raw_obs_keys={list(raw_obs.keys())}")

        # TODO: add explicit multi-camera/image extraction + preprocessing path for future vision policies.
        return obs

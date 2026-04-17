# isaaclab_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch


ObsType = Union[torch.Tensor, np.ndarray, Dict[str, Any]]
InfoType = Dict[str, Any]


def _is_tensor_like(x: Any) -> bool:
    return isinstance(x, (torch.Tensor, np.ndarray))


def _to_torch(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert numpy/tensor/scalar to torch tensor on device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if isinstance(x, np.ndarray):
        # torch.from_numpy keeps dtype; cast below
        t = torch.from_numpy(x)
        return t.to(device=device, dtype=dtype)
    # scalars / lists
    return torch.as_tensor(x, device=device, dtype=dtype)


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten everything after batch dim.
    Expects shape: [B, ...] and returns [B, N].
    """
    if x.ndim == 1:
        return x.unsqueeze(-1)
    return x.reshape(x.shape[0], -1)


def _flatten_dict_obs(
    obs_dict: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    keys: Optional[Tuple[str, ...]] = None,
) -> torch.Tensor:
    """
    Convert dict of tensors to a single [B, N] tensor by concatenating flattened parts.
    If keys is None, uses stable key ordering (sorted).
    """
    if keys is None:
        keys = tuple(sorted(obs_dict.keys()))

    parts = []
    for k in keys:
        v = obs_dict[k]
        if isinstance(v, dict):
            # Nested dicts are flattened recursively with sorted keys
            v = _flatten_dict_obs(v, device=device, dtype=dtype, keys=None)
        else:
            v = _to_torch(v, device=device, dtype=dtype)
            # If no batch dim, add one
            if v.ndim == 0:
                v = v.view(1, 1)
            elif v.ndim == 1:
                # Ambiguous: could be [B] or [N]. Assume [B] if matches num_envs later.
                v = v.unsqueeze(-1)
            v = _flatten_batch(v)
        parts.append(v)

    if len(parts) == 0:
        raise ValueError("Observation dict is empty; cannot build policy observation.")

    # Ensure same batch dimension
    b0 = parts[0].shape[0]
    for p in parts[1:]:
        if p.shape[0] != b0:
            raise ValueError(f"Batch size mismatch in obs dict: {b0} vs {p.shape[0]}")

    return torch.cat(parts, dim=1)


@dataclass
class StepOutput:
    obs: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    info: InfoType


class IsaacLabEnvWrapper:
    """
    A lightweight adapter around an IsaacLab vectorized environment.

    Responsibilities:
      - Normalize reset/step outputs to torch tensors on env device.
      - Extract "policy observation" from dict observations (common in IsaacLab).
      - Provide a consistent (obs, reward, done, info) interface for BFM code.

    Notes:
      - Supports Gymnasium-style step: (obs, reward, terminated, truncated, info)
        and Gym-style step: (obs, reward, done, info).
      - If obs is a dict, default priority:
          1) obs_key if provided
          2) "policy" key if exists
          3) single-key dict -> that value
          4) else -> concatenate all keys (sorted) into one flat tensor
    """

    def __init__(
        self,
        env: Any,
        obs_key: Optional[str] = None,
        flatten_dict: bool = True,
        clip_actions: Optional[Tuple[float, float]] = None,
        obs_dtype: torch.dtype = torch.float32,
        act_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.env = env
        self.obs_key = obs_key
        self.flatten_dict = flatten_dict
        self.clip_actions = clip_actions
        self.obs_dtype = obs_dtype
        self.act_dtype = act_dtype

        self._device = self._infer_device(env)
        self._num_envs = self._infer_num_envs(env)

    @staticmethod
    def _infer_device(env: Any) -> torch.device:
        d = getattr(env, "device", None)
        if d is None:
            return torch.device("cpu")
        if isinstance(d, torch.device):
            return d
        return torch.device(str(d))

    @staticmethod
    def _infer_num_envs(env: Any) -> int:
        n = getattr(env, "num_envs", None)
        if n is None:
            return 1
        return int(n)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[torch.Tensor, InfoType]:
        # IsaacLab often follows gymnasium: obs, info = env.reset()
        if seed is None and options is None:
            out = self.env.reset()
        else:
            # Be defensive: some envs accept seed/options, some don't.
            try:
                out = self.env.reset(seed=seed, options=options)
            except TypeError:
                out = self.env.reset()

        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        obs_t = self._extract_policy_obs(obs)
        return obs_t, info

    def step(self, action: Union[torch.Tensor, np.ndarray]) -> StepOutput:
        act = _to_torch(action, device=self.device, dtype=self.act_dtype)

        if self.clip_actions is not None:
            lo, hi = self.clip_actions
            act = torch.clamp(act, lo, hi)

        out = self.env.step(act)

        # Gymnasium: obs, reward, terminated, truncated, info
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = _to_torch(terminated, self.device, torch.bool) | _to_torch(truncated, self.device, torch.bool)
        # Gym: obs, reward, done, info
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            done = _to_torch(done, self.device, torch.bool)
        else:
            raise ValueError(f"Unexpected env.step() output format: type={type(out)} value={out}")

        obs_t = self._extract_policy_obs(obs)
        reward_t = _to_torch(reward, self.device, self.obs_dtype)

        # Ensure [B] shape for reward/done when vectorized
        if self.num_envs > 1:
            if reward_t.ndim == 0:
                reward_t = reward_t.repeat(self.num_envs)
            reward_t = reward_t.view(self.num_envs)
            done = done.view(self.num_envs)

        return StepOutput(obs=obs_t, reward=reward_t, done=done, info=info if isinstance(info, dict) else {})

    def _extract_policy_obs(self, obs: ObsType) -> torch.Tensor:
        # Direct tensor/ndarray
        if _is_tensor_like(obs):
            t = _to_torch(obs, device=self.device, dtype=self.obs_dtype)
            # If scalar, make [1,1]
            if t.ndim == 0:
                t = t.view(1, 1)
            # If [N] and vectorized, treat as [B,N]? unclear; keep as [1,N] in single env.
            if t.ndim == 1 and self.num_envs == 1:
                t = t.unsqueeze(0)
            return _flatten_batch(t) if t.ndim >= 2 else t

        # Dict obs (common in IsaacLab)
        if isinstance(obs, dict):
            # priority: explicit key -> "policy" -> single-key -> concat
            if self.obs_key is not None and self.obs_key in obs:
                return self._extract_policy_obs(obs[self.obs_key])

            if "policy" in obs:
                return self._extract_policy_obs(obs["policy"])

            if len(obs) == 1:
                only_val = next(iter(obs.values()))
                return self._extract_policy_obs(only_val)

            if not self.flatten_dict:
                raise ValueError(
                    "Observation is a dict with multiple keys, but flatten_dict=False. "
                    "Set obs_key or enable flatten_dict."
                )

            t = _flatten_dict_obs(obs, device=self.device, dtype=self.obs_dtype, keys=None)
            return t

        raise TypeError(f"Unsupported observation type: {type(obs)}")

    # Optional helpers (useful later in training script)
    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()
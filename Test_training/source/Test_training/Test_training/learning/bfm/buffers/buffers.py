from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class Batch:
    obs: torch.Tensor        # [B, obs_dim]
    act: torch.Tensor        # [B, act_dim]
    rew: torch.Tensor        # [B]
    done: torch.Tensor       # [B] bool
    next_obs: torch.Tensor   # [B, obs_dim]


class ReplayBuffer:
    """
    Simple ring replay buffer for vectorized RL envs.

    Stores flattened observations/actions:
      obs:      [N, obs_dim]
      act:      [N, act_dim]
      rew:      [N]
      done:     [N]
      next_obs: [N, obs_dim]
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        obs_dim: int,
        act_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self.device = device
        self.dtype = dtype

        self.obs = torch.zeros((capacity, obs_dim), device=device, dtype=dtype)
        self.act = torch.zeros((capacity, act_dim), device=device, dtype=dtype)
        self.rew = torch.zeros((capacity,), device=device, dtype=dtype)
        self.done = torch.zeros((capacity,), device=device, dtype=torch.bool)
        self.next_obs = torch.zeros((capacity, obs_dim), device=device, dtype=dtype)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """
        Supports vectorized add:
          obs:      [B, obs_dim]
          act:      [B, act_dim]
          rew:      [B] or [B,1]
          done:     [B] bool
          next_obs: [B, obs_dim]
        """
        if obs.ndim != 2 or next_obs.ndim != 2:
            raise ValueError(f"obs/next_obs must be [B, obs_dim], got {obs.shape} / {next_obs.shape}")
        if act.ndim != 2:
            raise ValueError(f"act must be [B, act_dim], got {act.shape}")

        rew = rew.view(-1).to(device=self.device, dtype=self.dtype)
        done = done.view(-1).to(device=self.device, dtype=torch.bool)

        b = obs.shape[0]
        if act.shape[0] != b or next_obs.shape[0] != b or rew.shape[0] != b or done.shape[0] != b:
            raise ValueError("Batch size mismatch in add()")

        obs = obs.to(device=self.device, dtype=self.dtype)
        act = act.to(device=self.device, dtype=self.dtype)
        next_obs = next_obs.to(device=self.device, dtype=self.dtype)

        for i in range(b):
            self.obs[self._ptr] = obs[i]
            self.act[self._ptr] = act[i]
            self.rew[self._ptr] = rew[i]
            self.done[self._ptr] = done[i]
            self.next_obs[self._ptr] = next_obs[i]

            self._ptr = (self._ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty buffer")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return Batch(
            obs=self.obs[idx],
            act=self.act[idx],
            rew=self.rew[idx],
            done=self.done[idx],
            next_obs=self.next_obs[idx],
        )

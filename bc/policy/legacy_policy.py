from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    activation: type[nn.Module] = nn.ReLU,
    out_activation: Optional[type[nn.Module]] = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(activation())
        last = h
    layers.append(nn.Linear(last, out_dim))
    if out_activation is not None:
        layers.append(out_activation())
    return nn.Sequential(*layers)


class MLPEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden: Sequence[int], latent_dim: int) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden, latent_dim, activation=nn.ReLU, out_activation=None)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TanhDiagGaussianPolicy(nn.Module):
    """
    Minimal continuous policy:
      mean = MLP(obs)
      std  = exp(log_std)
      action = tanh(mean + std * eps)  (optional)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (256, 256),
        log_std_init: float = -0.5,
        use_tanh: bool = True,
    ) -> None:
        super().__init__()
        self.mean_net = mlp(obs_dim, hidden, act_dim, activation=nn.ReLU, out_activation=None)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.use_tanh = bool(use_tanh)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_net(obs)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        if deterministic:
            a = mean
        else:
            std = torch.exp(log_std)
            a = mean + std * torch.randn_like(mean)
        if self.use_tanh:
            a = torch.tanh(a)
        return a

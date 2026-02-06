from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from ..nn_models import TanhDiagGaussianPolicy


@dataclass
class ModelConfig:
    obs_dim: int
    act_dim: int
    hidden: Sequence[int] = (256, 256)
    log_std_init: float = -0.5
    use_tanh: bool = True


class FBCPRModel(nn.Module):
    """
    Minimal placeholder for "FB-CPR style" model:
    for now it is just a stochastic continuous policy network.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.policy = TanhDiagGaussianPolicy(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            hidden=cfg.hidden,
            log_std_init=cfg.log_std_init,
            use_tanh=cfg.use_tanh,
        )

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.policy.act(obs, deterministic=deterministic)
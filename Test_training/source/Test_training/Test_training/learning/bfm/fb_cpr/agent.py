from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .model import FBCPRModel, ModelConfig


@dataclass
class AgentConfig:
    device: str = "cuda"
    lr: float = 3e-4  # for future
    deterministic_eval: bool = True


class FBCPRAgent:
    """
    Minimal agent wrapper:
      - act(obs) -> action
      - save/load checkpoints
      - update(batch) is a stub for now
    """

    def __init__(self, model_cfg: ModelConfig, agent_cfg: AgentConfig) -> None:
        self.model_cfg = model_cfg
        self.agent_cfg = agent_cfg
        self.device = torch.device(agent_cfg.device)

        self.model = FBCPRModel(model_cfg).to(self.device)
        self.model.eval()

        # placeholder optimizer (not used until real update is implemented)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(agent_cfg.lr))

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: Optional[bool] = None) -> torch.Tensor:
        if deterministic is None:
            deterministic = bool(self.agent_cfg.deterministic_eval)
        obs = obs.to(self.device)
        return self.model.act(obs, deterministic=deterministic)

    def update(self, batch: Any) -> Dict[str, float]:
        """
        Stub. Later: implement FB-CPR losses here.
        """
        return {"loss": 0.0}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_cfg": self.model_cfg.__dict__,
            "agent_cfg": self.agent_cfg.__dict__,
            "state_dict": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }
        torch.save(payload, str(path))

    def load(self, path: str | Path, map_location: Optional[str] = None) -> None:
        path = Path(path)
        if map_location is None:
            map_location = str(self.device)
        payload = torch.load(str(path), map_location=map_location)
        self.model.load_state_dict(payload["state_dict"])
        if "optim_state" in payload:
            self.optimizer.load_state_dict(payload["optim_state"])

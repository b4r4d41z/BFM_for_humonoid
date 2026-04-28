from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch


class RolloutRecorder:
    def __init__(self, enabled: bool = False, output_dir: str | None = None):
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir or "runs/bc/isaaclab_rollouts")
        self.steps: list[dict[str, Any]] = []

    def _to_cpu(self, x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        if isinstance(x, dict):
            return {k: self._to_cpu(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self._to_cpu(v) for v in x]
        if isinstance(x, tuple):
            return tuple(self._to_cpu(v) for v in x)
        return x

    def log_step(
        self,
        step: int,
        obs: Any = None,
        model_action: Any = None,
        env_action: Any = None,
        reward: Any = None,
        done: Any = None,
        info: Any = None,
    ) -> None:
        if not self.enabled:
            return

        item: dict[str, Any] = {
            "step": int(step),
            "obs_shape": tuple(obs.shape) if isinstance(obs, torch.Tensor) else None,
            "model_action": self._to_cpu(model_action),
            "env_action": self._to_cpu(env_action),
            "reward": self._to_cpu(reward),
            "done": self._to_cpu(done),
            "info": self._to_cpu(info),
        }
        self.steps.append(item)

    def save(self) -> Path | None:
        if not self.enabled:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"rollout_{stamp}.pt"
        torch.save({"steps": self.steps}, out_path)
        print(f"[RolloutRecorder] Saved rollout to: {out_path}")
        return out_path

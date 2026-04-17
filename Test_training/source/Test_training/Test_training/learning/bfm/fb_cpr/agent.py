from __future__ import annotations

import dataclasses
from typing import Any

import torch
import torch.nn.functional as F

from .model import FBCPRModel, ModelConfig


@dataclasses.dataclass
class TrainConfig:
    """Training hyperparameters for the offline BC stage."""

    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0
    batch_size: int = 256
    action_loss: str = "mse"  # {"mse", "smooth_l1"}


@dataclasses.dataclass
class AgentConfig:
    """Top-level agent config for offline training."""

    device: str = "cpu"
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    compile: bool = False


class FBCPRAgent:
    """
    Minimal offline agent for the first vision-aware policy stage.

    Responsibilities on this stage:
      - hold model + optimizer
      - compute supervised BC loss against batch["action"]["full"]
      - run one optimizer step

    Future FB-CPR-specific logic (latent z, discriminator, relabeling, etc.)
    can be added later without breaking the current batch contract.
    """

    def __init__(self, model_cfg: ModelConfig | None = None, agent_cfg: AgentConfig | None = None) -> None:
        self.model_cfg = model_cfg or ModelConfig()
        self.cfg = agent_cfg or AgentConfig()

        self._device = torch.device(self.cfg.device)
        self._model = FBCPRModel(self.model_cfg).to(self._device)

        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )

        if self.cfg.compile:
            self._model = torch.compile(self._model)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> FBCPRModel:
        return self._model

    def train(self, mode: bool = True) -> None:
        self._model.train(mode)

    def eval(self) -> None:
        self._model.eval()

    def _to_device(self, x: Any) -> Any:
        if isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self._to_device(v) for v in x]
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return x

    def _flatten_bt(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten [B, T, ...] -> [B*T, ...]. Keep [B, ...] unchanged."""
        if x.ndim < 2:
            return x
        if x.ndim >= 3 and x.shape[0] > 0 and x.shape[1] > 0:
            # sequence tensors in this project are [B, T, ...]
            return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        return x

    def _is_sequence_batch(self, action_full: torch.Tensor) -> bool:
        return action_full.ndim == 3

    def _compute_action_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.cfg.train.action_loss == "mse":
            return F.mse_loss(pred, target)
        if self.cfg.train.action_loss == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        raise ValueError(f"Unsupported action_loss: {self.cfg.train.action_loss}")

    def _prepare_batch_for_model(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Convert both transition and sequence batches to model-ready transition form.

        Transition batch contract (unchanged):
          obs.state.full: [B, 26]
          obs.images.* : [B, H, W, C]
          action.full  : [B, 26]

        Sequence batch contract:
          same keys with leading [B, T, ...] dims.
        """
        batch = self._to_device(batch)

        action_full = batch["action"]["full"]
        if not self._is_sequence_batch(action_full):
            return batch

        model_batch = {
            "obs": {
                "state": {
                    "full": self._flatten_bt(batch["obs"]["state"]["full"]),
                },
                "images": {
                    "head": self._flatten_bt(batch["obs"]["images"]["head"]),
                    "left_wrist": self._flatten_bt(batch["obs"]["images"]["left_wrist"]),
                    "right_wrist": self._flatten_bt(batch["obs"]["images"]["right_wrist"]),
                },
            },
            "action": {
                "full": self._flatten_bt(batch["action"]["full"]),
            },
        }
        return model_batch

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """One supervised offline BC update step."""
        self.train(True)

        model_batch = self._prepare_batch_for_model(batch)
        target_action = model_batch["action"]["full"]

        pred_action = self._model(model_batch)
        loss = self._compute_action_loss(pred_action, target_action)

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm_value = 0.0
        if self.cfg.train.grad_clip_norm > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.cfg.train.grad_clip_norm)
            grad_norm_value = float(grad_norm.detach().item())

        self._optimizer.step()

        with torch.no_grad():
            mae = torch.mean(torch.abs(pred_action - target_action))

        return {
            "loss": float(loss.detach().item()),
            "action_mae": float(mae.detach().item()),
            "grad_norm": grad_norm_value,
        }

    @torch.no_grad()
    def act(
        self,
        batch: dict[str, Any],
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Predict action.full for a nested obs batch.

        `deterministic` is kept for API compatibility.
        """
        _ = deterministic
        self.eval()
        batch = self._to_device(batch)
        return self._model.act(batch)

    def save(self, path: str) -> None:
        payload = {
            "model_state": self._model.state_dict(),
            "optimizer_state": self._optimizer.state_dict(),
            "model_cfg": dataclasses.asdict(self.model_cfg),
            "agent_cfg": dataclasses.asdict(self.cfg),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self._model.load_state_dict(ckpt["model_state"])

        if "optimizer_state" in ckpt:
            self._optimizer.load_state_dict(ckpt["optimizer_state"])


# Backward-compatible aliases used by some old scripts
FBcprAgent = FBCPRAgent

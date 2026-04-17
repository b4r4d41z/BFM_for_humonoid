from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..model_blocks import (
    ConcatFusion,
    DeterministicPolicyHead,
    MLPConfig,
    MultiViewVisionEncoder,
    StateEncoder,
    VisionEncoderConfig,
)


@dataclass
class ModelConfig:
    """Configuration for the first vision-aware offline BFM policy."""

    state_dim: int = 26
    action_dim: int = 26

    state_feat_dim: int = 256
    per_view_feat_dim: int = 128
    vision_feat_dim: int = 256
    fused_feat_dim: int = 256

    state_encoder_cfg: MLPConfig = field(
        default_factory=lambda: MLPConfig(hidden_dim=256, hidden_layers=2, activation="mish")
    )
    vision_encoder_cfg: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    vision_fusion_cfg: MLPConfig = field(
        default_factory=lambda: MLPConfig(hidden_dim=256, hidden_layers=2, activation="mish")
    )
    fusion_cfg: MLPConfig = field(
        default_factory=lambda: MLPConfig(hidden_dim=256, hidden_layers=2, activation="mish")
    )
    policy_head_cfg: MLPConfig = field(
        default_factory=lambda: MLPConfig(hidden_dim=256, hidden_layers=2, activation="mish")
    )
    squash_output: bool = False


class FBCPRModel(nn.Module):
    """
    Minimal vision-aware offline policy model.

    The model intentionally keeps only the core behavior-cloning path:
      1) Encode proprioceptive state (obs.state.full)
      2) Encode three camera views (head/left_wrist/right_wrist)
      3) Fuse proprio + vision features
      4) Predict deterministic action.full

    This is designed to be easy to extend later toward full FB-CPR logic.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.state_encoder = StateEncoder(
            input_dim=cfg.state_dim,
            output_dim=cfg.state_feat_dim,
            cfg=cfg.state_encoder_cfg,
        )

        self.vision_encoder = MultiViewVisionEncoder(
            view_encoder_out_dim=cfg.per_view_feat_dim,
            fused_out_dim=cfg.vision_feat_dim,
            view_cfg=cfg.vision_encoder_cfg,
            fusion_cfg=cfg.vision_fusion_cfg,
        )

        self.fusion = ConcatFusion(
            input_dim=cfg.state_feat_dim + cfg.vision_feat_dim,
            output_dim=cfg.fused_feat_dim,
            cfg=cfg.fusion_cfg,
        )

        self.policy_head = DeterministicPolicyHead(
            input_dim=cfg.fused_feat_dim,
            action_dim=cfg.action_dim,
            cfg=cfg.policy_head_cfg,
            squash_output=cfg.squash_output,
        )

    def _extract_inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract required tensors from the nested training batch dict."""
        obs = batch["obs"]
        state_full = obs["state"]["full"]
        images = obs["images"]
        head = images["head"]
        left_wrist = images["left_wrist"]
        right_wrist = images["right_wrist"]
        return state_full, head, left_wrist, right_wrist

    def forward(self, batch: dict[str, dict[str, dict[str, torch.Tensor]]]) -> torch.Tensor:
        """
        Predict action.full from state + multi-view images.

        Expected batch keys:
            batch["obs"]["state"]["full"]        -> [B, 26]
            batch["obs"]["images"]["head"]       -> [B, H, W, C] or [B, C, H, W]
            batch["obs"]["images"]["left_wrist"] -> [B, H, W, C] or [B, C, H, W]
            batch["obs"]["images"]["right_wrist"]-> [B, H, W, C] or [B, C, H, W]

        Returns:
            Predicted action.full tensor with shape [B, action_dim].
        """
        state_full, head, left_wrist, right_wrist = self._extract_inputs(batch)

        state_feat = self.state_encoder(state_full)
        vision_feat = self.vision_encoder(head, left_wrist, right_wrist)
        fused_feat = self.fusion(state_feat, vision_feat)

        return self.policy_head(fused_feat)

    @torch.no_grad()
    def act(self, batch: dict[str, dict[str, dict[str, torch.Tensor]]]) -> torch.Tensor:
        """Inference helper equivalent to forward() for deterministic policy output."""
        return self.forward(batch)

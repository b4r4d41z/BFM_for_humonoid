from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bc.fb_cpr.model import ModelConfig
from bc.model_blocks import MLPConfig, VisionEncoderConfig
from bc.nn_models import TanhDiagGaussianPolicy


class BCPolicyRunner:
    def __init__(self, checkpoint_path: str, device: str = "cuda:0", debug: bool = False, **model_kwargs):
        self.device = torch.device(device)
        self.debug = debug
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")

        payload = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict, meta = self._extract_state_dict(payload)

        self.model, self.model_kind = self._build_model(state_dict=state_dict, meta=meta, model_kwargs=model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        if self.debug:
            print(f"[BCPolicyRunner] loaded checkpoint={self.checkpoint_path}")
            print(f"[BCPolicyRunner] model_kind={self.model_kind}")

    def _extract_state_dict(self, payload: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if isinstance(payload, dict):
            if "model_state" in payload and isinstance(payload["model_state"], dict):
                return payload["model_state"], payload
            for key in ("model", "policy", "state_dict", "agent"):
                if key in payload and isinstance(payload[key], dict):
                    return payload[key], payload
            if payload and all(torch.is_tensor(v) for v in payload.values()):
                return payload, {}

        raise ValueError(
            "Unsupported checkpoint format. Expected raw state_dict or dict with one of keys: "
            "model_state/model/policy/state_dict/agent"
        )

    def _build_model(
        self,
        *,
        state_dict: dict[str, torch.Tensor],
        meta: dict[str, Any],
        model_kwargs: dict[str, Any],
    ) -> tuple[torch.nn.Module, str]:
        model_cfg_raw = meta.get("model_cfg", {}) if isinstance(meta, dict) else {}

        # Keep FBCPR config kwargs separate from generic fallback kwargs.
        fbcpr_allowed_keys = {
            "state_dim",
            "action_dim",
            "state_feat_dim",
            "per_view_feat_dim",
            "vision_feat_dim",
            "fused_feat_dim",
            "state_encoder_cfg",
            "vision_encoder_cfg",
            "vision_fusion_cfg",
            "fusion_cfg",
            "policy_head_cfg",
            "squash_output",
            # compatibility aliases
            "obs_dim",
            "act_dim",
        }
        fbcpr_kwargs = dict(model_cfg_raw) if isinstance(model_cfg_raw, dict) else {}
        for k, v in model_kwargs.items():
            if v is not None and k in fbcpr_allowed_keys:
                fbcpr_kwargs[k] = v

        if "obs_dim" in fbcpr_kwargs and "state_dim" not in fbcpr_kwargs:
            fbcpr_kwargs["state_dim"] = int(fbcpr_kwargs.pop("obs_dim"))
        if "act_dim" in fbcpr_kwargs and "action_dim" not in fbcpr_kwargs:
            fbcpr_kwargs["action_dim"] = int(fbcpr_kwargs.pop("act_dim"))

        fbcpr_kwargs = self._rehydrate_nested_model_cfg(fbcpr_kwargs)

        try:
            cfg = ModelConfig(**fbcpr_kwargs)
            from bc.fb_cpr.model import FBCPRModel

            model = FBCPRModel(cfg)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if self.debug and (missing or unexpected):
                print(f"[BCPolicyRunner] FBCPR load missing={len(missing)} unexpected={len(unexpected)}")
            return model, "fbcpr_model"
        except Exception as fbcpr_err:
            if self.debug:
                print(f"[BCPolicyRunner] FBCPRModel load failed: {fbcpr_err}")

        # Fallback: minimal MLP Gaussian policy from bc.nn_models.
        obs_dim = model_kwargs.get("obs_dim")
        action_dim = model_kwargs.get("action_dim")
        hidden_dim = int(model_kwargs.get("hidden_dim", 256))
        hidden_layers = int(model_kwargs.get("hidden_layers", 2))
        if obs_dim is None or action_dim is None:
            raise ValueError(
                "Could not build model from checkpoint automatically. "
                "Pass --obs_dim and --action_dim for fallback MLP policy loading."
            )

        model = TanhDiagGaussianPolicy(
            obs_dim=int(obs_dim),
            action_dim=int(action_dim),
            hidden=(hidden_dim,) * hidden_layers,
        )
        model.load_state_dict(state_dict, strict=False)
        return model, "mlp_policy"

    def _rehydrate_nested_model_cfg(self, cfg_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Rebuild nested dataclass config objects if they were serialized as dicts.

        Checkpoints are saved with dataclasses.asdict(...), so nested fields like
        state_encoder_cfg / vision_encoder_cfg arrive as plain dicts.
        FBCPRModel expects MLPConfig/VisionEncoderConfig objects.
        """
        out = dict(cfg_kwargs)

        mlp_cfg_keys = ("state_encoder_cfg", "vision_fusion_cfg", "fusion_cfg", "policy_head_cfg")
        for key in mlp_cfg_keys:
            value = out.get(key)
            if isinstance(value, dict):
                out[key] = MLPConfig(**value)

        vision_value = out.get("vision_encoder_cfg")
        if isinstance(vision_value, dict):
            out["vision_encoder_cfg"] = VisionEncoderConfig(**vision_value)

        return out

    @torch.no_grad()
    def act(self, model_obs: torch.Tensor) -> torch.Tensor:
        model_obs = model_obs.to(self.device)

        if self.model_kind == "fbcpr_model":
            batch_size = model_obs.shape[0]
            # Proprio-only bridge for the current visual policy architecture.
            # TODO: feed real Isaac Lab camera tensors when visual observations are integrated.
            zeros_img = torch.zeros((batch_size, 64, 64, 3), dtype=torch.float32, device=self.device)
            batch = {
                "obs": {
                    "state": {"full": model_obs},
                    "images": {
                        "head": zeros_img,
                        "left_wrist": zeros_img,
                        "right_wrist": zeros_img,
                    },
                }
            }
            action = self.model.act(batch)
        else:
            out = self.model(model_obs)
            action = out[0] if isinstance(out, tuple) else out

        if self.debug:
            print(f"[BCPolicyRunner] model_action_shape={tuple(action.shape)}")
        return action

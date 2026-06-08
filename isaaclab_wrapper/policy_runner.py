from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bc.fb_cpr.model import ModelConfig
from bc.model_blocks import MLPConfig, VisionEncoderConfig
from bc.nn_models import TanhDiagGaussianPolicy
from bc.temporal import temporal_contract_from_checkpoint_meta, validate_runtime_temporal_contract


class BCPolicyRunner:
    def __init__(self, checkpoint_path: str, device: str = "cuda:0", debug: bool = False, **model_kwargs):
        self.device = torch.device(device)
        self.debug = debug
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")

        payload = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict, meta = self._extract_state_dict(payload)
        self.checkpoint_meta: dict[str, Any] = meta if isinstance(meta, dict) else {}
        self.temporal_contract, self.legacy_temporal_contract = temporal_contract_from_checkpoint_meta(
            self.checkpoint_meta, warn_legacy=True
        )
        validate_runtime_temporal_contract(self.temporal_contract)

        self.model, self.model_kind = self._build_model(
            state_dict=state_dict,
            meta=self.checkpoint_meta,
            model_kwargs=model_kwargs,
        )
        self.expected_obs_dim: int | None = None
        self.expected_action_dim: int | None = None
        if self.model_kind == "fbcpr_model":
            self.expected_obs_dim = int(getattr(self.model.cfg, "state_dim", 0))
            self.expected_action_dim = int(getattr(self.model.cfg, "action_dim", 0))
        elif self.model_kind == "mlp_policy":
            self.expected_obs_dim = int(getattr(self.model, "obs_dim", model_kwargs.get("obs_dim", 0)))
            self.expected_action_dim = int(getattr(self.model, "action_dim", model_kwargs.get("action_dim", 0)))

        self.model.to(self.device)
        self.model.eval()

        if self.debug:
            print(f"[BCPolicyRunner] loaded checkpoint={self.checkpoint_path}")
            print(f"[BCPolicyRunner] model_kind={self.model_kind}")
            print(f"[BCPolicyRunner] temporal_contract={self.temporal_contract}")
            print(
                f"[BCPolicyRunner] expected dims: obs={self.expected_obs_dim} "
                f"action={self.expected_action_dim}"
            )

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
    def act(self, model_obs: torch.Tensor, images: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        model_obs = model_obs.to(self.device)

        if self.model_kind == "fbcpr_model":
            if images is None:
                raise RuntimeError(
                    "FBCPR visual policy requires real IsaacSim camera images. "
                    "No obs.images were provided; refusing to use placeholder/zero images."
                )
            required_image_keys = ("head", "left_wrist", "right_wrist")
            missing = [key for key in required_image_keys if key not in images]
            if missing:
                raise RuntimeError(f"FBCPR visual policy missing required obs.images keys: {missing}")
            policy_images: dict[str, torch.Tensor] = {}
            for key in required_image_keys:
                image = images[key].to(self.device)
                if image.ndim != 4:
                    raise ValueError(f"obs.images.{key} must be [B,H,W,C], got {tuple(image.shape)}")
                if image.shape[-1] not in (1, 3, 4):
                    raise ValueError(f"obs.images.{key} must be channel-last [B,H,W,C], got {tuple(image.shape)}")
                if int(image.shape[0]) != int(model_obs.shape[0]):
                    raise ValueError(
                        f"obs.images.{key} batch size {int(image.shape[0])} does not match "
                        f"obs.state.full batch size {int(model_obs.shape[0])}"
                    )
                policy_images[key] = image
            batch = {
                "obs": {
                    "state": {"full": model_obs},
                    "images": policy_images,
                }
            }
            action = self.model.act(batch)
        else:
            out = self.model(model_obs)
            action = out[0] if isinstance(out, tuple) else out

        if self.debug:
            print(f"[BCPolicyRunner] model_action_shape={tuple(action.shape)}")
        return action

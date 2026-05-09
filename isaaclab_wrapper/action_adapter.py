from __future__ import annotations

import torch
from bc.data import schema as data_schema


class ActionAdapter:
    def __init__(
        self,
        expected_action_dim: int | None = None,
        env_device: str = "cuda:0",
        action_scale: float = 1.0,
        clip_actions: bool = True,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        debug: bool = False,
        action_mode: str = "arm_only",
        model_action_dim: int | None = None,
        env_action_dim: int | None = None,
        model_action_joint_names: list[str] | None = None,
        env_ctrl_joint_names: list[str] | None = None,
        allow_schema_fallback: bool = True,
    ):
        self.expected_action_dim = expected_action_dim
        self.env_device = torch.device(env_device)
        self.action_scale = float(action_scale)
        self.clip_actions = bool(clip_actions)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.debug = debug
        self.action_mode = action_mode
        self.model_action_dim = int(model_action_dim) if model_action_dim is not None else expected_action_dim
        self.env_action_dim = int(env_action_dim) if env_action_dim is not None else expected_action_dim
        self.model_action_joint_names = list(model_action_joint_names or [])
        self.env_ctrl_joint_names = list(env_ctrl_joint_names or [])
        self.allow_schema_fallback = bool(allow_schema_fallback)
        self._used_name_map = False
        self._arm_index_map = self._build_arm_index_map()
        self._map_source = "name_map" if self._used_name_map else "schema_fallback_[0:14]"

    def _build_arm_index_map(self) -> list[int]:
        if self.action_mode != "arm_only":
            return []
        if (
            self.model_action_joint_names
            and self.env_ctrl_joint_names
            and len(self.model_action_joint_names) >= data_schema.ACTION_ARM_DIM
            and len(self.env_ctrl_joint_names) == data_schema.ACTION_ARM_DIM
        ):
            index_by_name = {name: idx for idx, name in enumerate(self.model_action_joint_names)}
            if all(name in index_by_name for name in self.env_ctrl_joint_names):
                self._used_name_map = True
                return [index_by_name[name] for name in self.env_ctrl_joint_names]
        if self.allow_schema_fallback:
            return list(range(data_schema.ACTION_ARM_DIM))
        raise ValueError(
            "Failed to build joint-name arm mapping and schema fallback is disabled. "
            "Provide verified model_action_joint_names or enable fallback explicitly."
        )

    def __call__(self, model_action: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(model_action):
            raise TypeError(f"model_action must be torch.Tensor, got {type(model_action).__name__}")

        action_2d = model_action.to(device=self.env_device, dtype=torch.float32)

        if action_2d.ndim == 1:
            action_2d = action_2d.unsqueeze(0)
        elif action_2d.ndim > 2:
            action_2d = action_2d.reshape(action_2d.shape[0], -1)

        if action_2d.ndim != 2:
            raise ValueError(f"model_action must be 2D [num_envs, action_dim], got {tuple(action_2d.shape)}")

        if self.model_action_dim is not None and int(action_2d.shape[-1]) != int(self.model_action_dim):
            raise ValueError(
                f"Model action dim mismatch: expected={self.model_action_dim}, got={int(action_2d.shape[-1])}"
            )

        if self.action_mode == "arm_only":
            idx = torch.as_tensor(self._arm_index_map, dtype=torch.long, device=action_2d.device)
            env_action = action_2d.index_select(dim=-1, index=idx)
        else:
            env_action = action_2d

        if self.env_action_dim is not None and int(env_action.shape[-1]) != int(self.env_action_dim):
            raise ValueError(
                f"Env action dim mismatch: expected={self.env_action_dim}, got={int(env_action.shape[-1])}"
            )

        env_action = env_action * self.action_scale

        if self.clip_actions:
            env_action = torch.clamp(env_action, min=self.clip_min, max=self.clip_max)

        if self.debug:
            print(
                f"[ActionAdapter] mode={self.action_mode} map_source={self._map_source} "
                f"model_action_shape={tuple(model_action.shape)} env_action_shape={tuple(env_action.shape)}"
            )
            if self.action_mode == "arm_only" and self._map_source != "name_map":
                print("[ActionAdapter][WARNING] arm mapping uses fallback [0:14]. Verify joint-order metadata.")

        return env_action

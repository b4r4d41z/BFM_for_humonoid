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
        gripper_bridge_aggregator: str = "mean",
        gripper_open_threshold: float = 0.15,
        gripper_close_threshold: float = -0.15,
        gripper_open_prototype: list[float] | None = None,
        gripper_closed_prototype: list[float] | None = None,
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
        self.gripper_bridge_aggregator = gripper_bridge_aggregator
        self.gripper_open_threshold = float(gripper_open_threshold)
        self.gripper_close_threshold = float(gripper_close_threshold)
        open_proto = gripper_open_prototype or [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]
        closed_proto = gripper_closed_prototype or [69.0, 99.0, 42.0, 44.0, 61.0, 60.0]
        self._open_proto = torch.tensor(open_proto, dtype=torch.float32, device=self.env_device).view(1, -1)
        self._closed_proto = torch.tensor(closed_proto, dtype=torch.float32, device=self.env_device).view(1, -1)
        proto_range = (self._closed_proto - self._open_proto).abs()
        self._active_dims = (proto_range > 1e-6).to(torch.float32)
        self._active_count = torch.clamp(self._active_dims.sum(dim=-1), min=1.0)
        self._gripper_state = torch.zeros((1, 2), dtype=torch.float32, device=self.env_device)
        self._gripper_switches = torch.zeros((1, 2), dtype=torch.int64, device=self.env_device)
        self._gripper_stats = {"open_ratio_left": 0.0, "open_ratio_right": 0.0, "switches_left": 0, "switches_right": 0}
        self._used_name_map = False
        self._arm_index_map = self._build_arm_index_map()
        self._map_source = "name_map" if self._used_name_map else "schema_fallback_[0:14]"

    def _build_arm_index_map(self) -> list[int]:
        if self.action_mode not in ("arm_only", "arm_plus_gripper_bridge"):
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
        elif self.action_mode == "arm_plus_gripper_bridge":
            idx = torch.as_tensor(self._arm_index_map, dtype=torch.long, device=action_2d.device)
            env_action = action_2d.index_select(dim=-1, index=idx)
            self._update_gripper_bridge(action_2d)
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

    def _update_gripper_bridge(self, action_2d: torch.Tensor) -> None:
        if action_2d.shape[-1] < data_schema.ACTION_FULL_DIM:
            return
        hand = action_2d[:, data_schema.ACTION_ARM_DIM : data_schema.ACTION_FULL_DIM]
        left = hand[:, : data_schema.ACTION_HAND_DIM // 2]
        right = hand[:, data_schema.ACTION_HAND_DIM // 2 :]
        left_score = self._closure_score(left)
        right_score = self._closure_score(right)
        scores = torch.stack([left_score, right_score], dim=-1)

        if self._gripper_state.shape[0] != scores.shape[0]:
            self._gripper_state = torch.zeros((scores.shape[0], 2), dtype=torch.float32, device=scores.device)
            self._gripper_switches = torch.zeros((scores.shape[0], 2), dtype=torch.int64, device=scores.device)

        prev = self._gripper_state.clone()
        open_mask = scores <= self.gripper_open_threshold
        close_mask = scores >= self.gripper_close_threshold
        self._gripper_state[open_mask] = 1.0
        self._gripper_state[close_mask] = 0.0
        switched = (prev != self._gripper_state).to(torch.int64)
        self._gripper_switches += switched
        self._gripper_stats = {
            "open_ratio_left": float(self._gripper_state[:, 0].mean().item()),
            "open_ratio_right": float(self._gripper_state[:, 1].mean().item()),
            "switches_left": int(self._gripper_switches[:, 0].sum().item()),
            "switches_right": int(self._gripper_switches[:, 1].sum().item()),
            "closure_score_left": float(left_score.mean().item()),
            "closure_score_right": float(right_score.mean().item()),
            "left_cmd": int(self._gripper_state[:, 0].mean().item() >= 0.5),
            "right_cmd": int(self._gripper_state[:, 1].mean().item() >= 0.5),
            "left_hand_raw": left[0].detach().cpu().tolist(),
            "right_hand_raw": right[0].detach().cpu().tolist(),
        }

    @property
    def gripper_bridge_stats(self) -> dict[str, float | int]:
        return dict(self._gripper_stats)

    def _closure_score(self, hand6: torch.Tensor) -> torch.Tensor:
        # Distance-based score calibrated by dataset prototypes:
        # 0.0 -> closer to open prototype, 1.0 -> closer to closed prototype.
        active = self._active_dims.to(hand6.device)
        open_d = (((hand6 - self._open_proto.to(hand6.device)) ** 2) * active).sum(dim=-1) / self._active_count.to(hand6.device)
        closed_d = (((hand6 - self._closed_proto.to(hand6.device)) ** 2) * active).sum(dim=-1) / self._active_count.to(hand6.device)
        return open_d / (open_d + closed_d + 1e-6)

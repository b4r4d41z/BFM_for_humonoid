# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import UsdFileCfg

from bc.data.schema import (
    ACTION_FULL_DIM,
    ACTION_HAND_DIM,
    ACTION_TYPE,
    STATE_ARM_DIM,
    HAND_CLOSED_PROTOTYPE_6,
    HAND_OPEN_PROTOTYPE_6,
    STATE_FULL_DIM,
    STATE_HAND_DIM,
)

from .test_training_env_cfg import TestTrainingEnvCfg


class TestTrainingEnv(DirectRLEnv):
    cfg: TestTrainingEnvCfg

    def __init__(self, cfg: TestTrainingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._ctrl_joint_ids, self._ctrl_joint_names = self.robot.find_joints(self.cfg.ctrl_joint_names)
        self._left_finger_joint_ids, self._left_finger_joint_names = self.robot.find_joints(
            ["zarm_l_finger_left_joint", "zarm_l_finger_right_joint"]
        )
        self._right_finger_joint_ids, self._right_finger_joint_names = self.robot.find_joints(
            ["zarm_r_finger_left_joint", "zarm_r_finger_right_joint"]
        )
        if len(self._left_finger_joint_ids) != 2 or len(self._right_finger_joint_ids) != 2:
            raise RuntimeError(
                "Could not find required simplified claw/finger joints. "
                f"left={self._left_finger_joint_names} ids={self._left_finger_joint_ids}, "
                f"right={self._right_finger_joint_names} ids={self._right_finger_joint_ids}"
            )

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self._joint_pos_target = torch.zeros(
            (self.num_envs, len(self._ctrl_joint_ids)),
            device=self.device,
            dtype=torch.float32,
        )
        self._left_finger_pos_target = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._right_finger_pos_target = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)

        self._hand_open_prototype_6 = torch.tensor(
            HAND_OPEN_PROTOTYPE_6, device=self.device, dtype=torch.float32
        )
        self._hand_closed_prototype_6 = torch.tensor(
            HAND_CLOSED_PROTOTYPE_6, device=self.device, dtype=torch.float32
        )
        self._left_claw_closed = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._right_claw_closed = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._arm_action_target = torch.zeros((self.num_envs, STATE_ARM_DIM), device=self.device, dtype=torch.float32)
        self._left_hand_action_6 = self._hand_open_prototype_6.expand(self.num_envs, -1).clone()
        self._right_hand_action_6 = self._hand_open_prototype_6.expand(self.num_envs, -1).clone()
        self._last_joint_limit_clipped = False
        self._last_gripper_log_state: tuple[tuple[int, int], bool] | None = None

        self._hand_open_prototype_6 = torch.tensor(
            HAND_OPEN_PROTOTYPE_6, device=self.device, dtype=torch.float32
        )
        self._hand_closed_prototype_6 = torch.tensor(
            HAND_CLOSED_PROTOTYPE_6, device=self.device, dtype=torch.float32
        )
        self._left_claw_closed = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._right_claw_closed = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        self._printed_robot_info = False
        self._printed_action_contract_info = False

    def _setup_scene(self):
        # Spawn per-env static scene under env_0 so it gets cloned
        scene_cfg = UsdFileCfg(usd_path=str(self.cfg.scene_usd_path))
        scene_cfg.func("/World/envs/env_0/Scene", scene_cfg)

        # Spawn robot (must be inside env_0 before cloning)
        self.robot = Articulation(self.cfg.robot_cfg)

        self.red_ball = RigidObject(self.cfg.red_ball_cfg)
        self.yellow_ball = RigidObject(self.cfg.yellow_ball_cfg)
        self.container_base = RigidObject(self.cfg.container_base_cfg)
        self.container_left_wall = RigidObject(self.cfg.container_left_wall_cfg)
        self.container_right_wall = RigidObject(self.cfg.container_right_wall_cfg)
        self.container_front_wall = RigidObject(self.cfg.container_front_wall_cfg)
        self.container_back_wall = RigidObject(self.cfg.container_back_wall_cfg)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["red_ball"] = self.red_ball
        self.scene.rigid_objects["yellow_ball"] = self.yellow_ball
        self.scene.rigid_objects["container_base"] = self.container_base
        self.scene.rigid_objects["container_left_wall"] = self.container_left_wall
        self.scene.rigid_objects["container_right_wall"] = self.container_right_wall
        self.scene.rigid_objects["container_front_wall"] = self.container_front_wall
        self.scene.rigid_objects["container_back_wall"] = self.container_back_wall

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if not self._printed_robot_info:
            self._printed_robot_info = True
            print(f"[INFO]: Robot joints ({len(self.robot.joint_names)}): {', '.join(self.robot.joint_names)}")
            print(
                f"[INFO]: Controlled arm joints ({len(self._ctrl_joint_names)}): "
                f"{', '.join(self._ctrl_joint_names)} | ids: {self._ctrl_joint_ids}"
            )
            print(
                f"[INFO]: Left claw joints ({len(self._left_finger_joint_names)}): "
                f"{', '.join(self._left_finger_joint_names)} | ids: {self._left_finger_joint_ids}"
            )
            print(
                f"[INFO]: Right claw joints ({len(self._right_finger_joint_names)}): "
                f"{', '.join(self._right_finger_joint_names)} | ids: {self._right_finger_joint_ids}"
            )

        actions = actions.to(self.device, dtype=torch.float32)
        if actions.ndim != 2:
            raise ValueError(f"Expected policy actions [num_envs, {ACTION_FULL_DIM}], got shape {tuple(actions.shape)}")
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected policy actions for {self.num_envs} envs, got {actions.shape[0]}")
        if actions.shape[-1] != ACTION_FULL_DIM:
            raise ValueError(f"Expected policy action dim {ACTION_FULL_DIM}, got {actions.shape[-1]}")

        self.actions = actions
        self._arm_action_target = actions[:, :STATE_ARM_DIM]
        self._left_hand_action_6 = actions[:, STATE_ARM_DIM : STATE_ARM_DIM + ACTION_HAND_DIM // 2]
        self._right_hand_action_6 = actions[:, STATE_ARM_DIM + ACTION_HAND_DIM // 2 : ACTION_FULL_DIM]

        if self._arm_action_target.shape[-1] != STATE_ARM_DIM:
            raise ValueError(f"Expected arm action slice dim {STATE_ARM_DIM}, got {self._arm_action_target.shape[-1]}")
        if self._left_hand_action_6.shape[-1] != ACTION_HAND_DIM // 2:
            raise ValueError(f"Expected left hand action slice dim 6, got {self._left_hand_action_6.shape[-1]}")
        if self._right_hand_action_6.shape[-1] != ACTION_HAND_DIM // 2:
            raise ValueError(f"Expected right hand action slice dim 6, got {self._right_hand_action_6.shape[-1]}")

    def _apply_action(self) -> None:
        self._joint_pos_target, self._last_joint_limit_clipped = self._clip_joint_targets_to_limits(
            self._arm_action_target, self._ctrl_joint_ids
        )
        self.robot.set_joint_position_target(self._joint_pos_target, joint_ids=self._ctrl_joint_ids)

        self._left_claw_closed = self._hand_targets_to_closed(self._left_hand_action_6)
        self._right_claw_closed = self._hand_targets_to_closed(self._right_hand_action_6)
        self._apply_claw_position_targets()
        self._log_action_debug()

    def _clip_joint_targets_to_limits(
        self, targets: torch.Tensor, joint_ids: Sequence[int]
    ) -> tuple[torch.Tensor, bool]:
        limits = getattr(self.robot.data, "soft_joint_pos_limits", None)
        if limits is None:
            limits = getattr(self.robot.data, "joint_pos_limits", None)
        if limits is None:
            return targets, False

        joint_ids_tensor = torch.as_tensor(joint_ids, device=self.device, dtype=torch.long)
        if limits.ndim == 2:
            selected_limits = limits[joint_ids_tensor, :].unsqueeze(0).expand(targets.shape[0], -1, -1)
        else:
            selected_limits = limits[:, joint_ids_tensor, :]
        lower = selected_limits[..., 0]
        upper = selected_limits[..., 1]
        clipped = torch.minimum(torch.maximum(targets, lower), upper)
        was_clipped = bool(torch.any(torch.ne(clipped, targets)).item())
        return clipped, was_clipped

    def _hand_targets_to_closed(self, hand6: torch.Tensor) -> torch.Tensor:
        if hand6.shape[-1] != ACTION_HAND_DIM // 2:
            raise ValueError(f"Expected 6D hand target, got shape {tuple(hand6.shape)}")
        open_proto = self._hand_open_prototype_6.expand_as(hand6)
        closed_proto = self._hand_closed_prototype_6.expand_as(hand6)
        open_dist = torch.sum((hand6 - open_proto) ** 2, dim=-1)
        closed_dist = torch.sum((hand6 - closed_proto) ** 2, dim=-1)
        return closed_dist <= open_dist

    def _finger_open_closed_targets(
        self, joint_ids: Sequence[int], joint_names: Sequence[str], env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = self.num_envs if env_ids is None else int(env_ids.numel())
        joint_ids_tensor = torch.as_tensor(joint_ids, device=self.device, dtype=torch.long)
        limits = getattr(self.robot.data, "soft_joint_pos_limits", None)
        if limits is None:
            limits = getattr(self.robot.data, "joint_pos_limits", None)

        if limits is not None:
            if limits.ndim == 2:
                selected_limits = limits[joint_ids_tensor, :].unsqueeze(0).expand(num_envs, -1, -1)
            else:
                selected_limits = limits[:, joint_ids_tensor, :] if env_ids is None else limits[env_ids][:, joint_ids_tensor, :]
            lower = selected_limits[..., 0]
            upper = selected_limits[..., 1]
            open_targets = torch.minimum(torch.maximum(torch.zeros_like(lower), lower), upper)
            lower_delta = torch.abs(lower - open_targets)
            upper_delta = torch.abs(upper - open_targets)
            closed_targets = torch.where(upper_delta >= lower_delta, upper, lower)
            return open_targets, closed_targets

        open_targets = torch.zeros((num_envs, len(joint_ids)), device=self.device, dtype=torch.float32)
        closed_values = []
        for name in joint_names:
            closed_values.append(-0.78539816339 if "right" in name else 0.78539816339)
        closed_targets = torch.tensor(closed_values, device=self.device, dtype=torch.float32).expand(num_envs, -1)
        return open_targets, closed_targets

    def _apply_claw_position_targets(self, env_ids: torch.Tensor | None = None) -> None:
        left_open, left_closed = self._finger_open_closed_targets(
            self._left_finger_joint_ids, self._left_finger_joint_names, env_ids
        )
        right_open, right_closed = self._finger_open_closed_targets(
            self._right_finger_joint_ids, self._right_finger_joint_names, env_ids
        )

        if env_ids is None:
            left_closed_mask = self._left_claw_closed.unsqueeze(-1)
            right_closed_mask = self._right_claw_closed.unsqueeze(-1)
        else:
            left_closed_mask = self._left_claw_closed[env_ids].unsqueeze(-1)
            right_closed_mask = self._right_claw_closed[env_ids].unsqueeze(-1)

        self._left_finger_pos_target = torch.where(left_closed_mask, left_closed, left_open)
        self._right_finger_pos_target = torch.where(right_closed_mask, right_closed, right_open)
        self.robot.set_joint_position_target(
            self._left_finger_pos_target, joint_ids=self._left_finger_joint_ids, env_ids=env_ids
        )
        self.robot.set_joint_position_target(
            self._right_finger_pos_target, joint_ids=self._right_finger_joint_ids, env_ids=env_ids
        )

    def _log_action_debug(self) -> None:
        state = (
            (int(self._left_claw_closed.sum().item()), int(self._right_claw_closed.sum().item())),
            self._last_joint_limit_clipped,
        )
        if self._printed_action_contract_info and state == self._last_gripper_log_state:
            return

        if not self._printed_action_contract_info:
            self._printed_action_contract_info = True
            print(
                f"[INFO]: policy_action_dim={ACTION_FULL_DIM} action_type={ACTION_TYPE} "
                "arm_target_mode=direct_absolute_target gripper_bridge=enabled"
            )

        self._last_gripper_log_state = state
        print(
            "[INFO]: gripper_state "
            f"left_closed_envs={state[0][0]}/{self.num_envs} "
            f"right_closed_envs={state[0][1]}/{self.num_envs} "
            f"joint_limit_clipping={'applied' if self._last_joint_limit_clipped else 'not_applied'}"
        )

    def _get_simulated_hand_state_12(self) -> torch.Tensor:
        left_hand_6 = torch.where(
            self._left_claw_closed.unsqueeze(-1),
            self._hand_closed_prototype_6.expand(self.num_envs, -1),
            self._hand_open_prototype_6.expand(self.num_envs, -1),
        )
        right_hand_6 = torch.where(
            self._right_claw_closed.unsqueeze(-1),
            self._hand_closed_prototype_6.expand(self.num_envs, -1),
            self._hand_open_prototype_6.expand(self.num_envs, -1),
        )
        hand_state_12 = torch.cat((left_hand_6, right_hand_6), dim=-1)
        if hand_state_12.shape[-1] != STATE_HAND_DIM:
            raise ValueError(
                f"Expected simulated hand state dim {STATE_HAND_DIM}, got {hand_state_12.shape[-1]}"
            )
        return hand_state_12

    def _get_observations(self) -> dict:
        q_arm = self.joint_pos[:, self._ctrl_joint_ids]
        hand_state_12 = self._get_simulated_hand_state_12()

        if q_arm.shape[-1] != STATE_ARM_DIM:
            raise ValueError(f"Expected arm joint position dim {STATE_ARM_DIM}, got {q_arm.shape[-1]}")
        if hand_state_12.shape[-1] != STATE_HAND_DIM:
            raise ValueError(f"Expected hand state dim {STATE_HAND_DIM}, got {hand_state_12.shape[-1]}")

        obs = torch.cat((q_arm, hand_state_12), dim=-1)
        if obs.shape[-1] != STATE_FULL_DIM:
            raise ValueError(f"Expected policy observation dim {STATE_FULL_DIM}, got {obs.shape[-1]}")
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_objects(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        num_envs = len(env_ids)

        def _sample_xy(x_range: tuple[float, float], y_range: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
            x = x_range[0] + (x_range[1] - x_range[0]) * torch.rand((num_envs,), device=self.device)
            y = y_range[0] + (y_range[1] - y_range[0]) * torch.rand((num_envs,), device=self.device)
            return x, y

        z = float(self.cfg.table_top_z) + float(self.cfg.ball_radius) + float(self.cfg.ball_spawn_margin)

        red_x, red_y = _sample_xy(self.cfg.red_ball_x_range, self.cfg.red_ball_y_range)
        yellow_x, yellow_y = _sample_xy(self.cfg.yellow_ball_x_range, self.cfg.yellow_ball_y_range)

        balls_setup = (
            ("red_ball", red_x, red_y),
            ("yellow_ball", yellow_x, yellow_y),
        )
        for name, x_vals, y_vals in balls_setup:
            obj = self.scene.rigid_objects[name]
            root_state = obj.data.default_root_state[env_ids].clone()
            root_state[:, :3] = 0.0
            root_state[:, 0] = x_vals
            root_state[:, 1] = y_vals
            root_state[:, 2] = z
            root_state[:, 3:7] = 0.0
            root_state[:, 3] = 1.0
            root_state[:, :3] += self.scene.env_origins[env_ids]
            root_state[:, 7:] = 0.0
            obj.write_root_pose_to_sim(root_state[:, :7], env_ids)
            obj.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._left_claw_closed[env_ids] = False
        self._right_claw_closed[env_ids] = False
        self._apply_claw_position_targets(env_ids=env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        if float(getattr(self.cfg, "reset_noise_joint_pos", 0.0)) > 0.0:
            noise = (2.0 * torch.rand((len(env_ids), len(self._ctrl_joint_ids)), device=self.device) - 1.0) * float(
                self.cfg.reset_noise_joint_pos
            )
            joint_pos[:, self._ctrl_joint_ids] += noise

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._reset_objects(env_ids)

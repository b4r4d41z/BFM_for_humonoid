# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import UsdFileCfg

from .test_training_env_cfg import TestTrainingEnvCfg


class TestTrainingEnv(DirectRLEnv):
    cfg: TestTrainingEnvCfg

    def __init__(self, cfg: TestTrainingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._ctrl_joint_ids, self._ctrl_joint_names = self.robot.find_joints(self.cfg.ctrl_joint_names)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self._joint_pos_target = torch.zeros(
            (self.num_envs, len(self._ctrl_joint_ids)),
            device=self.device,
            dtype=torch.float32,
        )

        self._printed_robot_info = False

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
                f"[INFO]: Controlled joints ({len(self._ctrl_joint_names)}): "
                f"{', '.join(self._ctrl_joint_names)} | ids: {self._ctrl_joint_ids}"
            )

        self.actions = actions.to(self.device).clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        q = self.joint_pos[:, self._ctrl_joint_ids]
        dq = self.actions * float(self.cfg.action_scale)
        self._joint_pos_target = q + dq
        self.robot.set_joint_position_target(self._joint_pos_target, joint_ids=self._ctrl_joint_ids)

    def _get_observations(self) -> dict:
        q = self.joint_pos[:, self._ctrl_joint_ids]
        qd = self.joint_vel[:, self._ctrl_joint_ids]
        obs = torch.cat((q, qd), dim=-1)
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

# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .test_training_env_cfg import TestTrainingEnvCfg


class TestTrainingEnv(DirectRLEnv):
    cfg: TestTrainingEnvCfg

    def __init__(self, cfg: TestTrainingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Controlled joints (arms only)
        self._ctrl_joint_ids, self._ctrl_joint_names = self.robot.find_joints(self.cfg.ctrl_joint_names)

        # Cache joint state tensors
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Position targets buffer for controlled joints
        self._joint_pos_target = torch.zeros(
            (self.cfg.scene.num_envs, len(self._ctrl_joint_ids)),
            device=self.device,
            dtype=torch.float32,
        )

        # One-time print
        self._printed_robot_info = False

    def _setup_scene(self):
        # Spawn robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Ground
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone envs
        self.scene.clone_environments(copy_from_source=False)

        # Collision filtering (CPU only)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Register robot
        self.scene.articulations["robot"] = self.robot

        # Light (optional)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if not self._printed_robot_info:
            self._printed_robot_info = True

            joints_line = ", ".join(self.robot.joint_names)
            print(f"[INFO]: Robot joints ({len(self.robot.joint_names)}): {joints_line}")

            ctrl_line = ", ".join(self._ctrl_joint_names)
            print(f"[INFO]: Controlled joints ({len(self._ctrl_joint_names)}): {ctrl_line} | ids: {self._ctrl_joint_ids}")

        self.actions = actions.to(self.device).clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        q = self.joint_pos[:, self._ctrl_joint_ids]  # [num_envs, 14]
        dq = self.actions * float(self.cfg.action_scale)
        self._joint_pos_target = q + dq
        self.robot.set_joint_position_target(self._joint_pos_target, joint_ids=self._ctrl_joint_ids)

    def _get_observations(self) -> dict:
        q = self.joint_pos[:, self._ctrl_joint_ids]
        qd = self.joint_vel[:, self._ctrl_joint_ids]
        obs = torch.cat((q, qd), dim=-1)  # [num_envs, 28]
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros((self.cfg.scene.num_envs,), device=self.device, dtype=torch.float32)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros((self.cfg.scene.num_envs,), device=self.device, dtype=torch.bool)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

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

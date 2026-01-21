# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils import configclass


@configclass
class TestTrainingEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 10.0

    action_space = 14
    observation_space = 28
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    ctrl_joint_names = [
        "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint",
        "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
        "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint",
        "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
    ]

    action_scale = 0.25

    reset_noise_joint_pos = 0.05
    reset_noise_joint_vel = 0.0

    _ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "robots" / "urdf" /"biped_s40"
    _USD_PATH = _ASSET_DIR / "biped_s40.usd"

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(usd_path=str(_USD_PATH)),
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    r"zarm_l[1-7]_joint",
                    r"zarm_r[1-7]_joint",
                ],
                stiffness=80.0,
                damping=4.0,
            ),
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    r"leg_l[1-6]_joint",
                    r"leg_r[1-6]_joint",
                ],
                stiffness=200.0,
                damping=10.0,
            ),
            "head": ImplicitActuatorCfg(
                joint_names_expr=[r"zhead_[1-2]_joint"],
                stiffness=50.0,
                damping=2.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[
                    r"zarm_l_finger_left_joint",
                    r"zarm_l_finger_right_joint",
                    r"zarm_r_finger_left_joint",
                    r"zarm_r_finger_right_joint",
                ],
                stiffness=30.0,
                damping=1.0,
            ),
        },
    )
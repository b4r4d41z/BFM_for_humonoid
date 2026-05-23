# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
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
        #num_envs=1,
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

    # Package root: .../source/Test_training/Test_training
    _PKG_DIR = Path(__file__).resolve().parents[3]
    _ASSETS_DIR = _PKG_DIR / "assets"

    # Static scene USD (ground + table, without robot)
    scene_usd_path: str = str(_ASSETS_DIR / "base_scene.usd")

    # Robot USD
    _ROBOT_DIR = _ASSETS_DIR / "robots" / "urdf" / "supported_biped_s40_v2"
    _ROBOT_USD_PATH = _ROBOT_DIR / "supported_biped_s40_sensor.usd"

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(usd_path=str(_ROBOT_USD_PATH)),
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


    table_top_z = 1
    ball_radius = 0.035
    ball_spawn_margin = 0.01
    red_ball_x_range = (0.40, 0.58)
    red_ball_y_range = (-0.22, -0.06)
    yellow_ball_x_range = (0.40, 0.58)
    yellow_ball_y_range = (0.06, 0.22)

    red_ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/RedBall",
        spawn=sim_utils.SphereCfg(
            radius=ball_radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.06),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.12, table_top_z + ball_radius + ball_spawn_margin)),
    )

    yellow_ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/YellowBall",
        spawn=sim_utils.SphereCfg(
            radius=ball_radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.06),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.12, table_top_z + ball_radius + ball_spawn_margin)),
    )

    container_base_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BlueContainerBase",
        spawn=sim_utils.CuboidCfg(
            size=(0.22, 0.22, 0.025),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65, 0.0, table_top_z + 0.0125)),
    )

    container_left_wall_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BlueContainerLeftWall",
        spawn=sim_utils.CuboidCfg(
            size=(0.025, 0.22, 0.09),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7475, 0.0, table_top_z + 0.045)),
    )

    container_right_wall_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BlueContainerRightWall",
        spawn=sim_utils.CuboidCfg(
            size=(0.025, 0.22, 0.09),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5525, 0.0, table_top_z + 0.045)),
    )

    container_front_wall_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BlueContainerFrontWall",
        spawn=sim_utils.CuboidCfg(
            size=(0.22, 0.025, 0.09),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65, 0.0975, table_top_z + 0.045)),
    )

    container_back_wall_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BlueContainerBackWall",
        spawn=sim_utils.CuboidCfg(
            size=(0.22, 0.025, 0.09),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65, -0.0975, table_top_z + 0.045)),
    )


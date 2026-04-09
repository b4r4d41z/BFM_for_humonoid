from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STATE_ARM_DIM = 14
STATE_HAND_DIM = 12
STATE_FULL_DIM = STATE_ARM_DIM + STATE_HAND_DIM

ACTION_ARM_DIM = 14
ACTION_HAND_DIM = 12
ACTION_FULL_DIM = ACTION_ARM_DIM + ACTION_HAND_DIM

IMAGE_KEYS = ("head", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class HDF5Paths:
    obs_joint_pos: str = "/obs/joint_pos"
    obs_hand_pos: str = "/obs/hand_pos"
    obs_state: str = "/obs/state"

    next_obs_state: str = "/next_obs/state"

    act_joint_target: str = "/act/joint_target"
    act_hand_target: str = "/act/hand_target"
    act_action: str = "/act/action"

    images_head: str = "/images/head"
    images_left_wrist: str = "/images/left_wrist"
    images_right_wrist: str = "/images/right_wrist"

    done: str = "/done"
    reward: str = "/reward"
    timestamps: str = "/timestamps"

    meta_instruction: str = "/meta/instruction"
    meta_bag_name: str = "/meta/bag_name"
    meta_joint_names: str = "/meta/joint_names"
    meta_obs_dim: str = "/meta/obs_dim"
    meta_act_dim: str = "/meta/act_dim"
    meta_state_definition: str = "/meta/state_definition"


PATHS = HDF5Paths()


def split_state_vector(x: Any) -> dict[str, Any]:
    """
    devide the 26-dim state vector into arm + hand + full.
    expected order
    [0:14]  -> arm joints
    [14:26] -> hand state
    """
    if len(x) != STATE_FULL_DIM:
        raise ValueError(f"Expected state dim {STATE_FULL_DIM}, got {len(x)}")

    return {
        "arm_joints": x[:STATE_ARM_DIM],
        "hand_state": x[STATE_ARM_DIM:STATE_FULL_DIM],
        "full": x,
    }


def split_action_vector(x: Any) -> dict[str, Any]:
    """
    devide the 26-dim action vector into arm + hand + full.
    expected order
    [0:14]  -> arm joint target
    [14:26] -> hand target
    """
    if len(x) != ACTION_FULL_DIM:
        raise ValueError(f"Expected action dim {ACTION_FULL_DIM}, got {len(x)}")

    return {
        "joint_target": x[:ACTION_ARM_DIM],
        "hand_target": x[ACTION_ARM_DIM:ACTION_FULL_DIM],
        "full": x,
    }


def get_image_path(key: str) -> str:
    image_map = {
        "head": PATHS.images_head,
        "left_wrist": PATHS.images_left_wrist,
        "right_wrist": PATHS.images_right_wrist,
    }
    if key not in image_map:
        raise KeyError(f"Unknown image key: {key}")
    return image_map[key]


def canonical_sample_description() -> dict[str, Any]:
    """
    Только описание схемы sample, без реальных данных.
    Удобно для документации и отладки.
    """
    return {
        "obs": {
            "state": {
                "arm_joints": f"[{STATE_ARM_DIM}]",
                "hand_state": f"[{STATE_HAND_DIM}]",
                "full": f"[{STATE_FULL_DIM}]",
            },
            "images": {
                "head": "[H, W, 3]",
                "left_wrist": "[H, W, 3]",
                "right_wrist": "[H, W, 3]",
            },
            "text": "str",
            "timestamp": "scalar",
        },
        "action": {
            "joint_target": f"[{ACTION_ARM_DIM}]",
            "hand_target": f"[{ACTION_HAND_DIM}]",
            "full": f"[{ACTION_FULL_DIM}]",
        },
        "next_obs": {
            "state": {
                "arm_joints": f"[{STATE_ARM_DIM}]",
                "hand_state": f"[{STATE_HAND_DIM}]",
                "full": f"[{STATE_FULL_DIM}]",
            }
        },
        "reward": "scalar",
        "done": "bool",
        "meta": {
            "bag_name": "str",
            "joint_names": f"[{STATE_ARM_DIM}]",
            "obs_dim": "scalar",
            "act_dim": "scalar",
            "state_definition": "str",
            "source": "str",
        },
    }
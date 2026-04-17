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
    """
    Canonical raw HDF5 dataset paths.

    Important:
    - These are raw dataset paths inside the .h5 file.
    - They do not have to match the internal Python sample keys one-to-one.
    - Internal Python naming is frozen separately as:
      state -> {arm, hand, full}
      action -> {arm, hand, full}
    """

    # Main raw state paths
    obs_state: str = "/obs/state"
    next_obs_state: str = "/next_obs/state"

    # Optional compatibility raw paths
    obs_joint_pos: str = "/obs/joint_pos"
    obs_hand_pos: str = "/obs/hand_pos"

    # Raw action paths
    act_joint_target: str = "/act/joint_target"
    act_hand_target: str = "/act/hand_target"
    act_action: str = "/act/action"

    # Vision paths
    images_head: str = "/images/head"
    images_left_wrist: str = "/images/left_wrist"
    images_right_wrist: str = "/images/right_wrist"

    # Transition paths
    done: str = "/done"
    reward: str = "/reward"
    timestamps: str = "/timestamps"

    # Meta paths
    meta_instruction: str = "/meta/instruction"
    meta_bag_name: str = "/meta/bag_name"
    meta_joint_names: str = "/meta/joint_names"
    meta_obs_dim: str = "/meta/obs_dim"
    meta_act_dim: str = "/meta/act_dim"
    meta_state_definition: str = "/meta/state_definition"


PATHS = HDF5Paths()


def _infer_last_dim(x: Any) -> int:
    """
    Return the size of the last dimension for vector-like objects.

    Supports:
    - torch.Tensor
    - numpy.ndarray
    - python lists
    - other objects exposing .shape or __len__
    """
    if hasattr(x, "shape"):
        shape = tuple(x.shape)
        if len(shape) == 0:
            raise ValueError("Input is scalar, expected vector-like object")
        return int(shape[-1])

    if hasattr(x, "__len__"):
        return len(x)

    raise TypeError(f"Cannot infer vector dimension from type: {type(x).__name__}")


def _slice_last_dim(x: Any, start: int, end: int) -> Any:
    """
    Slice along the last dimension.

    Works for:
    - [D]
    - [B, D]
    - [..., D]
    """
    try:
        return x[..., start:end]
    except Exception as e:
        raise TypeError(
            f"Object of type {type(x).__name__} does not support last-dim slicing"
        ) from e


def split_state_vector(x: Any) -> dict[str, Any]:
    """
    Split a 26-dim state vector into the frozen internal representation:

    - arm  -> [0:14]
    - hand -> [14:26]
    - full -> original input

    Supports:
    - [26]
    - [..., 26]
    """
    last_dim = _infer_last_dim(x)
    if last_dim != STATE_FULL_DIM:
        raise ValueError(f"Expected state dim {STATE_FULL_DIM}, got {last_dim}")

    return {
        "arm": _slice_last_dim(x, 0, STATE_ARM_DIM),
        "hand": _slice_last_dim(x, STATE_ARM_DIM, STATE_FULL_DIM),
        "full": x,
    }


def split_action_vector(x: Any) -> dict[str, Any]:
    """
    Split a 26-dim action vector into the frozen internal representation:

    - arm  -> [0:14]
    - hand -> [14:26]
    - full -> original input

    Supports:
    - [26]
    - [..., 26]
    """
    last_dim = _infer_last_dim(x)
    if last_dim != ACTION_FULL_DIM:
        raise ValueError(f"Expected action dim {ACTION_FULL_DIM}, got {last_dim}")

    return {
        "arm": _slice_last_dim(x, 0, ACTION_ARM_DIM),
        "hand": _slice_last_dim(x, ACTION_ARM_DIM, ACTION_FULL_DIM),
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


def all_image_paths() -> dict[str, str]:
    return {key: get_image_path(key) for key in IMAGE_KEYS}


def canonical_sample_description() -> dict[str, Any]:
    """
    Canonical in-memory sample structure used everywhere in Python code.

    This contract is frozen and should be used consistently by:
    - stream_loader.py
    - batch_assembly.py
    - check_data.py
    - future buffer / train code

    Required core fields:
    - obs.state.{arm, hand, full}
    - action.{arm, hand, full}
    - next_obs.state.{arm, hand, full}

    Optional fields:
    - obs.images
    - reward
    - done
    - meta
    """
    return {
        "obs": {
            "state": {
                "arm": f"[{STATE_ARM_DIM}]",
                "hand": f"[{STATE_HAND_DIM}]",
                "full": f"[{STATE_FULL_DIM}]",
            },
            "images": {
                "head": "[H, W, 3] optional",
                "left_wrist": "[H, W, 3] optional",
                "right_wrist": "[H, W, 3] optional",
            },
        },
        "action": {
            "arm": f"[{ACTION_ARM_DIM}]",
            "hand": f"[{ACTION_HAND_DIM}]",
            "full": f"[{ACTION_FULL_DIM}]",
        },
        "next_obs": {
            "state": {
                "arm": f"[{STATE_ARM_DIM}]",
                "hand": f"[{STATE_HAND_DIM}]",
                "full": f"[{STATE_FULL_DIM}]",
            }
        },
        "reward": "scalar optional",
        "done": "bool or scalar optional",
        "meta": {
            "instruction": "str optional",
            "bag_name": "str optional",
            "joint_names": f"[{STATE_ARM_DIM}] optional",
            "obs_dim": "scalar optional",
            "act_dim": "scalar optional",
            "state_definition": "str optional",
        },
    }
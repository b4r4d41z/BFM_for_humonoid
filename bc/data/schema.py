from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STATE_ARM_DIM = 14
STATE_HAND_DIM = 12
STATE_FULL_DIM = 26

ACTION_ARM_DIM = 14
ACTION_HAND_DIM = 12
ACTION_FULL_DIM = 26

ARM_JOINT_NAMES = [
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
    "zarm_r5_joint",
    "zarm_r6_joint",
    "zarm_r7_joint",
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_l5_joint",
    "zarm_l6_joint",
    "zarm_l7_joint",
]

HAND_VALUE_NAMES = [
    "l_hand_cmd_0",
    "l_hand_cmd_1",
    "l_hand_cmd_2",
    "l_hand_cmd_3",
    "l_hand_cmd_4",
    "l_hand_cmd_5",
    "r_hand_cmd_0",
    "r_hand_cmd_1",
    "r_hand_cmd_2",
    "r_hand_cmd_3",
    "r_hand_cmd_4",
    "r_hand_cmd_5",
]

ACTION_TYPE = "absolute_joint_target"
GRIPPER_MODE = "real_robot_hand_values_to_sim_claw_bridge"
STATE_LAYOUT = "arm14_plus_hand12"
ACTION_LAYOUT = "arm14_absolute_plus_hand12_target"
HAND_OPEN_PROTOTYPE_6 = [0, 100, 0, 0, 0, 0]
HAND_CLOSED_PROTOTYPE_6 = [69, 99, 42, 44, 61, 60]
LEFT_HAND_SLICE = [14, 20]
RIGHT_HAND_SLICE = [20, 26]

IMAGE_KEYS = ("head", "left_wrist", "right_wrist")

_CONTRACT_ALIASES = {
    "act_dim": "action_dim",
    "joint_names": "arm_joint_names",
}

_CONTRACT_DEFAULT_TO_LEGACY_KEYS = {
    "action_dim": "act_dim",
    "arm_joint_names": "joint_names",
}


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
    meta_hand_value_names: str = "/meta/hand_value_names"
    meta_obs_dim: str = "/meta/obs_dim"
    meta_act_dim: str = "/meta/act_dim"
    meta_action_dim: str = "/meta/action_dim"
    meta_action_type: str = "/meta/action_type"
    meta_gripper_mode: str = "/meta/gripper_mode"
    meta_state_layout: str = "/meta/state_layout"
    meta_action_layout: str = "/meta/action_layout"
    meta_state_definition: str = "/meta/state_definition"
    meta_hand_open_prototype_6: str = "/meta/hand_open_prototype_6"
    meta_hand_closed_prototype_6: str = "/meta/hand_closed_prototype_6"


PATHS = HDF5Paths()


def get_default_contract_metadata() -> dict[str, Any]:
    """Return the canonical 26D data/model contract metadata."""
    return {
        "obs_dim": STATE_FULL_DIM,
        "action_dim": ACTION_FULL_DIM,
        "arm_dim": STATE_ARM_DIM,
        "hand_dim": STATE_HAND_DIM,
        "arm_joint_names": list(ARM_JOINT_NAMES),
        "hand_value_names": list(HAND_VALUE_NAMES),
        "state_layout": STATE_LAYOUT,
        "action_layout": ACTION_LAYOUT,
        "action_type": ACTION_TYPE,
        "gripper_mode": GRIPPER_MODE,
        "hand_open_prototype_6": list(HAND_OPEN_PROTOTYPE_6),
        "hand_closed_prototype_6": list(HAND_CLOSED_PROTOTYPE_6),
        "left_hand_slice": list(LEFT_HAND_SLICE),
        "right_hand_slice": list(RIGHT_HAND_SLICE),
        "image_keys": list(IMAGE_KEYS),
    }


def _canonicalize_contract_keys(meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(meta)
    for legacy_key, canonical_key in _CONTRACT_ALIASES.items():
        if canonical_key not in out and legacy_key in out:
            out[canonical_key] = out[legacy_key]
    return out


def _as_plain_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError(f"expected list-like value, got {type(value).__name__}")
    return [x.decode("utf-8") if isinstance(x, bytes) else x for x in value]


def _as_string_list(value: Any) -> list[str]:
    return [str(x) for x in _as_plain_list(value)]


def _as_number_list(value: Any) -> list[float | int]:
    items = _as_plain_list(value)
    out: list[float | int] = []
    for item in items:
        number = float(item)
        out.append(int(number) if number.is_integer() else number)
    return out


def _same_number_list(actual: Any, expected: list[int | float]) -> bool:
    try:
        actual_numbers = [float(x) for x in _as_number_list(actual)]
    except Exception:
        return False
    return actual_numbers == [float(x) for x in expected]


def validate_contract_metadata(meta: dict[str, Any], strict: bool = True) -> dict[str, Any]:
    """
    Validate semantic metadata against the fixed 26D project contract.

    Args:
        meta: Metadata loaded from HDF5/checkpoint/config. The legacy keys
            ``act_dim`` and ``joint_names`` are accepted as aliases for
            ``action_dim`` and ``arm_joint_names``.
        strict: When true, missing contract keys are errors. When false, only
            present keys are validated.

    Returns:
        A canonicalized shallow copy of ``meta``.
    """
    canonical_meta = _canonicalize_contract_keys(meta)
    expected = get_default_contract_metadata()
    errors: list[str] = []

    def require_key(key: str) -> bool:
        if key not in canonical_meta:
            if strict:
                errors.append(f"missing required contract metadata key: {key}")
            return False
        return True

    for key in ("obs_dim", "action_dim", "arm_dim", "hand_dim"):
        if require_key(key):
            actual = int(canonical_meta[key])
            if actual != int(expected[key]):
                errors.append(f"{key}={actual}, expected {expected[key]}")

    for key in ("arm_joint_names", "hand_value_names"):
        if require_key(key):
            actual_list = _as_string_list(canonical_meta[key])
            expected_list = expected[key]
            if len(actual_list) != len(expected_list):
                errors.append(f"{key} count={len(actual_list)}, expected {len(expected_list)}")
            elif actual_list != expected_list:
                errors.append(f"{key}={actual_list}, expected {expected_list}")
            canonical_meta[key] = actual_list

    for key in ("action_type", "gripper_mode", "state_layout", "action_layout"):
        if require_key(key):
            actual = str(canonical_meta[key])
            if actual != expected[key]:
                errors.append(f"{key}={actual!r}, expected {expected[key]!r}")
            canonical_meta[key] = actual

    for key in ("hand_open_prototype_6", "hand_closed_prototype_6"):
        if require_key(key):
            if not _same_number_list(canonical_meta[key], expected[key]):
                errors.append(f"{key}={canonical_meta[key]}, expected {expected[key]}")
            canonical_meta[key] = _as_number_list(canonical_meta[key])

    for key in ("left_hand_slice", "right_hand_slice"):
        if require_key(key):
            if not _same_number_list(canonical_meta[key], expected[key]):
                errors.append(f"{key}={canonical_meta[key]}, expected {expected[key]}")
            canonical_meta[key] = [int(x) for x in _as_number_list(canonical_meta[key])]

    if require_key("image_keys"):
        actual_image_keys = _as_string_list(canonical_meta["image_keys"])
        if actual_image_keys != expected["image_keys"]:
            errors.append(f"image_keys={actual_image_keys}, expected {expected['image_keys']}")
        canonical_meta["image_keys"] = actual_image_keys

    if errors:
        raise ValueError("Contract metadata mismatch: " + "; ".join(errors))

    # Preserve legacy aliases for current callers/checkpoints while making the
    # canonical names available everywhere.
    for canonical_key, legacy_key in _CONTRACT_DEFAULT_TO_LEGACY_KEYS.items():
        if canonical_key in canonical_meta and legacy_key not in canonical_meta:
            canonical_meta[legacy_key] = canonical_meta[canonical_key]

    return canonical_meta


def fill_and_validate_contract_metadata(
    meta: dict[str, Any],
    *,
    context: str = "metadata",
    warn: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """
    Preserve present contract metadata, fill absent keys from defaults, then validate.

    Present-but-contradictory values raise ``ValueError``. Missing values are
    filled from ``get_default_contract_metadata()`` and returned as warnings.
    """
    validate_contract_metadata(meta, strict=False)
    defaults = get_default_contract_metadata()
    filled = _canonicalize_contract_keys(dict(meta))
    warnings: list[str] = []
    for key, value in defaults.items():
        if key not in filled:
            filled[key] = value
            warnings.append(f"{context}: missing {key}; using project default {value!r}")
    filled = validate_contract_metadata(filled, strict=True)
    for canonical_key, legacy_key in _CONTRACT_DEFAULT_TO_LEGACY_KEYS.items():
        filled.setdefault(legacy_key, filled[canonical_key])
    if warn:
        for message in warnings:
            print(f"[data contract warning] {message}", flush=True)
    return filled, warnings


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
            "joint_names": f"[{STATE_ARM_DIM}] optional legacy alias for arm_joint_names",
            "arm_joint_names": f"[{STATE_ARM_DIM}] canonical",
            "hand_value_names": f"[{STATE_HAND_DIM}] canonical",
            "obs_dim": "scalar canonical 26",
            "act_dim": "scalar optional legacy alias for action_dim",
            "action_dim": "scalar canonical 26",
            "state_layout": STATE_LAYOUT,
            "action_layout": ACTION_LAYOUT,
            "action_type": ACTION_TYPE,
            "gripper_mode": GRIPPER_MODE,
            "hand_open_prototype_6": HAND_OPEN_PROTOTYPE_6,
            "hand_closed_prototype_6": HAND_CLOSED_PROTOTYPE_6,
            "left_hand_slice": LEFT_HAND_SLICE,
            "right_hand_slice": RIGHT_HAND_SLICE,
            "state_definition": "str optional",
        },
    }
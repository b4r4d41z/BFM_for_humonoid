import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from torch.utils.data import DataLoader


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
PROJECT_ROOT = _REPO_ROOT

HDF5_DIR = Path("/media/lab/New Volume/hdf5/Sorting_food")
REPORT_DIR = HDF5_DIR / "_check_reports"

BATCH_SIZE = 8
SPLIT_SEED = 42
VAL_RATIO = 0.2
RANDOM_SAMPLE_COUNT = 2

SKIP_HIDDEN_DOTFILES = True


if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bc.data.stream_loader import HDF5DataStreamLoader
from bc.data.batch_assembly import assemble_bfm_batch
from bc.data.schema import (
    ACTION_ARM_DIM,
    ACTION_FULL_DIM,
    ACTION_HAND_DIM,
    IMAGE_KEYS,
    STATE_ARM_DIM,
    STATE_FULL_DIM,
    STATE_HAND_DIM,
    get_default_contract_metadata,
    fill_and_validate_contract_metadata,
)


REQUIRED_H5_PATHS = [
    "obs/state",
    "next_obs/state",
    "act/action",
    "act/joint_target",
    "act/hand_target",
    "done",
    "images/head",
    "images/left_wrist",
    "images/right_wrist",
]

OPTIONAL_H5_PATHS = [
    "meta/joint_names",
    "meta/instruction",
    "meta/bag_name",
    "meta/state_definition",
    "meta/obs_dim",
    "meta/act_dim",
    "meta/action_dim",
    "meta/hand_value_names",
    "meta/action_type",
    "meta/gripper_mode",
    "meta/state_layout",
    "meta/action_layout",
    "meta/hand_open_prototype_6",
    "meta/hand_closed_prototype_6",
    "reward",
]

REQUIRED_SAMPLE_ARRAY_KEY_PATHS = {
    "obs/state/arm": ["obs", "state", "arm"],
    "obs/state/hand": ["obs", "state", "hand"],
    "obs/state/full": ["obs", "state", "full"],
    "action/arm": ["action", "arm"],
    "action/hand": ["action", "hand"],
    "action/full": ["action", "full"],
    "next_obs/state/arm": ["next_obs", "state", "arm"],
    "next_obs/state/hand": ["next_obs", "state", "hand"],
    "next_obs/state/full": ["next_obs", "state", "full"],
    "done": ["done"],
}

OPTIONAL_SAMPLE_ARRAY_KEY_PATHS = {
    "reward": ["reward"],
}

REQUIRED_IMAGE_KEY_PATHS = {
    "head": ["obs", "images", "head"],
    "left_wrist": ["obs", "images", "left_wrist"],
    "right_wrist": ["obs", "images", "right_wrist"],
}

EXPECTED_LAST_DIMS = {
    "obs/state/arm": STATE_ARM_DIM,
    "obs/state/hand": STATE_HAND_DIM,
    "obs/state/full": STATE_FULL_DIM,
    "action/arm": ACTION_ARM_DIM,
    "action/hand": ACTION_HAND_DIM,
    "action/full": ACTION_FULL_DIM,
    "next_obs/state/arm": STATE_ARM_DIM,
    "next_obs/state/hand": STATE_HAND_DIM,
    "next_obs/state/full": STATE_FULL_DIM,
}

META_OPTIONAL_KEYS = {
    "instruction": "string",
    "bag_name": "string",
    "joint_names": "string_array",
    "arm_joint_names": "string_array",
    "hand_value_names": "string_array",
    "obs_dim": "scalar",
    "act_dim": "scalar",
    "action_dim": "scalar",
    "state_definition": "string",
    "action_type": "string",
    "gripper_mode": "string",
    "state_layout": "string",
    "action_layout": "string",
    "hand_open_prototype_6": "numeric_array",
    "hand_closed_prototype_6": "numeric_array",
    "left_hand_slice": "numeric_array",
    "right_hand_slice": "numeric_array",
}


def is_valid_hdf5(file_path: Path) -> tuple[bool, str]:
    if not file_path.exists():
        return False, "file does not exist"

    if not file_path.is_file():
        return False, "not a regular file"

    try:
        with h5py.File(file_path, "r"):
            pass
        return True, "OK"
    except Exception as e:
        return False, str(e)


def should_skip_file(file_path: Path) -> bool:
    name = file_path.name

    if SKIP_HIDDEN_DOTFILES and name.startswith("."):
        return True

    temp_suffixes = (".tmp", ".partial", ".swp", ".crdownload")
    if name.endswith(temp_suffixes):
        return True

    return False


def safe_shape(obj: Any) -> str:
    if hasattr(obj, "shape"):
        try:
            return str(tuple(obj.shape))
        except Exception:
            return str(obj.shape)
    return f"<no shape, type={type(obj).__name__}>"


def safe_dtype(obj: Any) -> str:
    if hasattr(obj, "dtype"):
        try:
            return str(obj.dtype)
        except Exception:
            return "<dtype unavailable>"
    return f"<no dtype, type={type(obj).__name__}>"


def get_nested(d: dict, keys: list[str]) -> Any:
    cur = d
    path_so_far = []

    for key in keys:
        path_so_far.append(key)
        if not isinstance(cur, dict):
            raise KeyError(
                f"Expected dict at {'/'.join(path_so_far[:-1])}, got {type(cur).__name__}"
            )
        if key not in cur:
            raise KeyError(f"Missing key: {'/'.join(path_so_far)}")
        cur = cur[key]

    return cur


def try_get_nested(d: dict, keys: list[str]) -> tuple[bool, Any | None, str | None]:
    try:
        return True, get_nested(d, keys), None
    except Exception as e:
        return False, None, str(e)


def to_numpy(obj: Any) -> np.ndarray | None:
    if obj is None:
        return None

    if isinstance(obj, np.ndarray):
        return obj

    if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        try:
            return obj.detach().cpu().numpy()
        except Exception:
            pass

    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        try:
            return obj.cpu().numpy()
        except Exception:
            pass

    try:
        return np.asarray(obj)
    except Exception:
        return None


def read_scalar_from_h5(h5f: h5py.File, path: str) -> Any:
    value = h5f[path][()]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return str(value)
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def read_list_from_h5(h5f: h5py.File, path: str) -> list[Any]:
    raw = np.asarray(h5f[path][()])
    if raw.shape == ():
        raw = raw.reshape(1)
    out = []
    for item in raw.tolist():
        out.append(item.decode("utf-8") if isinstance(item, bytes) else item)
    return out


def read_contract_meta_from_h5(h5f: h5py.File) -> tuple[dict[str, Any], bool]:
    paths = {
        "obs_dim": "meta/obs_dim",
        "act_dim": "meta/act_dim",
        "action_dim": "meta/action_dim",
        "joint_names": "meta/joint_names",
        "arm_joint_names": "meta/joint_names",
        "hand_value_names": "meta/hand_value_names",
        "action_type": "meta/action_type",
        "gripper_mode": "meta/gripper_mode",
        "state_layout": "meta/state_layout",
        "action_layout": "meta/action_layout",
        "state_definition": "meta/state_definition",
        "hand_open_prototype_6": "meta/hand_open_prototype_6",
        "hand_closed_prototype_6": "meta/hand_closed_prototype_6",
    }
    meta: dict[str, Any] = {}
    metadata_exists = "meta" in h5f
    for key, path in paths.items():
        if path not in h5f:
            continue
        if key in {"joint_names", "arm_joint_names", "hand_value_names"}:
            meta[key] = [str(x) for x in read_list_from_h5(h5f, path)]
        elif key in {"hand_open_prototype_6", "hand_closed_prototype_6"}:
            meta[key] = read_list_from_h5(h5f, path)
        elif key in {"obs_dim", "act_dim", "action_dim"}:
            meta[key] = int(np.asarray(h5f[path][()]).reshape(-1)[0])
        else:
            meta[key] = str(read_scalar_from_h5(h5f, path))
    return meta, metadata_exists


def dataset_info_h5(h5f: h5py.File, path: str) -> dict[str, Any]:
    obj = h5f[path]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"{path} exists but is not an HDF5 dataset")
    return {
        "shape": tuple(obj.shape),
        "dtype": str(obj.dtype),
        "ndim": int(obj.ndim),
    }


def get_first_dim_h5(h5f: h5py.File, path: str) -> int | None:
    ds = h5f[path]
    if not isinstance(ds, h5py.Dataset):
        return None
    if ds.ndim == 0:
        return None
    return int(ds.shape[0])


def choose_indices(length: int, extra_random: int = RANDOM_SAMPLE_COUNT) -> list[int]:
    if length <= 0:
        return []

    indices = {0, max(0, length // 2), max(0, length - 1)}

    if length > 3:
        rng = random.Random(SPLIT_SEED)
        for _ in range(extra_random):
            indices.add(rng.randint(0, length - 1))

    return sorted(indices)


def check_numeric_array(arr: np.ndarray, name: str) -> tuple[list[str], list[str], dict[str, Any]]:
    issues: list[str] = []
    warnings: list[str] = []

    meta: dict[str, Any] = {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "ndim": int(arr.ndim),
    }

    if arr.size == 0:
        issues.append(f"{name}: empty array")
        return issues, warnings, meta

    if np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_:
        finite_mask = np.isfinite(arr.astype(np.float32)) if arr.dtype == np.bool_ else np.isfinite(arr)
        if not np.all(finite_mask):
            bad_count = int((~finite_mask).sum())
            issues.append(f"{name}: contains non-finite values (count={bad_count})")

        try:
            meta["min"] = float(np.nanmin(arr.astype(np.float32)))
            meta["max"] = float(np.nanmax(arr.astype(np.float32)))
        except Exception:
            pass

        if np.all(arr == 0):
            warnings.append(f"{name}: all values are zero")

    return issues, warnings, meta


def check_image_array(arr: np.ndarray, name: str) -> tuple[list[str], list[str], dict[str, Any]]:
    issues: list[str] = []
    warnings: list[str] = []

    meta: dict[str, Any] = {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "ndim": int(arr.ndim),
    }

    if arr.size == 0:
        issues.append(f"{name}: empty image array")
        return issues, warnings, meta

    if arr.ndim not in (3, 4):
        warnings.append(f"{name}: unusual image ndim={arr.ndim}, expected 3 or 4")

    if arr.ndim >= 3:
        last_dim = arr.shape[-1]
        if last_dim not in (1, 3, 4):
            warnings.append(
                f"{name}: last dim is {last_dim}, expected 1/3/4 for channel-last image"
            )

    if np.issubdtype(arr.dtype, np.number):
        finite_mask = np.isfinite(arr)
        if not np.all(finite_mask):
            bad_count = int((~finite_mask).sum())
            issues.append(f"{name}: contains non-finite values (count={bad_count})")

        try:
            meta["min"] = float(np.nanmin(arr))
            meta["max"] = float(np.nanmax(arr))
        except Exception:
            pass

        if np.all(arr == 0):
            warnings.append(f"{name}: image values are all zero")

        if arr.dtype != np.uint8:
            warnings.append(f"{name}: image dtype is {arr.dtype}, expected uint8 in many setups")

    return issues, warnings, meta


def check_any_array(obj: Any, name: str, treat_as_image: bool = False) -> tuple[list[str], list[str], dict[str, Any]]:
    arr = to_numpy(obj)
    if arr is None:
        return [f"{name}: cannot convert to numpy array"], [], {}

    if treat_as_image:
        return check_image_array(arr, name)

    return check_numeric_array(arr, name)


def check_expected_last_dim(obj: Any, expected_dim: int, name: str) -> list[str]:
    arr = to_numpy(obj)
    if arr is None:
        return [f"{name}: cannot convert to numpy array for dimension check"]

    if arr.ndim == 0:
        return [f"{name}: scalar found, expected last dim = {expected_dim}"]

    if arr.shape[-1] != expected_dim:
        return [f"{name}: expected last dim {expected_dim}, got {arr.shape[-1]}"]

    return []


def sample_h5_dataset_slices(h5f: h5py.File, path: str, max_samples: int = 3) -> list[np.ndarray]:
    ds = h5f[path]
    if not isinstance(ds, h5py.Dataset):
        return []

    if ds.ndim == 0:
        return [np.asarray(ds[()])]

    length = ds.shape[0]
    if length == 0:
        return []

    indices = choose_indices(length, extra_random=0)[:max_samples]
    out = []
    for idx in indices:
        try:
            out.append(np.asarray(ds[idx]))
        except Exception:
            pass
    return out


def inspect_string_value(value: Any, name: str) -> tuple[list[str], list[str], dict[str, Any]]:
    issues: list[str] = []
    warnings: list[str] = []
    meta: dict[str, Any] = {"type": type(value).__name__}

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
            meta["decoded"] = True
        except Exception:
            warnings.append(f"{name}: bytes value could not be decoded as utf-8")

    if isinstance(value, str):
        meta["length"] = len(value)
        if len(value.strip()) == 0:
            warnings.append(f"{name}: empty string")
    else:
        warnings.append(f"{name}: expected string-like value, got {type(value).__name__}")

    return issues, warnings, meta


def inspect_scalar_value(value: Any, name: str) -> tuple[list[str], list[str], dict[str, Any]]:
    issues: list[str] = []
    warnings: list[str] = []
    meta: dict[str, Any] = {"type": type(value).__name__}

    arr = to_numpy(value)
    if arr is None:
        warnings.append(f"{name}: could not convert to numpy scalar")
        return issues, warnings, meta

    if arr.ndim > 0 and arr.size > 1:
        warnings.append(f"{name}: expected scalar-like value, got shape {arr.shape}")
    else:
        try:
            meta["value"] = arr.item()
        except Exception:
            pass

    return issues, warnings, meta


def inspect_joint_names_value(value: Any, name: str) -> tuple[list[str], list[str], dict[str, Any]]:
    issues: list[str] = []
    warnings: list[str] = []
    meta: dict[str, Any] = {"type": type(value).__name__}

    if isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        arr = to_numpy(value)
        if arr is not None and arr.ndim >= 1:
            items = arr.tolist()
        else:
            warnings.append(f"{name}: expected array/list-like value")
            return issues, warnings, meta

    meta["count"] = len(items)

    expected_count = ACTION_HAND_DIM if "hand_value_names" in name else ACTION_ARM_DIM
    label = "hand value names" if "hand_value_names" in name else "joint names"
    if len(items) != expected_count:
        issues.append(
            f"{name}: expected {expected_count} {label}, got {len(items)}"
        )

    return issues, warnings, meta


def validate_sample_meta(sample: dict[str, Any], idx: int) -> tuple[list[str], list[str], dict[str, Any]]:
    issues: list[str] = []
    warnings: list[str] = []
    meta_report: dict[str, Any] = {}

    ok, meta_obj, err = try_get_nested(sample, ["meta"])
    if not ok:
        warnings.append(f"dataset[{idx}]/meta: missing optional meta block")
        return issues, warnings, meta_report

    if not isinstance(meta_obj, dict):
        issues.append(f"dataset[{idx}]/meta: expected dict, got {type(meta_obj).__name__}")
        return issues, warnings, meta_report

    for key, kind in META_OPTIONAL_KEYS.items():
        if key not in meta_obj:
            continue

        value = meta_obj[key]
        name = f"dataset[{idx}]/meta/{key}"

        if kind == "string":
            sub_issues, sub_warnings, sub_meta = inspect_string_value(value, name)
        elif kind == "scalar":
            sub_issues, sub_warnings, sub_meta = inspect_scalar_value(value, name)
        elif kind == "string_array":
            sub_issues, sub_warnings, sub_meta = inspect_joint_names_value(value, name)
        elif kind == "numeric_array":
            sub_issues, sub_warnings, sub_meta = check_any_array(value, name)
        else:
            sub_issues, sub_warnings, sub_meta = [], [], {}

        issues.extend(sub_issues)
        warnings.extend(sub_warnings)
        meta_report[f"meta/{key}"] = sub_meta

        if key == "obs_dim":
            arr = to_numpy(value)
            if arr is not None and arr.size == 1:
                try:
                    scalar = int(arr.item())
                    if scalar != STATE_FULL_DIM:
                        issues.append(
                            f"{name}: expected obs_dim={STATE_FULL_DIM}, got {scalar}"
                        )
                except Exception:
                    warnings.append(f"{name}: could not parse integer value")

        if key == "act_dim":
            arr = to_numpy(value)
            if arr is not None and arr.size == 1:
                try:
                    scalar = int(arr.item())
                    if scalar != ACTION_FULL_DIM:
                        issues.append(
                            f"{name}: expected act_dim={ACTION_FULL_DIM}, got {scalar}"
                        )
                except Exception:
                    warnings.append(f"{name}: could not parse integer value")

    if isinstance(meta_obj, dict):
        try:
            contract_meta, contract_warnings = fill_and_validate_contract_metadata(
                meta_obj, context=f"dataset[{idx}]/meta", warn=False
            )
            warnings.extend(contract_warnings)
            meta_report["contract"] = {
                "compatible": True,
                "metadata_exists": contract_meta.get("metadata_exists", True),
                "metadata_source": contract_meta.get("metadata_source", "file"),
                "metadata_defaulted_fields": contract_meta.get("metadata_defaulted_fields", []),
                "arm_joint_order": contract_meta["arm_joint_names"],
                "hand_value_order": contract_meta["hand_value_names"],
                "action_type": contract_meta["action_type"],
                "gripper_mode": contract_meta["gripper_mode"],
                "state_layout": contract_meta["state_layout"],
                "action_layout": contract_meta["action_layout"],
                "hand_open_prototype_6": contract_meta["hand_open_prototype_6"],
                "hand_closed_prototype_6": contract_meta["hand_closed_prototype_6"],
                "left_hand_slice": contract_meta["left_hand_slice"],
                "right_hand_slice": contract_meta["right_hand_slice"],
            }
        except Exception as e:
            issues.append(f"dataset[{idx}]/meta contract incompatible: {e}")

    return issues, warnings, meta_report


def validate_batch_meta(batch: dict[str, Any], batch_size: int) -> tuple[list[str], list[str], dict[str, Any]]:
    issues: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {}

    if "meta" not in batch:
        warnings.append("batch/meta: missing optional meta block")
        return issues, warnings, report

    meta_obj = batch["meta"]
    report["type"] = type(meta_obj).__name__

    if not isinstance(meta_obj, list):
        issues.append(f"batch/meta: expected list, got {type(meta_obj).__name__}")
        return issues, warnings, report

    report["length"] = len(meta_obj)

    if len(meta_obj) != batch_size:
        issues.append(f"batch/meta: length {len(meta_obj)} != batch_size {batch_size}")

    if len(meta_obj) > 0:
        first = meta_obj[0]
        report["first_item_type"] = type(first).__name__
        if not isinstance(first, dict):
            issues.append(
                f"batch/meta[0]: expected dict, got {type(first).__name__}"
            )
        else:
            report["first_item_keys"] = sorted(first.keys())

    return issues, warnings, report


def validate_h5_file(file_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file": str(file_path),
        "status": "UNKNOWN",
        "issues": [],
        "warnings": [],
        "h5_info": {},
        "lengths": {},
        "meta": {},
    }

    valid, reason = is_valid_hdf5(file_path)
    if not valid:
        result["status"] = "INVALID_HDF5"
        result["issues"].append(f"Invalid HDF5: {reason}")
        return result

    with h5py.File(file_path, "r") as h5f:
        for path in REQUIRED_H5_PATHS:
            if path not in h5f:
                result["issues"].append(f"Missing required path: {path}")
            else:
                try:
                    result["h5_info"][path] = dataset_info_h5(h5f, path)
                except Exception as e:
                    result["issues"].append(f"{path}: {type(e).__name__}: {e}")

        for path in OPTIONAL_H5_PATHS:
            if path in h5f:
                try:
                    if isinstance(h5f[path], h5py.Dataset):
                        result["h5_info"][path] = dataset_info_h5(h5f, path)
                    else:
                        result["warnings"].append(f"{path}: exists but is not a dataset")
                except Exception as e:
                    result["warnings"].append(f"{path}: {type(e).__name__}: {e}")

        length_paths = [
            "obs/state",
            "next_obs/state",
            "act/action",
            "act/joint_target",
            "act/hand_target",
            "done",
            "images/head",
            "images/left_wrist",
            "images/right_wrist",
        ]

        lengths_found = {}
        for path in length_paths:
            if path in h5f:
                first_dim = get_first_dim_h5(h5f, path)
                result["lengths"][path] = first_dim
                if first_dim is not None:
                    lengths_found[path] = first_dim

        if lengths_found:
            unique_lengths = set(lengths_found.values())
            if len(unique_lengths) != 1:
                result["issues"].append(
                    f"Inconsistent first dimension across modalities: {lengths_found}"
                )

        # Strict shape consistency with frozen dimensions
        expected_second_dims = {
            "obs/state": STATE_FULL_DIM,
            "next_obs/state": STATE_FULL_DIM,
            "act/action": ACTION_FULL_DIM,
            "act/joint_target": ACTION_ARM_DIM,
            "act/hand_target": ACTION_HAND_DIM,
        }

        for path, expected_dim in expected_second_dims.items():
            if path in h5f and isinstance(h5f[path], h5py.Dataset):
                ds = h5f[path]
                if ds.ndim < 2:
                    result["issues"].append(
                        f"{path}: expected ndim >= 2, got {ds.ndim}"
                    )
                elif ds.shape[1] != expected_dim:
                    result["issues"].append(
                        f"{path}: expected second dim {expected_dim}, got {ds.shape[1]}"
                    )

        # Semantic project contract metadata checks. Missing fields are filled from
        # the hard-coded 26D defaults with warnings; contradictory fields fail.
        raw_contract_meta, metadata_exists = read_contract_meta_from_h5(h5f)
        result["meta"]["metadata_exists"] = metadata_exists
        try:
            contract_meta, contract_warnings = fill_and_validate_contract_metadata(
                raw_contract_meta, context=f"{file_path}/meta", warn=False
            )
            result["warnings"].extend(contract_warnings)
            result["meta"].update(
                {
                    "contract_compatible": True,
                    "metadata_source": "file" if not contract_warnings else "file_with_defaults",
                    "metadata_defaulted_fields": [
                        msg.split(": missing ", 1)[1].split(";", 1)[0]
                        for msg in contract_warnings
                    ],
                    "obs_dim": contract_meta["obs_dim"],
                    "act_dim": contract_meta["act_dim"],
                    "action_dim": contract_meta["action_dim"],
                    "arm_joint_order": contract_meta["arm_joint_names"],
                    "hand_value_order": contract_meta["hand_value_names"],
                    "action_type": contract_meta["action_type"],
                    "gripper_mode": contract_meta["gripper_mode"],
                    "state_layout": contract_meta["state_layout"],
                    "action_layout": contract_meta["action_layout"],
                    "state_definition": contract_meta.get("state_definition"),
                    "hand_open_prototype_6": contract_meta["hand_open_prototype_6"],
                    "hand_closed_prototype_6": contract_meta["hand_closed_prototype_6"],
                    "left_hand_slice": contract_meta["left_hand_slice"],
                    "right_hand_slice": contract_meta["right_hand_slice"],
                    "image_keys": contract_meta["image_keys"],
                }
            )
        except Exception as e:
            result["meta"]["contract_compatible"] = False
            result["issues"].append(f"Semantic contract metadata incompatible: {e}")

        # Meta checks
        try:
            if "meta/act_dim" in h5f:
                act_dim_meta = int(read_scalar_from_h5(h5f, "meta/act_dim"))
                result["meta"]["act_dim"] = act_dim_meta
                if act_dim_meta != ACTION_FULL_DIM:
                    result["issues"].append(
                        f"meta/act_dim={act_dim_meta}, expected {ACTION_FULL_DIM}"
                    )
        except Exception as e:
            result["warnings"].append(f"Could not validate meta/act_dim: {e}")

        try:
            if "meta/obs_dim" in h5f:
                obs_dim_meta = int(read_scalar_from_h5(h5f, "meta/obs_dim"))
                result["meta"]["obs_dim"] = obs_dim_meta
                if obs_dim_meta != STATE_FULL_DIM:
                    result["issues"].append(
                        f"meta/obs_dim={obs_dim_meta}, expected {STATE_FULL_DIM}"
                    )
        except Exception as e:
            result["warnings"].append(f"Could not validate meta/obs_dim: {e}")

        try:
            if "meta/joint_names" in h5f:
                joint_names_ds = h5f["meta/joint_names"]
                if isinstance(joint_names_ds, h5py.Dataset):
                    joint_names_count = int(joint_names_ds.shape[0])
                    result["meta"]["joint_names_count"] = joint_names_count
                    if joint_names_count != ACTION_ARM_DIM:
                        result["issues"].append(
                            f"meta/joint_names count={joint_names_count}, expected {ACTION_ARM_DIM}"
                        )
        except Exception as e:
            result["warnings"].append(f"Could not validate meta/joint_names: {e}")

        numeric_paths = [
            "obs/state",
            "next_obs/state",
            "act/action",
            "act/joint_target",
            "act/hand_target",
            "done",
        ]

        image_paths = [
            "images/head",
            "images/left_wrist",
            "images/right_wrist",
        ]

        for path in numeric_paths:
            if path in h5f and isinstance(h5f[path], h5py.Dataset):
                slices = sample_h5_dataset_slices(h5f, path)
                for i, arr in enumerate(slices):
                    issues, warnings, meta = check_any_array(arr, f"{path}[sample_{i}]")
                    result["issues"].extend(issues)
                    result["warnings"].extend(warnings)
                    if meta:
                        result["h5_info"][f"{path}[sample_{i}]"] = meta

        for path in image_paths:
            if path in h5f and isinstance(h5f[path], h5py.Dataset):
                slices = sample_h5_dataset_slices(h5f, path)
                for i, arr in enumerate(slices):
                    issues, warnings, meta = check_any_array(
                        arr, f"{path}[sample_{i}]", treat_as_image=True
                    )
                    result["issues"].extend(issues)
                    result["warnings"].extend(warnings)
                    if meta:
                        result["h5_info"][f"{path}[sample_{i}]"] = meta

    if result["issues"]:
        result["status"] = "FAIL_H5_VALIDATION"
    else:
        result["status"] = "OK_H5_VALIDATION"

    return result


def validate_loader_and_batch(file_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file": str(file_path),
        "status": "UNKNOWN",
        "issues": [],
        "warnings": [],
        "dataset_len": None,
        "sample_checks": {},
        "batch_checks": {},
    }

    dataset = HDF5DataStreamLoader(str(file_path))

    try:
        dataset_len = len(dataset)
    except Exception as e:
        result["status"] = "FAIL_LOADER"
        result["issues"].append(f"Cannot get dataset length: {e}")
        return result

    result["dataset_len"] = dataset_len

    if dataset_len <= 0:
        result["status"] = "FAIL_LOADER"
        result["issues"].append("Dataset length is zero")
        return result

    sample_indices = choose_indices(dataset_len)

    for idx in sample_indices:
        try:
            sample = dataset[idx]
        except Exception as e:
            result["issues"].append(f"Cannot read dataset[{idx}]: {e}")
            continue

        sample_result = {
            "index": idx,
            "found": {},
            "missing_optional": [],
        }

        # Required canonical arrays
        for path_str, key_path in REQUIRED_SAMPLE_ARRAY_KEY_PATHS.items():
            ok, value, err = try_get_nested(sample, key_path)
            if not ok:
                result["issues"].append(f"dataset[{idx}] missing required key {path_str}: {err}")
                continue

            entry = {
                "shape": safe_shape(value),
                "dtype": safe_dtype(value),
            }

            issues, warnings, meta = check_any_array(value, f"dataset[{idx}]/{path_str}")
            result["issues"].extend(issues)
            result["warnings"].extend(warnings)
            if meta:
                entry.update(meta)

            if path_str in EXPECTED_LAST_DIMS:
                dim_issues = check_expected_last_dim(
                    value, EXPECTED_LAST_DIMS[path_str], f"dataset[{idx}]/{path_str}"
                )
                result["issues"].extend(dim_issues)

            sample_result["found"][path_str] = entry

        # Optional sample numeric/scalar tensors
        for path_str, key_path in OPTIONAL_SAMPLE_ARRAY_KEY_PATHS.items():
            ok, value, _ = try_get_nested(sample, key_path)
            if not ok:
                sample_result["missing_optional"].append(path_str)
                continue

            entry = {
                "shape": safe_shape(value),
                "dtype": safe_dtype(value),
            }

            issues, warnings, meta = check_any_array(value, f"dataset[{idx}]/{path_str}")
            result["issues"].extend(issues)
            result["warnings"].extend(warnings)
            if meta:
                entry.update(meta)

            sample_result["found"][path_str] = entry

        # Required images
        for cam_name, key_path in REQUIRED_IMAGE_KEY_PATHS.items():
            path_str = "/".join(key_path)
            ok, value, err = try_get_nested(sample, key_path)
            if not ok:
                result["issues"].append(f"dataset[{idx}] missing required image {cam_name}: {err}")
                continue

            entry = {
                "shape": safe_shape(value),
                "dtype": safe_dtype(value),
            }

            issues, warnings, meta = check_any_array(
                value, f"dataset[{idx}]/{path_str}", treat_as_image=True
            )
            result["issues"].extend(issues)
            result["warnings"].extend(warnings)
            if meta:
                entry.update(meta)

            sample_result["found"][path_str] = entry

        # Optional meta block
        meta_issues, meta_warnings, meta_report = validate_sample_meta(sample, idx)
        result["issues"].extend(meta_issues)
        result["warnings"].extend(meta_warnings)
        if meta_report:
            sample_result["meta"] = meta_report

        result["sample_checks"][str(idx)] = sample_result

    batch_size = min(BATCH_SIZE, dataset_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(dataset_len > 1),
        collate_fn=assemble_bfm_batch,
        num_workers=0,
    )

    try:
        batch = next(iter(loader))
    except Exception as e:
        result["status"] = "FAIL_BATCH"
        result["issues"].append(f"Cannot assemble batch: {e}")
        return result

    # Required canonical arrays in batch
    for path_str, key_path in REQUIRED_SAMPLE_ARRAY_KEY_PATHS.items():
        ok, value, err = try_get_nested(batch, key_path)
        if not ok:
            result["issues"].append(f"Batch missing required key {path_str}: {err}")
            continue

        entry = {
            "shape": safe_shape(value),
            "dtype": safe_dtype(value),
        }

        issues, warnings, meta = check_any_array(value, f"batch/{path_str}")
        result["issues"].extend(issues)
        result["warnings"].extend(warnings)
        if meta:
            entry.update(meta)

        arr = to_numpy(value)
        if arr is not None and arr.ndim >= 1 and arr.shape[0] != batch_size:
            result["issues"].append(
                f"batch/{path_str}: first dimension {arr.shape[0]} != batch_size {batch_size}"
            )

        if path_str in EXPECTED_LAST_DIMS:
            dim_issues = check_expected_last_dim(
                value, EXPECTED_LAST_DIMS[path_str], f"batch/{path_str}"
            )
            result["issues"].extend(dim_issues)

        result["batch_checks"][path_str] = entry

    # Optional batch arrays
    for path_str, key_path in OPTIONAL_SAMPLE_ARRAY_KEY_PATHS.items():
        ok, value, _ = try_get_nested(batch, key_path)
        if not ok:
            continue

        entry = {
            "shape": safe_shape(value),
            "dtype": safe_dtype(value),
        }

        issues, warnings, meta = check_any_array(value, f"batch/{path_str}")
        result["issues"].extend(issues)
        result["warnings"].extend(warnings)
        if meta:
            entry.update(meta)

        arr = to_numpy(value)
        if arr is not None and arr.ndim >= 1 and arr.shape[0] != batch_size:
            result["issues"].append(
                f"batch/{path_str}: first dimension {arr.shape[0]} != batch_size {batch_size}"
            )

        result["batch_checks"][path_str] = entry

    # Required images in batch
    for cam_name, key_path in REQUIRED_IMAGE_KEY_PATHS.items():
        path_str = "/".join(key_path)
        ok, value, err = try_get_nested(batch, key_path)
        if not ok:
            result["issues"].append(f"Batch missing required image {cam_name}: {err}")
            continue

        entry = {
            "shape": safe_shape(value),
            "dtype": safe_dtype(value),
        }

        issues, warnings, meta = check_any_array(
            value, f"batch/{path_str}", treat_as_image=True
        )
        result["issues"].extend(issues)
        result["warnings"].extend(warnings)
        if meta:
            entry.update(meta)

        arr = to_numpy(value)
        if arr is not None and arr.ndim >= 1 and arr.shape[0] != batch_size:
            result["issues"].append(
                f"batch/{path_str}: first dimension {arr.shape[0]} != batch_size {batch_size}"
            )

        result["batch_checks"][path_str] = entry

    # Optional batch meta
    meta_issues, meta_warnings, meta_report = validate_batch_meta(batch, batch_size)
    result["issues"].extend(meta_issues)
    result["warnings"].extend(meta_warnings)
    if meta_report:
        result["batch_checks"]["meta"] = meta_report

    if result["issues"]:
        result["status"] = "FAIL_LOADER_OR_BATCH"
    else:
        result["status"] = "OK_LOADER_AND_BATCH"

    return result


def build_train_val_split(
    valid_files: list[str],
    seed: int = SPLIT_SEED,
    val_ratio: float = VAL_RATIO,
) -> dict[str, Any]:
    files = sorted(valid_files)
    rng = random.Random(seed)
    rng.shuffle(files)

    if not files:
        return {
            "seed": seed,
            "val_ratio": val_ratio,
            "train_files": [],
            "val_files": [],
        }

    if len(files) == 1:
        return {
            "seed": seed,
            "val_ratio": val_ratio,
            "train_files": files,
            "val_files": [],
        }

    val_count = max(1, int(round(len(files) * val_ratio)))
    val_count = min(val_count, len(files) - 1)

    val_files = sorted(files[:val_count])
    train_files = sorted(files[val_count:])

    return {
        "seed": seed,
        "val_ratio": val_ratio,
        "train_files": train_files,
        "val_files": val_files,
    }


def save_report(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "check_data_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    csv_path = report_dir / "check_data_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "h5_status",
                "loader_status",
                "final_status",
                "issue_count",
                "warning_count",
                "dataset_len",
            ],
        )
        writer.writeheader()

        for row in report["files"]:
            writer.writerow(
                {
                    "file": row["file"],
                    "h5_status": row.get("h5_status", ""),
                    "loader_status": row.get("loader_status", ""),
                    "final_status": row.get("final_status", ""),
                    "issue_count": len(row.get("issues", [])),
                    "warning_count": len(row.get("warnings", [])),
                    "dataset_len": row.get("dataset_len", ""),
                }
            )

    split_path = report_dir / "train_val_split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(report["split"], f, ensure_ascii=False, indent=2)

    print(f"\nSaved JSON report : {json_path}")
    print(f"Saved CSV summary : {csv_path}")
    print(f"Saved split file  : {split_path}")


def main() -> None:
    if not HDF5_DIR.exists():
        raise FileNotFoundError(f"HDF5 directory not found: {HDF5_DIR}")

    files = sorted(HDF5_DIR.glob("*.h5"))
    files = [p for p in files if not should_skip_file(p)]

    if not files:
        raise FileNotFoundError(f"No usable .h5 files found in: {HDF5_DIR}")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"HDF5 dir     : {HDF5_DIR}")
    print(f"Report dir   : {REPORT_DIR}")
    print(f"Found files  : {len(files)}")

    final_report: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "hdf5_dir": str(HDF5_DIR),
        "report_dir": str(REPORT_DIR),
        "files": [],
        "summary": {},
        "split": {},
    }

    valid_for_training: list[str] = []

    ok_all = 0
    fail_h5 = 0
    fail_loader = 0

    for file_path in files:
        print("\n" + "=" * 100)
        print(f"Checking: {file_path}")

        h5_result = validate_h5_file(file_path)
        loader_result = {
            "status": "SKIPPED",
            "issues": [],
            "warnings": [],
            "dataset_len": None,
            "sample_checks": {},
            "batch_checks": {},
        }

        if h5_result["status"] == "OK_H5_VALIDATION":
            try:
                loader_result = validate_loader_and_batch(file_path)
            except Exception as e:
                loader_result = {
                    "status": "FAIL_LOADER_OR_BATCH",
                    "issues": [f"Unexpected loader/batch exception: {type(e).__name__}: {e}"],
                    "warnings": [],
                    "dataset_len": None,
                    "sample_checks": {},
                    "batch_checks": {},
                }

        combined_issues = h5_result["issues"] + loader_result["issues"]
        combined_warnings = h5_result["warnings"] + loader_result["warnings"]

        if h5_result["status"] == "OK_H5_VALIDATION" and loader_result["status"] == "OK_LOADER_AND_BATCH":
            final_status = "OK"
            ok_all += 1
            valid_for_training.append(str(file_path))
        elif h5_result["status"] != "OK_H5_VALIDATION":
            final_status = "FAIL_H5"
            fail_h5 += 1
        else:
            final_status = "FAIL_LOADER_OR_BATCH"
            fail_loader += 1

        file_report = {
            "file": str(file_path),
            "h5_status": h5_result["status"],
            "loader_status": loader_result["status"],
            "final_status": final_status,
            "dataset_len": loader_result.get("dataset_len"),
            "issues": combined_issues,
            "warnings": combined_warnings,
            "h5_info": h5_result.get("h5_info", {}),
            "lengths": h5_result.get("lengths", {}),
            "meta": h5_result.get("meta", {}),
            "sample_checks": loader_result.get("sample_checks", {}),
            "batch_checks": loader_result.get("batch_checks", {}),
        }
        final_report["files"].append(file_report)

        print(f"H5 status      : {h5_result['status']}")
        print(f"Loader status  : {loader_result['status']}")
        print(f"Final status   : {final_status}")
        print(f"Issues         : {len(combined_issues)}")
        print(f"Warnings       : {len(combined_warnings)}")
        contract_meta = h5_result.get("meta", {})
        print(f"Metadata exists: {contract_meta.get('metadata_exists')}")
        print(f"Metadata source: {contract_meta.get('metadata_source')}")
        print(f"Contract OK    : {contract_meta.get('contract_compatible')}")
        print(f"Arm order      : {contract_meta.get('arm_joint_order')}")
        print(f"Hand order     : {contract_meta.get('hand_value_order')}")
        print(f"Action type    : {contract_meta.get('action_type')}")
        print(f"Gripper mode   : {contract_meta.get('gripper_mode')}")
        print(f"State layout   : {contract_meta.get('state_layout')}")
        print(f"Action layout  : {contract_meta.get('action_layout')}")
        print(f"Hand open      : {contract_meta.get('hand_open_prototype_6')}")
        print(f"Hand closed    : {contract_meta.get('hand_closed_prototype_6')}")
        print(f"Left slice     : {contract_meta.get('left_hand_slice')}")
        print(f"Right slice    : {contract_meta.get('right_hand_slice')}")

        if combined_issues:
            print("Issue list:")
            for msg in combined_issues:
                print(f"  - {msg}")

        if combined_warnings:
            print("Warning list:")
            for msg in combined_warnings:
                print(f"  - {msg}")

    split = build_train_val_split(valid_for_training, seed=SPLIT_SEED, val_ratio=VAL_RATIO)
    final_report["split"] = split

    final_report["summary"] = {
        "total_files": len(files),
        "ok_all": ok_all,
        "fail_h5": fail_h5,
        "fail_loader_or_batch": fail_loader,
        "valid_for_training": len(valid_for_training),
        "train_count": len(split["train_files"]),
        "val_count": len(split["val_files"]),
        "split_seed": SPLIT_SEED,
        "val_ratio": VAL_RATIO,
    }

    print("\n" + "=" * 100)
    print("SUMMARY")
    print(f"Total files            : {final_report['summary']['total_files']}")
    print(f"Fully OK               : {final_report['summary']['ok_all']}")
    print(f"Failed H5 validation   : {final_report['summary']['fail_h5']}")
    print(f"Failed loader/batch    : {final_report['summary']['fail_loader_or_batch']}")
    print(f"Valid for training     : {final_report['summary']['valid_for_training']}")
    print(f"Train split count      : {final_report['summary']['train_count']}")
    print(f"Val split count        : {final_report['summary']['val_count']}")

    save_report(final_report, REPORT_DIR)


if __name__ == "__main__":
    main()
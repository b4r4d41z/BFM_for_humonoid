from __future__ import annotations

from typing import Any

import torch

from learning.bc.data.schema import IMAGE_KEYS


def _get_nested(d: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = d
    path_so_far: list[str] = []

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


def _has_nested(d: dict[str, Any], keys: list[str]) -> bool:
    try:
        _get_nested(d, keys)
        return True
    except Exception:
        return False


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _stack_required(samples: list[dict[str, Any]], keys: list[str]) -> torch.Tensor:
    path_str = "/".join(keys)
    values: list[torch.Tensor] = []

    for i, sample in enumerate(samples):
        try:
            value = _get_nested(sample, keys)
        except Exception as e:
            raise KeyError(
                f"Sample {i} is missing required field '{path_str}': {e}"
            ) from e

        values.append(_to_tensor(value))

    try:
        return torch.stack(values, dim=0)
    except Exception as e:
        shapes = [tuple(v.shape) for v in values]
        raise RuntimeError(
            f"Failed to stack required field '{path_str}'. "
            f"Shapes: {shapes}. Reason: {e}"
        ) from e


def _field_presence_state(samples: list[dict[str, Any]], keys: list[str]) -> str:
    present = [_has_nested(s, keys) for s in samples]

    if all(present):
        return "all"
    if not any(present):
        return "none"
    return "partial"


def _stack_optional(samples: list[dict[str, Any]], keys: list[str]) -> torch.Tensor | None:
    state = _field_presence_state(samples, keys)
    path_str = "/".join(keys)

    if state == "none":
        return None

    if state == "partial":
        missing = [i for i, s in enumerate(samples) if not _has_nested(s, keys)]
        raise ValueError(
            f"Optional field '{path_str}' exists only in some samples. "
            f"Missing in indices: {missing}"
        )

    values = [_to_tensor(_get_nested(s, keys)) for s in samples]

    try:
        return torch.stack(values, dim=0)
    except Exception as e:
        shapes = [tuple(v.shape) for v in values]
        raise RuntimeError(
            f"Failed to stack optional field '{path_str}'. "
            f"Shapes: {shapes}. Reason: {e}"
        ) from e


def _collect_optional_list(samples: list[dict[str, Any]], keys: list[str]) -> list[Any] | None:
    state = _field_presence_state(samples, keys)
    path_str = "/".join(keys)

    if state == "none":
        return None

    if state == "partial":
        missing = [i for i, s in enumerate(samples) if not _has_nested(s, keys)]
        raise ValueError(
            f"Optional field '{path_str}' exists only in some samples. "
            f"Missing in indices: {missing}"
        )

    return [_get_nested(s, keys) for s in samples]


def _collect_required_images(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Collect and stack required camera images from:
    obs/images/{head,left_wrist,right_wrist}

    Rules:
    - every sample must have obs/images
    - every sample must contain exactly the keys defined in IMAGE_KEYS
    - each image tensor/array must be stackable across the batch
    """
    root_keys = ["obs", "images"]

    for i, sample in enumerate(samples):
        if not _has_nested(sample, root_keys):
            raise KeyError(f"Sample {i} is missing required field 'obs/images'")

    first_images = _get_nested(samples[0], root_keys)
    if not isinstance(first_images, dict):
        raise TypeError(
            f"Expected 'obs/images' to be dict, got {type(first_images).__name__}"
        )

    expected_keys = set(IMAGE_KEYS)
    first_keys = set(first_images.keys())

    if first_keys != expected_keys:
        raise ValueError(
            f"Sample 0 image keys mismatch. "
            f"Expected: {sorted(expected_keys)}, got: {sorted(first_keys)}"
        )

    for i, sample in enumerate(samples[1:], start=1):
        images = _get_nested(sample, root_keys)

        if not isinstance(images, dict):
            raise TypeError(
                f"Expected sample {i} 'obs/images' to be dict, got {type(images).__name__}"
            )

        keys_i = set(images.keys())
        if keys_i != expected_keys:
            raise ValueError(
                f"Inconsistent image keys across samples. "
                f"Expected: {sorted(expected_keys)}, "
                f"sample {i} keys: {sorted(keys_i)}"
            )

    batch_images: dict[str, torch.Tensor] = {}

    for key in IMAGE_KEYS:
        values: list[torch.Tensor] = []

        for i, sample in enumerate(samples):
            try:
                image_value = _get_nested(sample, ["obs", "images", key])
            except Exception as e:
                raise KeyError(
                    f"Sample {i} is missing required image field 'obs/images/{key}': {e}"
                ) from e

            values.append(_to_tensor(image_value))

        try:
            batch_images[key] = torch.stack(values, dim=0)
        except Exception as e:
            shapes = [tuple(v.shape) for v in values]
            raise RuntimeError(
                f"Failed to stack image field 'obs/images/{key}'. "
                f"Shapes: {shapes}. Reason: {e}"
            ) from e

    return batch_images


def assemble_bfm_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Assemble canonical BFM samples into a batch.

    Frozen required sample fields:
    - obs/state/arm
    - obs/state/hand
    - obs/state/full
    - obs/images/head
    - obs/images/left_wrist
    - obs/images/right_wrist
    - action/arm
    - action/hand
    - action/full
    - next_obs/state/arm
    - next_obs/state/hand
    - next_obs/state/full
    - done

    Optional sample fields:
    - reward
    - meta
    """
    if len(samples) == 0:
        raise ValueError("samples must not be empty")

    batch: dict[str, Any] = {
        "obs": {
            "state": {
                "arm": _stack_required(samples, ["obs", "state", "arm"]),
                "hand": _stack_required(samples, ["obs", "state", "hand"]),
                "full": _stack_required(samples, ["obs", "state", "full"]),
            },
            "images": _collect_required_images(samples),
        },
        "action": {
            "arm": _stack_required(samples, ["action", "arm"]),
            "hand": _stack_required(samples, ["action", "hand"]),
            "full": _stack_required(samples, ["action", "full"]),
        },
        "next_obs": {
            "state": {
                "arm": _stack_required(samples, ["next_obs", "state", "arm"]),
                "hand": _stack_required(samples, ["next_obs", "state", "hand"]),
                "full": _stack_required(samples, ["next_obs", "state", "full"]),
            }
        },
        "done": _stack_required(samples, ["done"]),
    }

    reward_batch = _stack_optional(samples, ["reward"])
    if reward_batch is not None:
        batch["reward"] = reward_batch

    meta_list = _collect_optional_list(samples, ["meta"])
    if meta_list is not None:
        batch["meta"] = meta_list

    return batch
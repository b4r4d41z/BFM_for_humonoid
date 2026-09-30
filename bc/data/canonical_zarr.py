"""Canonical, versioned Zarr storage shared by dataset importers."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bc.data.schema import (
    ACTION_FULL_DIM,
    ARM_JOINT_NAMES,
    HAND_VALUE_NAMES,
    STATE_FULL_DIM,
)
from bc.temporal import validate_episode_timestamps


SCHEMA_NAME = "bfm.canonical.transitions"
SCHEMA_VERSION = "1.0.0"
REQUIRED_ARRAYS = (
    "obs/state",
    "action",
    "next_obs/state",
    "done",
    "timestamp",
)


def _zarr():
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError(
            "Zarr support requires the optional 'zarr' package"
        ) from exc
    return zarr


def _image_compressor():
    try:
        from numcodecs import Blosc
    except ImportError as exc:
        raise RuntimeError(
            "Image compression requires the 'numcodecs' package"
        ) from exc

    return Blosc(
        cname="zstd",
        clevel=5,
        shuffle=Blosc.BITSHUFFLE,
    )


def _open_write_group(path: Path):
    zarr = _zarr()

    try:
        return zarr.open_group(
            str(path),
            mode="w",
            zarr_format=2,
        )
    except TypeError:
        return zarr.open_group(
            str(path),
            mode="w",
        )


def _put(
    group: Any,
    name: str,
    data: np.ndarray,
    *,
    compressor: Any = None,
    chunks: tuple[int, ...] | None = None,
) -> None:
    kwargs = {
        "data": data,
        "overwrite": True,
    }

    if compressor is not None:
        kwargs["compressor"] = compressor

    if chunks is not None:
        kwargs["chunks"] = chunks

    try:
        group.create_array(name, **kwargs)
    except AttributeError:
        group.create_dataset(
            name,
            shape=data.shape,
            **kwargs,
        )


def validate_canonical_zarr(
    store: str | os.PathLike[str] | Any,
) -> dict[str, Any]:
    root = (
        _zarr().open_group(str(store), mode="r")
        if isinstance(store, (str, os.PathLike))
        else store
    )

    missing = [
        name
        for name in REQUIRED_ARRAYS
        if name not in root
    ]

    if missing:
        raise ValueError(f"missing required arrays: {missing}")

    obs = np.asarray(root["obs/state"][:])
    action = np.asarray(root["action"][:])
    next_obs = np.asarray(root["next_obs/state"][:])
    done = np.asarray(root["done"][:])
    timestamp = np.asarray(root["timestamp"][:])

    t = len(timestamp)

    expected = {
        "obs/state": (t, STATE_FULL_DIM),
        "action": (t, ACTION_FULL_DIM),
        "next_obs/state": (t, STATE_FULL_DIM),
        "done": (t,),
        "timestamp": (t,),
    }

    for name, shape in expected.items():
        if tuple(root[name].shape) != shape:
            raise ValueError(
                f"{name} shape={tuple(root[name].shape)}, expected {shape}"
            )

    if t == 0:
        raise ValueError("canonical dataset has no transitions")

    for name, value in (
        ("obs/state", obs),
        ("action", action),
        ("next_obs/state", next_obs),
        ("timestamp", timestamp),
    ):
        if (
            not np.issubdtype(value.dtype, np.number)
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(
                f"{name} must contain only finite numeric values"
            )

    if not np.issubdtype(done.dtype, np.bool_):
        raise ValueError("done must have boolean dtype")

    validate_episode_timestamps(timestamp, done)

    if t > 1 and np.any(np.diff(timestamp) <= 0):
        raise ValueError(
            "timestamps must be strictly increasing"
        )

    for i in range(t - 1):
        if (
            not done[i]
            and not np.array_equal(
                next_obs[i],
                obs[i + 1],
            )
        ):
            raise ValueError(
                f"non-terminal next_obs[{i}] != obs[{i + 1}]"
            )

    attrs = dict(root.attrs)

    required_meta = (
        "schema_name",
        "schema_version",
        "obs_dim",
        "action_dim",
        "arm_dim",
        "hand_dim",
        "arm_joint_names",
        "hand_value_names",
        "source_bag",
        "camera_mapping",
        "temporal",
        "units",
    )

    absent = [
        key
        for key in required_meta
        if key not in attrs
    ]

    if absent:
        raise ValueError(
            f"missing required metadata: {absent}"
        )

    if (
        attrs["schema_name"] != SCHEMA_NAME
        or attrs["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError(
            "unsupported canonical schema name/version"
        )

    dims = tuple(
        int(attrs[key])
        for key in (
            "obs_dim",
            "action_dim",
            "arm_dim",
            "hand_dim",
        )
    )

    if (
        dims != (26, 26, 14, 12)
        or dims[2] + dims[3] != dims[0]
    ):
        raise ValueError(
            f"metadata dimensions violate 14+12 contract: {dims}"
        )

    if (
        list(attrs["arm_joint_names"]) != ARM_JOINT_NAMES
        or list(attrs["hand_value_names"]) != HAND_VALUE_NAMES
    ):
        raise ValueError(
            "metadata value names/order disagree with the canonical contract"
        )

    if "reward" in root:
        reward = np.asarray(root["reward"][:])

        if (
            reward.shape != (t,)
            or not np.issubdtype(reward.dtype, np.number)
            or not np.all(np.isfinite(reward))
        ):
            raise ValueError(
                "reward must be a finite numeric [T] array"
            )

    if "images" in root:
        for camera in root["images"].keys():
            arr = root[f"images/{camera}"]

            if arr.shape[0] != t:
                raise ValueError(
                    f"images/{camera} length={arr.shape[0]}, expected {t}"
                )

            if (
                arr.dtype != np.dtype("uint8")
                or len(arr.shape) != 4
                or arr.shape[-1] != 3
            ):
                raise ValueError(
                    f"images/{camera} must be decoded uint8 [T,H,W,3]"
                )

    mapped_cameras = sorted(
        dict(attrs["camera_mapping"]).keys()
    )

    stored_cameras = (
        sorted(root["images"].keys())
        if "images" in root
        else []
    )

    if mapped_cameras != stored_cameras:
        raise ValueError(
            f"camera_mapping keys {mapped_cameras} "
            f"disagree with stored images {stored_cameras}"
        )

    return {
        "transitions": t,
        "obs_dim": 26,
        "action_dim": 26,
        "cameras": (
            list(root["images"].keys())
            if "images" in root
            else []
        ),
    }


def write_canonical_zarr(
    output: str | os.PathLike[str],
    episode: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    overwrite: bool = False,
    compress_images: bool = False,
) -> dict[str, Any]:
    output = Path(output)

    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output}"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output.with_name(
        f".{output.name}.tmp-{uuid.uuid4().hex}"
    )

    try:
        root = _open_write_group(temporary)

        for name in REQUIRED_ARRAYS:
            _put(
                root,
                name,
                np.asarray(episode[name]),
            )

        if "reward" in episode:
            _put(
                root,
                "reward",
                np.asarray(episode["reward"]),
            )

        compressor = (
            _image_compressor()
            if compress_images
            else None
        )

        for camera, frames in episode.get("images", {}).items():
            frames = np.asarray(
                frames,
                dtype=np.uint8,
            )

            chunks = None

            if compress_images:
                chunks = (
                    1,
                    frames.shape[1],
                    frames.shape[2],
                    frames.shape[3],
                )

            _put(
                root,
                f"images/{camera}",
                frames,
                compressor=compressor,
                chunks=chunks,
            )

        root.attrs.update(
            dict(metadata)
        )

        summary = validate_canonical_zarr(
            root
        )

        if output.exists():
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()

        temporary.replace(output)

        return summary

    except Exception:
        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )
        raise
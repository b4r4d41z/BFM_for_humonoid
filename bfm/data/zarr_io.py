"""Validated episode stores, staged beside the destination before publication.

Reuse the existing canonical writer's staging approach, independent of its BC schema.
An overwrite keeps a recoverable backup until replacement succeeds.
"""
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import zarr
from numcodecs import Blosc

from .validation import validate_episode


def read_episode(path):
    root = zarr.open_group(str(path), mode="r")
    arrays = {}
    root.visititems(lambda name, obj: arrays.update({name: np.asarray(obj[:])}) if isinstance(obj, zarr.Array) else None)
    metadata = dict(root.attrs)
    validate_episode(arrays, metadata)
    return arrays, metadata


def write_episode(path, arrays, metadata, *, overwrite=False):
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    summary = validate_episode(arrays, metadata)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    backup = path.with_name(f".{path.name}.backup-{uuid.uuid4().hex}")
    try:
        root = zarr.open_group(str(tmp), mode="w")
        for name, a in arrays.items():
            root.create_dataset(name, data=a, chunks=(min(len(a), 2048),) + a.shape[1:], compressor=Blosc(cname="zstd", clevel=3))
        root.attrs.update(metadata)
        read_episode(tmp)
        if path.exists():
            os.replace(path, backup)
        try:
            os.replace(tmp, path)
        except Exception:
            if backup.exists():
                os.replace(backup, path)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return summary

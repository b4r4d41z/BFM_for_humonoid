from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .schema import IMAGE_KEYS, PATHS, get_image_path, split_action_vector, split_state_vector


@dataclass(frozen=True)
class HDF5FileInfo:
    path: str
    frames: int


def _progress(iterable, *, total: int, desc: str):
    last_pct = -1
    start = perf_counter()
    for i, item in enumerate(iterable, start=1):
        pct = int(i / max(1, total) * 100)
        if pct >= last_pct + 10 or i == total:
            print(f"[{desc}] progress={pct}% files={i}/{total} elapsed={perf_counter() - start:.2f}s", flush=True)
            last_pct = pct
        yield item


def _decode_cameras(cameras: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if cameras is None:
        return tuple(IMAGE_KEYS)
    if isinstance(cameras, str):
        items = [x.strip() for x in cameras.split(",") if x.strip()]
    else:
        items = [str(x).strip() for x in cameras if str(x).strip()]
    if not items:
        return tuple(IMAGE_KEYS)
    unknown = sorted(set(items) - set(IMAGE_KEYS))
    if unknown:
        raise ValueError(f"Unknown camera(s): {unknown}. Valid cameras: {list(IMAGE_KEYS)}")
    return tuple(items)


class HDF5StreamingDataset(Dataset):
    """Lightweight streaming HDF5 dataset for offline BC training.

    The constructor opens each HDF5 file only briefly to validate shapes and build
    an index of ``(file_id, timestep)`` entries. Persistent ``h5py.File`` handles
    are created lazily inside ``__getitem__`` and cached per Dataset instance,
    which means each DataLoader worker owns its own handles.
    """

    def __init__(
        self,
        hdf5_paths: list[str | Path],
        *,
        use_images: bool = True,
        cameras: str | list[str] | tuple[str, ...] | None = None,
        image_size: int | None = None,
        frame_stride: int = 1,
        seq_len: int = 1,
        use_text: bool = False,
        log_prefix: str = "[BFM streaming dataset]",
    ) -> None:
        if frame_stride <= 0:
            raise ValueError(f"frame_stride must be positive, got {frame_stride}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        self.hdf5_paths = [str(Path(p).expanduser().resolve()) for p in hdf5_paths]
        self.use_images = bool(use_images)
        self.cameras = _decode_cameras(cameras)
        self.image_size = int(image_size) if image_size is not None else None
        self.frame_stride = int(frame_stride)
        self.seq_len = int(seq_len)
        self.use_text = bool(use_text)
        self.log_prefix = log_prefix

        self.file_infos: list[HDF5FileInfo] = []
        self.index: list[tuple[int, int]] = []
        self._handles: dict[int, h5py.File] = {}
        self._meta_cache: dict[int, dict[str, Any]] = {}
        self._dummy_image_cache: torch.Tensor | None = None

        start = perf_counter()
        print(
            f"{self.log_prefix} building lightweight index: files={len(self.hdf5_paths)} "
            f"use_images={self.use_images} cameras={','.join(self.cameras)} "
            f"image_size={self.image_size} frame_stride={self.frame_stride} seq_len={self.seq_len}",
            flush=True,
        )
        for file_id, path in enumerate(_progress(self.hdf5_paths, total=len(self.hdf5_paths), desc="scan h5")):
            with h5py.File(path, "r") as f:
                frames = self._validate_file(f, path)
            self.file_infos.append(HDF5FileInfo(path=path, frames=frames))
            max_start = frames - self.seq_len
            if max_start < 0:
                continue
            for t in range(0, max_start + 1, self.frame_stride):
                self.index.append((file_id, t))
        elapsed = perf_counter() - start
        print(
            f"{self.log_prefix} index ready: files={len(self.file_infos)} samples={len(self.index)} "
            f"elapsed={elapsed:.2f}s (no full dataset preload)",
            flush=True,
        )
        if not self.index:
            raise ValueError("Streaming dataset index is empty. Check file lengths, frame_stride, and seq_len.")

    def __len__(self) -> int:
        return len(self.index)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_meta_cache"] = {}
        state["_dummy_image_cache"] = None
        return state

    def _validate_file(self, f: h5py.File, path: str) -> int:
        required = [
            PATHS.obs_state,
            PATHS.next_obs_state,
            PATHS.act_joint_target,
            PATHS.act_hand_target,
            PATHS.act_action,
            PATHS.done,
        ]
        if self.use_images:
            required.extend(get_image_path(cam) for cam in self.cameras)
        for key in required:
            if key not in f:
                raise KeyError(f"{path}: missing required HDF5 entry {key}")
        frames = int(f[PATHS.done].shape[0])
        for key in required:
            if hasattr(f[key], "shape") and int(f[key].shape[0]) != frames:
                raise ValueError(f"{path}: length mismatch for {key}: expected {frames}, got {f[key].shape[0]}")
        return frames

    def _get_h5(self, file_id: int) -> h5py.File:
        handle = self._handles.get(file_id)
        if handle is None:
            handle = h5py.File(self.file_infos[file_id].path, "r")
            self._handles[file_id] = handle
        return handle

    @staticmethod
    def _decode_scalar(x: Any) -> Any:
        if isinstance(x, bytes):
            return x.decode("utf-8")
        if isinstance(x, np.ndarray) and x.shape == ():
            x = x.item()
            return x.decode("utf-8") if isinstance(x, bytes) else x
        return x

    def _load_meta_once(self, file_id: int, f: h5py.File) -> dict[str, Any]:
        if file_id in self._meta_cache:
            return self._meta_cache[file_id]
        meta: dict[str, Any] = {}
        for name, path in (("obs_dim", PATHS.meta_obs_dim), ("act_dim", PATHS.meta_act_dim)):
            if path in f:
                meta[name] = int(np.asarray(f[path][()]).reshape(-1)[0])
        if PATHS.meta_bag_name in f:
            meta["bag_name"] = str(self._decode_scalar(f[PATHS.meta_bag_name][()]))
        if self.use_text and PATHS.meta_instruction in f:
            meta["instruction"] = str(self._decode_scalar(f[PATHS.meta_instruction][()]))
        self._meta_cache[file_id] = meta
        return meta

    def _resize_image(self, img: torch.Tensor) -> torch.Tensor:
        if self.image_size is None:
            return img
        if img.ndim != 3:
            raise ValueError(f"Expected image [H,W,C] or [C,H,W], got {tuple(img.shape)}")
        channel_last = img.shape[-1] in (1, 3, 4)
        x = img.permute(2, 0, 1) if channel_last else img
        dtype = img.dtype
        x = F.interpolate(x.float().unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
        if dtype == torch.uint8:
            x = x.clamp(0, 255).to(torch.uint8)
        return x.permute(1, 2, 0).contiguous() if channel_last else x.contiguous()

    def _dummy_image(self) -> torch.Tensor:
        if self._dummy_image_cache is None:
            size = self.image_size or 224
            self._dummy_image_cache = torch.zeros((size, size, 3), dtype=torch.uint8)
        return self._dummy_image_cache.clone()

    def _read_one(self, file_id: int, t: int) -> dict[str, Any]:
        f = self._get_h5(file_id)
        obs_full = np.asarray(f[PATHS.obs_state][t], dtype=np.float32)
        next_obs_full = np.asarray(f[PATHS.next_obs_state][t], dtype=np.float32)
        act_full = np.asarray(f[PATHS.act_action][t], dtype=np.float32)
        obs_state = split_state_vector(obs_full)
        next_obs_state = split_state_vector(next_obs_full)
        act_state = split_action_vector(act_full)

        sample: dict[str, Any] = {
            "obs": {"state": {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in obs_state.items()}},
            "action": {
                "arm": torch.from_numpy(np.asarray(f[PATHS.act_joint_target][t], dtype=np.float32)),
                "hand": torch.from_numpy(np.asarray(f[PATHS.act_hand_target][t], dtype=np.float32)),
                "full": torch.from_numpy(np.asarray(act_state["full"], dtype=np.float32)),
            },
            "next_obs": {"state": {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in next_obs_state.items()}},
            "done": torch.tensor(bool(np.asarray(f[PATHS.done][t])), dtype=torch.bool),
        }
        if PATHS.reward in f:
            sample["reward"] = torch.tensor(float(np.asarray(f[PATHS.reward][t])), dtype=torch.float32)

        images: dict[str, torch.Tensor] = {}
        if self.use_images:
            read_images: dict[str, torch.Tensor] = {}
            for cam in self.cameras:
                img = torch.from_numpy(np.asarray(f[get_image_path(cam)][t]))
                read_images[cam] = self._resize_image(img)
            fallback = next(iter(read_images.values())).new_zeros(next(iter(read_images.values())).shape) if read_images else self._dummy_image()
            for cam in IMAGE_KEYS:
                images[cam] = read_images.get(cam, fallback.clone())
        else:
            for cam in IMAGE_KEYS:
                images[cam] = self._dummy_image()
        sample["obs"]["images"] = images

        meta = self._load_meta_once(file_id, f)
        if meta:
            sample["meta"] = dict(meta)
        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_id, start = self.index[idx]
        if self.seq_len == 1:
            return self._read_one(file_id, start)
        items = [self._read_one(file_id, start + offset) for offset in range(self.seq_len)]
        return _stack_sequence(items)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        self.close()


def _stack_sequence(items: list[dict[str, Any]]) -> dict[str, Any]:
    def stack(vals: list[Any]) -> Any:
        first = vals[0]
        if isinstance(first, dict):
            return {k: stack([v[k] for v in vals]) for k in first.keys()}
        if isinstance(first, torch.Tensor):
            return torch.stack(vals, dim=0)
        return vals

    return stack(items)

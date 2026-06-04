from __future__ import annotations

from functools import singledispatch
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import torch

from ..data.schema import (
    ACTION_ARM_DIM,
    ACTION_FULL_DIM,
    ACTION_HAND_DIM,
    IMAGE_KEYS,
    STATE_ARM_DIM,
    STATE_FULL_DIM,
    STATE_HAND_DIM,
    fill_and_validate_contract_metadata,
)
from ..data.stream_loader import HDF5DataStreamLoader


@singledispatch
def _to_torch(x: Any) -> torch.Tensor:
    return torch.as_tensor(x)


@_to_torch.register(torch.Tensor)
def _(x: torch.Tensor) -> torch.Tensor:
    return x


@_to_torch.register(np.ndarray)
def _(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x)


def _clone_if_tensor(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, dict):
        return {k: _clone_if_tensor(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clone_if_tensor(v) for v in x]
    return x


def _nested_get(d: dict[str, Any], keys: list[str]) -> Any:
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


def _nested_has(d: dict[str, Any], keys: list[str]) -> bool:
    try:
        _nested_get(d, keys)
        return True
    except Exception:
        return False


def _nested_apply(x: Any, fn) -> Any:
    if isinstance(x, dict):
        return {k: _nested_apply(v, fn) for k, v in x.items()}
    return fn(x)


def _nested_index_time(x: Any, t: int) -> Any:
    if isinstance(x, dict):
        return {k: _nested_index_time(v, t) for k, v in x.items()}
    return x[t]


def _nested_slice_time(x: Any, start: int, end: int) -> Any:
    if isinstance(x, dict):
        return {k: _nested_slice_time(v, start, end) for k, v in x.items()}
    return x[start:end]


def _nested_stack(items: list[Any], dim: int = 0) -> Any:
    if len(items) == 0:
        raise ValueError("Cannot stack empty list")

    first = items[0]

    if isinstance(first, dict):
        keys = set(first.keys())
        for i, item in enumerate(items[1:], start=1):
            if not isinstance(item, dict):
                raise TypeError(
                    f"Expected dict at stacked item {i}, got {type(item).__name__}"
                )
            if set(item.keys()) != keys:
                raise ValueError(
                    f"Inconsistent dict keys while stacking. "
                    f"Expected {sorted(keys)}, got {sorted(item.keys())} at item {i}"
                )

        return {k: _nested_stack([item[k] for item in items], dim=dim) for k in first.keys()}

    tensors = [_to_torch(item) for item in items]
    return torch.stack(tensors, dim=dim)


def _nested_to_device(x: Any, device: torch.device | str | None) -> Any:
    if device is None:
        return x

    if isinstance(x, dict):
        return {k: _nested_to_device(v, device) for k, v in x.items()}

    if isinstance(x, list):
        return [_nested_to_device(v, device) for v in x]

    if isinstance(x, torch.Tensor):
        return x.to(device)

    return x


def _infer_time_length(x: Any) -> int:
    if isinstance(x, dict):
        if len(x) == 0:
            raise ValueError("Cannot infer time length from empty dict")
        first_key = next(iter(x.keys()))
        return _infer_time_length(x[first_key])

    if not isinstance(x, torch.Tensor):
        x = _to_torch(x)

    if x.ndim == 0:
        raise ValueError("Expected time-major tensor, got scalar")

    return int(x.shape[0])


def _check_last_dim(x: Any, expected_dim: int, name: str) -> None:
    x = _to_torch(x)
    if x.ndim == 0:
        raise ValueError(f"{name}: scalar found, expected last dim {expected_dim}")
    if int(x.shape[-1]) != expected_dim:
        raise ValueError(
            f"{name}: expected last dim {expected_dim}, got {int(x.shape[-1])}"
        )


def _check_time_length(x: Any, expected_len: int, name: str) -> None:
    actual_len = _infer_time_length(x)
    if actual_len != expected_len:
        raise ValueError(
            f"{name}: expected first dim {expected_len}, got {actual_len}"
        )


def _check_image_tensor(x: Any, name: str) -> None:
    x = _to_torch(x)
    if x.ndim != 4:
        raise ValueError(f"{name}: expected image tensor [T, H, W, C], got shape {tuple(x.shape)}")

    channels = int(x.shape[-1])
    if channels not in (1, 3, 4):
        raise ValueError(
            f"{name}: expected channel-last image with 1/3/4 channels, got {channels}"
        )


def _copy_meta(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if meta is None:
        return None
    return _clone_if_tensor(meta)


def _canonical_transition_from_episode(
    episode: dict[str, Any],
    t: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "obs": _nested_index_time(episode["obs"], t),
        "action": _nested_index_time(episode["action"], t),
        "next_obs": _nested_index_time(episode["next_obs"], t),
        "done": _nested_index_time(episode["done"], t),
    }

    if "reward" in episode:
        out["reward"] = _nested_index_time(episode["reward"], t)

    if "meta" in episode:
        out["meta"] = _copy_meta(episode["meta"])

    return out


def _canonical_sequence_from_episode(
    episode: dict[str, Any],
    start: int,
    end: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "obs": _nested_slice_time(episode["obs"], start, end),
        "action": _nested_slice_time(episode["action"], start, end),
        "next_obs": _nested_slice_time(episode["next_obs"], start, end),
        "done": _nested_slice_time(episode["done"], start, end),
    }

    if "reward" in episode:
        out["reward"] = _nested_slice_time(episode["reward"], start, end)

    if "meta" in episode:
        out["meta"] = _copy_meta(episode["meta"])

    return out


class OfflineTrajectoryBuffer:
    """
    Offline episode-based buffer for demonstration data.

    Storage contract for each episode:
    {
        "obs": {
            "state": {
                "arm":  [T, 14],
                "hand": [T, 12],
                "full": [T, 26],
            },
            "images": {
                "head":        [T, H, W, C],
                "left_wrist":  [T, H, W, C],
                "right_wrist": [T, H, W, C],
            }  # optional
        },
        "action": {
            "arm":  [T, 14],
            "hand": [T, 12],
            "full": [T, 26],
        },
        "next_obs": {
            "state": {
                "arm":  [T, 14],
                "hand": [T, 12],
                "full": [T, 26],
            }
        },
        "done":   [T],
        "reward": [T],      # optional
        "meta":   {...},    # optional, per-episode
    }
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> None:
        self.storage_device = torch.device(device)
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(seed)

        self.episodes: list[dict[str, Any]] = []
        self.episode_lengths: list[int] = []
        self.transition_index: list[tuple[int, int]] = []
        self.sequence_index_cache: dict[int, list[tuple[int, int]]] = {}

        self.has_images: bool | None = None
        self.has_reward: bool | None = None
        self.has_meta: bool | None = None

    def __len__(self) -> int:
        return len(self.transition_index)

    def num_episodes(self) -> int:
        return len(self.episodes)

    def empty(self) -> bool:
        return len(self.episodes) == 0

    def total_num_steps(self) -> int:
        return sum(self.episode_lengths)

    def clear(self) -> None:
        self.episodes.clear()
        self.episode_lengths.clear()
        self.transition_index.clear()
        self.sequence_index_cache.clear()

        self.has_images = None
        self.has_reward = None
        self.has_meta = None

    def _validate_and_prepare_episode(self, episode: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(episode, dict):
            raise TypeError(f"Episode must be dict, got {type(episode).__name__}")

        required_paths = [
            ["obs", "state", "arm"],
            ["obs", "state", "hand"],
            ["obs", "state", "full"],
            ["action", "arm"],
            ["action", "hand"],
            ["action", "full"],
            ["next_obs", "state", "arm"],
            ["next_obs", "state", "hand"],
            ["next_obs", "state", "full"],
            ["done"],
        ]

        for path in required_paths:
            if not _nested_has(episode, path):
                raise KeyError(f"Episode missing required path: {'/'.join(path)}")

        prepared: dict[str, Any] = {
            "obs": {
                "state": {
                    "arm": _to_torch(_nested_get(episode, ["obs", "state", "arm"])).to(self.storage_device),
                    "hand": _to_torch(_nested_get(episode, ["obs", "state", "hand"])).to(self.storage_device),
                    "full": _to_torch(_nested_get(episode, ["obs", "state", "full"])).to(self.storage_device),
                }
            },
            "action": {
                "arm": _to_torch(_nested_get(episode, ["action", "arm"])).to(self.storage_device),
                "hand": _to_torch(_nested_get(episode, ["action", "hand"])).to(self.storage_device),
                "full": _to_torch(_nested_get(episode, ["action", "full"])).to(self.storage_device),
            },
            "next_obs": {
                "state": {
                    "arm": _to_torch(_nested_get(episode, ["next_obs", "state", "arm"])).to(self.storage_device),
                    "hand": _to_torch(_nested_get(episode, ["next_obs", "state", "hand"])).to(self.storage_device),
                    "full": _to_torch(_nested_get(episode, ["next_obs", "state", "full"])).to(self.storage_device),
                }
            },
            "done": _to_torch(_nested_get(episode, ["done"])).to(self.storage_device),
        }

        episode_len = _infer_time_length(prepared["obs"]["state"]["full"])

        _check_last_dim(prepared["obs"]["state"]["arm"], STATE_ARM_DIM, "obs/state/arm")
        _check_last_dim(prepared["obs"]["state"]["hand"], STATE_HAND_DIM, "obs/state/hand")
        _check_last_dim(prepared["obs"]["state"]["full"], STATE_FULL_DIM, "obs/state/full")

        _check_last_dim(prepared["action"]["arm"], ACTION_ARM_DIM, "action/arm")
        _check_last_dim(prepared["action"]["hand"], ACTION_HAND_DIM, "action/hand")
        _check_last_dim(prepared["action"]["full"], ACTION_FULL_DIM, "action/full")

        _check_last_dim(prepared["next_obs"]["state"]["arm"], STATE_ARM_DIM, "next_obs/state/arm")
        _check_last_dim(prepared["next_obs"]["state"]["hand"], STATE_HAND_DIM, "next_obs/state/hand")
        _check_last_dim(prepared["next_obs"]["state"]["full"], STATE_FULL_DIM, "next_obs/state/full")

        _check_time_length(prepared["obs"]["state"]["arm"], episode_len, "obs/state/arm")
        _check_time_length(prepared["obs"]["state"]["hand"], episode_len, "obs/state/hand")
        _check_time_length(prepared["obs"]["state"]["full"], episode_len, "obs/state/full")

        _check_time_length(prepared["action"]["arm"], episode_len, "action/arm")
        _check_time_length(prepared["action"]["hand"], episode_len, "action/hand")
        _check_time_length(prepared["action"]["full"], episode_len, "action/full")

        _check_time_length(prepared["next_obs"]["state"]["arm"], episode_len, "next_obs/state/arm")
        _check_time_length(prepared["next_obs"]["state"]["hand"], episode_len, "next_obs/state/hand")
        _check_time_length(prepared["next_obs"]["state"]["full"], episode_len, "next_obs/state/full")

        _check_time_length(prepared["done"], episode_len, "done")

        has_images = _nested_has(episode, ["obs", "images"])
        if has_images:
            images_root = _nested_get(episode, ["obs", "images"])
            if not isinstance(images_root, dict):
                raise TypeError(
                    f"obs/images must be dict, got {type(images_root).__name__}"
                )

            actual_image_keys = set(images_root.keys())
            expected_image_keys = set(IMAGE_KEYS)
            if actual_image_keys != expected_image_keys:
                raise ValueError(
                    f"obs/images keys mismatch. "
                    f"Expected {sorted(expected_image_keys)}, got {sorted(actual_image_keys)}"
                )

            prepared["obs"]["images"] = {}
            for key in IMAGE_KEYS:
                image_tensor = _to_torch(images_root[key]).to(self.storage_device)
                _check_time_length(image_tensor, episode_len, f"obs/images/{key}")
                _check_image_tensor(image_tensor, f"obs/images/{key}")
                prepared["obs"]["images"][key] = image_tensor

        has_reward = _nested_has(episode, ["reward"])
        if has_reward:
            reward_tensor = _to_torch(_nested_get(episode, ["reward"])).to(self.storage_device)
            _check_time_length(reward_tensor, episode_len, "reward")
            prepared["reward"] = reward_tensor

        has_meta = _nested_has(episode, ["meta"])
        if has_meta:
            meta_obj = _nested_get(episode, ["meta"])
            if not isinstance(meta_obj, dict):
                raise TypeError(f"meta must be dict, got {type(meta_obj).__name__}")
            contract_meta, _ = fill_and_validate_contract_metadata(
                meta_obj, context="OfflineTrajectoryBuffer episode/meta", warn=True
            )
            prepared["meta"] = _copy_meta(contract_meta)

        if self.has_images is None:
            self.has_images = has_images
        elif self.has_images != has_images:
            raise ValueError(
                f"Inconsistent image presence across episodes. "
                f"Buffer expects has_images={self.has_images}, got {has_images}"
            )

        if self.has_reward is None:
            self.has_reward = has_reward
        elif self.has_reward != has_reward:
            raise ValueError(
                f"Inconsistent reward presence across episodes. "
                f"Buffer expects has_reward={self.has_reward}, got {has_reward}"
            )

        if self.has_meta is None:
            self.has_meta = has_meta
        elif self.has_meta != has_meta:
            raise ValueError(
                f"Inconsistent meta presence across episodes. "
                f"Buffer expects has_meta={self.has_meta}, got {has_meta}"
            )

        return prepared

    def add_episode(self, episode: dict[str, Any]) -> None:
        prepared = self._validate_and_prepare_episode(episode)

        ep_id = len(self.episodes)
        ep_len = _infer_time_length(prepared["obs"]["state"]["full"])

        self.episodes.append(prepared)
        self.episode_lengths.append(ep_len)

        for t in range(ep_len):
            self.transition_index.append((ep_id, t))

        self.sequence_index_cache.clear()

    def extend(self, episodes: Iterable[dict[str, Any]]) -> None:
        for episode in episodes:
            self.add_episode(episode)

    def _get_sequence_index(self, seq_len: int) -> list[tuple[int, int]]:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")

        if seq_len in self.sequence_index_cache:
            return self.sequence_index_cache[seq_len]

        index: list[tuple[int, int]] = []

        for ep_id, ep_len in enumerate(self.episode_lengths):
            if ep_len < seq_len:
                continue

            max_start = ep_len - seq_len
            for start in range(max_start + 1):
                index.append((ep_id, start))

        self.sequence_index_cache[seq_len] = index
        return index

    def sample_transitions(
        self,
        batch_size: int,
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        if self.empty():
            raise RuntimeError("Cannot sample from empty buffer")

        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        n = len(self.transition_index)
        idx_tensor = torch.randint(
            low=0,
            high=n,
            size=(batch_size,),
            generator=self.rng,
        )

        transitions = []
        for idx in idx_tensor.tolist():
            ep_id, t = self.transition_index[idx]
            transitions.append(_canonical_transition_from_episode(self.episodes[ep_id], t))

        batch: dict[str, Any] = {
            "obs": _nested_stack([x["obs"] for x in transitions], dim=0),
            "action": _nested_stack([x["action"] for x in transitions], dim=0),
            "next_obs": _nested_stack([x["next_obs"] for x in transitions], dim=0),
            "done": _nested_stack([x["done"] for x in transitions], dim=0),
        }

        if self.has_reward:
            batch["reward"] = _nested_stack([x["reward"] for x in transitions], dim=0)

        if self.has_meta:
            batch["meta"] = [_copy_meta(x["meta"]) for x in transitions]

        return _nested_to_device(batch, device)

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int,
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        if self.empty():
            raise RuntimeError("Cannot sample from empty buffer")

        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        sequence_index = self._get_sequence_index(seq_len)
        if len(sequence_index) == 0:
            raise RuntimeError(
                f"No valid sequences of length {seq_len} in buffer"
            )

        idx_tensor = torch.randint(
            low=0,
            high=len(sequence_index),
            size=(batch_size,),
            generator=self.rng,
        )

        sequences = []
        for idx in idx_tensor.tolist():
            ep_id, start = sequence_index[idx]
            end = start + seq_len
            sequences.append(
                _canonical_sequence_from_episode(self.episodes[ep_id], start, end)
            )

        batch: dict[str, Any] = {
            "obs": _nested_stack([x["obs"] for x in sequences], dim=0),
            "action": _nested_stack([x["action"] for x in sequences], dim=0),
            "next_obs": _nested_stack([x["next_obs"] for x in sequences], dim=0),
            "done": _nested_stack([x["done"] for x in sequences], dim=0),
        }

        if self.has_reward:
            batch["reward"] = _nested_stack([x["reward"] for x in sequences], dim=0)

        if self.has_meta:
            batch["meta"] = [_copy_meta(x["meta"]) for x in sequences]

        return _nested_to_device(batch, device)

    def get_episode(self, episode_id: int) -> dict[str, Any]:
        if episode_id < 0 or episode_id >= len(self.episodes):
            raise IndexError(
                f"episode_id out of range: {episode_id}, num_episodes={len(self.episodes)}"
            )
        return _clone_if_tensor(self.episodes[episode_id])

    def summary(self) -> dict[str, Any]:
        return {
            "num_episodes": self.num_episodes(),
            "total_num_steps": self.total_num_steps(),
            "has_images": self.has_images,
            "has_reward": self.has_reward,
            "has_meta": self.has_meta,
            "min_episode_length": min(self.episode_lengths) if self.episode_lengths else 0,
            "max_episode_length": max(self.episode_lengths) if self.episode_lengths else 0,
        }

    @staticmethod
    def build_episode_from_loader_dataset(
        dataset: HDF5DataStreamLoader,
    ) -> dict[str, Any]:
        dataset_len = len(dataset)
        if dataset_len <= 0:
            raise ValueError("Cannot build episode from empty dataset")

        first = dataset[0]

        has_images = _nested_has(first, ["obs", "images"])
        has_reward = "reward" in first
        has_meta = "meta" in first

        obs_arm: list[torch.Tensor] = []
        obs_hand: list[torch.Tensor] = []
        obs_full: list[torch.Tensor] = []

        act_arm: list[torch.Tensor] = []
        act_hand: list[torch.Tensor] = []
        act_full: list[torch.Tensor] = []

        next_obs_arm: list[torch.Tensor] = []
        next_obs_hand: list[torch.Tensor] = []
        next_obs_full: list[torch.Tensor] = []

        done_list: list[torch.Tensor] = []
        reward_list: list[torch.Tensor] = []

        image_lists: dict[str, list[torch.Tensor]] = {k: [] for k in IMAGE_KEYS}

        meta_obj: dict[str, Any] | None = _copy_meta(first["meta"]) if has_meta else None

        for i in range(dataset_len):
            sample = dataset[i]

            obs_arm.append(_to_torch(_nested_get(sample, ["obs", "state", "arm"])))
            obs_hand.append(_to_torch(_nested_get(sample, ["obs", "state", "hand"])))
            obs_full.append(_to_torch(_nested_get(sample, ["obs", "state", "full"])))

            act_arm.append(_to_torch(_nested_get(sample, ["action", "arm"])))
            act_hand.append(_to_torch(_nested_get(sample, ["action", "hand"])))
            act_full.append(_to_torch(_nested_get(sample, ["action", "full"])))

            next_obs_arm.append(_to_torch(_nested_get(sample, ["next_obs", "state", "arm"])))
            next_obs_hand.append(_to_torch(_nested_get(sample, ["next_obs", "state", "hand"])))
            next_obs_full.append(_to_torch(_nested_get(sample, ["next_obs", "state", "full"])))

            done_list.append(_to_torch(_nested_get(sample, ["done"])))

            if has_reward:
                if "reward" not in sample:
                    raise ValueError(
                        f"Inconsistent reward presence inside dataset at step {i}"
                    )
                reward_list.append(_to_torch(sample["reward"]))

            if has_images:
                if not _nested_has(sample, ["obs", "images"]):
                    raise ValueError(
                        f"Inconsistent image presence inside dataset at step {i}"
                    )
                for key in IMAGE_KEYS:
                    image_lists[key].append(_to_torch(_nested_get(sample, ["obs", "images", key])))

            if has_meta:
                if "meta" not in sample:
                    raise ValueError(
                        f"Inconsistent meta presence inside dataset at step {i}"
                    )

        episode: dict[str, Any] = {
            "obs": {
                "state": {
                    "arm": torch.stack(obs_arm, dim=0),
                    "hand": torch.stack(obs_hand, dim=0),
                    "full": torch.stack(obs_full, dim=0),
                }
            },
            "action": {
                "arm": torch.stack(act_arm, dim=0),
                "hand": torch.stack(act_hand, dim=0),
                "full": torch.stack(act_full, dim=0),
            },
            "next_obs": {
                "state": {
                    "arm": torch.stack(next_obs_arm, dim=0),
                    "hand": torch.stack(next_obs_hand, dim=0),
                    "full": torch.stack(next_obs_full, dim=0),
                }
            },
            "done": torch.stack(done_list, dim=0),
        }

        if has_reward:
            episode["reward"] = torch.stack(reward_list, dim=0)

        if has_images:
            episode["obs"]["images"] = {
                key: torch.stack(image_lists[key], dim=0) for key in IMAGE_KEYS
            }

        if has_meta and meta_obj is not None:
            episode["meta"] = meta_obj

        return episode

    def add_hdf5_file(
        self,
        hdf5_path: str | Path,
        use_images: bool = True,
        use_text: bool = True,
        log_prefix: str | None = None,
    ) -> int:
        start_time = perf_counter()
        if log_prefix is not None:
            print(f"{log_prefix} opening HDF5: {hdf5_path}", flush=True)

        dataset = HDF5DataStreamLoader(
            hdf5_path=hdf5_path,
            use_images=use_images,
            use_text=use_text,
        )
        open_elapsed = perf_counter() - start_time
        dataset_len = len(dataset)
        if log_prefix is not None:
            print(
                f"{log_prefix} opened/validated HDF5: {hdf5_path} "
                f"frames={dataset_len} elapsed={open_elapsed:.2f}s; reading into buffer...",
                flush=True,
            )

        read_start = perf_counter()
        try:
            episode = self.build_episode_from_loader_dataset(dataset)
        finally:
            dataset.close()
        self.add_episode(episode)
        total_elapsed = perf_counter() - start_time
        read_elapsed = perf_counter() - read_start

        if log_prefix is not None:
            print(
                f"{log_prefix} loaded HDF5: {hdf5_path} "
                f"frames={dataset_len} read_elapsed={read_elapsed:.2f}s total_elapsed={total_elapsed:.2f}s "
                f"episodes_loaded={self.num_episodes()} total_frames_loaded={self.total_num_steps()}",
                flush=True,
            )

        return dataset_len

    @classmethod
    def from_hdf5_files(
        cls,
        hdf5_paths: Iterable[str | Path],
        use_images: bool = True,
        use_text: bool = True,
        device: str | torch.device = "cpu",
        seed: int = 42,
        log_prefix: str | None = None,
    ) -> "OfflineTrajectoryBuffer":
        buffer = cls(device=device, seed=seed)
        paths = list(hdf5_paths)

        if log_prefix is not None:
            print(
                f"{log_prefix} loading {len(paths)} HDF5 file(s) into OfflineTrajectoryBuffer "
                f"on device={buffer.storage_device}",
                flush=True,
            )

        for i, path in enumerate(paths, start=1):
            if log_prefix is not None:
                pct = int(i / max(1, len(paths)) * 100)
                print(f"{log_prefix} eager load progress={pct}% files={i}/{len(paths)}", flush=True)
            file_prefix = f"{log_prefix} [{i}/{len(paths)}]" if log_prefix is not None else None
            buffer.add_hdf5_file(
                hdf5_path=path,
                use_images=use_images,
                use_text=use_text,
                log_prefix=file_prefix,
            )

        if log_prefix is not None:
            print(
                f"{log_prefix} finished buffer load: episodes={buffer.num_episodes()} "
                f"frames={buffer.total_num_steps()} transitions={len(buffer)}",
                flush=True,
            )

        return buffer

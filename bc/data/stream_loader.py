from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import (
    ACTION_ARM_DIM,
    ACTION_FULL_DIM,
    ACTION_HAND_DIM,
    IMAGE_KEYS,
    PATHS,
    fill_and_validate_contract_metadata,
    STATE_FULL_DIM,
    get_image_path,
    split_action_vector,
    split_state_vector,
)


class HDF5DataStreamLoader(Dataset):
    def __init__(
        self,
        hdf5_path: str | Path,
        use_images: bool = True,
        use_text: bool = True,
    ) -> None:
        """
        Canonical loader output format:

        {
            "obs": {
                "state": {
                    "arm": ...,
                    "hand": ...,
                    "full": ...,
                },
                "images": {
                    "head": ...,
                    "left_wrist": ...,
                    "right_wrist": ...,
                },   # present only when use_images=True
            },
            "action": {
                "arm": ...,
                "hand": ...,
                "full": ...,
            },
            "next_obs": {
                "state": {
                    "arm": ...,
                    "hand": ...,
                    "full": ...,
                }
            },
            "done": ...,
            "reward": ...,    # present only if raw HDF5 contains /reward
            "meta": {         # present if at least one meta field is available
                "instruction": ...,
                "bag_name": ...,
                "joint_names": ...,
                "arm_joint_names": ...,
                "hand_value_names": ...,
                "obs_dim": ...,
                "act_dim": ...,
                "action_dim": ...,
                "action_type": ...,
                "gripper_mode": ...,
                "state_layout": ...,
                "action_layout": ...,
                "hand_open_prototype_6": ...,
                "hand_closed_prototype_6": ...,
                "state_definition": ...,
            }
        }

        Notes:
        - `use_text` is kept only for backward compatibility of the constructor.
          It now controls whether meta["instruction"] is populated or replaced with "".
        - No old internal names are used anymore.
        """
        self.hdf5_path = str(hdf5_path)
        self.use_images = use_images
        self.use_text = use_text

        self._h5: h5py.File | None = None
        self._meta_cache: dict[str, Any] | None = None

        with h5py.File(self.hdf5_path, "r") as f:
            self._validate_structure(f)
            raw_meta = self._read_contract_meta_from_file(f)
            fill_and_validate_contract_metadata(
                raw_meta, context=f"{self.hdf5_path}/meta", warn=True
            )
            self.length = int(f[PATHS.done].shape[0])

    def __len__(self) -> int:
        return self.length

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def _validate_structure(self, f: h5py.File) -> None:
        """
        Validate only the raw datasets that are actually required for the
        frozen internal Python contract.
        """
        required = [
            PATHS.obs_state,
            PATHS.next_obs_state,
            PATHS.act_joint_target,
            PATHS.act_hand_target,
            PATHS.act_action,
            PATHS.done,
        ]

        if self.use_images:
            required.extend(
                [
                    PATHS.images_head,
                    PATHS.images_left_wrist,
                    PATHS.images_right_wrist,
                ]
            )

        for key in required:
            if key not in f:
                raise KeyError(f"Missing required HDF5 entry: {key}")

        n = int(f[PATHS.done].shape[0])

        expected_same_len = [
            PATHS.obs_state,
            PATHS.next_obs_state,
            PATHS.act_joint_target,
            PATHS.act_hand_target,
            PATHS.act_action,
        ]

        if PATHS.reward in f:
            expected_same_len.append(PATHS.reward)

        if self.use_images:
            expected_same_len.extend(
                [
                    PATHS.images_head,
                    PATHS.images_left_wrist,
                    PATHS.images_right_wrist,
                ]
            )

        for key in expected_same_len:
            if int(f[key].shape[0]) != n:
                raise ValueError(
                    f"Length mismatch for {key}: expected {n}, got {f[key].shape[0]}"
                )

        if f[PATHS.obs_state].ndim != 2 or f[PATHS.obs_state].shape[1] != STATE_FULL_DIM:
            raise ValueError(
                f"{PATHS.obs_state} must have shape [N, {STATE_FULL_DIM}]"
            )

        if (
            f[PATHS.next_obs_state].ndim != 2
            or f[PATHS.next_obs_state].shape[1] != STATE_FULL_DIM
        ):
            raise ValueError(
                f"{PATHS.next_obs_state} must have shape [N, {STATE_FULL_DIM}]"
            )

        if f[PATHS.act_action].ndim != 2 or f[PATHS.act_action].shape[1] != ACTION_FULL_DIM:
            raise ValueError(
                f"{PATHS.act_action} must have shape [N, {ACTION_FULL_DIM}]"
            )

        if (
            f[PATHS.act_joint_target].ndim != 2
            or f[PATHS.act_joint_target].shape[1] != ACTION_ARM_DIM
        ):
            raise ValueError(
                f"{PATHS.act_joint_target} must have shape [N, {ACTION_ARM_DIM}]"
            )

        if (
            f[PATHS.act_hand_target].ndim != 2
            or f[PATHS.act_hand_target].shape[1] != ACTION_HAND_DIM
        ):
            raise ValueError(
                f"{PATHS.act_hand_target} must have shape [N, {ACTION_HAND_DIM}]"
            )

    @staticmethod
    def _decode_scalar(x: Any) -> Any:
        if isinstance(x, bytes):
            return x.decode("utf-8")

        if isinstance(x, np.ndarray) and x.shape == ():
            x = x.item()
            if isinstance(x, bytes):
                return x.decode("utf-8")
            return x

        return x

    def _read_optional_scalar_dataset(self, f: h5py.File, path: str) -> Any | None:
        if path not in f:
            return None
        return self._decode_scalar(f[path][()])

    def _read_optional_list_dataset(self, f: h5py.File, path: str) -> list[Any] | None:
        if path not in f:
            return None

        raw = np.asarray(f[path][()])
        if raw.shape == ():
            raw = raw.reshape(1)
        items: list[Any] = []
        for item in raw.tolist():
            if isinstance(item, bytes):
                items.append(item.decode("utf-8"))
            else:
                items.append(item)
        return items

    def _read_optional_string_list_dataset(self, f: h5py.File, path: str) -> list[str] | None:
        items = self._read_optional_list_dataset(f, path)
        if items is None:
            return None
        return [str(item) for item in items]

    def _read_contract_meta_from_file(self, f: h5py.File) -> dict[str, Any]:
        meta: dict[str, Any] = {}

        for key, path, caster in (
            ("obs_dim", PATHS.meta_obs_dim, int),
            ("act_dim", PATHS.meta_act_dim, int),
            ("action_dim", PATHS.meta_action_dim, int),
        ):
            value = self._read_optional_scalar_dataset(f, path)
            if value is not None:
                meta[key] = caster(np.asarray(value).reshape(-1)[0])

        # Optional strings
        for key, path in (
            ("bag_name", PATHS.meta_bag_name),
            ("state_definition", PATHS.meta_state_definition),
            ("action_type", PATHS.meta_action_type),
            ("gripper_mode", PATHS.meta_gripper_mode),
            ("state_layout", PATHS.meta_state_layout),
            ("action_layout", PATHS.meta_action_layout),
        ):
            value = self._read_optional_scalar_dataset(f, path)
            if value is not None:
                meta[key] = str(value)

        instruction = self._read_optional_scalar_dataset(f, PATHS.meta_instruction)
        if instruction is not None:
            meta["instruction"] = str(instruction) if self.use_text else ""

        joint_names = self._read_optional_string_list_dataset(f, PATHS.meta_joint_names)
        if joint_names is not None:
            meta["joint_names"] = joint_names
            meta["arm_joint_names"] = joint_names

        hand_value_names = self._read_optional_string_list_dataset(f, PATHS.meta_hand_value_names)
        if hand_value_names is not None:
            meta["hand_value_names"] = hand_value_names

        for key, path in (
            ("hand_open_prototype_6", PATHS.meta_hand_open_prototype_6),
            ("hand_closed_prototype_6", PATHS.meta_hand_closed_prototype_6),
        ):
            value = self._read_optional_list_dataset(f, path)
            if value is not None:
                meta[key] = value

        return meta

    def _load_meta_once(self) -> dict[str, Any]:
        """
        Load per-file meta once and keep only the frozen meta keys.
        """
        if self._meta_cache is not None:
            return self._meta_cache

        f = self._get_h5()
        raw_meta = self._read_contract_meta_from_file(f)
        meta, warnings = fill_and_validate_contract_metadata(
            raw_meta, context=f"{self.hdf5_path}/meta", warn=True
        )
        meta["metadata_exists"] = bool("meta" in f)
        meta["metadata_source"] = "file" if not warnings else "file_with_defaults"
        meta["metadata_defaulted_fields"] = [msg.split(": missing ", 1)[1].split(";", 1)[0] for msg in warnings]

        self._meta_cache = meta
        return self._meta_cache

    def __getitem__(self, idx: int) -> dict[str, Any]:
        f = self._get_h5()
        meta = self._load_meta_once()

        # Raw states
        obs_state_full = np.asarray(f[PATHS.obs_state][idx], dtype=np.float32)
        next_obs_state_full = np.asarray(f[PATHS.next_obs_state][idx], dtype=np.float32)

        # Raw actions
        act_full = np.asarray(f[PATHS.act_action][idx], dtype=np.float32)
        act_arm = np.asarray(f[PATHS.act_joint_target][idx], dtype=np.float32)
        act_hand = np.asarray(f[PATHS.act_hand_target][idx], dtype=np.float32)

        # Required transition flag
        done = bool(np.asarray(f[PATHS.done][idx]))

        # Split canonical state/action
        obs_state = split_state_vector(obs_state_full)
        next_obs_state = split_state_vector(next_obs_state_full)
        act_state = split_action_vector(act_full)

        sample: dict[str, Any] = {
            "obs": {
                "state": {
                    "arm": torch.from_numpy(np.asarray(obs_state["arm"], dtype=np.float32)),
                    "hand": torch.from_numpy(np.asarray(obs_state["hand"], dtype=np.float32)),
                    "full": torch.from_numpy(np.asarray(obs_state["full"], dtype=np.float32)),
                },
            },
            "action": {
                "arm": torch.from_numpy(act_arm),
                "hand": torch.from_numpy(act_hand),
                "full": torch.from_numpy(np.asarray(act_state["full"], dtype=np.float32)),
            },
            "next_obs": {
                "state": {
                    "arm": torch.from_numpy(np.asarray(next_obs_state["arm"], dtype=np.float32)),
                    "hand": torch.from_numpy(np.asarray(next_obs_state["hand"], dtype=np.float32)),
                    "full": torch.from_numpy(np.asarray(next_obs_state["full"], dtype=np.float32)),
                }
            },
            "done": torch.tensor(done, dtype=torch.bool),
        }

        # Optional reward
        if PATHS.reward in f:
            reward = float(np.asarray(f[PATHS.reward][idx]))
            sample["reward"] = torch.tensor(reward, dtype=torch.float32)

        # Optional images
        if self.use_images:
            images: dict[str, torch.Tensor] = {}

            for key in IMAGE_KEYS:
                image_path = get_image_path(key)
                img = np.asarray(f[image_path][idx])
                images[key] = torch.from_numpy(img)

            sample["obs"]["images"] = images

        # Optional meta block
        if len(meta) > 0:
            sample["meta"] = dict(meta)

        return sample

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        self.close()
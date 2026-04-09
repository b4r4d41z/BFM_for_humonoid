from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import IMAGE_KEYS, PATHS, split_action_vector, split_state_vector


class HDF5DataStreamLoader(Dataset):
    def __init__(
        self,
        hdf5_path: str | Path,
        use_images: bool = True,
        use_text: bool = True,
    ) -> None:
        self.hdf5_path = str(hdf5_path)
        self.use_images = use_images
        self.use_text = use_text

        self._h5: h5py.File | None = None
        self._meta_cache: dict[str, Any] | None = None

        with h5py.File(self.hdf5_path, "r") as f:
            self._validate_structure(f)
            self.length = int(f[PATHS.done].shape[0])

    def __len__(self) -> int:
        return self.length

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def _validate_structure(self, f: h5py.File) -> None:
        required = [
            PATHS.obs_joint_pos,
            PATHS.obs_hand_pos,
            PATHS.obs_state,
            PATHS.next_obs_state,
            PATHS.act_joint_target,
            PATHS.act_hand_target,
            PATHS.act_action,
            PATHS.done,
            PATHS.reward,
            PATHS.timestamps,
            PATHS.meta_bag_name,
            PATHS.meta_instruction,
            PATHS.meta_joint_names,
            PATHS.meta_obs_dim,
            PATHS.meta_act_dim,
            PATHS.meta_state_definition,
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
            PATHS.obs_joint_pos,
            PATHS.obs_hand_pos,
            PATHS.obs_state,
            PATHS.next_obs_state,
            PATHS.act_joint_target,
            PATHS.act_hand_target,
            PATHS.act_action,
            PATHS.reward,
            PATHS.timestamps,
        ]

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

        if f[PATHS.obs_state].shape[1] != 26:
            raise ValueError(f"{PATHS.obs_state} must have shape [N, 26]")

        if f[PATHS.next_obs_state].shape[1] != 26:
            raise ValueError(f"{PATHS.next_obs_state} must have shape [N, 26]")

        if f[PATHS.act_action].shape[1] != 26:
            raise ValueError(f"{PATHS.act_action} must have shape [N, 26]")

        if f[PATHS.obs_joint_pos].shape[1] != 14:
            raise ValueError(f"{PATHS.obs_joint_pos} must have shape [N, 14]")

        if f[PATHS.obs_hand_pos].shape[1] != 12:
            raise ValueError(f"{PATHS.obs_hand_pos} must have shape [N, 12]")

        if f[PATHS.act_joint_target].shape[1] != 14:
            raise ValueError(f"{PATHS.act_joint_target} must have shape [N, 14]")

        if f[PATHS.act_hand_target].shape[1] != 12:
            raise ValueError(f"{PATHS.act_hand_target} must have shape [N, 12]")

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

    def _load_meta_once(self) -> dict[str, Any]:
        if self._meta_cache is not None:
            return self._meta_cache

        f = self._get_h5()

        bag_name = self._decode_scalar(f[PATHS.meta_bag_name][()])
        instruction = self._decode_scalar(f[PATHS.meta_instruction][()])
        state_definition = self._decode_scalar(f[PATHS.meta_state_definition][()])
        obs_dim = int(np.asarray(f[PATHS.meta_obs_dim][()]).reshape(-1)[0])
        act_dim = int(np.asarray(f[PATHS.meta_act_dim][()]).reshape(-1)[0])

        joint_names_raw = np.asarray(f[PATHS.meta_joint_names][()])
        joint_names = []
        for item in joint_names_raw:
            if isinstance(item, bytes):
                joint_names.append(item.decode("utf-8"))
            else:
                joint_names.append(str(item))

        self._meta_cache = {
            "bag_name": bag_name,
            "instruction": instruction,
            "joint_names": joint_names,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "state_definition": state_definition,
            "source": "hdf5",
            "path": self.hdf5_path,
        }
        return self._meta_cache

    def __getitem__(self, idx: int) -> dict[str, Any]:
        f = self._get_h5()
        meta = self._load_meta_once()

        obs_joint_pos = np.asarray(f[PATHS.obs_joint_pos][idx], dtype=np.float32)
        obs_hand_pos = np.asarray(f[PATHS.obs_hand_pos][idx], dtype=np.float32)
        obs_state_full = np.asarray(f[PATHS.obs_state][idx], dtype=np.float32)

        next_obs_state_full = np.asarray(f[PATHS.next_obs_state][idx], dtype=np.float32)

        act_joint_target = np.asarray(f[PATHS.act_joint_target][idx], dtype=np.float32)
        act_hand_target = np.asarray(f[PATHS.act_hand_target][idx], dtype=np.float32)
        act_full = np.asarray(f[PATHS.act_action][idx], dtype=np.float32)

        reward = float(np.asarray(f[PATHS.reward][idx]))
        done = bool(np.asarray(f[PATHS.done][idx]))
        timestamp = float(np.asarray(f[PATHS.timestamps][idx]))

        obs_state = split_state_vector(obs_state_full)
        next_obs_state = split_state_vector(next_obs_state_full)
        act_state = split_action_vector(act_full)

        images: dict[str, torch.Tensor] = {}
        if self.use_images:
            for key in IMAGE_KEYS:
                if key == "head":
                    img = np.asarray(f[PATHS.images_head][idx])
                elif key == "left_wrist":
                    img = np.asarray(f[PATHS.images_left_wrist][idx])
                elif key == "right_wrist":
                    img = np.asarray(f[PATHS.images_right_wrist][idx])
                else:
                    raise KeyError(f"Unknown image key: {key}")

                images[key] = torch.from_numpy(img)

        sample = {
            "obs": {
                "state": {
                    "arm_joints": torch.from_numpy(obs_joint_pos),
                    "hand_state": torch.from_numpy(obs_hand_pos),
                    "full": torch.from_numpy(obs_state["full"]),
                },
                "images": images,
                "text": meta["instruction"] if self.use_text else "",
                "timestamp": torch.tensor(timestamp, dtype=torch.float32),
            },
            "action": {
                "joint_target": torch.from_numpy(act_joint_target),
                "hand_target": torch.from_numpy(act_hand_target),
                "full": torch.from_numpy(act_state["full"]),
            },
            "next_obs": {
                "state": {
                    "arm_joints": torch.from_numpy(next_obs_state["arm_joints"]),
                    "hand_state": torch.from_numpy(next_obs_state["hand_state"]),
                    "full": torch.from_numpy(next_obs_state["full"]),
                }
            },
            "reward": torch.tensor(reward, dtype=torch.float32),
            "done": torch.tensor(done, dtype=torch.bool),
            "meta": {
                "bag_name": meta["bag_name"],
                "joint_names": meta["joint_names"],
                "obs_dim": meta["obs_dim"],
                "act_dim": meta["act_dim"],
                "state_definition": meta["state_definition"],
                "source": meta["source"],
                "path": meta["path"],
                "index": idx,
            },
        }

        return sample

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        self.close()
from __future__ import annotations

import dataclasses
import functools
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

Device = str | torch.device
TensorDict = dict[str, Any]


@functools.singledispatch
def _to_torch(value: Any, device: Device | None = None) -> Any:
    raise TypeError(
        f"No conversion to torch is registered for type: {type(value)}"
    )


@_to_torch.register(numbers.Number)
def _number_to_torch(value: numbers.Number, device: Device | None = None) -> torch.Tensor:
    tensor = torch.tensor(value)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


@_to_torch.register(np.ndarray)
def _ndarray_to_torch(value: np.ndarray, device: Device | None = None) -> torch.Tensor:
    tensor = torch.tensor(value)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


@_to_torch.register(torch.Tensor)
def _tensor_to_torch(value: torch.Tensor, device: Device | None = None) -> torch.Tensor:
    tensor = value.detach().clone()
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def dtype_numpytotorch(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == np.float16:
        return torch.float16
    if dtype == np.float32:
        return torch.float32
    if dtype == np.float64:
        return torch.float64
    if dtype == np.int16:
        return torch.int16
    if dtype == np.int32:
        return torch.int32
    if dtype == np.int64:
        return torch.int64
    if dtype == np.uint8:
        return torch.uint8
    if dtype == bool or dtype == np.bool_:
        return torch.bool
    raise ValueError(f"Unknown dtype: {dtype}")


def _first_leaf(d: Mapping) -> torch.Tensor | np.ndarray:
    for v in d.values():
        if isinstance(v, Mapping):
            leaf = _first_leaf(v)
            if leaf is not None:
                return leaf
        else:
            return v
    raise ValueError("Could not find a leaf tensor/array in nested dict.")


def _validate_numeric_nested_dict(d: Mapping, parent_key: str = "") -> None:
    for k, v in d.items():
        full_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, Mapping):
            _validate_numeric_nested_dict(v, parent_key=full_key)
        else:
            if not isinstance(v, (torch.Tensor, np.ndarray, numbers.Number)):
                raise TypeError(
                    f"Unsupported value in buffer input at key '{full_key}': {type(v)}"
                )


def _validate_batch_dimensions(d: Mapping, expected_batch_size: int, parent_key: str = "") -> None:
    for k, v in d.items():
        full_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, Mapping):
            _validate_batch_dimensions(v, expected_batch_size=expected_batch_size, parent_key=full_key)
        else:
            if isinstance(v, numbers.Number):
                raise ValueError(
                    f"Leaf '{full_key}' is a scalar. Buffer expects batched tensors/arrays."
                )
            if v.shape[0] != expected_batch_size:
                raise ValueError(
                    f"Batch size mismatch at key '{full_key}': "
                    f"expected {expected_batch_size}, got {v.shape[0]}"
                )


def initialize_storage(data: Mapping, storage: dict, capacity: int, device: Device) -> None:
    def recursive_initialize(src: Mapping, dst: dict) -> None:
        for k, v in src.items():
            if isinstance(v, Mapping):
                dst[k] = {}
                recursive_initialize(v, dst[k])
            else:
                if isinstance(v, numbers.Number):
                    raise ValueError(
                        f"Leaf '{k}' is a scalar. Expected batched tensor/array."
                    )

                leaf_shape = tuple(v.shape[1:]) if len(v.shape) > 1 else ()
                dst[k] = torch.zeros(
                    (capacity, *leaf_shape),
                    device=device,
                    dtype=dtype_numpytotorch(v.dtype if hasattr(v, "dtype") else type(v)),
                )

    recursive_initialize(data, storage)


def extract_values(d: Mapping, idxs: torch.Tensor | np.ndarray | list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, Mapping):
            result[k] = extract_values(v, idxs)
        else:
            result[k] = v[idxs]
    return result


def dict_cat(d: Mapping) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, Mapping):
            result[k] = dict_cat(v)
        else:
            result[k] = torch.cat(v, dim=0)
    return result


def to_buffer_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """
    Converts your assembled HDF5 batch into a pure numeric transition batch
    suitable for DictBuffer.

    Input expected:
        batch["obs"]["state"]["full"]         -> [B, 26]
        batch["action"]["full"]               -> [B, 26]
        batch["next_obs"]["state"]["full"]    -> [B, 26]
        batch["reward"]                       -> [B]
        batch["done"]                         -> [B]

    Output:
        {
            "observation": [B, 26],
            "action": [B, 26],
            "next": {
                "observation": [B, 26],
            },
            "reward": [B, 1],
            "done": [B, 1],
        }
    """
    required_keys = [
        "obs",
        "action",
        "next_obs",
        "reward",
        "done",
    ]
    for key in required_keys:
        if key not in batch:
            raise KeyError(f"Missing required batch key: {key}")

    observation = batch["obs"]["state"]["full"]
    action = batch["action"]["full"]
    next_observation = batch["next_obs"]["state"]["full"]
    reward = batch["reward"]
    done = batch["done"]

    if reward.ndim == 1:
        reward = reward.reshape(-1, 1)
    if done.ndim == 1:
        done = done.reshape(-1, 1)

    return {
        "observation": observation,
        "action": action,
        "next": {
            "observation": next_observation,
        },
        "reward": reward,
        "done": done,
    }


@dataclasses.dataclass
class DictBuffer:
    capacity: int
    device: Device = "cpu"

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {self.capacity}")
        self.storage: dict[str, Any] | None = None
        self._idx = 0
        self._is_full = False

    def __len__(self) -> int:
        return self.capacity if self._is_full else self._idx

    def empty(self) -> bool:
        return len(self) == 0

    @torch.no_grad()
    def extend(self, data: dict[str, Any]) -> None:
        _validate_numeric_nested_dict(data)

        first_leaf = _first_leaf(data)
        if isinstance(first_leaf, numbers.Number):
            raise ValueError("Buffer input must be batched, got scalar leaf.")
        batch_size = int(first_leaf.shape[0])

        _validate_batch_dimensions(data, expected_batch_size=batch_size)

        if self.storage is None:
            self.storage = {}
            initialize_storage(data, self.storage, capacity=self.capacity, device=self.device)
            self._idx = 0
            self._is_full = False

        def add_new_data(src: Mapping, dst: dict) -> None:
            for k, v in src.items():
                if isinstance(v, Mapping):
                    add_new_data(v, dst[k])
                else:
                    tensor_v = _to_torch(v, device=self.device)
                    end = self._idx + batch_size

                    if end >= self.capacity:
                        first_part = self.capacity - self._idx
                        second_part = batch_size - first_part

                        dst[k][self._idx:] = tensor_v[:first_part]
                        if second_part > 0:
                            dst[k][:second_part] = tensor_v[first_part:]

                        self._is_full = True
                    else:
                        dst[k][self._idx:end] = tensor_v

        add_new_data(data, self.storage)
        self._idx = (self._idx + batch_size) % self.capacity

    @torch.no_grad()
    def extend_stream_batch(self, batch: dict[str, Any]) -> None:
        """
        Directly accepts your current assembled batch from batch_assembly.py,
        converts it with to_buffer_batch(...), and writes it to the buffer.
        """
        buffer_batch = to_buffer_batch(batch)
        self.extend(buffer_batch)

    @torch.no_grad()
    def sample(self, batch_size: int) -> dict[str, Any]:
        if self.empty():
            raise ValueError("Cannot sample from an empty buffer.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        idxs = torch.randint(0, len(self), (batch_size,), device=self.device)
        return extract_values(self.storage, idxs)

    def get_full_buffer(self) -> dict[str, Any]:
        if self.storage is None:
            raise ValueError("Buffer is empty; no storage initialized yet.")
        if self._is_full:
            return self.storage
        idxs = torch.arange(0, len(self), device=self.device)
        return extract_values(self.storage, idxs)
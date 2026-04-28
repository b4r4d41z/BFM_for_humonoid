from __future__ import annotations

from pathlib import Path

import torch


def check_checkpoint_exists(path: str) -> Path:
    ckpt = Path(path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt}")
    if not ckpt.is_file():
        raise ValueError(f"Checkpoint path is not a file: {ckpt}")
    return ckpt


def check_tensor_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    finite_mask = torch.isfinite(tensor)
    if not bool(finite_mask.all().item()):
        bad = int((~finite_mask).sum().item())
        raise ValueError(f"{name} contains non-finite values (NaN/Inf). bad_count={bad}")


def check_shape(name: str, tensor: torch.Tensor, expected_last_dim: int | None = None) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [num_envs, dim], got {tuple(tensor.shape)}")
    if expected_last_dim is not None and int(tensor.shape[-1]) != int(expected_last_dim):
        raise ValueError(
            f"{name} last dim mismatch. expected={expected_last_dim}, got={int(tensor.shape[-1])}"
        )


def check_device(name: str, tensor: torch.Tensor, expected_device: str | torch.device) -> None:
    want = torch.device(expected_device)
    if tensor.device != want:
        raise ValueError(f"{name} device mismatch. expected={want}, got={tensor.device}")


def print_debug_tensor(name: str, tensor: torch.Tensor) -> None:
    print(
        f"[debug] {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} min={float(tensor.min().item()):.4f} "
        f"max={float(tensor.max().item()):.4f}"
    )

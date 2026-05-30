from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bc.buffers.buffers import OfflineTrajectoryBuffer
from bc.data.hdf5_discovery import discover_h5_files, limit_h5_files, print_h5_dataset_summary


def tensor_shape_str(x: Any) -> str:
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    return f"<{type(x).__name__}>"


def print_nested_shapes(x: Any, prefix: str = "") -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            print_nested_shapes(v, new_prefix)
        return

    if isinstance(x, list):
        print(f"{prefix}: list(len={len(x)})")
        return

    if isinstance(x, torch.Tensor):
        print(f"{prefix}: shape={tuple(x.shape)}, dtype={x.dtype}")
        return

    print(f"{prefix}: type={type(x).__name__}")


def assert_transition_batch(batch: dict[str, Any], batch_size: int) -> None:
    assert batch["obs"]["state"]["arm"].shape == (batch_size, 14)
    assert batch["obs"]["state"]["hand"].shape == (batch_size, 12)
    assert batch["obs"]["state"]["full"].shape == (batch_size, 26)

    assert batch["action"]["arm"].shape == (batch_size, 14)
    assert batch["action"]["hand"].shape == (batch_size, 12)
    assert batch["action"]["full"].shape == (batch_size, 26)

    assert batch["next_obs"]["state"]["arm"].shape == (batch_size, 14)
    assert batch["next_obs"]["state"]["hand"].shape == (batch_size, 12)
    assert batch["next_obs"]["state"]["full"].shape == (batch_size, 26)

    assert batch["done"].shape == (batch_size,)

    if "reward" in batch:
        assert batch["reward"].shape == (batch_size,)

    if "images" in batch["obs"]:
        head = batch["obs"]["images"]["head"]
        left = batch["obs"]["images"]["left_wrist"]
        right = batch["obs"]["images"]["right_wrist"]

        assert head.ndim == 4
        assert left.ndim == 4
        assert right.ndim == 4

        assert head.shape[0] == batch_size
        assert left.shape[0] == batch_size
        assert right.shape[0] == batch_size


def assert_sequence_batch(batch: dict[str, Any], batch_size: int, seq_len: int) -> None:
    assert batch["obs"]["state"]["arm"].shape == (batch_size, seq_len, 14)
    assert batch["obs"]["state"]["hand"].shape == (batch_size, seq_len, 12)
    assert batch["obs"]["state"]["full"].shape == (batch_size, seq_len, 26)

    assert batch["action"]["arm"].shape == (batch_size, seq_len, 14)
    assert batch["action"]["hand"].shape == (batch_size, seq_len, 12)
    assert batch["action"]["full"].shape == (batch_size, seq_len, 26)

    assert batch["next_obs"]["state"]["arm"].shape == (batch_size, seq_len, 14)
    assert batch["next_obs"]["state"]["hand"].shape == (batch_size, seq_len, 12)
    assert batch["next_obs"]["state"]["full"].shape == (batch_size, seq_len, 26)

    assert batch["done"].shape == (batch_size, seq_len)

    if "reward" in batch:
        assert batch["reward"].shape == (batch_size, seq_len)

    if "images" in batch["obs"]:
        head = batch["obs"]["images"]["head"]
        left = batch["obs"]["images"]["left_wrist"]
        right = batch["obs"]["images"]["right_wrist"]

        assert head.ndim == 5
        assert left.ndim == 5
        assert right.ndim == 5

        assert head.shape[0] == batch_size
        assert left.shape[0] == batch_size
        assert right.shape[0] == batch_size

        assert head.shape[1] == seq_len
        assert left.shape[1] == seq_len
        assert right.shape[1] == seq_len


def main() -> None:
    parser = argparse.ArgumentParser(description="BFM buffer smoke test")
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths: .h5 files and/or directories with .h5 files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If set, scan data directories recursively for .h5 files",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=2,
        help="How many .h5 files to load at most",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for transition sampling",
    )
    parser.add_argument(
        "--seq_batch_size",
        type=int,
        default=2,
        help="Batch size for sequence sampling",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=8,
        help="Sequence length for sequence sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Buffer storage device",
    )
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="Disable image loading",
    )
    args = parser.parse_args()

    data_paths = [Path(path).expanduser().resolve() for path in args.data]
    discovered_h5_files = discover_h5_files(data_paths, recursive=args.recursive)
    h5_files = limit_h5_files(discovered_h5_files, max_files=args.max_files)

    print_h5_dataset_summary(
        data_roots=data_paths,
        recursive=args.recursive,
        discovered_h5_files=discovered_h5_files,
        selected_h5_files=h5_files,
        max_files=args.max_files,
        prefix="[BFM buffer check]",
    )

    buffer = OfflineTrajectoryBuffer.from_hdf5_files(
        hdf5_paths=h5_files,
        use_images=not args.no_images,
        use_text=True,
        device=args.device,
        seed=42,
        log_prefix="[BFM buffer check][hdf5]",
    )

    print("\nBuffer summary:")
    summary = buffer.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\nSampling transition batch...")
    transition_batch = buffer.sample_transitions(
        batch_size=args.batch_size,
        device="cpu",
    )
    print_nested_shapes(transition_batch)
    assert_transition_batch(transition_batch, batch_size=args.batch_size)
    print("Transition batch check: OK")

    print("\nSampling sequence batch...")
    sequence_batch = buffer.sample_sequences(
        batch_size=args.seq_batch_size,
        seq_len=args.seq_len,
        device="cpu",
    )
    print_nested_shapes(sequence_batch)
    assert_sequence_batch(
        sequence_batch,
        batch_size=args.seq_batch_size,
        seq_len=args.seq_len,
    )
    print("Sequence batch check: OK")

    print("\nAll buffer smoke tests passed successfully.")


if __name__ == "__main__":
    main()
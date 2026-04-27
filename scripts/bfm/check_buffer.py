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

from bfm.buffers.buffers import OfflineTrajectoryBuffer


def collect_h5_files(path: Path, max_files: int) -> list[Path]:
    if path.is_file():
        if path.suffix != ".h5":
            raise ValueError(f"Expected .h5 file, got: {path}")
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    files = sorted(path.glob("*.h5"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {path}")

    return files[:max_files]


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
        required=True,
        help="Path to one .h5 file or a directory with .h5 files",
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

    data_path = Path(args.data).expanduser().resolve()
    h5_files = collect_h5_files(data_path, max_files=args.max_files)

    print("Selected H5 files:")
    for i, path in enumerate(h5_files):
        print(f"  [{i}] {path}")

    buffer = OfflineTrajectoryBuffer.from_hdf5_files(
        hdf5_paths=h5_files,
        use_images=not args.no_images,
        use_text=True,
        device=args.device,
        seed=42,
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
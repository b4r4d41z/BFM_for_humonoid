from __future__ import annotations

import argparse
import resource
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from bc.buffers.buffers import OfflineTrajectoryBuffer
from bc.data.hdf5_discovery import discover_h5_files, limit_h5_files
from bc.data.hdf5_streaming_dataset import HDF5StreamingDataset


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def rss_mib() -> float:
    # Linux reports KiB; macOS reports bytes. This project primarily runs on Linux.
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0


def warn_path(paths: list[Path]) -> None:
    for path in paths:
        text = str(path)
        if text.startswith("/run/user/1000/gvfs/") or "/gvfs/" in text:
            print("[benchmark][warn] You are using a GVFS SMB path. For HDF5 training, CIFS mount or local SSD is recommended.")
        if text.startswith("/mnt/tank6124_sharefolders"):
            print(f"[benchmark] CIFS NAS path detected: {path}")


def run_streaming(files: list[Path], args: argparse.Namespace, num_workers: int) -> None:
    start = perf_counter()
    dataset = HDF5StreamingDataset(
        files,
        use_images=args.use_images,
        cameras=args.cameras,
        image_size=args.image_size,
        frame_stride=args.frame_stride,
        seq_len=args.seq_len,
        log_prefix=f"[benchmark][streaming][workers={num_workers}]",
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        if args.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    build_elapsed = perf_counter() - start

    it = iter(loader)
    first_start = perf_counter()
    first = next(it)
    time_to_first = perf_counter() - first_start
    del first

    batches = 1
    bench_start = perf_counter()
    for _ in range(max(0, args.batches - 1)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        del batch
        batches += 1
        if batches == args.batches or batches % max(1, args.batches // 4) == 0:
            print(f"[benchmark][streaming][workers={num_workers}] progress={batches / args.batches * 100:.0f}% batches={batches}/{args.batches}", flush=True)
    elapsed = perf_counter() - bench_start
    print(
        f"[benchmark][streaming][workers={num_workers}] build_index_sec={build_elapsed:.3f} "
        f"time_to_first_batch_sec={time_to_first:.3f} batches_per_sec={batches / max(elapsed, 1e-9):.3f} "
        f"samples_per_sec={batches * args.batch_size / max(elapsed, 1e-9):.3f} max_rss_mib={rss_mib():.1f}"
    )



def run_eager(files: list[Path], args: argparse.Namespace) -> None:
    start = perf_counter()
    buffer = OfflineTrajectoryBuffer.from_hdf5_files(
        files,
        use_images=args.use_images,
        use_text=False,
        device="cpu",
        seed=42,
        log_prefix="[benchmark][eager]",
    )
    load_elapsed = perf_counter() - start
    first_start = perf_counter()
    if args.seq_len > 1:
        batch = buffer.sample_sequences(args.batch_size, args.seq_len, device="cpu")
    else:
        batch = buffer.sample_transitions(args.batch_size, device="cpu")
    time_to_first = perf_counter() - first_start
    del batch
    bench_start = perf_counter()
    for i in range(args.batches):
        if args.seq_len > 1:
            batch = buffer.sample_sequences(args.batch_size, args.seq_len, device="cpu")
        else:
            batch = buffer.sample_transitions(args.batch_size, device="cpu")
        del batch
        if (i + 1) == args.batches or (i + 1) % max(1, args.batches // 4) == 0:
            print(f"[benchmark][eager] progress={(i + 1) / args.batches * 100:.0f}% batches={i + 1}/{args.batches}", flush=True)
    elapsed = perf_counter() - bench_start
    print(
        f"[benchmark][eager] preload_sec={load_elapsed:.3f} time_to_first_batch_after_preload_sec={time_to_first:.3f} "
        f"batches_per_sec={args.batches / max(elapsed, 1e-9):.3f} samples_per_sec={args.batches * args.batch_size / max(elapsed, 1e-9):.3f} "
        f"max_rss_mib={rss_mib():.1f}"
    )

def main() -> None:
    parser = argparse.ArgumentParser("Benchmark BFM BC HDF5 loaders")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max_files", type=int, default=8)
    parser.add_argument("--mode", choices=["streaming", "eager", "both"], default="streaming")
    parser.add_argument("--num_workers", type=str, default="0,2,4,8", help="Comma-separated worker counts to test.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--use_images", type=str_to_bool, default=True)
    parser.add_argument("--cameras", type=str, default="head,left_wrist,right_wrist")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    args = parser.parse_args()

    roots = [Path(p).expanduser().resolve() for p in args.data]
    warn_path(roots)
    files = limit_h5_files(discover_h5_files(roots, recursive=args.recursive), args.max_files)
    print(f"[benchmark] files={len(files)} batch_size={args.batch_size} batches={args.batches}")
    if args.mode in {"streaming", "both"}:
        for workers in [int(x) for x in args.num_workers.split(",") if x.strip()]:
            run_streaming(files, args, workers)
    if args.mode in {"eager", "both"}:
        run_eager(files, args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from bc.data.hdf5_discovery import discover_h5_files, limit_h5_files
from bc.data.schema import IMAGE_KEYS, PATHS, get_image_path



def fmt_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def path_warning(paths: list[Path]) -> None:
    for path in paths:
        text = str(path)
        if text.startswith("/run/user/1000/gvfs/") or "/gvfs/" in text:
            print("[inspect][warn] You are using a GVFS SMB path. For HDF5 training, CIFS mount or local SSD is recommended.")
        if text.startswith("/mnt/tank6124_sharefolders"):
            print(f"[inspect] CIFS NAS path detected: {path}")


def dataset_bytes(ds: h5py.Dataset) -> int:
    return int(np.prod(ds.shape)) * int(ds.dtype.itemsize)


def main() -> None:
    parser = argparse.ArgumentParser("Inspect BFM HDF5 dataset")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--use_images", action="store_true", help="Include image datasets in eager-size estimate.")
    args = parser.parse_args()

    roots = [Path(p).expanduser().resolve() for p in args.data]
    path_warning(roots)
    start = perf_counter()
    files = limit_h5_files(discover_h5_files(roots, recursive=args.recursive), args.max_files)
    print(f"[inspect] files={len(files)} discovery_elapsed={perf_counter() - start:.2f}s")

    total_transitions = 0
    raw_state_action_bytes = 0
    raw_image_bytes = 0
    state_dim = None
    action_dim = None
    image_shapes: dict[str, tuple[int, ...]] = {}

    scan_start = perf_counter()
    for i, path in enumerate(files, start=1):
        if i == len(files) or i % max(1, len(files) // 10) == 0:
            print(f"[inspect] progress={i / max(1, len(files)) * 100:.0f}% files={i}/{len(files)}", flush=True)
        with h5py.File(path, "r") as f:
            n = int(f[PATHS.done].shape[0])
            total_transitions += n
            if PATHS.obs_state in f:
                state_dim = int(f[PATHS.obs_state].shape[-1])
                raw_state_action_bytes += dataset_bytes(f[PATHS.obs_state])
            if PATHS.next_obs_state in f:
                raw_state_action_bytes += dataset_bytes(f[PATHS.next_obs_state])
            if PATHS.act_action in f:
                action_dim = int(f[PATHS.act_action].shape[-1])
                raw_state_action_bytes += dataset_bytes(f[PATHS.act_action])
            for path_name in (PATHS.act_joint_target, PATHS.act_hand_target, PATHS.done, PATHS.reward):
                if path_name in f:
                    raw_state_action_bytes += dataset_bytes(f[path_name])
            for cam in IMAGE_KEYS:
                img_path = get_image_path(cam)
                if img_path in f:
                    image_shapes.setdefault(cam, tuple(f[img_path].shape))
                    raw_image_bytes += dataset_bytes(f[img_path])

    eager_with_images = raw_state_action_bytes + raw_image_bytes
    streaming_index_bytes = total_transitions * 16  # rough (file_id,timestep) Python-free lower bound
    print(f"[inspect] scan_elapsed={perf_counter() - scan_start:.2f}s")
    print(f"[inspect] number of files: {len(files)}")
    print(f"[inspect] transitions: {total_transitions}")
    print(f"[inspect] state_dim: {state_dim}")
    print(f"[inspect] action_dim: {action_dim}")
    print("[inspect] image shapes (first observed per camera):")
    for cam in IMAGE_KEYS:
        print(f"[inspect]   {cam}: {image_shapes.get(cam, 'missing')}")
    print(f"[inspect] estimated raw state/action/done/reward size: {fmt_bytes(raw_state_action_bytes)}")
    print(f"[inspect] estimated raw image size: {fmt_bytes(raw_image_bytes)}")
    print(f"[inspect] eager mode raw tensor estimate with images: {fmt_bytes(eager_with_images)}")
    print(f"[inspect] eager mode raw tensor estimate state-only: {fmt_bytes(raw_state_action_bytes)}")
    print(f"[inspect] streaming mode steady-state estimate: DataLoader batches + worker HDF5 caches; lightweight index lower-bound {fmt_bytes(streaming_index_bytes)}")
    print('[inspect] recommended NAS path: /mnt/tank6124_sharefolders/Datasets/BFM_dataset/hdf5')


if __name__ == "__main__":
    main()

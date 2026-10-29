"""Convert episodes, validate stores, or create explicit episode-level manifests."""
import argparse
import json
from pathlib import Path
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", nargs="*")
    p.add_argument("--output", help="One output .zarr for one input")
    p.add_argument("--output-dir", help="Batch output directory")
    p.add_argument("--collection-commit")
    p.add_argument("--max-camera-skew", type=float, default=None, help="Optional inter-camera skew limit, seconds")
    p.add_argument("--max-ages", nargs=5, type=float, metavar=("STATE", "JOINT", "HEAD", "LEFT", "RIGHT"))
    p.add_argument("--validate", metavar="ZARR")
    p.add_argument("--manifest", help="Create manifest only; assign episodes below explicitly")
    p.add_argument("--train", nargs="*", default=[])
    p.add_argument("--validation", nargs="*", default=[])
    p.add_argument("--test", nargs="*", default=[])
    a = p.parse_args()
    from bfm.data.sharpa import validate, SharpaDataset
    import zarr
    if a.validate:
        print(validate(zarr.open_group(a.validate, mode="r"))); return
    if a.manifest:
        path = Path(a.manifest).resolve()
        if path.exists(): raise FileExistsError(path)
        entries = []
        import os
        for split in ("train", "validation", "test"):
            for name in getattr(a, split):
                validate(zarr.open_group(name, mode="r"))
                entries.append({"path": os.path.relpath(Path(name).resolve(), path.parent), "split": split})
        content = {"schema_version": "3.0.0", "dataset_id": uuid.uuid4().hex, "episodes": entries}
        path.parent.mkdir(parents=True, exist_ok=True)
        # Validate before exposing the final manifest; always require train data.
        temporary = path.with_suffix('.json.tmp')
        try:
            temporary.write_text(json.dumps(content, indent=2))
            SharpaDataset(temporary)
            temporary.rename(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(path); return
    if not a.episodes or not a.max_ages or not a.collection_commit or len(a.collection_commit) != 40 or bool(a.output) == bool(a.output_dir):
        p.error("Specify episode(s), --output OR --output-dir, --collection-commit (40 chars), and five --max-ages seconds")
    if a.output and len(a.episodes) != 1:
        p.error("--output requires exactly one episode")
    from bfm.data.mcap import convert
    for episode in a.episodes:
        output = a.output or str(Path(a.output_dir)/(Path(episode).name + '.zarr'))
        print(convert(episode, output, max_age_s=a.max_ages, collection_commit=a.collection_commit,
                      max_camera_skew_s=a.max_camera_skew))


if __name__ == "__main__":
    main()

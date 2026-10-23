import hashlib
import json
from pathlib import Path

from bfm.contracts.kuavo import BFMStateAdapter, BFMActionAdapter, KuavoContract
from .schema import SCHEMA_VERSION
from .zarr_io import read_episode


class BFMDataset:
    """Small-state dataset loaded once in chunked Zarr reads, without images.

    Manifest splits are explicit and cannot silently leak test trajectories into training.
    """
    def __init__(self, manifest, split="train"):
        path = Path(manifest)
        self.manifest = json.loads(path.read_text())
        if self.manifest.get("schema_version") != SCHEMA_VERSION or not self.manifest.get("dataset_id"):
            raise ValueError("Manifest requires schema_version=2.0.0 and dataset_id")
        self.episodes, self.ids = [], []
        self.contract = None
        digest = hashlib.sha256()
        digest.update(json.dumps(self.manifest, sort_keys=True).encode())
        paths = set()
        for entry in self.manifest["episodes"]:
            if entry["split"] not in ("train", "validation", "test"):
                raise ValueError("Invalid manifest split")
            p = (path.parent / entry["path"]).resolve()
            if p in paths:
                raise ValueError("Duplicate episode path across manifest splits")
            paths.add(p)
            if entry["split"] != split:
                continue
            arrays, meta = read_episode(p)
            if meta["split"] != split or meta["episode_id"] in self.ids:
                raise ValueError("Episode split mismatch or duplicate episode id")
            c = KuavoContract(**meta["contract"])
            if self.contract is not None and c != self.contract:
                raise ValueError("All episodes must share the same robot/action/timing contract")
            self.contract = c
            self.ids.append(meta["episode_id"])
            digest.update(json.dumps(meta, sort_keys=True).encode())
            for k, a in sorted(arrays.items()):
                digest.update(k.encode()); digest.update(str(a.dtype).encode()); digest.update(a.tobytes())
            adapter = BFMStateAdapter()
            state = lambda prefix: adapter({k: arrays[f"{prefix}/{k}"] for k in adapter.fields})
            self.episodes.append({
                "observation": state("obs"), "next_observation": state("next_obs"),
                "action": BFMActionAdapter(c).normalize(arrays["action"]),
                "terminated": arrays["terminated"], "truncated": arrays["truncated"],
                "duration": arrays["next_timestamp"] - arrays["timestamp"],
            })
        if not self.episodes:
            raise ValueError(f"No {split} episodes")
        self.fingerprint = digest.hexdigest()

"""Versioned superset episode storage; task56 is a training adapter, not raw data."""
import hashlib
import json
from pathlib import Path
import os
import shutil
import uuid
import numpy as np
import torch
import zarr
from numcodecs import Blosc
from bfm.contracts.sharpa import SCHEMA, VERSION, PROFILE, FIELDS, GROUPS, JOINTS, SharpaContract, ActionNormalizer, validate_states

CAMERAS = ("head", "left_wrist", "right_wrist")
SOURCES = ("state", "joints", *CAMERAS)


def validate(root, *, check_image_pixels=True):
    m = dict(root.attrs)
    if (m.get("schema_name"), m.get("schema_version")) != (SCHEMA, VERSION):
        raise ValueError("Requires Sharpa schema 3.0.0; legacy data cannot be reinterpreted")
    expected = SharpaContract().to_dict()
    if m.get("contract") != expected or m.get("state_order") != list(FIELDS):
        raise ValueError("State/action contract mismatch")
    if m.get("profile") != PROFILE or m.get("sync_clock") != "mcap_log_time_ns":
        raise ValueError("Unsupported observation profile / clock")
    if m.get("source_columns") != list(SOURCES) or m.get("state_units") != {k: u for k, (_, u) in FIELDS.items()}:
        raise ValueError("State units or source-column ordering mismatch")
    names = m.get("joint_names", [])
    if m.get('joint_units') != {'position': 'rad', 'velocity': 'rad/s', 'effort': 'N*m'}:
        raise ValueError('Measured joint units must be rad, rad/s and N*m')
    if len(set(names)) != len(names) or not set(JOINTS["left"] + JOINTS["right"]) <= set(names):
        raise ValueError("Measured joint names must be unique and include both arms")
    if not m.get("source_episode_id") or not m.get("source_metadata") or len(m.get("collection_commit", "")) != 40:
        raise ValueError("Missing source provenance")
    if m['source_metadata'].get('bfm', {}).get('contract_version') != 3:
        raise ValueError('Collection v3 source metadata required; collection-v2 data cannot be relabelled')
    t = len(root["action"])
    shapes = {"action": (t, 26), "action_header_ns": (t,), "timestamp": (t,), "next_timestamp": (t,), "record_ns": (t,), "next_record_ns": (t,),
              "terminated": (t,), "truncated": (t,), "segment_id": (t,)}
    for p in ("obs", "next_obs"):
        shapes.update({f"{p}/state": (t, 56), f"{p}/joint_position": (t, len(names)), f"{p}/joint_velocity": (t, len(names)),
                       f"{p}/source_record_ns": (t, 5), f"{p}/source_header_ns": (t, 5), f"{p}/sync_age_ns": (t, 5)})
        if m.get("has_effort"):
            shapes[f"{p}/joint_effort"] = (t, len(names))
    if t < 1:
        raise ValueError("Empty episode")
    present = set()
    root.visititems(lambda n, v: present.add(n) if isinstance(v, zarr.Array) else None)
    images = {f"{p}/images/{c}" for p in ("obs", "next_obs") for c in CAMERAS}
    if present != set(shapes) | images:
        raise ValueError("Missing/unexpected arrays")
    for k, shape in shapes.items():
        a = root[k][:]
        if a.shape != shape or a.dtype.kind not in "fiub" or not np.isfinite(a).all():
            raise ValueError(f"Invalid finite array {k}, expected {shape}")
        dtype = np.bool_ if k in ("terminated", "truncated") else np.int64 if k.endswith(("_ns", "segment_id")) else np.float64
        if a.dtype != dtype:
            raise ValueError(f"Wrong dtype for {k}: expected {dtype}")
    ActionNormalizer().normalize(root["action"][:])
    for p in ("obs", "next_obs"):
        validate_states(root[f"{p}/state"][:])
        tick = root["record_ns" if p == "obs" else "next_record_ns"][:]
        age = tick[:, None] - root[f"{p}/source_record_ns"][:]
        limits = np.asarray(m["max_age_ns"])
        if limits.shape != (5,) or (limits <= 0).any() or (age < 0).any() or (age > limits).any() or not np.array_equal(age, root[f"{p}/sync_age_ns"][:]):
            raise ValueError("Future/stale source sample or incorrect sync age")
        for c in CAMERAS:
            arr = root[f"{p}/images/{c}"]
            if arr.dtype != np.uint8 or len(arr.shape) != 4 or arr.shape[0] != t or arr.shape[-1] != 3 or min(arr.shape[1:3]) < 1:
                raise ValueError("Cameras must be RGB uint8 [T,H,W,3]")
            if list(arr.shape[1:]) != m["image_shapes"][c]:
                raise ValueError("Camera shape/metadata mismatch")
    if m.get("stored_image_encoding") != "rgb8":
        raise ValueError("Canonical image encoding must be rgb8")
    r, nr = root["record_ns"][:], root["next_record_ns"][:]
    dt = (nr-r)/1e9
    if (np.diff(r) <= 0).any() or (dt < .05).any() or (dt > .15).any():
        raise ValueError("Duplicate/invalid action ticks; expected 10 Hz within [0.05,0.15] s")
    origin = m["record_origin_ns"]
    if not np.allclose(root["timestamp"][:], (r-origin)/1e9, atol=1e-9, rtol=0) or not np.allclose(root["next_timestamp"][:], (nr-origin)/1e9, atol=1e-9, rtol=0):
        raise ValueError("Relative times differ from record timestamps")
    term, trunc, seg = root["terminated"][:], root["truncated"][:], root["segment_id"][:]
    # No task terminal event is available in this source version.
    ends = np.r_[seg[1:] != seg[:-1], True]
    if term.any() or not np.array_equal(trunc, ends) or seg[0] != 0 or (np.diff(seg) < 0).any() or (np.diff(seg) > 1).any():
        raise ValueError("Recording/gap boundaries must be truncations, never task-success terminals")
    continuous = ~ends[:-1]
    if not np.array_equal(nr[:-1][continuous], r[1:][continuous]):
        raise ValueError("Transition jumped across a missing tick")
    if (nr[:-1][~continuous] >= r[1:][~continuous]).any():
        raise ValueError("Segments overlap")
    keys = ["state", "joint_position", "joint_velocity", "source_record_ns", "source_header_ns", "sync_age_ns"]
    if m.get("has_effort"):
        keys.append("joint_effort")
    for key in keys:
        a, b = root[f"next_obs/{key}"][:-1][continuous], root[f"obs/{key}"][1:][continuous]
        same = np.array_equal(a, b) if a.dtype.kind in 'iu' else np.allclose(a, b, atol=1e-7, rtol=0)
        if not same:
            raise ValueError(f"next_obs continuity failed: {key}")
    if check_image_pixels:
        for c in CAMERAS:
            for i in np.flatnonzero(continuous):
                if not np.array_equal(root[f"next_obs/images/{c}"][i], root[f"obs/images/{c}"][i+1]):
                    raise ValueError(f"Image continuity failed: {c}")
    return {"transitions": t, "segments": int(ends.sum()), "state_dim": 56, "action_dim": 26}


def write(path, arrays, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}")
    try:
        root = zarr.open_group(str(temporary), mode="w")
        root.attrs.update(metadata)
        for k, v in arrays.items():
            root.create_dataset(k, data=v, chunks=(1 if '/images/' in k else min(len(v), 1024),) + v.shape[1:], compressor=Blosc(cname="zstd", clevel=3))
        result = validate(root)
        os.rename(temporary, path)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class SharpaDataset:
    def __init__(self, manifest, split="train", observation_profile="task56", outcome="all"):
        from bfm.contracts.vision import PROFILE as VISION, proprioception
        if observation_profile not in (PROFILE, VISION) or outcome not in ('all', 'successful', 'failed'):
            raise ValueError("Unknown observation_profile or outcome selection")
        self.observation_profile = observation_profile
        self.image_rows, self.roots = [], []
        path = Path(manifest)
        self.manifest = json.loads(path.read_text())
        if self.manifest.get("schema_version") != VERSION or not self.manifest.get("dataset_id"):
            raise ValueError("Requires explicit Sharpa v3 manifest")
        self.contract, self.ids, self.episodes = SharpaContract(), [], []
        self.schema_version, self.state_fields = VERSION, list(FIELDS)
        digest = hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode())
        seen_ids, seen_paths = set(), set()
        for entry in self.manifest["episodes"]:
            p = (path.parent / entry["path"]).resolve()
            root = zarr.open_group(str(p), mode="r")
            identity = root.attrs["source_episode_id"]
            if identity in seen_ids or p in seen_paths or entry["split"] not in ("train", "validation", "test"):
                raise ValueError("Duplicate source episode/path or invalid split; never split a recording's frames")
            seen_ids.add(identity); seen_paths.add(p)
            if entry["split"] != split:
                continue
            if outcome != 'all' and root.attrs.get('success') is not (outcome == 'successful'):
                continue
            # Full image continuity is checked at conversion / explicit validation.
            # Training validates metadata without decompressing every RGB frame.
            validate(root, check_image_pixels=False)
            digest.update(json.dumps(dict(root.attrs), sort_keys=True).encode())
            if observation_profile == VISION:
                # Source MCAP hashes do not identify a subsequently edited Zarr.
                # Hash stored RGB chunks/array metadata without decompressing them.
                # task56 deliberately does not read these bytes.
                prefix = root.path + '/' if root.path else ''
                image_prefixes = tuple(f'{prefix}{p}/images/{c}/' for p in ('obs', 'next_obs') for c in CAMERAS)
                for key in sorted(k for k in root.store if k.startswith(image_prefixes)):
                    digest.update(key.encode())
                    digest.update(root.store[key])
            keys = ("obs/state", "next_obs/state", "action", "terminated", "truncated", "timestamp", "next_timestamp", "segment_id")
            data = {k: root[k][:] for k in keys}
            if observation_profile == VISION:
                for p in ('obs', 'next_obs'):
                    data[f'{p}/proprio'] = proprioception(root.attrs['joint_names'],
                        root[f'{p}/joint_position'][:], root[f'{p}/joint_velocity'][:])
            for k, v in data.items():
                digest.update(k.encode()); digest.update(v.tobytes())
            for seg in np.unique(data["segment_id"]):
                idx = data["segment_id"] == seg
                view = 'proprio' if observation_profile == VISION else 'state'
                self.episodes.append({"observation": torch.tensor(data[f"obs/{view}"][idx], dtype=torch.float32),
                                      "next_observation": torch.tensor(data[f"next_obs/{view}"][idx], dtype=torch.float32),
                                      "action": ActionNormalizer().normalize(data["action"][idx]),
                                      "terminated": data["terminated"][idx], "truncated": data["truncated"][idx],
                                      "duration": (data["next_timestamp"]-data["timestamp"])[idx]})
                if observation_profile == VISION:
                    # Privileged state is a reward/evaluation sidecar, never observation.
                    self.episodes[-1]['evaluation_state'] = data['next_obs/state'][idx]
                    self.image_rows.extend((len(self.roots), int(i)) for i in np.flatnonzero(idx))
            self.roots.append(root)
            self.ids.append(identity)
        if not self.episodes:
            raise ValueError(f"No {split} episodes")
        self.fingerprint = digest.hexdigest()

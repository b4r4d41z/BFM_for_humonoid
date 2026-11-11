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
from .policy_time import continuity_reason

CAMERAS = ("head", "left_wrist", "right_wrist")
SOURCES = ("state", "joints", *CAMERAS)


def validate(root, *, check_image_pixels=True):
    m = dict(root.attrs)
    if m.get("schema_name") != SCHEMA or m.get("schema_version") not in (VERSION, "4.0.0"):
        raise ValueError("Requires Sharpa schema 3.0.0; legacy data cannot be reinterpreted")
    from bfm.contracts.sharpa import contract_for, execution_configuration
    version=m.get('schema_version');v4=version=='4.0.0'
    expected = contract_for(version).to_dict()
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
    if m['source_metadata'].get('bfm', {}).get('contract_version') != (4 if v4 else 3):
        raise ValueError('Collection v3 source metadata required; collection-v2 data cannot be relabelled')
    t = len(root["action"])
    shapes = {"action": (t, 26), "action_header_ns": (t,), "timestamp": (t,), "next_timestamp": (t,), "record_ns": (t,), "next_record_ns": (t,),
              "terminated": (t,), "truncated": (t,), "segment_id": (t,)}
    simulation = m['source_metadata']['bfm'].get('policy_clock', {}).get('domain') == 'isaac_simulation'
    if simulation:
        shapes['next_action_header_ns'] = (t,)
    if v4:
        execution_configuration(m['execution'])
        if not m.get('execution_checks') or any(c.get('arm_updates_checked',0)<1 or c.get('controller_config_sha256')!=m['execution']['controller_config_sha256'] for c in m['execution_checks']):raise ValueError('Missing verified executor provenance')
        if m['execution'] != m['source_metadata']['bfm']['execution']:raise ValueError('Executor identity mismatch')
        for prefix in ('','next_'):
            for key in ('acceptance_ns','acceptance_record_ns','action_id','producer_session_index'):shapes[prefix+key]=(t,)
        shapes['duration']=(t,)
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
        if v4 and k.endswith("action_id"):dtype=np.uint64
        if v4 and k.endswith("producer_session_index"):dtype=np.int64
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
    if (np.diff(r) <= 0).any() or (nr <= r).any():
        raise ValueError('Duplicate/backward action record timestamps')
    if simulation:
        timing = m.get('policy_timing', {})
        period = 1e9/15 if v4 else m['source_metadata']['bfm']['policy_clock']['period_ns']
        if (timing.get('domain') != 'action_header_simulation_ns' or timing.get('period_ns') != period
                or type(timing.get('tolerance_ns')) is not int or not 0 <= timing['tolerance_ns'] < period / 2):
            raise ValueError('Invalid stored simulation policy timing')
        p, np_ = root['action_header_ns'][:], root['next_action_header_ns'][:]
        if any(continuity_reason(a, b, timing) for a, b in zip(p, np_)):
            raise ValueError('Invalid simulation policy adjacency')
        if v4:
            from .policy_time import policy_timing
            if timing!=policy_timing(m['source_metadata']['bfm']):raise ValueError('v4 timing metadata mismatch')
            for prefix in ('','next_'):
                ids=root[prefix+'action_id'][:];session=root[prefix+'producer_session_index'][:]
                if (ids==0).any() or (session<0).any() or (session>=len(m['producer_sessions'])).any():raise ValueError('Invalid accepted action identity')
                if (root[prefix+'acceptance_ns'][:]<root[prefix+'action_header_ns'][:]).any():raise ValueError('Acceptance precedes generation')
                obs='obs' if not prefix else 'next_obs'
                if (root[f'{obs}/source_record_ns'][:]>root[prefix+'acceptance_record_ns'][:][:,None]).any():raise ValueError('Future acceptance observation')
            if (root['producer_session_index'][:]!=root['next_producer_session_index'][:]).any() or (root['next_action_id'][:]<=root['action_id'][:]).any():raise ValueError('Invalid v4 transition identity')
            p,np_=root['acceptance_ns'][:],root['next_acceptance_ns'][:]
            if any(continuity_reason(a,b,timing) for a,b in zip(p,np_)):raise ValueError('Invalid acceptance adjacency')
            if not np.allclose(root['duration'][:],(np_-p)/1e9,atol=1e-12,rtol=0):raise ValueError('Invalid v4 duration')
        origin = m['policy_origin_ns']
    else:
        p, np_, origin = r, nr, m['record_origin_ns']
        if ((nr-r) < 50_000_000).any() or ((nr-r) > 150_000_000).any():
            raise ValueError('Invalid legacy wall policy adjacency')
    if not np.allclose(root['timestamp'][:], (p-origin)/1e9, atol=1e-9, rtol=0) or not np.allclose(root['next_timestamp'][:], (np_-origin)/1e9, atol=1e-9, rtol=0):
        raise ValueError('Relative times differ from policy timestamps')
    term, trunc, seg = root["terminated"][:], root["truncated"][:], root["segment_id"][:]
    # No task terminal event is available in this source version.
    ends = np.r_[seg[1:] != seg[:-1], True]
    if term.any() or not np.array_equal(trunc, ends) or seg[0] != 0 or (np.diff(seg) < 0).any() or (np.diff(seg) > 1).any():
        raise ValueError("Recording/gap boundaries must be truncations, never task-success terminals")
    continuous = ~ends[:-1]
    if not np.array_equal(nr[:-1][continuous], r[1:][continuous]):
        raise ValueError("Transition jumped across a missing tick")
    if simulation and not np.array_equal(np_[:-1][continuous], p[1:][continuous]):
        raise ValueError('Simulation policy timestamps are discontinuous')
    if (nr[:-1][~continuous] >= r[1:][~continuous]).any():
        raise ValueError("Segments overlap")
    if v4:
        for key in ('action_id','producer_session_index','acceptance_ns','acceptance_record_ns'):
            if not np.array_equal(root['next_'+key][:-1][continuous],root[key][1:][continuous]):raise ValueError('v4 execution identity continuity failed')
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
        self.image_rows, self.roots, self.episode_fingerprints = [], [], []
        path = Path(manifest)
        self.manifest = json.loads(path.read_text())
        if self.manifest.get("schema_version") not in (VERSION,"4.0.0") or not self.manifest.get("dataset_id"):
            raise ValueError("Requires explicit Sharpa v3 manifest")
        from bfm.contracts.sharpa import contract_for
        version=self.manifest['schema_version']
        self.contract, self.ids, self.episodes = contract_for(version), [], []
        self.schema_version, self.state_fields = version, list(FIELDS)
        digest = hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode())
        seen_ids, seen_paths = set(), set()
        for entry in self.manifest["episodes"]:
            p = (path.parent / entry["path"]).resolve()
            root = zarr.open_group(str(p), mode="r")
            if root.attrs['schema_version']!=version:raise ValueError('Mixed/relabeled Sharpa schemas')
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
            episode_digest = hashlib.sha256()
            def update(value):
                digest.update(value)
                episode_digest.update(value)
            update(json.dumps(dict(root.attrs), sort_keys=True).encode())
            if observation_profile == VISION:
                # Source MCAP hashes do not identify a subsequently edited Zarr.
                # Hash stored RGB chunks/array metadata without decompressing them.
                # task56 deliberately does not read these bytes.
                prefix = root.path + '/' if root.path else ''
                image_prefixes = tuple(f'{prefix}{p}/images/{c}/' for p in ('obs', 'next_obs') for c in CAMERAS)
                for key in sorted(k for k in root.store if k.startswith(image_prefixes)):
                    update(key.encode())
                    update(root.store[key])
            keys = ("obs/state", "next_obs/state", "action", "terminated", "truncated", "timestamp", "next_timestamp", "segment_id")
            if version=='4.0.0':keys += tuple(k for k in root.array_keys() if k not in keys)
            data = {k: root[k][:] for k in keys}
            if observation_profile == VISION:
                for p in ('obs', 'next_obs'):
                    data[f'{p}/proprio'] = proprioception(root.attrs['joint_names'],
                        root[f'{p}/joint_position'][:], root[f'{p}/joint_velocity'][:])
            for k, v in data.items():
                update(k.encode()); update(v.tobytes())
            self.episode_fingerprints.append(episode_digest.hexdigest())
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

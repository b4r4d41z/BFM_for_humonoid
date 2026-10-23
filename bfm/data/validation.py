import numpy as np
import torch

from bfm.contracts.kuavo import BFMActionAdapter, STATE_FIELDS, KuavoContract
from .schema import SCHEMA_NAME, SCHEMA_VERSION


def validate_episode(arrays, metadata):
    if (metadata.get("schema_name"), metadata.get("schema_version")) != (SCHEMA_NAME, SCHEMA_VERSION):
        raise ValueError("Requires measured BFM schema 2.0.0; legacy command-proxy trajectories cannot be converted to measurements")
    if metadata.get("state_source") != "measured_simulator" or metadata.get("coordinate_frame") != "environment_local":
        raise ValueError("Measured simulator state in environment_local coordinates is required")
    if metadata.get("terminal_semantics") != "terminated_no_bootstrap_truncated_bootstrap_pre_reset":
        raise ValueError("Unsupported terminal semantics")
    for k in ("episode_id", "provenance", "initial_scene"):
        if not metadata.get(k):
            raise ValueError(f"Missing episode metadata: {k}")
    if metadata.get("split") not in ("train", "validation", "test"):
        raise ValueError("Episode must declare a train/validation/test split")
    c = KuavoContract(**metadata["contract"])
    fields = metadata.get("state_fields", {})
    for k, (w, u) in STATE_FIELDS.items():
        if fields.get(k) != {"width": w, "unit": u}:
            raise ValueError(f"Invalid measured field contract: {k}")
    extras = set(fields) - set(STATE_FIELDS)
    if extras:
        names = metadata.get("all_joint_names", [])
        if not names or len(set(names)) != len(names) or not set(c.arm_joint_names + c.finger_joint_names) <= set(names):
            raise ValueError("Rich joint observations require complete unique all_joint_names")
        for k in extras:
            if k not in ("all_joint_q", "all_joint_dq") or fields[k] != {"width": len(names), "unit": "rad" if k.endswith('_q') else "rad/s"}:
                raise ValueError(f"Unsupported rich observation field: {k}")
    if metadata.get("action_units") != ["rad"] * 7 + ["binary_closed_signed"]:
        raise ValueError("Invalid physical action units")
    required = ["action", "isaac_action", "terminated", "truncated", "timestamp", "next_timestamp", "step_index", "next_step_index"]
    required += [f"{p}/{k}" for p in ("obs", "next_obs") for k in fields]
    if set(arrays) != set(required):
        raise ValueError(f"Episode array mismatch: missing={set(required)-set(arrays)}, unexpected={set(arrays)-set(required)}")
    n = len(arrays["action"])
    if n < 1:
        raise ValueError("Empty episode")
    for name, value in arrays.items():
        a = np.asarray(value)
        width = fields[name.split('/')[1]]["width"] if '/' in name else {"action": 8, "isaac_action": 26}.get(name)
        shape = (n, width) if width is not None else (n,)
        if a.shape != shape or a.dtype.kind not in "fiub" or not np.isfinite(a).all():
            raise ValueError(f"{name}: expected finite numeric {shape}, got {a.shape}")
    term, trunc = arrays["terminated"], arrays["truncated"]
    if term.dtype != np.bool_ or trunc.dtype != np.bool_ or np.any(term & trunc):
        raise ValueError("Terminal flags must be boolean and mutually exclusive")
    if np.any((term | trunc)[:-1]) or not (term | trunc)[-1]:
        raise ValueError("One complete episode per store; boundary only at final transition")
    t, nt = arrays["timestamp"], arrays["next_timestamp"]
    step, ns = arrays["step_index"], arrays["next_step_index"]
    if step.dtype.kind not in "iu" or ns.dtype.kind not in "iu" or np.any(step < 0):
        raise ValueError("Control-step indices must be nonnegative integers")
    span = ns - step
    max_span = round(c.policy_dt / c.control_dt)
    if np.any(span <= 0) or np.any(span > max_span) or np.any(span[:-1] != max_span):
        raise ValueError("Actions must span a policy period (final terminal/truncated step may be shorter)")
    if not np.allclose(nt - t, span * c.control_dt, atol=1e-8, rtol=1e-6):
        raise ValueError("Timestamp/control-step duration mismatch")
    if np.any(t < 0) or not np.allclose(t, step * c.control_dt, atol=1e-8, rtol=1e-6):
        raise ValueError("Episode timestamps must match control-step indices")
    if not np.array_equal(ns[:-1], step[1:]) or not np.allclose(nt[:-1], t[1:], atol=1e-8):
        raise ValueError("Noncontiguous episode timing")
    for k in fields:
        if not np.allclose(arrays[f"next_obs/{k}"][:-1], arrays[f"obs/{k}"][1:], atol=1e-6, rtol=1e-6):
            raise ValueError(f"Inconsistent next_observation / reset crossing: {k}")
    adapter = BFMActionAdapter(c)
    normalized = adapter.normalize(arrays["action"])
    if not torch.allclose(adapter(normalized), torch.as_tensor(arrays["isaac_action"], dtype=torch.float32), atol=1e-5, rtol=1e-5):
        raise ValueError("Executed Isaac action disagrees with joint ordering, left hold or claw command")
    return {"transitions": n, "state_dim": sum(w for w, _ in STATE_FIELDS.values()), "action_dim": 8}

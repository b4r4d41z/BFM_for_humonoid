"""Collection v3 bounded commands; legacy kuavo.py is explicitly unrelated."""
from dataclasses import dataclass
import numpy as np
import torch

SOURCE_COMMIT = "08162e1da037a2bb1827cad33e71c3e7a183de53"
MODEL_SHA256 = "4ac4549699cd3de774b676d2acaf9d0f35f1c2b041587ed7f44b535f80e43dcf"
SCHEMA = "bfm.sharpa.recorded"
VERSION = "3.0.0"
PROFILE = "task56"
GROUPS = {"left_arm": slice(0, 7), "left_hand": slice(7, 13),
          "right_arm": slice(13, 20), "right_hand": slice(20, 26)}
JOINTS = {side: tuple(f"zarm_{side[0]}{i}_joint" for i in range(1, 8)) for side in ("left", "right")}
FIELDS = {"left_eef_position": (3, "m"), "right_eef_position": (3, "m"),
          "left_arm_position": (7, "rad"), "right_arm_position": (7, "rad"),
          "left_arm_velocity": (7, "rad/s"), "right_arm_velocity": (7, "rad/s"),
          "object_position": (3, "m"), "object_velocity": (3, "m/s"),
          "goal_center": (3, "m"), "goal_half_extents": (3, "m"),
          "goal_orientation": (4, "xyzw"), "goal_linear_velocity": (3, "m/s"),
          "goal_angular_velocity": (3, "rad/s")}
SLICES, offset = {}, 0
for name, (width, _) in FIELDS.items():
    SLICES[name] = slice(offset, offset + width)
    offset += width
assert offset == 56
# Frozen radians from assembly/s40_sharpa_two_cameras.xml at SOURCE_COMMIT.
ARM_LOW = [-2.61799, -.349066, -1.5708, -1.5708, -1.5708, -1.5708, -1.5708]
ARM_HIGH = [.523599, 2.0944, 1.5708, 0., 1.5708, 1.5708, 1.5708]
LOW = ARM_LOW + [0.] * 6 + [ARM_LOW[0], -2.0944] + ARM_LOW[2:] + [0.] * 6
HIGH = ARM_HIGH + [100.] * 6 + [ARM_HIGH[0], .349066] + ARM_HIGH[2:] + [100.] * 6


def flatten_state(message):
    values = []
    for key, (n, _) in FIELDS.items():
        v = message[key] if isinstance(message, dict) else getattr(message, key)
        a = np.asarray(v, dtype=np.float64)
        if a.shape != (n,) or not np.isfinite(a).all():
            raise ValueError(f"Invalid measured state field {key}")
        values.extend(a)
    result = np.asarray(values)
    validate_states(result[None])
    return result


def unflatten_state(state):
    a = np.asarray(state)
    if a.shape != (56,):
        raise ValueError("task56 requires exactly 56 values")
    return {k: a[s].copy() for k, s in SLICES.items()}


def validate_states(states):
    a = np.asarray(states)
    if a.ndim != 2 or a.shape[1] != 56 or not np.isfinite(a).all():
        raise ValueError("Expected finite measured [N,56] task56 states")
    if (a[:, SLICES["goal_half_extents"]] <= 0).any():
        raise ValueError("Goal half extents must be positive")
    if not np.allclose(np.linalg.norm(a[:, SLICES["goal_orientation"]], axis=1), 1., atol=1e-5):
        raise ValueError("Goal orientation must be unit xyzw")


def flatten_action(message):
    for side in JOINTS:
        if tuple(getattr(message, f"{side}_arm_joint_names")) != JOINTS[side]:
            raise ValueError("BfmAction arm names/order mismatch")
    a = np.concatenate([np.asarray(getattr(message, f"{k}_target")) for k in GROUPS])
    ActionNormalizer().normalize(a)
    return a.astype(np.float64)


class ActionNormalizer:
    """One invertible physical scaling; never fitted to demonstrations."""
    def __init__(self, low=LOW, high=HIGH):
        self.low, self.high = torch.tensor(low, dtype=torch.float64), torch.tensor(high, dtype=torch.float64)
        if self.low.shape != (26,) or self.high.shape != (26,) or not torch.isfinite(self.low).all() or not torch.isfinite(self.high).all() or not (self.high > self.low).all():
            raise ValueError("Invalid action bounds")

    def normalize(self, physical):
        a = torch.as_tensor(physical, dtype=torch.float64)
        low, high = self.low.to(a.device), self.high.to(a.device)
        if a.shape[-1] != 26 or not torch.isfinite(a).all() or (a < low - 1e-7).any() or (a > high + 1e-7).any():
            raise ValueError("Physical action outside frozen arm radians / hand 0..100 bounds")
        return (2 * (a - low) / (high-low) - 1).clamp(-1, 1).float()

    def physical(self, normalized):
        a = torch.as_tensor(normalized, dtype=torch.float64)
        if a.shape[-1] != 26 or not torch.isfinite(a).all() or (a.abs() > 1.000001).any():
            raise ValueError("Model actions must be finite [...,26] in [-1,1]")
        return self.low.to(a.device) + (a.clamp(-1, 1)+1)/2 * (self.high-self.low).to(a.device)


@dataclass
class SharpaContract:
    policy_dt: float = .1

    def to_dict(self):
        return {"version": 3, "profile": PROFILE, "policy_dt": self.policy_dt,
                "arm_semantics": "bounded_absolute_joint_position_target_post_clamp",
                "groups": list(GROUPS), "joints": {k: list(v) for k, v in JOINTS.items()},
                "low": LOW, "high": HIGH, "bounds_source_commit": SOURCE_COMMIT,
                "bounds_source_path": "models/robots/s40_sharpa/assembly/s40_sharpa_two_cameras.xml",
                "bounds_source_sha256": MODEL_SHA256,
                "action_units": ["rad"]*7 + ["controller_0_100"]*6 + ["rad"]*7 + ["controller_0_100"]*6}

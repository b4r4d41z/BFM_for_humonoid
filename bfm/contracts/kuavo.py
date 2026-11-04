"""Measured KUAVO contract. Positions are in metres in the environment-local frame."""
from dataclasses import asdict, dataclass, field
import math

import numpy as np
import torch

RIGHT_ARM = tuple(f"zarm_r{i}_joint" for i in range(1, 8))
LEFT_ARM = tuple(f"zarm_l{i}_joint" for i in range(1, 8))
RIGHT_FINGERS = ("zarm_r_finger_left_joint", "zarm_r_finger_right_joint")
ARM_JOINTS = RIGHT_ARM + LEFT_ARM
OPEN_HAND = (0., 100., 0., 0., 0., 0.)
CLOSED_HAND = (69., 99., 42., 44., 61., 60.)
STATE_FIELDS = {
    "right_arm_q": (7, "rad"), "right_arm_dq": (7, "rad/s"),
    "right_gripper_q": (2, "rad"), "right_eef_position": (3, "m"),
    "red_ball_position": (3, "m"), "red_ball_velocity": (3, "m/s"),
    "container_position": (3, "m"),
}
STATE_DIM = sum(width for width, _ in STATE_FIELDS.values())
ACTION_DIM = 8


def ordered_indices(available, requested):
    if len(set(available)) != len(available) or any(available.count(n) != 1 for n in requested):
        raise ValueError(f"Missing/duplicate joint or body names: required={requested}, available={available}")
    return [available.index(n) for n in requested]


@dataclass
class KuavoContract:
    """Bounds and left hold are explicit physical joint targets, captured from Isaac."""
    right_low: list[float]
    right_high: list[float]
    left_hold: list[float]
    left_low: list[float]
    left_high: list[float]
    eef_body: str
    eef_offset: list[float]
    control_dt: float = 1 / 60
    policy_dt: float = 1 / 10
    arm_joint_names: tuple[str, ...] = ARM_JOINTS
    finger_joint_names: tuple[str, ...] = RIGHT_FINGERS
    left_gripper_closed: bool = False
    container_target_offset: list[float] = field(default_factory=lambda: [0., 0., 0.0475])

    def __post_init__(self):
        self.arm_joint_names = tuple(self.arm_joint_names)
        self.finger_joint_names = tuple(self.finger_joint_names)
        if self.arm_joint_names != ARM_JOINTS or self.finger_joint_names != RIGHT_FINGERS:
            raise ValueError("Unsupported KUAVO joint ordering")
        for name in ("right_low", "right_high", "left_hold", "left_low", "left_high"):
            value = np.asarray(getattr(self, name))
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain seven finite radians")
        for side in ("right", "left"):
            if np.any(np.asarray(getattr(self, side + "_low")) >= np.asarray(getattr(self, side + "_high"))):
                raise ValueError(f"Invalid {side} joint limits")
        if np.any(np.asarray(self.left_hold) < self.left_low) or np.any(np.asarray(self.left_hold) > self.left_high):
            raise ValueError("Left-arm hold is outside limits")
        if not self.eef_body or np.asarray(self.eef_offset).shape != (3,) or not np.isfinite(self.eef_offset).all():
            raise ValueError("An explicit end-effector body and local offset are required")
        if np.asarray(self.container_target_offset).shape != (3,) or not np.isfinite(self.container_target_offset).all():
            raise ValueError("Invalid container target offset")
        if not (math.isfinite(self.control_dt) and math.isfinite(self.policy_dt) and 0 < self.control_dt <= self.policy_dt):
            raise ValueError("Invalid control/policy period")
        if not math.isclose(self.policy_dt / self.control_dt, round(self.policy_dt / self.control_dt), abs_tol=1e-6):
            raise ValueError("Policy period must be an integer number of control steps")

    def to_dict(self):
        # JSON round-trip canonicalizes tuples for checkpoint/manifest comparisons.
        import json
        return json.loads(json.dumps(asdict(self)))


class BFMStateAdapter:
    dim = STATE_DIM
    fields = STATE_FIELDS

    def __call__(self, state):
        values = []
        prefix = None
        for name, (width, _) in self.fields.items():
            if name not in state:
                raise ValueError(f"Missing measured state field: {name}")
            v = torch.as_tensor(state[name], dtype=torch.float32)
            if v.ndim < 1 or v.shape[-1] != width or not torch.isfinite(v).all():
                raise ValueError(f"{name} must be finite [...,{width}]")
            if prefix is not None and v.shape[:-1] != prefix:
                raise ValueError("Inconsistent measured-state batch shapes")
            prefix = v.shape[:-1]
            values.append(v)
        return torch.cat(values, dim=-1)


class BFMActionAdapter:
    """[-1,1]^8 -> absolute arm radians + two existing six-value hand prototypes.

    The current environment has binary claws: scalar >= 0 closes the right claw.
    Record the executed scalar as +/-1, never as an unexecuted analogue command.
    """
    def __init__(self, contract: KuavoContract, env_joint_names=ARM_JOINTS, device="cpu"):
        self.contract, self.device = contract, torch.device(device)
        self.names = list(env_joint_names)
        if set(self.names) != set(ARM_JOINTS) or len(self.names) != 14:
            raise ValueError("Isaac arm interface must contain exactly the 14 named KUAVO joints")
        self.right_indices = ordered_indices(self.names, RIGHT_ARM)
        self.left_indices = ordered_indices(self.names, LEFT_ARM)
        self.low = torch.tensor(contract.right_low, device=self.device)
        self.high = torch.tensor(contract.right_high, device=self.device)

    def to_physical(self, action):
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if a.ndim != 2 or a.shape[-1] != 8 or not torch.isfinite(a).all() or (a.abs() > 1.000001).any():
            raise ValueError("Normalized actions must be finite [N,8] in [-1,1]")
        q = self.low + (a[:, :7].clamp(-1, 1) + 1) * (self.high - self.low) / 2
        return torch.cat((q, torch.where(a[:, 7:] >= 0, 1., -1.)), -1)

    def normalize(self, physical):
        p = torch.as_tensor(physical, dtype=torch.float32, device=self.device)
        if p.ndim != 2 or p.shape[-1] != 8 or not torch.isfinite(p).all():
            raise ValueError("Executed physical actions must be finite [N,8]")
        q = 2 * (p[:, :7] - self.low) / (self.high - self.low) - 1
        if (q.abs() > 1.00001).any() or not ((p[:, 7] == -1) | (p[:, 7] == 1)).all():
            raise ValueError("Executed action outside limits or non-binary claw command")
        return torch.cat((q.clamp(-1, 1), p[:, 7:]), -1)

    def __call__(self, normalized):
        physical = self.to_physical(normalized)
        out = torch.empty((len(physical), 26), device=self.device)
        out[:, self.right_indices] = physical[:, :7]
        out[:, self.left_indices] = torch.tensor(self.contract.left_hold, device=self.device)
        open_hand = torch.tensor(OPEN_HAND, device=self.device)
        closed_hand = torch.tensor(CLOSED_HAND, device=self.device)
        out[:, 14:20] = closed_hand if self.contract.left_gripper_closed else open_hand
        out[:, 20:26] = torch.where(physical[:, 7:] > 0, closed_hand, open_hand)
        return out

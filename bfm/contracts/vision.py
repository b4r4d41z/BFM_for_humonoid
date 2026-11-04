"""Explicit measured proprioception and camera identity for vision_proprio_v1."""
import numpy as np
from .sharpa import JOINTS

PROFILE = "vision_proprio_v1"
CAMERAS = ("head", "left_wrist", "right_wrist")
# All 22 physical DOFs per hand from the frozen Sharpa XML, including abduction.
FINGERS = ("thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
           "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
           "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
           "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
           "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP")
PROPRIO_JOINTS = JOINTS['left'] + JOINTS['right'] + tuple(
    f"{side}_{finger}" for side in ('left', 'right') for finger in FINGERS)
PROPRIO_DIM = 2 * len(PROPRIO_JOINTS)  # q[58], dq[58]; no task56 fields.


def proprio_indices(names):
    names = list(names)
    missing = set(PROPRIO_JOINTS) - set(names)
    if len(set(names)) != len(names) or missing:
        raise ValueError(f"vision_proprio_v1 requires unique measured arm/hand joints; missing {sorted(missing)}")
    return [names.index(name) for name in PROPRIO_JOINTS]


def proprioception(names, position, velocity):
    idx = proprio_indices(names)
    q, dq = np.asarray(position), np.asarray(velocity)
    if q.shape != dq.shape or q.shape[-1] != len(names) or not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError("Measured q/dq must be finite with one value per joint")
    return np.concatenate((q[..., idx], dq[..., idx]), axis=-1).astype(np.float32)

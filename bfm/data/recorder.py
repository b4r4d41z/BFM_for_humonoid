"""Reference single-environment recorder for measured policy-rate transitions."""
import numpy as np
import torch

from bfm.contracts.kuavo import STATE_FIELDS
from .schema import episode_metadata
from .zarr_io import write_episode


def cpu_state(state):
    result = {}
    for k, v in state.items():
        a = torch.as_tensor(v).detach().cpu().numpy()
        if a.ndim != 2 or a.shape[0] != 1:
            raise ValueError("Reference recorder requires exactly one environment")
        result[k] = a[0].copy()
    return result


class BFMRecorder:
    def __init__(self, contract, *, episode_id, split, provenance, initial_scene, joint_names):
        self.contract = contract
        self.metadata = episode_metadata(contract, episode_id=episode_id, split=split, provenance=provenance, initial_scene=initial_scene)
        self.metadata["all_joint_names"] = list(joint_names)
        self.transitions = []
        self.next_step = 0

    def append(self, state, physical_action, isaac_action, next_state, *, terminated, truncated, control_steps):
        if self.transitions and (self.transitions[-1]["terminated"] or self.transitions[-1]["truncated"]):
            raise ValueError("Episode already ended; create a new recorder after reset")
        s, ns = cpu_state(state), cpu_state(next_state)
        if set(s) != set(ns):
            raise ValueError("Measured observation fields changed within a transition")
        for k, v in s.items():
            if k not in STATE_FIELDS:
                if k not in ("all_joint_q", "all_joint_dq") or len(v) != len(self.metadata["all_joint_names"]):
                    raise ValueError(f"Unsupported rich measured field: {k}")
                self.metadata["state_fields"][k] = {"width": len(v), "unit": "rad" if k.endswith('_q') else "rad/s"}
        row = {f"{prefix}/{k}": v for prefix, values in (("obs", s), ("next_obs", ns)) for k, v in values.items()}
        row.update(action=torch.as_tensor(physical_action).detach().cpu().numpy().reshape(-1),
                   isaac_action=torch.as_tensor(isaac_action).detach().cpu().numpy().reshape(-1),
                   terminated=bool(terminated), truncated=bool(truncated),
                   step_index=self.next_step, next_step_index=self.next_step + control_steps,
                   timestamp=self.next_step * self.contract.control_dt,
                   next_timestamp=(self.next_step + control_steps) * self.contract.control_dt)
        self.transitions.append(row)
        self.next_step += control_steps

    def save(self, path):
        if not self.transitions:
            raise ValueError("No measured transitions recorded")
        arrays = {k: np.stack([row[k] for row in self.transitions]) for k in self.transitions[0]}
        return write_episode(path, arrays, self.metadata)

import json
import numpy as np
import pytest

from bfm.contracts.kuavo import KuavoContract, BFMActionAdapter, STATE_FIELDS
from bfm.data.schema import episode_metadata, SCHEMA_VERSION
from bfm.data.zarr_io import write_episode


@pytest.fixture
def contract():
    # Synthetic mechanics only; these are not claimed to be KUAVO hardware limits.
    return KuavoContract([-1.] * 7, [1.] * 7, [0.] * 7, [-1.] * 7, [1.] * 7,
                         "zarm_r7_link", [0., 0., 0.])


def make_episode(contract, count=12, episode_id="toy0", split="train"):
    """Toy first-order joint dynamics, explicit applied targets and measured next state."""
    rng = np.random.default_rng(12 + sum(map(ord, episode_id)))
    q = np.zeros(7, dtype=np.float32)
    states, actions = [], []
    dq = np.zeros(7, dtype=np.float32)
    for t in range(count + 1):
        states.append({
            "right_arm_q": q.copy(), "right_arm_dq": dq.copy(), "right_gripper_q": np.zeros(2),
            "right_eef_position": q[:3].copy(), "red_ball_position": np.array([0.4, -0.1, 0.2]),
            "red_ball_velocity": np.zeros(3), "container_position": np.array([0.5, 0.1, 0.2]),
        })
        action = np.r_[rng.uniform(-0.8, 0.8, 7), -1.].astype(np.float32)
        actions.append(action)
        dq = (action[:7] - q) * 0.5
        q = q + dq * contract.policy_dt
    arrays = {f"{p}/{k}": np.stack([s[k] for s in values]).astype(np.float32)
              for p, values in (("obs", states[:-1]), ("next_obs", states[1:])) for k in STATE_FIELDS}
    a = np.stack(actions[:-1])
    adapter = BFMActionAdapter(contract)
    arrays.update(action=a, isaac_action=adapter(adapter.normalize(a)).numpy(),
                  terminated=np.zeros(count, dtype=bool), truncated=np.arange(count) == count - 1,
                  timestamp=np.arange(count) * contract.policy_dt,
                  next_timestamp=np.arange(1, count + 1) * contract.policy_dt,
                  step_index=np.arange(count) * 6, next_step_index=np.arange(1, count + 1) * 6)
    meta = episode_metadata(contract, episode_id=episode_id, split=split,
                            provenance={"kind": "synthetic_unit_test", "dynamics": "dq=0.5*(target-q)"},
                            initial_scene={"note": "toy, not robot physics"})
    return arrays, meta


@pytest.fixture
def episode(contract):
    return make_episode(contract)


@pytest.fixture
def manifest(tmp_path, contract):
    entries = []
    for i, split in enumerate(["train", "train", "test"]):
        a, m = make_episode(contract, episode_id=f"toy{i}", split=split)
        name = f"episode{i}.zarr"
        write_episode(tmp_path / name, a, m)
        entries.append({"path": name, "split": split})
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "dataset_id": "toy", "episodes": entries}))
    return p

import copy
import numpy as np
import pytest
import torch

from bfm.contracts.kuavo import BFMActionAdapter, BFMStateAdapter, ARM_JOINTS
from bfm.data.zarr_io import read_episode, write_episode
from bfm.data.validation import validate_episode
from bfm.data.dataset import BFMDataset
from bfm.data.replay_buffer import BFMReplayBuffer
from bfm.data.normalization import StateNormalizer


def test_zarr_roundtrip_and_protect_existing(tmp_path, episode):
    a, m = episode
    p = tmp_path / "episode.zarr"
    assert write_episode(p, a, m)["state_dim"] == 28
    b, n = read_episode(p)
    assert m == n
    assert all(np.array_equal(a[k], b[k]) for k in a)
    with pytest.raises(FileExistsError):
        write_episode(p, a, m)
    invalid = copy.deepcopy(a)
    invalid["next_obs/right_arm_q"][0] = 100
    with pytest.raises(ValueError):
        write_episode(p, invalid, m, overwrite=True)
    assert np.array_equal(read_episode(p)[0]["action"], a["action"])


@pytest.mark.parametrize("defect", ["version", "joint_order", "nan", "dimensions", "bounds", "boundary", "next_obs", "missing", "terminal", "time", "proxy"])
def test_strict_contract_rejection(episode, defect):
    a, m = copy.deepcopy(episode)
    if defect == "version": m["schema_version"] = "1.0.0"
    if defect == "joint_order": m["contract"]["arm_joint_names"].reverse()
    if defect == "nan": a["obs/right_arm_q"][0, 0] = np.nan
    if defect == "dimensions": a["action"] = a["action"][:, :7]
    if defect == "bounds": a["action"][0, 0] = 100
    if defect == "boundary": a["truncated"][0] = True
    if defect == "next_obs": a["next_obs/right_arm_q"][0] += 1
    if defect == "missing": del a["obs/red_ball_velocity"]
    if defect == "terminal": a["terminated"][-1] = True
    if defect == "time": a["next_timestamp"][0] += 1
    if defect == "proxy": m["state_source"] = "recorded_command_proxy"
    with pytest.raises(ValueError): validate_episode(a, m)


def test_replay_reproducibility_and_sequences(manifest):
    data = BFMDataset(manifest)
    assert len(data.episodes) == 2  # held-out episode is not training/reference data
    a, b = [BFMReplayBuffer(data.episodes, seed=7) for _ in range(2)]
    sa, sb = a.sample_transitions(64), b.sample_transitions(64)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert sa["observation"].shape == (64, 28)
    assert sa["action"].shape == (64, 8)
    seq = a.sample_sequences(64, 5)
    assert torch.equal(seq["next_observation"][:, :-1], seq["observation"][:, 1:])
    assert not seq["truncated"][:, :-1].any()
    with pytest.raises(ValueError): a.sample_sequences(1, 999)
    assert torch.equal(a.reference_states(10, 3)[0], a.reference_states(10, 3)[0])
    n = StateNormalizer(28)
    n.fit(a.data["observation"])
    assert torch.allclose(n.inverse(n(sa["observation"])), sa["observation"], atol=1e-6)


def test_named_action_mapping_and_limits(contract):
    names = list(reversed(ARM_JOINTS))
    adapter = BFMActionAdapter(contract, names)
    action = torch.tensor([[-1., -.5, 0., .2, .4, .7, 1., 1.]])
    out = adapter(action)
    assert out.shape == (1, 26)
    for i in range(7):
        assert torch.isclose(out[0, names.index(ARM_JOINTS[i])], action[0, i])
        assert out[0, names.index(ARM_JOINTS[i + 7])] == 0
    assert out[0, 20] == 69
    assert out[0, 14] == 0
    assert torch.allclose(adapter.normalize(adapter.to_physical(action)), action, atol=1e-6)
    with pytest.raises(ValueError): adapter(action * 2)
    with pytest.raises(ValueError): BFMActionAdapter(contract, ["bad"] * 14)


def test_state_no_padding_or_missing_goal(episode):
    a, _ = episode
    states = {k.split('/')[1]: v for k, v in a.items() if k.startswith('obs/')}
    assert BFMStateAdapter()(states).shape == (12, 28)
    del states["right_eef_position"]
    with pytest.raises(ValueError): BFMStateAdapter()(states)

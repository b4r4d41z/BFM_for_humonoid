from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from bc.data.schema import (
    ACTION_ARM_DIM,
    ACTION_FULL_DIM,
    ACTION_HAND_DIM,
    ARM_JOINT_NAMES,
    HAND_VALUE_NAMES,
    PATHS,
    STATE_FULL_DIM,
)
from bc.temporal import (
    LEGACY_TEMPORAL_CONTRACT_WARNING,
    find_future_target_indices,
    temporal_contract_from_checkpoint_meta,
)
from isaaclab_wrapper.policy_scheduler import PolicyScheduler


def test_policy_scheduler_10hz_on_60hz_control_and_holds_cached_actions():
    scheduler = PolicyScheduler(policy_hz=10.0, control_hz=60.0)
    calls = []

    def infer():
        calls.append(len(calls))
        return torch.full((1, 26), float(len(calls)))

    actions = []
    inferred = []
    for _ in range(60):
        action, did_infer = scheduler.tick(infer)
        actions.append(action.clone())
        inferred.append(did_infer)

    assert sum(inferred) == pytest.approx(10, abs=1)
    assert len(calls) == sum(inferred)
    assert torch.equal(actions[1], actions[0])
    assert torch.equal(actions[5], actions[0])
    assert inferred[0] is True
    assert inferred[1] is False
    assert inferred[6] is True


def test_policy_scheduler_supports_non_integer_ratios_with_accumulated_time():
    scheduler = PolicyScheduler(policy_hz=7.0, control_hz=60.0)
    call_count = 0

    def infer():
        nonlocal call_count
        call_count += 1
        return torch.full((1, 26), float(call_count))

    inferred_steps = []
    for step in range(120):
        _, did_infer = scheduler.tick(infer)
        if did_infer:
            inferred_steps.append(step)

    # Around two seconds at 7 Hz, including the mandatory initial action.
    assert len(inferred_steps) == pytest.approx(14, abs=1)
    gaps = np.diff(inferred_steps)
    assert set(gaps).issubset({8, 9})


def test_future_targets_are_timestamp_based_half_second_and_do_not_cross_done():
    timestamps = np.array([0.0, 0.1, 0.2, 0.7, 0.8, 0.9, 1.0])
    done = np.array([False, False, True, False, False, False, True])

    pairs = find_future_target_indices(timestamps, done, horizon_s=0.5)

    assert (0, 2) in pairs  # closest in first episode is t=0.2, not frame +5.
    assert all(src <= 2 and tgt <= 2 or src >= 3 and tgt >= 3 for src, tgt in pairs)
    for src, tgt in pairs:
        assert abs((timestamps[tgt] - timestamps[src]) - 0.5) <= 0.3


def _write_h5(path, timestamps, done, *, action_offset=100.0, include_metadata=True):
    h5py = pytest.importorskip("h5py")
    n = len(timestamps)
    state = np.zeros((n, STATE_FULL_DIM), dtype=np.float32)
    for i in range(n):
        state[i, :] = float(i)
    next_state = state.copy()
    for i in range(n - 1):
        if not done[i]:
            next_state[i] = state[i + 1]
    action = state + action_offset
    with h5py.File(path, "w") as f:
        f.create_dataset(PATHS.obs_state, data=state)
        f.create_dataset(PATHS.next_obs_state, data=next_state)
        f.create_dataset(PATHS.act_action, data=action)
        f.create_dataset(PATHS.act_joint_target, data=action[:, :ACTION_ARM_DIM])
        f.create_dataset(PATHS.act_hand_target, data=action[:, ACTION_ARM_DIM:])
        f.create_dataset(PATHS.done, data=np.asarray(done, dtype=np.bool_))
        f.create_dataset(PATHS.timestamps, data=np.asarray(timestamps, dtype=np.float64))
        if include_metadata:
            f.create_dataset(PATHS.meta_obs_dim, data=STATE_FULL_DIM)
            f.create_dataset(PATHS.meta_action_dim, data=ACTION_FULL_DIM)
            f.create_dataset(PATHS.meta_joint_names, data=np.asarray(ARM_JOINT_NAMES, dtype="S"))
            f.create_dataset(PATHS.meta_hand_value_names, data=np.asarray(HAND_VALUE_NAMES, dtype="S"))


def test_streaming_dataset_returns_stored_action_and_immediate_transition(tmp_path):
    pytest.importorskip("h5py")
    from bc.data.hdf5_streaming_dataset import HDF5StreamingDataset
    path = tmp_path / "demo.h5"
    timestamps = [0.0, 0.11, 0.19, 0.31, 0.49, 0.52, 0.61, 0.75]
    done = [False, False, False, False, False, False, False, True]
    _write_h5(path, timestamps, done)

    dataset = HDF5StreamingDataset([path], use_images=False)
    sample = dataset[0]

    # A horizon target for t=0 would be obs[4] == 4, but action[0] is the
    # independently stored command 100. This guards against the original bug.
    assert torch.allclose(sample["obs"]["state"]["full"], torch.zeros(STATE_FULL_DIM))
    assert torch.allclose(sample["action"]["full"], torch.full((STATE_FULL_DIM,), 100.0))
    assert not torch.allclose(sample["action"]["full"], torch.full((STATE_FULL_DIM,), 4.0))
    assert torch.allclose(sample["next_obs"]["state"]["full"], torch.ones(STATE_FULL_DIM))
    assert sample["timestamp"].item() == timestamps[0]
    assert torch.equal(
        torch.cat([sample["action"]["arm"], sample["action"]["hand"]]),
        sample["action"]["full"],
    )
    assert torch.equal(
        torch.cat([sample["obs"]["state"]["arm"], sample["obs"]["state"]["hand"]]),
        sample["obs"]["state"]["full"],
    )
    assert sample["meta"]["actual_dataset_hz"] > 0.0


def test_streaming_dataset_exposes_terminal_transitions_and_never_shifts_episode(tmp_path):
    from bc.data.hdf5_streaming_dataset import HDF5StreamingDataset

    path = tmp_path / "episodes.h5"
    _write_h5(path, [0.0, 0.1, 0.2, 0.0, 0.1], [False, False, True, False, True])
    dataset = HDF5StreamingDataset([path], use_images=False)

    assert len(dataset) == 5
    assert dataset[2]["done"].item() is True
    assert torch.allclose(dataset[3]["obs"]["state"]["full"], torch.full((26,), 3.0))
    assert torch.allclose(dataset[3]["next_obs"]["state"]["full"], torch.full((26,), 4.0))


@pytest.mark.parametrize(
    "failure", ["components", "dimension", "metadata", "metadata_order", "timestamps", "next_obs"]
)
def test_streaming_dataset_rejects_ambiguous_or_invalid_contract(tmp_path, failure):
    h5py = pytest.importorskip("h5py")
    from bc.data.hdf5_streaming_dataset import HDF5StreamingDataset

    path = tmp_path / f"bad-{failure}.h5"
    _write_h5(path, [0.0, 0.1, 0.2], [False, False, True], include_metadata=failure != "metadata")
    if failure != "metadata":
        with h5py.File(path, "r+") as f:
            if failure == "components":
                f[PATHS.act_hand_target][0, 0] += 1
            elif failure == "dimension":
                del f[PATHS.act_action]
                f.create_dataset(PATHS.act_action, data=np.zeros((3, 25), dtype=np.float32))
            elif failure == "metadata_order":
                f[PATHS.meta_joint_names][0] = b"wrong_joint"
            elif failure == "timestamps":
                f[PATHS.timestamps][1] = f[PATHS.timestamps][0]
            elif failure == "next_obs":
                f[PATHS.next_obs_state][0] = -1

    with pytest.raises(ValueError):
        HDF5StreamingDataset([path], use_images=False)


def test_temporal_sampling_rejects_non_monotonic_episode_timestamps():
    with pytest.raises(ValueError, match="strictly increasing"):
        find_future_target_indices([0.0, 0.2, 0.1], [False, False, True])


def test_legacy_checkpoint_temporal_warning_is_explicit():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        contract, legacy = temporal_contract_from_checkpoint_meta({"model_state": {}}, warn_legacy=True)

    assert legacy is True
    assert contract["policy_hz"] == 10.0
    assert any(LEGACY_TEMPORAL_CONTRACT_WARNING in str(w.message) for w in caught)

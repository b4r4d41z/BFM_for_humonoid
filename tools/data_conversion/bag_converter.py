"""Pure timestamp synchronization and canonical transition construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from bc.data.schema import ACTION_ARM_DIM, ACTION_HAND_DIM, STATE_ARM_DIM, STATE_HAND_DIM


@dataclass(frozen=True)
class Timestamped:
    timestamp: float
    value: Any


def _validated(stream: Sequence[Timestamped], name: str, dim: int | None = None) -> tuple[np.ndarray, list[Any]]:
    if not stream:
        raise ValueError(f"missing required stream: {name}")
    times = np.asarray([sample.timestamp for sample in stream], dtype=np.float64)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError(f"{name} timestamps must be finite and strictly increasing")
    values = [sample.value for sample in stream]
    if dim is not None:
        for value in values:
            array = np.asarray(value)
            if array.shape != (dim,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} values must be finite vectors of shape ({dim},)")
    return times, values


def nearest(stream: Sequence[Timestamped], timestamp: float, max_skew_s: float, name: str) -> Any:
    times, values = _validated(stream, name)
    pos = int(np.searchsorted(times, timestamp))
    candidates = [i for i in (pos - 1, pos) if 0 <= i < len(times)]
    index = min(candidates, key=lambda i: (abs(float(times[i]) - timestamp), i))
    skew = abs(float(times[index]) - timestamp)
    if skew > max_skew_s:
        raise ValueError(f"{name} has no sample within max skew at {timestamp:.9f}s (nearest skew {skew:.9f}s)")
    return values[index]


def build_canonical_episode(streams: Mapping[str, Sequence[Timestamped]], *, max_skew_s: float) -> dict[str, Any]:
    """Build one episode on the arm-state clock without interpolation/resampling."""
    if max_skew_s < 0 or not np.isfinite(max_skew_s):
        raise ValueError("max_skew_s must be finite and non-negative")
    arm_times, arm_states = _validated(streams.get("arm_state", ()), "arm_state", STATE_ARM_DIM)
    _validated(streams.get("hand_state", ()), "hand_state", STATE_HAND_DIM)
    _validated(streams.get("arm_command", ()), "arm_command", ACTION_ARM_DIM)
    _validated(streams.get("hand_command", ()), "hand_command", ACTION_HAND_DIM)
    if len(arm_times) < 2:
        raise ValueError("at least two arm-state samples are required")

    states, actions = [], []
    cameras = {key.removeprefix("camera:"): value for key, value in streams.items() if key.startswith("camera:")}
    aligned_images: dict[str, list[np.ndarray]] = {key: [] for key in cameras}
    transition_count = len(arm_times) - 1
    for state_index, (timestamp, arm) in enumerate(zip(arm_times, arm_states)):
        hand = nearest(streams["hand_state"], float(timestamp), max_skew_s, "hand_state")
        states.append(np.concatenate((np.asarray(arm), np.asarray(hand))))
        if state_index == transition_count:
            continue
        arm_cmd = nearest(streams["arm_command"], float(timestamp), max_skew_s, "arm_command")
        hand_cmd = nearest(streams["hand_command"], float(timestamp), max_skew_s, "hand_command")
        actions.append(np.concatenate((np.asarray(arm_cmd), np.asarray(hand_cmd))))
        for camera, samples in cameras.items():
            frame = np.asarray(nearest(samples, float(timestamp), max_skew_s, f"camera:{camera}"))
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"camera:{camera} frames must be decoded uint8 [H,W,3]")
            aligned_images[camera].append(frame)
    state = np.asarray(states, dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)
    t = len(state) - 1
    result = {
        "obs/state": state[:-1], "action": action, "next_obs/state": state[1:],
        "done": np.concatenate((np.zeros(t - 1, dtype=np.bool_), np.ones(1, dtype=np.bool_))),
        "timestamp": arm_times[:-1].copy(),
        "images": {name: np.stack(frames) for name, frames in aligned_images.items()},
    }
    return result

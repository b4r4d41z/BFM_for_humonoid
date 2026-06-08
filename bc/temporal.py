from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import warnings

import numpy as np


@dataclass(frozen=True)
class TemporalContract:
    """Shared temporal contract for data, training targets, and playback."""

    dataset_hz: float = 10.0
    policy_hz: float = 10.0
    prediction_horizon_s: float = 0.5
    isaaclab_physics_hz: float = 120.0
    isaaclab_control_hz: float = 60.0

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


DEFAULT_TEMPORAL_CONTRACT = TemporalContract()
LEGACY_TEMPORAL_CONTRACT_WARNING = (
    "Checkpoint does not contain temporal_contract metadata; assuming legacy "
    "defaults dataset_hz=10.0, policy_hz=10.0, prediction_horizon_s=0.5."
)


def get_default_temporal_contract() -> dict[str, float]:
    return DEFAULT_TEMPORAL_CONTRACT.to_dict()


def measure_dataset_hz(timestamps: Any) -> float:
    """Measure sampling frequency from monotonically increasing timestamps."""

    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size < 2:
        return float("nan")
    diffs = np.diff(ts)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return float("nan")
    return float(1.0 / np.median(diffs))


def build_temporal_contract_metadata(actual_dataset_hz: float | None = None) -> dict[str, float]:
    meta = get_default_temporal_contract()
    meta["actual_dataset_hz"] = float(actual_dataset_hz) if actual_dataset_hz is not None else float("nan")
    return meta


def temporal_contract_from_checkpoint_meta(
    meta: dict[str, Any],
    *,
    warn_legacy: bool = True,
) -> tuple[dict[str, float], bool]:
    """Return temporal metadata and whether it came from a legacy checkpoint."""

    raw = meta.get("temporal_contract") if isinstance(meta, dict) else None
    if not isinstance(raw, dict) and isinstance(meta, dict) and isinstance(meta.get("extra"), dict):
        raw = meta["extra"].get("temporal_contract")
    legacy = not isinstance(raw, dict)
    if legacy:
        if warn_legacy:
            warnings.warn(LEGACY_TEMPORAL_CONTRACT_WARNING, RuntimeWarning, stacklevel=2)
        return build_temporal_contract_metadata(actual_dataset_hz=float("nan")), True

    contract = build_temporal_contract_metadata(actual_dataset_hz=raw.get("actual_dataset_hz", float("nan")))
    for key in ("dataset_hz", "policy_hz", "prediction_horizon_s", "isaaclab_physics_hz", "isaaclab_control_hz"):
        if key in raw:
            contract[key] = float(raw[key])
    return contract, False


def validate_runtime_temporal_contract(
    checkpoint_contract: dict[str, Any],
    runtime_contract: TemporalContract = DEFAULT_TEMPORAL_CONTRACT,
    *,
    rtol: float = 1e-6,
) -> dict[str, float]:
    """Validate checkpoint temporal contract against runtime playback defaults."""

    runtime = runtime_contract.to_dict()
    errors: list[str] = []
    for key in ("policy_hz", "prediction_horizon_s"):
        actual = float(checkpoint_contract[key])
        expected = float(runtime[key])
        if not np.isclose(actual, expected, rtol=rtol, atol=1e-9):
            errors.append(f"{key} checkpoint={actual}, runtime={expected}")

    for key in ("isaaclab_physics_hz", "isaaclab_control_hz"):
        if key in checkpoint_contract:
            actual = float(checkpoint_contract[key])
            expected = float(runtime[key])
            if not np.isclose(actual, expected, rtol=rtol, atol=1e-9):
                errors.append(f"{key} checkpoint={actual}, runtime={expected}")

    if errors:
        raise RuntimeError("Incompatible temporal contract: " + "; ".join(errors))
    return {k: float(v) for k, v in checkpoint_contract.items()}


def find_future_target_indices(
    timestamps: Any,
    done: Any,
    *,
    horizon_s: float = DEFAULT_TEMPORAL_CONTRACT.prediction_horizon_s,
) -> list[tuple[int, int]]:
    """Map each valid source frame to future frame closest to t + horizon_s.

    Samples whose future target would cross a done boundary or leave the file are
    excluded. A done at source frame marks the end of that episode and is not a
    valid source sample for a future action target.
    """

    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    done_arr = np.asarray(done, dtype=bool).reshape(-1)
    if ts.shape[0] != done_arr.shape[0]:
        raise ValueError(f"timestamps/done length mismatch: {ts.shape[0]} vs {done_arr.shape[0]}")
    n = int(ts.shape[0])
    if n == 0:
        return []

    out: list[tuple[int, int]] = []
    episode_start = 0
    for i in range(n + 1):
        at_end = i == n
        if not at_end and not done_arr[i]:
            continue
        episode_end_exclusive = i + 1 if not at_end else n
        _append_episode_target_indices(ts, done_arr, episode_start, episode_end_exclusive, horizon_s, out)
        episode_start = i + 1
    return out


def _append_episode_target_indices(
    ts: np.ndarray,
    done_arr: np.ndarray,
    start: int,
    end: int,
    horizon_s: float,
    out: list[tuple[int, int]],
) -> None:
    if end - start < 2:
        return
    ep_ts = ts[start:end]
    for local_i, t in enumerate(ep_ts):
        src = start + local_i
        if done_arr[src]:
            continue
        target_time = float(t) + float(horizon_s)
        insertion = int(np.searchsorted(ep_ts, target_time, side="left"))
        candidates: list[int] = []
        if insertion < len(ep_ts):
            candidates.append(insertion)
        if insertion - 1 > local_i:
            candidates.append(insertion - 1)
        # Future means strictly after the source sample and within the same episode.
        candidates = [c for c in candidates if c > local_i and c < len(ep_ts)]
        if not candidates:
            continue
        best_local = min(candidates, key=lambda c: abs(float(ep_ts[c]) - target_time))
        out.append((src, start + best_local))

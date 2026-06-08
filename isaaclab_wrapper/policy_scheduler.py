from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from bc.temporal import DEFAULT_TEMPORAL_CONTRACT


@dataclass
class PolicySchedulerStats:
    control_steps: int = 0
    inference_calls: int = 0


class PolicyScheduler:
    """Time-accumulating scheduler for low-rate policy inference in control loops.

    The scheduler advances at IsaacLab control time (normally 60 Hz), calls the
    model at the temporal contract policy rate (10 Hz), and returns the cached
    absolute 26D action on in-between control steps. It never fabricates zero
    actions; the first tick always runs inference to initialize the cache.
    """

    def __init__(self, *, policy_hz: float = DEFAULT_TEMPORAL_CONTRACT.policy_hz, control_hz: float = DEFAULT_TEMPORAL_CONTRACT.isaaclab_control_hz) -> None:
        if policy_hz <= 0.0:
            raise ValueError(f"policy_hz must be > 0, got {policy_hz}")
        if control_hz <= 0.0:
            raise ValueError(f"control_hz must be > 0, got {control_hz}")
        self.policy_hz = float(policy_hz)
        self.control_hz = float(control_hz)
        self.control_dt = 1.0 / self.control_hz
        self.policy_period = 1.0 / self.policy_hz
        self._time_since_inference = 0.0
        self._cached_action: torch.Tensor | None = None
        self.stats = PolicySchedulerStats()

    @property
    def cached_action(self) -> torch.Tensor | None:
        return self._cached_action

    def reset(self) -> None:
        self._time_since_inference = 0.0
        self._cached_action = None
        self.stats = PolicySchedulerStats()

    def should_infer(self) -> bool:
        return self._cached_action is None or self._time_since_inference + 1e-12 >= self.policy_period

    def tick(self, infer_fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, bool]:
        did_infer = self.should_infer()
        if did_infer:
            action = infer_fn()
            if not isinstance(action, torch.Tensor):
                raise TypeError(f"infer_fn must return torch.Tensor, got {type(action).__name__}")
            self._cached_action = action.detach().clone()
            if self._time_since_inference >= self.policy_period:
                self._time_since_inference -= self.policy_period
            else:
                self._time_since_inference = 0.0
            self.stats.inference_calls += 1

        if self._cached_action is None:
            raise RuntimeError("PolicyScheduler has no cached action; refusing to emit zero placeholder action")

        action_out = self._cached_action.clone()
        self._time_since_inference += self.control_dt
        self.stats.control_steps += 1
        return action_out, did_infer

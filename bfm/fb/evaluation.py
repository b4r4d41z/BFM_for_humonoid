"""ROS-independent bounded evaluation, physical safety and JSONL rollout logs."""
import json
import math
from pathlib import Path
import time

import numpy as np

from bfm.contracts.sharpa import ActionNormalizer, GROUPS

ARMS = np.r_[np.arange(26)[GROUPS['left_arm']], np.arange(26)[GROUPS['right_arm']]]


def validate_evaluation(mode, warmup_ticks, max_policy_steps, max_arm_delta_rad):
    if mode not in ('shadow', 'closed-loop'):
        raise ValueError('mode must be shadow or closed-loop')
    if type(warmup_ticks) is not int or warmup_ticks < 0:
        raise ValueError('warmup_ticks must be a nonnegative integer')
    if type(max_policy_steps) is not int or max_policy_steps < 1:
        raise ValueError('max_policy_steps must be a positive integer')
    if isinstance(max_arm_delta_rad, bool) or not math.isfinite(max_arm_delta_rad) or max_arm_delta_rad <= 0:
        raise ValueError('max_arm_delta_rad must be positive and finite')


class RuntimeSafetyFilter:
    def __init__(self, max_arm_delta_rad):
        validate_evaluation('shadow', 0, 1, max_arm_delta_rad)
        self.limit = max_arm_delta_rad
        self.normalizer = ActionNormalizer()
        self.previous = None

    def apply(self, physical, measured_proprio):
        raw = np.asarray(physical, dtype=np.float64)
        if raw.shape != (26,):
            raise ValueError('Physical action must be exactly 26D')
        self.normalizer.normalize(raw)  # Validate before limiting; never repair corruption.
        if self.previous is None:
            q = np.asarray(measured_proprio).reshape(-1)
            if q.shape != (116,) or not np.isfinite(q).all():
                raise ValueError('Initial safety anchor requires finite measured 116D proprio')
            previous = raw.copy()
            previous[ARMS] = q[:len(ARMS)]
            self.normalizer.normalize(previous)
        else:
            previous = self.previous
        applied = raw.copy()
        applied[ARMS] = previous[ARMS] + np.clip(raw[ARMS]-previous[ARMS], -self.limit, self.limit)
        self.normalizer.normalize(applied)
        limited = np.flatnonzero(np.abs(applied-raw) > 1e-12).tolist()
        return applied, limited

    def commit(self, applied):
        self.previous = np.asarray(applied).copy()


class RolloutLogger:
    def __init__(self, directory, *, mode, checkpoint, task_z):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory/'ticks.jsonl').open('w')
        self.summary = dict(mode=mode, checkpoint=str(checkpoint), task_z=str(task_z),
                            total_policy_steps=0, published_policy_steps=0, warmup_steps=0,
                            safety_interventions=0, invalid_stale_observations=0)
        self.latencies = []
        self.closed = False

    def record(self, row):
        self.stream.write(json.dumps(row, allow_nan=False)+'\n')
        self.stream.flush()
        self.summary['total_policy_steps'] += 1
        self.summary['published_policy_steps'] += int(row['published'])
        self.summary['warmup_steps'] += int(row['warmup'])
        self.summary['safety_interventions'] += int(row['safety_intervention'])
        self.summary['invalid_stale_observations'] += int(not row['observation_valid'])
        if row['inference_latency_s'] is not None:
            self.latencies.append(row['inference_latency_s'])

    def close(self, reason):
        if self.closed:
            return
        self.closed = True
        self.stream.close()
        self.summary.update(termination_reason=reason,
                            mean_inference_latency_s=float(np.mean(self.latencies)) if self.latencies else None,
                            max_inference_latency_s=max(self.latencies) if self.latencies else None)
        (self.directory/'summary.json').write_text(json.dumps(self.summary, allow_nan=False, indent=2)+'\n')


class Evaluation:
    def __init__(self, runtime, buffer, logger, *, mode='shadow', warmup_ticks=10,
                 max_policy_steps=300, max_arm_delta_rad=.1, clock=time.monotonic):
        validate_evaluation(mode, warmup_ticks, max_policy_steps, max_arm_delta_rad)
        self.runtime, self.buffer, self.logger = runtime, buffer, logger
        self.mode, self.warmup_ticks, self.maximum = mode, warmup_ticks, max_policy_steps
        self.clock, self.safety = clock, RuntimeSafetyFilter(max_arm_delta_rad)
        self.steps = self.ready = 0
        self.stopped = False

    def stop(self, reason, *, invalid_observation=False):
        if invalid_observation and not self.stopped:
            self.logger.summary['invalid_stale_observations'] += 1
        self.stopped = True
        self.logger.close(reason)

    def tick(self, publish, check_publishers=lambda: None):
        if self.stopped:
            return
        self.steps += 1
        row = dict(policy_step=self.steps, runtime_timestamp=self.clock(), mode=self.mode,
                   observation_valid=False, source_ages_s=None, camera_skew_s=None,
                   inference_latency_s=None, normalized_action=None, raw_physical_action=None,
                   applied_physical_action=None, safety_intervention=False, limited_dimensions=[],
                   published=False, publication_status='waiting', warmup=False)
        try:
            row.update(self.buffer.diagnostics(self.clock()))
            obs = self.buffer.snapshot(self.clock())
            if obs is None:
                if self.ready > self.warmup_ticks:
                    raise RuntimeError('Missing observations after activation')
                self.ready = 0
            else:
                row['observation_valid'] = True
                row['warmup'] = self.ready < self.warmup_ticks
                start = self.clock()
                normalized = self.runtime.act(obs)
                row['inference_latency_s'] = self.clock()-start
                raw = ActionNormalizer().physical(normalized).numpy()
                applied, limited = self.safety.apply(raw, obs['proprio'])
                row.update(normalized_action=np.asarray(normalized).tolist(), raw_physical_action=raw.tolist(),
                           applied_physical_action=applied.tolist(), safety_intervention=bool(limited), limited_dimensions=limited)
                # Revalidate receive ages after inference, before any publication.
                row.update(self.buffer.diagnostics(self.clock()))
                row['observation_valid'] = False
                if self.buffer.snapshot(self.clock()) is None:
                    raise RuntimeError('Observations disappeared during inference')
                row['observation_valid'] = True
                self.ready += 1
                if row['warmup']:
                    row['publication_status'] = 'warmup'
                elif self.mode == 'shadow':
                    row['publication_status'] = 'shadow'
                    self.safety.commit(applied)  # Virtual previous applied target for diagnostics.
                else:
                    check_publishers()
                    publish(applied)
                    self.safety.commit(applied)
                    row.update(published=True, publication_status='published')
        except Exception as exc:
            row.update(error=f'{type(exc).__name__}: {exc}', publication_status='failed')
            self.logger.record(row)
            self.stop(row['error'])
            raise
        self.logger.record(row)
        if self.steps >= self.maximum:
            self.stop('max_policy_steps')

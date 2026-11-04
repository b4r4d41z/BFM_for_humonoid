"""Policy decision cadence is separate from causal observation receive time."""
import numpy as np


def policy_timing(bfm, clock_samples=()):
    clock = bfm.get('policy_clock')
    if clock is None:
        # Older collection-v3 used a wall timer and source-native action headers.
        return {'domain': 'mcap_log_time_ns', 'period_ns': 100_000_000,
                'tolerance_ns': 50_000_000, 'tolerance_source': 'legacy_wall_timer'}
    period = clock.get('period_ns')
    if (clock.get('domain') != 'isaac_simulation' or clock.get('topic') != '/clock'
            or type(period) is not int or period <= 0
            or not np.isclose(period / 1e9, bfm.get('policy_dt', 1 / bfm['policy_hz']), rtol=0, atol=1e-9)
            or not np.isclose(period * bfm['policy_hz'], 1e9, rtol=0, atol=1)):
        raise ValueError('Invalid simulation policy clock/period metadata')
    tolerance, source = 1, 'integer_nanosecond_rounding'
    if clock.get('quantization') == 'first physics clock sample at or after deadline' and len(clock_samples) > 1:
        stamps = np.array([m.clock.sec * 1_000_000_000 + m.clock.nanosec for _, m in clock_samples], dtype=np.int64)
        positive = np.diff(stamps)
        positive = positive[positive > 0]
        if len(positive):
            # Receive gaps are not physics steps. Use the typical observed step,
            # never a large maximum gap; one ns covers alternating rounded steps.
            step = int(np.ceil(np.median(positive)))
            if step * 2 >= period:
                raise ValueError('Cannot establish sub-policy physics-step quantization')
            tolerance, source = step + 1, 'median_positive_clock_step_plus_1ns'
    return {'domain': 'action_header_simulation_ns', 'period_ns': period,
            'tolerance_ns': tolerance, 'tolerance_source': source}


def continuity_reason(previous, current, timing):
    if previous < 0 or current < 0:
        return 'invalid_policy_stamp'
    delta = int(current) - int(previous)
    if delta == 0:
        return 'duplicate_policy_stamp'
    if delta < 0:
        return 'backward_policy_stamp'
    if delta > timing['period_ns'] + timing['tolerance_ns']:
        return 'skipped_policy_interval'
    if delta < timing['period_ns'] - timing['tolerance_ns']:
        return 'policy_burst'
    return None

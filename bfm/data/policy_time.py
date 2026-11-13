"""Policy decision cadence is separate from causal observation receive time."""
import numpy as np


def policy_timing(bfm, clock_samples=()):
    clock = bfm.get('policy_clock')
    if bfm.get('control_architecture') == 'decoupled_teleoperation_v1':
        control = bfm.get('control',{});hz = control.get('physics_hz')
        if (type(hz) not in (int,float) or not np.isfinite(hz) or hz<=0 or not clock
                or clock.get('domain')!='isaac_simulation' or clock.get('topic')!='/clock'
                or clock.get('period_ns_numerator')!=10**9 or clock.get('period_ns_denominator')!=15
                or bfm.get('policy_hz')!=15 or not np.isclose(bfm.get('policy_dt',0),1/15)):
            raise ValueError('Invalid decoupled simulation policy clock')
        step = int(np.ceil(1e9/hz))
        for _,msg in clock_samples:
            if msg.clock.sec<0 or not 0<=msg.clock.nanosec<10**9:raise ValueError('Invalid /clock sample')
        return dict(domain='action_header_simulation_ns',period_ns=1e9/15,tolerance_ns=min(step+2,int(1e9/30)-1),
                    tolerance_source='recorded_physics_hz_plus_ns_rounding',physics_hz=hz)
    if bfm.get('contract_version') == 4:
        if (not clock or clock.get('domain') != 'isaac_simulation' or clock.get('topic') != '/clock'
                or clock.get('period_ns_numerator') != 1_000_000_000 or clock.get('period_ns_denominator') != 15
                or bfm.get('policy_hz') != 15 or not np.isclose(bfm.get('policy_dt', 0), 1/15)):
            raise ValueError('Invalid v4 rational policy clock')
        return dict(domain='action_header_simulation_ns', period_ns=1_000_000_000/15,
                    period_ns_numerator=1_000_000_000, period_ns_denominator=15,
                    tolerance_ns=3_333_335, tolerance_source='300Hz_physics_step_plus_rounding')
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
        if timing.get('physics_hz',float('inf'))<=30:return 'unverifiable_coarse_clock_interval'
        return 'skipped_policy_interval'
    if delta < timing['period_ns'] - timing['tolerance_ns']:
        return 'policy_burst'
    return None


class SimPolicyClock:
    """Synthetic v4 rational deadlines, 1ns tolerance, one update without catch-up."""
    def __init__(self, hz=15):
        if type(hz) is not int or hz <= 0: raise ValueError('Positive integer Hz required')
        self.hz = hz
        self.last_ns = self.origin = None
        self.index = 1

    def advance(self, now_ns):
        if type(now_ns) is not int or now_ns < 0: raise ValueError('Invalid simulation ns')
        if self.last_ns is None or now_ns < self.last_ns:
            self.last_ns = self.origin = now_ns
            self.index = 1
            return False
        if now_ns == self.last_ns: return False
        self.last_ns = now_ns
        reached = (now_ns-self.origin+1)*self.hz//1_000_000_000
        if reached < self.index: return False
        self.index = reached+1
        return True

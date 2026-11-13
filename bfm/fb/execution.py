"""v4 runtime identity/acceptance watchdog; no low-level control here."""
import time
import uuid
from bfm.contracts.sharpa import execution_configuration, ActionNormalizer, GROUPS, JOINTS
from bfm.data.execution_v4 import identity
from bfm.data.mcap import header_ns


class ExecutionMonitor:
    def __init__(self, execution, timeout_s=.5, clock=time.monotonic):
        import math
        execution_configuration(execution)
        if not math.isfinite(timeout_s) or timeout_s<=0:raise ValueError('Positive acknowledgement timeout required')
        self.digest=execution['controller_config_sha256'];self.timeout=timeout_s;self.clock=clock
        self.session=uuid.uuid4().hex;self.counter=0;self.pending={};self.completed=set();self.events=[]

    def publish(self, message, simulation_ns):
        self.check();self.counter+=1
        message.producer_session_id=self.session;message.action_id=self.counter
        message.header.stamp.sec, message.header.stamp.nanosec=divmod(simulation_ns,10**9)
        self.pending[self.counter]=(self.clock(),simulation_ns)
        self.events.append(dict(event='published_not_accepted',producer_session_id=self.session,action_id=self.counter,simulation_ns=simulation_ns))
        return message

    def accept(self, status):
        session,aid=identity(status)
        if session!=self.session:
            if status.accepted:
                self.events.append(dict(event='competing_acceptance',producer_session_id=session,action_id=aid))
                raise RuntimeError('Competing executor producer')
            return
        if aid not in self.pending or aid in self.completed:raise RuntimeError('Unknown/duplicate executor acknowledgement')
        sent,stamp=self.pending[aid];elapsed=self.clock()-sent
        self.events.append(dict(event='accepted' if status.accepted else 'rejected',producer_session_id=session,action_id=aid,
                                acceptance_ns=header_ns(status),execution_latency_wall_s=elapsed,acceptance_delay_sim_s=(header_ns(status)-stamp)/1e9,reason=status.reason,
                                controller_config_sha256=status.controller_config_sha256))
        if (elapsed>self.timeout or not status.accepted or status.controller_config_sha256!=self.digest
                or not 0<=header_ns(status)-stamp<=200_000_000):raise RuntimeError('Executor rejection/identity/deadline mismatch')
        del self.pending[aid];self.completed.add(aid)

    def check(self):
        for aid,(sent,_) in self.pending.items():
            if self.clock()-sent>self.timeout:
                self.events.append(dict(event='missing_acknowledgement',producer_session_id=self.session,action_id=aid))
                raise RuntimeError('Missing executor acknowledgement deadline')


def fill_action(physical, action):
    import numpy as np
    a=np.asarray(physical,dtype=float)
    if a.shape!=(26,):raise ValueError('Canonical physical action must be 26D')
    ActionNormalizer().normalize(a)
    for side in JOINTS:setattr(action,side+'_arm_joint_names',list(JOINTS[side]))
    for group,sl in GROUPS.items():setattr(action,group+'_target',a[sl].tolist())
    return action


def output_types(version, action, arm, hand):
    if version == 4:return {'action':action}
    if version == 3:return {'arm':arm,'hand':hand,'action':action}
    raise ValueError('Unsupported publication contract')


class PolicyIdentity:
    """Direct bridge publication identity; deliberately no pending/ACK state."""
    def __init__(self):
        self.session=uuid.uuid4().hex;self.counter=0;self.last_ns=None

    def publish(self,message,simulation_ns):
        if type(simulation_ns) is not int or simulation_ns<0 or (self.last_ns is not None and simulation_ns<=self.last_ns):
            raise ValueError('Duplicate/backward policy simulation time')
        self.last_ns=simulation_ns;self.counter+=1
        if self.counter>=2**64:raise ValueError('Policy action identity exhausted')
        message.producer_session_id=self.session;message.action_id=self.counter
        message.header.stamp.sec,message.header.stamp.nanosec=divmod(simulation_ns,10**9)
        return message

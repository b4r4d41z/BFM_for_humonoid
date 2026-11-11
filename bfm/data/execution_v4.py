"""Strict read-only verification of v4 acceptance and commanded trajectories."""
import numpy as np
from bfm.contracts.sharpa import execution_configuration, flatten_action, JOINTS, GROUPS, LOW, HIGH

STATUS = '/bfm/execution_status'
FINGERS = ('/left_hand_joint_command', '/right_hand_joint_command')
EXTRA = (STATUS, *FINGERS, '/robot/command/hand_target', '/robot/command/arm_joint_target')


def identity(message):
    session, aid = message.producer_session_id, message.action_id
    if (not isinstance(session, str) or not session.strip() or len(session) > 128
            or type(aid) is not int or not 0 < aid < 2**64):
        raise ValueError('Invalid v4 producer/action identity')
    return session, aid


def acknowledgements(records, metadata):
    from .mcap import ACTION, header_ns
    config = execution_configuration(metadata['bfm']['execution'])
    digest = metadata['bfm']['execution']['controller_config_sha256']
    statuses = {}
    for record, msg in records.get(STATUS, []):
        key = identity(msg)
        if key in statuses: raise ValueError('Duplicate/conflicting execution acknowledgement')
        if not msg.accepted or msg.controller_config_sha256 != digest or header_ns(msg) < 0:
            raise ValueError('Rejected/invalid executor acknowledgement')
        statuses[key] = (record, msg)
    actions, watermarks = [], {}
    for record, msg in records[ACTION]:
        key = identity(msg); flatten_action(msg)
        if any(np.asarray(getattr(msg, group+'_target')).shape != (sl.stop-sl.start,) for group,sl in GROUPS.items()):
            raise ValueError('Invalid v4 action group dimensions')
        if key[1] <= watermarks.get(key[0], 0): raise ValueError('Duplicate/backward action ID')
        watermarks[key[0]] = key[1]
        if key not in statuses: raise ValueError('Missing execution acknowledgement')
        accepted_record, accepted = statuses.pop(key)
        delay = header_ns(accepted)-header_ns(msg)
        if not 0 <= delay <= config['action_max_age_sim_sec']*1e9:
            raise ValueError('Invalid generation/acceptance simulation interval')
        actions.append((record, msg, accepted_record, accepted))
    if statuses: raise ValueError('Unmatched/competing executor action')
    return actions, config


def verify_streams(records, actions, config):
    """Verify each recorded update; unknown leading velocity uses an interval.

    This checks existence of the commanded state, not physical target attainment.
    Recording loss/inhibition gaps are not silently interpolated.
    """
    from .mcap import arm_values, header_ns, HELD
    ai = np.r_[0:7,13:20];lo,hi = np.array(LOW)[ai],np.array(HIGH)[ai]
    active = [(header_ns(ack), flatten_action(msg)) for _,msg,_,ack in actions]
    stamps = np.array([t for t,_ in active])
    # Receipt order is used for interval association when simulation epochs reset.
    last_q = None;last_time = None;vlow = -np.array(config['max_velocity']);vhigh = -vlow
    checked = 0
    for record,msg in records.get(HELD[0],[]):
        t = header_ns(msg);idx = np.searchsorted(stamps,t,side='right')-1
        if idx < 0: continue
        q = arm_values(msg)
        if (q < lo-1e-7).any() or (q > hi+1e-7).any(): raise ValueError('Unbounded executor arm command')
        if last_q is not None:
            if not 6_666_665 <= t-last_time <= 13_333_335: raise ValueError('Missing/duplicate executor arm update')
            possible = [idx]+([idx-1] if idx and stamps[idx]==t else [])
            matches = []
            for action_index in possible:
                target = active[action_index][1][ai]
                desired = np.clip((target-last_q)/config['response_time'], -np.array(config['max_velocity']), config['max_velocity'])
                step = np.array(config['max_acceleration'])*.01
                lower = np.clip(desired,vlow-step,vlow+step);upper = np.clip(desired,vhigh-step,vhigh+step)
                if not ((q < np.clip(last_q+lower*.01,lo,hi)-1e-6) | (q > np.clip(last_q+upper*.01,lo,hi)+1e-6)).any():
                    matches.append((lower,upper))
            if not matches:raise ValueError('Executor trajectory differs from bounded velocity/acceleration rule')
            lower=np.min([a for a,_ in matches],axis=0);upper=np.max([b for _,b in matches],axis=0)
            velocity = (q-last_q)/.01
            interior = (q > lo+1e-7)&(q < hi-1e-7)
            vlow = np.where(interior,velocity,lower);vhigh = np.where(interior,velocity,upper)
            checked += 1
        last_q,last_time = q,t
    if checked < 1: raise ValueError('Insufficient executor arm evidence')
    for topic,slack in [(HELD[0],13_333_335),*[(t,23_333_335) for t in FINGERS]]:
        times=np.array([header_ns(m) for _,m in records.get(topic,[])])
        for (begin,_),(end,_) in zip(active,active[1:]):
            selected=times[(times>=begin)&(times<=end)]
            if not len(selected) or selected[0]-begin>slack or end-selected[-1]>slack:
                raise ValueError('Missing execution interval coverage: '+topic)
    if len(records.get(HELD[1],[])) != len(actions):
        raise ValueError('Unexpected/missing logical hand forwarding evidence')
    for _,msg,_,ack in actions:
        expected = np.rint(np.r_[msg.left_hand_target,msg.right_hand_target])
        found = [m for _,m in records.get(HELD[1],[]) if header_ns(m)==header_ns(ack)]
        if len(found)!=1 or not np.array_equal(np.array(list(found[0].left_hand_position)+list(found[0].right_hand_position)),expected):
            raise ValueError('Accepted logical hand forwarding mismatch')
    for topic in FINGERS:
        samples = records.get(topic,[])
        if not samples: raise ValueError('Missing executor finger stream')
        previous = None
        for _,msg in samples:
            q = np.asarray(msg.position);t = header_ns(msg)
            if len(msg.name)!=len(set(msg.name)) or q.shape!=(len(msg.name),) or not np.isfinite(q).all():
                raise ValueError('Invalid finger targets')
            side = 'left' if topic == FINGERS[0] else 'right'
            sign = 1 if side == 'left' else -1
            angles = dict(zip(msg.name, np.rad2deg(q)*sign))
            required = {side+'_thumb_MCP_FE':50, side+'_thumb_IP':65, side+'_thumb_CMC_FE':90}
            logical = [angles[side+'_thumb_MCP_FE']/50*100, angles[side+'_thumb_CMC_FE']/90*100]
            expected = {n: logical[0]/100*scale for n,scale in required.items() if not n.endswith('CMC_FE')}
            expected[side+'_thumb_CMC_FE'] = logical[1]*.9
            expected[side+'_thumb_CMC_AA'] = -20
            for finger in ('index','middle','ring','pinky'):
                base = 100 if finger == 'index' else 90
                value = angles[side+'_'+finger+'_MCP_FE']/base*100;logical.append(value)
                for joint,scale in [('MCP_FE',base),('PIP',100),('DIP',100)]:expected[side+'_'+finger+'_'+joint]=value*scale/100
            if set(angles)!=set(expected) or any(abs(angles[n]-v)>1e-5 for n,v in expected.items()) or min(logical)<-1e-6 or max(logical)>100+1e-6:
                raise ValueError('Finger mapping differs from Synthetic v4')
            if previous is not None:
                pt,pq,names,old_logical = previous
                if names != list(msg.name) or not 16_666_665 <= t-pt <= 23_333_335 or (np.abs(q-pq)>np.deg2rad(6.01)).any():
                    raise ValueError('Invalid 50Hz finger execution continuity')
                # At a shared clock instant the hand callback can precede/follow the finger tick.
                choices = [j for j, (stamp,_) in enumerate(active) if stamp <= t]
                if choices:
                    index = choices[-1];candidates = [index]
                    if active[index][0] == t and index: candidates.append(index-1)
                    valid = False
                    for j in candidates:
                        target = np.rint(active[j][1][7:13] if side=='left' else active[j][1][20:26])
                        nxt = np.array(old_logical)+np.clip(target-np.array(old_logical),-np.array([5,5,6,6,6,6]), [5,5,6,6,6,6])
                        if target[0] >= old_logical[0] and nxt[2:].mean() < 40:nxt[0]=old_logical[0]
                        if np.allclose(nxt,logical,atol=1e-5,rtol=0): valid=True
                    if not valid:raise ValueError('Finger interpolation/acceptance mismatch')
            previous = t,q,list(msg.name),logical
    return dict(arm_updates_checked=checked, controller_config_sha256=actions[0][3].controller_config_sha256)

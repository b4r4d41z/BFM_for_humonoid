"""Lossless decoded command history and causal sampled-target evidence (schema 4.1)."""
import base64
import json
import hashlib
import numpy as np
from .mcap import ACTION, RAW_ARM, HELD, CLOCK, EPISODE, RECORDING, arm_values, header_ns, asof
from bfm.contracts.sharpa import LOW, HIGH

HISTORY = dict(raw_arm='/robot/command/arm_joint_target',raw_hand='/robot/command/hand_target',
               mapped_arm=RAW_ARM,arm=HELD[0],hand=HELD[1],left_fingers='/left_hand_joint_command',
               right_fingers='/right_hand_joint_command',clock=CLOCK,success='/bfm/success',
               episode=EPISODE,recording=RECORDING,action=ACTION,
               input_provenance='/bfm/source/input_provenance')
REQUIRED = ('raw_arm','raw_hand','mapped_arm','arm','hand','left_fingers','right_fingers','clock','action')
ARMS = np.r_[0:7,13:20]
HANDS = np.r_[7:13,20:26]


def plain(value):
    if isinstance(value,(bytes,bytearray)):return {'__bytes_base64__':base64.b64encode(value).decode()}
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if hasattr(value,'get_fields_and_field_types'):return {k:plain(getattr(value,k)) for k in value.get_fields_and_field_types()}
    if hasattr(value,'__slots__'):return {k:plain(getattr(value,k)) for k in value.__slots__}
    if hasattr(value,'__dict__'):return {k:plain(v) for k,v in vars(value).items()}
    return [plain(v) for v in value]


def hand_values(msg):
    a=np.asarray([*msg.left_hand_position,*msg.right_hand_position],dtype=float)
    if a.shape!=(12,) or not np.isfinite(a).all() or (a<0).any() or (a>100).any():raise ValueError('Invalid logical hand input')
    return a


def verify_commands(records):
    from .source_provenance import header_identity
    for key in REQUIRED:
        if not records.get(HISTORY[key]):raise ValueError('Missing decoupled stream: '+HISTORY[key])
    warnings=[];raw={}
    for _,msg in records[HISTORY['raw_arm']]:arm_values(msg)
    for _,msg in records[RAW_ARM]:
        if header_ns(msg)<0:raise ValueError('Invalid mapped source header')
        raw.setdefault(header_identity(msg),[]).append(arm_values(msg))
    for record,msg in records[HELD[0]]:
        q=arm_values(msg);matches=raw.get(header_identity(msg),[])
        if (q<np.array(LOW)[ARMS]-1e-7).any() or (q>np.array(HIGH)[ARMS]+1e-7).any():raise ValueError('Unbounded direct arm command')
        if not matches and record<records[RAW_ARM][0][0]:
            warnings.append(dict(topic=HELD[0],record_ns=record,reason='leading_source_chain_outside_bag'));continue
        if not any(np.allclose(np.clip(v,np.array(LOW)[ARMS],np.array(HIGH)[ARMS]),q,atol=1e-7,rtol=0) for v in matches):
            raise ValueError('Mapped raw/controller header identity or clamp mismatch')
    raw_hands={}
    for _,msg in records[HISTORY['raw_hand']]:
        if header_ns(msg)<0:raise ValueError('Invalid raw hand header')
        raw_hands.setdefault(header_identity(msg),[]).append(hand_values(msg))
    for record,msg in records[HELD[1]]:
        candidates=raw_hands.get(header_identity(msg),[])
        if not candidates and record<records[HISTORY['raw_hand']][0][0]:
            warnings.append(dict(topic=HELD[1],record_ns=record,reason='leading_source_chain_outside_bag'));continue
        if not candidates or not any(np.array_equal(hand_values(msg),v) for v in candidates):raise ValueError('Logical hand source/controller identity mismatch')
    for key in ('left_fingers','right_fingers'):
        for _,msg in records[HISTORY[key]]:
            q=np.asarray(msg.position)
            if not len(msg.name) or len(set(msg.name))!=len(msg.name) or q.shape!=(len(msg.name),) or not np.isfinite(q).all():
                raise ValueError('Invalid physical finger command')
    return warnings


def history_arrays(records,starts,ends):
    arrays={};meta={}
    for key,topic in HISTORY.items():
        samples=records.get(topic,[]);times=np.array([t for t,_ in samples],dtype=np.int64)
        blobs=[json.dumps(plain(m),sort_keys=True,separators=(',',':'),allow_nan=False).encode() for _,m in samples]
        offsets=np.r_[0,np.cumsum([len(b) for b in blobs])].astype(np.int64)
        payload=np.frombuffer(b''.join(blobs),dtype=np.uint8)
        values=dict(record_ns=times,header_ns=np.array([header_ns(m) for _,m in samples],dtype=np.int64),
                    offsets=offsets,payload=payload,interval_start=np.searchsorted(times,starts,side='right').astype(np.int64),
                    interval_stop=np.searchsorted(times,ends,side='right').astype(np.int64))
        arrays.update({f'control_history/{key}/{name}':a for name,a in values.items()})
        meta[key]=dict(topic=topic,count=len(samples),encoding='canonical_JSON_decoded_ROS_fields_v1',
                       sha256=hashlib.sha256(payload.tobytes()).hexdigest())
    return arrays,meta


def validate_history(root):
    """Return history array names after checking offsets, interval indices and hash."""
    expected=set();starts,ends=root['record_ns'][:],root['next_record_ns'][:]
    for key,topic in HISTORY.items():
        if key=='input_provenance' and key not in root.attrs['control_history']:continue
        m=root.attrs['control_history'][key]
        if m['topic']!=topic or m['encoding']!='canonical_JSON_decoded_ROS_fields_v1':raise ValueError('Unknown history encoding/topic')
        g=root['control_history/'+key];times=g['record_ns'][:];headers=g['header_ns'][:];offsets=g['offsets'][:];payload=g['payload'][:]
        if (times.dtype!=np.int64 or times.ndim!=1 or (np.diff(times)<0).any() or len(times)!=m['count']
                or headers.dtype!=np.int64 or headers.shape!=times.shape or offsets.dtype!=np.int64
                or offsets.shape!=(len(times)+1,) or offsets[0]!=0 or (np.diff(offsets)<=0).any()
                or offsets[-1]!=len(payload) or payload.dtype!=np.uint8 or payload.ndim!=1
                or hashlib.sha256(payload.tobytes()).hexdigest()!=m['sha256']):raise ValueError('Corrupt control history')
        if key in REQUIRED and not len(times):raise ValueError('Missing required control history')
        for begin,end in zip(offsets[:-1],offsets[1:]):
            try:
                msg=json.loads(payload[begin:end].tobytes(),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
                if not isinstance(msg,dict):raise ValueError('Expected ROS message fields')
            except (ValueError,UnicodeError) as exc:raise ValueError('Corrupt history JSON') from exc
        for name,bounds in [('interval_start',starts),('interval_stop',ends)]:
            a=g[name][:]
            if a.dtype!=np.int64 or not np.array_equal(a,np.searchsorted(times,bounds,side='right')):raise ValueError('Incorrect history interval index')
        expected.update(f'control_history/{key}/{n}' for n in ('record_ns','header_ns','offsets','payload','interval_start','interval_stop'))
    return expected

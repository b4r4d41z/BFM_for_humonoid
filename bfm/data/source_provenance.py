"""Synthetic gateway_input v1 and explicitly weaker historical input evidence.

Record time never identifies a Gateway callback. Trace hashes bind exact UTF-8
bytes; they provide integrity, not authentication or physical execution evidence.
"""
import base64
import hashlib
import json
import struct
from types import SimpleNamespace

import numpy as np

from bfm.contracts.sharpa import JOINTS, LOW, HIGH, flatten_action
from .mcap import ACTION, RAW_ARM, HELD, CLOCK, EPISODE, asof, header_ns, arm_values

TOPIC = '/bfm/source/input_provenance'
PROTOCOL = 'synthetic.gateway_input'
REORDER_NS = 50_000_000  # Existing converter transport budget, not policy cadence.
ATOL = 1e-7  # Historical float representation tolerance; never an encoder count.
NAMES = JOINTS['left'] + JOINTS['right']
ARMS = np.r_[0:7, 13:20]
HANDS = np.r_[7:13, 20:26]
HAND_TOPIC = '/robot/command/hand_target'


def boundaries(records):
    """Exactly the events that clear Synthetic's passive input buffers."""
    result = []; previous = None; boundary = None
    for record, msg in records.get(CLOCK, []):
        stamp = msg.clock.sec * 10**9 + msg.clock.nanosec
        if previous is not None and stamp < previous:
            result.append(record)
        previous = stamp
    for record, msg in records.get(EPISODE, []):
        event = json.loads(msg.data)
        if 'phase' not in event:
            continue  # Recorder start/stop alone does not clear Gateway inputs.
        if event.get('boundary_id') != boundary or event['phase'] in ('stopped', 'failed', 'closed'):
            result.append(record)
        boundary = event.get('boundary_id')
    return sorted(set(result))


def fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError('Invalid provenance fields')


def uint64(value):
    if type(value) is not int or not 0 < value < 2**64:
        raise ValueError('Invalid provenance sequence/action ID')


def values(value, size):
    if (not isinstance(value, list) or len(value) != size
            or any(type(v) not in (int, float) for v in value)):
        raise ValueError('Invalid provenance numeric dimensions/type')
    a = np.asarray(value, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite provenance values')
    return a


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate provenance JSON key')
        result[key] = value
    return result


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def header_identity(msg):
    h = msg.header
    return h.stamp.sec, h.stamp.nanosec, h.frame_id


class Provenance:
    def __init__(self, records, metadata):
        self.records = records
        self.episode = metadata['bfm']['episode']['episode_id']
        self.cutoffs = boundaries(records)
        self.processing = metadata['bfm']['arm_target_processing']
        declaration = metadata['bfm'].get('operator_input_provenance')
        declared = 'operator_input_provenance' in metadata['bfm']
        self.explicit = (declared or bool(records.get(TOPIC))
                         or any(hasattr(m, 'source_provenance_version') or hasattr(m, 'source_provenance_sha256')
                                for _, m in records[ACTION]))
        self.mode = 'explicit_gateway_input_v1' if self.explicit else 'historical_bounded_value_chain_v1'
        self.traces = {}; self.inputs = {}; self.selected = {}
        if declared and (not isinstance(declaration, dict)
                or declaration.get('protocol') != PROTOCOL or type(declaration.get('version')) is not int
                or declaration['version'] != 1 or declaration.get('topic') != TOPIC
                or declaration.get('message_type') != 'std_msgs/msg/String'):
            raise ValueError(f'episode={self.episode}: unsupported operator provenance declaration')
        for record, msg in records.get(TOPIC, []):
            try:
                self.add_trace(record, msg.data)
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
                raise ValueError(f'episode={self.episode} mode={self.mode} trace_record_ns={record}: {exc}') from exc

    def add_trace(self, record, text):
        if not isinstance(text, str):
            raise ValueError('Expected provenance String.data')
        data = json.loads(text, object_pairs_hook=unique_keys)
        fields(data, 'protocol version producer_session_id action_id decision_stamp arm hand')
        if data['protocol'] != PROTOCOL or type(data['version']) is not int or data['version'] != 1:
            raise ValueError('Unsupported trace protocol/version')
        session = data['producer_session_id']; uint64(data['action_id'])
        if not isinstance(session, str) or not session.strip():
            raise ValueError('Invalid provenance session')
        stamp = data['decision_stamp']; fields(stamp, 'sec nanosec')
        if (type(stamp['sec']) is not int or not 0 <= stamp['sec'] < 2**31
                or type(stamp['nanosec']) is not int or not 0 <= stamp['nanosec'] < 10**9):
            raise ValueError('Invalid trace decision stamp')
        for kind in ('arm', 'hand'):
            sample = data[kind]
            extra = 'joint_names position_rad processing bounds_source_sha256' if kind == 'arm' else 'left_hand_position right_hand_position'
            fields(sample, 'kind input_sequence topic header ' + extra)
            uint64(sample['input_sequence'])
            topic = sample['topic']
            if (sample['kind'] != kind or not isinstance(topic,str) or not topic.startswith('/')
                    or any(c.isspace() for c in topic) or (kind=='arm' and topic!=RAW_ARM)):
                raise ValueError('Incorrect source kind/topic')
            h = sample['header']; fields(h, 'stamp_sec stamp_nanosec frame_id')
            if (type(h['stamp_sec']) is not int or not -2**31 <= h['stamp_sec'] < 2**31
                    or type(h['stamp_nanosec']) is not int or not 0 <= h['stamp_nanosec'] < 2**32
                    or not isinstance(h['frame_id'], str)):
                raise ValueError('Invalid source-native header')
            if kind == 'arm':
                names = sample['joint_names']
                if not isinstance(names, list) or len(names) != 14 or set(names) != set(NAMES):
                    raise ValueError('Invalid source joint names/order')
                values(sample['position_rad'], 14)
                if (sample['processing'] != 'per_joint_saturation_rad'
                        or sample['bounds_source_sha256'] != self.processing['bounds_source_sha256']):
                    raise ValueError('Incorrect source arm bounds/processing')
            else:
                q = np.r_[values(sample['left_hand_position'], 6), values(sample['right_hand_position'], 6)]
                if (q < 0).any() or (q > 100).any():
                    raise ValueError('Invalid logical hand range')
            key = (session, kind, sample['input_sequence']); encoded = canonical(sample)
            if key in self.inputs and self.inputs[key] != encoded:
                raise ValueError('Conflicting source identity: ' + str(key))
            self.inputs[key] = encoded
        key = (session, data['action_id'])
        if key in self.traces:
            if self.traces[key][0] != text:
                raise ValueError('Conflicting action trace identity: ' + str(key))
            self.traces[key][2].append(record)
        else:
            self.traces[key] = (text, data, [record])

    def explicit_match(self, record, msg, action, lower):
        version = getattr(msg, 'source_provenance_version', None)
        digest = getattr(msg, 'source_provenance_sha256', None)
        if type(version) is not int or version != 1:
            raise ValueError('Missing/unsupported operator provenance claim; no historical fallback')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid Action provenance SHA-256')
        key = (msg.producer_session_id, msg.action_id)
        if key not in self.traces:
            raise ValueError('Missing referenced arm/hand trace for ' + str(key))
        text, trace, times = self.traces[key]
        if hashlib.sha256(text.encode('utf-8')).hexdigest() != digest:
            raise ValueError('Action provenance digest mismatch')
        if trace['decision_stamp'] != dict(sec=msg.header.stamp.sec, nanosec=msg.header.stamp.nanosec):
            raise ValueError('Action/trace simulation timestamp mismatch')
        arm = trace['arm']; hand = trace['hand']; limits = self.processing['joint_limits']
        bounded = {n: min(max(v, limits[n][0]), limits[n][1]) for n, v in zip(arm['joint_names'], arm['position_rad'])}
        expected = [*[bounded[n] for n in JOINTS['left']], *hand['left_hand_position'],
                    *[bounded[n] for n in JOINTS['right']], *hand['right_hand_position']]
        if struct.pack('<26d', *expected) != struct.pack('<26d', *action):
            delta = np.abs(np.asarray(expected) - action)
            raise ValueError(f'Exact arm/hand trace target mismatch max_diff={delta.max()} indices={np.flatnonzero(delta).tolist()}')
        for kind in ('arm', 'hand'):
            identity = (key[0], kind); sequence = trace[kind]['input_sequence']
            previous = self.selected.get(identity)
            if previous and (sequence < previous[0] or (lower > previous[1] and sequence == previous[0])):
                raise ValueError(f'{kind} source sequence went backward or crossed STOP/reset boundary: {sequence}')
            self.selected[identity] = (sequence, record)
        return dict(confidence='gateway_selected_identity',sha256=digest,trace_record_ns=times,
                    arm_input_sequence=arm['input_sequence'],hand_input_sequence=hand['input_sequence'])

    def historical_match(self, record, action, lower, upper):
        from .decoupled import hand_values
        result = {}
        for kind, topic, controller, extract, indices in (
                ('arm', RAW_ARM, HELD[0], arm_values, ARMS),
                ('hand', HAND_TOPIC, HELD[1], hand_values, HANDS)):
            samples = self.records[topic]
            # At the start of the transport window the last input can be held
            # indefinitely (no Gateway watchdog). Later inputs are not allowed
            # to revive an arbitrarily old, superseded value.
            anchor = asof(samples, record - REORDER_NS)
            candidates = ([anchor] if anchor else []) + [p for p in samples if record - REORDER_NS < p[0] <= upper]
            candidates = [p for p in candidates if lower <= p[0]]
            matches = []; source_values = set()
            for time, source in candidates:
                raw = extract(source)
                bounded = np.clip(raw, np.array(LOW)[ARMS], np.array(HIGH)[ARMS]) if kind == 'arm' else raw
                if not np.allclose(bounded, action[indices], atol=ATOL if kind == 'arm' else 0, rtol=0):
                    continue
                for held_time, held in self.records[controller]:
                    if (lower <= held_time <= upper and abs(held_time-time) <= REORDER_NS
                            and header_identity(held) == header_identity(source)
                            and np.allclose(extract(held), bounded, atol=ATOL if kind == 'arm' else 0, rtol=0)):
                        matches.append(dict(source_record_ns=time,controller_record_ns=held_time,header=list(header_identity(source))))
                        source_values.add(tuple(raw))
            if not matches:
                if not asof(samples, record) or asof(samples, record)[0] < lower:
                    return None  # Incomplete leading/boundary evidence, visible rejection.
                diagnostics = [dict(record_ns=t,header=list(header_identity(m)),max_diff=float(np.max(abs(extract(m)-action[indices]))),
                                    indices=np.flatnonzero(abs(extract(m)-action[indices])>ATOL).tolist()) for t,m in candidates]
                raise ValueError(f'{kind} missing exact source/header/clamp chain candidates={diagnostics}')
            if len(source_values) != 1:
                raise ValueError(f'{kind} ambiguous pre-clamp sources (saturation collision): {matches}')
            result[kind] = dict(confidence='unique_recorded_chain' if len(matches)==1 else 'equivalent_values_only',candidates=matches)
        return result

    def match(self, index, record, msg):
        lower = max((r for r in self.cutoffs if r <= record), default=-1)
        upper = min([record + REORDER_NS] + [r-1 for r in self.cutoffs if r > record]
                    + [r-1 for r,_ in self.records[ACTION][index+1:index+2]])
        try:
            action = flatten_action(msg)
            evidence = (self.explicit_match(record, msg, action, lower) if self.explicit
                        else self.historical_match(record, action, lower, upper))
            # These four recorder-causal times describe boundary controller/input
            # context, NOT the matching input identity or its callback time.
            context = [asof(self.records[t], record) for t in (RAW_ARM, HELD[0], HAND_TOPIC, HELD[1])]
            complete = all(p is not None and p[0] >= lower for p in context)
            association = dict(index=index,producer_session_id=msg.producer_session_id,action_id=msg.action_id,
                               record_ns=record,action_header_ns=header_ns(msg),mode=self.mode,boundary_record_ns=lower,evidence=evidence)
            return (np.array([p[0] for p in context], dtype=np.int64) if complete and evidence is not None else None), association
        except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError(f'episode={self.episode} action_index={index} action_id={msg.action_id} '
                             f'session={msg.producer_session_id} record_ns={record} sim_ns={header_ns(msg)} '
                             f'mode={self.mode} boundary_record_ns={lower}: {exc}') from exc


def validate_provenance(root):
    """Replay evidence from hashed full histories, including rejected decisions."""
    from .decoupled import HISTORY
    metadata = root.attrs['source_metadata']
    declared = 'operator_input_provenance' in metadata['bfm']
    saved = root.attrs.get('action_provenance')
    if saved is None:
        if declared or 'input_provenance' in root.attrs['control_history']:
            raise ValueError('Missing explicit provenance validation metadata')
        return  # Existing schema 4.1 stores retain their original validation.
    def message(value):
        if isinstance(value, dict):
            if set(value)=={'__bytes_base64__'}:
                return base64.b64decode(value['__bytes_base64__'],validate=True)
            return SimpleNamespace(**{k:message(v) for k,v in value.items()})
        if isinstance(value, list):
            return [message(v) for v in value]
        return value
    records = {}
    for key in ('mapped_arm','arm','raw_hand','hand','action','clock','episode','input_provenance'):
        if key not in root.attrs['control_history']:
            continue
        g = root['control_history/'+key]; offsets = g['offsets'][:]; payload = g['payload'][:].tobytes()
        records[HISTORY[key]] = [(int(t),message(json.loads(payload[a:b])))
                                for t,a,b in zip(g['record_ns'][:],offsets[:-1],offsets[1:])]
    verifier = Provenance(records,metadata); expected = []; contexts = {}; actions = {}
    for i,(record,msg) in enumerate(records[ACTION]):
        context,evidence = verifier.match(i,record,msg); expected.append(evidence)
        contexts[record] = context; actions[record] = flatten_action(msg)
    if saved != expected:
        raise ValueError('Action provenance metadata differs from recorded evidence')
    for prefix in ('','next_'):
        for i,record in enumerate(root[prefix+'record_ns'][:]):
            if contexts.get(record) is None or not np.array_equal(contexts[record],root[prefix+'command_source_record_ns'][i]):
                raise ValueError('Incorrect recorder-causal command context')
            if not prefix and not np.array_equal(actions[record],root['action'][i]):
                raise ValueError('Stored action differs from provenance history')

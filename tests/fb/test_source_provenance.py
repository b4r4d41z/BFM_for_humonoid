"""Synthetic d13fedf gateway_input v1 wire contract; no ROS/Isaac dependency."""
import copy
import hashlib
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import zarr

from test_sharpa import source
from test_decoupled_v4 import decoupled, header, convert_fixture
from bfm.contracts.sharpa import FIELDS, JOINTS, HIGH, flatten_action
from bfm.data.mcap import ACTION, RAW_ARM, HELD, TOPICS, CLOCK, EPISODE, asof, convert, read_records
from bfm.data.decoupled import HISTORY, plain, hand_values
from bfm.data.source_provenance import TOPIC, PROTOCOL, Provenance, canonical
from bfm.data.sharpa import write, validate, SharpaDataset


def seal(action, trace):
    text = canonical(trace)
    action.source_provenance_version = 1
    action.source_provenance_sha256 = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return NS(data=text)


def explicit(fixture):
    records, metadata = fixture
    metadata['bfm']['operator_input_provenance'] = dict(protocol=PROTOCOL,version=1,topic=TOPIC,
        message_type='std_msgs/msg/String',required_for_operator_actions=True)
    records[TOPIC] = []
    for record, action in records[ACTION]:
        _, arm = asof(records[RAW_ARM], record)
        _, hand = asof(records[HISTORY['raw_hand']], record)
        def native(msg):
            return dict(stamp_sec=msg.header.stamp.sec,stamp_nanosec=msg.header.stamp.nanosec,frame_id=msg.header.frame_id)
        trace = dict(protocol=PROTOCOL,version=1,producer_session_id=action.producer_session_id,action_id=action.action_id,
            decision_stamp=dict(sec=action.header.stamp.sec,nanosec=action.header.stamp.nanosec),
            arm=dict(kind='arm',input_sequence=records[RAW_ARM].index(asof(records[RAW_ARM],record))+1,topic=RAW_ARM,
                     header=native(arm),joint_names=list(arm.name),position_rad=list(arm.position),
                     processing='per_joint_saturation_rad',bounds_source_sha256=metadata['bfm']['arm_target_processing']['bounds_source_sha256']),
            hand=dict(kind='hand',input_sequence=records[HISTORY['raw_hand']].index(asof(records[HISTORY['raw_hand']],record))+1,
                      topic=HISTORY['raw_hand'],header=native(hand),left_hand_position=list(hand.left_hand_position),right_hand_position=list(hand.right_hand_position)))
        records[TOPIC].append((record+5_000_000,seal(action,trace)))
    return records, metadata


def alter(fixture, index, change):
    r,_ = fixture; record,msg = r[TOPIC][index]; trace = json.loads(msg.data)
    change(trace); r[TOPIC][index] = record, seal(r[ACTION][index][1],trace)


def test_explicit_reordering_reuse_order_and_uint64(decoupled,tmp_path):
    r,m = explicit(decoupled)
    # Evidence arrives after the Action; a newer, one-count input arrives before
    # it. The exact selected trace still names the older input.
    record,action = r[ACTION][3]; newer = copy.deepcopy(asof(r[RAW_ARM],record)[1])
    newer.position[0] += 2*np.pi/4096; newer.header.stamp.nanosec += 123
    for topic in (RAW_ARM,HELD[0]):r[topic].append((record-100_000,copy.deepcopy(newer)))
    # Source order is by name, not assumed to be canonical. Large IDs must never
    # round through binary64. Headers and values may repeat across callbacks.
    for i in range(len(r[ACTION])):
        def change(t):
            t['arm']['joint_names'].reverse();t['arm']['position_rad'].reverse()
            t['arm']['input_sequence'] += 2**53
            t['hand']['input_sequence'] += 2**53
        alter((r,m),i,change)
    prior = json.loads(r[TOPIC][3][1].data)
    r[ACTION][4][1].left_arm_target = list(r[ACTION][3][1].left_arm_target)
    alter((r,m),4,lambda t:t.update(arm=prior['arm']))
    r[TOPIC].append(copy.deepcopy(r[TOPIC][0]))  # Identical duplicate is idempotent.
    arrays,attrs = convert_fixture((r,m));path=tmp_path/'verified.zarr';write(path,arrays,attrs)
    assert validate(zarr.open_group(str(path),mode='r'))['transitions']==9
    assert attrs['action_provenance'][3]['evidence']['arm_input_sequence']>2**53
    assert attrs['action_provenance'][4]['evidence']['arm_input_sequence']==attrs['action_provenance'][3]['evidence']['arm_input_sequence']
    assert arrays['action'].shape==(9,26) and arrays['obs/state'].shape==(9,56)
    for pre in ('obs','next_obs'):
        assert all(arrays[pre+'/images/'+c].dtype==np.uint8 for c in ('head','left_wrist','right_wrist'))
        assert (arrays[pre+'/sync_age_ns']>=0).all()
    np.testing.assert_allclose(arrays['duration'],1/15,atol=1e-9)
    writable=zarr.open_group(str(path),mode='a'); evidence=copy.deepcopy(attrs['action_provenance'])
    evidence[3]['evidence']['arm_input_sequence']+=1;writable.attrs['action_provenance']=evidence
    with pytest.raises(ValueError,match='differs from recorded evidence'):validate(writable)


@pytest.mark.parametrize('fault',['missing','digest','session','conflict','input_conflict','action','names','dimensions','nan','hand_range','bounds','zero_claim','declaration','null_declaration','duplicate_json'])
def test_explicit_fail_closed(decoupled,fault):
    r,m=explicit(decoupled)
    if fault=='missing':r[TOPIC].pop(3)
    if fault=='digest':r[ACTION][3][1].source_provenance_sha256='0'*64
    if fault=='session':alter((r,m),3,lambda t:t.update(producer_session_id='other'))
    if fault=='conflict':
        trace=json.loads(r[TOPIC][3][1].data);trace['hand']['input_sequence']+=100
        r[TOPIC].append((r[TOPIC][3][0]+1,NS(data=canonical(trace))))
    if fault=='input_conflict':alter((r,m),3,lambda t:t['arm'].update(input_sequence=1))
    if fault=='action':r[ACTION][3][1].left_arm_target[0]+=2*np.pi/4096
    if fault=='names':alter((r,m),3,lambda t:t['arm']['joint_names'].__setitem__(0,'wrong'))
    if fault=='dimensions':alter((r,m),3,lambda t:t['arm']['position_rad'].pop())
    if fault=='nan':r[TOPIC][3][1].data=r[TOPIC][3][1].data.replace('"version":1','"version":NaN')
    if fault=='hand_range':alter((r,m),3,lambda t:t['hand']['left_hand_position'].__setitem__(0,101))
    if fault=='bounds':alter((r,m),3,lambda t:t['arm'].update(bounds_source_sha256='0'*64))
    if fault=='zero_claim':r[ACTION][3][1].source_provenance_version=0
    if fault=='declaration':m['bfm']['operator_input_provenance']['version']=2
    if fault=='null_declaration':m['bfm']['operator_input_provenance']=None
    if fault=='duplicate_json':r[TOPIC][3][1].data=r[TOPIC][3][1].data.replace('"version":1','"version":1,"version":1')
    with pytest.raises(ValueError,match='provenance|source|trace|operator|identity|bounds|joint|finite|target|hand|fields'):
        convert_fixture((r,m))


@pytest.mark.parametrize('event',['stop','rollback'])
def test_explicit_no_source_reuse_across_boundary(decoupled,event):
    r,m=explicit(decoupled);record,_=r[ACTION][4];prior=json.loads(r[TOPIC][3][1].data)
    r[ACTION][4][1].left_arm_target=list(r[ACTION][3][1].left_arm_target)
    alter((r,m),4,lambda t:t.update(arm=prior['arm']))
    if event=='stop':
        r[EPISODE]=[(record-500_000,NS(data=json.dumps(dict(phase='stopped',boundary_id='reset',episode=m['bfm']['episode'])))),
                    (record-400_000,NS(data=json.dumps(dict(phase='playing',boundary_id='reset',episode=m['bfm']['episode']))))]
    else:r[CLOCK].append((record-500_000,NS(clock=header(0).stamp)))
    with pytest.raises(ValueError,match='STOP/reset boundary'):convert_fixture((r,m))


def test_explicit_clamp_and_native_header(decoupled):
    r,m=explicit(decoupled)
    for i,(_,a) in enumerate(r[ACTION]):
        a.left_arm_target[0]=HIGH[0]
        def change(t):
            t['arm']['position_rad'][0]=HIGH[0]+1
            t['arm']['header'].update(stamp_sec=-1,stamp_nanosec=2**32-1)
            t['hand']['topic']='/configured/operator_hand_input'
        alter((r,m),i,change)
    arrays,_=convert_fixture((r,m));assert (arrays['action'][:,0]==HIGH[0]).all()
    r[ACTION][3][1].left_arm_target[0]-=.01
    with pytest.raises(ValueError,match='Exact arm/hand trace target mismatch'):convert_fixture((r,m))


def test_repeated_native_stamps_and_session_changes(decoupled):
    r,m=explicit(decoupled)
    for i in range(len(r[ACTION])):
        alter((r,m),i,lambda t:[t[k]['header'].update(stamp_sec=7,stamp_nanosec=0) for k in ('arm','hand')])
    # Header time is not a callback ID; distinct sequences can share it.
    assert len(convert_fixture((r,m))[0]['action'])==9
    record,_=r[ACTION][5]
    for i in range(5,len(r[ACTION])):
        r[ACTION][i][1].producer_session_id='second'
        alter((r,m),i,lambda t:t.update(producer_session_id='second'))
    with pytest.raises(ValueError,match='Competing passive sampler session'):convert_fixture((r,m))
    r[EPISODE]=[(record-500_000,NS(data=json.dumps(dict(phase='playing',boundary_id='new',episode=m['bfm']['episode']))))]
    _,attrs=convert_fixture((r,m))
    assert attrs['producer_sessions']==['fixture','second']
    assert any(i['reason']=='episode_or_simulation_reset' for i in attrs['rejected_intervals'])


def test_historical_repeated_headers_and_equivalent_values(decoupled):
    r,m=decoupled
    for topic in (RAW_ARM,HELD[0]):
        time,msg=r[topic][4];r[topic].append((time+1,copy.deepcopy(msg)))
    arrays,attrs=convert_fixture((r,m))
    assert len(arrays['action'])==9
    assert attrs['action_provenance'][2]['evidence']['arm']['confidence']=='equivalent_values_only'


def test_historical_anchor_before_reordering_window(decoupled):
    r,m=decoupled;record,action=r[ACTION][4]
    older=copy.deepcopy(asof(r[RAW_ARM],record)[1]);newer=copy.deepcopy(older)
    newer.position[0]+=2*np.pi/4096;newer.header.stamp.nanosec+=1000
    for topic in (RAW_ARM,HELD[0]):
        r[topic]=[(record-120_000_000,copy.deepcopy(older)),(record-174_000,copy.deepcopy(newer))]
    context,association=Provenance(r,m).match(4,record,action)
    assert context[0]==record-174_000  # Truthful recorder context remains separate.
    assert association['evidence']['arm']['candidates'][0]['source_record_ns']==record-120_000_000
    # A conflicting input already established before the transport window
    # prevents revival of that older value.
    for topic in (RAW_ARM,HELD[0]):r[topic].insert(1,(record-60_000_000,copy.deepcopy(newer)))
    with pytest.raises(ValueError,match='arm missing exact'):Provenance(r,m).match(4,record,action)


def test_historical_saturation_ambiguity_rejected(decoupled):
    r,m=decoupled;record,action=r[ACTION][3];action.left_arm_target[0]=HIGH[0]
    for delta in (1,2):
        raw=copy.deepcopy(r[RAW_ARM][6][1]);raw.position[0]=HIGH[0]+delta;raw.header.stamp.nanosec+=delta
        held=copy.deepcopy(raw);held.position[0]=HIGH[0]
        r[RAW_ARM].append((record-delta*1000,raw));r[HELD[0]].append((record-delta*1000,held))
    with pytest.raises(ValueError,match='ambiguous pre-clamp'):convert_fixture((r,m))


def test_historical_interior_corruption_and_boundary_prefix(decoupled):
    r,m=decoupled;record,_=r[ACTION][3]
    r[ACTION][3][1].left_arm_target[0]+=.1
    with pytest.raises(ValueError,match='action_index=3.*arm missing exact'):convert_fixture((r,m))
    r[ACTION][3][1].left_arm_target[0]-=.1
    r[EPISODE]=[(r[ACTION][0][0]-100,NS(data=json.dumps(dict(phase='playing',boundary_id='new',episode=m['bfm']['episode']))))]
    arrays,attrs=convert_fixture((r,m))
    assert attrs['rejected_decisions'][0]['index']==0
    assert attrs['rejected_decisions'][0]['reasons']==['incomplete_boundary_command_provenance','pre_boundary_observation']
    assert r[ACTION][0][0] not in arrays['record_ns']


def test_existing_41_store_remains_valid(decoupled,tmp_path):
    arrays,attrs=convert_fixture(decoupled)
    # Pre-extension stores have neither association metadata nor the optional
    # trace stream. Their schema and prior validation remain supported.
    del attrs['action_provenance'];del attrs['command_source_role']
    del attrs['control_history']['input_provenance']
    arrays={k:v for k,v in arrays.items() if not k.startswith('control_history/input_provenance/')}
    assert write(tmp_path/'existing.zarr',arrays,attrs)['transitions']==9


def test_actual_v1_cdr_schema_and_persistent_history(decoupled,tmp_path):
    """Encode the authoritative appended fields, then use production MCAP reader."""
    from mcap_ros2.writer import Writer
    import yaml
    r,m=explicit(decoupled)
    record,_=r[ACTION][3];newer=copy.deepcopy(asof(r[RAW_ARM],record)[1])
    newer.position[0]+=2*np.pi/4096;newer.header.stamp.nanosec+=123
    for topic in (RAW_ARM,HELD[0]):r[topic].append((record-100_000,copy.deepcopy(newer)))
    for i,(_,msg) in enumerate(r[ACTION]):
        msg.action_id+=2**53
        alter((r,m),i,lambda t:t.update(action_id=t['action_id']+2**53))
    raw=tmp_path/'raw';(raw/'bag').mkdir(parents=True)
    (raw/'episode.json').write_text(json.dumps(m))
    (raw/'bag/metadata.yaml').write_text(yaml.safe_dump(dict(rosbag2_bagfile_information=dict(storage_identifier='mcap',relative_file_paths=['bag_0.mcap']))))
    deps='\n'+'='*80+'\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n'+'='*80+'\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n'
    action_def='std_msgs/Header header\n'+''.join(f'string[7] {s}_arm_joint_names\nfloat64[7] {s}_arm_target\nfloat64[6] {s}_hand_target\n' for s in JOINTS)
    action_def+='string producer_session_id\nuint64 action_id\nuint32 source_provenance_version\nstring source_provenance_sha256\n'
    defs={ACTION:('dynamic_biped/msg/BfmAction',action_def),TOPIC:('std_msgs/msg/String','string data\n'),
          CLOCK:('rosgraph_msgs/msg/Clock','builtin_interfaces/Time clock\n'),
          TOPICS[0]:('dynamic_biped/msg/BfmState','std_msgs/Header header\n'+''.join(f'float64[{n}] {k}\n' for k,(n,_) in FIELDS.items()))}
    for topic in (RAW_ARM,HELD[0],HISTORY['raw_arm'],HISTORY['left_fingers'],HISTORY['right_fingers'],TOPICS[1]):
        defs[topic]=('sensor_msgs/msg/JointState','std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n')
    for topic in (HISTORY['raw_hand'],HELD[1]):defs[topic]=('dynamic_biped/msg/RobotHandPosition','std_msgs/Header header\nuint8[6] left_hand_position\nuint8[6] right_hand_position\n')
    for topic in TOPICS[2:]:defs[topic]=('sensor_msgs/msg/Image','std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n')
    def dictionary(x):
        if isinstance(x,NS):return {k:dictionary(v) for k,v in vars(x).items()}
        return x
    with (raw/'bag/bag_0.mcap').open('wb') as f:
        writer=Writer(f);schemas={t:writer.register_msgdef(name,d+(deps if t!=TOPIC else '')) for t,(name,d) in defs.items()}
        for topic,samples in r.items():
            for record,msg in samples:writer.write_message(topic,schemas[topic],dictionary(msg),log_time=record,publish_time=record)
        writer.finish()
    decoded,*_=read_records(raw)
    assert decoded[ACTION][0][1].source_provenance_version==1
    assert plain(decoded[ACTION][0][1])['action_id']==2**53+1  # Dynamic messages have empty __dict__ plus populated slots.
    assert convert(raw,tmp_path/'converted.zarr',max_age_s=[.05]*5,collection_commit='a'*40)['transitions']==9
    root=zarr.open_group(str(tmp_path/'converted.zarr'),mode='r')
    assert root.attrs['control_history']['input_provenance']['count']==10
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema_version='4.1.0',dataset_id='wire-v1',episodes=[dict(path='converted.zarr',split='train')])))
    from bfm.data.visual_replay import VisualReplayBuffer
    batch=VisualReplayBuffer(SharpaDataset(manifest,observation_profile='vision_proprio_v1')).sample_transitions(2)
    assert batch['observation']['proprio'].shape==(2,116) and batch['action'].shape==(2,26)

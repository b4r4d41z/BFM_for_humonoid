"""Generated CDR MCAP integration tests; no claim of robot physics validation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
import zarr
import yaml
from bfm.contracts.sharpa import *
from bfm.data.mcap import ACTION, TOPICS, build, convert, decode_image
from bfm.data.sharpa import validate, write, SharpaDataset
from bfm.fb.trainer import train
from bfm.sharpa_runtime import SharpaRunner, fill_messages


def header(native=3):
    return NS(stamp=NS(sec=native, nanosec=0), frame_id="world")


@pytest.fixture
def source():
    state = {k: np.zeros(n).tolist() for k, (n, _) in FIELDS.items()}
    state.update(goal_orientation=[0., 0., 0., 1.], goal_half_extents=[.1]*3,
                 object_position=[1., 0., 0.], goal_center=[1.2, .1, 0.])
    physical = ((np.array(LOW)+HIGH)/2)
    action = NS(header=header(1793156940))
    for side in JOINTS:
        setattr(action, f"{side}_arm_joint_names", list(JOINTS[side]))
    for group, sl in GROUPS.items():
        setattr(action, f"{group}_target", physical[sl].tolist())
    names = list(JOINTS["left"]+JOINTS["right"])+["left_thumb_IP", "right_thumb_IP"]
    image = NS(header=header(), height=2, width=2, encoding="bgr8", is_bigendian=1,
               step=8, data=bytes([1, 2, 3, 4, 5, 6, 99, 99]*2))
    joint = NS(header=header(), name=names, position=[0.]*len(names), velocity=[0.]*len(names), effort=[])
    records = {k: [] for k in (ACTION, *TOPICS)}
    for i in range(10):
        t = 1_793_156_940_000_000_000 + i*100_000_000
        measured = NS(header=header(), **copy.deepcopy(state))
        measured.right_arm_position[0] = i*.01
        records[ACTION].append((t, copy.deepcopy(action)))
        for topic, msg in zip(TOPICS, [measured, joint, image, image, image]):
            records[topic].append((t-1_000_000, copy.deepcopy(msg)))
    semantics = {f"{s}_arm": {"joint_names": list(j), "unit": "rad", "dim": 7, "type": "absolute_joint_position_target"} for s, j in JOINTS.items()}
    semantics.update({f"{s}_hand": {"unit": "controller_0_100", "range": [0, 100], "dim": 6, "type": "logical_hand_target"} for s in JOINTS})
    metadata = {"time": {"synchronization_source": "rosbag_record_timestamp"}, "success": None,
                "bfm": {"contract_version": 2, "action_dim": 26, "action_group_order": list(GROUPS), "policy_hz": 10.,
                        "action_semantics": semantics, "task_state": {"frame": "world"}, "episode": {"episode_id": "generated"}}}
    return records, metadata


def make(source):
    return build(*copy.deepcopy(source), max_age_s=[.05]*5, collection_commit=SOURCE_COMMIT)


def test_layout_state_and_action_roundtrip(source):
    records, _ = source
    a = flatten_action(records[ACTION][0][1])
    assert np.allclose(ActionNormalizer().physical(ActionNormalizer().normalize(a)), a, atol=2e-6)
    assert GROUPS["right_arm"] == slice(13, 20)
    s = flatten_state(records[TOPICS[0]][0][1])
    assert np.array_equal(flatten_state(unflatten_state(s)), s) and len(s) == 56
    arm, hand, act = fill_messages(a, NS(), NS(), NS())
    assert len(arm.position) == 14 and arm.name == list(JOINTS["left"]+JOINTS["right"])
    assert act.left_hand_target == hand.left_hand_position
    assert flatten_action(act).shape == (26,)


def test_causal_clock_missing_tick_and_no_future_leak(source):
    records, metadata = copy.deepcopy(source)
    t = records[ACTION][4][0]
    records[TOPICS[0]] = [s for s in records[TOPICS[0]] if s[0] != t-1_000_000]
    future = copy.deepcopy(records[TOPICS[0]][0][1]); future.object_position = [99.]*3
    records[TOPICS[0]].append((t+1, future))
    arrays, m = make((records, metadata))
    assert t not in arrays['record_ns'] and t not in arrays['next_record_ns']
    assert arrays['truncated'].sum() == 2  # Gap splits, never jumps over tick 4.
    assert (arrays['obs/sync_age_ns'] >= 0).all()
    assert m['source_metadata']['success'] is None and not arrays['terminated'].any()
    assert arrays['obs/source_header_ns'][0, 0] == 3_000_000_000


@pytest.mark.parametrize('encoding,expected', [('rgb8',[1,2,3]),('bgr8',[3,2,1]),('rgba8',[1,2,3]),('bgra8',[3,2,1]),('mono8',[1,1,1])])
def test_raw_decode_padding_channels_and_endianness(encoding, expected):
    n = 1 if encoding == 'mono8' else 4 if 'a8' in encoding else 3
    image = NS(height=1, width=1, encoding=encoding, is_bigendian=1, step=n+2, data=bytes(range(1,n+1))+b'xx')
    assert decode_image(image).tolist() == [[expected]]
    image.encoding = 'mono16'
    with pytest.raises(ValueError): decode_image(image)


def test_zarr_dataset_model_checkpoint_and_runtime(source, tmp_path):
    arrays, m = make(source)
    path = tmp_path/'episode.zarr'
    write(path, arrays, m)
    root = zarr.open_group(str(path), mode='r')
    assert validate(root)['action_dim'] == 26
    assert np.array_equal(root['action'][:], arrays['action'])  # raw not normalized
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'schema_version':VERSION,'dataset_id':'generated','episodes':[{'path':path.name,'split':'train'}]}))
    data = SharpaDataset(manifest)
    assert data.episodes[0]['observation'].shape == (9,56)
    config = {'dataset_schema':'sharpa_v3','dataset_manifest':str(manifest),'output_dir':str(tmp_path/'run'),
              'batch_size':4,'training_steps':4,'logging_interval':1,'checkpoint_interval':2,
              'model':{'state_dim':56,'action_dim':26,'hidden_dim':16,'latent_dim':4}}
    checkpoint = train(config)
    runner = SharpaRunner(checkpoint)
    action = runner.act(source[0][TOPICS[0]][0][1])
    assert action.shape == (26,) and np.isfinite(action).all()
    assert torch.isfinite(runner.z).all()
    assert not any(p.requires_grad for p in runner.model.parameters())
    train({**config, 'training_steps':6}, resume=checkpoint)
    # A source episode cannot be assigned to another split even by another path.
    spec = json.loads(manifest.read_text()); spec['episodes'].append({'path':path.name,'split':'test'})
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError): SharpaDataset(manifest)


@pytest.mark.parametrize('bad', ['schema','order','future','age','nan','quaternion','extent','range','duplicate','boundary','continuity','camera','joints'])
def test_reject_invalid_store(source,tmp_path,bad):
    a,m = make(source)
    if bad=='schema': m['schema_version']='2.0.0'
    if bad=='order': m['contract']['groups'].reverse()
    if bad=='future': a['obs/source_record_ns'][0,0] += 2_000_000
    if bad=='age': a['obs/sync_age_ns'][0,0] = 999999999
    if bad=='nan': a['obs/state'][0,0] = np.nan
    if bad=='quaternion': a['obs/state'][0,SLICES['goal_orientation']] = 0
    if bad=='extent': a['obs/state'][0,SLICES['goal_half_extents']] = -1
    if bad=='range': a['action'][0,GROUPS['left_hand']] = 101
    if bad=='duplicate': a['record_ns'][1] = a['record_ns'][0]
    if bad=='boundary': a['terminated'][-1] = True
    if bad=='continuity': a['next_obs/state'][0,0] += 1
    if bad=='camera': a['obs/images/head'] = a['obs/images/head'].astype(float)
    if bad=='joints': m['joint_names'][0]=m['joint_names'][1]
    with pytest.raises(ValueError): write(tmp_path/'bad.zarr',a,m)


def test_missing_topics_and_held_command_checks(source):
    records, meta = copy.deepcopy(source)
    records.pop(TOPICS[0])
    with pytest.raises(ValueError, match='Missing required topic'): make((records,meta))
    records, meta = copy.deepcopy(source)
    records['/robot_arm_joint_command'] = []
    for t, action in records[ACTION]:
        a = flatten_action(action)
        arm, _, _ = fill_messages(a,NS(),NS(),NS())
        records['/robot_arm_joint_command'].append((t,arm))
    assert make((records,meta))[1]['held_command_checks']['/robot_arm_joint_command']==10
    records['/robot_arm_joint_command'][0][1].position[0] += .5
    with pytest.raises(ValueError,match='Held command'): make((records,meta))


def test_real_embedded_schema_cdr_mcap_reader(source,tmp_path):
    from mcap_ros2.writer import Writer
    records, meta = source
    raw = tmp_path/'record'
    (raw/'bag').mkdir(parents=True)
    (raw/'episode.json').write_text(json.dumps(meta))
    bag_meta = {'rosbag2_bagfile_information':{'storage_identifier':'mcap','relative_file_paths':['bag_0.mcap']}}
    (raw/'bag/metadata.yaml').write_text(yaml.safe_dump(bag_meta))
    deps = '\n'+'='*80+'\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n'+'='*80+'\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n'
    defs = {ACTION:'std_msgs/Header header\n'+''.join(f'string[7] {s}_arm_joint_names\nfloat64[7] {s}_arm_target\nfloat64[6] {s}_hand_target\n' for s in JOINTS),
            TOPICS[0]:'std_msgs/Header header\n'+''.join(f'float64[{w}] {k}\n' for k,(w,_) in FIELDS.items()),
            TOPICS[1]:'std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n'}
    for topic in TOPICS[2:]:
        defs[topic]='std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n'
    def dictionary(v):
        return {k:dictionary(x) for k,x in vars(v).items()} if isinstance(v,NS) else v
    with (raw/'bag/bag_0.mcap').open('wb') as out:
        writer = Writer(out)
        schemas = {t:writer.register_msgdef('test_msgs/msg/Message'+str(i),definition+deps) for i,(t,definition) in enumerate(defs.items())}
        for topic, samples in records.items():
            for t,msg in samples:
                writer.write_message(topic,schemas[topic],dictionary(msg),log_time=t,publish_time=7)
        writer.finish()
    output=tmp_path/'from_mcap.zarr'
    assert convert(raw,output,max_age_s=[.05]*5,collection_commit=SOURCE_COMMIT)['transitions']==9

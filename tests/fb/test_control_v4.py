import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import zarr
from test_sharpa import source
from test_runtime_tasks import trained, visual_manifest
from bfm.contracts.sharpa import (SharpaV4Contract, SharpaContract, SOURCE_COMMIT, MODEL_SHA256, GROUPS, JOINTS, execution_configuration)
from bfm.data.mcap import build, ACTION, TOPICS, HELD, CLOCK
from bfm.data.execution_v4 import STATUS, FINGERS
from bfm.data.policy_time import SimPolicyClock, policy_timing, continuity_reason
from bfm.data.sharpa import write, validate, SharpaDataset
from bfm.fb.execution import ExecutionMonitor, fill_action, output_types


def stamp(t):return NS(stamp=NS(sec=t//10**9,nanosec=t%10**9),frame_id='world')


def execution():
    content=(Path(__file__).parents[2]/'configs/data_contract/bfm_executor_v4.json').read_bytes()
    return dict(configuration=json.loads(content),controller_config_sha256=hashlib.sha256(content).hexdigest(),entry_topic=ACTION,status_topic=STATUS)


@pytest.fixture
def v4(source):
    records,meta=copy.deepcopy(source);origin=records[ACTION][0][0]
    for topic,samples in records.items():
        records[topic]=[(origin+i*10**9//15+(t-origin-i*100_000_000),m) for i,(t,m) in enumerate(samples)]
    records[STATUS]=[];records[HELD[1]]=[]
    for i,(t,m) in enumerate(records[ACTION]):
        m.header=stamp(i*10**9//15);m.producer_session_id='test';m.action_id=i+1
        records[STATUS].append((t+500,NS(header=stamp(i*10**9//15),producer_session_id='test',action_id=i+1,accepted=True,reason='accepted_not_attained',controller_config_sha256=execution()['controller_config_sha256'])))
        records[HELD[1]].append((t+600,NS(header=stamp(i*10**9//15),left_hand_position=[50]*6,right_hand_position=[50]*6)))
    initial=records[HELD[0]][0][1];records[HELD[0]]=[]
    for tick in range(61):
        msg=copy.deepcopy(initial);msg.header=stamp(tick*10_000_000);records[HELD[0]].append((origin+tick*10_000_000+1000,msg))
    for side,topic in zip(['left','right'],FINGERS):
        angles={side+'_thumb_MCP_FE':25,side+'_thumb_IP':32.5,side+'_thumb_CMC_FE':45,side+'_thumb_CMC_AA':-20}
        for finger in ('index','middle','ring','pinky'):
            for joint,scale in [('MCP_FE',100 if finger=='index' else 90),('PIP',100),('DIP',100)]:angles[side+'_'+finger+'_'+joint]=scale*.5
        records[topic]=[(origin+tick*20_000_000+2000,NS(header=stamp(tick*20_000_000),name=list(angles),position=(np.deg2rad(list(angles.values()))*(1 if side=='left' else -1)).tolist())) for tick in range(31)]
    records[CLOCK]=[(origin+i*10**9//300,NS(clock=stamp(i*10**9//300).stamp)) for i in range(181)]
    meta.update(recording_code={'commit_sha':SOURCE_COMMIT},camera_configuration={'model_sha256':MODEL_SHA256})
    meta['bfm'].update(contract_version=4,policy_hz=15,policy_dt=1/15,execution=execution(),policy_clock=dict(domain='isaac_simulation',topic=CLOCK,period_ns_numerator=10**9,period_ns_denominator=15))
    return records,meta


def convert_fixture(v4):return build(*copy.deepcopy(v4),max_age_s=[.05]*5,collection_commit=SOURCE_COMMIT)


def test_v4_store_and_legacy_identity(v4,tmp_path):
    a,m=convert_fixture(v4);assert m['schema_version']=='4.0.0' and a['action'].shape==(9,26)
    assert m['contract']==SharpaV4Contract().to_dict()!=SharpaContract().to_dict()
    np.testing.assert_allclose(a['duration'],1/15,atol=1e-9)
    assert (a['obs/source_record_ns']<=a['record_ns'][:,None]).all()
    path=tmp_path/'v4.zarr';write(path,a,m);assert validate(zarr.open_group(str(path),mode='r'))['transitions']==9
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema_version='4.0.0',dataset_id='v4',episodes=[dict(path=path.name,split='train')])))
    data=SharpaDataset(manifest,observation_profile='vision_proprio_v1');assert data.contract.policy_dt==1/15
    spec=json.loads(manifest.read_text());spec['schema_version']='3.0.0';manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='schema'):SharpaDataset(manifest)


@pytest.mark.parametrize('fault',['missing','duplicate','reject','hash','command','future','skip','session','groups','extra_hand'])
def test_execution_fail_closed(v4,fault):
    r,m=v4
    if fault=='missing':r[STATUS].pop(3)
    if fault=='duplicate':r[STATUS].append(r[STATUS][3])
    if fault=='reject':r[STATUS][3][1].accepted=False
    if fault=='hash':r[STATUS][3][1].controller_config_sha256='x'*64
    if fault=='command':r[HELD[0]][20][1].position[0]+=.5
    if fault=='future':
        for topic in TOPICS:r[topic]=[(t+10**9,msg) for t,msg in r[topic]]
    if fault=='skip':
        for _,msg in r[ACTION][4:]:msg.header=stamp(msg.header.stamp.nanosec+100_000_000)
    if fault=='session':r[ACTION][4][1].producer_session_id='other'
    if fault=='groups':
        r[ACTION][4][1].left_hand_target.pop();r[ACTION][4][1].right_hand_target.append(50)
    if fault=='extra_hand':
        record,msg=copy.deepcopy(r[HELD[1]][3]);msg.header=stamp(210_000_000);r[HELD[1]].append((record+1,msg))
    if fault=='skip':
        with pytest.raises(ValueError):convert_fixture(v4)
    else:
        with pytest.raises(ValueError):convert_fixture(v4)


def test_rational_clock_pause_reset_and_discount():
    clock=SimPolicyClock();ticks=[]
    for i in range(3001):
        ns=i*10**9//300
        if clock.advance(ns):ticks.append(ns)
        assert not clock.advance(ns)
    assert len(ticks)==150 and ticks[-1]==10*10**9
    assert not clock.advance(0);assert clock.advance(10**9//15)
    assert clock.advance(10**10);assert not clock.advance(10**10)
    assert .993322**15==pytest.approx(.99**10,abs=1e-5)


def test_ack_watchdog_and_exact_physical_action():
    now=[0.];monitor=ExecutionMonitor(execution(),clock=lambda:now[0]);a=np.array([-1.]*7+[20.5]*6+[-1.]*7+[30.5]*6);a[1]=.1;a[14]=-.1
    msg=fill_action(a,NS(header=stamp(0)));monitor.publish(msg,0)
    assert msg.left_hand_target==[20.5]*6
    ack=NS(producer_session_id=monitor.session,action_id=1,header=stamp(0),accepted=True,reason='accepted_not_attained',controller_config_sha256=monitor.digest)
    monitor.accept(ack);assert not monitor.pending
    with pytest.raises(RuntimeError):monitor.accept(ack)
    monitor.publish(NS(header=stamp(0)),100_000_000);now[0]=1
    with pytest.raises(RuntimeError,match='deadline'):monitor.check()


def test_v4_embedded_cdr_mcap(v4,tmp_path):
    from mcap_ros2.writer import Writer
    import yaml
    from bfm.data.mcap import convert, RAW_ARM
    from bfm.contracts.sharpa import FIELDS
    r,meta=v4;raw=tmp_path/'raw';(raw/'bag').mkdir(parents=True)
    (raw/'episode.json').write_text(json.dumps(meta))
    (raw/'bag/metadata.yaml').write_text(yaml.safe_dump({'rosbag2_bagfile_information':{'storage_identifier':'mcap','relative_file_paths':['bag_0.mcap']}}))
    deps='\n'+'='*80+'\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n'+'='*80+'\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n'
    defs={ACTION:'std_msgs/Header header\nstring producer_session_id\nuint64 action_id\n'+''.join(f'string[7] {s}_arm_joint_names\nfloat64[7] {s}_arm_target\nfloat64[6] {s}_hand_target\n' for s in JOINTS),
          STATUS:'std_msgs/Header header\nstring producer_session_id\nuint64 action_id\nbool accepted\nstring reason\nstring controller_config_sha256\n',
          TOPICS[0]:'std_msgs/Header header\n'+''.join(f'float64[{n}] {k}\n' for k,(n,_) in FIELDS.items()),
          TOPICS[1]:'std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n',
          CLOCK:'builtin_interfaces/Time clock\n',HELD[1]:'std_msgs/Header header\nuint8[6] left_hand_position\nuint8[6] right_hand_position\n'}
    for topic in (HELD[0],RAW_ARM,*FINGERS):defs[topic]='std_msgs/Header header\nstring[] name\nfloat64[] position\n'
    for topic in TOPICS[2:]:defs[topic]='std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n'
    def dictionary(x):return {k:dictionary(v) for k,v in vars(x).items()} if isinstance(x,NS) else x
    with (raw/'bag/bag_0.mcap').open('wb') as f:
        writer=Writer(f);schemas={t:writer.register_msgdef('test_msgs/msg/M'+str(i),d+deps) for i,(t,d) in enumerate(defs.items())}
        for topic,samples in r.items():
            for record,msg in samples:writer.write_message(topic,schemas[topic],dictionary(msg),log_time=record,publish_time=record)
        writer.finish()
    assert convert(raw,tmp_path/'converted.zarr',max_age_s=[.05]*5,collection_commit=SOURCE_COMMIT)['transitions']==9


def test_runtime_gate_shadow_and_config(trained,tmp_path):
    from bfm.fb.runtime import VisualRuntime, validate_ros_config
    from bfm.fb.checkpoint import load_payload
    import yaml
    runtime,_,checkpoint=trained
    with pytest.raises(ValueError,match='v4-trained'):runtime.require_v4()
    p=copy.deepcopy(runtime.payload);p.update(format='vanilla_fb_sharpa_v4',schema_version='4.0.0',contract=SharpaV4Contract().to_dict(),execution=execution())
    path=tmp_path/'fixture-v4.pt';torch.save(p,path)
    restored=VisualRuntime(path);restored.require_v4()
    p['execution']['controller_config_sha256']='x'*64;torch.save(p,tmp_path/'bad.pt')
    with pytest.raises(ValueError):load_payload(tmp_path/'bad.pt')
    cfg=yaml.safe_load((Path(__file__).parents[2]/'configs/runtime/isaac_ros2.yaml').read_text());validate_ros_config(cfg)
    assert cfg['output_topics']=={'action':'/bfm/action'} and cfg['policy_hz']==15
    assert output_types(4,'canonical','arm','hand')=={'action':'canonical'}
    assert set(output_types(3,'canonical','arm','hand'))=={'action','arm','hand'}
    from bfm.fb.evaluation import Evaluation, RolloutLogger
    class Buffer:
        def diagnostics(self,t):return {}
        def snapshot(self,t):return {'proprio':np.zeros(116)}
    restored.act=lambda obs:np.zeros(26)
    published=[]
    e=Evaluation(restored,Buffer(),RolloutLogger(tmp_path/'shadow',mode='shadow',checkpoint=path,task_z='fixture'),mode='shadow',warmup_ticks=0,max_policy_steps=1,control_contract_version=4)
    e.tick(published.append);assert published==[] and e.safety.limit is None
    e=Evaluation(restored,Buffer(),RolloutLogger(tmp_path/'closed',mode='closed-loop',checkpoint=path,task_z='fixture'),mode='closed-loop',warmup_ticks=1,max_policy_steps=2,control_contract_version=4)
    e.tick(published.append);assert published==[]
    e.tick(published.append);assert len(published)==1



def test_interpolated_arm_commands_are_not_policy_targets(v4):
    r,m=v4
    baseline=r[HELD[0]][0][1].position[0]
    for _,action in r[ACTION]:action.left_arm_target[0]=baseline+1
    for tick,(_,command) in enumerate(r[HELD[0]]):
        command.position[0]=baseline+.01*sum(min(.02*i,1.) for i in range(1,tick+2))
    arrays,_=convert_fixture(v4)
    assert len(arrays['action'])==9
    assert r[HELD[0]][20][1].position[0]!=r[ACTION][2][1].left_arm_target[0]
    r[HELD[0]]=r[HELD[0]][:15]
    with pytest.raises(ValueError,match='coverage'):convert_fixture(v4)


def test_boundary_and_stale_ticks_do_not_bridge(v4,tmp_path):
    from bfm.data.mcap import EPISODE
    r,m=v4;origin=r[ACTION][0][0]
    def event(boundary):return NS(data=json.dumps(dict(phase='playing',boundary_id=boundary,episode={'episode_id':'generated'})))
    r[EPISODE]=[(origin-2_000_000,event('first')),(r[ACTION][5][0]-100,event('second'))]
    a,attrs=convert_fixture(v4)
    assert len(a['action'])==8 and a['truncated'].sum()==2
    write(tmp_path/'boundaries.zarr',a,attrs)
    r[TOPICS[2]].pop(2)
    a,attrs=convert_fixture(v4)
    assert attrs['rejected_ticks']==1 and attrs['rejected_decisions'][0]['index']==2
    time=r[ACTION][2][0]
    assert time not in a['record_ns'] and time not in a['next_record_ns']

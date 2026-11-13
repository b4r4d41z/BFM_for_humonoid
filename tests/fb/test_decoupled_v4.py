"""Generated decoupled timelines; Synthetic 131cfb5 interface, no ROS/Isaac needed."""
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
from bfm.contracts.sharpa import (DECOUPLED,DECOUPLED_SCHEMA,MODEL_SHA256,LOW,HIGH,JOINTS,GROUPS,FIELDS,
                                  SharpaDecoupledContract,decoupled_control)
from bfm.data.mcap import ACTION,RAW_ARM,HELD,TOPICS,CLOCK,build
from bfm.data.decoupled import HISTORY,plain,validate_history
from bfm.data.sharpa import write,validate,SharpaDataset
from bfm.data.visual_replay import VisualReplayBuffer


def header(ns):return NS(stamp=NS(sec=ns//10**9,nanosec=ns%10**9),frame_id='world')


@pytest.fixture
def decoupled(source):
    r,m=copy.deepcopy(source);origin=r[ACTION][0][0]
    for topic,samples in r.items():r[topic]=[(origin+i*10**9//15+(t-origin-i*100_000_000),msg) for i,(t,msg) in enumerate(samples)]
    arm=copy.deepcopy(r[HELD[0]][0][1]);hand=copy.deepcopy(r[HELD[1]][0][1])
    r[RAW_ARM]=[];r[HELD[0]]=[];r[HISTORY['raw_arm']]=[];r[HELD[1]]=[];r[HISTORY['raw_hand']]=[]
    for i,(record,action) in enumerate(r[ACTION]):
        action.header=header(i*10**9//15);action.producer_session_id='fixture';action.action_id=i+1
        action.left_arm_target[0]+=.005*i
        for offset,change in [(-2_000_000,0),(30_000_000,.002)]:
            target=copy.deepcopy(arm);target.header=header(400_000_000_000+2*i+(offset>0));target.position[0]=action.left_arm_target[0]+change
            r[HELD[0]].append((record+offset,target));r[RAW_ARM].append((record+offset+100,copy.deepcopy(target)))
            r[HISTORY['raw_arm']].append((record+offset-100,copy.deepcopy(target)))
        logical=copy.deepcopy(hand);logical.header=header(600_000_000_000+i)
        r[HISTORY['raw_hand']].append((record-1_000_000,copy.deepcopy(logical)));r[HELD[1]].append((record-999_900,logical))
    for side in ('left','right'):
        r[HISTORY[side+'_fingers']]=[(origin+j*20_000_000,NS(header=header(700_000_000_000+j*20_000_000),name=[side+'_index_PIP'],position=[.1])) for j in range(32)]
    r[CLOCK]=[(origin+j*10**9//60,NS(clock=header(j*10**9//60).stamp)) for j in range(39)]
    m['camera_configuration']={'model_sha256':MODEL_SHA256};m['success']=True
    m['robot']='kuavo_s40_sharpa';m['recording_code']={'commit_sha':'a'*40,'status':'available','dirty':True,'dirty_sha256':'b'*64}
    m['bfm'].update(contract_version=4,control_architecture=DECOUPLED,action_role='sampled_operator_target_not_execution_acknowledgement',policy_hz=15,policy_dt=1/15,
                    policy_clock=dict(domain='isaac_simulation',topic=CLOCK,period_ns_numerator=10**9,period_ns_denominator=15),
                    control=dict(physics_hz=60,hand_hz=50,hand_clock='steady_wall_time',execution_acknowledgements=False,sampling_independent_of_control=True))
    m['bfm']['arm_target_processing'].update(clamping_enabled=True,transformation='per_joint_saturation',unit='rad',bounds_source_sha256=MODEL_SHA256,
        joint_limits=dict(zip(JOINTS['left']+JOINTS['right'],[[LOW[i],HIGH[i]] for i in np.r_[0:7,13:20]])))
    return r,m


def convert_fixture(fixture):return build(*fixture,max_age_s=[.05]*5,collection_commit='a'*40)


def dataset(tmp_path,fixture):
    a,m=convert_fixture(fixture);path=tmp_path/'episode.zarr';write(path,a,m)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema_version=DECOUPLED_SCHEMA,dataset_id='fixture',episodes=[dict(path=path.name,split='train')])))
    return SharpaDataset(manifest,observation_profile='vision_proprio_v1'),manifest


def test_history_causality_replay_and_cache_identity(decoupled,tmp_path):
    data,manifest=dataset(tmp_path,decoupled);root=data.roots[0]
    assert validate(root)==dict(transitions=9,segments=1,state_dim=56,action_dim=26)
    assert root.attrs['contract']==SharpaDecoupledContract().to_dict() and 'execution' not in root.attrs
    assert len(root['action'])==9 and root['intervening_arm_max_delta'][:].max()>.001
    for pre in ('obs','next_obs'):
        times=root['record_ns' if pre=='obs' else 'next_record_ns'][:]
        assert (root[f'{pre}/source_record_ns'][:]<=times[:,None]).all()
    for name,topic in HISTORY.items():
        g=root['control_history/'+name];samples=decoupled[0].get(topic,[])
        assert len(g['record_ns'])==len(samples)
        offsets=g['offsets'][:];payload=g['payload'][:].tobytes()
        for i,(_,msg) in enumerate(sorted(samples,key=lambda x:x[0])):assert json.loads(payload[offsets[i]:offsets[i+1]])==plain(msg)
    batch=VisualReplayBuffer(data).sample_transitions(2)
    assert batch['observation']['proprio'].shape==(2,116) and batch['action'].shape==(2,26)
    np.testing.assert_allclose(batch['duration'],1/15,atol=1e-7)
    assert all(batch['observation'][c].dtype==torch.uint8 for c in ('head','left_wrist','right_wrist'))
    original=data.fingerprint
    g=zarr.open_group(str(manifest.parent/'episode.zarr'),mode='a')['control_history/raw_arm']
    payload=g['payload'][:];payload[-2]=ord(' ');g['payload'][:]=payload
    with pytest.raises(ValueError,match='Corrupt'):SharpaDataset(manifest)
    # Even a correctly re-hashed history edit invalidates dataset/cache identity.
    payload=np.frombuffer(b''.join(json.dumps(plain(msg),sort_keys=True,separators=(',',':'),allow_nan=False).encode() for _,msg in sorted(decoupled[0][HISTORY['raw_arm']],key=lambda s:s[0])),dtype=np.uint8).copy()
    mark=payload.tobytes().index(b'-1.');payload[mark+1]=ord('0');g['payload'][:]=payload
    rootw=zarr.open_group(str(manifest.parent/'episode.zarr'),mode='a');meta=dict(rootw.attrs['control_history']);meta['raw_arm']['sha256']=hashlib.sha256(payload.tobytes()).hexdigest();rootw.attrs['control_history']=meta
    assert SharpaDataset(manifest,observation_profile='vision_proprio_v1').fingerprint!=original


@pytest.mark.parametrize('fault',['ack_metadata','missing_arm','bad_action','bad_clamp','future_source','nan','bad_clock','future_observation','duplicate_stamp'])
def test_invalid_evidence(decoupled,fault):
    r,m=decoupled
    if fault=='duplicate_stamp':r[ACTION][2][1].header=copy.deepcopy(r[ACTION][1][1].header)
    if fault=='ack_metadata':m['bfm']['execution']={}
    if fault=='missing_arm':r[HELD[0]]=[]
    if fault=='bad_action':r[ACTION][2][1].left_arm_target[0]+=.3
    if fault=='bad_clamp':r[HELD[0]][3][1].position[0]+=.3
    if fault=='future_source':r[RAW_ARM]=[(t+10**9,msg) for t,msg in r[RAW_ARM]]
    if fault=='nan':r[TOPICS[1]][2][1].velocity[0]=float('nan')
    if fault=='bad_clock':m['bfm']['control']['physics_hz']=float('nan')
    if fault=='future_observation':
        for topic in TOPICS:r[topic]=[(t+10**9,msg) for t,msg in r[topic]]
    with pytest.raises(ValueError):convert_fixture(decoupled)


@pytest.mark.parametrize('hz',[60,100,120,300])
def test_configurable_physics_quantization(decoupled,hz):
    r,m=decoupled;m['bfm']['control']['physics_hz']=hz;origin=r[ACTION][0][0]
    r[CLOCK]=[(origin+j*10**9//hz,NS(clock=header(j*10**9//hz).stamp)) for j in range(hz+1)]
    for i,(_,msg) in enumerate(r[ACTION]):msg.header=header(((i*hz+14)//15)*10**9//hz)
    a,attrs=convert_fixture(decoupled)
    assert len(a['action'])==9 and attrs['control']['physics_hz']==hz
    assert (a['duration']>0).all()
    np.testing.assert_allclose(a['duration'],(a['next_action_header_ns']-a['action_header_ns'])/1e9)


def test_skips_stale_and_lifecycle_not_bridged(decoupled,tmp_path):
    from bfm.data.mcap import EPISODE
    r,m=decoupled;origin=r[ACTION][0][0]
    r[TOPICS[2]].pop(2)
    a,attrs=convert_fixture(decoupled)
    assert attrs['rejected_decisions'][0]['index']==2 and len(a['action'])==7
    assert a['truncated'].sum()==2
    write(tmp_path/'gaps.zarr',a,attrs)
    # A missed policy message removes the interval, rather than inventing a held target.
    r[ACTION].pop(5);a,attrs=convert_fixture(decoupled)
    assert any(e['reason']=='skipped_policy_interval' for e in attrs['rejected_intervals'])


def test_checkpoint_task_binding_and_shadow(trained,decoupled,tmp_path):
    from bfm.fb.runtime import VisualRuntime,validate_ros_config
    from bfm.fb.latents import episode_embeddings,save_task
    from bfm.fb.checkpoint import save_checkpoint,load_payload
    from bfm.fb.evaluation import Evaluation,RolloutLogger
    from bfm.fb.execution import PolicyIdentity,fill_action,output_types
    import yaml
    runtime,_,checkpoint=trained
    with pytest.raises(ValueError,match='decoupled-trained'):runtime.require_decoupled()
    data,manifest=dataset(tmp_path/'converted',decoupled)
    from bfm.fb.trainer import train
    from test_vision import mock_config
    ckpt=train(dict(dataset_schema='sharpa_decoupled_v4',allow_sampled_control_approximation=True,observation_profile='vision_proprio_v1',
                    dataset_manifest=str(manifest),output_dir=str(tmp_path/'new-run'),batch_size=2,training_steps=1,
                    checkpoint_interval=1,reference_count=2,vision=mock_config(),model=dict(hidden_dim=16,latent_dim=4),train=dict(discount=.993322)))
    p=load_payload(ckpt)
    assert p['contract']['policy_dt']==1/15 and p['robot']=='kuavo_s40_sharpa' and p['train_config']['discount']==.993322
    restored=VisualRuntime(ckpt);restored.require_decoupled(60)
    with pytest.raises(ValueError,match='physics'):restored.require_decoupled(300)
    with pytest.raises(ValueError,match='architecture mismatch'):episode_embeddings(runtime,manifest)
    episodes=episode_embeddings(restored,manifest,3);task=tmp_path/'task.pt';save_task(restored,episodes,task);restored.load_task(task,require_bound=True)
    obs=VisualReplayBuffer(data).observations([0]);assert restored.act(obs).shape==(26,)
    from bfm.data.feature_cache import build_cache,FeatureCache
    cache=build_cache(data,restored.model.observation_encoder,tmp_path/'cache',batch_size=3)
    replay=VisualReplayBuffer(data);replay.feature_cache=FeatureCache(cache,data,restored.model.observation_encoder)
    cached=replay.sample_transitions(2)['observation']
    assert set(cached)=={'proprio','features/head','features/left_wrist','features/right_wrist'}
    from bfm.fb.reward_weighted_z import build as build_reward
    template=tmp_path/'rewards.json';build_reward(restored,manifest,template,template=True,batch_size=3)
    annotations=json.loads(template.read_text())
    for entry in annotations['episodes']:entry['rewards']=list(range(entry['states']))
    template.write_text(json.dumps(annotations));build_reward(restored,manifest,tmp_path/'weighted.pt',annotations=template,batch_size=3)
    restored.load_task(tmp_path/'weighted.pt',require_bound=True)
    from bfm.fb.latent_analysis import analyze
    from bfm.fb.cluster_analysis import build_clusters
    analyze(restored,manifest,task,tmp_path/'analysis',3)
    build_clusters(restored,manifest,tmp_path/'analysis/episode_z.pt',tmp_path/'clusters',1,3)
    restored.load_task(tmp_path/'clusters/cluster_0_task_z.pt',require_bound=True)
    cfg=yaml.safe_load((Path(__file__).parents[2]/'configs/runtime/isaac_ros2.yaml').read_text());validate_ros_config(cfg)
    assert 'execution_status_topic' not in cfg and output_types(4,'A','B','C')=={'action':'A'}
    state=PolicyIdentity();msg=fill_action(np.asarray(decoupled[0][ACTION][0][1].left_arm_target+[50]*6+decoupled[0][ACTION][0][1].right_arm_target+[50]*6),NS(header=header(0)))
    state.publish(msg,0);assert msg.action_id==1 and not hasattr(state,'pending')
    with pytest.raises(ValueError):state.publish(msg,0)
    class Buffer:
        def diagnostics(self,now):return {}
        def snapshot(self,now):return obs
    log=RolloutLogger(tmp_path/'shadow',mode='shadow',checkpoint=ckpt,task_z=task);calls=[]
    evaluation=Evaluation(restored,Buffer(),log,mode='shadow',warmup_ticks=0,max_policy_steps=1,control_contract_version=4,control_architecture=DECOUPLED)
    evaluation.tick(calls.append);assert not calls and evaluation.safety.limit is None
    p['control']['execution_acknowledgements']=True;torch.save(p,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='control'):load_payload(tmp_path/'bad.pt')


def test_generated_cdr_conversion_and_batch_reuse(decoupled,tmp_path):
    from mcap_ros2.writer import Writer
    import yaml
    from bfm.data.mcap import convert
    from bfm.data.recordings import prepare
    r,m=decoupled;raw=tmp_path/'records'/'generated';(raw/'bag').mkdir(parents=True)
    (raw/'episode.json').write_text(json.dumps(m))
    (raw/'bag/metadata.yaml').write_text(yaml.safe_dump({'rosbag2_bagfile_information':dict(storage_identifier='mcap',relative_file_paths=['bag_0.mcap'])}))
    deps='\n'+'='*80+'\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n'+'='*80+'\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n'
    defs={ACTION:'std_msgs/Header header\nstring producer_session_id\nuint64 action_id\n'+''.join(f'string[7] {s}_arm_joint_names\nfloat64[7] {s}_arm_target\nfloat64[6] {s}_hand_target\n' for s in JOINTS),
          TOPICS[0]:'std_msgs/Header header\n'+''.join(f'float64[{n}] {k}\n' for k,(n,_) in FIELDS.items()),
          TOPICS[1]:'std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n',CLOCK:'builtin_interfaces/Time clock\n'}
    for topic in (HELD[0],RAW_ARM,HISTORY['raw_arm'],HISTORY['left_fingers'],HISTORY['right_fingers']):defs[topic]='std_msgs/Header header\nstring[] name\nfloat64[] position\n'
    for topic in (HELD[1],HISTORY['raw_hand']):defs[topic]='std_msgs/Header header\nuint8[6] left_hand_position\nuint8[6] right_hand_position\n'
    for topic in TOPICS[2:]:defs[topic]='std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n'
    def dictionary(x):return {k:dictionary(v) for k,v in vars(x).items()} if isinstance(x,NS) else x
    with (raw/'bag/bag_0.mcap').open('wb') as f:
        writer=Writer(f);schemas={t:writer.register_msgdef('test_msgs/msg/M'+str(i),d+deps) for i,(t,d) in enumerate(defs.items())}
        for topic,samples in r.items():
            for record,msg in samples:writer.write_message(topic,schemas[topic],dictionary(msg),log_time=record,publish_time=record)
        writer.finish()
    assert convert(raw,tmp_path/'converted.zarr',max_age_s=[.05]*5,collection_commit='a'*40)['transitions']==9
    reports=[];manifest=prepare(raw.parent,tmp_path/'batch',dict(max_age_s=[.05]*5),expected_version=4,expected_architecture=DECOUPLED,report=lambda *a:reports.append(a))
    data=SharpaDataset(manifest,observation_profile='vision_proprio_v1');assert data.schema_version==DECOUPLED_SCHEMA
    prepare(raw.parent,tmp_path/'batch',dict(max_age_s=[.05]*5),expected_version=4,expected_architecture=DECOUPLED,report=lambda *a:reports.append(a))
    assert reports[-1][-1]=='skipped'


def test_reset_and_stop_boundary(decoupled,tmp_path):
    from bfm.data.mcap import EPISODE
    r,m=decoupled;boundary=r[ACTION][5][0]-1_000_000
    def event(phase,name):return NS(data=json.dumps(dict(phase=phase,boundary_id=name,episode={'episode_id':'generated'})))
    r[EPISODE]=[(r[ACTION][0][0]-4_000_000,event('playing','a')),(boundary,event('stopped','a')),
                (boundary+100,event('playing','b'))]
    a,attrs=convert_fixture(decoupled)
    assert a['truncated'].sum()==2 and not a['terminated'].any()
    assert any('pre_boundary_observation' in d['reasons'] or 'incomplete_boundary_command_provenance' in d['reasons'] for d in attrs['rejected_decisions'])
    write(tmp_path/'boundary.zarr',a,attrs)
    r.pop(EPISODE)
    for _,msg in r[ACTION][5:]:msg.header=header(msg.header.stamp.sec*10**9+msg.header.stamp.nanosec-5*10**9//15)
    for _,msg in r[CLOCK][20:]:msg.clock=header(msg.clock.sec*10**9+msg.clock.nanosec-20*10**9//60).stamp
    a,attrs=convert_fixture(decoupled)
    assert a['truncated'].sum()==2
    write(tmp_path/'reset.zarr',a,attrs)


@pytest.mark.parametrize('hz',[15,20,30])
def test_coarse_physics_excludes_only_ambiguous_intervals(decoupled,hz):
    r,m=decoupled;m['bfm']['control']['physics_hz']=hz;origin=r[ACTION][0][0]
    r[CLOCK]=[(origin+j*10**9//hz,NS(clock=header(j*10**9//hz).stamp)) for j in range(hz+1)]
    for i,(_,msg) in enumerate(r[ACTION]):msg.header=header(((i*hz+14)//15)*10**9//hz)
    a,attrs=convert_fixture(decoupled)
    assert len(a['action'])>0 and (a['duration']>0).all()
    if hz==20:assert any(d['reason']=='unverifiable_coarse_clock_interval' for d in attrs['rejected_intervals'])


def test_independent_hand_callback_is_not_fake_acceptance(decoupled):
    r,m=decoupled;record,msg=r[ACTION][4]
    msg.left_hand_target=[70.]*6
    r[HISTORY['raw_hand']][4][1].left_hand_position=[70]*6
    _,forwarded=r[HELD[1]][4];forwarded.left_hand_position=[70]*6
    r[HELD[1]][4]=(record+100,forwarded)
    a,attrs=convert_fixture(decoupled)
    assert len(a['action'])==9 and 'execution' not in attrs
    assert (a['boundary_controller_hand_delta'][4,:6]==-20).all()
    assert a['intervening_hand_max_delta'][4].max()==20

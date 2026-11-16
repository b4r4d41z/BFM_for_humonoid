"""Camera-only simulation freshness; tiny generated timelines, no training or ROS."""
import copy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import zarr

from test_sharpa import source
from test_decoupled_v4 import decoupled, header, convert_fixture
from bfm.data.camera_sync import CameraSync, POLICY
from bfm.data.mcap import ACTION, CLOCK, EPISODE, RECORDING, TOPICS, header_ns
from bfm.data.sharpa import SharpaDataset, validate, write


def slow(fixture):
    r,m=fixture;origin=r[ACTION][0][0]
    for topic,samples in r.items():r[topic]=[(origin+(t-origin)*8,msg) for t,msg in samples]
    for topic in TOPICS[2:]:
        for i in range(1,len(r[topic])):
            captured=header_ns(r[ACTION][i][1])-50_000_000
            _,msg=r[topic][i];msg.header=header(captured)
            r[topic][i]=(origin+captured*8,msg)
    return r,m


def test_low_rtf_keeps_wall_diagnostics_and_validates_zarr(decoupled,tmp_path):
    arrays,attrs=convert_fixture(slow(decoupled))
    assert len(arrays['action'])==9 and attrs['rejected_ticks']==0
    assert attrs['synchronization_policy']==POLICY and attrs['sync_clock']=='mcap_log_time_ns'
    assert attrs['source_age_ns_p50_p95_max'][2][2]>150_000_000
    assert (arrays['obs/source_header_ns'][:,:2]>arrays['action_header_ns'][:,None]).all()  # Other headers remain source-native.
    for p in ('obs','next_obs'):
        record=arrays['record_ns' if p=='obs' else 'next_record_ns']
        stamp=arrays['action_header_ns' if p=='obs' else 'next_action_header_ns']
        np.testing.assert_array_equal(arrays[p+'/sync_age_ns'],record[:,None]-arrays[p+'/source_record_ns'])
        np.testing.assert_array_equal(arrays[p+'/camera_age_sim_ns'],stamp[:,None]-arrays[p+'/source_header_ns'][:,2:])
        assert (arrays[p+'/sync_age_ns'][:,2:]>150_000_000).any()
        assert (arrays[p+'/camera_age_sim_ns']<=100_000_000).all()
    assert validate_store(tmp_path,arrays,attrs)['transitions']==9


def validate_store(tmp_path,arrays,attrs):
    return write(tmp_path/'episode.zarr',arrays,attrs)


@pytest.mark.parametrize('fault,reason',[
    ('stale','stale_camera_observation'),('future','future_camera_frame'),
    ('negative','invalid_camera_timestamp'),('missing_stamp','invalid_camera_timestamp'),
    ('invalid_ns','invalid_camera_timestamp'),('missing','missing_observation')])
@pytest.mark.parametrize('topic',TOPICS[2:])
def test_camera_failures_reject_ticks_without_bridging(decoupled,fault,reason,topic):
    r,m=decoupled
    if fault=='missing':r[topic].pop(0)
    else:
        _,msg=r[topic][4];stamp=header_ns(r[ACTION][4][1])
        if fault=='stale':msg.header=header(stamp-100_000_001)
        if fault=='future':msg.header=header(stamp+1)
        if fault=='negative':msg.header=header(-1)
        if fault=='missing_stamp':msg.header=NS()
        if fault=='invalid_ns':msg.header.stamp.nanosec=10**9
    arrays,attrs=convert_fixture((r,m));index=0 if fault=='missing' else 4
    assert reason in next(t['reasons'] for t in attrs['rejected_decisions'] if t['index']==index)
    record=r[ACTION][index][0]
    assert record not in arrays['record_ns'] and record not in arrays['next_record_ns']
    if index:assert arrays['truncated'].sum()==2


def test_causal_selection_never_uses_a_later_record(decoupled):
    r,m=decoupled;record,_=r[ACTION][4];_,image=r[TOPICS[2]][4]
    r[TOPICS[2]][4]=(record+1,image)
    arrays,attrs=convert_fixture((r,m));i=list(arrays['record_ns']).index(record)
    assert arrays['obs/source_record_ns'][i,2]==r[TOPICS[2]][3][0]
    assert arrays['obs/source_header_ns'][i,2]==header_ns(r[ACTION][3][1])
    assert arrays['obs/camera_age_sim_ns'][i,0]>66_000_000
    assert attrs['rejected_ticks']==0


@pytest.mark.parametrize('topic',TOPICS[:2])
def test_non_camera_wall_freshness_is_unchanged(decoupled,topic):
    r,m=slow(decoupled);r[topic].pop(4)
    _,attrs=convert_fixture((r,m))
    assert 'stale_observation' in next(t['reasons'] for t in attrs['rejected_decisions'] if t['index']==4)


def test_camera_skew_keeps_its_recorder_time_meaning(decoupled):
    from bfm.data.mcap import build
    r,m=slow(decoupled);record,image=r[TOPICS[2]][4];r[TOPICS[2]][4]=(record-30_000_000,image)
    _,attrs=build(r,m,max_age_s=[.05]*5,max_camera_skew_s=.02,collection_commit='a'*40)
    assert 'camera_skew' in next(t['reasons'] for t in attrs['rejected_decisions'] if t['index']==4)


def test_100ms_inclusive_and_unwitnessed_header(decoupled):
    r,m=decoupled;sync=CameraSync(r,m);record,msg=r[ACTION][4];stamp=header_ns(msg)
    age,reason=sync.ages(record,stamp,[record-1]*3,[stamp-100_000_000]*3)
    assert reason is None and (age==100_000_000).all()
    assert sync.ages(record,stamp,[record-1]*3,[stamp-100_000_001]*3)[1]=='stale_camera_observation'
    r[CLOCK]=r[CLOCK][8:];sync=CameraSync(r,m)
    assert sync.ages(record,stamp,[record-1]*3,[0]*3)[1]=='unverified_camera_clock_epoch'


def test_reset_epochs_and_delayed_overlapping_frames(decoupled):
    r,m=decoupled;r[CLOCK]=[(0,NS(clock=header(10_000_000_000).stamp)),(10,NS(clock=header(10_016_666_666).stamp)),
        (20,NS(clock=header(0).stamp)),(30,NS(clock=header(16_666_666).stamp)),(40,NS(clock=header(33_333_333).stamp))]
    sync=CameraSync(r,m)
    assert sync.ages(40,33_333_333,[30]*3,[16_666_666]*3)[1] is None
    assert sync.ages(40,33_333_333,[10]*3,[16_666_666]*3)[1]=='camera_epoch_mismatch'
    assert sync.ages(40,33_333_333,[30]*3,[10_016_666_666]*3)[1]=='future_camera_frame'
    r[CLOCK][0]=(0,NS(clock=header(0).stamp));r[CLOCK][1]=(10,NS(clock=header(16_666_666).stamp))
    assert CameraSync(r,m).ages(40,33_333_333,[30]*3,[16_666_666]*3)[1]=='ambiguous_camera_clock_epoch'


@pytest.mark.parametrize('boundary',['phase','recording_started','recording_bool'])
def test_camera_cannot_cross_lifecycle_boundary(decoupled,boundary):
    r,m=decoupled;record,msg=r[ACTION][4];before=r[TOPICS[2]][4][0]
    if boundary=='recording_bool':r[RECORDING]=[(record-500_000,NS(data=True))]
    else:
        event=dict(episode=m['bfm']['episode'])
        if boundary=='phase':event.update(phase='playing',boundary_id='new')
        else:event.update(event_type='recording_started',event_schema_version=1)
        r[EPISODE]=[(record-500_000,NS(data=json.dumps(event)))]
    sync=CameraSync(r,m)
    assert sync.ages(record,header_ns(msg),[before]*3,[header_ns(msg)]*3)[1]=='camera_epoch_mismatch'


@pytest.mark.parametrize('declaration',[None,{}, {'camera_image.header.stamp':'system wall time'}])
def test_uncertain_clock_domains_fail_closed(decoupled,declaration):
    r,m=decoupled;m['clock_domains']=declaration
    with pytest.raises(ValueError,match='clock relationship'):convert_fixture((r,m))


@pytest.mark.parametrize('fault',['sim_age','header','record','wall_age','policy','missing_array'])
def test_zarr_rechecks_freshness(decoupled,tmp_path,fault):
    arrays,attrs=convert_fixture(decoupled);validate_store(tmp_path,arrays,attrs)
    root=zarr.open_group(str(tmp_path/'episode.zarr'),mode='a')
    if fault=='sim_age':root['next_obs/camera_age_sim_ns'][1,0]=1
    if fault=='header':root['obs/source_header_ns'][1,2]=root['action_header_ns'][1]+1
    if fault=='record':root['obs/source_record_ns'][1,2]=root['record_ns'][1]+1
    if fault=='wall_age':root['obs/sync_age_ns'][1,2]=1
    if fault=='policy':root.attrs['synchronization_policy']={'name':'unknown'}
    if fault=='missing_array':del root['obs/camera_age_sim_ns']
    with pytest.raises(ValueError):validate(root)


def test_legacy_policy_and_cache_identity(decoupled,tmp_path):
    arrays,attrs=convert_fixture(decoupled);validate_store(tmp_path,arrays,attrs)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema_version='4.1.0',dataset_id='sync',episodes=[dict(path='episode.zarr',split='train')])))
    original=SharpaDataset(manifest,observation_profile='vision_proprio_v1').fingerprint
    root=zarr.open_group(str(tmp_path/'episode.zarr'),mode='a');del root.attrs['synchronization_policy']
    for p in ('obs','next_obs'):del root[p+'/camera_age_sim_ns']
    assert validate(root)['transitions']==9
    assert SharpaDataset(manifest,observation_profile='vision_proprio_v1').fingerprint!=original
    # Legacy stores continue enforcing recorder-time camera age, not the new rule.
    a,m=convert_fixture(slow(copy.deepcopy(decoupled)));del m['synchronization_policy']
    a={k:v for k,v in a.items() if not k.endswith('/camera_age_sim_ns')}
    with pytest.raises(ValueError,match='stale source'):write(tmp_path/'legacy-stale.zarr',a,m)

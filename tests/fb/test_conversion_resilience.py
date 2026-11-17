"""Evidence exclusions and independent batch failures through real CDR conversion."""
import copy
import errno
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from bfm.contracts.sharpa import flatten_action
from test_sharpa import source
from test_decoupled_v4 import decoupled, convert_fixture, header
from test_source_provenance import explicit, alter, write_recording
from bfm.data.mcap import ACTION, CLOCK, convert, header_ns
from bfm.data.source_provenance import TOPIC
from bfm.data.mcap_decoupled import NoVerifiableTransitions
from bfm.data.sharpa import validate, SharpaDataset
from bfm.data.visual_replay import VisualReplayBuffer
from bfm.data import recordings

POLICY=dict(max_age_s=[.05]*5)


def remove_clock(records, indices):
    stamps={header_ns(records[ACTION][i][1]) for i in indices}
    records[CLOCK]=[(t,m) for t,m in records[CLOCK] if m.clock.sec*10**9+m.clock.nanosec not in stamps]


@pytest.mark.parametrize('kind',['clock','trace'])
@pytest.mark.parametrize('indices',[[0,1,2],[3,4]])
def test_evidence_gaps_preserve_only_adjacent_transitions(decoupled,tmp_path,kind,indices):
    r,m=explicit(decoupled)
    if kind=='clock':
        if indices==[0,1,2]:r[CLOCK]=r[CLOCK][12:]
        else:remove_clock(r,indices)
    else:r[TOPIC]=[s for i,s in enumerate(r[TOPIC]) if i not in indices]
    raw=tmp_path/'raw';write_recording(raw,r,m);out=tmp_path/'episode.zarr'
    convert(raw,out,collection_commit='a'*40,**POLICY)
    root=zarr.open_group(str(out),mode='r');assert validate(root)['action_dim']==26
    kept=[i for i in range(9) if i not in indices and i+1 not in indices]
    np.testing.assert_array_equal(root['action_id'][:],np.asarray(kept)+1)
    np.testing.assert_array_equal(root['next_action_id'][:],np.asarray(kept)+2)
    np.testing.assert_array_equal(root['action'][:],np.stack([flatten_action(r[ACTION][i][1]) for i in kept]))
    expected_ends=np.r_[[b!=a+1 for a,b in zip(kept,kept[1:])],True]
    np.testing.assert_array_equal(root['truncated'][:],expected_ends)
    np.testing.assert_array_equal(root['segment_id'][:],np.cumsum(np.r_[0,expected_ends[:-1]]))
    assert not root['terminated'][:].any()
    rejected=root.attrs['rejected_decisions'];assert [e['index'] for e in rejected]==indices
    reason='action_stamp_not_witnessed_by_clock' if kind=='clock' else 'missing_explicit_provenance_trace'
    assert all(reason in e['reasons'] and e['action_id']==e['index']+1 and e['producer_session_id']=='fixture' for e in rejected)
    assert root.attrs['total_action_ticks']==10 and root.attrs['valid_decision_ticks']==10-len(indices)
    np.testing.assert_allclose(root['duration'][:],1/15,atol=1e-9)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema_version='4.1.0',dataset_id='gaps',episodes=[dict(path=out.name,split='train')])))
    batch=VisualReplayBuffer(SharpaDataset(manifest,observation_profile='vision_proprio_v1')).sample_transitions(2)
    assert batch['action'].shape==(2,26) and batch['observation']['proprio'].shape==(2,116)


def test_physics_gap_without_missing_action_stamp_still_splits(decoupled):
    r,m=explicit(decoupled);r[CLOCK].pop(13)
    arrays,attrs=convert_fixture((r,m))
    assert attrs['rejected_ticks']==0 and len(arrays['action'])==8
    assert arrays['truncated'].sum()==2
    assert attrs['rejected_intervals']==[dict(index=3,reason='missing_physics_clock_evidence')]


@pytest.mark.parametrize('fault',['digest','conflict','identity','version','nan','groups','duplicate_id','backward_stamp','invalid_stamp','duplicate_record'])
def test_critical_errors_are_fatal_even_at_unwitnessed_ticks(decoupled,fault):
    r,m=explicit(decoupled);remove_clock(r,[3]);action=r[ACTION][3][1]
    if fault=='digest':action.source_provenance_sha256='0'*64
    if fault=='conflict':r[TOPIC].append((r[TOPIC][3][0]+1,type(r[TOPIC][3][1])(data=r[TOPIC][3][1].data+' ')))
    if fault=='identity':alter((r,m),3,lambda t:t['arm'].update(input_sequence=1))
    if fault=='version':action.source_provenance_version=2
    if fault=='nan':action.left_arm_target[0]=float('nan')
    if fault=='groups':action.left_arm_target.pop()
    if fault=='duplicate_id':action.action_id=r[ACTION][2][1].action_id
    if fault=='backward_stamp':action.header=copy.deepcopy(r[ACTION][2][1].header)
    if fault=='invalid_stamp':action.header=header(-1)
    if fault=='duplicate_record':r[ACTION][3]=(r[ACTION][2][0],action)
    with pytest.raises(ValueError):convert_fixture((r,m))


def test_missing_trace_cannot_hide_contradictory_identity(decoupled):
    fixture=explicit(decoupled)
    alter(fixture,3,lambda t:t.update(producer_session_id='different'))
    with pytest.raises(ValueError,match='different provenance identity'):convert_fixture(fixture)


def test_no_transitions_reports_real_exclusions(decoupled):
    r,m=explicit(decoupled);r[TOPIC]=[]
    with pytest.raises(NoVerifiableTransitions) as error:convert_fixture((r,m))
    assert error.value.conversion_details['total_actions']==10
    assert error.value.conversion_details['excluded_actions']==10
    assert error.value.conversion_details['valid_transitions']==0


def episodes(tmp_path,fixture,names=('a','b','c')):
    paths=[]
    for name in names:
        r,m=copy.deepcopy(fixture);m['bfm']['episode']['episode_id']=name
        path=tmp_path/name;write_recording(path,r,m);paths.append(path)
    return paths


def test_batch_recovery_report_reuse_and_determinism(decoupled,tmp_path,capsys):
    fixture=explicit(decoupled);paths=episodes(tmp_path/'raw',fixture)
    # A loses two traces but remains useful; B is corrupt; C must still convert.
    r,m=copy.deepcopy(fixture);m['bfm']['episode']['episode_id']='a';r[TOPIC]=[s for i,s in enumerate(r[TOPIC]) if i not in (3,4)]
    # Rewrite only our generated fixture bag, never repository recordings.
    import shutil
    shutil.rmtree(paths[0]);write_recording(paths[0],r,m)
    (paths[1]/'bag/bag_0.mcap').write_bytes(b'corrupt MCAP fixture')
    dest=tmp_path/'zarr';manifest=recordings.prepare(paths[0].parent,dest,POLICY,expected_version=4,expected_architecture='decoupled_teleoperation_v1')
    report=json.loads((dest/'conversion_report.json').read_text());entries=report['recordings']
    assert [r['status'] for r in entries]==['CONVERTED_WITH_SKIPS','FAILED','CONVERTED']
    assert entries[0]['total_actions']==10 and entries[0]['excluded_actions']==2 and entries[0]['segments']==2
    assert entries[0]['valid_transitions']==6 and entries[2]['valid_transitions']==9
    assert entries[1]['total_actions'] is None and entries[1]['excluded_actions'] is None
    assert entries[1]['failure']['exception_type']=='InvalidMagic'
    assert report['summary']['valid_transitions']==15 and report['summary']['failed']==1
    spec=json.loads(manifest.read_text());assert len(spec['episodes'])==2
    assert sorted(SharpaDataset(manifest,observation_profile='vision_proprio_v1').ids)==['a','c']
    text=capsys.readouterr().out
    assert '[01/03] CONVERTED_WITH_SKIPS' in text and '[02/03] FAILED' in text and '[03/03] CONVERTED' in text
    assert 'Valid transitions: 15' in text and 'Failed recordings:' in text
    before={p.name:(p/'.zattrs').read_bytes() for p in dest.glob('*.zarr')}
    recordings.prepare(paths[0].parent,dest,POLICY,report=lambda *args:None,expected_version=4,expected_architecture='decoupled_teleoperation_v1')
    assert [r['status'] for r in json.loads((dest/'conversion_report.json').read_text())['recordings']]==['REUSED','FAILED','REUSED']
    assert before=={p.name:(p/'.zattrs').read_bytes() for p in dest.glob('*.zarr')}
    fresh=tmp_path/'fresh';recordings.prepare(paths[0],fresh,POLICY,report=lambda *args:None)
    same=next(fresh.glob('*.zarr'));assert (same/'.zattrs').read_bytes()==before[same.name]
    assert not list(dest.glob('.conversion-*')) and not list(dest.glob('.manifest-*')) and not list(dest.glob('.report-*'))


def test_all_failed_batch_still_publishes_report(decoupled,tmp_path):
    r,m=explicit(decoupled);r[TOPIC]=[];paths=episodes(tmp_path/'raw',(r,m),names=('empty',))
    with pytest.raises(recordings.NoSuccessfulRecordings):recordings.prepare(paths[0],tmp_path/'zarr',POLICY)
    report=json.loads((tmp_path/'zarr/conversion_report.json').read_text())
    assert report['recordings'][0]['excluded_actions']==10 and report['recordings'][0]['segments']==0
    assert report['summary']['failed']==1 and report['summary']['valid_transitions']==0
    assert not (tmp_path/'zarr/manifest.json').exists() and not list((tmp_path/'zarr').glob('*.zarr'))


def test_cli_partial_failure_exit_and_canonical_report(decoupled,tmp_path,monkeypatch):
    import runpy
    import sys
    root=Path(__file__).resolve().parents[2]
    main=runpy.run_path(str(root/'tools/convert_recordings.py'))['main']
    paths=episodes(tmp_path/'raw',explicit(decoupled),names=('a','b'))
    (paths[0]/'bag/bag_0.mcap').write_bytes(b'invalid MCAP')
    config=tmp_path/'project/configs/data_conversion/sharpa_v3.yaml';config.parent.mkdir(parents=True)
    config.write_bytes((root/'configs/data_conversion/sharpa_v3.yaml').read_bytes())
    main.__globals__['ROOT']=tmp_path/'project'
    monkeypatch.setattr(sys,'argv',['convert_recordings.py',str(paths[0].parent),'--contract','decoupled-v4'])
    assert main()==1
    report=json.loads((tmp_path/'project/data/zarr_decoupled_v4/conversion_report.json').read_text())
    assert [e['status'] for e in report['recordings']]==['FAILED','CONVERTED']
    assert report['summary']['valid_transitions']==9


@pytest.mark.parametrize('fault',['json','identity','code','missing_mcap'])
def test_bad_input_metadata_or_missing_bag_does_not_abort(decoupled,tmp_path,fault):
    paths=episodes(tmp_path/'raw',explicit(decoupled),names=('a','b'));path=paths[0]/'episode.json'
    metadata=json.loads(path.read_text())
    if fault=='json':path.write_text('invalid JSON')
    elif fault=='missing_mcap':(paths[0]/'bag/bag_0.mcap').unlink()
    else:
        if fault=='identity':metadata['bfm']['episode']['episode_id']={}
        if fault=='code':metadata['recording_code']=None
        path.write_text(json.dumps(metadata))
    dest=tmp_path/'zarr';recordings.prepare(paths[0].parent,dest,POLICY,report=lambda *args:None)
    report=json.loads((dest/'conversion_report.json').read_text())
    assert [e['status'] for e in report['recordings']]==['FAILED','CONVERTED']
    assert report['recordings'][0]['total_actions'] is None
    assert report['summary']['valid_transitions']==9


@pytest.mark.parametrize('fault',['permission','space','unexpected','cancel','manifest'])
def test_infrastructure_failures_abort_and_keep_manifest(source,tmp_path,monkeypatch,fault):
    from test_recordings_workflow import raw, generated_converter
    paths=[raw(tmp_path/'raw',source,n) for n in ('a','b')];dest=tmp_path/'zarr';calls=[]
    normal=generated_converter(source,calls)
    if fault=='manifest':
        dest.mkdir();(dest/'manifest.json').write_text('{invalid')
        monkeypatch.setattr(recordings,'convert',normal)
        expected=ValueError
    else:
        def failed(*args,**kwargs):
            calls.append(args[0])
            if fault=='permission':raise PermissionError(errno.EACCES,'output denied')
            if fault=='space':raise OSError(errno.ENOSPC,'no space')
            if fault=='cancel':raise KeyboardInterrupt()
            raise RuntimeError('unclassified error')
        monkeypatch.setattr(recordings,'convert',failed)
        expected={'permission':PermissionError,'space':OSError,'unexpected':RuntimeError,'cancel':KeyboardInterrupt}[fault]
    with pytest.raises(expected):recordings.prepare(paths[0].parent,dest,POLICY)
    assert len(calls)==(0 if fault=='manifest' else 1)
    report=json.loads((dest/'conversion_report.json').read_text());assert report['aborted']['exception_type']==expected.__name__ or fault=='manifest'
    if fault=='manifest':assert (dest/'manifest.json').read_text()=='{invalid'
    else:assert not (dest/'manifest.json').exists()


def test_global_config_and_contract_abort(source,tmp_path,monkeypatch):
    from test_recordings_workflow import raw, generated_converter
    path=raw(tmp_path/'raw',source,'a');calls=[];monkeypatch.setattr(recordings,'convert',generated_converter(source,calls))
    with pytest.raises(ValueError,match='global'):recordings.prepare(path,tmp_path/'bad',dict(max_age_s=[0]*5))
    assert not calls
    dest=tmp_path/'zarr';manifest=recordings.prepare(path,dest,POLICY);before=manifest.read_bytes()
    with pytest.raises(recordings.BatchCompatibilityError):recordings.prepare(path,dest,POLICY,expected_version=4,expected_architecture='decoupled_teleoperation_v1')
    assert manifest.read_bytes()==before and len(calls)==1

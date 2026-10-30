"""Small generated episodes exercise orchestration; real MCAP is opt-in integration."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml
import zarr

from test_sharpa import source, make
from bfm.data.mcap import ACTION, HELD, RAW_ARM
from bfm.data.sharpa import write, SharpaDataset, validate
from bfm.data.visual_replay import VisualReplayBuffer
from bfm.data import recordings
from bfm.fb.trainer import train
from bfm.fb.checkpoint import load_payload

POLICY = {'max_age_s':[.15]*5, 'max_camera_skew_s':None}


def raw(tmp_path, source, name):
    episode = tmp_path/name
    (episode/'bag').mkdir(parents=True)
    metadata = copy.deepcopy(source[1])
    metadata['bfm']['episode']['episode_id'] = name
    from bfm.contracts.sharpa import SOURCE_COMMIT
    metadata['recording_code'] = {'status':'available', 'commit_sha':SOURCE_COMMIT}
    (episode/'episode.json').write_text(json.dumps(metadata))
    (episode/'bag/metadata.yaml').write_text(yaml.safe_dump({'rosbag2_bagfile_information':{'relative_file_paths':['small.mcap']}}))
    (episode/'bag/small.mcap').write_bytes(b'small fixture; converter mocked in orchestration-only tests')
    return episode


def generated_converter(source, calls):
    def convert(episode, output, **kwargs):
        calls.append(episode)
        arrays, attrs = make(source)
        metadata = json.loads((episode/'episode.json').read_text())
        attrs['source_metadata'] = metadata
        attrs['source_episode_id'] = metadata['bfm']['episode']['episode_id']
        return write(output, arrays, attrs)
    return convert


def test_batch_reuse_progress_manifest_and_real_training(source, tmp_path, monkeypatch):
    inputs = tmp_path/'raw'
    for name in ('b','a'): raw(inputs, source, name)
    assert [p.name for p in recordings.discover(inputs)] == ['a','b']
    calls, progress = [], []
    monkeypatch.setattr(recordings, 'convert', generated_converter(source,calls))
    output = tmp_path/'zarr'
    manifest = recordings.prepare(inputs, output, POLICY, report=lambda *v:progress.append(v))
    assert len(calls)==2 and [(p[0],p[1],p[3]) for p in progress]==[(1,2,'converting'),(1,2,'validated'),(2,2,'converting'),(2,2,'validated')]
    names = sorted(p.name for p in output.glob('*.zarr'))
    assert len(names)==2
    recordings.prepare(inputs, output, POLICY, report=lambda *v:progress.append(v))
    assert len(calls)==2 and progress[-1][3]=='skipped'
    assert names==sorted(p.name for p in output.glob('*.zarr'))
    data = SharpaDataset(manifest, observation_profile='vision_proprio_v1')
    batch = VisualReplayBuffer(data).sample_transitions(2)
    assert batch['action'].shape==(2,26) and batch['observation']['proprio'].shape==(2,116)
    assert batch['observation']['head'].shape==(2,2,2,3)
    # Actual trainer/model/loss/backward/optimizer. Mock visual features are only
    # a unit fixture; the separate real integration uses pretrained local DINO.
    checkpoint = train({'dataset_schema':'sharpa_v3','observation_profile':'vision_proprio_v1',
        'dataset_manifest':str(manifest),'output_dir':str(tmp_path/'run'),'batch_size':2,
        'training_steps':2,'checkpoint_interval':2,'reference_count':2,
        'vision':{'backbone':'mock','image_size':14,'embedding_dim':16,'projection_dim':8},
        'model':{'hidden_dim':16,'latent_dim':4}})
    assert load_payload(checkpoint)['step']==2
    # Explicit validation assignments survive manifest regeneration.
    spec=json.loads(manifest.read_text()); spec['episodes'][0]['split']='validation'
    manifest.write_text(json.dumps(spec))
    recordings.manifest_for(output)
    assert json.loads(manifest.read_text())['episodes'][0]['split']=='validation'


@pytest.mark.parametrize('failure',['conversion','provenance','changed_source','invalid_existing'])
def test_fail_closed_and_atomic_outputs(source,tmp_path,monkeypatch,failure):
    episode=raw(tmp_path/'raw',source,'bad')
    destination=tmp_path/'zarr'
    calls=[]
    monkeypatch.setattr(recordings,'convert',generated_converter(source,calls))
    if failure in ('changed_source','invalid_existing'):
        recordings.prepare(episode,destination,POLICY)
        if failure=='changed_source': (episode/'bag/small.mcap').write_bytes(b'changed')
        else:
            root=zarr.open_group(str(next(destination.glob('*.zarr'))),mode='a')
            root['action'][0]=np.nan
    elif failure=='provenance':
        meta=json.loads((episode/'episode.json').read_text());meta['recording_code']['commit_sha']='bad'
        (episode/'episode.json').write_text(json.dumps(meta))
    else:
        def broken(episode,path,**kwargs):
            Path(path).mkdir(); (Path(path)/'.zgroup').write_text('{}')
            raise ValueError('interrupted')
        monkeypatch.setattr(recordings,'convert',broken)
    with pytest.raises(ValueError,match='bad'):
        recordings.prepare(episode,destination,POLICY)
    if failure in ('conversion','provenance'):
        assert not list(destination.glob('*.zarr')) and not (destination/'manifest.json').exists()
        assert not list(destination.glob('.conversion-*'))


def test_high_rate_held_target_has_no_policy_proximity_window(source):
    records,meta=copy.deepcopy(source)
    meta['bfm']['arm_target_processing'].update(controller_cadence='validated source callback rate',
        canonical_action_cadence='0.1 simulation seconds; latest bounded target snapshot')
    # Single controller request held for many policy ticks; raw receipt can be
    # reordered relative to its controller publication. Header proves association.
    for topic in HELD: records[topic]=[(records[ACTION][0][0]-400_000_000,records[topic][0][1])]
    records[RAW_ARM]=[(records[ACTION][0][0]-300_000_000,records[RAW_ARM][0][1])]
    assert len(make((records,meta))[0]['action'])==9
    # A newer unrelated target must not be skipped to borrow the older match.
    wrong=copy.deepcopy(records[HELD[0]][0][1]); wrong.position[0]+=.2
    records[HELD[0]].append((records[ACTION][4][0]-1,wrong))
    with pytest.raises(ValueError,match='Held command'): make((records,meta))


@pytest.mark.parametrize('field',['position','velocity'])
def test_nonfinite_required_measurement_rejects(source,field):
    records,meta=copy.deepcopy(source)
    from bfm.data.mcap import TOPICS
    getattr(records[TOPICS[1]][4][1],field)[0]=np.nan
    with pytest.raises(ValueError,match='invalid measured'):make((records,meta))


def test_progress_text(capsys):
    recordings.progress(7,24,'episode','validated')
    assert '7/24 episode' in capsys.readouterr().out


def test_high_rate_reordered_controller_requires_latest_raw_identity(source):
    records,meta=copy.deepcopy(source)
    meta['bfm']['arm_target_processing'].update(controller_cadence='validated source callback rate',
        canonical_action_cadence='0.1 simulation seconds; latest bounded target snapshot')
    records[HELD[1]]=[(t-5_000,m) for t,m in records[HELD[1]]]
    # Fixture publishes each arm controller just after action, raw source before it.
    assert len(make((records,meta))[0]['action'])==9
    records[RAW_ARM][0][1].header.stamp.nanosec=99999
    with pytest.raises(ValueError,match='Held command'):make((records,meta))

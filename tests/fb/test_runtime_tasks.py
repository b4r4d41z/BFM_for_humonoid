"""Focused offline tests; no simulator or ROS middleware required."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
import zarr
from test_sharpa import source
from test_vision import visual_manifest, mock_config
from bfm.data.training import validate_training
from bfm.data.sharpa import SharpaDataset
from bfm.data.visual_replay import VisualReplayBuffer
from bfm.fb.trainer import train
from bfm.fb.runtime import VisualRuntime, validate_ros_config
from bfm.fb.latents import episode_embeddings, balanced_task, save_task, candidates, rank_candidates, pca, visualize


@pytest.fixture
def trained(visual_manifest,tmp_path):
    checkpoint=train({'dataset_schema':'sharpa_v3','observation_profile':'vision_proprio_v1',
        'dataset_manifest':str(visual_manifest),'output_dir':str(tmp_path/'run'),'batch_size':2,
        'training_steps':1,'checkpoint_interval':1,'reference_count':2,
        'vision':mock_config(),'model':{'hidden_dim':16,'latent_dim':4}})
    return VisualRuntime(checkpoint),visual_manifest,checkpoint


def test_zarr_only_training_and_completeness(visual_manifest,monkeypatch):
    import bfm.data.mcap
    monkeypatch.setattr(bfm.data.mcap,'read_records',lambda *a:pytest.fail('Training opened MCAP'))
    assert validate_training(visual_manifest)==visual_manifest.resolve()
    root=zarr.open_group(str(visual_manifest.parent/'episode.zarr'),mode='a')
    del root['next_obs/images/head']
    with pytest.raises(ValueError):validate_training(visual_manifest)
    with pytest.raises(ValueError,match='Missing converted-data'):validate_training(visual_manifest.parent/'missing')


def test_split_entrypoints_do_not_cross_layers(monkeypatch,tmp_path):
    import importlib.util,sys
    def module(name):
        spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[2]/'tools'/f'{name}.py')
        value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
    import bfm.data.recordings,bfm.fb.trainer
    monkeypatch.setattr(bfm.fb.trainer,'train',lambda *a,**k:pytest.fail('Conversion started training'))
    calls=[]
    monkeypatch.setattr(bfm.data.recordings,'prepare',lambda *a,**k:calls.append(a) or tmp_path/'manifest.json')
    monkeypatch.setattr(sys,'argv',['convert_recordings.py',str(tmp_path)])
    module('convert_recordings').main();assert len(calls)==1
    monkeypatch.setattr(bfm.data.recordings,'prepare',lambda *a,**k:pytest.fail('Training called converter'))
    monkeypatch.setattr(sys,'argv',['train_from_zarr.py',str(tmp_path)])
    with pytest.raises(ValueError,match='Missing converted-data'):module('train_from_zarr').main()


def test_runtime_restore_and_validation(trained,tmp_path):
    runtime,manifest,checkpoint=trained
    obs=VisualReplayBuffer(SharpaDataset(manifest,observation_profile='vision_proprio_v1')).observations([0])
    z=runtime.model.sample_z(1)
    assert runtime.act(obs,z).shape==(26,)
    with pytest.raises(ValueError):runtime.act(obs)
    with pytest.raises(ValueError):runtime.act({**obs,'task56':torch.zeros(56)},z)
    with pytest.raises(ValueError):runtime.act({**obs,'head':obs['head'].float()},z)
    with pytest.raises(ValueError):runtime.act({**obs,'proprio':torch.zeros(1,115)},z)
    for bad in (torch.zeros(3),torch.zeros(4),torch.tensor([float('nan')]*4)):
        with pytest.raises(ValueError):runtime.task(bad)
    path=tmp_path/'task.pt';torch.save({'z':z,'checkpoint_sha256':runtime.checkpoint_sha256},path)
    runtime.load_task(path)
    assert torch.equal(runtime.act(obs),VisualRuntime(checkpoint).act(obs,z))
    torch.save({'z':z,'checkpoint_sha256':'wrong'},path)
    with pytest.raises(ValueError,match='different checkpoint'):runtime.load_task(path)
    cfg=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/runtime/isaac_ros2.yaml').read_text())
    validate_ros_config(cfg);cfg['cameras'].reverse()
    with pytest.raises(ValueError):validate_ros_config(cfg)


def test_full_reference_embeddings_and_balanced_task(trained,tmp_path):
    runtime,manifest,_=trained
    root=zarr.open_group(str(manifest.parent/'episode.zarr'),mode='a');root.attrs['success']=True
    episodes=episode_embeddings(runtime,manifest,batch_size=3)
    assert len(episodes)==1 and episodes[0]['embeddings'].shape==(10,4)
    first=save_task(runtime,episodes,tmp_path/'task.pt')
    second=save_task(runtime,episodes,tmp_path/'task2.pt')
    assert torch.equal(first['z'],second['z'])
    assert first['z'].norm()==pytest.approx(2.)
    a={'success':True,'embeddings':torch.tensor([[1.,0,0,0]]).repeat(100,1)}
    b={'success':True,'embeddings':torch.tensor([[0.,1,0,0]])}
    expected=runtime.model.project_z(torch.tensor([[.5,.5,0,0]]))
    assert torch.equal(balanced_task(runtime.model,[a,b]),expected)
    with pytest.raises(ValueError):balanced_task(runtime.model,[{'success':False}])
    bad=json.loads(manifest.read_text());bad['schema_version']='2.0.0'
    invalid=tmp_path/'invalid.json';invalid.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='Unsupported reference manifest'):episode_embeddings(runtime,invalid)
    # Confirm the first state is encoded by the checkpoint's own backward path.
    from bfm.contracts.vision import proprioception
    obs={'proprio':proprioception(root.attrs['joint_names'],root['obs/joint_position'][0],root['obs/joint_velocity'][0])[None],
         **{c:root[f'obs/images/{c}'][0][None] for c in ('head','left_wrist','right_wrist')}}
    with torch.no_grad():assert torch.allclose(runtime.model.backward_map(obs)[0],episodes[0]['embeddings'][0],atol=1e-6)


def test_search_and_joint_visualization(trained,tmp_path,monkeypatch):
    runtime,_,_=trained
    original=runtime.model.sample_z;calls=[]
    def sample(n):calls.append(n);return original(n)
    monkeypatch.setattr(runtime.model,'sample_z',sample)
    proposal=candidates(runtime,5,seed=12)
    assert calls==[5] and proposal['vectors'].shape==(5,4)
    assert torch.allclose(proposal['vectors'].norm(dim=-1),torch.full((5,),2.))
    result=rank_candidates(runtime,proposal,lambda identity,z:int(identity.split('-')[1]))
    assert torch.equal(result['z'][0],proposal['vectors'][-1])
    with pytest.raises(ValueError):rank_candidates(runtime,proposal,lambda *a:float('nan'))
    points=torch.cat([proposal['vectors'],result['z']]).numpy()
    labels=[{'source':'random'}]*5+[{'source':'reference_task'}]
    xy=visualize(points,labels,tmp_path/'latent')
    assert xy.shape==(6,2) and np.isfinite(xy).all()
    assert np.array_equal(xy,pca(points))
    assert (tmp_path/'latent.csv').is_file() and (tmp_path/'latent.svg').is_file()

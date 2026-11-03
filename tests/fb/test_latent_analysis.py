import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from test_runtime_tasks import trained, visual_manifest, source
from bfm.fb.latent_analysis import (analyze, validate_artifact, diversity, representatives,
                                    visualize_analysis, compare_actions, action_metrics)


def test_geometry_and_metrics():
    z=np.array([[2.,0],[0,2.],[-2.,0]])
    pair,sim=diversity(z,z[:1])
    np.testing.assert_allclose(pair,pair.T);np.testing.assert_allclose(np.diag(pair),1)
    np.testing.assert_allclose(sim,[1,0,-1]);np.testing.assert_allclose(1-sim,[0,1,2])
    assert representatives(['b','a','c'],[0,0,1])==['a','b','c']
    metrics=action_metrics(np.ones(26))
    assert metrics['l2']==pytest.approx(26**.5) and metrics['right_hand_mae']==1
    for bad in (np.zeros((3,3)), np.full((3,2),np.nan)):
        with pytest.raises(ValueError):diversity(bad,z[:1])


def snapshot(path):
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}


def test_analysis_visualization_actions_and_provenance(trained,tmp_path,monkeypatch):
    import shutil
    runtime,manifest,checkpoint=trained
    root=zarr.open_group(str(manifest.parent/'episode.zarr'),mode='a');root.attrs['success']=True
    shutil.copytree(manifest.parent/'episode.zarr',manifest.parent/'second.zarr')
    second=zarr.open_group(str(manifest.parent/'second.zarr'),mode='a')
    metadata=second.attrs['source_metadata'];metadata['bfm']['episode']['episode_id']='second'
    second.attrs.update(source_episode_id='second',source_metadata=metadata)
    spec=json.loads(manifest.read_text());spec['episodes'].append({'path':'second.zarr','split':'train'})
    manifest.write_text(json.dumps(spec))
    task=tmp_path/'task.pt';torch.save({'z':runtime.model.sample_z(1),'checkpoint_sha256':runtime.checkpoint_sha256},task)
    before=[snapshot(manifest.parent/'episode.zarr'),snapshot(manifest.parent/'second.zarr'),task.read_bytes(),checkpoint.read_bytes(),manifest.read_bytes()]
    summary=analyze(runtime,manifest,task,tmp_path/'analysis',batch_size=2)
    artifact=tmp_path/'analysis/episode_z.pt';saved=torch.load(artifact,weights_only=True)
    z,g=validate_artifact(saved);assert z.shape==(2,4) and np.isfinite(z).all()
    assert summary['successful_episodes']==2
    assert summary['pairwise_cosine_similarity']['mean']==pytest.approx(1,abs=1e-6)
    # Stored values really are projected episode means, not raw B vectors.
    from bfm.fb.latents import balanced_task, episode_embeddings
    expected=balanced_task(runtime.model,[episode_embeddings(runtime,manifest.parent/'episode.zarr',2)[0]])
    torch.testing.assert_close(saved['episode_z'][0],expected[0])
    pca=visualize_analysis(artifact,tmp_path/'plot')
    assert 0<=pca['pc1_pc2_cumulative']<=1.000001
    assert len(list(csv.DictReader((tmp_path/'plot/pca_coordinates.csv').open())))==3
    assert (tmp_path/'plot/episode_z_pca.svg').is_file()
    result=compare_actions(runtime,artifact,tmp_path/'actions',timesteps=[0,2])
    rows=list(csv.DictReader((tmp_path/'actions/action_comparison.csv').open()))
    assert len(rows)==6 and result['timesteps']==[0,2]
    assert all(float(r['max_abs'])==0 for r in rows if r['latent_id']=='global')
    assert before==[snapshot(manifest.parent/'episode.zarr'),snapshot(manifest.parent/'second.zarr'),task.read_bytes(),checkpoint.read_bytes(),manifest.read_bytes()]
    saved['latent_dim']=5
    with pytest.raises(ValueError):validate_artifact(saved)
    with pytest.raises(ValueError,match='checkpoint'):validate_artifact(torch.load(artifact,weights_only=True),'wrong')
    torch.save({'z':runtime.model.sample_z(1),'checkpoint_sha256':'wrong'},tmp_path/'wrong.pt')
    with pytest.raises(ValueError,match='different checkpoint'):analyze(runtime,manifest,tmp_path/'wrong.pt',tmp_path/'bad')
    base=Path(__file__).resolve().parents[2]
    commands=[['analyze_episode_z.py','--checkpoint',str(checkpoint),'--references',str(manifest),
               '--task-z',str(task),'--output-dir',str(tmp_path/'cli-analysis'),'--batch-size','2'],
              ['visualize_latent_space.py','--analysis',str(tmp_path/'cli-analysis/episode_z.pt'),
               '--output-dir',str(tmp_path/'cli-plot')],
              ['compare_z_actions.py','--checkpoint',str(checkpoint),'--analysis',str(tmp_path/'cli-analysis/episode_z.pt'),
               '--output-dir',str(tmp_path/'cli-actions'),'--timesteps','0','2']]
    for command in commands:
        subprocess.run([sys.executable,str(base/'tools'/command[0]),*command[1:]],check=True,capture_output=True)
    root['obs/images/head'][0]=255
    with pytest.raises(ValueError,match='source changed'):compare_actions(runtime,artifact,tmp_path/'changed')


def test_pca_explained_variance_and_invalid_artifacts(tmp_path):
    saved=dict(format='bfm.episode_z.v1',checkpoint_sha256='test',latent_dim=2,
               episodes=[{'episode_id':'a'},{'episode_id':'b'}],episode_z=torch.tensor([[2**.5,0],[-2**.5,0]]),global_z=torch.tensor([[1.,1.]]))
    # Nonzero variance on one axis, fixed sqrt(2) spherical radius.
    p=tmp_path/'z.pt';torch.save(saved,p)
    summary=visualize_analysis(p,tmp_path/'plot')
    assert summary['pc1_explained_variance_ratio']==pytest.approx(1)
    assert summary['pc2_explained_variance_ratio']==pytest.approx(0)
    saved['episode_z'][0,0]=float('nan')
    with pytest.raises(ValueError):validate_artifact(saved)

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from test_runtime_tasks import trained, visual_manifest, source
from test_latent_analysis import snapshot
from bfm.contracts.sharpa import SLICES
from bfm.fb.latents import EpisodeBalancedZ, balanced_task, episode_embeddings, retained_endpoints, save_task
from bfm.fb.model import FBModel, ModelConfig
from bfm.fb.reward_weighted_z import (RewardWeightedZ, finite_rewards, measured_rewards,
                                      annotation_rewards, reference_rows, build, compare_tasks)


def test_weighting_equation_signed_rewards_and_episode_balance():
    model = FBModel(ModelConfig(state_dim=56,action_dim=26,latent_dim=4,hidden_dim=16))
    episodes = [dict(episode_id='a',success=True,embeddings=torch.tensor([[1.,0,0,0],[0,1,0,0]])),
                dict(episode_id='b',success=True,embeddings=torch.tensor([[0.,0,1,0]]))]
    rewards = {'a':[-1.,2.], 'b':[0.]}
    z, diagnostics = RewardWeightedZ(2).infer(model,episodes,rewards)
    r = torch.tensor([-1.,2.]);weighted = (r*torch.softmax(2*r,0))[:,None]*episodes[0]['embeddings']
    expected = model.project_z((weighted.sum(0)/2)[None])
    torch.testing.assert_close(z,expected)
    assert z.shape==(1,4) and z.norm()==pytest.approx(2.) and diagnostics['states']==3
    longer = copy.deepcopy(episodes);longer[0]['embeddings']=longer[0]['embeddings'].repeat(50,1)
    again,_ = RewardWeightedZ(2).infer(model,longer,{'a':rewards['a']*50,'b':[0.]})
    torch.testing.assert_close(again,z)
    assert torch.equal(RewardWeightedZ(2).infer(model,episodes,rewards)[0],z)
    assert EpisodeBalancedZ is balanced_task
    old = model.project_z(torch.stack([e['embeddings'].mean(0) for e in episodes]).mean(0,keepdim=True))
    assert torch.equal(EpisodeBalancedZ(model,episodes),old)
    # Signed weighting can point opposite to B; no clipping or reward shifting.
    negative,_ = RewardWeightedZ(0).infer(model,episodes,{'a':[-2.,-1.],'b':[0.]})
    assert negative[0,0]<0 and negative[0,1]<0
    extremes,_ = RewardWeightedZ(0).infer(model,episodes,{'a':[-1e308,1e308],'b':[0.]})
    assert torch.isfinite(extremes).all()


@pytest.mark.parametrize('values',[[1.], [0.,float('nan')], [1.,float('inf')], [[1.],[2.]]])
def test_invalid_rewards(values):
    with pytest.raises(ValueError):finite_rewards(values,2)


def test_uninformative_invalid_and_cancelling_directions():
    model = FBModel(ModelConfig(state_dim=56,action_dim=26,latent_dim=4,hidden_dim=16))
    e = [dict(episode_id='a',success=True,embeddings=torch.eye(4)[:2])]
    for r in ([0.,0.],[1.,1.],[-2.,-2.]):
        with pytest.raises(ValueError,match='Constant rewards'):RewardWeightedZ().infer(model,e,{'a':r})
    for beta in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):RewardWeightedZ(beta)
    for bad in (torch.zeros(2,3),torch.full((2,4),float('nan'))):
        with pytest.raises(ValueError,match='embeddings'):RewardWeightedZ().infer(model,[{**e[0],'embeddings':bad}],{'a':[0,1]})
    with pytest.raises(ValueError,match='unique'):RewardWeightedZ().infer(model,e,{'other':[0,1]})
    zero = [{**e[0],'embeddings':torch.zeros(2,4)}]
    with pytest.raises(ValueError,match='cancels'):RewardWeightedZ().infer(model,zero,{'a':[0,1]})


def task_states():
    a = np.zeros((2,56));a[:,SLICES['goal_half_extents']]=1
    a[:,SLICES['goal_orientation']]=[0,0,0,1]
    a[0,SLICES['object_position']]=[0,0,0];a[1,SLICES['object_position']]=[.1,0,0]
    a[:,SLICES['right_eef_position']]=[1,0,0]
    return a


def test_measured_proxies_and_unavailable_components():
    a = task_states()
    r,d = measured_rewards(a,'approach',hand='left',scale_m=.1)
    np.testing.assert_allclose(r,[1,np.exp(-1)])
    assert d['kind']=='positional_proxy_not_physical_success'
    goal,_ = measured_rewards(a,'goal_proximity',scale_m=.1)
    np.testing.assert_allclose(goal,r)
    for name in ('grasp','transport','placement'):
        with pytest.raises(ValueError,match='Unsupported reward component'):measured_rewards(a,name)
    for bad in (None, np.zeros((2,55)), np.full((2,56),np.nan)):
        with pytest.raises(ValueError):measured_rewards(bad,'approach')
    with pytest.raises(ValueError):measured_rewards(a,'approach',scale_m=0)
    with pytest.raises(ValueError):measured_rewards(a,'approach',hand='unknown')


def test_annotation_alignment_and_corruption():
    s = dict(episode_id='a',states=2,source_fingerprint='digest',endpoints=[{'record_ns':10},{'record_ns':20}])
    annotation = dict(format='bfm.reward_annotations.v1',definition='external measurements',episodes=[{**s,'rewards':[-1,2]}])
    assert annotation_rewards(annotation,[s])['a'].tolist()==[-1,2]
    for field,value in [('source_fingerprint','wrong'),('states',3),('endpoints',list(reversed(s['endpoints']))),('rewards',[1]),('rewards',[None,1])]:
        bad = copy.deepcopy(annotation);bad['episodes'][0][field]=value
        with pytest.raises(ValueError):annotation_rewards(bad,[s])
    bad = copy.deepcopy(annotation);bad['episodes']*=2
    with pytest.raises(ValueError,match='IDs'):annotation_rewards(bad,[s])
    with pytest.raises(ValueError):annotation_rewards({**annotation,'episodes':[]},[s])
    with pytest.raises(ValueError):annotation_rewards([], [s])


def test_real_adapter_binding_artifacts_actions_and_cli(trained,tmp_path):
    runtime,manifest,checkpoint = trained
    root = zarr.open_group(str(manifest.parent/'episode.zarr'),mode='a');root.attrs['success']=True
    before = snapshot(manifest.parent/'episode.zarr');weights = checkpoint.read_bytes();manifest_before=manifest.read_bytes()
    episodes = episode_embeddings(runtime,manifest,3)
    ep = episodes[0];s,states = reference_rows(ep)
    points = retained_endpoints(root)
    assert len(points)==10 and s['states']==10 and points[-1][1]=='next_obs'
    expected = {}
    for p,key in [('obs','record_ns'),('next_obs','next_record_ns')]:
        for row,t in enumerate(root[key][:]):expected[int(t)]=(p,row)
    assert points==[(t,*expected[t]) for t in sorted(expected)]
    # This exact backward path and state set are shared with EpisodeBalancedZ.
    from bfm.data.visual_replay import VisualReplayBuffer
    from bfm.data.sharpa import SharpaDataset
    obs = VisualReplayBuffer(SharpaDataset(manifest,observation_profile='vision_proprio_v1')).observations([0])
    with torch.no_grad():torch.testing.assert_close(ep['embeddings'][0],runtime.model.backward_map(obs)[0])
    baseline = tmp_path/'baseline.pt';save_task(runtime,episodes,baseline)
    old_z = runtime.load_task(baseline,require_bound=True).clone();old_action=runtime.act(obs).clone()
    template = tmp_path/'annotations.json';build(runtime,manifest,template,template=True,batch_size=3)
    annotation = json.loads(template.read_text());annotation['definition']='synthetic TEST annotations, not physical success'
    annotation['episodes'][0]['rewards']=list(np.linspace(-1,2,10));template.write_text(json.dumps(annotation))
    original_annotation = template.read_bytes()
    output = tmp_path/'weighted.pt';result=build(runtime,manifest,output,annotations=template,beta=2,batch_size=3)
    assert torch.equal(runtime.z,old_z) and torch.equal(runtime.act(obs),old_action)
    assert result['method']=='reward_weighted_z_v1' and result['reference_state_count']==10
    fresh = type(runtime)(checkpoint);fresh.load_task(output,require_bound=True)
    assert fresh.act(obs).shape==(26,) and torch.isfinite(fresh.act(obs)).all()
    runtime.load_task(baseline,require_bound=True)
    comparison=compare_tasks(runtime,baseline,output,[baseline],tmp_path/'compare',timesteps=[0,2])
    assert len(comparison['cosine_similarity'])==3
    rows=list(csv.DictReader((tmp_path/'compare/action_comparison.csv').open()))
    assert len(rows)==6 and all(float(r['max_abs'])==0 for r in rows if r['method']=='EpisodeBalancedZ')
    wrong=tmp_path/'wrong.pt';torch.save({**result,'checkpoint_sha256':'wrong'},wrong)
    with pytest.raises(ValueError,match='different checkpoint'):runtime.load_task(wrong,require_bound=True)
    with pytest.raises(ValueError,match='different checkpoint'):compare_tasks(runtime,baseline,output,[wrong],tmp_path/'bad-compare')
    assert torch.equal(runtime.z,old_z)
    assert before==snapshot(manifest.parent/'episode.zarr') and checkpoint.read_bytes()==weights and manifest.read_bytes()==manifest_before
    assert template.read_bytes()==original_annotation and torch.equal(runtime.z,old_z)
    command=[sys.executable,str(Path(__file__).parents[2]/'tools/build_reward_weighted_z.py'),
             '--checkpoint',str(checkpoint),'--references',str(manifest),'--annotations',str(template),
             '--output',str(tmp_path/'cli.pt'),'--baseline-task',str(baseline),
             '--cluster-tasks',str(baseline),'--timesteps','0','--batch-size','3']
    subprocess.run(command,check=True,capture_output=True)
    assert (tmp_path/'cli_comparison/comparison_summary.json').is_file()
    with pytest.raises(FileExistsError):build(runtime,manifest,output,annotations=template)


def test_existing_reward_inference_is_independent():
    model=FBModel(ModelConfig(state_dim=56,action_dim=26,latent_dim=4,hidden_dim=16)).eval()
    states=torch.arange(168,dtype=torch.float32).reshape(3,56)/100
    r=torch.tensor([-1.,0,2.])
    with torch.no_grad():b=model.backward_map(states)
    e=[dict(episode_id='one',success=True,embeddings=b)]
    z,_=RewardWeightedZ(beta=0).infer(model,e,{'one':r})
    with torch.no_grad():expected=model.reward_inference(states,r)
    torch.testing.assert_close(z,expected)

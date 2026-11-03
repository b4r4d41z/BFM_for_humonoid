"""Offline diagnostics of projected episode directions; no production policy changes."""
import csv
import html
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from bfm.contracts.sharpa import ActionNormalizer, GROUPS
from bfm.contracts.vision import CAMERAS, proprioception
from bfm.data.sharpa import SharpaDataset
from .latents import balanced_task, episode_embeddings, pca
from .runtime import checkpoint_digest


def write_csv(path, fields, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def source_dataset(path):
    with tempfile.TemporaryDirectory() as temp:
        manifest = Path(temp)/'reference.json'
        write_json(manifest, {'schema_version':'3.0.0', 'dataset_id':'episode-z-diagnostic',
                             'episodes':[{'path':str(Path(path).resolve()), 'split':'train'}]})
        return SharpaDataset(manifest, observation_profile='vision_proprio_v1')


def validate_artifact(saved, checkpoint=None):
    if saved.get('format') != 'bfm.episode_z.v1' or (checkpoint and saved['checkpoint_sha256'] != checkpoint):
        raise ValueError('Analysis format/checkpoint mismatch')
    z = torch.as_tensor(saved['episode_z']).double()
    g = torch.as_tensor(saved['global_z']).double()
    n, dim = len(saved['episodes']), saved['latent_dim']
    ids = [e['episode_id'] for e in saved['episodes']]
    if not n or len(set(ids)) != n or z.shape != (n,dim) or g.shape != (1,dim):
        raise ValueError('Invalid episode IDs or latent dimensions')
    for vectors in (z,g):
        if not torch.isfinite(vectors).all() or not torch.allclose(vectors.norm(dim=1), torch.full((len(vectors),), dim**.5, dtype=torch.float64), atol=1e-5, rtol=1e-5):
            raise ValueError('Latents must be finite projected task directions')
    return z.numpy(), g.numpy()


def diversity(z, global_z):
    z, g = np.asarray(z,dtype=np.float64), np.asarray(global_z,dtype=np.float64)
    if z.ndim != 2 or not len(z) or g.shape != (1,z.shape[1]) or not np.isfinite(z).all() or not np.isfinite(g).all():
        raise ValueError('Invalid finite latent dimensions')
    norms = np.linalg.norm(z,axis=1,keepdims=True); gn = np.linalg.norm(g)
    if (norms == 0).any() or gn == 0: raise ValueError('Zero latent direction')
    unit = z/norms
    return np.clip(unit@unit.T,-1,1), np.clip((unit@(g/gn).T)[:,0],-1,1)


def statistics(values):
    a = np.asarray(values)
    return {k:float(f(a)) if a.size else None for k,f in
            [('min',np.min),('mean',np.mean),('median',np.median),('max',np.max),('std',np.std)]}


def representatives(ids, distances):
    order = sorted(range(len(ids)), key=lambda i:(distances[i],ids[i]))
    return list(dict.fromkeys(ids[order[i]] for i in (0,len(order)//2,len(order)-1)))


def analyze(runtime, references, task_z, output, batch_size=8):
    global_z = runtime.load_task(task_z, require_bound=True).cpu()
    episodes = sorted((e for e in episode_embeddings(runtime,references,batch_size) if e['success'] is True), key=lambda e:e['episode_id'])
    if not episodes: raise ValueError('No successful reference episodes')
    vectors = torch.cat([balanced_task(runtime.model,[e]) for e in episodes])
    sources = []
    for e in episodes:
        data = source_dataset(e['path'])
        sources.append({'episode_id':e['episode_id'], 'path':e['path'], 'states':len(e['embeddings']),
                        'source_fingerprint':data.episode_fingerprints[0]})
    saved = dict(format='bfm.episode_z.v1', checkpoint_sha256=runtime.checkpoint_sha256,
                 task_z_path=str(Path(task_z).resolve()), task_z_sha256=checkpoint_digest(task_z),
                 references=str(Path(references).resolve()), latent_dim=runtime.model.cfg.latent_dim,
                 method='project_z(mean_episode_B); balanced_task(single_episode)', episodes=sources,
                 episode_z=vectors, global_z=global_z)
    z,g = validate_artifact(saved); pair,sim = diversity(z,g); distance=1-sim
    ids=[e['episode_id'] for e in sources]; upper=pair[np.triu_indices(len(z),1)]
    order=sorted(range(len(ids)),key=lambda i:(distance[i],ids[i]))
    summary=dict(successful_episodes=len(z),latent_dim=z.shape[1],checkpoint_sha256=runtime.checkpoint_sha256,
                 pairwise_cosine_similarity=statistics(upper),pairwise_cosine_distance=statistics(1-upper),
                 similarity_to_global=statistics(sim),cosine_distance_to_global=statistics(distance),
                 nearest_episode=ids[order[0]],farthest_episode=ids[order[-1]],
                 note='Off-diagonal pairs only; one episode has no pairwise diversity estimate. No scientific conclusion is inferred automatically.')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.save(saved,output/'episode_z.pt')
    rows=[dict(episode_id=e['episode_id'],path=e['path'],states=e['states'],cosine_to_global=float(sim[i]),
               cosine_distance_to_global=float(distance[i]),euclidean_to_global=float(np.linalg.norm(z[i]-g[0]))) for i,e in enumerate(sources)]
    write_csv(output/'episode_metrics.csv',list(rows[0]),rows)
    write_csv(output/'pairwise_cosine_similarity.csv',['episode_id',*ids],[dict(episode_id=identity,**dict(zip(ids,pair[i]))) for i,identity in enumerate(ids)])
    write_json(output/'summary.json',summary)
    return summary


def visualize_analysis(artifact, output):
    saved=torch.load(artifact,map_location='cpu',weights_only=True);z,g=validate_artifact(saved)
    # Fit only episodes; the global task is an out-of-sample projection.
    coordinates=pca(z)
    centered=z-z.mean(0);_,s,axes=np.linalg.svd(centered,full_matrices=False)
    energy=float(np.square(s).sum());ratios=np.square(s[:2])/energy if energy else np.zeros(2)
    global_xy=(g-z.mean(0))@axes[:2].T
    ids=[e['episode_id'] for e in saved['episodes']]
    rows=[dict(episode_id=identity,kind='episode',pc1=float(x),pc2=float(y)) for identity,(x,y) in zip(ids,coordinates)]
    rows.append(dict(episode_id='global_task_z',kind='global',pc1=float(global_xy[0,0]),pc2=float(global_xy[0,1])))
    summary=dict(pc1_explained_variance_ratio=float(ratios[0]),pc2_explained_variance_ratio=float(ratios[1]),
                 pc1_pc2_cumulative=float(ratios.sum()),variance_defined=energy>0,fit='episode_z only',
                 checkpoint_sha256=saved['checkpoint_sha256'],analysis_sha256=checkpoint_digest(artifact))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_csv(output/'pca_coordinates.csv',list(rows[0]),rows);write_json(output/'pca_summary.json',summary)
    # Standalone SVG avoids a plotting dependency and offers full-ID hover labels.
    points=np.r_[coordinates,global_xy];span=np.ptp(points,axis=0);xy=(points-points.min(0))/np.where(span>0,span,1)*640+60
    marks=[]
    for i,((x,y),row) in enumerate(zip(xy,rows)):
        color='crimson' if row['kind']=='global' else '#2563eb';radius=8 if row['kind']=='global' else 4
        label='GLOBAL' if row['kind']=='global' else str(i+1)
        marks.append(f'<g><title>{html.escape(row["episode_id"])}</title><circle cx="{x}" cy="{760-y}" r="{radius}" fill="{color}"/><text x="{x+7}" y="{753-y}" font-size="10">{label}</text></g>')
    (output/'episode_z_pca.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="820" height="820"><rect width="100%" height="100%" fill="white"/><text x="25" y="25">Episode z (blue, row numbers); global task (red). PC1 horizontal, PC2 vertical.</text>'+''.join(marks)+'</svg>')
    return summary


def action_metrics(delta):
    a=np.asarray(delta,dtype=np.float64)
    if a.shape != (26,) or not np.isfinite(a).all(): raise ValueError('Action difference must be finite 26D')
    result=dict(l2=float(np.linalg.norm(a)),mae=float(np.abs(a).mean()),max_abs=float(np.abs(a).max()))
    for group,sl in GROUPS.items(): result[group+'_mae']=float(np.abs(a[sl]).mean())
    return result


@torch.no_grad()
def compare_actions(runtime, artifact, output, episode_id=None, timesteps=None, z_ids=None):
    saved=torch.load(artifact,map_location='cpu',weights_only=True)
    z,g=validate_artifact(saved,runtime.checkpoint_sha256);_,sim=diversity(z,g)
    ids=[e['episode_id'] for e in saved['episodes']]
    selected=z_ids if z_ids is not None else representatives(ids,1-sim)
    if not selected or len(set(selected))!=len(selected) or any(i not in ids for i in selected): raise ValueError('Unknown/duplicate episode z IDs')
    episode_id=episode_id or ids[0]
    if episode_id not in ids: raise ValueError('Unknown observation episode ID')
    episode=saved['episodes'][ids.index(episode_id)];data=source_dataset(episode['path']);root=data.roots[0]
    if data.ids[0]!=episode_id or data.episode_fingerprints[0]!=episode['source_fingerprint']: raise ValueError('Reference source changed since analysis')
    steps=timesteps if timesteps is not None else np.unique(np.linspace(0,len(root['action'])-1,min(3,len(root['action'])) ,dtype=int)).tolist()
    if not steps or any(type(i) is not int or not 0<=i<len(root['action']) for i in steps): raise ValueError('Invalid retained obs row indices')
    rows=[];metrics=[];normalizer=ActionNormalizer()
    for step in sorted(set(steps)):
        obs={'proprio':proprioception(root.attrs['joint_names'],root['obs/joint_position'][step],root['obs/joint_velocity'][step]),
             **{c:root['obs/images/'+c][step] for c in CAMERAS}}
        h=runtime.model.encode_observation(runtime.observation(obs))
        actions=[]
        for latent in [g[0],*[z[ids.index(i)] for i in selected]]:
            task=runtime.task(latent)
            a=runtime.model.act_encoded(h,task,deterministic=True)[0].cpu()
            if a.shape!=(26,) or not torch.isfinite(a).all() or (a.abs()>1).any(): raise ValueError('Invalid deterministic action')
            actions.append(a.numpy())
        for identity,a in zip(['global',*selected],actions):
            physical=normalizer.physical(a).numpy();delta=a-actions[0];pd=physical-normalizer.physical(actions[0]).numpy()
            row=dict(observation_episode_id=episode_id,timestep=step,record_ns=int(root['record_ns'][step]),
                     segment_id=int(root['segment_id'][step]),latent_id=identity,**action_metrics(delta))
            for i in range(26):row.update({f'normalized_{i}':float(a[i]),f'physical_{i}':float(physical[i]),f'delta_{i}':float(delta[i]),f'physical_delta_{i}':float(pd[i])})
            rows.append(row)
            if identity!='global':metrics.append(action_metrics(delta))
    summary=dict(checkpoint_sha256=runtime.checkpoint_sha256,analysis_sha256=checkpoint_digest(artifact),
                 observation_episode_id=episode_id,timesteps=sorted(set(steps)),selected_z_ids=selected,
                 selection='nearest, median-rank, farthest by cosine distance; ties by episode ID' if z_ids is None else 'explicit IDs',
                 normalized_difference={k:statistics([m[k] for m in metrics]) for k in metrics[0]},
                 note='Deterministic actions on selected retained observations; sensitivity does not establish rollout quality or causality.')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_csv(output/'action_comparison.csv',list(rows[0]),rows);write_json(output/'action_comparison_summary.json',summary)
    return summary

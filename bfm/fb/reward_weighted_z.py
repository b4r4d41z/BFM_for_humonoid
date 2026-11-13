"""Independent post-training reward-weighted task inference; no actor/training changes."""
from pathlib import Path

import numpy as np
import torch

from bfm.contracts.sharpa import SLICES, validate_states, ActionNormalizer
from bfm.contracts.vision import CAMERAS, proprioception
from .latent_analysis import source_dataset, statistics, action_metrics, write_csv, write_json
from .latents import episode_embeddings, retained_endpoints
from .runtime import checkpoint_digest


METHOD = 'reward_weighted_z_v1'
ANNOTATIONS = 'bfm.reward_annotations.v1'


def finite_rewards(values, count):
    try:rewards = torch.as_tensor(values, dtype=torch.float64, device='cpu')
    except (TypeError, ValueError) as exc:raise ValueError('Rewards must be real numbers; annotate every null endpoint') from exc
    if rewards.shape != (count,) or not count or not torch.isfinite(rewards).all():
        raise ValueError('Rewards must be finite [N] aligned with retained endpoints')
    return rewards


def reward_statistics(values):
    # Scaling also keeps std/median/mean finite for very large finite rewards.
    a = np.asarray(values,dtype=np.float64);scale = float(np.abs(a).max()) or 1.
    return {key:value*scale for key,value in statistics(a/scale).items()}


class RewardWeightedZ:
    """project_z(mean_e sum_t softmax(beta*r_e)[t] * r_et * B(s_et))."""
    def __init__(self, beta=1.):
        if not np.isfinite(beta) or beta < 0:
            raise ValueError('beta must be finite and nonnegative (inverse reward units)')
        self.beta = float(beta)

    @torch.no_grad()
    def infer(self, model, episodes, rewards):
        selected = sorted((e for e in episodes if e['success'] is True), key=lambda e:e['episode_id'])
        ids = [e['episode_id'] for e in selected]
        if not ids or len(set(ids)) != len(ids) or set(rewards) != set(ids):
            raise ValueError('Exactly one reward sequence per unique successful episode required')
        values = [finite_rewards(rewards[e['episode_id']], len(e['embeddings'])) for e in selected]
        all_rewards = torch.cat(values)
        if all_rewards.max() == all_rewards.min():
            raise ValueError('Constant rewards contain no task preference; use EpisodeBalancedZ or informative rewards')
        # One common positive scale cancels under project_z and avoids overflow.
        scale = all_rewards.abs().max()
        directions, magnitudes, diagnostics = [], [], []
        for episode, r in zip(selected, values):
            b = torch.as_tensor(episode['embeddings'], dtype=torch.float64, device='cpu')
            if b.shape != (len(r), model.cfg.latent_dim) or not torch.isfinite(b).all():
                raise ValueError('B embeddings must be finite [states, checkpoint latent_dim]')
            # Subtract before multiplying: logits have max zero even for large beta.
            shifted = r/scale-r.max()/scale
            factor = self.beta*float(scale)
            logits = torch.where(shifted==0, 0., -torch.inf) if np.isinf(factor) else shifted*factor
            weights = torch.softmax(logits, dim=0)
            weighted = weights*(r/scale)
            directions.append((weighted[:,None]*b).sum(0))
            magnitudes.append((weighted.abs()*b.norm(dim=1)).sum())
            diagnostics.append(dict(episode_id=episode['episode_id'], states=len(r),
                                    rewards=reward_statistics(r.numpy()), constant=bool(r.max()==r.min()),
                                    effective_weighted_states=float(1/weights.square().sum())))
        direction = torch.stack(directions).mean(0, keepdim=True)
        magnitude = torch.stack(magnitudes).mean()
        if (not torch.isfinite(direction).all() or
                direction.norm() <= torch.finfo(torch.float64).eps*len(selected)*magnitude):
            raise ValueError('Reward-weighted B cancels or has no usable latent direction')
        # Direction-only rescaling avoids underflow in the existing FP32 projector.
        z = model.project_z((direction/direction.abs().max()).float().to(model.device)).detach().cpu()
        if z.shape != (1, model.cfg.latent_dim) or not torch.isfinite(z).all():
            raise ValueError('Invalid projected reward-weighted task')
        return z, dict(beta=self.beta, episodes=len(selected), states=len(all_rewards),
                       reward_scale=float(scale), rewards=reward_statistics(all_rewards.numpy()), per_episode=diagnostics,
                       aggregation='equal episodes; within-episode softmax(beta*r) times signed r; project once')


def measured_rewards(states, component, *, hand='either', scale_m=.1):
    """Bounded positional proxies, never grasp/placement success indicators."""
    required = {'approach':'EEF and object positions', 'goal_proximity':'object and goal center positions'}
    if component not in required:
        raise ValueError('Unsupported reward component: grasp/contact and verified placement are not recorded; '
                         'transport requires grasp evidence. Use approach, goal_proximity, or external annotations')
    if hand not in ('left','right','either') or not np.isfinite(scale_m) or scale_m <= 0:
        raise ValueError('Choose left/right/either hand and a positive finite scale_m')
    a = np.asarray(states, dtype=np.float64)
    validate_states(a)
    obj = a[:,SLICES['object_position']]
    if component == 'approach':
        sides = ('left','right') if hand == 'either' else (hand,)
        distance = np.min([np.linalg.norm(a[:,SLICES[s+'_eef_position']]-obj, axis=1) for s in sides], axis=0)
    else:
        distance = np.linalg.norm(obj-a[:,SLICES['goal_center']], axis=1)
    return np.exp(-distance/scale_m), dict(component=component, kind='positional_proxy_not_physical_success',
                                          required_fields=required[component], hand=hand if component=='approach' else None,
                                          scale_m=scale_m, equation='exp(-Euclidean_distance_m/scale_m)', range=[0.,1.])


def reference_rows(episode):
    """Bind rewards to exactly the endpoints encoded by episode_embeddings()."""
    data = source_dataset(episode['path']);root = data.roots[0]
    if root.attrs['source_episode_id'] != episode['episode_id'] or root.attrs.get('success') is not True:
        raise ValueError('Reference episode identity/success changed')
    points = retained_endpoints(root)
    if len(points) != len(episode['embeddings']):
        raise ValueError('Reference endpoint/embedding count mismatch')
    endpoints = [dict(record_ns=t, prefix=p, row=int(row), segment_id=int(root['segment_id'][row]),
                      source_record_ns=root[f'{p}/source_record_ns'][row].tolist()) for t,p,row in points]
    source = dict(episode_id=episode['episode_id'], path=str(Path(episode['path']).resolve()),
                  source_fingerprint=data.episode_fingerprints[0], states=len(points), endpoints=endpoints)
    states = np.stack([root[f'{p}/state'][row] for _,p,row in points])
    return source, states


def annotation_rewards(saved, sources):
    if not isinstance(saved,dict) or saved.get('format') != ANNOTATIONS or not isinstance(saved.get('definition'), str) or not saved['definition'].strip():
        raise ValueError('Annotations require bfm.reward_annotations.v1 and a nonempty definition')
    entries = saved.get('episodes', [])
    if not isinstance(entries,list) or any(not isinstance(e,dict) or not isinstance(e.get('episode_id'),str) or 'rewards' not in e for e in entries):
        raise ValueError('Annotations require episode IDs and complete numeric rewards')
    ids = [e['episode_id'] for e in entries]
    if len(set(ids)) != len(ids) or set(ids) != {s['episode_id'] for s in sources}:
        raise ValueError('Annotation episode IDs must exactly match successful references, without duplicates')
    by_id = {e['episode_id']:e for e in entries};result = {}
    for source in sources:
        entry = by_id[source['episode_id']]
        if (entry.get('states') != source['states'] or entry.get('source_fingerprint') != source['source_fingerprint']
                or entry.get('endpoints') != source['endpoints']):
            raise ValueError('Annotation source fingerprint/order/retained endpoints mismatch: '+source['episode_id'])
        result[source['episode_id']] = finite_rewards(entry['rewards'], source['states'])
    return result


def build(runtime, references, output, *, beta=1., component=None, annotations=None,
          hand='either', scale_m=.1, batch_size=8, template=False):
    import json
    inference = RewardWeightedZ(beta)
    if not template and (component is None) == (annotations is None):
        raise ValueError('Choose exactly one calculated reward component or annotation file')
    output = Path(output)
    if output.exists():raise FileExistsError(output)
    # Authoritative B(s), observation selection, success filter and endpoint semantics.
    episodes = sorted((e for e in episode_embeddings(runtime,references,batch_size) if e['success'] is True), key=lambda e:e['episode_id'])
    if not episodes:raise ValueError('No successful reference trajectories')
    rows = [reference_rows(e) for e in episodes];sources = [s for s,_ in rows]
    if template:
        payload = dict(format=ANNOTATIONS, definition='REPLACE with measured reward definition and provenance',
                       episodes=[{**s,'rewards':[None]*s['states']} for s in sources])
        output.parent.mkdir(parents=True,exist_ok=True);write_json(output,payload)
        return payload
    if annotations is not None:
        path = Path(annotations)
        saved = json.loads(path.read_text());rewards = annotation_rewards(saved,sources)
        definition = dict(kind='external_per_endpoint_annotations', definition=saved['definition'],
                          path=str(path.resolve()), sha256=checkpoint_digest(path))
    else:
        calculated = [measured_rewards(state,component,hand=hand,scale_m=scale_m) for _,state in rows]
        rewards = {s['episode_id']:r for s,(r,_) in zip(sources,calculated)}
        definition = calculated[0][1]
    z, diagnostics = inference.infer(runtime.model,episodes,rewards)
    payload = dict(z=z, checkpoint_sha256=runtime.checkpoint_sha256, latent_dim=runtime.model.cfg.latent_dim,
                   method=METHOD, reward_definition=definition, weighting=diagnostics['aggregation'], beta=float(beta),
                   reference_episode_count=len(sources), reference_state_count=sum(s['states'] for s in sources),
                   references=sources, reward_diagnostics=diagnostics)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,output)
    original = runtime.z
    try:runtime.load_task(output,require_bound=True)
    finally:runtime.z = original
    return payload


@torch.no_grad()
def compare_tasks(runtime, baseline, weighted, clusters, output, *, timesteps=None):
    """Same checkpoint and same obs for all tasks; reports sensitivity, not quality."""
    output = Path(output)
    if output.exists():raise FileExistsError(output)
    paths = [Path(baseline),Path(weighted),*[Path(p) for p in clusters]]
    names = ['EpisodeBalancedZ','RewardWeightedZ',*[f'ClusterZ_{i}' for i in range(len(clusters))]]
    tasks, metadata = [], [];original = runtime.z
    try:
        for path in paths:
            tasks.append(runtime.load_task(path,require_bound=True).detach().cpu())
            payload = torch.load(path,map_location='cpu',weights_only=True)
            refs = payload.get('references',[])
            metadata.append(dict(path=str(path.resolve()),sha256=checkpoint_digest(path),method=payload.get('method'),
                                 episodes=len(refs),states=sum(e.get('states',0) for e in refs),
                                 reward_diagnostics=payload.get('reward_diagnostics')))
    finally:runtime.z = original
    z = torch.cat(tasks).double().numpy();unit=z/np.linalg.norm(z,axis=1,keepdims=True)
    similarity = np.clip(unit@unit.T,-1,1)
    weighted_meta = torch.load(weighted,map_location='cpu',weights_only=True)
    source = weighted_meta['references'][0];data = source_dataset(source['path']);root = data.roots[0]
    if data.episode_fingerprints[0] != source['source_fingerprint']:
        raise ValueError('Comparison observation source fingerprint changed')
    steps = timesteps if timesteps is not None else np.unique(np.linspace(0,len(root['action'])-1,min(3,len(root['action'])),dtype=int)).tolist()
    if not steps or any(type(i) is not int or not 0<=i<len(root['action']) for i in steps):
        raise ValueError('Invalid comparison obs row indices')
    rows = [];normalizer = ActionNormalizer()
    for step in sorted(set(steps)):
        obs = dict(proprio=proprioception(root.attrs['joint_names'],root['obs/joint_position'][step],root['obs/joint_velocity'][step]),
                   **{c:root[f'obs/images/{c}'][step] for c in CAMERAS})
        h = runtime.model.encode_observation(runtime.observation(obs))
        actions = [runtime.model.act_encoded(h,runtime.task(t),deterministic=True)[0].cpu().numpy() for t in tasks]
        for name,a in zip(names,actions):
            if a.shape!=(26,) or not np.isfinite(a).all() or (np.abs(a)>1).any():raise ValueError('Invalid deterministic 26D action')
            physical = normalizer.physical(a).numpy();delta = a-actions[0]
            rows.append(dict(episode_id=source['episode_id'],timestep=step,method=name,**action_metrics(delta),
                             **{f'normalized_{j}':float(v) for j,v in enumerate(a)},
                             **{f'physical_{j}':float(v) for j,v in enumerate(physical)},
                             **{f'delta_{j}':float(v) for j,v in enumerate(delta)}))
    summary = dict(checkpoint_sha256=runtime.checkpoint_sha256,methods=names,
                   norms=np.linalg.norm(z,axis=1).tolist(),cosine_similarity=similarity.tolist(),artifacts=metadata,
                   observation_episode_id=source['episode_id'],timesteps=sorted(set(steps)),
                   note='Fixed observations and deterministic actor; differences do not establish rollout performance.')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'comparison_summary.json',summary)
    write_csv(output/'task_similarity.csv',['method',*names],[dict(method=name,**dict(zip(names,similarity[i]))) for i,name in enumerate(names)])
    write_csv(output/'action_comparison.csv',list(rows[0]),rows)
    return summary

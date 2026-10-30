"""Post-training reference task inference, callback-based search, and diagnostics."""
import csv
import html
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import zarr
from bfm.contracts.vision import CAMERAS, proprioception
from bfm.data.sharpa import SharpaDataset, validate


def episode_embeddings(runtime, source, batch_size=8):
    if batch_size < 1:
        raise ValueError('Positive embedding batch size required')
    source = Path(source).resolve()
    if source.is_file():
        manifest = json.loads(source.read_text())
        if manifest.get('schema_version') != '3.0.0' or not manifest.get('dataset_id') or any(e['split'] not in ('train','validation','test') for e in manifest['episodes']):
            raise ValueError('Unsupported reference manifest schema/split')
        paths = [(source.parent/e['path']).resolve() for e in manifest['episodes']]
    else:
        paths = [source] if source.suffix == '.zarr' else sorted(source.glob('*.zarr'))
    if not paths:
        raise ValueError('No reference Zarr episodes')
    # Use the real dataset adapter to validate all reference sources and names.
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp)/'references.json'
        manifest.write_text(json.dumps({'schema_version':'3.0.0','dataset_id':'task-references',
            'episodes':[{'path':str(p),'split':'train'} for p in paths]}))
        data = SharpaDataset(manifest, observation_profile='vision_proprio_v1')
    episodes = []
    for root in data.roots:
        validate(root)
        # Full retained trajectory, including final next_obs; shared endpoints
        # appear once. Segments are averaged together within the source episode.
        endpoints = {}
        for prefix, key in [('obs','record_ns'),('next_obs','next_record_ns')]:
            for row, time in enumerate(root[key][:]):
                endpoints[int(time)] = (prefix,row)
        points = [endpoints[t] for t in sorted(endpoints)]
        chunks = []
        for start in range(0,len(points),batch_size):
            selected = points[start:start+batch_size]
            obs = {'proprio':np.stack([proprioception(root.attrs['joint_names'],
                root[f'{p}/joint_position'][r],root[f'{p}/joint_velocity'][r]) for p,r in selected])}
            obs.update({c:np.stack([root[f'{p}/images/{c}'][r] for p,r in selected]) for c in CAMERAS})
            with torch.no_grad():
                chunks.append(runtime.model.backward_map(obs).cpu())
        embeddings = torch.cat(chunks)
        if not torch.isfinite(embeddings).all():
            raise ValueError('Nonfinite B embeddings')
        episodes.append({'episode_id':root.attrs['source_episode_id'], 'success':root.attrs.get('success'),
                         'embeddings':embeddings, 'path':str(root.store.path)})
    return episodes


def balanced_task(model, episodes):
    successful = [e for e in episodes if e['success'] is True]
    if not successful or any(not len(e['embeddings']) for e in successful):
        raise ValueError('No usable successful reference trajectories')
    mean = torch.stack([e['embeddings'].mean(0) for e in successful]).mean(0,keepdim=True)
    return model.project_z(mean.to(model.device)).detach().cpu()


def save_task(runtime, episodes, output):
    z = balanced_task(runtime.model, episodes)
    payload = {'z':z, 'checkpoint_sha256':runtime.checkpoint_sha256,
               'latent_dim':runtime.model.cfg.latent_dim, 'method':'episode_balanced_full_trajectory_B_v1',
               'references':[{'episode_id':e['episode_id'],'states':len(e['embeddings']),'path':e['path']}
                             for e in episodes if e['success'] is True]}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,output)
    return payload


def candidates(runtime, count, seed=0):
    if count < 1:
        raise ValueError('Positive candidate count required')
    device = runtime.model.device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        torch.manual_seed(seed)
        vectors = runtime.model.sample_z(count).detach().cpu()
    return {'checkpoint_sha256':runtime.checkpoint_sha256,'vectors':vectors,
            'ids':[f'z-{i:06d}' for i in range(count)],'seed':seed,'method':'model.sample_z'}


def rank_candidates(runtime, proposal, rollout):
    if proposal['checkpoint_sha256'] != runtime.checkpoint_sha256:
        raise ValueError('Candidate checkpoint mismatch')
    vectors, ids = proposal['vectors'], proposal['ids']
    if vectors.shape != (len(ids),runtime.model.cfg.latent_dim) or not len(ids) or len(set(ids))!=len(ids):
        raise ValueError('Invalid candidate dimensions/IDs')
    if not torch.isfinite(vectors).all() or not torch.allclose(vectors.norm(dim=-1), torch.full((len(ids),),runtime.model.cfg.latent_dim**.5),atol=1e-5):
        raise ValueError('Candidates must use model.sample_z spherical normalization')
    scores = [float(rollout(identity,z.clone())) for identity,z in zip(ids,vectors)]
    if not np.isfinite(scores).all():
        raise ValueError('Rollout scores must be finite')
    order = sorted(range(len(ids)),key=lambda i:(-scores[i],ids[i]))
    best = order[0]
    return {'z':vectors[best:best+1],'checkpoint_sha256':runtime.checkpoint_sha256,
            'latent_dim':runtime.model.cfg.latent_dim,'method':'random_z_rollout_search',
            'ranking':[{'id':ids[i],'score':scores[i]} for i in order]}


def pca(points):
    points = np.asarray(points,dtype=np.float64)
    if points.ndim!=2 or len(points)<2 or points.shape[1]<2 or not np.isfinite(points).all():
        raise ValueError('PCA needs at least two finite latent vectors')
    centered = points-points.mean(0)
    _,_,axes = np.linalg.svd(centered,full_matrices=False)
    return centered@axes[:2].T


def visualize(points, labels, output, method='pca'):
    if len(points)!=len(labels):
        raise ValueError('One metadata label per latent point required')
    if method=='pca':
        coordinates = pca(points)
    elif method=='umap':
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ImportError('Optional umap-learn is not installed; use PCA') from exc
        coordinates = UMAP(random_state=0).fit_transform(np.asarray(points))
    else:
        raise ValueError('Unknown projection method')
    if not np.isfinite(coordinates).all():
        raise ValueError('Nonfinite projected coordinates')
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['x','y','metadata'])
        for xy,label in zip(coordinates,labels):writer.writerow([*xy,json.dumps(label,sort_keys=True)])
    xy=(coordinates-coordinates.min(0))/(np.ptp(coordinates,axis=0)+1e-12)*560+20
    palette=['#2563eb','#16a34a','#dc2626','#9333ea','#ea580c']
    groups=sorted({str(label.get('source','unknown')) for label in labels})
    marks=[f'<circle cx="{x:.3f}" cy="{600-y:.3f}" r="4" fill="{palette[groups.index(str(label.get("source","unknown")))%len(palette)]}"><title>{html.escape(json.dumps(label))}</title></circle>' for (x,y),label in zip(xy,labels)]
    output.with_suffix('.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600">'+''.join(marks)+'</svg>')
    return coordinates

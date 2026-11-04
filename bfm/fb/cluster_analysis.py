"""Post-training cosine clustering and checkpoint-bound balanced task construction."""
from pathlib import Path

import numpy as np
import torch

from .latent_analysis import diversity, source_dataset, validate_artifact, write_csv, write_json
from .latents import balanced_task, episode_embeddings
from .runtime import checkpoint_digest


def cosine_clusters(z, count):
    """Average-linkage cosine clustering; ties use input order, no random seed."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or not len(z):
        raise ValueError('Expected nonempty episode latent matrix')
    if type(count) is not int or not 1 <= count <= len(z):
        raise ValueError('clusters must be between 1 and the episode count')
    pair, _ = diversity(z, z[:1])
    groups = [(i,) for i in range(len(z))]
    while len(groups) > count:
        _, a, b = min(
            (float((1-pair[np.ix_(groups[a], groups[b])]).mean()), a, b)
            for a in range(len(groups)) for b in range(a+1, len(groups)))
        merged = tuple(sorted(groups[a]+groups[b]))
        groups = sorted([g for i, g in enumerate(groups) if i not in (a, b)]+[merged])
    labels = np.empty(len(z), dtype=int)
    for label, members in enumerate(groups):
        labels[list(members)] = label
    return labels


def load_cluster_tasks(directory, artifact, saved):
    """Validate visualization inputs without loading a model or recomputing B."""
    import json
    directory = Path(directory)
    summary = json.loads((directory/'cluster_summary.json').read_text())
    if (summary['analysis_sha256'] != checkpoint_digest(artifact)
            or summary['checkpoint_sha256'] != saved['checkpoint_sha256']
            or summary['latent_dim'] != saved['latent_dim']):
        raise ValueError('Cluster analysis/checkpoint mismatch')
    membership, vectors = {}, []
    for i, group in enumerate(summary['clusters']):
        if group['cluster_id'] != i or not group['episode_ids']:
            raise ValueError('Invalid cluster IDs or empty cluster')
        task = torch.load(directory/f'cluster_{i}_task_z.pt', map_location='cpu', weights_only=True)
        if (task['checkpoint_sha256'] != saved['checkpoint_sha256']
                or task['analysis_sha256'] != summary['analysis_sha256']
                or task['cluster_id'] != i or task['episode_ids'] != group['episode_ids']):
            raise ValueError('Cluster task provenance mismatch')
        # Reuse the analysis validator for finite, dimensional and sphere checks.
        validate_artifact({**saved, 'global_z': task['z']})
        vectors.append(torch.as_tensor(task['z']).double().numpy()[0])
        for identity in group['episode_ids']:
            if identity in membership:
                raise ValueError('Duplicate cluster member')
            membership[identity] = i
    if (set(membership) != {e['episode_id'] for e in saved['episodes']}
            or len(vectors) != summary['number_of_clusters']):
        raise ValueError('Cluster members differ from analysis')
    return np.stack(vectors), membership


def build_clusters(runtime, references, artifact, output, count=3, batch_size=8):
    saved = torch.load(artifact, map_location='cpu', weights_only=True)
    z, global_z = validate_artifact(saved, runtime.checkpoint_sha256)
    if saved['latent_dim'] != runtime.model.cfg.latent_dim:
        raise ValueError('Analysis latent dimension differs from checkpoint')
    order = sorted(range(len(z)), key=lambda i: saved['episodes'][i]['episode_id'])
    z = z[order]
    sources = [saved['episodes'][i] for i in order]
    labels = cosine_clusters(z, count)
    output = Path(output)
    if output.exists():
        raise ValueError('Output directory must be new')
    episodes = {e['episode_id']: e for e in episode_embeddings(runtime, references, batch_size)
                if e['success'] is True}
    if set(episodes) != {e['episode_id'] for e in sources}:
        raise ValueError('Successful reference episode IDs differ from analysis')
    for source in sources:
        episode = episodes[source['episode_id']]
        data = source_dataset(episode['path'])
        if (data.episode_fingerprints[0] != source['source_fingerprint']
                or len(episode['embeddings']) != source['states']):
            raise ValueError(f'Reference source changed since analysis: {source["episode_id"]}')
    digest = checkpoint_digest(artifact)
    tasks, groups = [], []
    pair, to_global = diversity(z, global_z)
    for i in range(count):
        indices = np.flatnonzero(labels == i)
        members = [episodes[sources[j]['episode_id']] for j in indices]
        # Equal episode weighting of unprojected B means, exactly as global task-z.
        task = balanced_task(runtime.model, members)
        validate_artifact({**saved, 'global_z': task})
        tasks.append(task)
        within = pair[np.ix_(indices, indices)][np.triu_indices(len(indices), 1)]
        groups.append(dict(cluster_id=i, size=len(members), episode_ids=[e['episode_id'] for e in members],
                           average_within_cluster_cosine_similarity=float(within.mean()) if within.size else None,
                           average_within_cluster_cosine_distance=float((1-within).mean()) if within.size else None,
                           minimum_similarity=float(within.min()) if within.size else None,
                           maximum_similarity=float(within.max()) if within.size else None))
    task_vectors = torch.cat(tasks).double().numpy()
    task_pair, task_global = diversity(task_vectors, global_z)
    to_task = np.stack([diversity(z, t[None])[1] for t in task_vectors], axis=1)
    rows = []
    for j, source in enumerate(sources):
        i = int(labels[j]); sim = float(to_task[j, i])
        rows.append(dict(episode_id=source['episode_id'], path=episodes[source['episode_id']]['path'],
                         cluster_id=i, cosine_similarity_to_cluster_direction=sim,
                         cosine_distance_to_cluster_direction=1-sim,
                         cosine_similarity_to_global=float(to_global[j]), cosine_distance_to_global=float(1-to_global[j])))
    similarity = [dict(source=f'cluster_{i}', target=f'cluster_{j}', cosine_similarity=float(task_pair[i, j]),
                       cosine_distance=float(1-task_pair[i, j])) for i in range(count) for j in range(count)]
    similarity += [dict(source=f'cluster_{i}', target='global', cosine_similarity=float(task_global[i]),
                        cosine_distance=float(1-task_global[i])) for i in range(count)]
    summary = dict(latent_dim=saved['latent_dim'], checkpoint_sha256=runtime.checkpoint_sha256,
                   analysis_sha256=digest, references=str(Path(references).resolve()),
                   number_of_clusters=count, total_episodes=len(z),
                   clustering_method='average_linkage_cosine_v1', random_seed=None,
                   tie_break='sorted episode IDs; lexicographic member-index tuples',
                   clusters=groups, cluster_task_cosine_similarity=task_pair.tolist(),
                   cluster_task_similarity_to_global=task_global.tolist(),
                   note='Within-cluster statistics use distinct unordered episode pairs; singletons are null. '
                        'Clustering uses full-dimensional episode z. No labels, optimal-k or behavioral claims.')
    output.mkdir(parents=True, exist_ok=False)
    for i, task in enumerate(tasks):
        members = groups[i]['episode_ids']
        payload = dict(z=task, checkpoint_sha256=runtime.checkpoint_sha256, latent_dim=saved['latent_dim'],
                       method='cluster_balanced_full_trajectory_B_v1', cluster_id=i, cluster_size=len(members),
                       episode_ids=members, analysis_sha256=digest,
                       references=[{**source, 'path': episodes[source['episode_id']]['path']}
                                   for source in sources if source['episode_id'] in members])
        path = output/f'cluster_{i}_task_z.pt'
        torch.save(payload, path)
        runtime.load_task(path, require_bound=True)
    write_csv(output/'clusters.csv', list(rows[0]), rows)
    write_csv(output/'cluster_task_similarity.csv', list(similarity[0]), similarity)
    # Written last: the visualization requires this completion/identity metadata.
    write_json(output/'cluster_summary.json', summary)
    return summary

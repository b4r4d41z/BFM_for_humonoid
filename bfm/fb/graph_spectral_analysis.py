"""Normalized kNN Laplacians of projected episode directions (NumPy only)."""
from pathlib import Path
import numpy as np
import torch
from .latent_analysis import validate_artifact, diversity, write_csv, write_json
from .runtime import checkpoint_digest
from .spectral_utils import candidates, agreement, matrix_csv, scatter, heatmap, lines


def graph(z, neighbors):
    pair, _ = diversity(z, z[:1]); n = len(z)
    if type(neighbors) is not int or not 1 <= neighbors < n: raise ValueError('neighbors must be in [1,N-1]')
    # Angular Gaussian, positive even at antipodal directions; bandwidth fixed at 1.
    affinity = np.exp(-(1-pair)); np.fill_diagonal(affinity, 0)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        order = sorted((j for j in range(n) if j != i), key=lambda j: (-pair[i, j], j))
        mask[i, order[:neighbors]] = True
    a = affinity*(mask | mask.T)
    return a, pair


def laplacian(a):
    a = np.asarray(a, dtype=float)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all()
            or (a < 0).any() or not np.allclose(a, a.T) or np.diag(a).any()): raise ValueError('Invalid affinity')
    degree = a.sum(1); inv = np.zeros_like(degree)
    inv[degree > 0] = 1/np.sqrt(degree[degree > 0])
    l = np.diag((degree > 0).astype(float))-inv[:, None]*a*inv[None, :]
    components, unseen = [], set(range(len(a)))
    while unseen:
        todo = [min(unseen)]; found = set()
        while todo:
            i = todo.pop()
            if i in found: continue
            found.add(i); todo.extend(j for j in np.flatnonzero(a[i] > 0) if j not in found)
        unseen -= found; components.append(sorted(map(int, found)))
    values, vectors = np.linalg.eigh(l)
    # Fix sign ambiguity for exported coordinates; repeated eigenspaces remain nonunique.
    for column in vectors.T:
        if column[np.argmax(np.abs(column))] < 0: column *= -1
    return values, vectors, components, np.flatnonzero(degree == 0).tolist()


def kmeans(x, k, seed):
    rng = np.random.default_rng(seed); centers = [x[int(rng.integers(len(x)))]]
    while len(centers) < k:
        d = np.min(np.sum((x[:, None]-np.array(centers)[None])**2, axis=2), axis=1)
        centers.append(x[int(np.argmax(d))])
    centers = np.array(centers)
    for _ in range(100):
        labels = np.argmin(np.sum((x[:, None]-centers[None])**2, axis=2), axis=1)
        # Deterministically split largest residuals to repair empty clusters.
        for c in range(k):
            if c not in labels:
                eligible = [i for i in range(len(x)) if np.sum(labels == labels[i]) > 1]
                i = max(eligible, key=lambda i: (float(np.sum((x[i]-centers[labels[i]])**2)), -i))
                labels[i] = c
        updated = np.array([x[labels == c].mean(0) for c in range(k)])
        if np.allclose(updated, centers, atol=1e-10, rtol=0): break
        centers = updated
    return labels


def analyze_graph(artifact, output, neighbors=(3, 5, 10), seeds=(0, 1, 2)):
    saved = torch.load(artifact, map_location='cpu', weights_only=True); z, _ = validate_artifact(saved)
    if len(z) < 4 or not seeds: raise ValueError('Graph study needs >=4 episodes and seeds')
    ids = [e['episode_id'] for e in saved['episodes']]; distance = np.clip(1-diversity(z, z[:1])[0], 0, 2)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    studies = []
    for nn in sorted(set(neighbors)):
        a, pair = graph(z, nn); values, vectors, components, isolated = laplacian(a)
        def partition(k, seed=seeds[0]):
            x = vectors[:, :k]; norm = np.linalg.norm(x, axis=1, keepdims=True)
            return kmeans(x/np.where(norm > 0, norm, 1), k, seed)
        trials, selected = candidates(distance, partition)
        stability = {str(r['k']): [agreement(r['labels'], partition(r['k'], s))['ari'] for s in seeds] for r in trials}
        gaps = np.diff(values)
        study = dict(neighbors=nn, candidates=trials, selected_k=selected['k'], silhouette=selected['silhouette'],
                     seed_stability_ari=stability, components=components, isolated_indices=isolated,
                     eigenvalues=values.tolist(), eigengaps=gaps.tolist(),
                     eigengap_candidate_k=int(np.argmax(gaps[:min(8, len(gaps))])+1))
        studies.append(study)
        dest = output/f'neighbors_{nn}'; dest.mkdir()
        write_csv(dest/'eigenvalues.csv', ['index', 'eigenvalue'], [dict(index=i+1, eigenvalue=float(v)) for i, v in enumerate(values)])
        write_csv(dest/'eigengaps.csv', ['after_k', 'gap'], [dict(after_k=i+1, gap=float(v)) for i, v in enumerate(gaps)])
        xy = vectors[:, 1:3]
        write_csv(dest/'spectral_coordinates.csv', ['episode_id', 'x', 'y'], [dict(episode_id=e, x=float(x), y=float(y)) for e, (x, y) in zip(ids, xy)])
        write_csv(dest/'graph_clusters.csv', ['episode_id', 'cluster_id'], [dict(episode_id=e, cluster_id=c) for e, c in zip(ids, selected['labels'])])
        matrix_csv(dest/'affinity.csv', ids, a)
        lines(dest/'eigenvalues.svg', np.arange(1, len(values)+1), [('eigenvalues', values)], 'Normalized Laplacian spectrum')
        lines(dest/'eigengap.svg', np.arange(1, len(values)), [('gaps', gaps)], 'Gap after k eigenvalues')
        scatter(dest/'spectral_embedding.svg', xy, ids, selected['labels'], 'Eigenvectors 2 and 3; repeated eigenspaces nonunique')
        order = np.argsort(selected['labels'], kind='stable')
        heatmap(dest/'similarity_heatmap.svg', a[np.ix_(order, order)], np.array(ids)[order], 'Affinity sorted by selected cluster')
    primary = studies[0]
    write_csv(output/'graph_clusters.csv', ['episode_id', 'cluster_id'], [dict(episode_id=e, cluster_id=c) for e, c in zip(ids, next(r for r in primary['candidates'] if r['k'] == primary['selected_k'])['labels'])])
    summary = dict(checkpoint_sha256=saved['checkpoint_sha256'], analysis_sha256=checkpoint_digest(artifact), episode_ids=ids,
                   parameters=dict(neighbors=list(sorted(set(neighbors))), seeds=list(seeds), affinity='exp(-(1-cosine)); union kNN; no self edges',
                                   clustering='row-normalized first k eigenvectors; farthest-initialized Lloyd; empty repair',
                                   selection='max cosine-distance silhouette; smaller k ties; smallest neighborhood is primary'),
                   selected_k=primary['selected_k'], silhouette=primary['silhouette'], studies=studies, warnings=[f"neighbors={s['neighbors']}: {len(s['components'])} disconnected components" for s in studies if len(s['components']) > 1],
                   note='Components are topological; eigengaps and silhouette are diagnostics, not proof of optimal k or semantic skills.')
    write_json(output/'graph_summary.json', summary)
    return summary

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
import zarr

from test_runtime_tasks import trained, visual_manifest, source
from test_latent_analysis import snapshot
from bfm.fb.cluster_analysis import build_clusters, cosine_clusters, load_cluster_tasks
from bfm.fb.latent_analysis import analyze, visualize_analysis
from bfm.fb.latents import balanced_task, episode_embeddings


def test_cosine_clustering():
    z = np.array([[1., 0], [.9, .1], [-1., 0], [-.9, -.1]])
    labels = cosine_clusters(z, 2)
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])
    np.testing.assert_array_equal(labels, cosine_clusters(z*7, 2))
    np.testing.assert_array_equal(labels, cosine_clusters(z, 2))
    for k in (1, 2, 4):
        assert len(np.unique(cosine_clusters(np.ones((4, 3)), k))) == k
    for k in (0, -1, 5, 1.5, True):
        with pytest.raises(ValueError, match='clusters'):
            cosine_clusters(z, k)
    with pytest.raises(ValueError):
        cosine_clusters(np.full((3, 2), np.nan), 2)


def test_cluster_tasks_and_visualization(trained, tmp_path, monkeypatch):
    runtime, manifest, checkpoint = trained
    root = zarr.open_group(str(manifest.parent/'episode.zarr'), mode='a')
    root.attrs['success'] = True
    spec = json.loads(manifest.read_text())
    for identity in ('second', 'third'):
        shutil.copytree(manifest.parent/'episode.zarr', manifest.parent/f'{identity}.zarr')
        copied = zarr.open_group(str(manifest.parent/f'{identity}.zarr'), mode='a')
        metadata = copied.attrs['source_metadata']
        metadata['bfm']['episode']['episode_id'] = identity
        copied.attrs.update(source_episode_id=identity, source_metadata=metadata)
        spec['episodes'].append({'path': f'{identity}.zarr', 'split': 'train'})
    manifest.write_text(json.dumps(spec))
    task = tmp_path/'global.pt'
    torch.save({'z': runtime.model.sample_z(1), 'checkpoint_sha256': runtime.checkpoint_sha256}, task)
    analyze(runtime, manifest, task, tmp_path/'analysis', 2)
    artifact = tmp_path/'analysis/episode_z.pt'
    saved = torch.load(artifact, weights_only=True)
    before = {str(p): snapshot(p) for p in manifest.parent.glob('*.zarr')}
    originals = [p.read_bytes() for p in (artifact, task, checkpoint, manifest)]
    out = tmp_path/'clusters'
    summary = build_clusters(runtime, manifest, artifact, out, 2, 2)
    assert summary['number_of_clusters'] == 2 and summary['total_episodes'] == 3
    assert np.shape(summary['cluster_task_cosine_similarity']) == (2, 2)
    episodes = episode_embeddings(runtime, manifest, 2)
    for group in summary['clusters']:
        path = out/f'cluster_{group["cluster_id"]}_task_z.pt'
        actual = runtime.load_task(path, require_bound=True).cpu()
        expected = balanced_task(runtime.model, [e for e in episodes if e['episode_id'] in group['episode_ids']])
        torch.testing.assert_close(actual, expected)
        assert actual.shape == (1, runtime.model.cfg.latent_dim) and torch.isfinite(actual).all()
        assert actual.norm().item() == pytest.approx(runtime.model.cfg.latent_dim**.5)
    plain = visualize_analysis(artifact, tmp_path/'plain')
    plotted = visualize_analysis(artifact, tmp_path/'plot', out)
    assert plotted['pc1_pc2_cumulative'] == plain['pc1_pc2_cumulative']
    rows = list(csv.DictReader((tmp_path/'plot/pca_coordinates.csv').open()))
    original_rows = list(csv.DictReader((tmp_path/'plain/pca_coordinates.csv').open()))
    assert [r['kind'] for r in rows] == ['episode']*3+['global']+['cluster_task']*2
    assert [(r['pc1'], r['pc2']) for r in rows[:4]] == [(r['pc1'], r['pc2']) for r in original_rows]
    vectors, _ = load_cluster_tasks(out, artifact, saved)
    z = saved['episode_z'].double().numpy()
    _, _, axes = np.linalg.svd(z-z.mean(0), full_matrices=False)
    np.testing.assert_allclose([[float(r['pc1']), float(r['pc2'])] for r in rows[4:]],
                               (vectors-z.mean(0))@axes[:2].T)
    assert 'GLOBAL' in (tmp_path/'plot/episode_z_cluster_pca.svg').read_text()
    assert before == {str(p): snapshot(p) for p in manifest.parent.glob('*.zarr')}
    assert originals == [p.read_bytes() for p in (artifact, task, checkpoint, manifest)]
    tools = Path(__file__).resolve().parents[2]/'tools'
    for command in (
        ['build_cluster_task_z.py', '--checkpoint', str(checkpoint), '--references', str(manifest),
         '--analysis', str(artifact), '--clusters', '2', '--batch-size', '2', '--output-dir', str(tmp_path/'cli')],
        ['visualize_latent_space.py', '--analysis', str(artifact), '--cluster-dir', str(tmp_path/'cli'),
         '--output-dir', str(tmp_path/'cli-plot')]):
        subprocess.run([sys.executable, str(tools/command[0]), *command[1:]], check=True, capture_output=True)
    bad = {**saved, 'checkpoint_sha256': 'wrong'}
    torch.save(bad, tmp_path/'wrong.pt')
    with pytest.raises(ValueError, match='checkpoint'):
        build_clusters(runtime, manifest, tmp_path/'wrong.pt', tmp_path/'bad', 2)
    with pytest.raises(ValueError, match='mismatch'):
        load_cluster_tasks(out, tmp_path/'wrong.pt', saved)
    wrong_dim = {**saved, 'latent_dim': 2,
                 'episode_z': torch.ones(3, 2), 'global_z': torch.ones(1, 2)}
    torch.save(wrong_dim, tmp_path/'dimension.pt')
    with pytest.raises(ValueError, match='dimension differs'):
        build_clusters(runtime, manifest, tmp_path/'dimension.pt', tmp_path/'dimension', 2)
    # Controlled unprojected B magnitudes prove balancing is not mean(project_z(B)).
    controlled = []
    for i, e in enumerate(episodes):
        embeddings = torch.zeros_like(e['embeddings'])
        embeddings[:, i] = 1 if i == 0 else 10
        controlled.append({**e, 'embeddings': embeddings})
    monkeypatch.setattr('bfm.fb.cluster_analysis.episode_embeddings', lambda *args: controlled)
    build_clusters(runtime, manifest, artifact, tmp_path/'balanced', 1)
    result = runtime.load_task(tmp_path/'balanced/cluster_0_task_z.pt', require_bound=True).cpu()
    torch.testing.assert_close(result, balanced_task(runtime.model, controlled))
    projected_mean = runtime.model.project_z(torch.cat([balanced_task(runtime.model, [e]) for e in controlled]).mean(0, keepdim=True)).cpu()
    assert not torch.allclose(result, projected_mean)
    monkeypatch.setattr('bfm.fb.cluster_analysis.episode_embeddings', lambda *args: controlled[:-1])
    with pytest.raises(ValueError, match='IDs differ'):
        build_clusters(runtime, manifest, artifact, tmp_path/'missing', 2)
    monkeypatch.setattr('bfm.fb.cluster_analysis.episode_embeddings', lambda *args: controlled)
    root['obs/images/head'][0] = 255
    with pytest.raises(ValueError, match='source changed'):
        build_clusters(runtime, manifest, artifact, tmp_path/'changed', 2)

"""Time-axis Hann periodograms of unprojected B; no FFT across latent coordinates."""
from pathlib import Path
import numpy as np
import torch
from .latent_analysis import source_dataset, validate_artifact, write_csv, write_json
from .latents import episode_embeddings, pca
from .runtime import checkpoint_digest
from .spectral_utils import candidates, matrix_csv, heatmap, scatter, lines


def spectrum(times, embeddings, grid):
    t, b = np.asarray(times, dtype=float), np.asarray(embeddings, dtype=float)
    if (t.ndim != 1 or b.ndim != 2 or len(t) != len(b) or len(t) < 4
            or not np.isfinite(t).all() or not np.isfinite(b).all() or (np.diff(t) <= 0).any()):
        raise ValueError('Finite embeddings and at least four strictly increasing timestamps required')
    dt = float(np.median(np.diff(t)))
    if np.max(np.diff(t)) > 1.5*dt: raise ValueError('Unsplit temporal gap')
    # Small irregularities resampled in seconds, never duration-normalized.
    uniform = np.arange(int(np.floor((t[-1]-t[0])/dt))+1)*dt+t[0]
    b = np.stack([np.interp(uniform, t, column) for column in b.T], axis=1)
    b -= b.mean(0); window = np.hanning(len(b))
    power = np.abs(np.fft.rfft(b*window[:, None], axis=0))**2
    power = power.sum(1)*dt/np.square(window).sum()
    power[1:-1 if len(b)%2 == 0 else None] *= 2
    freq = np.fft.rfftfreq(len(b), dt)
    return np.interp(grid, freq, power, left=0, right=0), float(np.mean(np.sum(b*b, axis=1)))


def analyze_frequency(runtime, references, artifact, output, batch_size=8, bins=128):
    saved = torch.load(artifact, map_location='cpu', weights_only=True)
    validate_artifact(saved, runtime.checkpoint_sha256)
    if saved['latent_dim'] != runtime.model.cfg.latent_dim or bins < 8: raise ValueError('Invalid dimension/grid')
    episodes = {e['episode_id']: e for e in episode_embeddings(runtime, references, batch_size) if e['success'] is True}
    if set(episodes) != {e['episode_id'] for e in saved['episodes']}: raise ValueError('Reference episode IDs changed')
    prepared, warnings, domains = [], [], set()
    for source in sorted(saved['episodes'], key=lambda e: e['episode_id']):
        e = episodes[source['episode_id']]; data = source_dataset(e['path']); root = data.roots[0]
        domains.add(getattr(root, 'attrs', {}).get('policy_timing', {}).get('domain', 'record_time_ns'))
        if data.episode_fingerprints[0] != source['source_fingerprint']: raise ValueError('Reference fingerprint changed')
        endpoints = {}
        for prefix, key, time_key in [('obs', 'record_ns', 'timestamp'), ('next_obs', 'next_record_ns', 'next_timestamp')]:
            for row, record in enumerate(root[key][:]):
                point = (float(root[time_key][row]), int(root['segment_id'][row]))
                if int(record) in endpoints and endpoints[int(record)] != point: raise ValueError('Ambiguous duplicate endpoint')
                endpoints[int(record)] = point
        points = [endpoints[r] for r in sorted(endpoints)]
        b = e['embeddings'].numpy()
        if b.shape != (len(points), saved['latent_dim']) or len(points) != source['states']: raise ValueError('Endpoint/embedding mismatch')
        segments = []
        for segment in sorted({p[1] for p in points}):
            indices = np.array([i for i, p in enumerate(points) if p[1] == segment])
            t = np.array([points[i][0] for i in indices]); values = b[indices]
            if len(t) > 1 and (np.diff(t) <= 0).any(): raise ValueError('Duplicate/backward policy timestamp')
            cuts = np.flatnonzero(np.diff(t) > 1.5*np.median(np.diff(t)))+1 if len(t) > 1 else []
            for chunk in np.split(np.arange(len(t)), cuts):
                if len(chunk) < 4: warnings.append(f'{e["episode_id"]}: short segment excluded'); continue
                segments.append((t[chunk], values[chunk]))
        if not segments: warnings.append(f'{e["episode_id"]}: no usable segment'); continue
        prepared.append((e['episode_id'], segments))
    if len(domains) != 1: raise ValueError('Cannot compare spectra from different policy clock domains')
    if not prepared: raise ValueError('No usable temporal episodes')
    nyquist = min(.5/np.median(np.diff(t)) for _, seg in prepared for t, _ in seg)
    grid = np.linspace(0, nyquist, bins); rows, signatures, ids, metadata = [], [], [], []
    for identity, segments in prepared:
        results = [spectrum(t, b, grid) for t, b in segments]
        duration = np.array([t[-1]-t[0] for t, _ in segments])
        power = np.average([r[0] for r in results], axis=0, weights=duration)
        energy = float(np.average([r[1] for r in results], weights=duration))
        if power.sum() <= 1e-20: warnings.append(f'{identity}: zero-energy episode excluded from clustering'); continue
        shape = power/power.sum(); ids.append(identity); signatures.append(shape)
        metadata.append(dict(episode_id=identity, duration_s=float(duration.sum()), samples=sum(len(t) for t, _ in segments), energy=energy))
        rows.extend(dict(episode_id=identity, frequency_hz=float(f), power=float(p), normalized_power=float(s)) for f, p, s in zip(grid, power, shape))
    if len(signatures) < 3: raise ValueError(f'At least three nonzero-energy episodes required; exclusions: {warnings}')
    signatures = np.asarray(signatures)
    distance = np.linalg.norm(np.sqrt(signatures)[:, None]-np.sqrt(signatures)[None, :], axis=2)/np.sqrt(2)
    trials, selected = candidates(distance); labels = selected['labels']
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'episode_spectra.csv', list(rows[0]), rows)
    write_csv(output/'frequency_clusters.csv', ['episode_id', 'cluster_id'], [dict(episode_id=e, cluster_id=c) for e, c in zip(ids, labels)])
    matrix_csv(output/'frequency_distance.csv', ids, distance)
    xy = pca(np.sqrt(signatures))
    write_csv(output/'frequency_coordinates.csv', ['episode_id', 'pc1', 'pc2'], [dict(episode_id=e, pc1=float(x), pc2=float(y)) for e, (x, y) in zip(ids, xy)])
    summary = dict(checkpoint_sha256=runtime.checkpoint_sha256, analysis_sha256=checkpoint_digest(artifact),
                   episode_ids=ids, parameters=dict(clock_domain=next(iter(domains)), bins=bins, units='policy-time Hz', nyquist_hz=nyquist,
                   method='duration-weighted segment Hann periodograms; mean removal; median-dt linear resampling',
                   distance='Hellinger of unit-sum PSD bins', selection='maximum silhouette; smaller k breaks ties'),
                   candidates=trials, selected_k=selected['k'], silhouette=selected['silhouette'], episodes=metadata, warnings=warnings)
    write_json(output/'frequency_summary.json', summary)
    heatmap(output/'frequency_heatmap.svg', distance, ids, 'Hellinger distance; input episode order')
    scatter(output/'frequency_pca.svg', xy, ids, labels, 'PCA of square-root spectral shapes')
    curves = [(e, s) for e, s in zip(ids, signatures)]
    curves += [(f'cluster {c} mean', signatures[np.array(labels) == c].mean(0)) for c in sorted(set(labels))]
    lines(output/'frequency_spectra.svg', grid, curves, 'Normalized spectra; x policy-time Hz; individual curves and cluster means')
    return summary

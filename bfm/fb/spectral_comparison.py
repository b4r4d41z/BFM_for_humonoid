"""ID-aligned partition comparison; shared episode-z PCA in every panel."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .latent_analysis import validate_artifact, diversity, write_csv, write_json
from .latents import pca
from .runtime import checkpoint_digest
from .spectral_utils import assignments, agreement, silhouette, heatmap, scatter


def compare(artifact, cluster_dir, frequency_dir, graph_dir, output):
    saved = torch.load(artifact, map_location='cpu', weights_only=True); z, _ = validate_artifact(saved)
    directories = [Path(p) for p in (cluster_dir, frequency_dir, graph_dir)]
    summaries = [json.loads((p/name).read_text()) for p, name in zip(directories, ['cluster_summary.json', 'frequency_summary.json', 'graph_summary.json'])]
    for summary in summaries:
        if (summary['checkpoint_sha256'] != saved['checkpoint_sha256'] or summary['analysis_sha256'] != checkpoint_digest(artifact)):
            raise ValueError('Analysis/checkpoint metadata mismatch')
    ids = [e['episode_id'] for e in saved['episodes']]
    labels = [assignments(list(csv.DictReader((p/name).open())), ids) for p, name in zip(directories, ['clusters.csv', 'frequency_clusters.csv', 'graph_clusters.csv'])]
    names = ['cosine', 'frequency', 'graph']; rows = []; details = {}
    for i in range(3):
        for j in range(i+1, 3):
            result = agreement(labels[i], labels[j]); key = names[i]+'_'+names[j]
            details[key] = result
            rows.append(dict(method_a=names[i], method_b=names[j], ari=result['ari'], nmi=result['nmi']))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'cluster_agreement.csv', list(rows[0]), rows)
    xy = pca(z); panels = []
    for name, lab in zip(names, labels):
        temp = output/f'{name}_shared_pca.svg'; scatter(temp, xy, ids, lab, name+' (shared episode-z PCA)')
        text = temp.read_text(); panels.append(text[text.index('>')+1:text.rindex('</svg>')])
    (output/'cluster_comparison.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="3000" height="760">'+''.join(f'<g transform="translate({i*1000},0)">{body}</g>' for i, body in enumerate(panels))+'</svg>')
    for key, result in details.items(): heatmap(output/f'{key}_contingency.svg', result['contingency'], range(len(result['contingency'])), key+' contingency')
    a = np.eye(3)
    for row in rows:
        i, j = names.index(row['method_a']), names.index(row['method_b']); a[i,j] = a[j,i] = row['ari']
    heatmap(output/'cluster_agreement.svg', a, names, 'ARI (negative values shown at zero intensity; exact values in CSV)')
    distance = np.clip(1-diversity(z, z[:1])[0], 0, 2)
    confounds = {}
    for metric in ('duration_s', 'samples', 'energy'):
        values = {e['episode_id']: e[metric] for e in summaries[1]['episodes']}
        confounds[metric] = {str(c): [values[e] for e, lab in zip(ids, labels[1]) if lab == c] for c in sorted(set(labels[1]))}
    confound_eta_squared = {}
    for metric, groups in confounds.items():
        all_values = np.array([v for values in groups.values() for v in values], dtype=float)
        total = np.sum((all_values-all_values.mean())**2)
        between = sum(len(values)*(np.mean(values)-all_values.mean())**2 for values in groups.values())
        confound_eta_squared[metric] = float(between/total) if total else None
    summary = dict(checkpoint_sha256=saved['checkpoint_sha256'], analysis_sha256=checkpoint_digest(artifact), episode_ids=ids,
                   methods={name: dict(cluster_sizes={str(c): lab.count(c) for c in sorted(set(lab))},
                                      cosine_silhouette=silhouette(distance, lab), source_summary=source)
                            for name, lab, source in zip(names, labels, summaries)},
                   agreement=details, frequency_confounds=confounds, confound_eta_squared=confound_eta_squared,
                   interpretation='Selected counts and seed/neighborhood sensitivity are recorded per method. '
                   'Frequency shape removes amplitude scale but duration/resolution and sample count can confound it. '
                   'Confound eta-squared is descriptive and not a significance test. Spectral-processing sensitivity requires reruns with --bins. '
                   'Agreement measures partition consistency, not object identity, motor primitives or control quality.')
    write_json(output/'comparison_summary.json', summary)
    write_csv(output/'shared_coordinates.csv', ['episode_id', 'pc1', 'pc2', *names],
              [dict(episode_id=e, pc1=float(x), pc2=float(y), **{name: lab[i] for name, lab in zip(names, labels)}) for i, (e, (x,y)) in enumerate(zip(ids, xy))])
    return summary

"""Small NumPy-only offline analysis metrics and standalone SVG figures."""
import html
from pathlib import Path
import numpy as np
from .latent_analysis import write_csv, write_json


def matrix_csv(path, ids, matrix):
    write_csv(path, ['episode_id', *ids], [dict(episode_id=e, **dict(zip(ids, row))) for e, row in zip(ids, matrix)])


def silhouette(distance, labels):
    labels = np.asarray(labels); groups = np.unique(labels)
    if not 1 < len(groups) < len(labels): return None
    values = []
    for i, label in enumerate(labels):
        own = np.flatnonzero(labels == label); own = own[own != i]
        if not len(own): values.append(0.); continue
        a = distance[i, own].mean()
        b = min(distance[i, labels == other].mean() for other in groups if other != label)
        values.append((b-a)/max(a, b) if max(a, b) else 0.)
    return float(np.mean(values))


def linkage(distance, k):
    groups = [(i,) for i in range(len(distance))]
    while len(groups) > k:
        _, a, b = min((float(distance[np.ix_(x, y)].mean()), a, b)
                      for a, x in enumerate(groups) for b, y in enumerate(groups) if a < b)
        merged = tuple(sorted(groups[a]+groups[b]))
        groups = sorted([g for i, g in enumerate(groups) if i not in (a, b)]+[merged])
    labels = np.empty(len(distance), dtype=int)
    for i, g in enumerate(groups): labels[list(g)] = i
    return labels


def candidates(distance, partition=None):
    result = []
    for k in range(2, min(8, len(distance)-1)+1):
        labels = linkage(distance, k) if partition is None else partition(k)
        result.append(dict(k=k, labels=labels.tolist(), silhouette=silhouette(distance, labels),
                           sizes=np.bincount(labels, minlength=k).tolist()))
    if not result: raise ValueError('At least three usable episodes required')
    return result, max(result, key=lambda r: (r['silhouette'], -r['k']))


def agreement(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 1 or not len(a): raise ValueError('Invalid assignments')
    _, ai = np.unique(a, return_inverse=True); _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max()+1, bi.max()+1), dtype=int)
    np.add.at(table, (ai, bi), 1)
    choose = lambda x: np.sum(x*(x-1)/2)
    total = choose(np.array([len(a)])); index = choose(table)
    x, y = choose(table.sum(1)), choose(table.sum(0))
    expected = x*y/total if total else 0.; maximum = (x+y)/2
    ari = (index-expected)/(maximum-expected) if maximum != expected else 1.
    p = table/len(a); px, py = p.sum(1), p.sum(0)
    mask = p > 0; product = px[:, None]*py[None]
    mi = np.sum(p[mask]*np.log(p[mask]/product[mask]))
    entropy = lambda q: -np.sum(q[q > 0]*np.log(q[q > 0]))
    h = (entropy(px)+entropy(py))/2
    return dict(ari=float(ari), nmi=float(mi/h) if h else 1., contingency=table.tolist())


def assignments(rows, ids):
    mapping = {r['episode_id']: int(r['cluster_id']) for r in rows}
    if len(mapping) != len(rows) or set(mapping) != set(ids): raise ValueError('Missing/duplicate episode IDs')
    return [mapping[e] for e in ids]


def svg(path, body):
    Path(path).write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="760">'
                         '<rect width="100%" height="100%" fill="white"/>'+body+'</svg>')


def scatter(path, xy, ids, labels, title):
    xy = np.asarray(xy); span = np.ptp(xy, axis=0)
    points = 60+(xy-xy.min(0))/np.where(span > 0, span, 1)*620
    body = f'<text x="20" y="25">{html.escape(title)}</text>'
    for (x, y), identity, label in zip(points, ids, labels):
        body += f'<g><title>{html.escape(identity)}</title><circle cx="{x}" cy="{720-y}" r="5" fill="hsl({int(label)*137.5%360},65%,40%)"/><text x="{x+7}" y="{720-y}" font-size="10">{html.escape(identity)} C{label}</text></g>'
    svg(path, body)


def heatmap(path, matrix, ids, title):
    a = np.asarray(matrix); scale = max(float(np.max(np.abs(a))), 1e-15)
    h, w = a.shape; body = f'<text x="20" y="25">{html.escape(title)}</text>'
    for i in range(h):
        for j in range(w):
            color = int(255*(1-max(0, a[i, j])/scale))
            body += f'<rect x="{40+j*650/w}" y="{50+i*650/h}" width="{650/w+0.1}" height="{650/h+0.1}" fill="rgb({color},{color},255)"><title>{html.escape(str(ids[i]))} / {j}: {a[i,j]:.5g}</title></rect>'
    svg(path, body)


def lines(path, x, series, title):
    x = np.asarray(x); all_y = np.asarray([y for _, y in series]); lo, hi = all_y.min(), all_y.max()
    xp = 50+(x-x.min())/max(float(np.ptp(x)), 1e-15)*850
    body = f'<text x="20" y="25">{html.escape(title)}</text>'
    for i, (name, y) in enumerate(series):
        yp = 700-(np.asarray(y)-lo)/max(float(hi-lo), 1e-15)*630
        points = ' '.join(f'{a},{b}' for a, b in zip(xp, yp))
        body += f'<polyline points="{points}" fill="none" stroke="hsl({i*137.5%360},65%,40%)"><title>{html.escape(name)}</title></polyline>'
    body += f'<text x="30" y="740">x: {x.min():.4g} to {x.max():.4g}; y: {lo:.4g} to {hi:.4g}; hover identifies curves</text>'
    svg(path, body)

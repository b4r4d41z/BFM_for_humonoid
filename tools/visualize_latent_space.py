"""Plot real B(s), episode means, task/search z, and training-distribution z."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from bfm.fb.runtime import VisualRuntime
    from bfm.fb.latents import candidates, episode_embeddings, visualize
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--dataset')
    p.add_argument('--embeddings',help='Cached trained B(s) from build_task_z.py')
    p.add_argument('--task-z',nargs='*',default=[])
    p.add_argument('--random-count',type=int,default=32)
    p.add_argument('--output',required=True,help='Output stem for CSV and SVG')
    p.add_argument('--method',choices=['pca','umap'],default='pca')
    a=p.parse_args();runtime=VisualRuntime(a.checkpoint)
    points=list(candidates(runtime,a.random_count)['vectors']);labels=[{'source':'random'} for _ in points]
    episodes=[]
    if a.embeddings:
        saved=torch.load(a.embeddings,weights_only=True,map_location='cpu')
        if saved['checkpoint_sha256']!=runtime.checkpoint_sha256:raise ValueError('Embedding checkpoint mismatch')
        episodes=saved['episodes']
    elif a.dataset:episodes=episode_embeddings(runtime,a.dataset)
    for e in episodes:
        metadata={'episode':e['episode_id'],'success':e['success']}
        points.extend(e['embeddings']);labels.extend([{'source':'B(s)',**metadata} for _ in e['embeddings']])
        points.append(e['embeddings'].mean(0));labels.append({'source':'episode_mean',**metadata})
    for path in a.task_z:
        points.append(runtime.load_task(path)[0].cpu());labels.append({'source':'task_z','path':path})
    visualize(torch.stack(points).numpy(),labels,a.output,a.method)


if __name__=='__main__':main()

"""Infer an episode-balanced task z from full successful reference trajectories."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from bfm.fb.runtime import VisualRuntime
    from bfm.fb.latents import episode_embeddings, save_task
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--references',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--embeddings-out',help='Optional reusable B(s) diagnostic artifact')
    a=p.parse_args()
    runtime=VisualRuntime(a.checkpoint,a.device)
    episodes=episode_embeddings(runtime,a.references,a.batch_size)
    save_task(runtime,episodes,a.output)
    if a.embeddings_out:
        torch.save({'checkpoint_sha256':runtime.checkpoint_sha256,'episodes':episodes},a.embeddings_out)
    print(a.output)


if __name__=='__main__':main()

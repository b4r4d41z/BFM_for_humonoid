"""Experimental external-rollout search: propose candidates, then rank scores."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from bfm.fb.runtime import VisualRuntime
    from bfm.fb.latents import candidates, rank_candidates
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=32)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--candidates',help='Previously proposed .pt artifact')
    p.add_argument('--scores',help='External rollout JSON object: candidate ID -> finite score')
    a=p.parse_args();runtime=VisualRuntime(a.checkpoint)
    if bool(a.candidates)!=bool(a.scores):p.error('Ranking requires both --candidates and --scores')
    if a.candidates:
        proposal=torch.load(a.candidates,weights_only=True,map_location='cpu')
        scores=json.loads(Path(a.scores).read_text())
        if set(scores)!=set(proposal['ids']):raise ValueError('Scores must cover exactly all candidate IDs')
        result=rank_candidates(runtime,proposal,lambda identity,z:scores[identity])
    else:result=candidates(runtime,a.count,a.seed)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,a.output)


if __name__=='__main__':main()

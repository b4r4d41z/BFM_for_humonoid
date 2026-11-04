"""Analyze projected per-successful-episode B means against a bound global task."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from bfm.fb.runtime import VisualRuntime
    from bfm.fb.latent_analysis import analyze
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--references',required=True)
    p.add_argument('--task-z',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    print(json.dumps(analyze(VisualRuntime(a.checkpoint,a.device),a.references,a.task_z,a.output_dir,a.batch_size),indent=2))


if __name__=='__main__':main()

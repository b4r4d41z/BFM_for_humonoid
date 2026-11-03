"""Compare deterministic actor sensitivity using real retained Zarr observations."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from bfm.fb.runtime import VisualRuntime
    from bfm.fb.latent_analysis import compare_actions
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--analysis',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--episode-id');p.add_argument('--timesteps',nargs='+',type=int)
    p.add_argument('--z-ids',nargs='+')
    a=p.parse_args()
    print(json.dumps(compare_actions(VisualRuntime(a.checkpoint,a.device),a.analysis,a.output_dir,
                                    a.episode_id,a.timesteps,a.z_ids),indent=2))


if __name__=='__main__':main()

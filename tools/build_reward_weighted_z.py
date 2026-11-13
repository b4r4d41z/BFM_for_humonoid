"""Build independent RewardWeightedZ; never train, convert, or change the actor."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from bfm.fb.runtime import VisualRuntime
    from bfm.fb.reward_weighted_z import build, compare_tasks
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--references',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--beta',type=float,default=1.,help='Nonnegative inverse reward scale; 0 gives uniform within-episode weights')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--reward',choices=['approach','goal_proximity'],help='Positional proxy, not measured physical success')
    source.add_argument('--annotations',type=Path)
    source.add_argument('--annotation-template',action='store_true',help='Write aligned JSON with null rewards; does not infer Z')
    p.add_argument('--hand',choices=['left','right','either'],default='either')
    p.add_argument('--scale-m',type=float,default=.1,help='Distance decay length in meters for positional proxy')
    p.add_argument('--baseline-task',type=Path,help='Optional EpisodeBalancedZ for bound latent/action comparison')
    p.add_argument('--cluster-tasks',type=Path,nargs='+',default=[])
    p.add_argument('--comparison-dir',type=Path)
    p.add_argument('--timesteps',type=int,nargs='+',help='Retained obs rows for comparison (default first/middle/last)')
    a = p.parse_args()
    if (a.cluster_tasks or a.comparison_dir or a.timesteps) and not a.baseline_task:
        p.error('Comparison options require --baseline-task')
    if a.annotation_template and a.baseline_task:p.error('Annotation template mode does not produce a task to compare')
    runtime = VisualRuntime(a.checkpoint,a.device)
    result = build(runtime,a.references,a.output,beta=a.beta,component=a.reward,annotations=a.annotations,
                   hand=a.hand,scale_m=a.scale_m,batch_size=a.batch_size,template=a.annotation_template)
    if a.annotation_template:
        print(f'Annotation template: {a.output}; replace nulls with measured rewards before inference')
        return
    print(json.dumps(result['reward_diagnostics'],indent=2))
    if a.baseline_task:
        directory = a.comparison_dir or a.output.with_name(a.output.stem+'_comparison')
        print(json.dumps(compare_tasks(runtime,a.baseline_task,a.output,a.cluster_tasks,directory,timesteps=a.timesteps),indent=2))
    print(f'RewardWeightedZ: {a.output}')


if __name__=='__main__':main()

"""Build post-training, label-blind cluster tasks from a bound episode analysis."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from bfm.fb.cluster_analysis import build_clusters
    from bfm.fb.runtime import VisualRuntime
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--references', required=True)
    p.add_argument('--analysis', required=True)
    p.add_argument('--clusters', type=int, default=3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    print(json.dumps(build_clusters(VisualRuntime(a.checkpoint, a.device), a.references,
                                   a.analysis, a.output_dir, a.clusters, a.batch_size), indent=2))


if __name__ == '__main__':
    main()

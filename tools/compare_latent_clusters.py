"""Offline spectral latent diagnostics; inputs are read-only."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    from bfm.fb.spectral_comparison import compare
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--cluster-dir', required=True); p.add_argument('--frequency-dir', required=True)
    p.add_argument('--graph-dir', required=True)
    a=p.parse_args()
    print(json.dumps(compare(a.analysis,a.cluster_dir,a.frequency_dir,a.graph_dir,a.output_dir),indent=2))

if __name__ == '__main__': main()

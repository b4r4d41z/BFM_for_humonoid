"""Offline spectral latent diagnostics; inputs are read-only."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    from bfm.fb.graph_spectral_analysis import analyze_graph
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--neighbors', nargs='+', type=int, default=[3,5,10])
    p.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
    a=p.parse_args()
    print(json.dumps(analyze_graph(a.analysis,a.output_dir,a.neighbors,a.seeds),indent=2))

if __name__ == '__main__': main()

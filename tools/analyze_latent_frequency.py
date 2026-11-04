"""Offline spectral latent diagnostics; inputs are read-only."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    from bfm.fb.frequency_analysis import analyze_frequency
    from bfm.fb.runtime import VisualRuntime
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--checkpoint', required=True); p.add_argument('--references', required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--bins', type=int, default=128)
    a=p.parse_args()
    print(json.dumps(analyze_frequency(VisualRuntime(a.checkpoint,a.device),a.references,a.analysis,a.output_dir,a.batch_size,a.bins),indent=2))

if __name__ == '__main__': main()

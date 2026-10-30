"""Convert raw recordings independently; never initialize a training model."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import yaml
    from bfm.data.recordings import prepare
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recordings', type=Path)
    args = parser.parse_args()
    policy = yaml.safe_load((ROOT/'configs/data_conversion/sharpa_v3.yaml').read_text())
    print(prepare(args.recordings, ROOT/'data/zarr', policy))


if __name__ == '__main__':
    main()

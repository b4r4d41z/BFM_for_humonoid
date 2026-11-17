"""Convert raw recordings independently; never initialize a training model."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import yaml
    import json
    from bfm.data.recordings import prepare, NoSuccessfulRecordings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recordings', type=Path)
    parser.add_argument('--contract',choices=['v3','v4','decoupled-v4'],default='decoupled-v4')
    args = parser.parse_args()
    policy = yaml.safe_load((ROOT/'configs/data_conversion/sharpa_v3.yaml').read_text())
    destination={'v3':'data/zarr','v4':'data/zarr_15hz','decoupled-v4':'data/zarr_decoupled_v4'}[args.contract]
    try:
        manifest=prepare(args.recordings,ROOT/destination,policy,expected_version=3 if args.contract=='v3' else 4,
                         expected_architecture='decoupled_teleoperation_v1' if args.contract=='decoupled-v4' else 'executor_v1' if args.contract=='v4' else None)
    except NoSuccessfulRecordings:return 1
    print(manifest)
    return int(json.loads((manifest.parent/'conversion_report.json').read_text())['summary']['failed']>0)


if __name__ == '__main__':
    raise SystemExit(main())

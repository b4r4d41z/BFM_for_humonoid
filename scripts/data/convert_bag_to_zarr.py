#!/usr/bin/env python3
"""Convert one explicitly mapped ROS1 bag episode to canonical Zarr."""
from __future__ import annotations

import argparse
from pathlib import Path

from bc.data.canonical_zarr import SCHEMA_NAME, SCHEMA_VERSION, write_canonical_zarr
from bc.data.ros_episode import build_canonical_episode
from bc.data.rosbag_reader import load_topic_config, read_rosbag
from bc.data.schema import ARM_JOINT_NAMES, HAND_VALUE_NAMES, get_default_contract_metadata
from bc.temporal import build_temporal_contract_metadata, measure_dataset_hz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--topics", required=True, type=Path, help="JSON with streams and optional camera role mappings")
    parser.add_argument("--max-skew-s", default=0.05, type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_topic_config(args.topics)
    episode = build_canonical_episode(read_rosbag(args.input, config), max_skew_s=args.max_skew_s)
    temporal = build_temporal_contract_metadata(measure_dataset_hz(episode["timestamp"]))
    temporal.update({"transition_clock": "arm_state", "alignment": "nearest_neighbor", "max_skew_s": args.max_skew_s, "resampled": False})
    contract = get_default_contract_metadata()
    metadata = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "obs_dim": 26, "action_dim": 26, "arm_dim": 14, "hand_dim": 12,
        "arm_joint_names": ARM_JOINT_NAMES, "hand_value_names": HAND_VALUE_NAMES,
        "source_bag": args.input.name, "instruction": config.get("instruction", ""),
        "camera_mapping": {role: spec["topic"] for role, spec in config.get("cameras", {}).items()},
        "temporal": temporal,
        "units": config.get("units", {}),
        "state_layout": contract["state_layout"], "action_layout": contract["action_layout"],
        "action_type": contract["action_type"],
    }
    summary = write_canonical_zarr(args.output, episode, metadata, overwrite=args.overwrite)
    print(f"Source: {args.input}")
    print(f"Transitions: {summary['transitions']}")
    print("State dim: 26\nAction dim: 26")
    print(f"Measured dataset Hz: {temporal['actual_dataset_hz']:.6g}")
    print(f"Cameras: {', '.join(summary['cameras']) or 'none'}")
    print("Zarr validation: PASS")


if __name__ == "__main__":
    main()

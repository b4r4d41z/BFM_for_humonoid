#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bc.data.canonical_zarr import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    write_canonical_zarr,
)
from bc.data.schema import (
    ARM_JOINT_NAMES,
    HAND_VALUE_NAMES,
    get_default_contract_metadata,
)
from bc.temporal import (
    build_temporal_contract_metadata,
    measure_dataset_hz,
)


# ============================================================
# Basic timestamped sample
# ============================================================

@dataclass
class Sample:
    timestamp: float
    value: Any


# ============================================================
# ROS helpers
# ============================================================ 

def message_timestamp(msg: Any, bag_timestamp: Any) -> float:
    """Prefer message header timestamp, otherwise use bag timestamp."""

    header = getattr(msg, "header", None)

    if header is not None and hasattr(header, "stamp"):
        stamp = header.stamp

        try:
            value = float(stamp.to_sec())
            if value > 0:
                return value
        except Exception:
            pass

    return float(bag_timestamp.to_sec())


def get_field(obj: Any, path: str) -> Any:
    value = obj

    for part in path.split("."):
        value = getattr(value, part)

    return value


def reorder_joint_state(msg: Any, values: np.ndarray) -> np.ndarray:
    """
    If the ROS message provides joint names, reorder values into the
    canonical BFM order.

    If names are unavailable, preserve the recorded order.
    """

    names = list(getattr(msg, "name", []))

    if not names:
        return values

    if not all(name in names for name in ARM_JOINT_NAMES):
        return values

    index = {name: i for i, name in enumerate(names)}

    return np.asarray(
        [values[index[name]] for name in ARM_JOINT_NAMES],
        dtype=np.float32,
    )


def read_arm_vector(
    msg: Any,
    field: str,
    unit: str,
) -> np.ndarray:

    values = np.asarray(
        get_field(msg, field),
        dtype=np.float32,
    ).reshape(-1)

    values = reorder_joint_state(msg, values)

    if values.shape != (14,):
        raise ValueError(
            f"Expected 14 arm values, got {values.shape}"
        )

    if unit == "deg":
        values = np.deg2rad(values)

    if not np.all(np.isfinite(values)):
        raise ValueError("Arm vector contains non-finite values")

    return values.astype(np.float32)


def read_hand_vector(msg: Any) -> np.ndarray:
    """
    Supports the KUAVO / ExoSuit hand message:
        left_hand_position[6]
        right_hand_position[6]

    Also accepts a generic 12-value .data field.
    """

    if (
        hasattr(msg, "left_hand_position")
        and hasattr(msg, "right_hand_position")
    ):
        left = np.asarray(
            msg.left_hand_position,
            dtype=np.float32,
        ).reshape(-1)

        right = np.asarray(
            msg.right_hand_position,
            dtype=np.float32,
        ).reshape(-1)

        values = np.concatenate([left, right])

    elif hasattr(msg, "data"):
        values = np.asarray(
            msg.data,
            dtype=np.float32,
        ).reshape(-1)

    else:
        raise ValueError(
            "Unsupported hand message: expected "
            "left_hand_position/right_hand_position or data"
        )

    if values.shape != (12,):
        raise ValueError(
            f"Expected 12 hand values, got {values.shape}"
        )

    return values


def decode_compressed_image(msg: Any) -> np.ndarray:

    from PIL import Image

    with Image.open(io.BytesIO(bytes(msg.data))) as image:
        return np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )


# ============================================================
# Stream helpers
# ============================================================

def prepare_stream(
    samples: list[Sample],
    name: str,
) -> list[Sample]:

    if not samples:
        raise ValueError(f"No samples found for {name}")

    samples = sorted(samples, key=lambda x: x.timestamp)

    # Remove duplicate timestamps.
    result: list[Sample] = []

    for sample in samples:
        if result and sample.timestamp == result[-1].timestamp:
            result[-1] = sample
        else:
            result.append(sample)

    timestamps = np.asarray(
        [x.timestamp for x in result],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(timestamps)):
        raise ValueError(f"{name}: invalid timestamps")

    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(
            f"{name}: timestamps are not strictly increasing"
        )

    return result


def nearest(
    samples: list[Sample],
    timestamp: float,
    max_skew: float,
    name: str,
) -> Any:

    times = np.asarray(
        [x.timestamp for x in samples],
        dtype=np.float64,
    )

    pos = int(np.searchsorted(times, timestamp))

    candidates = [
        i
        for i in (pos - 1, pos)
        if 0 <= i < len(samples)
    ]

    if not candidates:
        raise ValueError(f"{name}: no samples")

    index = min(
        candidates,
        key=lambda i: abs(times[i] - timestamp),
    )

    skew = abs(times[index] - timestamp)

    if skew > max_skew:
        raise ValueError(
            f"{name}: nearest sample is {skew:.4f}s away "
            f"(limit={max_skew:.4f}s)"
        )

    return samples[index].value


# ============================================================
# Bag reader
# ============================================================

def read_bag(args):
    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError(
            "ROS1 rosbag Python package is required"
        ) from exc

    arm_state: list[Sample] = []
    arm_command: list[Sample] = []

    hand_state: list[Sample] = []
    hand_command: list[Sample] = []

    cameras: dict[str, list[Sample]] = {}

    camera_topics = {}

    if args.head_topic:
        camera_topics[args.head_topic] = "head"

    if args.left_wrist_topic:
        camera_topics[args.left_wrist_topic] = "left_wrist"

    if args.right_wrist_topic:
        camera_topics[args.right_wrist_topic] = "right_wrist"

    for name in camera_topics.values():
        cameras[name] = []

    topics = {
        args.arm_state_topic,
        args.arm_command_topic,
        args.hand_command_topic,
        *camera_topics.keys(),
    }

    if args.hand_state_topic:
        topics.add(args.hand_state_topic)

    with rosbag.Bag(str(args.input), "r") as bag:

        for topic, msg, bag_time in bag.read_messages(
            topics=list(topics)
        ):

            timestamp = message_timestamp(msg, bag_time)

            if topic == args.arm_state_topic:
                value = read_arm_vector(
                    msg,
                    args.arm_state_field,
                    args.arm_state_unit,
                )

                arm_state.append(
                    Sample(timestamp, value)
                )

            if topic == args.arm_command_topic:
                value = read_arm_vector(
                    msg,
                    args.arm_command_field,
                    args.arm_command_unit,
                )

                arm_command.append(
                    Sample(timestamp, value)
                )

            if topic == args.hand_command_topic:
                value = read_hand_vector(msg)

                hand_command.append(
                    Sample(timestamp, value)
                )

                # If there is no independent hand feedback topic,
                # the latest commanded target is also used as the
                # observable hand state.
                if args.hand_state_topic is None:
                    hand_state.append(
                        Sample(timestamp, value.copy())
                    )

            if (
                args.hand_state_topic
                and topic == args.hand_state_topic
            ):
                hand_state.append(
                    Sample(
                        timestamp,
                        read_hand_vector(msg),
                    )
                )

            if topic in camera_topics:
                role = camera_topics[topic]

                cameras[role].append(
                    Sample(
                        timestamp,
                        decode_compressed_image(msg),
                    )
                )

    return {
        "arm_state": prepare_stream(
            arm_state,
            "arm_state",
        ),
        "arm_command": prepare_stream(
            arm_command,
            "arm_command",
        ),
        "hand_state": prepare_stream(
            hand_state,
            "hand_state",
        ),
        "hand_command": prepare_stream(
            hand_command,
            "hand_command",
        ),
        "cameras": {
            name: prepare_stream(stream, f"camera:{name}")
            for name, stream in cameras.items()
        },
    }


# ============================================================
# Synchronization
# ============================================================

def build_episode(streams, max_skew: float):

    arm_state = streams["arm_state"]

    # Arm state is the canonical transition clock.
    timestamps = np.asarray(
        [x.timestamp for x in arm_state],
        dtype=np.float64,
    )

    if len(timestamps) < 2:
        raise ValueError(
            "At least two arm-state samples are required"
        )

    states = []

    for sample in arm_state:

        hand = nearest(
            streams["hand_state"],
            sample.timestamp,
            max_skew,
            "hand_state",
        )

        state = np.concatenate(
            [
                sample.value,
                hand,
            ]
        ).astype(np.float32)

        states.append(state)

    states = np.stack(states)

    transition_times = timestamps[:-1]

    actions = []
    images: dict[str, list[np.ndarray]] = {
        name: []
        for name in streams["cameras"]
    }

    for timestamp in transition_times:

        arm_action = nearest(
            streams["arm_command"],
            timestamp,
            max_skew,
            "arm_command",
        )

        hand_action = nearest(
            streams["hand_command"],
            timestamp,
            max_skew,
            "hand_command",
        )

        action = np.concatenate(
            [
                arm_action,
                hand_action,
            ]
        ).astype(np.float32)

        actions.append(action)

        for name, camera_stream in streams["cameras"].items():

            frame = nearest(
                camera_stream,
                timestamp,
                max_skew,
                f"camera:{name}",
            )

            images[name].append(frame)

    actions = np.stack(actions)

    T = len(actions)

    done = np.zeros(T, dtype=np.bool_)
    done[-1] = True

    return {
        "obs/state": states[:-1],
        "action": actions,
        "next_obs/state": states[1:],
        "done": done,
        "timestamp": transition_times,
        "images": {
            name: np.stack(frames)
            for name, frames in images.items()
        },
    }


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Convert a ROS1 bag directly to canonical BFM Zarr"
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    # Observation source
    parser.add_argument(
        "--arm-state-topic",
        required=True,
    )

    parser.add_argument(
        "--arm-state-field",
        required=True,
    )

    parser.add_argument(
        "--arm-state-unit",
        choices=["rad", "deg"],
        default="rad",
    )

    # Recorded arm command
    parser.add_argument(
        "--arm-command-topic",
        default="/kuavo_arm_traj",
    )

    parser.add_argument(
        "--arm-command-field",
        default="position",
    )

    parser.add_argument(
        "--arm-command-unit",
        choices=["rad", "deg"],
        default="deg",
    )

    # Hands
    parser.add_argument(
        "--hand-command-topic",
        default="/control_robot_hand_position",
    )

    parser.add_argument(
        "--hand-state-topic",
        default=None,
        help=(
            "Optional real hand feedback topic. "
            "If omitted, commanded hand position is used."
        ),
    )

    # Cameras
    parser.add_argument(
        "--head-topic",
        default=None,
    )

    parser.add_argument(
        "--left-wrist-topic",
        default=None,
    )

    parser.add_argument(
        "--right-wrist-topic",
        default=None,
    )

    parser.add_argument(
        "--instruction",
        default="",
    )

    parser.add_argument(
        "--max-skew-s",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    print(f"Reading: {args.input}")

    streams = read_bag(args)

    episode = build_episode(
        streams,
        max_skew=args.max_skew_s,
    )

    temporal = build_temporal_contract_metadata(
        measure_dataset_hz(
            episode["timestamp"]
        )
    )

    temporal.update(
        {
            "transition_clock": "arm_state",
            "alignment": "nearest_neighbor",
            "max_skew_s": args.max_skew_s,
            "resampled": False,
        }
    )

    contract = get_default_contract_metadata()

    camera_mapping = {}

    if args.head_topic:
        camera_mapping["head"] = args.head_topic

    if args.left_wrist_topic:
        camera_mapping["left_wrist"] = (
            args.left_wrist_topic
        )

    if args.right_wrist_topic:
        camera_mapping["right_wrist"] = (
            args.right_wrist_topic
        )

    metadata = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,

        "obs_dim": 26,
        "action_dim": 26,
        "arm_dim": 14,
        "hand_dim": 12,

        "arm_joint_names": ARM_JOINT_NAMES,
        "hand_value_names": HAND_VALUE_NAMES,

        "source_bag": args.input.name,
        "instruction": args.instruction,

        "camera_mapping": camera_mapping,

        "temporal": temporal,

        "units": {
            "arm_state": "rad",
            "arm_action": "rad",
            "hand": "raw_robot_hand_values",
            "images": "uint8_rgb",
        },

        "state_layout": contract["state_layout"],
        "action_layout": contract["action_layout"],
        "action_type": contract["action_type"],

        "hand_state_source": (
            args.hand_state_topic
            if args.hand_state_topic
            else "command_hold"
        ),

        "topics": {
            "arm_state": args.arm_state_topic,
            "arm_command": args.arm_command_topic,
            "hand_command": args.hand_command_topic,
        },
    }

    summary = write_canonical_zarr(
        args.output,
        episode,
        metadata,
        overwrite=args.overwrite,
    )

    print()
    print("Conversion complete")
    print("-------------------")
    print(f"Transitions : {summary['transitions']}")
    print("State       : 26D")
    print("Action      : 26D")
    print(
        f"Dataset Hz  : "
        f"{temporal['actual_dataset_hz']:.3f}"
    )
    print(
        f"Cameras     : "
        f"{', '.join(summary['cameras']) or 'none'}"
    )
    print(f"Output      : {args.output}")
    print("Validation  : PASS")


if __name__ == "__main__":
    main()

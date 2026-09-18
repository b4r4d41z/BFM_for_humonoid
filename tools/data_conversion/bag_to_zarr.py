#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bc.data.canonical_zarr import SCHEMA_NAME, SCHEMA_VERSION, write_canonical_zarr
from bc.data.schema import ARM_JOINT_NAMES, HAND_VALUE_NAMES, get_default_contract_metadata
from bc.temporal import build_temporal_contract_metadata, measure_dataset_hz


OUTPUT_DIR = Path("/home/ivan/Desktop/ivan/zarr")

ARM_TOPIC = "/kuavo_arm_traj"
HAND_TOPIC = "/control_robot_hand_position"
HEAD_CAMERA_TOPIC = "/head_camera/color/image_raw/compressed"
LEFT_WRIST_CAMERA_TOPIC = "/left_hand_camera/left_hand_camera_node/image_raw/compressed"
RIGHT_WRIST_CAMERA_TOPIC = "/right_hand_camera/right_hand_camera_node/image_raw/compressed"

MAX_HAND_SKEW_S = 0.05
MAX_CAMERA_SKEW_S = 0.10

ARM_INPUT_IS_DEGREES = True

COMPRESSED_IMAGE_WIDTH = 320
COMPRESSED_IMAGE_HEIGHT = 240


@dataclass
class Sample:
    timestamp: float
    value: Any


def print_progress(current: int, total: int, label: str) -> None:
    width = 30
    ratio = current / total if total else 1.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)

    print(
        f"\r{label:<20} [{bar}] {current}/{total} ({ratio * 100:5.1f}%)",
        end="",
        flush=True,
    )

    if current >= total:
        print()


def get_timestamp(msg: Any, bag_time: Any) -> float:
    header = getattr(msg, "header", None)

    if header is not None:
        stamp = getattr(header, "stamp", None)

        if stamp is not None:
            try:
                timestamp = float(stamp.to_sec())

                if timestamp > 0:
                    return timestamp
            except Exception:
                pass

    return float(bag_time.to_sec())


def read_arm(msg: Any) -> np.ndarray:
    names = list(msg.name)
    positions = np.asarray(msg.position, dtype=np.float32).reshape(-1)

    if len(names) != len(positions):
        raise ValueError(
            f"JointState name/position mismatch: {len(names)} vs {len(positions)}"
        )

    joint_index = {name: i for i, name in enumerate(names)}
    missing = [name for name in ARM_JOINT_NAMES if name not in joint_index]

    if missing:
        raise ValueError(f"Missing arm joints: {missing}")

    positions = np.asarray(
        [positions[joint_index[name]] for name in ARM_JOINT_NAMES],
        dtype=np.float32,
    )

    if positions.shape != (14,):
        raise ValueError(f"Expected arm shape (14,), got {positions.shape}")

    if ARM_INPUT_IS_DEGREES:
        positions = np.deg2rad(positions)

    if not np.all(np.isfinite(positions)):
        raise ValueError("Arm message contains non-finite values")

    return positions.astype(np.float32)


def read_hand(msg: Any) -> np.ndarray:
    def to_array(values):
        if isinstance(values, (bytes, bytearray, memoryview)):
            return np.frombuffer(values, dtype=np.uint8).astype(np.float32)

        return np.asarray(values, dtype=np.float32).reshape(-1)

    left = to_array(msg.left_hand_position)
    right = to_array(msg.right_hand_position)

    if left.shape != (6,) or right.shape != (6,):
        raise ValueError(
            f"Expected hand shapes (6,) + (6,), got {left.shape} + {right.shape}"
        )

    return np.concatenate([left, right]).astype(np.float32)


def read_image(msg: Any, compress: bool) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to decode camera images") from exc

    with Image.open(io.BytesIO(bytes(msg.data))) as image:
        image = image.convert("RGB")

        if compress:
            resampling = getattr(Image, "Resampling", Image)

            image = image.resize(
                (COMPRESSED_IMAGE_WIDTH, COMPRESSED_IMAGE_HEIGHT),
                resampling.LANCZOS,
            )

        return np.asarray(image, dtype=np.uint8)


def prepare_stream(samples: list[Sample], name: str) -> list[Sample]:
    if not samples:
        raise ValueError(f"No messages found for {name}")

    samples.sort(key=lambda sample: sample.timestamp)

    cleaned: list[Sample] = []

    for sample in samples:
        if cleaned and sample.timestamp == cleaned[-1].timestamp:
            cleaned[-1] = sample
        else:
            cleaned.append(sample)

    timestamps = np.asarray(
        [sample.timestamp for sample in cleaned],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(timestamps)):
        raise ValueError(f"{name}: invalid timestamps")

    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{name}: timestamps are not strictly increasing")

    return cleaned


def nearest_sample(
    samples: list[Sample],
    timestamp: float,
) -> tuple[Any, float]:
    times = np.asarray(
        [sample.timestamp for sample in samples],
        dtype=np.float64,
    )

    pos = int(np.searchsorted(times, timestamp))
    candidates = [i for i in (pos - 1, pos) if 0 <= i < len(samples)]

    if not candidates:
        raise ValueError("Cannot find nearest sample")

    index = min(
        candidates,
        key=lambda i: abs(times[i] - timestamp),
    )

    skew = abs(times[index] - timestamp)

    return samples[index].value, skew


def read_bag(
    bag_path: Path,
    compress: bool,
) -> dict[str, Any]:
    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError("ROS1 Python rosbag package is required") from exc

    arm_samples: list[Sample] = []
    hand_samples: list[Sample] = []

    camera_samples = {
        "head": [],
        "left_wrist": [],
        "right_wrist": [],
    }

    camera_topics = {
        HEAD_CAMERA_TOPIC: "head",
        LEFT_WRIST_CAMERA_TOPIC: "left_wrist",
        RIGHT_WRIST_CAMERA_TOPIC: "right_wrist",
    }

    topics = [
        ARM_TOPIC,
        HAND_TOPIC,
        HEAD_CAMERA_TOPIC,
        LEFT_WRIST_CAMERA_TOPIC,
        RIGHT_WRIST_CAMERA_TOPIC,
    ]

    print(f"Reading: {bag_path}")

    with rosbag.Bag(str(bag_path), "r") as bag:
        total_messages = bag.get_message_count(topic_filters=topics)

        for i, (topic, msg, bag_time) in enumerate(
            bag.read_messages(topics=topics),
            start=1,
        ):
            timestamp = get_timestamp(msg, bag_time)

            if topic == ARM_TOPIC:
                arm_samples.append(
                    Sample(timestamp, read_arm(msg))
                )

            elif topic == HAND_TOPIC:
                hand_samples.append(
                    Sample(timestamp, read_hand(msg))
                )

            elif topic in camera_topics:
                name = camera_topics[topic]

                camera_samples[name].append(
                    Sample(
                        timestamp,
                        read_image(msg, compress),
                    )
                )

            if i % 100 == 0 or i == total_messages:
                print_progress(
                    i,
                    total_messages,
                    "Reading bag",
                )

    arm_samples = prepare_stream(
        arm_samples,
        "arm",
    )

    hand_samples = prepare_stream(
        hand_samples,
        "hand",
    )

    camera_samples = {
        name: prepare_stream(samples, f"camera:{name}")
        for name, samples in camera_samples.items()
    }

    print(f"Arm messages  : {len(arm_samples)}")
    print(f"Hand messages : {len(hand_samples)}")

    for name, samples in camera_samples.items():
        print(f"{name:12}: {len(samples)} frames")

    return {
        "arm": arm_samples,
        "hand": hand_samples,
        "cameras": camera_samples,
    }


def build_episode(streams: dict[str, Any]) -> dict[str, Any]:
    arm = streams["arm"]
    hand = streams["hand"]
    cameras = streams["cameras"]

    if len(arm) < 2:
        raise ValueError("At least two arm samples are required")

    obs = []
    actions = []
    next_obs = []
    timestamps = []
    source_indices = []

    aligned_images = {
        name: []
        for name in cameras
    }

    camera_skews = {
        name: []
        for name in cameras
    }

    hand_skews = []

    dropped = 0
    total = len(arm) - 1

    for i in range(total):
        current_arm = arm[i]
        next_arm = arm[i + 1]

        hand_now, hand_skew_now = nearest_sample(
            hand,
            current_arm.timestamp,
        )

        hand_next, hand_skew_next = nearest_sample(
            hand,
            next_arm.timestamp,
        )

        if (
            hand_skew_now > MAX_HAND_SKEW_S
            or hand_skew_next > MAX_HAND_SKEW_S
        ):
            dropped += 1
            print_progress(
                i + 1,
                total,
                "Building dataset",
            )
            continue

        current_state = np.concatenate(
            [
                current_arm.value,
                hand_now,
            ]
        ).astype(np.float32)

        next_state = np.concatenate(
            [
                next_arm.value,
                hand_next,
            ]
        ).astype(np.float32)

        if (
            current_state.shape != (26,)
            or next_state.shape != (26,)
        ):
            raise ValueError("Expected 26D states")

        frames = {}
        frame_skews = {}
        valid = True

        for camera_name, camera_stream in cameras.items():
            frame, skew = nearest_sample(
                camera_stream,
                current_arm.timestamp,
            )

            if skew > MAX_CAMERA_SKEW_S:
                valid = False
                break

            frames[camera_name] = frame
            frame_skews[camera_name] = skew

        if not valid:
            dropped += 1
            print_progress(
                i + 1,
                total,
                "Building dataset",
            )
            continue

        obs.append(current_state)
        actions.append(next_state)
        next_obs.append(next_state)
        timestamps.append(current_arm.timestamp)
        source_indices.append(i)

        hand_skews.append(
            max(
                hand_skew_now,
                hand_skew_next,
            )
        )

        for camera_name in cameras:
            aligned_images[camera_name].append(
                frames[camera_name]
            )

            camera_skews[camera_name].append(
                frame_skews[camera_name]
            )

        print_progress(
            i + 1,
            total,
            "Building dataset",
        )

    if not obs:
        raise ValueError(
            "No valid transitions remained after synchronization"
        )

    obs = np.stack(obs)
    actions = np.stack(actions)
    next_obs = np.stack(next_obs)

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    done = np.zeros(
        len(obs),
        dtype=np.bool_,
    )

    for j in range(len(source_indices) - 1):
        if source_indices[j + 1] != source_indices[j] + 1:
            done[j] = True

    done[-1] = True

    aligned_images = {
        name: np.stack(frames)
        for name, frames in aligned_images.items()
    }

    print()
    print(f"Valid transitions   : {len(obs)}")
    print(f"Dropped transitions : {dropped}")
    print(f"Continuous segments : {int(np.count_nonzero(done))}")

    if hand_skews:
        print(
            f"Max hand skew       : "
            f"{max(hand_skews) * 1000:.1f} ms"
        )

    for name, values in camera_skews.items():
        if values:
            print(
                f"Max {name} skew : "
                f"{max(values) * 1000:.1f} ms"
            )

    return {
        "obs/state": obs,
        "action": actions,
        "next_obs/state": next_obs,
        "done": done,
        "timestamp": timestamps,
        "images": aligned_images,
    }


def build_metadata(
    bag_path: Path,
    episode: dict[str, Any],
    compress: bool,
) -> dict[str, Any]:
    dataset_hz = measure_dataset_hz(
        episode["timestamp"]
    )

    temporal = build_temporal_contract_metadata(
        dataset_hz
    )

    temporal.update(
        {
            "transition_clock": "kuavo_arm_traj",
            "alignment": "nearest_neighbor",
            "max_hand_skew_s": MAX_HAND_SKEW_S,
            "max_camera_skew_s": MAX_CAMERA_SKEW_S,
            "resampled": False,
        }
    )

    contract = get_default_contract_metadata()

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "obs_dim": 26,
        "action_dim": 26,
        "arm_dim": 14,
        "hand_dim": 12,
        "arm_joint_names": ARM_JOINT_NAMES,
        "hand_value_names": HAND_VALUE_NAMES,
        "source_bag": bag_path.name,
        "instruction": "",
        "camera_mapping": {
            "head": HEAD_CAMERA_TOPIC,
            "left_wrist": LEFT_WRIST_CAMERA_TOPIC,
            "right_wrist": RIGHT_WRIST_CAMERA_TOPIC,
        },
        "topics": {
            "arm": ARM_TOPIC,
            "hand": HAND_TOPIC,
        },
        "units": {
            "arm": "radians",
            "hand": "raw_robot_hand_values",
            "images": "uint8_rgb",
        },
        "image_storage": {
            "compression": (
                "blosc_zstd"
                if compress
                else "default"
            ),
            "storage_compression_lossless": True,
            "resolution_changed": compress,
            "width": (
                COMPRESSED_IMAGE_WIDTH
                if compress
                else None
            ),
            "height": (
                COMPRESSED_IMAGE_HEIGHT
                if compress
                else None
            ),
        },
        "state_source": "recorded_command_proxy",
        "measured_robot_state": False,
        "transition_definition": (
            "obs=command_t, "
            "action=command_t+1, "
            "next_obs=command_t+1"
        ),
        "temporal": temporal,
        "state_layout": contract["state_layout"],
        "action_layout": contract["action_layout"],
        "action_type": contract["action_type"],
    }


def convert(
    bag_path: Path,
    compress: bool = False,
    overwrite: bool = False,
) -> Path:
    bag_path = bag_path.expanduser().resolve()

    if not bag_path.exists():
        raise FileNotFoundError(bag_path)

    if bag_path.suffix != ".bag":
        raise ValueError(
            f"Expected .bag file: {bag_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{bag_path.stem}.zarr"
    )

    print(f"Input       : {bag_path}")
    print(f"Output      : {output_path}")

    if compress:
        print(
            "Compression : Blosc/Zstd + "
            f"{COMPRESSED_IMAGE_WIDTH}x{COMPRESSED_IMAGE_HEIGHT} resize"
        )
    else:
        print(
            "Compression : disabled, original image resolution"
        )

    streams = read_bag(
        bag_path,
        compress,
    )

    episode = build_episode(
        streams
    )

    metadata = build_metadata(
        bag_path,
        episode,
        compress,
    )

    print("\nWriting Zarr...")

    summary = write_canonical_zarr(
        output_path,
        episode,
        metadata,
        overwrite=overwrite,
        compress_images=compress,
    )

    print("\nConversion complete")
    print(f"Transitions : {summary['transitions']}")
    print("State       : 26D")
    print("Action      : 26D")

    if compress:
        print(
            f"Images      : "
            f"{COMPRESSED_IMAGE_WIDTH}x{COMPRESSED_IMAGE_HEIGHT}"
        )
    else:
        print("Images      : original resolution")

    print(
        f"Cameras     : "
        f"{', '.join(summary['cameras'])}"
    )

    print(f"Output      : {output_path}")
    print("Validation  : PASS")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert KUAVO / ExoSuit ROS1 bag "
            "to canonical BFM Zarr"
        )
    )

    parser.add_argument(
        "bag",
        type=Path,
        help="Path to ROS1 .bag file",
    )

    parser.add_argument(
        "--compress",
        action="store_true",
        help=(
            "Resize images to 320x240 and enable "
            "Blosc/Zstd Zarr compression"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Zarr dataset",
    )

    args = parser.parse_args()

    convert(
        args.bag,
        compress=args.compress,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
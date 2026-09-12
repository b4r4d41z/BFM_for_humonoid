"""ROS1 bag adapter. ROS dependencies are deliberately isolated in this module."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.data_conversion.bag_converter import Timestamped

REQUIRED_ROLES = ("arm_state", "hand_state", "arm_command", "hand_command")


def load_topic_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    streams = config.get("streams", {})
    missing = [role for role in REQUIRED_ROLES if role not in streams]
    if missing:
        raise ValueError(f"topic config missing required stream roles: {missing}")
    for role, spec in streams.items():
        if not isinstance(spec, dict) or not spec.get("topic") or not spec.get("field"):
            raise ValueError(f"stream {role!r} requires explicit 'topic' and 'field'")
    cameras = config.get("cameras", {})
    for role, spec in cameras.items():
        if not isinstance(spec, dict) or not spec.get("topic"):
            raise ValueError(f"camera {role!r} requires a topic")
    return config


def _field(message: Any, path: str) -> Any:
    value = message
    for component in path.split("."):
        if not component:
            raise ValueError("message field path must not contain empty components")
        value = getattr(value, component)
    return value


def _decode_compressed(message: Any) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("decoding sensor_msgs/CompressedImage requires Pillow") from exc
    with Image.open(io.BytesIO(bytes(message.data))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def read_rosbag(path: str | Path, config: dict[str, Any]) -> dict[str, list[Timestamped]]:
    """Read configured vectors and decoded compressed images from a ROS1 bag."""
    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError("ROS bag conversion requires ROS1's Python 'rosbag' package") from exc
    streams: dict[str, list[Timestamped]] = {role: [] for role in REQUIRED_ROLES}
    streams.update({f"camera:{role}": [] for role in config.get("cameras", {})})
    vector_by_topic = {spec["topic"]: (role, spec) for role, spec in config["streams"].items()}
    camera_by_topic = {spec["topic"]: role for role, spec in config.get("cameras", {}).items()}
    topics = list(vector_by_topic) + list(camera_by_topic)
    with rosbag.Bag(str(path), "r") as bag:
        for topic, message, bag_stamp in bag.read_messages(topics=topics):
            header = getattr(message, "header", None)
            stamp = getattr(header, "stamp", bag_stamp)
            timestamp = float(stamp.to_sec())
            if topic in vector_by_topic:
                role, spec = vector_by_topic[topic]
                value = np.asarray(_field(message, spec["field"]), dtype=np.float64).reshape(-1)
                value = value * float(spec.get("scale", 1.0)) + float(spec.get("offset", 0.0))
                streams[role].append(Timestamped(timestamp, value))
            if topic in camera_by_topic:
                if not hasattr(message, "data") or "compressed" not in type(message).__name__.lower():
                    raise ValueError(f"camera topic {topic} is not a sensor_msgs/CompressedImage")
                streams[f"camera:{camera_by_topic[topic]}"].append(Timestamped(timestamp, _decode_compressed(message)))
    return streams

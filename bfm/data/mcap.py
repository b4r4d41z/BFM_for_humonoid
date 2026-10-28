"""ROS2 MCAP ingestion. All synchronization is causal in recorder log time."""
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from bfm.contracts.sharpa import FIELDS, GROUPS, JOINTS, SCHEMA, VERSION, PROFILE, MODEL_SHA256, SharpaContract, flatten_state, flatten_action
from .sharpa import CAMERAS, SOURCES, write

TOPICS = ("/bfm/state", "/robot/state/joint_states", *[f"/sensors/camera/{c}/image_raw" for c in CAMERAS])
ACTION = "/bfm/action"
HELD = ("/robot_arm_joint_command", "/bfm/command/hand_target")


def header_ns(msg):
    h = getattr(getattr(msg, "header", None), "stamp", None)
    return -1 if h is None else int(h.sec) * 1_000_000_000 + int(h.nanosec)


def decode_image(msg):
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    if msg.encoding not in channels:
        raise ValueError(f"Unsupported raw Image encoding: {msg.encoding}; supported: {list(channels)}")
    h, w, step, n = int(msg.height), int(msg.width), int(msg.step), channels[msg.encoding]
    if min(h, w) <= 0 or step < w*n or msg.is_bigendian not in (0, 1):
        raise ValueError("Invalid raw Image dimensions/stride/endianness")
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if len(raw) != h*step:
        raise ValueError("Raw Image data length differs from height*step")
    image = raw.reshape(h, step)[:, :w*n].reshape(h, w, n)
    if n == 1:
        image = np.repeat(image, 3, axis=2)
    else:
        image = image[:, :, :3]
    if msg.encoding in ("bgr8", "bgra8"):
        image = image[:, :, ::-1]
    return image.copy()  # 8-bit channels are independent of byte endianness.


def asof(samples, time):
    i = bisect_right([s[0] for s in samples], time)-1
    return samples[i] if i >= 0 else None


def read_records(episode):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    episode = Path(episode)
    meta = json.loads((episode / "episode.json").read_text())
    bag = episode / "bag"
    bag_meta = yaml.safe_load((bag / "metadata.yaml").read_text())
    info = bag_meta["rosbag2_bagfile_information"]
    if info["storage_identifier"] != "mcap":
        raise ValueError("Expected rosbag2 with MCAP storage")
    result = defaultdict(list)
    hashes = {}
    for name in info["relative_file_paths"]:
        path = (bag/name).resolve()
        if not path.is_relative_to(bag.resolve()):
            raise ValueError("MCAP path escapes episode bag directory")
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
        with path.open('rb') as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            for schema, channel, message, decoded in reader.iter_decoded_messages(topics=[ACTION, *TOPICS, *HELD, "/bfm/episode"]):
                result[channel.topic].append((message.log_time, decoded))
    return result, meta, bag_meta, hashes


def build(records, metadata, *, max_age_s, collection_commit):
    if len(collection_commit) != 40 or any(c not in '0123456789abcdef' for c in collection_commit):
        raise ValueError("Full collection commit SHA required")
    model_hash = metadata.get("camera_configuration", {}).get("model_sha256")
    if model_hash is not None and model_hash != MODEL_SHA256:
        raise ValueError("Recorded robot model differs from frozen bounds; review the physical contract")
    if metadata.get("time", {}).get("synchronization_source") != "rosbag_record_timestamp":
        raise ValueError("Episode must declare rosbag_record_timestamp synchronization")
    bfm = metadata["bfm"]
    if bfm.get("contract_version") != 2 or bfm.get("action_dim") != 26 or bfm.get("action_group_order") != list(GROUPS) or not np.isclose(bfm.get("policy_hz", 0), 10):
        raise ValueError("Incompatible collection action metadata (legacy order is not supported)")
    for side in JOINTS:
        sem = bfm["action_semantics"][f"{side}_arm"]
        if sem.get("joint_names") != list(JOINTS[side]) or sem.get("unit") != "rad" or sem.get("dim") != 7 or sem.get("type") != "absolute_joint_position_target":
            raise ValueError("Collection arm order/units mismatch")
        hand = bfm["action_semantics"][f"{side}_hand"]
        if hand.get("range") != [0, 100] or hand.get("unit") != "controller_0_100" or hand.get("dim") != 6 or hand.get("type") != "logical_hand_target":
            raise ValueError("Collection hand semantics mismatch")
    if bfm["task_state"].get("frame") != "world":
        raise ValueError("First profile requires measured world-frame task state")
    limits = np.asarray(max_age_s, dtype=float)
    if limits.shape != (5,) or not np.isfinite(limits).all() or (limits <= 0).any():
        raise ValueError("Explicit positive ages required: state,joints,head,left_wrist,right_wrist")
    limits = np.rint(limits*1e9).astype(np.int64)
    for topic in (ACTION, *TOPICS):
        if not records.get(topic):
            raise ValueError(f"Missing required topic: {topic}")
    for topic, samples in records.items():
        samples.sort(key=lambda s: s[0])
        if any(b[0] <= a[0] for a, b in zip(samples, samples[1:])):
            raise ValueError(f"Duplicate recorded timestamp in {topic}")
    frames = {t: getattr(getattr(records[t][0][1], 'header', None), 'frame_id', '') for t in (ACTION, *TOPICS)}
    for topic, frame in frames.items():
        if any(getattr(getattr(m, 'header', None), 'frame_id', '') != frame for _, m in records[topic]):
            raise ValueError(f"Source frame changed within episode: {topic}")
    actions = records[ACTION]
    ages, ticks, names, effort, encodings = [], [], None, None, {}
    for time, msg in actions:
        physical = flatten_action(msg)
        picked = [asof(records[topic], time) for topic in TOPICS]
        age = np.asarray([time-p[0] if p else np.iinfo(np.int64).max for p in picked], dtype=np.int64)
        ages.append(age)
        if any(p is None for p in picked) or (age > limits).any():
            ticks.append(None); continue
        event = asof(records.get("/bfm/episode", []), time)
        if event:
            status = json.loads(event[1].data)
            identity = (status.get("episode") or {}).get("episode_id")
            if status.get("phase") != "playing" or identity != bfm["episode"]["episode_id"]:
                ticks.append(None); continue
        messages = [p[1] for p in picked]
        if messages[0].header.frame_id != "world":
            raise ValueError("BfmState frame differs from episode metadata")
        state = flatten_state(messages[0])
        js = messages[1]
        if names is None:
            names = list(js.name)
        if list(js.name) != names or len(set(names)) != len(names):
            raise ValueError("Measured joint ordering changed or contains duplicates")
        row = {"state": state}
        for field, key in (("position", "joint_position"), ("velocity", "joint_velocity")):
            value = np.asarray(getattr(js, field), dtype=np.float64)
            if value.shape != (len(names),) or not np.isfinite(value).all():
                raise ValueError(f"Missing/invalid measured joint {field}")
            row[key] = value
        has_effort = len(js.effort) > 0
        if effort is not None and has_effort != effort:
            raise ValueError("Effort availability changed within episode")
        effort = has_effort
        if has_effort:
            row["joint_effort"] = np.asarray(js.effort, dtype=np.float64)
        for c, image in zip(CAMERAS, messages[2:]):
            encodings.setdefault(c, image.encoding)
            if image.encoding != encodings[c]:
                raise ValueError("Camera encoding changed within episode")
            row[f"images/{c}"] = decode_image(image)
        row.update(source_record_ns=np.array([p[0] for p in picked], dtype=np.int64),
                   source_header_ns=np.array([header_ns(m) for m in messages], dtype=np.int64), sync_age_ns=age)
        ticks.append((time, physical, row))
    # Command checks may inspect +/-half a tick for transport reordering only.
    # This never provides observations or changes causal sample selection.
    checks = {}
    for topic in HELD:
        count = 0
        if records.get(topic):
            for time, msg in actions:
                target = flatten_action(msg)
                candidates = [m for t, m in records[topic] if abs(t-time) <= 50_000_000]
                def value(m):
                    if topic == HELD[0]:
                        if len(set(m.name)) != len(m.name) or set(m.name) != set(JOINTS["left"]+JOINTS["right"]):
                            raise ValueError("Held arm joint metadata mismatch")
                        by_name = dict(zip(m.name, m.position))
                        return [by_name[n] for side in JOINTS for n in JOINTS[side]]
                    return [*m.left_hand_position, *m.right_hand_position]
                expected = np.r_[target[GROUPS["left_arm"]], target[GROUPS["right_arm"]]] if topic == HELD[0] else np.r_[target[GROUPS["left_hand"]], target[GROUPS["right_hand"]]]
                if not any(np.allclose(value(m), expected, atol=1e-6, rtol=1e-6) for m in candidates):
                    raise ValueError(f"Held command does not match /bfm/action: {topic} at {time}")
                count += 1
        checks[topic] = count
    pairs = []
    for i in range(len(ticks)-1):
        dt = (actions[i+1][0]-actions[i][0])/1e9
        # Never stitch across invalid intermediate ticks or episode control events.
        events = [json.loads(m.data) for t, m in records.get("/bfm/episode", []) if actions[i][0] < t <= actions[i+1][0]]
        boundary = any(e.get("phase") != "playing" or (e.get("episode") or {}).get("episode_id") != bfm["episode"]["episode_id"] for e in events)
        if ticks[i] is not None and ticks[i+1] is not None and .05 <= dt <= .15 and not boundary:
            pairs.append(i)
    if not pairs:
        raise ValueError("No valid causal adjacent action transitions")
    arrays = {"action": np.stack([ticks[i][1] for i in pairs]).astype(np.float64)}
    arrays["action_header_ns"] = np.array([header_ns(actions[i][1]) for i in pairs], dtype=np.int64)
    for p, shift in (("obs", 0), ("next_obs", 1)):
        for k in ticks[pairs[0]][2]:
            arrays[f"{p}/{k}"] = np.stack([ticks[i+shift][2][k] for i in pairs])
    r = np.array([actions[i][0] for i in pairs], dtype=np.int64)
    nr = np.array([actions[i+1][0] for i in pairs], dtype=np.int64)
    ends = np.r_[np.diff(pairs) != 1, True]
    arrays.update(record_ns=r, next_record_ns=nr, timestamp=(r-actions[0][0])/1e9, next_timestamp=(nr-actions[0][0])/1e9,
                  terminated=np.zeros(len(pairs), dtype=bool), truncated=ends, segment_id=np.cumsum(np.r_[0, ends[:-1]]).astype(np.int64))
    age_array = np.array(ages)
    stats = {k: {"available_ticks": int((age_array[:, j] < np.iinfo(np.int64).max).sum()),
                 "age_ms_p50_p95_max": np.percentile(age_array[age_array[:, j] < np.iinfo(np.int64).max, j]/1e6, [50, 95, 100]).tolist()}
             for j, k in enumerate(SOURCES) if (age_array[:, j] < np.iinfo(np.int64).max).any()}
    attrs = {"schema_name": SCHEMA, "schema_version": VERSION, "profile": PROFILE, "state_order": list(FIELDS),
             "state_units": {k: u for k, (_, u) in FIELDS.items()}, "contract": SharpaContract().to_dict(),
             "joint_names": names, "has_effort": bool(effort), "sync_clock": "mcap_log_time_ns", "source_columns": list(SOURCES),
             "joint_units": {"position": "rad", "velocity": "rad/s", "effort": "N*m"},
             "max_age_ns": limits.tolist(), "sync_statistics_before_filter": stats, "held_command_checks": checks,
             "action_tick_dt_s_p50_p95_max": np.percentile(np.diff([t for t, _ in actions])/1e9, [50, 95, 100]).tolist(),
             "rejected_ticks": sum(t is None for t in ticks), "record_origin_ns": actions[0][0], "stored_image_encoding": "rgb8",
             "image_shapes": {c: list(arrays[f"obs/images/{c}"].shape[1:]) for c in CAMERAS}, "source_image_encodings": encodings,
             "source_episode_id": bfm["episode"]["episode_id"], "collection_commit": collection_commit,
             "source_metadata": metadata, "success": metadata.get("success"), "terminal_policy": "recording_and_gap_truncations_only"}
    attrs['source_frame_ids'] = frames
    return arrays, attrs


def convert(episode, output, *, max_age_s, collection_commit):
    records, metadata, bag_metadata, hashes = read_records(episode)
    arrays, attrs = build(records, metadata, max_age_s=max_age_s, collection_commit=collection_commit)
    attrs.update(source_bag_metadata=bag_metadata, source_mcap_sha256=hashes)
    print(json.dumps({"sync_before_filter": attrs["sync_statistics_before_filter"], "rejected_ticks": attrs["rejected_ticks"]}))
    return write(output, arrays, attrs)

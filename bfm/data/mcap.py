"""ROS2 MCAP ingestion. All synchronization is causal in recorder log time."""
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from bfm.contracts.sharpa import FIELDS, GROUPS, JOINTS, LOW, HIGH, SCHEMA, VERSION, PROFILE, MODEL_SHA256, SharpaContract, flatten_state, flatten_action
from .sharpa import CAMERAS, SOURCES, write
from .policy_time import policy_timing, continuity_reason

TOPICS = ("/bfm/state", "/robot/state/joint_states", *[f"/sensors/camera/{c}/image_raw" for c in CAMERAS])
ACTION = "/bfm/action"
HELD = ("/robot_arm_joint_command", "/bfm/command/hand_target")
RAW_ARM = "/bfm/source/arm_joint_target_rad"
ARM_INDICES = list(range(7)) + list(range(13, 20))
EPISODE = '/bfm/episode'
RECORDING = '/data_collection/recording'
CLOCK = '/clock'


def arm_values(message):
    names, values = list(message.name), np.asarray(message.position, dtype=np.float64)
    expected = JOINTS["left"] + JOINTS["right"]
    if len(names) != 14 or set(names) != set(expected) or values.shape != (14,) or not np.isfinite(values).all():
        raise ValueError("Arm command requires 14 unique named finite radian targets")
    return values[[names.index(n) for n in expected]]


def header_ns(msg):
    h = getattr(getattr(msg, "header", None), "stamp", None)
    if h is None:
        return -1
    sec, ns = getattr(h, 'sec', None), getattr(h, 'nanosec', None)
    if not isinstance(sec, (int, np.integer)) or not isinstance(ns, (int, np.integer)) or sec < 0 or not 0 <= ns < 1_000_000_000:
        return -1
    return int(sec) * 1_000_000_000 + int(ns)


def lifecycle(records, episode_id, action_times):
    """Fold schema-v1 explicit events and legacy phases in recorder order.

    Final result fields come from episode.json, never from a provisional event.
    A recording Bool can close/gate an interval, but cannot reopen an explicit
    end request. Each actual boundary increments a generation, even between ticks.
    """
    events = [(t, json.loads(m.data)) for t, m in records.get(EPISODE, [])]
    explicit = any(e.get('event_type') == 'recording_started' for _, e in events)
    controls = [(t, 'episode', e) for t, e in events]
    controls += [(t, 'recording', bool(m.data)) for t, m in records.get(RECORDING, [])]
    controls.sort(key=lambda item: item[0])
    active = not events  # Legacy files without lifecycle evidence.
    if explicit:
        active = False
    recording = not records.get(RECORDING)
    generation, cursor, current_id, boundary_id = 0, 0, episode_id, None
    result = []
    for time in action_times:
        while cursor < len(controls) and controls[cursor][0] <= time:
            _, kind, event = controls[cursor]
            cursor += 1
            if kind == 'recording':
                if recording != event:
                    generation += 1
                recording = event
                continue
            identity = (event.get('episode') or {}).get('episode_id')
            if not identity:
                raise ValueError('Episode lifecycle event is missing episode_id')
            event_type = event.get('event_type')
            if event_type is not None:
                if event.get('event_schema_version') != 1 or event_type not in ('recording_started', 'recording_end_requested', 'recording_result'):
                    raise ValueError('Unsupported episode lifecycle event schema/type')
                generation += 1
                current_id = identity
                active = event_type == 'recording_started'
            elif 'phase' in event:
                next_active = event['phase'] == 'playing'
                next_boundary = event.get('boundary_id', boundary_id)
                if current_id != identity or active != next_active or next_boundary != boundary_id:
                    generation += 1
                current_id, boundary_id = identity, next_boundary
                if not explicit:
                    active = next_active
                elif not next_active:
                    active = False
            else:
                raise ValueError('Episode lifecycle event has no recognized phase/type')
        reason = ('episode_id_mismatch' if current_id != episode_id else
                  'outside_recording_interval' if not active or not recording else None)
        result.append((generation, reason))
    return result


def nearby_command(samples, time, window_ns=50_000_000):
    """Latest nearby causal target; only use a reordered future target if absent.

    Do not search past a newer, conflicting target for an older matching value.
    Header identity plus clamp(raw) is checked separately for arm provenance.
    """
    causal = asof(samples, time)
    if causal is not None and time - causal[0] <= window_ns:
        return causal
    future = next((s for s in samples if time < s[0] <= time + window_ns), None)
    return future


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
            for schema, channel, message, decoded in reader.iter_decoded_messages(topics=[ACTION, *TOPICS, *HELD, RAW_ARM, EPISODE, RECORDING, CLOCK]):
                result[channel.topic].append((message.log_time, decoded))
    return result, meta, bag_meta, hashes


def build(records, metadata, *, max_age_s, collection_commit, max_camera_skew_s=None):
    if len(collection_commit) != 40 or any(c not in '0123456789abcdef' for c in collection_commit):
        raise ValueError("Full collection commit SHA required")
    model_hash = metadata.get("camera_configuration", {}).get("model_sha256")
    if model_hash is not None and model_hash != MODEL_SHA256:
        raise ValueError("Recorded robot model differs from frozen bounds; review the physical contract")
    if metadata.get("time", {}).get("synchronization_source") != "rosbag_record_timestamp":
        raise ValueError("Episode must declare rosbag_record_timestamp synchronization")
    bfm = metadata["bfm"]
    recorded_commit = metadata.get('recording_code', {}).get('commit_sha')
    if recorded_commit is not None and recorded_commit != collection_commit:
        raise ValueError('Collection commit differs from authoritative episode recording_code')
    if bfm.get("contract_version") != 3:
        raise ValueError("Collection v3 required; collection-v2 recordings cannot be silently migrated")
    if bfm.get("action_dim") != 26 or bfm.get("action_group_order") != list(GROUPS) or not np.isclose(bfm.get("policy_hz", 0), 10):
        raise ValueError("Incompatible collection action metadata (legacy order is not supported)")
    for side in JOINTS:
        sem = bfm["action_semantics"][f"{side}_arm"]
        if sem.get("joint_names") != list(JOINTS[side]) or sem.get("unit") != "rad" or sem.get("dim") != 7 or sem.get("type") != "bounded_absolute_joint_position_target":
            raise ValueError("Collection arm order/units mismatch")
        hand = bfm["action_semantics"][f"{side}_hand"]
        if hand.get("range") != [0, 100] or hand.get("unit") != "controller_0_100" or hand.get("dim") != 6 or hand.get("type") != "logical_hand_target":
            raise ValueError("Collection hand semantics mismatch")
    if bfm["task_state"].get("frame") != "world":
        raise ValueError("First profile requires measured world-frame task state")
    processing = bfm.get("arm_target_processing", {})
    bounds = {n: [LOW[i], HIGH[i]] for n, i in zip(JOINTS['left'] + JOINTS['right'], ARM_INDICES)}
    required = {"clamping_enabled": True, "transformation": "per_joint_saturation", "unit": "rad",
                "source_topic": RAW_ARM, "controller_topic": HELD[0], "canonical_action_topic": ACTION,
                "joint_limits": bounds, "bounds_source_sha256": MODEL_SHA256,
                "bounds_source": SharpaContract().to_dict()["bounds_source_path"]}
    if any(processing.get(k) != v for k, v in required.items()):
        raise ValueError("Collection v3 clamp/bounds provenance mismatch")
    if max_camera_skew_s is not None and (not np.isfinite(max_camera_skew_s) or max_camera_skew_s <= 0):
        raise ValueError("max_camera_skew_s must be positive finite or None")
    limits = np.asarray(max_age_s, dtype=float)
    if limits.shape != (5,) or not np.isfinite(limits).all() or (limits <= 0).any():
        raise ValueError("Explicit positive ages required: state,joints,head,left_wrist,right_wrist")
    limits = np.rint(limits*1e9).astype(np.int64)
    for topic in (ACTION, *TOPICS, *HELD, RAW_ARM):
        if not records.get(topic):
            raise ValueError(f"Missing required topic: {topic}")
    for topic, samples in records.items():
        samples.sort(key=lambda s: s[0])
        if any(b[0] <= a[0] for a, b in zip(samples, samples[1:])):
            raise ValueError(f"Duplicate recorded timestamp in {topic}")
    # Missing action headers are rejected as individual policy decisions below.
    framed = {t: [m for _, m in records[t] if t != ACTION or header_ns(m) >= 0] for t in (ACTION, *TOPICS)}
    frames = {t: getattr(getattr(messages[0], 'header', None), 'frame_id', '') if messages else ''
              for t, messages in framed.items()}
    for topic, frame in frames.items():
        if any(getattr(getattr(m, 'header', None), 'frame_id', '') != frame for m in framed[topic]):
            raise ValueError(f"Source frame changed within episode: {topic}")
    actions = records[ACTION]
    timing = policy_timing(bfm, records.get(CLOCK, []))
    simulation = timing['domain'] == 'action_header_simulation_ns'
    policy_stamps = [header_ns(m) if simulation else t for t, m in actions]
    intervals = [continuity_reason(a, b, timing) for a, b in zip(policy_stamps, policy_stamps[1:])]
    lifecycle_ticks = lifecycle(records, bfm['episode']['episode_id'], [t for t, _ in actions])
    rejected = []
    ages, ticks, names, effort, encodings = [], [], None, None, {}
    for index, (time, msg) in enumerate(actions):
        physical = flatten_action(msg)
        picked = [asof(records[topic], time) for topic in TOPICS]
        age = np.asarray([time-p[0] if p else np.iinfo(np.int64).max for p in picked], dtype=np.int64)
        ages.append(age)
        reasons = [f'missing_observation:{SOURCES[j]}' for j, p in enumerate(picked) if p is None]
        reasons += [f'stale_observation:{SOURCES[j]}:age_ns={int(age[j])}:limit_ns={int(limits[j])}'
                    for j, p in enumerate(picked) if p is not None and age[j] > limits[j]]
        if simulation and policy_stamps[index] < 0:
            reasons.append('invalid_policy_stamp')
        if index and intervals[index-1] in ('duplicate_policy_stamp', 'backward_policy_stamp'):
            reasons.append(intervals[index-1])
        if lifecycle_ticks[index][1]:
            reasons.append(lifecycle_ticks[index][1])
        if max_camera_skew_s is not None and all(p is not None for p in picked[2:]) and np.ptp([p[0] for p in picked[2:]]) > max_camera_skew_s * 1e9:
            reasons.append('camera_skew_exceeded')
        if reasons:
            rejected.append({'index': index, 'record_ns': time, 'action_header_ns': header_ns(msg), 'reasons': reasons})
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
    high_rate = (processing.get('controller_cadence') == 'validated source callback rate'
                 and processing.get('canonical_action_cadence') == '0.1 simulation seconds; latest bounded target snapshot')
    def command(topic, time, expected=None):
        if high_rate:
            # A callback-published target stays in force until the next callback.
            # Policy receive time is not required to be close to source publication.
            causal = asof(records[topic], time)
            if topic == HELD[0] and expected is not None and (causal is None or not np.allclose(arm_values(causal[1]), expected, atol=1e-6, rtol=0)):
                # A raw callback may be recorded before its bounded publication.
                # Only the latest causal raw identity can justify such reordering;
                # never search arbitrary future commands for a matching value.
                raw = asof(records[RAW_ARM], time)
                following = next((t for t, _ in actions if t > time), None)
                if raw is not None and header_ns(raw[1]) >= 0:
                    for t, message in records[topic]:
                        if t > time and (following is None or t < following) and header_ns(message) == header_ns(raw[1]):
                            if np.allclose(arm_values(message), expected, atol=1e-6, rtol=0):
                                return t, message
            return causal
        return nearby_command(records[topic], time)

    checks, raw_ages, saturated = {}, [], 0
    saturation_counts = np.zeros(14, dtype=np.int64)
    command_provenance = []
    for topic in HELD:
        count = 0
        if records.get(topic):
            for time, msg in actions:
                target = flatten_action(msg)
                def value(m):
                    if topic == HELD[0]:
                        return arm_values(m)
                    values = np.asarray([*m.left_hand_position, *m.right_hand_position], dtype=np.float64)
                    if len(m.left_hand_position) != 6 or len(m.right_hand_position) != 6 or not np.isfinite(values).all():
                        raise ValueError('Held hand command requires six finite values per hand')
                    return values
                expected = np.r_[target[GROUPS["left_arm"]], target[GROUPS["right_arm"]]] if topic == HELD[0] else np.r_[target[GROUPS["left_hand"]], target[GROUPS["right_hand"]]]
                candidate = command(topic, time, expected)
                if candidate is None or not np.allclose(value(candidate[1]), expected, atol=1e-6, rtol=0):
                    raise ValueError(f"Held command does not match /bfm/action: {topic} at {time}")
                count += 1
        checks[topic] = count
    # The gateway preserves the source header in the held arm snapshot. Match
    # header identity AND clamped values, within one tick plus transport jitter.
    # Raw commands are provenance only: never used in an observation or action.
    for time, msg in actions:
        expected = flatten_action(msg)[ARM_INDICES]
        held = [command(HELD[0], time, expected)[1]]
        matches = [(t, m) for t, m in records[RAW_ARM] if (high_rate or abs(t-time) <= 150_000_000)
                   and any(header_ns(m) == header_ns(h) and header_ns(m) >= 0 for h in held)
                   and np.allclose(np.clip(arm_values(m), np.array(LOW)[ARM_INDICES], np.array(HIGH)[ARM_INDICES]),
                                   expected, atol=1e-6, rtol=0)]
        if not matches:
            raise ValueError(f"Raw arm provenance does not match held header and clamp(raw)==action at {time}")
        # Prefer a preceding equivalent request when repeated headers/values are
        # indistinguishable; use later record time only for transport reordering.
        t, raw = min(matches, key=lambda sample: (sample[0] > time, abs(sample[0]-time)))
        raw_ages.append((time-t)/1e6)
        saturated += int(not np.allclose(arm_values(raw), expected, atol=1e-6, rtol=0))
        raw_values = arm_values(raw)
        saturation_counts += (raw_values < np.array(LOW)[ARM_INDICES]) | (raw_values > np.array(HIGH)[ARM_INDICES])
        command_provenance.append({'action_record_ns': time,
                                   'controller_record_ns': command(HELD[0], time, expected)[0],
                                   'raw_record_ns': t, 'source_header_ns': header_ns(raw)})
    pairs = []
    rejected_intervals = []
    for i in range(len(ticks)-1):
        boundary = lifecycle_ticks[i][0] != lifecycle_ticks[i+1][0]
        if intervals[i] is not None or boundary:
            rejected_intervals.append({'from_index': i, 'to_index': i+1,
                                       'reasons': ([intervals[i]] if intervals[i] else []) + (['lifecycle_boundary'] if boundary else [])})
        if ticks[i] is not None and ticks[i+1] is not None and intervals[i] is None and not boundary:
            pairs.append(i)
    if not pairs:
        raise ValueError("No valid causal adjacent action transitions")
    arrays = {"action": np.stack([ticks[i][1] for i in pairs]).astype(np.float64)}
    arrays["action_header_ns"] = np.array([header_ns(actions[i][1]) for i in pairs], dtype=np.int64)
    if simulation:
        arrays['next_action_header_ns'] = np.array([policy_stamps[i+1] for i in pairs], dtype=np.int64)
    for p, shift in (("obs", 0), ("next_obs", 1)):
        for k in ticks[pairs[0]][2]:
            arrays[f"{p}/{k}"] = np.stack([ticks[i+shift][2][k] for i in pairs])
    r = np.array([actions[i][0] for i in pairs], dtype=np.int64)
    nr = np.array([actions[i+1][0] for i in pairs], dtype=np.int64)
    ends = np.r_[np.diff(pairs) != 1, True]
    origin = next((s for s in policy_stamps if s >= 0), 0)
    pt = np.array([policy_stamps[i] for i in pairs], dtype=np.int64)
    npt = np.array([policy_stamps[i+1] for i in pairs], dtype=np.int64)
    arrays.update(record_ns=r, next_record_ns=nr, timestamp=(pt-origin)/1e9, next_timestamp=(npt-origin)/1e9,
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
             "rejected_ticks": len(rejected), "rejected_decisions": rejected, "rejected_intervals": rejected_intervals,
             "total_action_ticks": len(actions), "valid_decision_ticks": sum(t is not None for t in ticks),
             "policy_timing": timing, "policy_origin_ns": origin,
             "record_origin_ns": actions[0][0], "stored_image_encoding": "rgb8",
             "image_shapes": {c: list(arrays[f"obs/images/{c}"].shape[1:]) for c in CAMERAS}, "source_image_encodings": encodings,
             "source_episode_id": bfm["episode"]["episode_id"], "collection_commit": collection_commit,
             "source_metadata": metadata, "success": metadata.get("success"), "terminal_policy": "recording_and_gap_truncations_only"}
    attrs['source_frame_ids'] = frames
    deltas = np.diff(policy_stamps)
    attrs['policy_tick_dt_s_min_p50_p95_max'] = np.percentile(deltas / 1e9, [0, 50, 95, 100]).tolist()
    attrs['policy_interval_reasons'] = {reason: intervals.count(reason) for reason in sorted(set(intervals) - {None})}
    attrs['controller_command_association'] = ('latest_causal_callback_target_and_raw_header' if high_rate else
                                               'latest_causal_within_50ms_else_first_reordered_within_50ms')
    attrs['command_provenance'] = command_provenance
    # Unique retained policy ticks, including final next_obs, not duplicated edges.
    kept = sorted(set(pairs) | {i+1 for i in pairs})
    skew = np.array([np.ptp(ticks[i][2]['source_record_ns'][2:])/1e6 for i in kept])
    attrs.update({f"camera_skew_ms_{key}": float(value) for key, value in
                  zip(('p50', 'p95', 'max'), np.percentile(skew, [50, 95, 100]))})
    attrs.update(max_camera_skew_s=max_camera_skew_s, camera_skew_tick_count=len(kept),
                 raw_command_checks={"count": len(raw_ages), "saturated_ticks": saturated,
                                     "saturation_per_joint": dict(zip(JOINTS['left'] + JOINTS['right'], saturation_counts.tolist())),
                                     "window_ms": None if high_rate else 150, "association": "held_source_header_and_clamped_values",
                                     "age_ms_p50_p95_max": np.percentile(raw_ages, [50, 95, 100]).tolist()})
    return arrays, attrs


def convert(episode, output, *, max_age_s, collection_commit, max_camera_skew_s=None):
    records, metadata, bag_metadata, hashes = read_records(episode)
    arrays, attrs = build(records, metadata, max_age_s=max_age_s, collection_commit=collection_commit,
                          max_camera_skew_s=max_camera_skew_s)
    attrs.update(source_bag_metadata=bag_metadata, source_mcap_sha256=hashes)
    print(json.dumps({"sync_before_filter": attrs["sync_statistics_before_filter"], "rejected_ticks": attrs["rejected_ticks"]}))
    return write(output, arrays, attrs)

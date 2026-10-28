# Current KUAVO S40 + Sharpa migration audit

BFM BASE_SHA: `18c239b9e3bf2fb8f3c6c3e754e94ad5e4c2d3d2` on local `zarr-pipeline`.
The local checkout is authoritative; this patch does not assume remote `fc40e4a`.
Collection main inspected read-only: `08162e1da037a2bb1827cad33e71c3e7a183de53`.
No collection source or simulator asset is changed/copied by this patch.

## Findings and decisions

The inspected messages, gateway, controller, PhysX sampler, recorder, snapshot,
scene JSON, camera metadata, assembly XML/USD physics and collection tests agree:

* Contract v2 action is left arm 7 rad, left logical hand 6 values 0..100,
  right arm 7 rad, right logical hand 6 values 0..100. It is not arm14+hand12.
* BfmState is 56 measured values, in the exact order listed below. Positions are
  world-frame metres, velocities SI, goal orientation xyzw; the goal is a moving
  oriented box. Object state is the selected fruit's live PhysX COM, not red ball.
* Logical hand commands are not measured finger joints. Full measured JointState
  is separate; the Sharpa mapping is stateful, mirrored and not invertible.
* Gateway decisions are 10 Hz; arm and hand inputs are held. Existing hand
  interpolation runs at 50 Hz with rate limits and thumb sequencing. Republishing
  `/bfm/action` alone does not control the simulator: it has no command subscriber.
* The snapshot is 47.37754052 s: action 473, state 884, measured joints 876,
  each camera 877, held arms/source arms/held hands 473 each, interpolated hand
  commands 2363 each. Counts/duration imply about 10, 18.7, 18.5 and 49.9 Hz,
  respectively; these are not measured interval/jitter distributions.
* Action headers are epoch-like, state headers simulation time. Only MCAP
  `Message.log_time` is used for synchronization; native headers remain provenance.
* Snapshot `success=null` supplies no task terminal. Operator stop, file end and
  dropped synchronization intervals are truncations. No success is invented.

Primary solution: use the current collection simulator through its existing ROS2
held-command interface. Do not move assets into old IsaacLab. The old S40 arm
names overlap, but mechanical/actuator/hand equivalence is not established: old
binary finger claws are not Sharpa hands, action order differs, and old table,
balls/container, cameras, 26D observation, TCP, reset/randomization and measured
task definition differ. Old physics is configured at 120 Hz / 60 Hz control;
new policy is 10 Hz and hand controller 50 Hz, while current native physics rate
cannot be established from the recording counts alone. Old-environment results
are not valid evidence for the current data distribution or manipulation task.

Legacy incompatibility inventory: ROS1 `tools/data_conversion/bag_to_zarr.py`
uses header clocks, command proxies, nearest matching, compressed images and
old names; `bc/data/schema.py`, HDF5/stream/batch loaders and canonical Zarr v1
assume 26=14+12; `bfm/contracts/kuavo.py`, measured schema v2, dataset/validation,
checkpoint checks, old runner/state extractor and IsaacLab assume 28/8 and binary
claws. BC policy/runtime, generic padding/trimming ObservationAdapter and
arm_plus_gripper_bridge cannot execute the new contract. `bfm/fb` loss/network
code itself supports arbitrary dimensions: it needs only measured s,a,s_next,
termination mask and dt. Targets, actor objective and reward-free training are
reused without algorithm changes. No new BC policy imports are introduced.

## Availability and consumers

| Requirement | Consumer / shape | Source | Available / conversion / recommendation |
|---|---|---|---|
| observation, next observation | FBAgent [B,56] | /bfm/state | Measured; causal as-of at adjacent action ticks |
| action | FBAgent [B,26] normalized | /bfm/action | Physical values; fixed scaling at loader/runtime only |
| terminal/truncated | update/replay [B] bool | episode/recording lifecycle | Task terminal missing; truncate recording/gap ends, bootstrap |
| timestamp/dt | FBAgent discount [B] | MCAP log time | int64 record ns and relative float64 seconds |
| episode boundaries/split | manifest, replay | episode.json + /bfm/episode | Source identity retained; split whole source recordings only |
| task/goal geometry | task56, reward prompt | /bfm/state | Measured oriented box; no success certificate |
| arm proprioception | task56 | /bfm/state | q/dq, rad/rad/s, not commands |
| hand/other proprioception | canonical / future profile | /robot/state/joint_states | All named q/dq and optional effort retained; no inverse logical mapping |
| images | canonical / visual BC comparison | three ROS2 raw Image topics | Decode to RGB; not consumed by vanilla FB |
| reward | post-training prompt only | distances over reference states | Derived after pretraining; not required/recorded for FB updates |
| language/task ID | future language models | episode.json | Preserved exactly; no language encoder claimed |
| success | future evaluation labels | success=null | Unknown; add explicit outcome/event in collector |
| normalization | model checkpoint | training split only | Fixed action bounds, state mean/std; raw arrays unchanged |

The same SharpaDataset episodes and BFMReplayBuffer.sample_sequences can supply
state-based BC/BC-RNN with the same normalized inputs/actions and split policy.
Raw cameras/joints remain for future visual baselines/Robomimic export. This
patch does not implement a new BC trainer or Robomimic exporter. Existing BC
HDF5/Zarr v1 and measured/claw v2 remain isolated legacy formats. They are not
auto-migrated, renamed, or silently reinterpreted. Old checkpoints use their old
runner; the old runner explicitly rejects Sharpa checkpoints.

## Canonical schema 3.0.0 (Zarr 2 storage)

Each store represents ONE source recording, possibly containing synchronization
segments. Let T=valid adjacent transitions, J=number of measured joint names.
No future observation is used; invalid tick i prevents both edges touching i.

```text
episode.zarr/
  action                         float64 [T,26] physical rad / logical 0..100
  action_header_ns               int64 [T] native action header, provenance only
  record_ns, next_record_ns       int64 [T] epoch-like recorder log nanoseconds
  timestamp, next_timestamp      float64 [T] seconds relative to first action
  terminated                     bool [T] false until a real terminal contract exists
  truncated                      bool [T] true at each recording/gap segment end
  segment_id                     int64 [T] contiguous segment IDs starting at zero
  obs/ and next_obs/ (same layout):
    state                        float64 [T,56] named task56 layout below
    joint_position               float64 [T,J] measured rad, attrs.joint_names order
    joint_velocity               float64 [T,J] measured rad/s, same order
    joint_effort                 float64 [T,J] optional measured N*m; omitted if absent
    source_record_ns             int64 [T,5] source recorder timestamps
    source_header_ns             int64 [T,5] native timestamps or -1 if absent
    sync_age_ns                  int64 [T,5] action tick minus source record time
    images/{head,left_wrist,right_wrist} uint8 [T,H,W,3] RGB at native resolution
```

The five source columns are state, joints, head, left_wrist, right_wrist.
Attrs contain schema name/version, task56 profile, exact state order/units,
physical bounds/joint groups and bounds source SHA/path/checksum, joint names,
source frame IDs, effort availability/units, image shapes/source encodings,
sync limits/statistics before filtering, actual action dt percentiles, rejected
tick count, held-command checks, source episode ID, collection commit, complete
episode.json and rosbag metadata, raw MCAP checksums and unknown success label.
Camera configuration, task/instruction, robot, scene, reset poses and asset
provenance are preserved within source_metadata; null calibration/frame values
remain unknown rather than fabricated. Images are neither resized nor normalized.
Eight-bit RGB/BGR/RGBA/BGRA/mono are supported, including padded step and either
endianness flag (8-bit channels are endian-independent). Other encodings fail.
No raw camera messages occur in the textual snapshot, so live encoding remains
unverified. Alpha is explicitly discarded and mono replicated to RGB.

task56 order: left_eef_position(3), right_eef_position(3), left_arm_position(7),
right_arm_position(7), left_arm_velocity(7), right_arm_velocity(7),
object_position(3), object_velocity(3), goal_center(3), goal_half_extents(3),
goal_orientation(4 xyzw), goal_linear_velocity(3), goal_angular_velocity(3).
FIELDS/SLICES in bfm/contracts/sharpa.py are authoritative. Named components are
lossless views into this flat array; unflatten_state provides them. The first
training profile excludes hands/images deliberately. It is a partially observed
robot state: hidden hand interpolation/contact/object orientation may violate a
strict Markov assumption. A later hand-inclusive/recurrent profile can use the
preserved measurements; task56 is not claimed to capture full robot dynamics.

Validation checks dtypes, shapes, finite values, causal ages, duplicate times,
10 Hz interval tolerance [0.05,0.15] s, goal quaternion norm/positive extents,
fixed command limits/order, source frame consistency and next-observation
continuity, including images/effort, within segments. Schema/order confusion is
rejected even when dimensions happen to match. Held commands, when present,
must match each decision within +/-50 ms recorded time, allowing transport
reordering only; this never supplies observations. A match is not an actuator
execution acknowledgement. Loader validates all selected train stores and reads
only small arrays into replay; image continuity validation streams frame pairs.
Conversion currently buffers an episode's messages/images; use bounded episode
lengths on memory-limited machines. Out-of-core conversion is not implemented.

## Model and runtime contract

Actor input is float32 [B,56] raw task56 passed through checkpointed training-only
mean/std inside FBModel, plus latent [B,d]. Output is float32 [B,26] in [-1,1]
in the canonical group order. Arms use fixed per-joint XML limits; hands use
0..100. Neither scaling nor bounds is fitted to demonstration min/max. Conversion
rejects out-of-range commands instead of clipping canonical data. Runtime inverses
the fixed affine scaling, rounds logical hands to uint8 nearest integers, and
records exactly those sent hand values. There is no binary gripper conversion.
Bounds come from assembly/s40_sharpa_two_cameras.xml, SHA256
`4ac4549699cd3de774b676d2acaf9d0f35f1c2b041587ed7f44b535f80e43dcf`.
Its radian bounds agree to export precision with authored USD Physics degree
limits; live composed physics overrides still need deployment verification.

Use current simulator checkout/assets at the pinned collection commit. Keep
its task-state publisher, ROS2 bridge, PhysX variant and BFM-mode eff_control_node
running; STOP the ExoSuit gateway and all other held-topic writers. New runtime
publishes named radian JointState to /robot_arm_joint_command and uint8 logical
RobotHandPosition to /bfm/command/hand_target, plus a BfmAction decision record.
It checks for competing publishers and local state-receive staleness. High-level
targets update at 10 Hz and remain held between ticks; existing 50 Hz Sharpa
interpolation remains authoritative. A stale stream stops new publication but
does not reset held actuator targets. Real ROS/Isaac transport is unverified.
Do not use the old eval_zero_shot_fb.py/IsaacLab environment for current claims.

## Reproducible commands (repository root)

```bash
python -m pip install -r requirements-fb.txt
python tools/data_conversion/mcap_to_zarr.py /data/record_xxx --output /data/episode.zarr --collection-commit 08162e1da037a2bb1827cad33e71c3e7a183de53 --max-ages .15 .15 .15 .15 .15
python tools/data_conversion/mcap_to_zarr.py --validate /data/episode.zarr
python tools/data_conversion/mcap_to_zarr.py --manifest /data/manifest.json --train /data/episode.zarr --test /data/other_episode.zarr
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/train_offline_fb.py --config configs/fb/vanilla_offline.yaml --dataset-manifest /data/manifest.json --output-dir runs/fb/sharpa --training-steps 10
python scripts/run_sharpa_fb.py --checkpoint runs/fb/sharpa/last.pt --task object_to_goal --max-state-age .15
python -m pytest tests/fb -q
```

The five max ages are REQUIRED user thresholds, not measured conclusions from
the snapshot. Conversion reports p50/p95/max causal age before filtering; inspect
these on real recordings and choose appropriate strict values. Batch conversion
accepts multiple episode paths with --output-dir instead of --output. Runtime
requires a ROS2 environment with the pinned dynamic_biped messages built/sourced.
Direct MCAP decoding uses embedded ros2msg schemas/CDR through mcap-ros2-support;
no ROS installation is needed for conversion. Missing/unsupported embedded schemas
fail; ros2idl-only or stripped-schema bags are not established compatible.

## Remaining validation / collection recommendations

Executed: `python -m pytest tests/fb -q`: 52 passed (23 migration tests plus 29
existing FB tests). Full `python -m pytest -q`: 64 passed, one pre-existing
`test_future_targets_are_timestamp_based_half_second_and_do_not_cross_done`
failure in unchanged legacy bc.temporal (0.4 s mismatch exceeds 0.3 s tolerance).
The generated MCAP fixture passed the conversion/validate/manifest CLI path;
default 56/26 architecture completed 10 real FB optimizer updates and wrote a
loadable checkpoint. New tests also exercise 4-step training + resume to 6 and
physical runtime output. These are synthetic software checks, not robot results.

Only textual snapshots are available, not a real captured MCAP. Tests generate
small genuine CDR/MCAP fixtures, including headers in incompatible domains.
There is no measured real-data synchronization distribution, no full training,
no real ROS/Isaac policy rollout and no manipulation success claim.
Future collection should add: actual terminal reason + episode-ID event/time,
explicit success/unknown outcome; controller sequence/execution acknowledgement;
live composed DOF limits and asset revision; measured camera encoding/frame IDs;
runtime-verified optical calibration where needed; physical simulation dt/real
time factor and reset markers. Keep the existing stop-ack/reset barrier.
The stored OBB was calibrated for an earlier object; for larger randomized fruit
it is not a containment certificate. Validate task success using actual object
geometry/settled state. No source-repository changes are made here.

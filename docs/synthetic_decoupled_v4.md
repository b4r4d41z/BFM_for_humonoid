# Synthetic decoupled V4 compatibility

Interface audited against Synthetic main
`131cfb57eadc18bce5a479fd50bf228e38b266a0`; BFM base
`4c0782083b8dce278877c2d52f2251b63e35142a`. Synthetic is read-only.
The source sets PHYSICS_HZ=60; older paragraphs and the BfmAction comment still
mention 300 Hz or held decisions. The implemented gateway/controller/bridge and
recorded metadata are authoritative: passive 15 sim Hz samples, source-rate arms,
50 wall Hz finger timer, no executor and no ACK endpoint.

## Explicit contracts

* 3.0.0: historical collection V3, unchanged.
* 4.0.0: historical executor V4, acceptance joins/configuration hashes unchanged.
* **4.1.0 / sharpa_decoupled_v4 / decoupled_teleoperation_v1**: sampled bounded
  operator targets plus full recorded command history. Not executor acceptance.

All retain canonical arm7/hand6/arm7/hand6, fixed bounds, units, normalization,
116D measured visual proprioception, three native RGB views, task56 sidecar and
existing FB/actor architecture. No BC, reward or loss changes. Schemas cannot be
mixed or relabeled. Different recorded physics configurations cannot silently
share one dataset/checkpoint. The new checkpoint binds architecture, robot/model
contract, timing, observation identity, data fingerprint and control configuration.
It has no execution configuration/hash or acceptance metadata.

## Conversion and causality

```bash
python tools/convert_recordings.py /path/to/new_recordings
# data/zarr_decoupled_v4/manifest.json (new, isolated destination)
python tools/convert_recordings.py /path/to/old_executor_records --contract v4
# data/zarr_15hz (unchanged)
python tools/convert_recordings.py /path/to/legacy_records --contract v3
# data/zarr (unchanged)
```

The new converter requires the recorded architecture, control metadata, source
commit, robot/model and bounds. Dirty-tree/module provenance and the entire source
metadata remain intact. Required raw/mapped/controller/finger/clock streams must
exist. Each bounded arm command must match the mapped radian source's exact header
identity and clamp; logical hand copies must match raw source headers/values.
Unverifiable leading command chains outside the bag are marked, never used to
admit an incomplete action. Interior inconsistent chains fail. Raw bridge arm
commands are preserved, but their environment-controlled unit/side transforms
cannot be certified from unrecorded overrides; the mapped radian stream is the
contract's authoritative policy source.

At each action boundary, the latest causal mapped arm/raw hand inputs validate
the exact 26D snapshot. The latest causal controller targets are separate context;
independent callbacks can make them differ. Signed boundary_controller_*_delta
arrays retain that difference. No future command is selected for an action.
Sensor selection is latest MCAP log time <= action record time, with explicit
freshness/skew limits and no pre-boundary observations. Native headers remain
separate provenance; no subtraction across clock domains. Repeated latest targets
are legitimate: source age is recorded, not replaced by a fabricated watchdog.
A sampler/recorder race without a provable causal chain fails closed.

Policy headers must be witnessed by /clock. Metadata physics Hz determines the
quantization tolerance; actual clock gaps/reset evidence splits transitions.
60/100/120/300 Hz are tested; 100 is not divisible by 15. Adjacent decisions permit
one physics-step quantization, never a missed policy period. At coarse physics rates,
intervals that could represent either quantized adjacency or a skipped decision
are excluded as unverifiable rather than inventing sampler phase. All metadata frequencies must be positive,
finite whole Hz as required by Synthetic's USD configuration. For example, 20 Hz
can retain unambiguous 50 ms intervals while excluding ambiguous 100 ms ones.
This is conservative interval admission, not a silent conversion to 300 Hz.

## Full-rate history and training approximation

Each `control_history/<stream>/` contains:

* record_ns/header_ns [M], preserving recorder and native clocks separately;
* payload uint8 and offsets [M+1]: canonical UTF-8 JSON of every decoded ROS field;
  byte fields have a base64 tag. Numeric values round-trip; this is decoded message
  preservation, not a byte-identical copy of CDR wire padding;
* interval_start/interval_stop [T] index messages in (record_t, next_record_t].

Streams include original arm/hand, mapped arms, bounded arms, logical hands,
physical fingers, clocks, sampled actions, lifecycle, recording and success.
Leading/trailing messages are retained too. Source bag metadata keeps ROS types;
MCAP hashes identify original wire data. Histories have hashes and validated
indices; their compressed bytes participate in dataset/cache fingerprints.
`intervening_arm_max_delta[T,14]` (rad) and
`intervening_hand_max_delta[T,12]` (logical units) expose changes relative to the
boundary target (including initial controller mismatch), without hiding them under a mean.
They are audit sidecars, never future policy inputs. Counts, source times, ages,
rejected decisions/intervals, boundaries and source metadata remain available.

The 26D interface is a **sampled-control approximation**, not an exact Markov or
zero-order-held executed transition: intervening human commands may affect the
next observation. Merely storing history does not remove this approximation.
Training requires explicit `allow_sampled_control_approximation: true`; the new
supplied config sets it visibly. Review target-change magnitudes before admission
for a particular experiment. No arbitrary change threshold silently filters data.
Recorded stream loss cannot be repaired or certified absent per-message sequence
IDs; the converter rejects observable provenance/clock gaps, not hypothetical ones.

```bash
python tools/train_from_zarr.py data/zarr_decoupled_v4 --device cuda
python tools/build_feature_cache.py --help
python tools/train_from_zarr.py data/zarr_decoupled_v4 --feature-cache /path/to/cache --device cuda
# Historical executor configuration remains available:
python tools/train_from_zarr.py data/zarr_15hz --config configs/fb/vision_executor_v4.yaml --device cuda
```

Replay uses actual header-time durations and measured observations; command history
is excluded from actor input. Duration-aware discount remains
`gamma ** (duration / dataset.policy_dt)` with gamma=0.993322 and policy_dt=1/15;
its per-second horizon approximates 0.99 at 10 Hz. Success metadata does not create
a terminal: recording ends/gaps truncate. No data is resampled or migrated.
Existing feature-cache preprocessing/storage is unchanged; new source fingerprints
prevent stale reuse. Existing episode-balanced, cluster and reward-weighted task-Z
algorithms accept the new references but reject cross-contract checkpoint sources.

## Runtime

Start Synthetic's policy bridge instead of collection, then:

```bash
python tools/build_task_z.py --checkpoint new.pt --references data/zarr_decoupled_v4/manifest.json --output task_z.pt --device cuda
python tools/run_policy_ros.py --checkpoint new.pt --task-z task_z.pt --mode shadow --device cuda
python tools/run_policy_ros.py --checkpoint new.pt --task-z task_z.pt --mode closed-loop --device cuda
# Historical executor runtime:
python tools/run_policy_ros.py --config configs/runtime/isaac_executor_v4.yaml --checkpoint old_v4.pt --task-z old_z.pt --mode shadow
```

New runtime mode is selected by the YAML architecture, not only version=4. It
requires a decoupled-trained checkpoint and matching declared physics Hz. Verify
the actual external simulator configuration separately; YAML is not a live physics
measurement. Only /bfm/action is published, with unique session, increasing IDs and
rational simulation scheduling. PAUSE produces no ticks; rollback stops. There is
no ACK subscriber, wait, timeout or acceptance claim. Logs say publication only.
No hidden trajectory limiter is used; finite/physical bounds, warm-up, observation
freshness (including the post-inference check), publisher protection and bounded
shutdown remain. Shadow has no control side effects. Raw direct targets differ
from the old executor trajectory, so legacy checkpoints cannot simply be reused.

## Validation limits

Synthetic snapshot metadata was inspected: 139 actions, 247 mapped/bounded arm
commands, 1,189 raw / 1,177 forwarded hands, 1,242 messages per physical hand,
554 clocks/joint states, 554 task states, 553 success messages, 138 frames/camera;
24.836 wall seconds, physics_hz=60, no execution_status. Its recording commit is
50c8432 with dirty/module checksums; no claim is made that this equals clean main.
The 387 MB MCAP could not be materialized within available disk space. Generated
CDR MCAP fixtures exercise production conversion, validation, replay, checkpoint,
task Z and a one-update mock-vision optimization smoke. This is technical pipeline
validation, not production DINO training or live ROS/Isaac validation. No production
data, model weights or datasets were changed. No full training/evaluation was run.

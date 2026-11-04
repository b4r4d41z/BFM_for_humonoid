# Sharpa collection v3 and visual Forward-Backward

Supported path: collection contract **3**, canonical Zarr schema **3.0.0**, and
explicit training profile `task56` or `vision_proprio_v1`. Collection and storage
versions describe different contracts. Collector code, topics, MCAP structure,
simulator assets, and action26 semantics are unchanged.

```text
collection v3 MCAP -> causal canonical Zarr 3.0.0 (native RGB retained)
  -> task56: privileged task state -> F / B / actor
  -> vision_proprio_v1: measured arm/hand q,dq + head/left/right RGB
     -> shared frozen DINOv2 -> camera projections + proprio MLP
     -> concat + fusion MLP -> 256D h -> F / B / actor
```

## Actions and provenance

`/bfm/action` is the model action: left arm7, left logical hand6, right arm7,
right logical hand6. Arms are bounded absolute position targets in radians,
**after** the gateway's clamp. Hands remain logical controller values 0..100.
The existing stateful hand mapping is not invertible to measured finger state.

Conversion requires collection-v3 metadata: bounded arm semantics, clamp
declaration, exact named limits, topics, model path and checksum. V2 recordings,
out-of-range canonical actions, mismatched held targets and invalid raw provenance
fail explicitly. Canonical data is never clipped or automatically migrated.
`/bfm/source/arm_joint_target_rad` is a pre-clamp request and may exceed bounds;
it is exclusively command provenance, never a model action or observation.

Frozen limits originate from model commit
`08162e1da037a2bb1827cad33e71c3e7a183de53`, path
`models/robots/s40_sharpa/assembly/s40_sharpa_two_cameras.xml`, SHA256
`4ac4549699cd3de774b676d2acaf9d0f35f1c2b041587ed7f44b535f80e43dcf`.
This is **bounds provenance**. The separate `--collection-commit` identifies the
actual recording checkout. Complete episode/rosbag metadata and MCAP hashes are
also retained. Limits are never fitted to demonstration extrema.
`ActionNormalizer` still maps bounded physical action26 to/from [-1,1]^26.

The current collector's gateway copies a raw snapshot, clamps positions, preserves
its source header in the held arm message, and creates a new BfmAction header.
The converter therefore:

1. Checks matching held arms/hands within +/-50 ms recorder time of the action.
2. Searches raw requests within +/-150 ms (one 100 ms policy tick plus 50 ms
   transport allowance), requiring both the held source header and
   `clamp(raw)==canonical arm`. This accepts the reported 94.85 ms delay.
3. Prefers preceding matching requests; later recorder timestamps are allowed
   solely for transport reordering. Records counts, signed ages and clamp events.

These command checks never supply observations and are not execution
acknowledgements. Sources older than the window fail explicitly; changing that
rule requires recorded-stream evidence. The supplied real MCAP was unavailable
here, so the association has generated-stream validation only.

## Storage and causal synchronization

Physical array layout is unchanged; storage schema remains **3.0.0** (Zarr 2):

```text
action                           float64 [T,26]
action_header_ns                  int64 [T]
record_ns, next_record_ns         int64 [T]
timestamp, next_timestamp         float64 [T] relative recorder seconds
terminated, truncated            bool [T]
segment_id                       int64 [T]
obs/ and next_obs/:
  state                          float64 [T,56]
  joint_position, joint_velocity float64 [T,J]
  joint_effort                   float64 [T,J], optional
  source_record_ns                int64 [T,5]
  source_header_ns                int64 [T,5]
  sync_age_ns                     int64 [T,5]
  images/head                     uint8 [T,H,W,3] RGB
  images/left_wrist                uint8 [T,H,W,3] RGB
  images/right_wrist               uint8 [T,H,W,3] RGB
```

Source columns are state, joints, head, left_wrist, right_wrist. Each selected
sample is the latest with `source_record_ns <= action_record_ns`, within its
configured maximum age. Native headers remain provenance; clocks are not mixed.
Invalid ticks remove both adjacent transitions. Segments never bridge missing
ticks or episode/reset boundaries. Continuous obs/next_obs must agree.

Images remain losslessly compressed RGB uint8 at capture resolution. Supported
encodings: rgb8/bgr8/rgba8/bgra8/mono8, including row padding; alpha is discarded,
mono replicated. Resizing/normalization/features exist only in the model adapter.
`attrs.profile=task56` describes the stored state column; training selects its
profile separately.

New `camera_skew_ms_p50`, `_p95`, `_max` measure max-minus-min camera record time
over unique retained ticks, including final next observations.
`camera_skew_tick_count` records the denominator. `--max-camera-skew SECONDS`
optionally rejects skewed ticks. Default is diagnostics only. Per-source age
p50/p95/max before filtering are retained.

Failed episodes are valid: success, episode identity, task and success evaluation
remain in source metadata; top-level success is also retained. Recording/gap ends
are truncations, not fabricated success terminals. Training `outcome: all` is
default; `successful`/`failed` select explicit true/false, excluding unknowns.
Filtering never changes conversion; splits remain episode-level.

Conversion/explicit validation check image continuity. Dataset construction
validates metadata, small arrays and image shapes without decoding pixels.
**task56 performs no image-pixel I/O.** Conversion still buffers one episode;
very long recordings need a future streaming converter.

## Input roles and exact proprioception

| Role | Fields / consumer |
| --- | --- |
| task56 training | All measured task56 fields, including privileged scene state |
| visual training | Named measured arm/finger q,dq and configured RGB cameras only |
| task conditioning | FB latent z inferred after training from reference rewards/goals |
| privileged evaluation | Object q/dq, goal geometry/velocities, EEF positions |
| command provenance | Raw requests, held targets, headers, clamp checks |

task56 keeps this order: left/right EEF position(3 each), left/right arm
position(7 each), left/right arm velocity(7 each), object position(3), object
velocity(3), goal center(3), half extents(3), orientation(4 xyzw), linear
velocity(3), angular velocity(3). FIELDS/SLICES in the contract are authoritative.

`vision_proprio_v1` uses **116 values**, selected by joint names from measured
JointState regardless of array ordering. Missing/duplicate joints or missing/
nonfinite velocities fail. No command-target fallback is permitted.

| Python slice | Values |
| --- | --- |
| 0:7 | left arm q: zarm_l1_joint through zarm_l7_joint |
| 7:14 | right arm q: zarm_r1_joint through zarm_r7_joint |
| 14:36 | left hand q: 22 physical joints below |
| 36:58 | right hand q: same suffix order |
| 58:65 | left arm dq |
| 65:72 | right arm dq |
| 72:94 | left hand dq |
| 94:116 | right hand dq |

Hand joint names prepend `left_` or `right_` to this frozen-model DOF order:

```text
thumb_CMC_FE, thumb_CMC_AA, thumb_MCP_FE, thumb_MCP_AA, thumb_IP,
index_MCP_FE, index_MCP_AA, index_PIP, index_DIP,
middle_MCP_FE, middle_MCP_AA, middle_PIP, middle_DIP,
ring_MCP_FE, ring_MCP_AA, ring_PIP, ring_DIP,
pinky_CMC, pinky_MCP_FE, pinky_MCP_AA, pinky_PIP, pinky_DIP
```

q is radians, dq rad/s. Training-split mean/std normalizes this vector and is
checkpointed. EEF, object, goal and logical commands are excluded. Canonical
task56 remains a reward/evaluation sidecar. Runtime task prompting uses rewards
over saved reference task states and aligned visual backward embeddings; the
visual runtime does not subscribe to current privileged object state.
A task definition/latent remains necessary; cameras do not specify intent.
Single-frame sensing does not expose all hidden contact/controller state.

## Architecture, preprocessing and FB

Production default: **facebook/dinov2-small**, via Transformers 4.46.3, one shared
pretrained backbone for all cameras and both transition endpoints. It remains in
eval mode, outside optimizers, without gradients. There is no random fallback.
`mock` is an explicit software-test fixture. Other DINOv2 artifacts can be selected
by ID or local directory; actual feature width is resolved from the model.

Preprocessing: uint8 RGB NHWC -> float /255 -> whole-frame aspect-preserving
bicubic antialiased resize -> mean-color letterbox -> ImageNet normalization,
mean [.485,.456,.406], std [.229,.224,.225]. Default 518x518 is the official DINOv2
pretrained size; configurable sizes must be multiples of patch size14. No center
crop removes workspace. Training/runtime call the same function. Torch version,
preprocessing and eager attention implementation are checkpointed and checked.

Each camera CLS feature (384D for small) has its own trainable 64D linear+tanh
projection. Proprio uses a 116 -> 64 -> 64 MLP. Three camera features plus proprio
concatenate to 256D; a 256 -> 256 -> 256 MLP produces h. MLP hidden layers use
LayerNorm+tanh. Projection/embedding dimensions are configurable. Camera order
is head, left_wrist, right_wrist; explicit subsets support ablations.

One online observation encoder produces h_t and h_next. Existing twin F, B and
actor consume h; FB losses, discounting, latent sampling and actor objective are
unchanged. FB gradients train projection/proprio/fusion. Actor updates use
detached h and update only the actor. A Polyak target of the small fusion module
accompanies target F/B. The frozen backbone/features are shared with targets:
current and next RGB features are extracted once per update.

Replay fetches only sampled RGB rows, deduplicated within a batch. Separate
tensors allow different resolutions between cameras; each camera must have
consistent resolution across episodes in a batch. Frozen inference uses
microbatches (default16). Benchmark GPU throughput before large training.
No persistent feature cache is implemented. The separate `features` API permits
a future derived cache keyed by backbone revision/weight hash, preprocessing,
feature dimension and source dataset fingerprint. Canonical RGB remains intact.

## Checkpoints and runtime

Visual checkpoints save collection3, Zarr3.0.0, profile, backbone ID/resolved remote
revision when available, architecture, full frozen weights/SHA256, Transformers
and Torch versions, frozen flag, preprocessing, camera order/projections,
proprio names/dimension, fusion dimensions, normalizer, target fusion, optimizers,
and replay/RNG state. Initial local artifacts are identified by weight hash even
when no remote revision is available. Loading/resuming needs no weight download.
Old collection-v2 checkpoints, profile/config mismatches and incompatible
software/preprocessing identities fail; weights checksums are verified.

ROS runtime selects its profile explicitly. task56 keeps /bfm/state. Vision
subscribes to measured joints and configured raw cameras only. At each 10Hz
decision it uses latest available samples in a common local monotonic receive
clock, with positive configurable freshness and optional camera skew checks.
Freshness is checked again after inference to include encoder latency. Header
clocks are not assumed comparable. Receive-age checks do not measure pre-receipt
capture/transport delay.

Actor output uses existing `ActionNormalizer.physical()` and held command
interfaces. Logical hands are rounded to the integers actually sent/recorded.
No second clamp/scaler is introduced. Stop gateway/teleoperation publishers before
model control; competing publishers fail. Stopping publication leaves previous
held targets active. Live ROS/Isaac transport and 10Hz inference remain deployment
checks.

## Commands

```bash
python -m pip install -r requirements-vision.txt
# Required by deterministic CUDA training; set before starting Python.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python tools/data_conversion/mcap_to_zarr.py /data/record_v3 \
  --output /data/episode.zarr --collection-commit RECORDING_FULL_SHA \
  --max-ages .15 .15 .15 .15 .15
python tools/data_conversion/mcap_to_zarr.py --validate /data/episode.zarr
python tools/data_conversion/mcap_to_zarr.py --manifest /data/manifest.json \
  --train /data/episode.zarr --test /data/independent_episode.zarr
python scripts/train_offline_fb.py --config configs/fb/vanilla_offline.yaml \
  --dataset-manifest /data/manifest.json --output-dir runs/fb/task56
python scripts/train_offline_fb.py --config configs/fb/vision_offline.yaml \
  --dataset-manifest /data/manifest.json --output-dir runs/fb/vision
python scripts/run_sharpa_fb.py --checkpoint runs/fb/vision/last.pt \
  --observation-profile vision_proprio_v1 --task object_to_goal \
  --device cuda --max-state-age .15
python scripts/smoke_sharpa_vision.py --episode /data/record_v3 \
  --collection-commit RECORDING_FULL_SHA --output-dir /data/visual-smoke \
  --device cuda --steps 1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests/fb -q
```

The smoke CLI writes report.json with conversion, age/skew, image, loss and
runtime-inference checks. `--mock-vision` explicitly tests software plumbing, not
pretrained perception. For offline production, use a locally supplied pretrained
DINOv2 artifact. Initial remote download requires model-host access; unit tests
never download.

Optional real-episode pytest uses `BFM_REAL_EPISODE` (defaulting to the supplied
record_20261028_141621_145555_a3fffe33 path) and requires
`BFM_REAL_COLLECTION_COMMIT` when the file exists. Absence is an explicit skip.
That test uses mock vision; run the smoke CLI without --mock-vision for a full
pretrained-backbone integration check. Neither is a policy-learning benchmark.

See [implementation_report.md](implementation_report.md) for actual checkout,
validation results and remaining unavailable checks.

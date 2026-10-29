# Real-sample BFM pipeline audit

## A. Executive result

**NO — not yet an approved, provenance-valid training sample.** The technical
data path passes after focused BFM fixes: the real MCAP converts to329
transitions in3 segments, measured visual proprioception is valid, and real-RGB
mock-backed FB training/checkpoint/inference passes. No current privileged scene
state enters the visual actor.

However, the actual recording-code revision is absent. The saved camera/asset
snapshot SHA is not proof of running recorder/controller code. The only conversion
performed here uses an explicitly diagnostic all-zero40-hex placeholder and is
**provenance-invalid**. Do not promote its manifest/checkpoints into a validated
corpus. Recover independently evidenced provenance or recollect; do not replace
the placeholder with a convenient model/camera SHA.

There is also a HIGH clock/design concern before collecting hundreds of hours:
wall-time policy ticks and physics time differ substantially in this recording.
This is not repaired by widening synchronization windows. DINOv2 pretrained
execution and live ROS/Isaac deployment remain unverified infrastructure/deployment
checks. No learning-quality claim follows from this smoke test.

## B. Exact tested repository state

- Requested branch: zarr-pipeline; starting SHA:
  `341c4bedd9e0db42f676f21bac3b724aa16dd497`.
- Isolated audit branch: `zarr-pipeline-real-audit`, repository root
  `/workspace/BFM_for_humonoid-audit`. Final changes are uncommitted against341c4be.
  No remote branch was pushed. Earlier worktrees/artifacts are preserved.
- Python3.12.14, PyTorch2.9.1+cpu, no CUDA runtime/GPU, Transformers4.46.3,
  Zarr2.18.7, NumPy2.2.6, numcodecs0.15.2, mcap-ros2-support0.5.7.
  Isolated environment: /workspace/bfm-venv.
- requirements-lock.txt contains ROS packages, CUDA13 packages and torch2.15.0;
  these are not suitable for this CPU executor. The tested environment satisfies
  requirements-fb.txt/requirements-vision.txt instead of forcing the machine lock.
- Sample: `data/current_sample/record_20261028_150427_892904_5714d4d7`.
- LFS fetched with git lfs install --local / git lfs pull; git lfs ls-files shows
  the materialized object, not a pointer. Binary MCAP magic, full SHA and CRCs checked.
- File size **1772669218 bytes**; SHA256
  `d4b7196385f50ac31f78092dd6756b7a966224aabd392a7398bebf0bdc401e48`.
- No Synthetic source or asset was modified.

## C. Raw MCAP facts

Valid MCAP, embedded ROS2 schemas decode, CRC checks pass. Duration:
**33.809824751s**, total messages:10532.
No topic has duplicate recorder timestamps or within-topic reversals.
Physical file order has128 cross-topic
log-time reversals, maximum936783ns; sorting by
recorder time is therefore correct, not an indication of future observations.

Rates below are (count−1)/(last−first); singleton rates are undefined.
Inter-message intervals are recorder-time milliseconds. Duplicates/reversals are
per topic, measured before converter sorting.

| Topic | Count | Hz | dt p50 / p95 / max ms | duplicates/reversals |
| --- | ---: | ---: | --- | --- |
| /clock | 640 | 18.913842 | 51.747486 / 59.907841 / 336.136936 | 0/0 |
| /robot/state/joint_states | 640 | 18.913865 | 51.721237 / 59.887035 / 336.09499 | 0/0 |
| /bfm/episode | 1 | — |  | 0/0 |
| /bfm/state | 640 | 19.095677 | 51.671827 / 60.948741 / 288.227411 | 0/0 |
| /left_hand_joint_command | 1691 | 50.490807 | 19.996765 / 20.122579 / 20.279968 | 0/0 |
| /right_hand_joint_command | 1691 | 50.490826 | 19.996657 / 20.130541 / 20.29153 | 0/0 |
| /bfm/source/arm_joint_target_rad | 338 | 10.087068 | 100.076506 / 103.136179 / 107.614689 | 0/0 |
| /sensors/camera/head/image_raw | 639 | 19.097923 | 51.871954 / 63.997285 / 287.621591 | 0/0 |
| /sensors/camera/right_wrist/image_raw | 639 | 19.110377 | 51.883976 / 63.96955 / 287.538335 | 0/0 |
| /sensors/camera/left_wrist/image_raw | 639 | 19.116522 | 51.795724 / 64.450488 / 277.50001 | 0/0 |
| /robot_arm_joint_command | 334 | 9.999956 | 100.001754 / 100.146653 / 100.273243 | 0/0 |
| /bfm/command/hand_target | 334 | 9.999956 | 100.001474 / 100.155791 / 100.285823 | 0/0 |
| /bfm/action | 334 | 9.999958 | 100.0012 / 100.155721 / 100.293749 | 0/0 |
| /data_collection/recording | 2 | 0.030462 | 32827.479976 / 32827.479976 / 32827.479976 | 0/0 |
| /robot/command/hand_target | 1642 | 50.002687 | 20.002953 / 21.89599 / 36.68646 | 0/0 |
| /robot/command/arm_joint_target | 328 | 9.999801 | 100.115588 / 103.207655 / 107.75866 | 0/0 |

Native timestamps prove multiple clock domains: BfmAction/source/held headers
are epoch-like; BfmState/joints/cameras are simulation-like. No cross-domain
header subtraction was used for synchronization.

| Topic | Recorder ns first … last | Native header ns min … max |
| --- | --- | --- |
| /clock | 1793199868182790000 … 1793199901967570400 | no header |
| /robot/state/joint_states | 1793199868182836000 … 1793199901967576000 | 5166666507 … 15816666603 |
| /bfm/episode | 1793199868189246000 … 1793199868189246000 | no header |
| /bfm/state | 1793199868518767400 … 1793199901981838600 | 5133333333 … 15783333333 |
| /left_hand_joint_command | 1793199868518727000 … 1793199901990166800 | 1793199868190159400 … 1793199901990103800 |
| /right_hand_joint_command | 1793199868518749200 … 1793199901990176000 | 1793199868190176300 … 1793199901990117000 |
| /bfm/source/arm_joint_target_rad | 1793199868518798300 … 1793199901927912400 | 1793199868227835100 … 1793199901927804700 |
| /sensors/camera/head/image_raw | 1793199868540317700 … 1793199901947092200 | 5150000000 … 15783333333 |
| /sensors/camera/right_wrist/image_raw | 1793199868559724000 … 1793199901944726500 | 5150000000 … 15783333333 |
| /sensors/camera/left_wrist/image_raw | 1793199868561514800 … 1793199901935786800 | 5150000000 … 15783333333 |
| /robot_arm_joint_command | 1793199868627954700 … 1793199901928101400 | 1793199868626397700 … 1793199901825854500 |
| /bfm/command/hand_target | 1793199868627967500 … 1793199901928113200 | 1793199868657488000 … 1793199901958012700 |
| /bfm/action | 1793199868627981600 … 1793199901928121600 | 1793199868627901200 … 1793199901927976400 |
| /data_collection/recording | 1793199869165134600 … 1793199901992614700 | no header |
| /robot/command/hand_target | 1793199869172951300 … 1793199901991187500 | 1793199869217887200 … 1793199902037776400 |
| /robot/command/arm_joint_target | 1793199869226983200 … 1793199901927633700 | 1793199869273134800 … 1793199901972973300 |

All camera messages are640×480 rgb8. All640 JointState messages have72 unique
names in stable order; position/velocity/effort shapes[640,72] are finite.
All58 visual-profile joints exist. Raw measured q spans−1.6117…1.572rad and raw
dq spans−40.9555…24.3533rad/s across all72 joints. These are actual physical-state
readbacks, not logical hand values. Retained visual dq reaches21.7933rad/s
(right_middle_DIP); such dynamics deserve physics/controller quality review,
not downstream clipping or target substitution. Effort is retained but not used
by either present observation profile.

BfmState:640×56, finite, header.frame_id=world. Camera frame_id is sim_camera for
all three; topic identity distinguishes cameras, but TF/calibration identity is
underspecified. JointState frame_id is empty; scalar joint-angle units are
declared by the collection contract.

Metadata: success=false, episode identity689479e7a2a3456baf6d6c1870841781,
task sorting_food, asset orange2.usd. Final success_evaluation uses
final_object_position_in_goal_obb, review_required=false, state_age_sec0.010733282.
This is an explicit failed outcome, not an unknown mapped to failure.
There is one matching playing episode event, and recording Bool true then false.
No genuine task-terminal/reset/end-reason event is in this bag.

The MCAP has two rosbag2 metadata records: an initial empty snapshot and a final
snapshot matching10532 messages and external metadata.yaml. Neither records the
running collection-code SHA. The event also lacks it. Saved camera snapshot
repository_commit=d263821ea10b2e8865c31d13ee97c24bb15254d6 has kind
saved_repository_usd_snapshot and runtime_verified=false; it cannot stand in for
the running checkout. Frozen bound source08162e1 and model SHA4ac454… are separate.

## D. MCAP → Zarr result

The **unchanged** converter passed technically before fixes, using max ages
[.15,.15,.15,.15,.15]s and diagnostic collection_commit=0000000000000000000000000000000000000000.
No contract relaxation, action clipping, fabricated measurement or future-source
selection was used. Output:
`/workspace/bfm-real-audit/provenance_invalid_technical.zarr`.
Validator passes including full RGB continuity.
Conversion+validation took10.279s, process peak RSS4648132KiB (~4.43GiB).
Schema remains3.0.0; raw physical arrays and RGB are retained.

| Field | Shape | dtype |
| --- | --- | --- |
| action | [329,26] | float64 |
| action_header_ns | [329] | int64 |
| next_obs/images/head | [329,480,640,3] | uint8 |
| next_obs/images/left_wrist | [329,480,640,3] | uint8 |
| next_obs/images/right_wrist | [329,480,640,3] | uint8 |
| next_obs/joint_effort | [329,72] | float64 |
| next_obs/joint_position | [329,72] | float64 |
| next_obs/joint_velocity | [329,72] | float64 |
| next_obs/source_header_ns | [329,5] | int64 |
| next_obs/source_record_ns | [329,5] | int64 |
| next_obs/state | [329,56] | float64 |
| next_obs/sync_age_ns | [329,5] | int64 |
| next_record_ns | [329] | int64 |
| next_timestamp | [329] | float64 |
| obs/images/head | [329,480,640,3] | uint8 |
| obs/images/left_wrist | [329,480,640,3] | uint8 |
| obs/images/right_wrist | [329,480,640,3] | uint8 |
| obs/joint_effort | [329,72] | float64 |
| obs/joint_position | [329,72] | float64 |
| obs/joint_velocity | [329,72] | float64 |
| obs/source_header_ns | [329,5] | int64 |
| obs/source_record_ns | [329,5] | int64 |
| obs/state | [329,56] | float64 |
| obs/sync_age_ns | [329,5] | int64 |
| record_ns | [329] | int64 |
| segment_id | [329] | int64 |
| terminated | [329] | bool |
| timestamp | [329] | float64 |
| truncated | [329] | bool |

Pixels were inspected throughout the raw streams and through the production
preprocessor. All639 frames per camera have unique RGB hashes; there is zero
exact frame-hash overlap between any camera pair. Contact sheets at raw frames100
and300 show three distinct viewpoints, visible fruit/robot/container and changing
content. Declared RGB decoding is consistent with these colors; no evident BGR
swap was observed. No claim of externally calibrated colorimetry/extrinsics.

| Camera | encoding | H×W×C | min…max | frame mean range | unique frames; adjacent changes |
| --- | --- | --- | --- | --- | --- |
| head | rgb8 | 480×640×3 | 3…253 | 207.576073…226.466865 | 639; 638/638 |
| right_wrist | rgb8 | 480×640×3 | 1…252 | 147.017291…240.306048 | 639; 638/638 |
| left_wrist | rgb8 | 480×640×3 | 0…252 | 143.927513…210.35677 | 639; 638/638 |

The arrays were not replaced with features. obs/next_obs image equality within
continuous segments is checked pixel-for-pixel by the validator.

## E. Synchronization report

Independently selected latest source_record_time<=action_record_time. Independent
action rows and obs/next_obs source timestamps match converted arrays **exactly**.
The Zarr action is action[i], and next observation is selected at action[i+1],
never action[i+1] as the stored action. All edges join consecutive original ticks.

| Source | All decisions age p50 / p95 / max ms | Stored obs age p50 / p95 / max ms |
| --- | --- | --- |
| state | 22.901868 / 51.619327 / 206.989381 | 22.63573 / 49.826212 / 57.252021 |
| joints | 26.791269 / 51.331861 / 220.370159 | 26.489905 / 50.07733 / 58.07025 |
| head | 24.935171 / 54.187413 / 201.559193 | 23.90883 / 52.335507 / 62.629339 |
| left_wrist | 25.075926 / 53.918668 / 192.8546 | 24.457934 / 52.162307 / 66.262888 |
| right_wrist | 25.309299 / 53.384881 / 204.2855 | 24.893105 / 52.209704 / 67.458545 |

Camera skew over unique retained ticks p50/p95/max:
**2.391814 / 44.526383 / 64.499934ms**.
Action dt p50/p95/max:**100.0012 / 100.155721 / 100.293749ms**.
There are334 decisions,2 rejected ticks (original indices20,196),332 retained
ticks,329 transitions,3 segments of19/174/136 transitions.
Edges touching either rejected tick are absent; no bridge crosses the gaps.
Every obs/next source recorder time is <= its corresponding action time.

The150ms threshold is conservative for this sample: typical p95 ages are~51–54ms;
outage ages reach193–220ms and are rejected. Retained sources peak~67.46ms.
Threshold150ms is not a guarantee of safe physical latency or generality to other
hardware. Camera skew p95~44.5ms reflects occasional different-frame selection;
a strict few-millisecond skew limit would unnecessarily discard real causal data.
Default skew diagnostics with opt-in rejection remain appropriate.

IMPORTANT physical-time distinction: /clock spans10.65 simulation seconds over
33.784780473 recorder seconds, real-time factor0.3152307001. Latest recorded /clock
increments between decisions range0…50ms, median33.333333ms, while recorder dt is
~100ms. This /clock as-of comparison is an arrival-based diagnostic, not an exact
application-time measurement. Recorded velocity units are simulation seconds.
The current FB discount uses recorder dt by its documented contract. It is coherent
for that wall-time environment, but physical horizon/controller evolution may vary
with render load. Capture/application time domains, policy/hand timer clock and
physics dt need explicit collection metadata and a coordinated design decision.
Do not silently change converter timing to unproven native-header synchronization.

## F. Action/provenance report

Every334 BfmAction has canonical26D ordering:
left arm7, left hand6, right arm7, right hand6. Joint names exactly match both
ordered arms. All canonical values are finite and bounded. Hands are0…100
(the second logical channel is99…100 in this episode).
Held named arm targets and held hand values match canonical actions at every tick.
Preserved source headers plus clamp(raw) match every canonical arm action.

Signed age is action recorder time minus matching command recorder time:

| Command | p50 / p95 / max ms |
| --- | --- |
| raw arm source | 96.103476 / 101.248182 / 102.31856 |
| held arm | 0.030303 / 0.061989 / 0.12391 |
| held hand | 0.016539 / 0.030681 / 0.09981 |

Raw minimum age:0.342761ms. No future raw sample was needed.
Existing +/-50ms held and +/-150ms raw windows pass all decisions. The held maximum
is only0.12391ms; raw maximum102.31856ms is consistent with the10Hz held-source
semantics. No window increase is justified. These checks prove request forwarding,
not an independent live drive-target acknowledgement.

155 decisions contain at least one saturation:
left arm1:15, left arm4:125, right arm4:17 (counts overlap).
Maximum overshoot: left arm1=0.189702066rad, left arm4=0.147258482rad,
right arm4=0.234699061rad; all other arm joints0.
Raw left arm4 min−1.718058482 becomes canonical lower−1.5708.
Raw right arm4 max0.234699061 becomes canonical upper0.
No measured q was replaced by a bounded target.

Bounds remain frozen at robot-model commit08162e1, XML path
models/robots/s40_sharpa/assembly/s40_sharpa_two_cameras.xml, SHA256
4ac4549699cd3de774b676d2acaf9d0f35f1c2b041587ed7f44b535f80e43dcf.
Recording-code provenance remains **unestablished**; it is a separate upstream gap.

## G. Observation/data-role matrix

Canonical per-channel ranges over all334 recorded decisions (rad for arms,
controller0..100 for hands):

| Offset | Canonical channel | Minimum | Maximum |
| --- | --- | --- | --- |
| 0 | zarm_l1_joint | -0.503145698 | 0.523599000 |
| 1 | zarm_l2_joint | -0.015339808 | 0.619728238 |
| 2 | zarm_l3_joint | -0.615126296 | 1.147417629 |
| 3 | zarm_l4_joint | -1.570800000 | -0.115048559 |
| 4 | zarm_l5_joint | -0.859029241 | 1.311553574 |
| 5 | zarm_l6_joint | -0.322135965 | 0.457126275 |
| 6 | zarm_l7_joint | -0.111980598 | 0.444854428 |
| 7 | left_hand_logical_0 | 0.000000000 | 69.000000000 |
| 8 | left_hand_logical_1 | 99.000000000 | 100.000000000 |
| 9 | left_hand_logical_2 | 0.000000000 | 42.000000000 |
| 10 | left_hand_logical_3 | 0.000000000 | 54.000000000 |
| 11 | left_hand_logical_4 | 0.000000000 | 61.000000000 |
| 12 | left_hand_logical_5 | 0.000000000 | 60.000000000 |
| 13 | zarm_r1_joint | -0.711767086 | 0.418776755 |
| 14 | zarm_r2_joint | -0.624330181 | -0.035281558 |
| 15 | zarm_r3_joint | -0.639669989 | 1.145883649 |
| 16 | zarm_r4_joint | -1.510971076 | 0.000000000 |
| 17 | zarm_r5_joint | -0.632000085 | 0.018407769 |
| 18 | zarm_r6_joint | -0.472466083 | -0.023009712 |
| 19 | zarm_r7_joint | -0.345145677 | 0.055223308 |
| 20 | right_hand_logical_0 | 0.000000000 | 69.000000000 |
| 21 | right_hand_logical_1 | 99.000000000 | 100.000000000 |
| 22 | right_hand_logical_2 | 0.000000000 | 42.000000000 |
| 23 | right_hand_logical_3 | 0.000000000 | 44.000000000 |
| 24 | right_hand_logical_4 | 0.000000000 | 61.000000000 |
| 25 | right_hand_logical_5 | 0.000000000 | 60.000000000 |


Columns trace every current model input and significant retained/excluded signal.
“forbidden” refers to the deployable visual observation, not the task56 oracle.

| Source | Zarr | Loader / replay | Model consumer | Role and permitted use |
| --- | --- | --- | --- | --- |
| /bfm/action four target groups | action[T,26] | fixed normalized action | F(h,a,z) training | action only; never proprio |
| same at following tick | defines next tick time; not stored as a_t | no shifted action | none in current edge | a_t remains aligned to o_t→o_next |
| measured JointState q | obs/next_obs joint_position | named q[58] in visual proprio | fusion, then F/B/actor | visual observation; no command substitution |
| measured JointState dq | joint_velocity | named dq[58] | fusion, then F/B/actor | visual observation, rad/s |
| measured JointState effort | optional joint_effort | not selected | none | retained physical analysis sidecar |
| leg/head q,dq | same72D arrays | excluded from116D view | none | retained; future explicit profile if movable base/head matters |
| head RGB | obs/next_obs images/head | lazy uint8 head tensor | shared backbone, head projection | visual observation |
| left wrist RGB | images/left_wrist | lazy named tensor | shared backbone, left projection | visual observation |
| right wrist RGB | images/right_wrist | lazy named tensor | shared backbone, right projection | visual observation |
| measured left/right EEF position | state slices0:6 | task56 tensor / evaluation sidecar | task56 F/B/actor; offline reach prompt | forbidden in visual observation |
| measured left/right arm q | state slices6:20 | task56 tensor | task56 F/B/actor | visual uses named JointState instead |
| measured left/right arm dq | state slices20:34 | task56 tensor | task56 F/B/actor | same distinction |
| object position | state34:37 | task56; visual evaluation_state(next) | oracle; offline reward reference | forbidden current visual input |
| object velocity | state37:40 | task56; sidecar | oracle/analysis | forbidden current visual input |
| goal center | state40:43 | task56; sidecar | oracle; offline distance prompt | task definition sidecar; not visual observation |
| goal half extents | state43:46 | task56; sidecar | oracle/evaluation | not visual observation |
| goal orientation xyzw | state46:50 | task56; sidecar | oracle/evaluation | not visual observation |
| goal linear velocity | state50:53 | task56; sidecar | oracle/evaluation | not visual observation |
| goal angular velocity | state53:56 | task56; sidecar | oracle/evaluation | not visual observation |
| next observations | next_obs same layout | next_observation dictionary/tensor | online B, target F/B, next actor | legitimate successor training input, never current deployed observation |
| source recorder stamps | source_record_ns | validation only | none | causal synchronization provenance |
| native source headers | source_header_ns | validation/analysis only | none | clock provenance, not cross-domain sync |
| source ages | sync_age_ns | validation only | none | quality metadata |
| action record/header times | record_ns/action_header_ns | timestamps vs provenance | dt for discount; header unused | recorder dt only |
| next tick time | next_record_ns/next_timestamp | duration | gamma^(duration/policy_dt) | temporal transition metadata |
| episode events/reset/gaps | source metadata; segment_id | whole segments / offsets | no policy input | boundary metadata; never join rejected ticks |
| terminated | false here | bool mask | zero bootstrap if true | no invented task terminals |
| truncated | segment ends | replay boundary | bootstrap remains enabled | file/gap interruption, not success |
| success/evaluation | attrs/source_metadata | optional manifest outcome filter | not in sampled model input | dataset selection/evaluation only |
| raw requested arm | source MCAP; clamp check summaries | not loaded | none | command provenance; unbounded legal raw requests |
| held arm / logical hand | MCAP; check counts | not loaded | none | command correspondence only |
| /robot/command arm/hand | source MCAP only | converter excludes | none | raw control debug, not measurements |
| /left_hand_joint_command,/right_hand_joint_command | source MCAP only | excluded | none | interpolated commands, never finger readback |
| /clock | source MCAP only | independent diagnostic | none currently | simulation-time evidence; coordinated clock contract needed |
| recording Bool | source MCAP only | not treated as task terminal | none | recorder-status notification; delayed true is not proof of a reset |
| task_id/instruction/robot/asset/calibration | source_metadata | identity/analysis | no language/VLM input | metadata; instruction is not automatically a supported runtime task |
| frozen bounds/model SHA | contract attrs | normalizer / checkpoint | fixed action scaling | model provenance, never recording revision |
| source MCAP hashes/collection SHA | attrs | fingerprint / provenance | compatibility only | source identity; zero diagnostic SHA is invalid provenance |
| train q/dq mean/std | checkpoint normalizer | fit training split only | visual proprio normalization | allowed offline statistics, not action extrema |
| sampled z | not a raw dataset field | uniform sphere | F/actor | reward-free training conditioning |
| reference_states/indices | checkpoint next-state sidecar | selected exact replay rows | post-training reward prompt | offline privileged task specification only |
| reference_backward | checkpoint B(h_next) | aligned saved embeddings | z=project(mean(B*r)) | same exact next states; no one-step shift |

No incorrect root/segment image assignment was found in normal flattened replay.
Tests now cover multiple roots, multiple segments, split/outcome exclusions,
unsorted indices, both endpoints, actions and evaluation sidecars.
No physical measurement is fabricated. Hand logical values appear only in action.
Current task reward/success/object coordinates never enter the visual FB update.

## H. task56 audit

Exact order: EEF left/right position(3+3), arm left/right q(7+7), arm left/right
dq(7+7), object position3, object velocity3, goal center3, half extents3,
orientation4 xyzw, linear velocity3, angular velocity3.
All raw state samples are finite; goal quaternion/unit/positive extents checks pass.
Frame is declared world by both metadata and header; object/EEF/goal values are
world-scale and visually plausible. Absolute live calibration/kinematics are not
independently certified by a bag. Arm q/dq are measured fields, not target arrays.
Object/goal state is privileged and task56 remains a separately configured oracle.
This profile intentionally omits hands, cameras, contact and object orientation;
it is not a guaranteed fully Markov simulator state.

## I. vision_proprio_v1 audit

**PASS**, exact116D q[58]+dq[58],14 arm+44 physical finger DOFs.
All names below exist in the real72-name JointState with unique stable spelling.
Name lookup reorders interleaved real source arrays; no arbitrary slice positions
are assumed. q units rad, dq rad/s are the declared producer contract; observed
values have appropriate angular structure, with high finger transients noted.
Measured state is never inferred from logical targets.
EEF/object/goal/success/effort/commands are not visual inputs.

The q ordering below is repeated in exactly the same order for dq at offsets58…115.
Ranges are over all raw640 JointStates, not just retained decision samples.

| q offset | ACTUAL recorded joint used | q min…max rad | dq min…max rad/s |
| --- | --- | --- | --- |
| 0 | zarm_l1_joint | -0.4927 … 0.5235 | -8.0445 … 3.165 |
| 1 | zarm_l2_joint | -0.016 … 0.6057 | -2.4001 … 3.7869 |
| 2 | zarm_l3_joint | -0.611 … 1.0744 | -5.142 … 6.5046 |
| 3 | zarm_l4_joint | -1.5686 … -0.1182 | -4.5598 … 4.8358 |
| 4 | zarm_l5_joint | -0.4974 … 1.3083 | -5.7972 … 2.8706 |
| 5 | zarm_l6_joint | -0.3017 … 0.4574 | -2.672 … 1.1416 |
| 6 | zarm_l7_joint | -0.106 … 0.3905 | -0.698 … 1.5563 |
| 7 | zarm_r1_joint | -0.7016 … 0.4521 | -6.5356 … 3.6245 |
| 8 | zarm_r2_joint | -0.6401 … -0.035 | -2.8221 … 3.7043 |
| 9 | zarm_r3_joint | -0.5897 … 1.1428 | -7.1555 … 3.8558 |
| 10 | zarm_r4_joint | -1.3883 … 0 | -5.6251 … 5.6146 |
| 11 | zarm_r5_joint | -0.527 … 0.0174 | -2.4726 … 1.2293 |
| 12 | zarm_r6_joint | -0.414 … 0.501 | -3.5918 … 3.5607 |
| 13 | zarm_r7_joint | -0.3414 … 0.0404 | -2.9236 … 1.4995 |
| 14 | left_thumb_CMC_FE | 1.4689 … 1.572 | -3.4525 … 0.3464 |
| 15 | left_thumb_CMC_AA | -0.3494 … -0.3489 | -1.6693 … 0.0441 |
| 16 | left_thumb_MCP_FE | -0.0001 … 0.6022 | -5.007 … 4.644 |
| 17 | left_thumb_MCP_AA | -0.016 … 0.0073 | -0.5324 … 1.9409 |
| 18 | left_thumb_IP | 0 … 0.7827 | -4.3229 … 4.0394 |
| 19 | left_index_MCP_FE | -0.0002 … 0.733 | -9.1334 … 9.1514 |
| 20 | left_index_MCP_AA | -0.0112 … 0.0144 | -0.5752 … 0.6259 |
| 21 | left_index_PIP | 0 … 0.7329 | -4.9289 … 4.9482 |
| 22 | left_index_DIP | 0 … 0.733 | -4.9272 … 4.9316 |
| 23 | left_middle_MCP_FE | -0.0003 … 0.8482 | -10.0151 … 8.859 |
| 24 | left_middle_MCP_AA | -0.0267 … 0.003 | -1.8947 … 1.1222 |
| 25 | left_middle_PIP | 0 … 0.9423 | -5.7546 … 5.0773 |
| 26 | left_middle_DIP | 0 … 0.9424 | -5.7537 … 5.436 |
| 27 | left_ring_MCP_FE | -0.0493 … 0.9582 | -10.0137 … 8.1522 |
| 28 | left_ring_MCP_AA | -0.0164 … 0.0465 | -1.9055 … 1.8454 |
| 29 | left_ring_PIP | 0 … 1.0646 | -6.7041 … 6.4634 |
| 30 | left_ring_DIP | 0 … 1.0646 | -6.7019 … 6.3887 |
| 31 | left_pinky_CMC | -0.0003 … 0.0026 | -12.2157 … 1.5306 |
| 32 | left_pinky_MCP_FE | -0.0003 … 0.9425 | -10.0133 … 8.6669 |
| 33 | left_pinky_MCP_AA | -0.0002 … 0.0056 | -0.361 … 1.1121 |
| 34 | left_pinky_PIP | 0 … 1.0471 | -6.5685 … 6.9199 |
| 35 | left_pinky_DIP | 0 … 1.0472 | -6.5665 … 7.5966 |
| 36 | right_thumb_CMC_FE | -1.6117 … -1.4357 | -2.3789 … 2.9078 |
| 37 | right_thumb_CMC_AA | 0.3485 … 0.3494 | -0.7142 … 4.8744 |
| 38 | right_thumb_MCP_FE | -0.6053 … 0.0001 | -3.9964 … 4.6971 |
| 39 | right_thumb_MCP_AA | -0.029 … 0.0268 | -0.5649 … 2.3017 |
| 40 | right_thumb_IP | -0.7812 … 0 | -3.8189 … 4.3532 |
| 41 | right_index_MCP_FE | -0.733 … 0.025 | -11.1851 … 10.6579 |
| 42 | right_index_MCP_AA | -0.0046 … 0.0546 | -3.1001 … 4.5741 |
| 43 | right_index_PIP | -0.7326 … 0.003 | -5.3837 … 5.1386 |
| 44 | right_index_DIP | -0.733 … 0.0009 | -5.1329 … 5.1375 |
| 45 | right_middle_MCP_FE | -0.6912 … 0.2647 | -16.4251 … 9.5911 |
| 46 | right_middle_MCP_AA | -0.4977 … 0.0002 | -13.5141 … 13.9668 |
| 47 | right_middle_PIP | -0.7679 … 0.2386 | -20.495 … 7.3934 |
| 48 | right_middle_DIP | -0.7679 … 0.1263 | -7.44 … 24.3533 |
| 49 | right_ring_MCP_FE | -0.9582 … 0.2099 | -18.1583 … 9.5895 |
| 50 | right_ring_MCP_AA | -0.5014 … 0.0002 | -11.2404 … 9.1287 |
| 51 | right_ring_PIP | -1.0602 … 0.0907 | -40.9555 … 18.8776 |
| 52 | right_ring_DIP | -1.0702 … 0.0579 | -6.4021 … 8.4189 |
| 53 | right_pinky_CMC | -0.3431 … 0.1095 | -12.2937 … 20.2439 |
| 54 | right_pinky_MCP_FE | -0.9425 … 0.2389 | -7.8872 … 9.5889 |
| 55 | right_pinky_MCP_AA | -0.4353 … 0.0638 | -4.2919 … 11.6355 |
| 56 | right_pinky_PIP | -1.074 … 0.1632 | -12.2837 … 6.0539 |
| 57 | right_pinky_DIP | -1.0599 … 0.1037 | -7.1494 … 6.0498 |

The14 other recorded DOFs are12 leg joints and2 head joints. They are retained in
canonical arrays but excluded by this explicit fixed-base/arm-hand profile.
A mobile base or actively moving neck may warrant a newly defined observation
profile; do not silently change116D semantics.

## J. Vision/DINO audit

Production preprocessing exercised on actual frames:
uint8 RGB NHWC /255; aspect-preserving bicubic antialiased resize of480×640 to
388×518; normalized mean-color padding65 pixels top/bottom;518×518 output.
Rounding introduces only the expected subpixel aspect approximation. No crop.
ImageNet mean/std and RGB channel order are correct; repeated identical inputs
produce identical tensors. Padding is zero in normalized coordinates.
Patch size14 divides518. Camera-major concatenation followed by equal batch split
preserves explicit head/left/right identity, even across feature microbatches.

Shared DINOv2-small architecture is384D CLS per view, frozen/no_grad/eval.
Real pretrained weights were not cached. Attempted
https://huggingface.co/facebook/dinov2-small/resolve/main/config.json failed with
proxy CONNECT403 Forbidden. **Pretrained DINOv2 smoke NOT_RUN (infrastructure)**.
The local tiny random-DINO unit fixture verifies real Transformers loading,
frozen gradients and offline restoration API, not pretrained perception quality.
The real RGB smoke uses an explicitly selected mock encoder and never presents it
as pretrained validation.

## K. FB algorithm audit

Inputs: normalized measured proprio+camera fusion h (or normalized task56 baseline),
fixed normalized bounded actions, and uniformly spherical z with norm sqrt(latent_dim).
Online F sees(h_t,a_t,z); online B sees h_next. Target policy acts on online
h_next without gradients; target F/B consume target-fusion h_next. This is
consistent with a target critic encoder and online actor. Frozen features are
computed once per endpoint and shared with the small target fusion.

The successor-measure objective is unchanged: twin F dot B, off-diagonal bootstrap
residual, immediate-successor diagonal term, and B orthogonality regularization.
Twin actor objective reduces to minimum latent Q. Discount is
gamma^(record_dt/policy_dt)*(not terminated). Real retained dt is~.1s. No task
reward, demonstrated-action BC objective, success label or object-state shortcut
is introduced into pretraining. Orthonormalization is a training objective,
not an exact finite-data whitening guarantee.

Optimizer membership: F+B+small fusion/proprio/projections in FB optimizer; actor
alone in actor optimizer; frozen backbone and targets excluded. Target gradients
remain absent. Polyak updates include target fusion/F/B. Actor gradients through
F to its actions are retained while F parameters are temporarily frozen.
A MEDIUM defect used pre-FB-step h with post-FB-step F in the actor step; patched
to recompute only small fusion from cached frozen features after the FB update.
This aligns representation versions without an extra vision forward or loss change.

Normalizer fits only training data and is checkpointed. It uses physical q/dq
for visual experiments, all56 for task56. Visual checkpoint identity, backbone
weights hash, target fusion, optimizer/replay/global RNG and data fingerprint are
checked on restore/resume. Existing exact resume-vs-uninterrupted tests pass.
The RGB fingerprint fix deliberately rejects resume from old fingerprints that
did not establish RGB identity; no silent rewrite/migration is appropriate.

## L. Privileged leakage audit

No forbidden current-state leakage found. SharpaDataset constructs visual
observation only from measured named joints; VisualReplayBuffer adds only named
RGB. sample_transitions excludes evaluation_state. FBModel requires exactly
proprio+configured camera keys. FBAgent reads only observation/next_observation,
action, terminated and duration. Current actor runtime subscribes only to visual
measurements in this profile, despite importing shared message/contract classes.
All training losses remain reward-free.

Allowed privileged uses are explicit: task56 oracle, outcome-based episode
selection, offline reference distance rewards and evaluation. The sidecar is
next_obs/state; B embeddings are computed from those same next observations.
The task z is privileged task specification, not a current object-coordinate
observation. This does not solve language-conditioned sorting or arbitrary
goal specification: the supplied instruction string is metadata, while supported
runtime tasks are object_to_goal and right_reach_object distance prompts.

## M. Reference/task-conditioning audit

Real checkpoint rows, reference_states and reference_backward were independently
reconstructed after reload and match **exactly**. reference_states is the
privileged next_obs/state sidecar at reference_indices. reference_backward is
B(encoder(next_obs at those same rows)), evaluated at checkpoint time.
No obs-vs-next_obs offset exists. Offline rewards are computed from those
reference task states; z projects their reward-weighted backward mean.
The new multi-root/multi-segment/outcome-filter test proves this alignment across
flattened indices, rather than relying only on a single recording.

A single failed trajectory provides weak task coverage and limited reference
support. A finite inferred z is not evidence of reachable goals or successful
learning; use multiple independent episodes/tasks and held-out evaluation.

## N. Runtime/action audit

Actor tanh/bounded noise emits[-1,1]^26. ActionNormalizer.physical reverses the
same frozen asymmetric bounds; no demonstration min/max or second incompatible
scaler. Arms retain left7, hand6, right7, hand6 ordering. Logical hands are
nearest-integer rounded0…100; BfmAction records the exact hand values sent.
Commands publish on /robot_arm_joint_command and /bfm/command/hand_target,
plus /bfm/action as the decision record. /bfm/action alone has no actuator effect.

Real-data offline runtime inference emits finite26D physical actions within bounds.
Static ROS audit confirms named measured subscriptions, competing-publisher check,
latest local receive-time snapshots, freshness before/after inference and optional
camera skew. No live ROS/Isaac deployment was run. Local receive age cannot bound
pre-receipt camera latency, and stopping publication leaves held targets active.
The rollout runner emits no raw-arm provenance stream; its logs are therefore not
automatically valid teleoperation collection-v3 training episodes. A separate
model-rollout provenance contract is needed before feeding such logs to this
strict converter; do not fabricate a raw teleoperation source.

## O. Performance findings

Actual native camera RGB batch32 for obs+next:176947200 bytes (168.75MiB).
Six preprocessed518 views for batch32:618218496 bytes (~589.58MiB), before model
workspace/temporaries. Lazy RGB batch loading:0.265943s.
Constructor including compressed RGB fingerprint:0.384557s for this
sample. Production preprocessing three views, batch32, one pass:
**0.774417s** on CPU.
A full batch32 **mock-backed** FB update:1.785072s.
Do not mistake mock timing for DINO timing. Three batch4 updates plus checkpoint:
2.023824s. Raw scan/CRC/pixel audit:11.992397s.

DINO feature extraction time and GPU peak memory:NOT_RUN.
At518 input,1370 tokens; eager attention microbatch16×6 heads×1370² float32 has
~687MiB for one attention-score tensor alone, plus softmax/temporary tensors,
pixels, model and other memory. Microbatch16 might fit a suitable GPU, but has not
been benchmarked; batch32 CPU visual training is unsuitable for10Hz deployment.
Start production profiling at feature microbatch1–4, then increase with measured
headroom. Keep canonical RGB unchanged.

A persistent frozen-feature cache is recommended for large repeated offline runs,
but not implemented without pretrained/GPU measurements for this immediate audit.
Design: separate derived Zarr/array store with obs/next camera features[T,3,384]
or deduplicated(camera,source_record_ns,frame_hash) rows and explicit row maps.
Key by source dataset fingerprint **including stored RGB**, camera identity/order,
backbone ID, immutable revision/weight SHA, preprocessing/Torch implementation,
image size and feature dimension/dtype. Mismatch of any key rejects/rebuilds the
cache; never silently serve stale features. Store complete provenance and atomic
completion markers. Full row-wise features here are~3.03MB vs1.82GB nominal RGB;
only fusion/F/B/actor run per update after a one-pass frozen extraction. Actual
speed benefit remains to be benchmarked, including decoding and feature I/O.

## P. BFM defects

| Severity | Root cause | Minimal fix | Verification |
| --- | --- | --- | --- |
| HIGH | Visual fingerprint hashed attrs/small arrays but not actual stored RGB; changing images with unchanged source hashes could silently resume on different inputs | Hash selected canonical RGB array metadata and compressed stored chunks for visual datasets, no pixel decompression; task56 reads no RGB bytes | image-mutation fingerprint test; existing exact resume and task56 no-pixel-I/O tests |
| MEDIUM | Actor used h computed before FB changed fusion, against F after the same update | Recompute current small fusion from cached frozen features, then detach for actor; two backbone calls per transition batch remain | post-step embedding equality, feature-call count and finite real update |
| LOW | Reversed NumPy index views had negative strides and failed torch indexing | Make flat replay indices contiguous | unsorted/reversed multi-store multi-segment test |
| LOW | Optional integration test defaulted to an old external episode and skipped despite the tracked real sample being present | Resolve the tracked episode relative to this repository; explicitly skip unmaterialized LFS pointers, and still require recording commit input | final real-episode test and default-path execution |

Patched files: bfm/data/sharpa.py, bfm/data/visual_replay.py, bfm/fb/agent.py;
new tests/fb/test_real_audit.py, and real-sample discovery in tests/fb/test_vision.py. Documentation includes this report and the
Synthetic prompt. No source-side defect was hidden by these fixes. All production
action, temporal and measurement validation contracts remain intact.
No normal root-index, segment-offset or reference one-step bug was found.

## Q. Synthetic-side required changes

1. **BLOCKER for validated corpus provenance:** no actual recording-code SHA in
   episode.json, external/final MCAP rosbag metadata or episode event. The camera
   snapshot SHA cannot prove runtime code. BFM cannot recover it downstream.
   Record clean/dirty/build identity at recording start and in a recorded
   provenance event; unknown must remain explicit. Acceptance: sidecar/event
   agree, installed/source/dirty/missing-Git cases covered, old recordings never
   relabeled without external evidence.
2. **MEDIUM lifecycle deficiency:** one playing event; no task-end reason/event,
   despite final sidecar success=false and recording Bool false. Current BFM
   correctly truncates/bootstraps; it cannot infer genuine task terminal timing.
   Record explicit identity, reason, terminal-vs-interruption and true/false/unknown
   outcome with evaluation semantics and reliable stop/flush ordering.
3. **LOW camera frame/calibration metadata:** all three headers sim_camera,
   saved frame=null/runtime_verified=false. Topic-based BFM view identity works;
   unique optical frames and runtime/configured calibration distinction are
   needed for TF/geometry. Acceptance: no duplicated view assignment, frames and
   topic roles agree, no false claim of physical/live calibration.

These are specified in the standalone ready-to-paste
[PROMPT FOR SYNTHETIC CODEX SESSION](synthetic_codex_prompt.md), including exact
sample evidence, likely producer files, acceptance criteria and regression tests.
No Synthetic file was edited.

## R. Contract/design concerns

- **HIGH:** explicitly define wall versus simulation policy/controller time,
  application/capture timestamps, physics dt and reset identity. Evidence and
  acceptance experiments are in sectionE and the Synthetic prompt. Current
  recorder causality is correct; physical-time deployment equivalence is unproven.
- **HIGH for deployment:** receive-time freshness does not establish capture-time
  freshness. Requires a declared observable source clock/application contract,
  not a guessed cross-domain header subtraction.
- **MEDIUM:** the final COM-in-OBB predicate is narrower than settled whole-object
  containment/grasp success. Keep the recorded false label as-is; do not
  reinterpret it as a general benchmark result.
- **MEDIUM:** model-controlled rollout recordings need controller/provenance
  identity instead of pretending to be ExoSuit raw requests.
- **INFO:** single-frame116D is partially observed; contact/controller internals,
  possible future head/base motion and task/context coverage warrant separately
  named experiments, not silent changes to this profile.

## S. Tests actually run

Baseline unmodified341c4be:
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HOME=/tmp/bfm-audit-hf PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /workspace/bfm-venv/bin/python -m pytest tests/fb -q`
→65 passed,1 skipped (the historical default real-episode path differs from the
new tracked sample; final test discovery now defaults to the tracked sample).

Final FB and full-suite commands use this exact prefix (the zero SHA marks a
technical provenance-invalid run, **not** validated recording provenance):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HOME=/tmp/bfm-audit-hf \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
BFM_REAL_EPISODE=/workspace/BFM_for_humonoid-audit/data/current_sample/record_20261028_150427_892904_5714d4d7 \
BFM_REAL_COLLECTION_COMMIT=0000000000000000000000000000000000000000 \
/workspace/bfm-venv/bin/python -m pytest tests/fb -q
# 69 passed in24.35s, including real MCAP conversion and real-RGB mock FB.
# Same prefix with: python -m pytest -q
# 81 passed,1 failed in25.26s.
```

The full-suite failure is unchanged legacy
tests/test_temporal_contract.py::test_future_targets_are_timestamp_based_half_second_and_do_not_cross_done:
0.4s error exceeds0.3s tolerance. It is outside the current Sharpa/FB path.
Targeted new regressions:3 passed. Python compilation and git diff --check pass.
The same legacy failure was independently reproduced on a clean341c4be worktree
(1 failed in1.02s). The corrected default tracked-sample discovery was also run
without BFM_REAL_EPISODE, using the same explicitly diagnostic zero SHA.
The independent scan uses make_reader(validate_crcs=True, log_time_order=False),
separate manual canonical/held/raw comparisons and searchsorted causal selection.
The actual converter and its complete pixel-continuity validator both ran.
Audit scripts/logs and detailed evidence are delivered separately from the source
patch; datasets, MCAP, model weights and caches are excluded from that patch.

## T. Real visual smoke result

**Mock visual plumbing:PASS on real RGB and measured proprio.**
329 transitions/3 segments; production116/3-camera adapter;518 preprocessing;
three batch4 FB optimizer updates;256D fusion/heads; checkpoint reload;
exact reference reward-state/visual-B alignment; finite bounded26D action.
Final FB loss-1.2132781744003296, actor loss-0.1851268857717514.
Separate batch32 check confirms fusion parameter changes, no target gradients and
frozen-backbone semantics. The mock has no trainable backbone parameters; the
local DINO API test additionally verifies frozen real-module parameters.

**Actual pretrained DINOv2 integration:NOT_RUN**, proxy403 and no cached weights.
**Live ROS/Isaac:NOT_RUN.**
Checkpoint identities restore exactly. These results demonstrate software/data
plumbing, not successful manipulation or policy learning. Every real smoke
artifact is provenance-invalid pending recording-code evidence.

## U. Remaining blockers before large-scale training

Establish actual recorder/controller code provenance, or recollect with the
requested immutable metadata. Resolve physical-clock/control-rate expectations
before comparing simulation loads or real-time deployment. Obtain real pretrained
DINO weights and benchmark GPU training/inference, freshness and publisher
exclusivity on the live simulator. Validate multiple independent episodes and
outcome/task coverage; one failed trajectory is not a scientific benchmark.
Consider measured finger-speed outliers, reset settling and realistic physics
quality before collecting hundreds of hours; do not silently filter or clamp them.

## V. Recommended next actions in priority order

1. Run the supplied Synthetic prompt: implement provenance capture, investigate
   clock semantics under multiple loads, and record a new independently traceable bag.
2. Apply the BFM patch relative to341c4be. Start a new run for the corrected visual
   fingerprint; do not silently migrate old incomplete input fingerprints.
3. Rerun real conversion and the same alignment/leakage tests with actual
   recording SHA and a production provenance-valid manifest.
4. Supply pretrained DINOv2 weights locally or configure model-host access; profile
   microbatch1–4 and real GPU memory/latency before selecting batch32/microbatch16.
5. Implement the derived frozen-feature cache only after a measured bottleneck;
   preserve canonical RGB and full invalidation keys.
6. Establish multi-episode train/held-out splits, terminal/evaluation semantics,
   task prompts and live rollout provenance before reporting learning results.

## Appendix: actual transition examples

Each JSON entry prints a_t, action recorder time, selected current sources,
following action time and selected next sources. Held commands match the printed
a_t by the independent header/value checks in sectionF. Gap edges are absent:
row18 ends segment0, row19 begins segment1; row192 ends segment1 and row193 begins
segment2. See the original indices in the separate technical evidence JSON.

```json
{
  "row": 0,
  "original_action_index": 0,
  "record_ns": 1793199868627981600,
  "next_record_ns": 1793199868728085500,
  "obs_source_record_ns": [
    1793199868586554400,
    1793199868617759200,
    1793199868601657000,
    1793199868590296300,
    1793199868590249200
  ],
  "next_source_record_ns": [
    1793199868678642200,
    1793199868709940700,
    1793199868693648400,
    1793199868682114300,
    1793199868680723700
  ],
  "action": [
    0.12885438618239387,
    -0.015339807878856412,
    0.3236699462438703,
    -0.2699806186678729,
    -0.14726215563702155,
    0.06596117387908257,
    -0.07516505860639641,
    0,
    100,
    0,
    0,
    0,
    0,
    -0.059825250727540004,
    -0.035281558121369745,
    0.1227184630308513,
    0,
    -0.09357282806102411,
    -0.1518640980006785,
    -0.007669903939428206,
    0,
    100,
    0,
    0,
    0,
    0
  ],
  "segment_id": 0,
  "truncated": false
}
```

```json
{
  "row": 100,
  "original_action_index": 102,
  "record_ns": 1793199878828093200,
  "next_record_ns": 1793199878928056600,
  "obs_source_record_ns": [
    1793199878784304600,
    1793199878816462800,
    1793199878789166600,
    1793199878787979500,
    1793199878786780700
  ],
  "next_source_record_ns": [
    1793199878883377400,
    1793199878914281000,
    1793199878889060400,
    1793199878887877000,
    1793199878886683600
  ],
  "action": [
    -0.010737865515199488,
    0.35128160042581186,
    -0.17640779060684875,
    -1.5708,
    -0.07976700097005335,
    0.009203884727313847,
    0.035281558121369745,
    0,
    100,
    0,
    0,
    0,
    0,
    -0.3574175235773544,
    -0.18407769454627695,
    0.504679679214376,
    -0.44792239006260726,
    0.01380582709097077,
    -0.23776702212227438,
    -0.24697090684958822,
    0,
    100,
    0,
    0,
    0,
    0
  ],
  "segment_id": 1,
  "truncated": false
}
```

```json
{
  "row": 328,
  "original_action_index": 332,
  "record_ns": 1793199901827983400,
  "next_record_ns": 1793199901928121600,
  "obs_source_record_ns": [
    1793199901781227500,
    1793199901816920800,
    1793199901785540900,
    1793199901784355800,
    1793199901783163100
  ],
  "next_source_record_ns": [
    1793199901881636600,
    1793199901917787400,
    1793199901886474200,
    1793199901885287200,
    1793199901884094700
  ],
  "action": [
    0.15339807878856412,
    -0.0046019423636569235,
    0.483203948183977,
    -0.11811652066719437,
    -0.31753402309232776,
    -0.21015536794033285,
    0.15493205957644976,
    0,
    100,
    0,
    0,
    0,
    0,
    -0.0030679615757712823,
    -0.06902913545485385,
    0.2761165418194154,
    0,
    -0.2699806186678729,
    -0.2761165418194154,
    -0.20401944478879028,
    0,
    100,
    0,
    0,
    0,
    0
  ],
  "segment_id": 2,
  "truncated": true
}
```

## Appendix: all recorded joint names

```text
leg_l1_joint
leg_r1_joint
zarm_l1_joint
zarm_r1_joint
zhead_1_joint
leg_l2_joint
leg_r2_joint
zarm_l2_joint
zarm_r2_joint
zhead_2_joint
leg_l3_joint
leg_r3_joint
zarm_l3_joint
zarm_r3_joint
leg_l4_joint
leg_r4_joint
zarm_l4_joint
zarm_r4_joint
leg_l5_joint
leg_r5_joint
zarm_l5_joint
zarm_r5_joint
leg_l6_joint
leg_r6_joint
zarm_l6_joint
zarm_r6_joint
zarm_l7_joint
zarm_r7_joint
left_thumb_CMC_FE
left_index_MCP_FE
left_middle_MCP_FE
left_ring_MCP_FE
left_pinky_CMC
right_thumb_CMC_FE
right_index_MCP_FE
right_middle_MCP_FE
right_ring_MCP_FE
right_pinky_CMC
left_thumb_CMC_AA
left_index_MCP_AA
left_middle_MCP_AA
left_ring_MCP_AA
left_pinky_MCP_FE
right_thumb_CMC_AA
right_index_MCP_AA
right_middle_MCP_AA
right_ring_MCP_AA
right_pinky_MCP_FE
left_thumb_MCP_FE
left_index_PIP
left_middle_PIP
left_ring_PIP
left_pinky_MCP_AA
right_thumb_MCP_FE
right_index_PIP
right_middle_PIP
right_ring_PIP
right_pinky_MCP_AA
left_thumb_MCP_AA
left_index_DIP
left_middle_DIP
left_ring_DIP
left_pinky_PIP
right_thumb_MCP_AA
right_index_DIP
right_middle_DIP
right_ring_DIP
right_pinky_PIP
left_thumb_IP
left_pinky_DIP
right_thumb_IP
right_pinky_DIP
```

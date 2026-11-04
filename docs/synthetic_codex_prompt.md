# PROMPT FOR SYNTHETIC CODEX SESSION

Work only in `b4r4d41z/Synthetic_data_collection_with_ExoSuit`. Inspect the actual
current branch before editing. This request comes from an independent end-to-end
audit of the BFM repository at `341c4bedd9e0db42f676f21bac3b724aa16dd497`.
Do not modify BFM in this session. Keep canonical collection-v3 action26 semantics
and existing topics unless a separately documented, coordinated contract change
is necessary. Do not fabricate provenance for historical recordings.

## Evidence you must preserve and reproduce

The real episode is `record_20261028_150427_892904_5714d4d7`, identity
`689479e7a2a3456baf6d6c1870841781`. BFM tracks it under
`data/current_sample/record_20261028_150427_892904_5714d4d7` through Git LFS.
MCAP size: **1,772,669,218 bytes**. SHA256:
`d4b7196385f50ac31f78092dd6756b7a966224aabd392a7398bebf0bdc401e48`.
The file is valid MCAP and CRC checks pass; duration33.809824751s,10532 messages.

Observed streams:

- `/bfm/action`, `/robot_arm_joint_command`, `/bfm/command/hand_target`:334 each.
- `/bfm/source/arm_joint_target_rad`:338.
- `/bfm/state`, `/robot/state/joint_states`, `/clock`:640 each.
- Each of head/left_wrist/right_wrist RGB streams:639, rgb8,640x480.
- `/bfm/episode`:one `playing` event with the matching episode identity.
- `/data_collection/recording`:true then false; no task-terminal event is recorded.

All334 canonical actions are bounded and match held arms/hands. Source headers
associate correctly with held commands, and clamp(raw)==canonical for every tick.
Raw age p50/p95/max:96.1034755/101.24818205/102.31856ms.155 ticks saturate:
left arm1:15, left arm4:125, right arm4:17 (joint counts can overlap).
All58 expected measured arm/finger DOFs, q and dq, exist and are finite within the
72-joint JointState. Three camera streams are distinct and change every frame.
Do not break these working contracts or replace measurements with commands.

The BFM converter, using causal recorder-time selection and150ms age thresholds,
retains332 decision ticks and329 transitions in3 segments (19/174/136 transitions).
It rejects action ticks20 and196 for stale sources. No future observations are
used. Camera recorder-time skew p50/p95/max:2.391814/44.5263827/64.499934ms.
Do not tune producer behavior solely to make BFM tests pass.

## 1. BLOCKER: record actual collection-code provenance

`episode.json.recording` contains storage_id, bag_directory and topics, but no
recording-code SHA. The only relevant repository_commit is under
`camera_configuration`, whose kind is `saved_repository_usd_snapshot` and whose
notes explicitly describe saved assets/calibration provenance. That SHA is
`d263821ea10b2e8865c31d13ee97c24bb15254d6`; it does not prove the recorder's checkout.
Other embedded SHAs are asset, bounds or scene-calibration revisions.
Both MCAP rosbag2 metadata records and the single `/bfm/episode` event also lack
the collection checkout SHA. BFM therefore cannot recover actual recording
provenance from this sample. Its technical audit used an explicitly invalid
all-zero diagnostic SHA, never a substitute bounds/camera revision.

Inspect the current metadata producer, likely
`exo2isaacsim/src/isaac_eff_control/isaac_eff_control/sim_recorder.py`
(`episode_metadata` / `start_recording`), and `bfm_contract.py` / episode events.
Snapshot actual software provenance **before** starting rosbag. Define a clear
recording provenance object including repository URL, full40-hex HEAD, dirty flag,
dirty-state diff/content digest when applicable, and installed/build artifact
identity when running copied/installed ROS packages. Preserve asset/model hashes
separately. Never infer runtime revision from the camera JSON or unrelated cwd.
Do not capture credentials, environment dumps or secrets.

Where a source Git checkout is unavailable, require an explicit build provenance
artifact generated from the real checkout/build. Represent unknown provenance
explicitly and fail the strict reproducible-collection path, rather than inventing
a SHA. Redundantly preserve the same identity in recorded MCAP metadata or a
recorded episode/provenance event. Verify that the event is recorded, including
startup ordering/transient-local delivery. Do not pretend existing bags are fixed
unless external evidence actually establishes their original source revision.

Acceptance/regression tests: clean checkout, dirty checkout, installed package,
missing Git, wrong cwd, inconsistent producer identities, and complete MCAP/sidecar
round-trip. Sidecar and recorded identities must agree and survive conversion.

## 2. HIGH coordinated concern: specify control, simulation and recorder clocks

Actual `/clock` spans10.65 simulation seconds over33.784780473 recorder seconds,
real-time factor0.3152307001. `/bfm/action` is10Hz in recorder time (dt median
100.0012ms), while the latest recorded `/clock` advances by median33.333333ms per
decision, with observed min0 and max50ms. This is an arrival-based diagnostic,
not exact controller-application timing. Nevertheless the sample plainly ran
slower than real time. Physics velocities use simulation seconds; recorder
durations use wall seconds. BFM currently discounts using wall dt by its existing
contract, which does not guarantee matching physical dynamics across machine loads.

Native action/source/held headers are epoch-like; state, measured joints and
cameras are simulation-like. Do not compare those domains directly. Investigate
policy timer clock, hand-controller timer clock, physics time step/substeps,
pause/reset behavior, real-time factor and message buffering. Document whether
the intended policy interval is0.1 wall seconds or0.1 simulation seconds. Do not
silently change timer behavior or change canonical action semantics in this fix.

Add explicit clock-domain/unit metadata, simulation episode/reset identity,
physics dt and policy/hand-controller clock choices. If feasible, record the
simulation/control application time of each canonical action and source capture
times in a common declared simulation domain, separately from rosbag log time.
Preserve source-header association and raw/held/canonical equality. Explain any
new application-time signal and coordinate its consumption with BFM separately.

Acceptance tests: run the same seeded control schedule under at least two render/
compute loads and a pause/reset. Show policy application intervals in the declared
clock, correct measured velocity units, reset isolation and source/canonical
identity. Recorder-time causal conversion must remain available; no future-source
selection is permitted to compensate for buffering. Report measured timing before
choosing enforcement thresholds or a contract-version change.

## 3. MEDIUM: explicit end reason and outcome event

The file has one playing event but no terminal, truncation/end reason, stop
acknowledgement event or recorded outcome. Sidecar success=false is valid and
has a fresh final-object-in-goal-OBB evaluation (`review_required=false`), evaluated
at2026-10-28T15:05:01.992737+00:00. That predicate is narrower than grasp stability,
settled containment or general task success. Preserve its exact meaning.

Extend the episode lifecycle contract with a recorded end event containing the
same identity, reason, terminal-vs-interruption meaning, relevant timestamps,
success true/false/unknown and evaluation method/status. Make stop/event/flush
ordering reliable. Operator stop and transport gaps are truncations; do not label
them task terminals. Unknown success must remain unknown. Distinguish final COM
inside a calibrated OBB from whole-object containment and settled success.

Acceptance tests: normal failure, success, unknown evaluation, operator stop,
crash/interrupted recorder, reset barrier and final-event delivery. BFM should
bootstrap truncations and mask only genuine task terminals once the contract
supports them. A producer event addition must not fabricate labels for old data.

## 4. LOW: camera/source frame identity and quality diagnostics

All three actual camera headers use `sim_camera`; topic identities are distinct,
so current BFM camera-aware fusion is correct, but TF/extrinsic interpretation is
ambiguous. Saved metadata frame fields are null and runtime_verified=false.
Record unique, documented camera optical-frame identities and runtime/configured
calibration provenance separately. Verify actual640x480 RGB encoding and each
frame's camera role. Do not claim physical calibration from saved USD optics.

Report source gaps (actual camera max inter-message gaps277.5–287.6ms, measured
joint gap336.095ms), capture/receipt latency where clocks permit, and camera
capture skew. Keep strictness based on measurements. Include frame-identity,
camera-duplication, empty-frame and RGB encoding regression tests.

## Deliverables

Implement minimal Synthetic fixes for provenance and lifecycle metadata, add
tests, and separately document the clock-contract investigation and any required
coordinated BFM work. Preserve the working collection-v3 bounded action pipeline,
measured physical joints and canonical raw RGB. Provide one clean source patch,
exact tested SHA/worktree, commands/results, a new real recording with verified
provenance, and a report that clearly separates implemented changes from measured
or unverified behavior. Do not claim learning quality from an integration smoke.

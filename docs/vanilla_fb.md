# Vanilla Forward-Backward BFM

This implementation is reward-free Vanilla FB, not the existing supervised
`bc.policy.FBCPRAgent`. It supports measured KUAVO transitions, offline training,
complete-state and reward prompts, and autonomous policy execution through the
existing Isaac interface. No CPR discriminator, BC loss, advantage weighting,
vision model, or scripted task controller is used.

## Repository audit and implementation checklist

Development base: `origin/zarr-pipeline`,
`d7b7e52d2ab33bc70eb8059d3c2fcbc09b7aae2f`. The original checkout was clean at
`e5463b2a47b74c927a89a54cc3dba068366c5e6f` before fetching/switching branches.

- [x] Inspect `TestTrainingEnv`, action application, robot configuration and joint mapping.
- [x] Inspect existing data conversion, replay, BC objectives, runtime, and FB stubs.
- [x] Read the official Meta Motivo basic FB implementation and original authors' FB agent; check licenses.
- [x] Implement separate measured schema-v2 contracts, recorder, validation and replay.
- [x] Implement B, twin F, latent actor, target updates, genuine FB objective.
- [x] Implement offline training, logging, complete checkpoints and exact CPU resume.
- [x] Implement goal/reward prompting, measured runtime and closed-loop evaluation.
- [x] Run analytic, data, algorithm, CLI and mocked-runtime tests.
- [ ] Validate real KUAVO data and physics rollouts (external prerequisites below).

Audit findings:

* `TestTrainingEnv` controls named right then left arm joints with absolute targets;
  `_apply_action` clips to articulation limits and maps two six-value hand
  prototypes into binary claw positions. The 26D policy observation combines
  measured arm angles with commanded hand prototypes. It is not the new FB state.
* `tools/data_conversion/bag_to_zarr.py` explicitly declares
  `state_source=recorded_command_proxy`, `measured_robot_state=False`. Its 26D
  Zarr v1 files cannot supply measured physical transitions. They are rejected,
  not converted into invented observations.
* `bc/buffers/offline_buffer.py` and the existing policy are tied to BC's 26D
  HDF5/action contract. The new small-state replay has independent dimensions.
* `bfm/fb_cpr/` and `scripts/bfm/` contain empty stubs on this base. They are left
  untouched and are not the implementation documented here.
* The existing `RolloutRecorder` stores actions and observation shapes, not all
  measured signals. `BFMRecorder` is an opt-in direct-export path.
* Reused infrastructure: scene/control code, canonical joint names verified
  against `configs/robot_mappings/kuavo_real_to_isaac_arm14.json`, hand prototypes
  and environment claw conversion, `create_isaaclab_env`, and `PolicyScheduler`.
  The Zarr writer follows the existing staged-write design, adds rollback on
  failed replacement, and avoids importing the old dimension-specific schema.
* The only simulator change is a pre-reset observer hook in `_get_dones`.
  Existing scene, actions, rewards and observations are unchanged.

No thesis proposal or the named Russian technical audit was present in the
provided workspace; neither is claimed as a source.

## Modules and dependencies

```text
TestTrainingEnv + IsaacTransitionBridge (measured signals before auto-reset)
  -> BFMRecorder -> bfm/data schema-v2 Zarr -> BFMDataset -> BFMReplayBuffer
  -> bfm/fb/FBAgent (B, twin F, actor, target B/F)
  -> complete checkpoint + fixed training-only reference states
  -> context.infer_task -> BFMPolicyRunner -> BFMActionAdapter -> TestTrainingEnv
```

`bfm/contracts/kuavo.py` defines the physical contract and deterministic adapters.
`bfm/data/` owns validation, atomic episode writing, manifests, replay and fixed
normalization. `bfm/fb/` owns the algorithm, inference and training. Runtime
modules do not import the BC trainer or policy. Existing wrapper exports are
lazy so importing the FB runner does not initialize BC components.

## Exact algorithm

For latent z, learn the discounted successor density relative to the next-state
replay distribution rho:

```text
M_z(s,a,s') = sum_{t>=0} gamma^t P(s_{t+1}=s' | s0=s,a0=a,pi_z) / rho(s')
M_z(s,a,s') ~= F(s,a,z)^T B(s')
Q_z(s,a) = F(s,a,z)^T z
```

For an iid replay batch `(s_i,a_i,s+_i,terminated_i,duration_i)`, define:

```text
d_i = gamma^(duration_i / policy_dt) * (1 - terminated_i)
a+_i = bounded_Gaussian(actor(s+_i,z_i), std=.2, noise_clip=.3)
M^k_ij = F_k(s_i,a_i,z_i)^T B(s+_j)
T_ij = mean_k [target_F_k(s+_i,a+_i,z_i)^T target_B(s+_j)]
D^k_ij = M^k_ij - stop_gradient(d_i T_ij)

L_off = 1/2 sum_k mean_{i != j} (D^k_ij)^2
L_diag = -sum_k mean_i D^k_ii
C_ij = B(s+_i)^T B(s+_j)
L_orth = 1/2 mean_{i != j} C_ij^2 - mean_i C_ii
L_FB = L_off + L_diag + ortho_coef * L_orth
```

Only the matching column `j=i` is the immediate successor. Other rows' iid
next states form independent reference columns; an unrelated reference bank
must never replace the diagonal. Repeated samples are allowed. Batch size must
be at least two. The detached target on the diagonal matches Meta Motivo's
reported scalar; omitting that detached constant gives identical gradients.
`L_orth` estimates `0.5 ||E[B B^T]-I||_F^2` up to an additive constant.
Negative losses are expected; these are not ordinary nonnegative MSE reports.

Target pessimism is configurable as `mean - p*abs(twin1-twin2)`; the default
`p=0` matches the basic Meta Motivo FB target. The actor always uses the twin
minimum:

```text
a_i = bounded_Gaussian(actor(s_i,z_i))
L_actor = -mean_i min_k [F_k(s_i,a_i,z_i)^T z_i]
target_theta <- (1-tau)*target_theta + tau*theta
```

Representation parameters are frozen during the actor update, while gradients
still flow through F's action input. There is no demonstrated-action objective,
entropy objective, target actor, or auxiliary reward/Q loss. Target F/B are
permanently excluded from autograd and updated after each FB+actor step.
Reported gradient norms are measured before clipping.

Latents are Gaussian directions projected to norm `sqrt(latent_dim)`:
`project_z(x)=sqrt(d)*x/||x||`. All training, goal and reward prompts use this
convention. Zero-norm prompts raise an explicit error rather than silently
inventing a task direction. Training samples only uniform spherical latents.

Post-training prompting uses:

```text
z_goal = project_z(B(full_goal_state))
z_reward = project_z(mean_reference [B(s) * r(s)])
```

The reference set is deterministically sampled **only from training next
states** and saved in the checkpoint. It is not recomputed each control step.
No inverse covariance or reward-weighted actor training is introduced.
Reach reward is negative measured EEF-to-ball distance; ball-to-container reward
is negative measured ball-to-target distance. These rewards never enter FB
pretraining. Complete-state goal prompts must supply every required field.

Intentional differences from Meta Motivo defaults: compact configurable MLPs
with LayerNorm/Tanh hidden layers; fixed training-only state statistics instead
of running BatchNorm; an unconstrained backward MLP (the reference supports
`norm=False`) with explicit orthogonality regularization; uniform rather than
mixed goal/uniform training latents; no optional Q auxiliary loss. The actor's
tanh mean, clipped Gaussian noise and straight-through boundary gradient follow
the reference update convention. These are Vanilla FB choices, not FB-CPR or
FB-AW.

## Measured state and executed action contract

The first model view is **28D state / 8D action**. Dimensions are derived from
named fields and validated; state and action dimensions are never equated.

| Field | Width | Unit / meaning |
|---|---:|---|
| right_arm_q | 7 | measured rad, `zarm_r1_joint` … `zarm_r7_joint` |
| right_arm_dq | 7 | measured rad/s, same order |
| right_gripper_q | 2 | measured rad, right finger-left then finger-right joint |
| right_eef_position | 3 | m, named body origin plus explicit TCP offset |
| red_ball_position | 3 | m, measured object center |
| red_ball_velocity | 3 | m/s, measured linear velocity |
| container_position | 3 | m, measured base center + fixed target offset |

All positions are in the environment-local frame (world minus environment
origin). Body-local TCP offsets are rotated using Isaac's wxyz quaternion.
`--eef-body` is required when collecting, is checked against actual articulation
body names and is persisted. `zarm_r7_link` exists in the checked-in URDF, but
the absent USD's body names still require runtime validation. Zero TCP offset
means body origin, **not an asserted fingertip location**. The container target
is base center plus half its thickness plus ball radius, putting the desired
ball center above the interior floor. This geometry is checked on runtime load.

Canonical storage additionally retains `all_joint_q`, `all_joint_dq` and their
complete name order when exported by the Isaac recorder. It does not replace
finger measurements with the old hand command prototypes.

Physical actions are seven absolute right-arm joint targets in radians and
one executed claw command (`-1=open`, `+1=closed`). Bounds come from the loaded
articulation's soft limits, not guessed constants. Normalized arm actions use
`a_norm=2*(q-low)/(high-low)-1`. The actor's eighth output is thresholded at zero
because this environment implements a binary claw. The recorder stores the
**executed endpoint**, not the unexecuted continuous scalar. This binary bridge
is an offline extrapolation risk and is explicit in the data/action contract.

The 26D environment action contains named right/left arm targets in the
environment's configured order, followed by left and right six-value hand
prototypes. The left arm holds an explicit saved seven-angle pose (default:
articulation default pose); the left claw is explicitly open by default. Limits,
joint names, hold pose, hand state, TCP and timestep configuration are checked.
The hold configuration still needs physical clearance validation in the real
scene; CPU mocks cannot certify collision-free operation.

## Zarr schema v2 and manifests

Install `requirements-fb.txt` with a compatible PyTorch build. Storage requires
Zarr **2.x**; schema version `2.0.0` is a separate concept from storage format.
Each `.zarr` directory contains one complete episode:

```text
obs/<field>          [T,width]   measured state before each applied action
next_obs/<field>     [T,width]   measured state after the held action, before reset
action              [T,8]       executed physical action (rad + signed binary claw)
isaac_action        [T,26]      actual clipped targets / hand prototypes
terminated          [T] bool   true MDP termination; disables bootstrapping
truncated           [T] bool   timeout/recording cutoff; preserves bootstrapping
timestamp           [T]        seconds since episode control-step zero
next_timestamp      [T]        seconds at the next measured state
step_index          [T] int    initial Isaac control-step index
next_step_index     [T] int    final Isaac control-step index
```

Attributes include schema name/version, episode ID, explicit split, provenance,
initial measured scene state, coordinate frame, field widths/units, joint and
action contract, and terminal semantics. Only the last transition ends an
episode. Every nonfinal next observation must match the following observation,
including extra measured fields. The final next observation is never a reset
state. Nonfinite values, missing fields, changed joint order, invalid action
bounds, unsupported versions, discontinuities and mismatched executed Isaac
targets are errors. Metadata cannot by itself prove physical provenance;
the direct recorder is the supported source of genuine simulator measurements.

At the default 120 Hz physics / 60 Hz control / 10 Hz policy schedule, each
stored transition spans six held control steps. A terminal interval may be
shorter; discounting uses its measured duration. Integer policy/control ratios
are required in v1. There are no sequences across episode boundaries.

Example manifest (`data/bfm/manifest.json`):

```json
{
  "schema_version": "2.0.0",
  "dataset_id": "kuavo-reach-master-v1",
  "episodes": [
    {"path": "seed0-episode.zarr", "split": "train"},
    {"path": "seed100-episode.zarr", "split": "test"}
  ]
}
```

Manifest paths are relative to the manifest directory. The trainer only loads
`train` entries and fits normalization only to their measured states. Selected
metadata and array contents are hashed for resume compatibility. Future nested
coverage subsets can use new manifests referencing the same immutable episodes.
Replay loads the small state/action arrays once; images are not loaded. This
first version is memory-resident, not an out-of-core or distributed replay.

A minimal measured episode using an existing collector:

```python
from bfm.data.recorder import BFMRecorder, cpu_state
from isaaclab_wrapper.bfm_state import IsaacTransitionBridge

bridge = IsaacTransitionBridge(env, contract)  # validates actual robot/scene
env.reset(seed=0)
s = bridge.extract()
recorder = BFMRecorder(
    contract, episode_id="episode0", split="train",
    provenance={"kind": "isaac_teleoperation", "seed": 0},
    initial_scene={k: v.tolist() for k, v in cpu_state(s).items()},
    joint_names=env.unwrapped.robot.joint_names,
)
# command is a bounded 26D action produced by BFMActionAdapter from a supplied
# normalized right-arm command. Hold it for one policy period.
for j in range(round(contract.policy_dt / contract.control_dt)):
    next_s, terminated, truncated, (physical, full), _ = bridge.step(command)
    if bool(terminated[0] or truncated[0]):
        break
recorder.append(s, physical, full, next_s,
                terminated=terminated[0], truncated=not bool(terminated[0]),
                control_steps=j + 1)  # explicitly end this one-transition recording
recorder.save("episode0.zarr")
bridge.close()
```

Use the CLI for complete episodes; it checks action constancy across each held
interval and writes/extends the manifest. Run these commands from repository
root using the **Isaac Lab Python interpreter**:

```bash
python scripts/collect_bfm_zarr.py --output-dir data/bfm \
  --eef-body zarm_r7_link --eef-offset 0 0 0 \
  --actions-npz /path/to/normalized_right_arm_commands.npz --seeds 0 1 2
```

The NPZ must contain `actions[N,8]` in `[-1,1]` in the documented right-arm
order. Commands are executed in the simulator and measured afterward; the
command file is never treated as observations. `--random-smoke` instead of
`--actions-npz` supplies small target perturbations for recorder validation.
It does not promise task coverage or useful manipulation demonstrations.

## Train and resume

From repository root, in a CPU or CUDA PyTorch environment:

```bash
python -m pip install -r requirements-fb.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/train_offline_fb.py \
  --config configs/fb/vanilla_offline.yaml \
  --dataset-manifest data/bfm/manifest.json --output-dir runs/fb/vanilla

python scripts/train_offline_fb.py --config configs/fb/vanilla_offline.yaml \
  --dataset-manifest data/bfm/manifest.json --output-dir runs/fb/vanilla \
  --resume runs/fb/vanilla/last.pt --training-steps 200000
```

Set `--device cuda:0` for GPU training; preserve batch/model/train/reference
settings on resume. `training_steps` is the absolute target update count, not an
additional count. Relative configuration paths are resolved from the working
directory. Use a fresh output directory for an independent run.

`metrics.jsonl` reports real FB/orthogonality/actor losses, representation and
latent norms, action statistics/boundary fraction, and unclipped gradient norms.
`last.pt` is atomically refreshed at the configured interval and final step.
It contains model/targets, optimizer states, configuration, exact dimensions,
normalization, action scaling/joint mapping, dataset fingerprint/provenance,
training step, CPU/CUDA/Python/NumPy/replay RNG state, and fixed reference states.
Inference loads it without reconstructing the replay. Deterministic CPU resume
is tested bit-for-bit; cross-device/CUDA reproducibility is not claimed.

## Goal prompting and autonomous evaluation

Use the Isaac Lab Python interpreter with the repository on its Python path:

```bash
python scripts/eval_zero_shot_fb.py --checkpoint runs/fb/vanilla/last.pt \
  --reward-task reach_red_ball --seeds 0 1 2 --device cuda:0 \
  --output runs/fb/reach-evaluation.json

python scripts/eval_zero_shot_fb.py --checkpoint runs/fb/vanilla/last.pt \
  --reward-task move_red_ball_to_container --hold-steps 30 \
  --target-half-extent .06 .06 .04 --seeds 0 1 2 --device cuda:0 \
  --output runs/fb/container-evaluation.json
```

The evaluator freezes weights, infers one task latent, resets with each seed,
executes only learned actor actions at the recorded policy frequency, and
computes success from pre-reset measured positions. Reaching uses a configurable
distance threshold. Container success requires the ball center to remain inside
the configured target box for consecutive **control** steps. Metrics include
initial/final/minimum reach distance, time to success, episode length, ball target
distance, termination/cutoff status, goal/latent, checkpoint hash, and reference
indices. `--render` enables the existing viewer; video recording is not added.

`--goal-json` accepts `{"type":"goal","state":{...all seven measured fields...}}`
for complete-state prompting. `--reward-task` still selects the evaluation metric
when a full-state goal is provided. No missing goal fields are synthesized.

## Validation and current limitations

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests/fb -q
python -m pytest -q
```

`pytest.ini` selects actual tests under `tests/`; Isaac environment source files
named `test_training_env*.py` are not unit tests and need a launched simulator.
See [validation results](vanilla_fb_results.md) for the executed run and counts.

The following external prerequisites remain for real robot research validation:

* Isaac Sim / Isaac Lab and a compatible NVIDIA GPU. This cloud instance exposes
  no NVIDIA device and has no `isaaclab` or `omni` installation.
* Scene `Test_training/.../assets/base_scene.usd` and robot
  `assets/robots/urdf/supported_biped_s40_v2/supported_biped_s40_sensor.usd`.
  Both configured files are absent from this checkout (USD files are gitignored).
* New measured simulation episodes with adequate task/goal coverage, a verified
  TCP/body and left-arm hold configuration. The documented NAS is not mounted.

Vanilla FB is not a guarantee of successful offline manipulation. Its actor may
maximize extrapolated values for unsupported actions, especially binary-claw
interpolation. Monitor the saved action/boundary and representation metrics and
actual held-out rollouts. The implementation deliberately does not switch to
BC, FB-AW, or CPR to conceal failure. Finite synthetic losses prove neither
manipulation competence nor offline policy quality. Reaching is an intermediate
validation; a measured successful ball-to-container rollout is a separate,
currently unachieved research milestone.

## Research sources and access record

Source implementations were inspected as references; the new modules are an
independent compact implementation rather than vendored Meta Motivo code.
The original authors' repository is MIT-licensed; Meta Motivo source is
CC BY-NC 4.0. No Meta Motivo dependency or source copy is shipped.

* Touati & Ollivier, [Learning One Representation to Optimize All Rewards](https://arxiv.org/abs/2103.07945).
  Official authors' [FB agent](https://github.com/facebookresearch/controllable_agent/blob/5a9950b07f1edcb4bddd04f5819379541793d975/url_benchmark/agent/fb_ddpg.py)
  and [license](https://github.com/facebookresearch/controllable_agent/blob/5a9950b07f1edcb4bddd04f5819379541793d975/LICENSE)
  were read, including the successor TD, diagonal correction, orthogonality and
  actor updates.
* [Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models](https://arxiv.org/abs/2504.11054),
  [Meta Motivo project](https://metamotivo.metademolab.com/).
  Official [basic FB agent](https://github.com/facebookresearch/metamotivo/blob/ff8dcc55cf58f766d365ab0be23a021a7e34d53d/metamotivo/fb/agent.py),
  [model](https://github.com/facebookresearch/metamotivo/blob/ff8dcc55cf58f766d365ab0be23a021a7e34d53d/metamotivo/fb/model.py),
  [network/distribution definitions](https://github.com/facebookresearch/metamotivo/blob/ff8dcc55cf58f766d365ab0be23a021a7e34d53d/metamotivo/nn_models.py),
  and [license](https://github.com/facebookresearch/metamotivo/blob/ff8dcc55cf58f766d365ab0be23a021a7e34d53d/LICENSE)
  were read. Basic FB, not FB-CPR, supplies the implementation conventions.
* [Finer Behavioral Foundation Models via Auto-Regressive Features and Advantage Weighting](https://arxiv.org/abs/2412.04368).
* [BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning](https://arxiv.org/abs/2511.04131).

**Research access limitation:** requests to all four arXiv papers returned
`Tunnel connection failed: 403 Forbidden` under this environment's network
policy. Full paper texts could not be read; no claim is made to have studied
their inaccessible appendices. The mathematical implementation was checked
against the two accessible official FB codebases and a known analytic
successor solution. References C and D are recorded as requested context, not
as fully reviewed or reproduced methods. In particular, no undocumented details
of FB-AW or BFM-Zero are claimed as implemented.

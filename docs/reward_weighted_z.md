# Post-training EpisodeBalancedZ and RewardWeightedZ

No retraining, conversion, controller or runtime changes are needed. The actor,
backward network B, projection, random-z FB training and existing task artifacts
are unchanged. `EpisodeBalancedZ` is an alias of `balanced_task`: mean B within
each successful episode, mean those episode means equally, then `project_z`.
The historical `episode_balanced_full_trajectory_B_v1` method ID stays intact.
Episode Z, Cluster Z, search, goal inference and `model.reward_inference` remain
independent. The latter uses uniform state sampling and signed rewards without
this tool's softmax or episode balancing.

## Schema audit and reward limits

Inspected contracts/data adapter, schema documentation, tracked data/zarr manifest
and a materialized training_analysis reference (schema 3.0.0, 78 transitions).
The current worktree's production checkpoint/Zarr are LFS pointers; the materialized
reference was read only from another BFM checkout. data/zarr's manifest alone
cannot supply missing episode stores.

Available: state[56] has left/right EEF world positions, arm q/dq, object
position/velocity, goal center/half extents/orientation/velocities. Measured
physical joints (72 in that reference; visual selection 58 q + 58 dq), three RGB
streams, canonical actions, source/header/record timestamps, segments and
terminated/truncated flags are retained. success is episode-level metadata.
No per-state grasp/contact flags, verified occupancy, object volume/extent or
validated RGB-to-3D perception interface is supplied by this schema. Logical
hand targets and measured finger closure do not prove contact/grasp.

Supported *positional proxies*, using recorded task state only for offline task
inference (never adding privileged fields to the visual actor observation):

* `approach`: exp(-d/scale_m), d = distance from selected EEF to object, meters;
  `--hand either` takes the nearer EEF for handover demonstrations.
* `goal_proximity`: exp(-d/scale_m), object-to-goal-center distance in meters.
  This is not grasp-conditioned transport or verified placement.

Both have range [0,1]; default decay length 0.1 m is an explicit tuning parameter,
not a success threshold. Missing/nonfinite state or invalid box pose fails.
`grasp`, grasp-conditioned `transport`, and physical `placement` are unsupported:
use separately measured external annotations if those rewards are required.
Being inside an oriented box as a point would not certify object occupancy or
stable placement. success=true is only the reference selection filter, never an
intermediate reward. No time-progress reward or zero/uniform fallback is used.
The dataset can generate positional-proxy Z without annotations, but cannot
supply the full intended physical approach/grasp/transport/placement reward.

## Exact weighting

For successful episode e, unprojected b_et = B(s_et), finite signed reward r_et:

    w_et = softmax_t(beta * r_et)
    u_e = sum_t w_et * r_et * b_et
    z = project_z(mean_e u_e)

beta >= 0 is in inverse reward units; default 1. beta=0 makes w uniform but still
multiplies by signed r. Negative rewards retain their signs. Duplicating every
state/reward of one episode leaves its contribution unchanged. Long episodes do
not gain weight just because they have more frames. Softmax is centered and
computed in float64; one common positive reward scaling and a final direction
rescaling prevent overflow/underflow without changing the projected direction.
The checkpoint's own project_z sets norm sqrt(latent_dim). Constant rewards
across all selected states (including all zeros) fail as uninformative. Constant
individual episodes are reported. Cancelling/zero B directions fail too.

Endpoints exactly reuse episode_embeddings: union obs record_ns and next_obs
next_record_ns, sorted; shared endpoints once, final next_obs included, no gap
interpolation. This refactoring preserves the original endpoint replacement rule.
Each reward binds to episode ID, source fingerprint, state count and full ordered
endpoint descriptors, including segment and source record times.

## Commands

Use a checkpoint-compatible Python/torch environment. Production assets may
require Git LFS materialization. Existing preprocessing/checkpoint checks apply.

```bash
# Original EpisodeBalancedZ (existing CLI and method metadata unchanged):
python tools/build_task_z.py --checkpoint "$CKPT" --references "$REFS" --output baseline.pt --device cuda
# Existing Cluster Z (requires prior episode analysis):
python tools/build_cluster_task_z.py --checkpoint "$CKPT" --references "$REFS" --analysis "$ANALYSIS" --clusters 3 --output-dir clusters --device cuda
# Independent proxy RewardWeightedZ:
python tools/build_reward_weighted_z.py --checkpoint "$CKPT" --references "$REFS" --reward approach --hand either --scale-m 0.1 --beta 1 --output reward_z.pt --device cuda
# Or choose --reward goal_proximity. Not a physical success certificate.
# Optional measured per-endpoint annotations, outside source stores:
python tools/build_reward_weighted_z.py --checkpoint "$CKPT" --references "$REFS" --annotation-template --output rewards.json --device cuda
# Fill every null in rewards.json with measured signed rewards and document the
# measurement definition/provenance. Do not reorder/change endpoint descriptors.
python tools/build_reward_weighted_z.py --checkpoint "$CKPT" --references "$REFS" --annotations rewards.json --beta 1 --output annotated_z.pt --device cuda
# Compare norms, all pairwise cosines, reward/reference statistics and actions
# at the same first/middle/last retained observations; --timesteps overrides rows:
python tools/build_reward_weighted_z.py --checkpoint "$CKPT" --references "$REFS" --reward approach --output compared_z.pt --baseline-task baseline.pt --cluster-tasks clusters/cluster_0_task_z.pt clusters/cluster_1_task_z.pt clusters/cluster_2_task_z.pt --comparison-dir runs/latent_analysis/reward_comparison --device cuda
```

Annotation JSON: format=`bfm.reward_annotations.v1`, a nonempty measurement
`definition`, and `episodes` containing exactly the successful source episodes.
Each entry has episode_id, source_fingerprint, states, endpoints and rewards[N].
Templates contain nulls deliberately and cannot be inferred until completed.
Fingerprints bind numeric and compressed RGB source content through the existing
adapter. Foreign, duplicate, incomplete, stale or reordered annotations fail.
Task artifacts contain z, checkpoint_sha256, latent_dim, method=
reward_weighted_z_v1, reward definition, beta/weighting, diagnostics and source
fingerprints/endpoints. Output files must be new; inputs are never rewritten.

## Runtime and experiments

Load via existing VisualRuntime.load_task(path, require_bound=True), then act(obs).
Use an explicit runtime config compatible with the checkpoint: current default
runtime YAML is v4, so a historical v3 checkpoint needs its existing v3 config.

```bash
python tools/run_policy_ros.py --config "$COMPATIBLE_RUNTIME_YAML" --checkpoint "$CKPT" --task-z reward_z.pt --mode shadow --device cuda
python tools/run_policy_ros.py --config "$COMPATIBLE_RUNTIME_YAML" --checkpoint "$CKPT" --task-z reward_z.pt --mode closed-loop --device cuda
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q tests/fb
```

Comparison JSON/CSV reports norms, pairwise cosine matrix (including every cluster
and EpisodeBalancedZ), reward/state counts and hashes; action CSV includes 26D
normalized/physical actions, per-dimension differences and canonical group MAEs.
It reuses one encoded observation for all task directions and deterministic actor
calls. No latent/action difference proves improved performance. Fixed Z is loaded
before each rollout; no tracking/adaptive Z is implemented.

For later Isaac trials use the same checkpoint, saved initial scene and runtime
config, changing only task-z. Reset between repeated trials and retain runtime
logs. Score physical placement/grasp only with an independent validated evaluator
or measured annotations. Report repeated-trial results and completion time from
that evaluator; do not substitute these positional proxies for success metrics.
Live Isaac evaluation has not been performed for this patch.

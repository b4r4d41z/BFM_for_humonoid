# Collection-v3 and visual FB implementation report

## 1. Starting branch and SHA

Requested target: `zarr-pipeline`.
The original /workspace/BFM_for_humonoid checkout was at
`18c239b9e3bf2fb8f3c6c3e754e94ad5e4c2d3d2`, with three existing untracked
patch/audit files. Fetching the actual remote found target HEAD
`7db82816a73eb0c6c6544f7385ad6d859c7ba353`
(`mod: .MCAP->.zarr updated`). Implementation starts from that remote HEAD.

Collector main inspected read-only:
`95db6129043bd741f7f9401313247c3166ebe495`.
Its current contract/gateway establishes bounded canonical commands and preserved
source headers. Its tracked data_snapshot still describes v2, not the supplied v3
episode. No collector files were changed.

## 2. Final working-tree state

Changes are uncommitted in isolated branch `zarr-pipeline-vision`, at
`/workspace/BFM_for_humonoid-vision`, based on the SHA above.
The original implementation checkout and its patch/audit artifacts are preserved.
No remote branch was pushed.

## 3. Files changed

- bfm/contracts/sharpa.py — collection-v3 identity, same fixed action bounds.
- bfm/contracts/vision.py — reproducible 116D measured joint layout.
- bfm/data/mcap.py — v3 metadata, clamp/held provenance, skew diagnostics.
- bfm/data/sharpa.py — explicit observation profile, outcome selection, no task56 image-pixel I/O.
- bfm/data/visual_replay.py — lazy named-camera transition batches.
- bfm/fb/observation.py — frozen DINOv2, letterboxing, camera-aware MLP fusion.
- bfm/fb/model.py — explicit profile and raw visual observation interfaces.
- bfm/fb/agent.py — shared observation encoding, fusion gradients and target fusion.
- bfm/fb/checkpoint.py — visual identity, weights, references, compatibility and resume.
- bfm/fb/trainer.py — profile dispatch, visual replay and reproducible training.
- bfm/fb/networks.py — documentation updated; head architectures unchanged.
- bfm/sharpa_runtime.py — visual prompting/inference and measurement buffer.
- scripts/run_sharpa_fb.py — visual subscriptions and freshness checks.
- scripts/smoke_sharpa_vision.py — reproducible conversion/training/runtime smoke CLI.
- tools/data_conversion/mcap_to_zarr.py — optional camera skew threshold.
- configs/fb/vanilla_offline.yaml — explicit task56.
- configs/fb/vision_offline.yaml — production DINOv2 visual baseline.
- requirements-vision.txt — optional Transformers dependency.
- tests/fb/test_sharpa.py — v3 CDR fixtures and baseline regression coverage.
- tests/fb/test_vision.py — visual, provenance, timing and compatibility tests.
- docs/sharpa_pipeline.md — current complete pipeline documentation.
- docs/implementation_report.md — this report.

## 4. Functionality already present

The remote implementation already had action26 bounds/scaling, measured task56,
causal as-of recorder-time synchronization, complete named measured joint storage,
lossless three-camera RGB storage, schema3.0.0, continuity/gap validation,
source/outcome metadata preservation, state-only FB, checkpoints and ROS runtime.
It still required collection **v2**; a correct v3 migration was not present there.

## 5. V3 changes

Accept only collection3; require bounded canonical semantics and matching frozen
model/clamp metadata. Reject v2 explicitly, without clipping/migration.
Canonical commands remain bounded; pre-clamp raw requests may exceed limits.
Require held arm/hand equality and raw header identity plus clamp(raw)==canonical.
Use a +/-150ms raw provenance window separately from observation age limits;
prefer prior matches, allow later recorder ordering only for provenance.
The raw command never enters observation or action. Frozen model provenance and
recording SHA remain distinct. Old collection-v2 checkpoints fail contract checks.

## 6. Real MCAP conversion result

**Not run: file unavailable.** The requested path
`/workspace/Synthetic_data_collection_with_ExoSuit/records/record_20261028_141621_145555_a3fffe33/bag/bag_0.mcap`
does not exist here, and this recording is not tracked in collector origin/main.
No real synchronization, pixel, clamp-count or multimodal result is claimed.
The optional pytest skips explicitly; the smoke CLI is ready for that episode.

## 7. Resulting Zarr structure

Schema stays3.0.0; physical layout is unchanged. The generated fixture produced
9 transitions in one segment: action[9,26], obs/next_obs state[9,56], measured
q/dq[9,58], timing/provenance arrays, and all six uint8 RGB arrays[9,2,2,3].
No embeddings replace RGB. Detailed production layout is in sharpa_pipeline.md.

## 8–9. Synchronization and inter-camera skew

Real episode: unavailable, so no measured statistics.

Generated MCAP smoke, ten ticks (p50/p95/max, milliseconds):

| Source/diagnostic | p50 | p95 | max |
| --- | ---: | ---: | ---: |
| state age | 1 | 1 | 1 |
| joints age | 1 | 1 | 1 |
| head age | 1 | 1 | 1 |
| left wrist age | 4 | 4 | 4 |
| right wrist age | 8 | 8 | 8 |
| camera skew | 7 | 7 | 7 |
| matched raw source age | 94.85 | 94.85 | 94.85 |

All retained observation source times are <= decision times. Zero ticks rejected.
Ten clamp events retained: raw left arm4=-1.668971, held/canonical=-1.5708.
success=false survives conversion. Native RGB channel range1..6 in this small
generated fixture is nonempty; it is not a camera recording from the simulator.

## 10. Exact visual proprioception

116D = q[58] followed by dq[58]. Within each half:
left arm7, right arm7, left physical hand22, right physical hand22.
The exact 22 hand suffixes and all offsets are listed in sharpa_pipeline.md and
PROPRIO_JOINTS in bfm/contracts/vision.py. Units are rad/rad/s.
Selection is by names, including all physical finger flexion/abduction joints.
Commands, object state, goal fields and EEF fields are excluded.

## 11–14. Backbone, preprocessing, fusion and dimension

Production: shared frozen facebook/dinov2-small via Transformers4.46.3.
Selected for a direct CLS feature interface and a compact pretrained backbone.
Remote revisions resolve when available; weights and SHA256 are checkpointed.
Local pretrained artifact paths are supported; mock is explicitly tests-only.

Whole-frame RGB /255, aspect-preserving bicubic antialiased resize, mean-color
letterbox to configurable518x518, ImageNet mean/std. No center crop. Input size
must be a multiple of14. Train/runtime use the same function and pinned eager
attention implementation. Torch release/Transformers version are checked.

Separate camera projections:384->64 each. Proprio MLP:116->64->64.
Concatenate three projected views and proprio, then256->256->256 fusion MLP.
Default observation embedding256; projection/fusion dimensions configurable.
Backbone inference is microbatched and its weights remain frozen.

## 15. FB interfaces

task56 keeps tensor inputs. Visual public interfaces receive a dictionary of
proprio plus named camera tensors. F/B/actor consume fused h; original FB loss,
latent sampling, discount and actor objective are retained.
One online encoder serves t and t+1. FB optimizes small fusion/projection/proprio
modules. Actor uses detached embeddings. Target fusion is Polyak-updated alongside
target F/B and shares the frozen extracted next features.

## 16. Runtime

Explicit --observation-profile selects state-only or visual mode. Visual mode
subscribes to measured JointState and selected cameras, not BfmState. It uses the
same encoder/preprocessing/normalizer from the checkpoint, with causal local
receive-time snapshots, freshness before/after inference and optional camera skew.
Action inversion, logical-hand rounding and existing publication endpoints remain.
No live ROS/Isaac rollout or10Hz throughput was available for verification.

Task z is inferred from saved reference reward states and aligned visual backward
embeddings. Privileged reference values define the task prompt; current privileged
scene state is not a policy observation.

## 17. Checkpoint behavior

Saves profile, full backbone architecture/weights/hash/revision, frozen state,
library/preprocessing identities, camera order, named proprio layout, fusion
configuration, dimensions, collection3/Zarr3.0.0, normalization, targets,
optimizer/replay/RNG state and reference embeddings.
Offline restore needs no pretrained-model download. Wrong profile, incompatible
configuration/software, checksum mismatch and collection-v2 identity fail.
Tests verify exact resumed-vs-uninterrupted mock-backed training.

## 18. Tests

- FB suite:65 passed,1 skipped (missing real v3 episode).
- Full suite:77 passed,1 skipped,1 pre-existing failure.
- The failure is tests/test_temporal_contract.py::
  test_future_targets_are_timestamp_based_half_second_and_do_not_cross_done.
  The same failure was reproduced on unchanged base7db8281: a0.4s discrepancy
  exceeds the legacy test's0.3s tolerance. Unrelated legacy temporal code is unchanged.
- Actual Transformers DINOv2 loading/restoration API exercised with a tiny
  randomly initialized LOCAL fixture; outputs restore exactly, backbone gradients
  remain absent, fusion gradients are finite, and restore performs no download.
- Full pretrained DINOv2 weights were not downloaded/validated.
- Python compilation and git diff --check pass.

The tests cover v2 rejection, exact/asymmetric limits, raw overshoot, canonical
range rejection, header/value association, causal/stale/gap handling, failed
episodes, skew, no task56 image reads, name-selected full hands, no privileged
observation leakage, RGB/letterbox determinism, camera roles/subsets, finite
updates, frozen parameters, profile/config/checkpoint rejection, and freshness.

## 19. Multimodal smoke

Generated CDR/MCAP -> canonical Zarr -> lazy three-camera batch -> explicit mock
encoder -> ten FB optimizer steps -> checkpoint -> physical26D runtime inference:
**passed**. All losses, gradients and actions finite. Final FB loss=-0.94725108,
actor loss=1.11499989. These numbers are plumbing diagnostics, not learning results.

Artifacts:
- /workspace/bfm-vision-smoke-final/episode.zarr
- /workspace/bfm-vision-smoke-final/run/last.pt
- /workspace/bfm-vision-smoke-final/report.json
- /workspace/bfm-vision-smoke-final.log

## 20. Remaining validation before large training

Run the supplied real episode through conversion and the pretrained smoke CLI.
Its reported counts/clamp event were not independently measured here.
Supply pretrained DINOv2 weights locally or permit initial model-host access;
the managed network's current allowed-host list does not include Hugging Face.
Benchmark GPU memory/throughput and live10Hz ROS/Isaac operation. Verify actual
measured finger names and command-header association on the real recording.
No real policy-learning, manipulation-success or sim2real claim is made.

Canonical conversion buffers an episode; training keeps small arrays in RAM and
fetches RGB lazily. Per-camera resolutions must be consistent across batch
episodes. Persistent feature caches and temporal models are not implemented;
the frozen-feature boundary permits a later fingerprinted derived cache.

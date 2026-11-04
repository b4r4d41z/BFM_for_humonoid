# Raw recordings to visual FB training

From the repository root, with `requirements-vision.txt` installed and the local
`models/dinov2-small` Git LFS weights materialized:

```bash
python tools/convert_recordings.py /path/to/raw_recordings
python tools/train_from_zarr.py data/zarr
```

The conversion command takes a raw episode directory or a directory recursively
containing episodes with `episode.json`. The launcher reads authoritative
`recording_code.commit_sha`, calls the strict converter, validates each store,
and automatically writes the existing Sharpa manifest. It never starts training.
The independent Zarr-only trainer validates the manifest and all visual training
fields, then calls `bfm.fb.trainer.train` with `configs/fb/vision_offline.yaml`.
It never reads MCAP or regenerates converted data. CUDA is selected when available.
It does not change the model, objective, action contract, or observation pipeline.

```text
data/
  current_sample/<raw regression episode>/
  zarr/
    .gitkeep
    episode-<sha256 of source episode ID>.zarr/
    manifest.json
```

Converted data and the manifest are generated and ignored by Git. There was no
`data/sorting_food` storage path in the current branch to migrate; task identifiers
remain unchanged. Legacy measured-v2 training configurations are unaffected.

Recorder-time freshness defaults are centralized in
`configs/data_conversion/sharpa_v3.yaml`: 150 ms for each required source, matching
the existing Sharpa conversion guide. Camera skew is measured; a hard skew limit
is optional there. Independently arriving observations always use causal MCAP
receive-time selection. Policy cadence remains in the declared simulation clock.

The callback-rate collection-v3 metadata declares that bounded targets are
published on source callbacks and actions snapshot the latest target. Such a
command remains in force until replaced, regardless of receive-time distance to a
policy tick. Association uses the latest causal command and exact raw source header
plus `clamp(raw)==action`. A reordered arm publication requires the latest causal
raw identity and must precede the next policy record. Legacy recordings keep their
existing association rules. No controller-frequency or message-count assumption
is made; invalid canonical actions/provenance still abort conversion.

Progress shows sample N/total, episode name, a bar, and converting/validated/skipped/
failed status. `skipped` means a validated identical conversion is reused, not that
invalid input is ignored. Any failed episode aborts conversion.
Completed earlier stores remain valid and reusable. Temporary conversions are
removed on ordinary failures; even after process termination their hidden temporary
directories are never discovered as final stores. An existing store is validated
and its source metadata, raw MCAP bytes, conversion policy and converter version
must match before reuse; changed sources require explicitly moving the old output
aside. The original valid store is never overwritten.

All valid `data/zarr/*.zarr` stores are discovered. Existing manifest split
assignments are preserved; new episodes are assigned wholly to train. No automatic
validation/test split or success filtering is invented. Duplicate episode IDs,
invalid stores or malformed existing split assignments abort preparation.

Each training invocation starts a new run under `runs/fb/zarr/`; it does not overwrite
or silently resume training. For explicit resume use the existing low-level trainer.
On CPU or limited GPUs, a short real pretrained smoke run is:

```bash
python tools/train_from_zarr.py data/zarr \
  --device cpu --steps 3 --batch-size 2 --feature-batch-size 1 --reference-count 2
```

`--config` optionally selects an existing training YAML. Count/device overrides are
optional diagnostics; normal use needs no output path, manifest, SHA or age flags.


## Post-training task inference and external simulator runtime

```text
raw MCAP -> convert_recordings.py -> Zarr + manifest
-> train_from_zarr.py -> last.pt
-> successful reference trajectories -> build_task_z.py -> task_z.pt
-> run_policy_ros.py -> ROS2 -> external Isaac simulator
```

Training Zarr contains actions, measured q/dq (adapted to 116D proprio), three
RGB views and their next observations, durations/timestamps, terminal/truncation
flags and reference-state sidecars. DINO weights, Python code and model YAML are
external resources, not episode arrays. A saved checkpoint contains the inference
model weights and preprocessing/normalization identity as before.

```bash
python tools/build_task_z.py --checkpoint runs/.../last.pt \
  --references data/zarr/manifest.json --output task_z/sorting_fruit.pt \
  --embeddings-out task_z/reference_embeddings.pt
python tools/run_policy_ros.py --checkpoint runs/.../last.pt \
  --task-z task_z/sorting_fruit.pt --device cuda
```

References may be a supported manifest, a directory of Zarr stores, or one store.
Only episodes explicitly labelled success=true contribute to the task vector.
The label is consumed as supplied, not independently certified by this tool.
All retained states along each source trajectory, including final next_obs and
unique shared endpoints, are encoded through the checkpoint's own visual encoder
and B. Missing/rejected raw frames cannot be reconstructed from Zarr. State means
are computed within source episodes (across retained segments), then averaged
across successful episodes and projected using model.project_z. Longer episodes
receive no additional weight. The saved vector records method, checkpoint SHA256,
latent dimension, reference episode IDs/paths and state counts. A bare tensor is
also accepted at runtime; metadata-bearing task files must match the checkpoint.
Task inference never alters random-z reward-free pretraining.

ROS topics/message types follow the existing scripts/run_sharpa_fb.py interface:
sensor_msgs/Image, sensor_msgs/JointState, and dynamic_biped BfmAction and
RobotHandPosition. Configure configs/runtime/isaac_ros2.yaml and source the ROS2
message packages. Normalized actor output is converted through ActionNormalizer
and the existing fill_messages adapter for physical controller publication.
The canonical joint order is PROPRIO_JOINTS; no privileged state enters the actor.
The task vector is loaded once and reused, with no per-frame resampling. Local
monotonic receive ages/skew include inference latency. Missing observations wait;
stale inputs or competing publishers stop publication. External gateway/teleop
publishers on the control topics must relinquish control. The simulator remains
external; live transport/control authority needs integration validation.

## Experimental search and diagnostics

```bash
python tools/search_task_z.py --checkpoint runs/.../last.pt --count 32 --output candidates.pt
# External simulator runs each candidate ID and writes {"z-000000": score, ...}.
python tools/search_task_z.py --checkpoint runs/.../last.pt \
  --candidates candidates.pt --scores scores.json --output best_task_z.pt
python tools/visualize_latent_space.py --checkpoint runs/.../last.pt \
  --embeddings task_z/reference_embeddings.pt \
  --task-z task_z/sorting_fruit.pt best_task_z.pt --output diagnostics/latent
```

Search uses model.sample_z, not a new distribution. The Python rank_candidates
API accepts a rollout callback; the CLI accepts externally produced finite scores
for exactly the candidate IDs. Scores are ranked deterministically and the best
vector is saved as a runtime-compatible task file. This is the experimental
alternative; successful-reference inference is primary. No simulator is embedded.

PCA uses NumPy SVD and emits CSV coordinates/metadata and an SVG scatter plot with
source colors and metadata tooltips, without new dependencies. Inputs can combine
random training-distribution z, full B(s), episode means and task/search vectors.
Only known episode/success/source metadata are used. Optional --method umap needs
an existing umap-learn installation; PCA does not require it. Cached embeddings
are checkpoint-bound and avoid duplicate DINO passes during diagnostics.

## Persistent frozen DINO feature cache

Existing canonical Zarr needs no reconversion. RGB uint8 and its Blosc/Zstd
compression remain unchanged. To avoid DINO inference and RGB decompression in
training updates, build a disposable derived cache, then use it in a later process:

```bash
python tools/build_feature_cache.py data/zarr --device cuda --batch-size 32
python tools/train_from_zarr.py data/zarr --device cuda --feature-cache data/feature_cache
# Original online RGB/DINO path:
python tools/train_from_zarr.py data/zarr --device cuda
```

Both tools accept the same `--config` YAML; use matching vision and outcome
selection. Cache construction also accepts `--output`. Its default directory is
`data/feature_cache/`, ignored by Git. `index.json` binds the selected training
manifest/dataset fingerprint to content-addressed `episodes/<hash>/` entries.
Each entry contains `entry.json` and a memory-mapped `features.npy` of float32
`[unique endpoints, cameras, backbone feature dimension]` (normally 384).
Only frozen CLS features are cached; projections, proprio encoding and fusion
remain trainable. Extraction calls the same encoder/preprocessor as online use.
Adjacent `next_obs`/`obs` share an endpoint only inside a fully validated segment.

Identity includes compressed RGB content through the existing dataset fingerprint,
source episode identity, camera order, observation profile, resolved backbone
configuration/revision/weight hash, image size, preprocessing, implementation and
library versions, float32 dtype and matmul precision settings. Changing these
requires rebuilding; incompatible or damaged caches fail closed. Batching and
trainable projection/fusion sizes do not invalidate frozen features. Small
float32 rounding differences across execution batch sizes/devices are possible;
no mixed-precision storage or training is introduced.

Rerun the builder after adding episodes: unchanged episode entries are validated
and reused; the dataset index is published only after all entries finish. Failed
builds never publish a partial index. Keep canonical input stores static while
building/training. Do not run concurrent builders into the same destination.
Local backbone identifiers include their configured path; moving model assets or
changing the vision implementation can conservatively require a rebuild.
Delete a corrupt cache directory and rebuild it; it contains no canonical data.

Cached startup still hashes compressed RGB bytes to detect stale data, but does
not decompress pixels. The model loads its frozen backbone once to verify weight
identity and preserve existing self-contained checkpoints/runtime inference;
training updates and checkpoint reference embeddings use cached features without
DINO forwards. Thus backbone parameter memory remains allocated. Without
`--feature-cache` (or the equivalent `feature_cache` training config key), behavior
is unchanged. CPU validation can use `--device cpu --batch-size 1` for the builder.

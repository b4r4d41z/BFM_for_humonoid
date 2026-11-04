# Episode task-direction diagnostics

These offline tools do not change training, task inference or runtime behavior.
Use an existing visual checkpoint and its checkpoint-bound global task file:

```bash
python tools/analyze_episode_z.py --checkpoint checkpoint.pt --references data/zarr \
  --task-z task_z.pt --device cuda --batch-size 8 --output-dir runs/latent_analysis/episodes
python tools/visualize_latent_space.py --analysis runs/latent_analysis/episodes/episode_z.pt \
  --output-dir runs/latent_analysis/pca
python tools/compare_z_actions.py --checkpoint checkpoint.pt \
  --analysis runs/latent_analysis/episodes/episode_z.pt --device cuda \
  --output-dir runs/latent_analysis/actions
```

Output directories must be new. Inputs are read-only. Reference formats are exactly
those of build_task_z.py: a Sharpa manifest, directory of Zarr episodes or one Zarr.
Only success=true episodes contribute directions; existing reference validation and
full retained endpoint/segment semantics apply. Per-episode z is balanced_task on a
single episode, hence project_z(mean B), never mean joint positions or mean of
already projected per-state vectors. Global task loading requires checkpoint binding.
The analysis records checkpoint/task hashes, episode paths, endpoint counts and
source-content fingerprints. Keep source stores static during analysis.

The analyzer saves episode_z.pt, episode_metrics.csv, pairwise_cosine_similarity.csv
and summary.json. Pairwise summaries exclude the diagonal and count each unordered
pair once. A single episode yields null pairwise statistics; PCA requires at least
two episodes and two latent dimensions. Identical vectors have undefined explained
variance, reported as variance_defined=false and zero ratios.

PCA fits episode vectors only and projects the global direction onto those same
axes. It saves pca_coordinates.csv, pca_summary.json and episode_z_pca.svg, a
standalone non-GUI figure. Blue numbered points correspond to CSV episode rows;
the global task is a larger red point. Hover shows full IDs. No DINO/B inference
occurs in this visualization mode. The existing visualization CLI remains available.

Action comparison defaults to the first episode in sorted ID order and the first,
midpoint and last retained obs rows (not raw decision indices). Override with
--episode-id ID --timesteps 0 10 20. Default latents are nearest, median-rank and
farthest from global by cosine distance, ties by episode ID, with duplicates
removed. Override with --z-ids ID1 ID2. Each real observation is encoded once;
act_encoded uses deterministic=True for each projected task. No ROS or simulator
is used. Analysis/checkpoint identity and source fingerprints must still match.

CSV outputs include normalized and physical 26D actions, signed differences per
dimension, normalized L2/MAE/max differences and per-canonical-group MAE. Physical
arm differences are radians and physical hand differences logical 0..100 units;
no mixed-unit physical norm is presented. Summary metrics exclude global self
comparisons. Sensitivity at a few states is not rollout quality. Compact directions
alone do not prove narrow behavior or model collapse; compare multiple checkpoints,
reference distributions and observation phases before drawing that conclusion.

## Post-training cluster tasks

```bash
python tools/build_cluster_task_z.py --checkpoint checkpoint.pt --references data/zarr \
  --analysis runs/latent_analysis/episodes/episode_z.pt --clusters 3 \
  --device cuda --batch-size 8 --output-dir runs/latent_analysis/clusters
python tools/visualize_latent_space.py --analysis runs/latent_analysis/episodes/episode_z.pt \
  --cluster-dir runs/latent_analysis/clusters --output-dir runs/latent_analysis/cluster_pca
```

Clustering is deterministic average-linkage agglomeration using cosine distance
in the original latent dimension. It uses no semantic labels or random seed.
Episode IDs are sorted first; equal-distance merges break ties by lexicographic
member-index order. Cluster IDs follow that same order. Any 1 <= k <= N is valid;
identical directions may be split by tie-breaking, which is not evidence for modes.
No claim is made that k=3 is optimal. Pairwise storage is quadratic in N; this tool
is intended for reference sets of tens/hundreds of episodes, not millions.

Membership uses projected episode z, but each final cluster task uses the existing
balanced_task on reconstructed full-trajectory B embeddings: mean within each
successful episode, equal-weight mean across member episodes, then project_z.
It is not the mean of projected episode directions. No training or weights change.
Reconstruction runs DINO/B again because episode_z.pt stores directions, not the
unprojected B means needed for correct balancing. The successful reference ID set,
endpoint counts, source fingerprints and checkpoint binding must match the analysis.
Paths may relocate if content and IDs match. Keep source stores static during use.

Each cluster_N_task_z.pt is checkpoint-bound, records member/source provenance and
the analysis hash, and is verified with runtime.load_task(require_bound=True).
Use it in place of the global task file with the existing runtime. The input global
task remains unchanged. clusters.csv compares each episode to its assigned final
cluster task and the analysis's bound global task. cluster_task_similarity.csv
contains the full k-by-k task cosine matrix plus k global comparisons.
cluster_summary.json records membership, pairwise within-cluster statistics (unique
off-diagonal pairs; null for singletons), task similarities and method/provenance.
Output directories must be new; summary is written last, after all task loads pass.

Cluster visualization needs only the saved artifacts. PCA still fits episodes only;
global/cluster tasks use those same axes out-of-sample. Outputs are
pca_coordinates.csv (including kind and cluster_id), pca_summary.json and
episode_z_cluster_pca.svg. Circles label episodes with row number and cluster ID,
squares mark cluster tasks, and a black cross marks global. Full IDs are in hover
labels. Explained variance describes a 2D projection, not proof of cluster structure.

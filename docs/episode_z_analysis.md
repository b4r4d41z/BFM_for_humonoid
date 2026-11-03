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

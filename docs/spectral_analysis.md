# Offline spectral diagnostics

Use a checkpoint-bound episode_z.pt from analyze_episode_z.py and the same static
successful references. New output directories are required; inputs remain read-only.
No training, task construction or runtime changes are made. No new dependencies.

```bash
python tools/analyze_latent_frequency.py --checkpoint "$CKPT" --references "$REFS" \
  --analysis "$ANALYSIS" --device cuda --batch-size 8 --bins 128 --output-dir /tmp/spectral/frequency
python tools/analyze_latent_graph.py --analysis "$ANALYSIS" --neighbors 3 5 10 \
  --seeds 0 1 2 --output-dir /tmp/spectral/graph
python tools/compare_latent_clusters.py --analysis "$ANALYSIS" --cluster-dir "$COSINE_CLUSTERS" \
  --frequency-dir /tmp/spectral/frequency --graph-dir /tmp/spectral/graph \
  --output-dir /tmp/spectral/comparison
```

Frequency uses full, unprojected chronological B embeddings from episode_embeddings,
including final next_obs. Shared record-time endpoints appear once, exactly as in
the existing reader. Inconsistent duplicate endpoints and nonincreasing policy times
fail. Canonical timestamp/next_timestamp are validated policy-time seconds (simulation
time for current recordings, recorder time for legacy); these clocks are not mixed
within an episode. Segment boundaries and gaps >1.5 median dt are split, never joined.
Chunks shorter than four samples are explicitly excluded. Small jitter is linearly
resampled at median dt without time-warping episode duration. Each chunk is centered
per dimension and Hann-windowed; one-sided PSDs sum latent-coordinate powers.
Duration-weighted segment PSDs are interpolated onto a common grid up to the smallest
Nyquist frequency. This deliberately drops higher frequencies unavailable in slower
sources. It does not recover missing temporal resolution from short episodes.
Frequency CSV stores raw PSD and unit-sum shape; energy is separately reported.
Zero-energy episodes cannot define spectral shape and are excluded with warnings.
At least three usable episodes are needed. Hellinger distance compares normalized
power shapes; candidate k=2..min(8,N-1) uses average linkage and silhouette selection.
Changing --bins allows processing sensitivity studies; use separate directories and
compare the resulting assignments. No automatic processing-invariance claim is made.

Graph uses exp(-(1-cosine)) affinities on a symmetric union-kNN graph, excluding
self edges. Each requested neighborhood must be in 1..N-1. At least four episodes
are required. Zero-degree nodes use zero Laplacian rows/columns and are listed.
Eigenvalues, components, gaps, affinities and coordinates are exported per neighborhood.
Clustering row-normalizes the first k eigenvectors and runs farthest-initialized
Lloyd iterations with seeded initial point and deterministic empty-cluster repair.
Candidate counts include k=3 whenever defined. Cosine-distance silhouettes assess
partitions in original z geometry; ARI across seeds reports clustering stability.
Eigenvectors in repeated eigenspaces are not unique across numerical libraries.
The smallest requested neighborhood is the explicitly designated primary result;
all neighborhood studies remain available. Eigengap candidates are independent
of silhouette-selected k. Disconnected components are not inferred semantic modes.

Comparison strictly aligns identical episode-ID sets and checkpoint/analysis hashes;
missing/excluded/duplicate IDs fail rather than silently intersecting datasets.
If temporal exclusions exist, resolve them or create a matching new analysis before
comparison. ARI, arithmetic-normalized NMI, contingency figures and cluster sizes
are saved, together with each source summary and silhouette. Duration, sample count
and embedding energy distributions by frequency cluster and descriptive eta-squared
help identify confounds; these are not significance tests. All three comparison
panels share one PCA fit of episode z. Frequency and graph also retain their own
coordinate CSVs and SVG figures. SVG curves/points expose IDs via hover; CSVs carry
all numeric values. No object labels are used. Silhouette chooses a candidate, not
proof of an optimal k; high agreement does not establish meaningful control skills.

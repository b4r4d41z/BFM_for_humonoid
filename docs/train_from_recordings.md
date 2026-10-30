# Raw recordings to visual FB training

From the repository root, with `requirements-vision.txt` installed and the local
`models/dinov2-small` Git LFS weights materialized:

```bash
python tools/train_from_recordings.py /path/to/raw_recordings
```

The sole required argument is a raw episode directory or a directory recursively
containing episodes with `episode.json`. The launcher reads authoritative
`recording_code.commit_sha`, calls the strict converter, validates each store,
automatically writes the existing Sharpa manifest, and calls `bfm.fb.trainer.train`
with `configs/fb/vision_offline.yaml`. It selects CUDA when available, otherwise CPU.
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
invalid input is ignored. Any failed episode aborts the batch before training.
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

Each invocation starts a new run under `runs/fb/recordings/`; it does not overwrite
or silently resume training. For explicit resume use the existing low-level trainer.
On CPU or limited GPUs, a short real pretrained smoke run is:

```bash
python tools/train_from_recordings.py data/current_sample \
  --device cpu --steps 3 --batch-size 2 --feature-batch-size 1 --reference-count 2
```

`--config` optionally selects an existing training YAML. Count/device overrides are
optional diagnostics; normal use needs no output path, manifest, SHA or age flags.

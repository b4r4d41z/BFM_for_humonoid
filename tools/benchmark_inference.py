"""Measure existing checkpoint inference on real Zarr RGB, without ROS or training."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import torch
    from bfm.data.sharpa import SharpaDataset
    from bfm.data.visual_replay import VisualReplayBuffer
    from bfm.fb.runtime import VisualRuntime

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--task-z', required=True)
    p.add_argument('--references', type=Path, required=True, help='Canonical Zarr manifest or its directory')
    p.add_argument('--device', default='cuda')
    p.add_argument('--precisions', nargs='+', choices=['fp32', 'bf16', 'fp16'], default=['fp32', 'bf16', 'fp16'])
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--samples', type=int, default=3)
    a = p.parse_args()
    if min(a.warmup, a.iterations, a.samples) < 1:
        p.error('warmup, iterations and samples must be positive')
    if a.iterations < a.samples:
        p.error('iterations must cover every selected sample')
    manifest = a.references/'manifest.json' if a.references.is_dir() else a.references
    data = SharpaDataset(manifest, observation_profile='vision_proprio_v1')
    replay = VisualReplayBuffer(data)
    indices = np.linspace(0, len(data.image_rows)-1, min(a.samples, len(data.image_rows)), dtype=int)
    # Disk I/O and RGB decompression are outside timed inference, as in ROS input.
    observations = [replay.observations([i]) for i in indices]
    runtime = VisualRuntime(a.checkpoint, a.device)
    runtime.load_task(a.task_z, require_bound=True)
    @torch.no_grad()
    def original_fp32(obs):
        action = runtime.model.act(runtime.observation(obs), runtime.z, deterministic=True)
        if action.shape != (1,26) or not torch.isfinite(action).all() or (action.abs()>1).any():
            raise ValueError('Invalid reference action')
        return action[0].cpu()

    baseline = [original_fp32(obs) for obs in observations]

    def stats(values):
        return dict(mean=float(np.mean(values)), p50=float(np.median(values)), p95=float(np.percentile(values, 95)))

    for i in range(a.warmup): original_fp32(observations[i % len(observations)])
    reference_times = []
    for i in range(a.iterations):
        if runtime.model.device.type == 'cuda': torch.cuda.synchronize(runtime.model.device)
        start = time.perf_counter()
        original_fp32(observations[i % len(observations)])
        if runtime.model.device.type == 'cuda': torch.cuda.synchronize(runtime.model.device)
        reference_times.append(time.perf_counter()-start)
    report = dict(original_fp32_latency_s=stats(reference_times), device=str(runtime.model.device), torch=str(torch.__version__),
                  checkpoint_sha256=runtime.checkpoint_sha256, dataset_fingerprint=data.fingerprint,
                  warmup=a.warmup, iterations=a.iterations, samples=indices.tolist(), modes={})
    for precision in a.precisions:
        try:
            candidate = VisualRuntime(a.checkpoint, a.device, precision=precision)
        except ValueError as exc:
            report['modes'][precision] = {'unavailable': str(exc)}
            continue
        candidate.load_task(a.task_z, require_bound=True)
        for i in range(a.warmup):
            candidate.act(observations[i % len(observations)])
        totals, differences = [], []
        for i in range(a.iterations):
            index = i % len(observations)
            if candidate.model.device.type == 'cuda':
                torch.cuda.synchronize(candidate.model.device)
            start = time.perf_counter()
            action = candidate.act(observations[index])
            # act returns a completed CPU action; synchronize explicitly for timing.
            if candidate.model.device.type == 'cuda':
                torch.cuda.synchronize(candidate.model.device)
            totals.append(time.perf_counter()-start)
            differences.extend((action-baseline[index]).abs().tolist())
        # Separate instrumented pass: synchronization overhead is not speedup data.
        candidate.profile = True
        stages = []
        for i in range(a.iterations):
            candidate.act(observations[i % len(observations)])
            stages.append(dict(candidate.last_profile))
        report['modes'][precision] = dict(total_latency_s=stats(totals),
            profiled_stages_s={key: stats([row[key] for row in stages]) for key in stages[0]},
            action_max_abs_difference=max(differences), action_mean_abs_difference=float(np.mean(differences)))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

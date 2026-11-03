# ROS2 simulation policy evaluation

Run from the BFM repository with the existing external simulator publishing the
configured joint states and three RGB topics. Source ROS2 and dynamic_biped
message packages. BFM has no Isaac Python dependency and does not manage the scene.

Evaluation requires checkpoint-bound task metadata; legacy bare tensors cannot
prove compatibility and are rejected by this entrypoint. Build a latent using
successful canonical Zarr references:

```bash
python tools/build_task_z.py --checkpoint checkpoint.pt --references data/zarr \
  --output task_z.pt --device cuda
python tools/run_policy_ros.py --checkpoint checkpoint.pt --task-z task_z.pt \
  --device cuda --mode shadow
python tools/run_policy_ros.py --checkpoint checkpoint.pt --task-z task_z.pt \
  --device cuda --mode closed-loop --warmup-ticks 10 --max-policy-steps 300
```

Shadow is the default and creates no output publishers. It runs the same inference,
physical conversion and safety logic, with virtual applied targets for diagnostics.
Before closed-loop evaluation, stop other publishers on ALL configured output
topics. Competition aborts publication, including duplicate runtime node names.

Each policy timer invocation counts toward max-policy-steps, including waiting and
warm-up. The first N consecutive complete valid snapshots run inference without
publication; publication can begin on valid tick N+1. Missing startup sources reset
this count. Stale/invalid observations, excessive camera skew, inference/action
errors and competing publishers stop the rollout. Freshness includes inference
latency and uses monotonic receive time, independently of ROS header clocks.

The mandatory arm limiter uses the canonical bounds and only the 14 arm targets.
Its initial anchor is measured physical arm q; subsequent anchors are previous
applied targets. Measured initial arms outside canonical bounds abort evaluation.
Warm-up does not advance the safety anchor. Hands retain existing logical 0..100
semantics and message rounding. `--max-arm-delta-rad` overrides the YAML default
0.1 rad per policy tick: an evaluation cap, not a certified robot velocity limit.
Review the limit for your scene. No training distribution or model changes occur.

Logs are printed at startup and stored at runs/eval/<run_id>/ticks.jsonl with
summary.json on clean stop, interruption or caught failure. Source ages are in
seconds on the monotonic clock; no images are logged. Invalid/missing snapshots
count as invalid observations. A failure while publishing multiple ROS messages
cannot undo a message already sent; ROS topic publication is not atomic. Stop
means no further commands, not an emergency stop or cancellation of a controller's
last held target. Use the external simulator's stop mechanism if needed.

CLI overrides mode/count/arm limit; topics, freshness, camera skew and log root
remain in configs/runtime/isaac_ros2.yaml. Policy frequency remains the existing
contract frequency. Live ROS2 graph, QoS, clock and controller behavior require
integration validation in the actual simulator environment.

## Inference precision and profiling

FP32 remains the compatibility default. CUDA-only BF16/FP16 are opt-in runtime
settings; existing checkpoints need no retraining or conversion:

```bash
python tools/run_policy_ros.py --checkpoint checkpoint.pt --task-z task_z.pt \
  --device cuda --mode shadow --precision bf16 --profile-inference
```

`--precision fp16` is an alternative; BF16 requires native hardware support and
retains FP32-like exponent range, whereas FP16 has greater overflow risk. Unsupported
modes fail explicitly. Nonfinite features/actions still stop the rollout. Compare
both on your GPU and review action deviations before closed-loop use. There is no
automatic precision fallback or claim that either mode always meets 100 ms.

Only backbone operations run under CUDA autocast. Existing preprocessing, image
size, camera order, normalization, projections/fusion and actor stay FP32, and
weights remain FP32. Runtime stages reuse the existing preprocessing and model
components; tests compare the split runtime against original model.act. Training,
feature-cache generation/identity, build_task_z and checkpoint format are unchanged.

`profile_inference` and `inference_precision` are also configurable in the existing
runtime YAML. CLI overrides them; `--no-profile-inference` disables instrumentation.
Profiled ticks add input, preprocessing, backbone, feature handling, fusion, policy
head and total policy latency fields in seconds to ticks.jsonl, including ticks
that subsequently fail freshness. GPU timings synchronize the selected CUDA device
at stage boundaries and include CPU dispatch. This diagnostic synchronization adds
overhead; disable profiling for normal execution. No explicit profiling
synchronization occurs when disabled. CPU action return completes inference before
the existing post-inference freshness check. Neither max_age_s nor publication
safety checks are relaxed. Faster inference alone may still leave callback/input
ages too large for the current freshness budget.

Benchmark the same checkpoint-bound task and real canonical RGB in a separate
process (no ROS, no checkpoint writes):

```bash
python tools/benchmark_inference.py --checkpoint checkpoint.pt --task-z task_z.pt \
  --references data/zarr --device cuda --warmup 5 --iterations 30 --samples 3
```

The JSON report contains original FP32 and runtime FP32/BF16/FP16 mean/p50/p95 wall
latencies, separately instrumented stage statistics, and maximum/mean normalized
action deviations from original FP32 on the same frames. Loading, RGB disk reads
and decompression are excluded. Each mode warms up; total timings explicitly
synchronize CUDA before/after each call. Stage profiling uses a separate pass and
is not used to calculate speedup. CPU runs report mixed modes unavailable; they
cannot establish GPU performance. Use representative frames/tasks and assess the
reported deviations for your policy: the CUDA unit fixture's .01 normalized error
budget corresponds to at most .016 rad in arm targets or .5 logical hand units;
it is not a universal trained-policy acceptance threshold.

No compile, TensorRT or ONNX path is introduced without GPU measurements showing
it is necessary. If mixed precision remains over budget on the target GPU, use
these stage timings to decide whether optional compilation is worth investigating.

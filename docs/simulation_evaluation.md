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

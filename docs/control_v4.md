# Unified executor v4

BFM main baseline b57716c; interface audited against Synthetic main
c7408e7c276b9f515f1f912064eaf211ff570e8b (read-only reference).
Rebuild/source Synthetic's dynamic_biped interfaces before running ROS.

V3 remains a separate 3.0.0 schema, contract, checkpoint format and 10 Hz path.
There is no migration, resampling or relabeling of existing episodes/checkpoints.
`SharpaV4Contract` preserves physical limits, ordering, units and normalization;
its 4.0.0 dataset contract records 15/100/50/300 Hz and rational policy timing.
The approved executor JSON is copied byte-for-byte into
configs/data_contract/bfm_executor_v4.json; its exact SHA256 must match recorded
execution metadata and every acceptance. An updated executor configuration needs
explicit BFM review, not silent cache/checkpoint reuse.

```bash
python tools/convert_recordings.py /path/to/new_v4_records
# Writes only data/zarr_15hz; fail-closed if any input has another contract.
python tools/convert_recordings.py /path/to/legacy_records --contract v3
# Writes only data/zarr; legacy semantics preserved.
python tools/build_feature_cache.py --help
python tools/train_from_zarr.py data/zarr_15hz --device cuda
```

The supplied training configuration now selects sharpa_v4 and gamma=0.993322.
The trainer still passes the dataset's policy_dt into duration-aware discounting;
0.993322**15 approximately equals 0.99**10. No FB objective/network change.
Explicit v3 training configs remain supported; do not feed v3 stores to a v4 config.
Task/reference tooling accepts either manifest schema, never mixed stores.
No training or conversion of user recordings was run for this implementation.

V4 conversion joins identities to explicit accepted statuses; missing, duplicate,
rejected, wrong-hash or foreign execution evidence fails closed. It validates the
bounded arm trajectory, rounded logical hand forwarding and physical finger
mapping/interpolation rather than requiring arm commands to equal policy targets.
The leading commanded velocity may be unknown at bag start: an admissible velocity
interval is propagated until constrained by observed updates. This does not certify
that the measured robot attained targets. Missing low-level coverage/inhibition
or transport loss fails closed; recording more complete evidence is required.
Same-stamp acceptance/control callback ordering permits the old or new target only
at that simulation instant. Observations always use record time <= both action
and acknowledgement record time; no command-provenance look-ahead selects images.

record_ns/action_header_ns preserve generation clocks; acceptance_record_ns and
acceptance_ns preserve execution evidence. Training timestamps/duration use actual
successive acceptance simulation times, with nominal cadence checks on both
accepted and generated decisions. Session indices and uint64 IDs are stored for
both endpoints. STOP/episode/clock-reset boundaries and invalid ticks split segments.
The validator and dataset fingerprints include the extra numeric evidence; cache
identity already incorporates source episode metadata/content and remains separate.

```bash
python tools/build_task_z.py --help
python tools/run_policy_ros.py --checkpoint v4.pt --task-z task_z.pt --mode shadow
python tools/run_policy_ros.py --checkpoint v4.pt --task-z task_z.pt --mode closed-loop
```

V4 runtime requires a v4-trained bound checkpoint and the existing bound task_z.
Only /bfm/action is published. Fractional logical hand values are preserved;
Synthetic owns integer rounding and all trajectory limiting. Arm-delta CLI overrides
are rejected in v4. Physical/finite/observation and post-inference freshness checks
remain. Use an explicit v3 runtime YAML (control_contract_version: 3, policy_hz: 10,
original three output topics) to retain legacy runtime behavior.

Clock callbacks use rational deadlines: frozen time produces no work, delayed
samples one decision without catch-up. Simulation rollback explicitly stops the
rollout; restart with a fresh session after reset. A monotonic wall watchdog runs
while paused so missing executor acknowledgements cannot be hidden by frozen time.
Each tick logs session/action ID and pending acceptance; execution.jsonl separately
logs accepted/rejected/missing acknowledgements and wall/simulation delays. Final
summary distinguishes published from accepted and pending actions. Bounded runs
wait for the final acknowledgement (or timeout) before shutdown. Acceptance is
never physical attainment. Publisher ownership checks cover the canonical policy
topic; the external executor protects its own low-level outputs.

Generated CDR MCAP, synthetic Zarr, metadata checkpoint and pure scheduler/runtime
unit tests require no ROS or Isaac. Live QoS, rebuilt ROS messages, actual 300 Hz
physics and collection-versus-policy replay remain deployment integration checks.

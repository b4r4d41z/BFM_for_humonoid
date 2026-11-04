# Vanilla FB implementation validation

Development branch: `zarr-pipeline`, tracking `origin/zarr-pipeline`.
Required base: `d7b7e52d2ab33bc70eb8059d3c2fcbc09b7aae2f`.
Initial onboarding checkout before the requested branch switch:
`e5463b2a47b74c927a89a54cc3dba068366c5e6f`.
The implementation commit is recorded in the Git history and final handoff;
`git rev-parse HEAD` identifies the final local revision. No remote push was made.

## Source changes

Created:

* `bfm/contracts/{__init__,kuavo}.py`: physical contracts, measured-state view,
  explicit named right-arm/left-hold/claw conversion.
* `bfm/data/{__init__,schema,validation,zarr_io,dataset,replay_buffer,normalization,recorder}.py`:
  measured Zarr v2 episodes, strict transition checks, manifest splits,
  normalization, replay and direct simulator export.
* `bfm/fb/{__init__,networks,model,losses,agent,context,checkpoint,trainer}.py`:
  actual Vanilla FB networks, objectives, optimization, prompting and persistence.
* `bfm/evaluation.py`: measured reaching/container metrics and autonomous loop.
* `isaaclab_wrapper/{bfm_state,bfm_policy_runner}.py`: articulation extraction,
  pre-reset transition capture and checkpoint-based actor execution.
* `scripts/{collect_bfm_zarr,train_offline_fb,eval_zero_shot_fb}.py`:
  executable collection, training and evaluation entry points.
* `configs/fb/vanilla_offline.yaml`, `requirements-fb.txt`, `pytest.ini`.
* `tests/fb/{conftest,test_data,test_algorithm,test_runtime,test_trainer}.py`.
* `docs/vanilla_fb.md` and this report.

Modified:

* `TestTrainingEnv._get_dones`: opt-in callback after physics and before reset.
* `isaaclab_wrapper/__init__.py`: preserve existing exports with lazy loading;
  new FB runtime imports do not pull in the BC policy.
* `isaaclab_wrapper/policy_scheduler.py`: preserve scheduling behavior/defaults
  while removing the import of BC's temporal constants.
* `.gitignore`: include the new FB test sources.
* `README.md`: link the implemented pipeline and results.

Existing BC policy/data files, legacy tests, joint mapping configuration and
scene setup were not rewritten. Existing FB-CPR stubs are not claimed as working.
Reusable infrastructure and reference algorithm details are listed in
[the implementation guide](vanilla_fb.md).

## Executed checks

Environment: Python 3.12, PyTorch 2.9.1 CPU, NumPy 2.2.6, Zarr 2.18.7,
pytest 8.4.2, PyYAML 6.0.3. CPU test runs use two OpenMP/MKL threads.

| Check | Outcome |
|---|---|
| All 29 new FB tests | Passed |
| Full `python -m pytest -q -p no:cacheprovider` | 41 passed, 1 pre-existing failure |
| Known tabular successor density | Expected TD semi-gradient zero at analytic solution |
| Explicit scalar reference loss comparison | Passed; off-diagonal, diagonal and orthogonality terms checked |
| Gradient checks | Actor updates through F; representation frozen for actor; targets have no gradients |
| Termination/truncation | True termination masks bootstrap; truncation retains duration-adjusted bootstrap |
| Target interpolation | Exact soft-update equation checked |
| CLI training resume | Four updates + resume to eight exactly matches uninterrupted eight-update model |
| Held-out exclusion | Training statistics and reference states use train split only |
| Zarr read/write and invalid-data rejection | Passed |
| Episode-aware sequence sampling | Passed; no reset crossings |
| Action mapping / bounds / left hold | Passed with permuted articulation names |
| Pre-reset capture | Mock automatically resets; stored terminal state/action still retain pre-reset values |
| Frozen runtime and goal/reward prompts | Passed |
| Entrypoint `--help` without Isaac | Passed for all three scripts |
| Dependency consistency (`pip check`) | Passed |
| Actual Isaac launch attempt | Blocked: `ModuleNotFoundError: isaaclab`, then `omni` |

The existing failure is
`tests/test_temporal_contract.py::test_future_targets_are_timestamp_based_half_second_and_do_not_cross_done`.
`bc.temporal.find_future_target_indices` returns a `(1,2)` pair with a 0.1-second
offset for a 0.5-second horizon; its 0.4-second error exceeds the test's 0.3-second
tolerance. This same failure was observed before implementation. Neither that
implementation nor that test was modified, and the new FB path does not use it.

## Training and checkpoint actually produced

A 100-update CLI run used three synthetic toy-dynamics training episodes, each
64 transitions, with separate validation/test episode entries. The toy uses
`dq=0.5*(target-q)` and integrates measured next angles; it is **not KUAVO physics**.

Configuration: batch 64, latent dimension 32, hidden dimension 256, two hidden
layers, seed 5, CPU, default Vanilla FB optimizer/loss settings. All logged
metrics were finite. The checkpoint contains all model/target/optimizer state,
normalization, joint/action contract, reference set and RNG state.

Local artifacts from this run (outside Git):

```text
/workspace/bfm-vanilla-smoke/manifest.json
/workspace/bfm-vanilla-smoke/config.yaml
/workspace/bfm-vanilla-smoke/train.log
/workspace/bfm-vanilla-smoke/run/metrics.jsonl
/workspace/bfm-vanilla-smoke/run/last.pt
/workspace/bfm-vanilla-smoke/mock_eval.json
```

Final update diagnostics:

| Metric | Value |
|---|---:|
| FB TD loss | -8.1130 |
| FB off-diagonal loss | 4.4048 |
| FB diagonal loss | -12.5178 |
| Backward orthogonality loss | -7.4534 |
| Total FB loss | -15.5664 |
| Actor loss | -1.2288 |
| Mean latent norm | 5.65685 = sqrt(32) |
| Mean backward embedding norm | 4.0005 |
| Mean forward prediction norm | 3.0401 |
| Sampled actor action standard deviation | 0.5815 |
| Sampled action boundary fraction | 0.06836 |

Negative losses are valid for the FB estimator with its linear diagonal and
orthogonality corrections. Finite losses alone are not the mathematical
correctness claim: the explicit formula, analytic fixed point and gradient
tests supply additional evidence.

The checkpoint was loaded into the actual `BFMPolicyRunner`, prompted for both
tasks, and executed in four 60-control-step **mock interface** rollouts (two seeds
per task). All finished with finite measurements and correctly mapped actions;
success was **0/4**. The mock has stationary objects and does not model contact
or KUAVO mechanics. These results establish neither reaching skill nor offline
policy quality. This synthetic dataset cannot teach manipulation. The boundary
fraction is a useful diagnostic, not proof of policy failure from extrapolation.

## Remaining research validation

**Actual Isaac physics rollouts executed: zero. Actual manipulation successes:
zero. Real-dataset FB training runs: zero.**

Required inputs are the missing scene/robot USD files, an Isaac Lab/Sim GPU
runtime, validated TCP/left-hold settings, and a new measured simulation dataset
with adequate behavioral coverage. Collection/training/evaluation commands are
implemented and documented, but real integration and research performance are
unverified until those inputs are available.

The four requested arXiv paper texts were blocked by network policy (403).
The official authors' FB implementation and Meta Motivo basic FB model/agent,
network definitions and licenses were accessible and inspected. See the
[source links and access record](vanilla_fb.md#research-sources-and-access-record).
No FB-AW, CPR or BFM-Zero reproduction is claimed.

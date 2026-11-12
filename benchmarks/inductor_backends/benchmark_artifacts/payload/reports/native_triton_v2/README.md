# Native Triton v2 experiment harness

`baseline_source/` was copied before implementation began. Its sibling
`baseline_source_fingerprint.json` records the original overlay files. It holds
all eight compiler modules intercepted by `run_source_overlay.py` and the FlyDSL
source subtree. The baseline always enables the previous native tier and
disables Gluon. Stock and new-native runs use the candidate overlay with their
respective optimization flags.

The runner freezes a separate copy of each compiler overlay plus a shared copy
of the benchmark helpers, worker, source loader, and subprocess startup hook.
Each worker checks those hashes before and after execution and verifies that
all eight compiler modules loaded from its expected snapshot. Each variant runs
in a fresh subprocess, with serial compilation and a separate compiler cache.
The runner serializes GPU use and rotates variant order deterministically across
cases. It supports strict resume of completed results under the same manifest.

## Cases and execution

`cases.json` contains 142 cases: 16 original anchors and 126 heldout cases.
The heldout matrix spans reduction widths 127 through 32768, odd rows 37/127,
tall matrices, few-row and many-row wide matrices, pointwise tails, and 2D/batched
transposes with FP32 and BF16 source tensors. Several workloads promote math
to FP32; softmax_backward specifically builds FP32 probabilities and gradients
before compilation for both dtype labels, preserving the original benchmark
protocol. Its labels refer to the dtype used to generate source scores. `cases_smoke.json` selects 20 cases. Any independent
JSON list with the same case schema can be supplied with `--cases`; inputs and
case names become part of the frozen protocol.

After the compiler source freeze, from this directory:

```bash
bash ../run_env.sh run.py --cases cases_smoke.json --output results_smoke --rounds 5 --repeats 5 --gpu 3 --freeze-only
bash ../run_env.sh run.py --cases cases_smoke.json --output results_smoke --rounds 5 --repeats 5 --gpu 3
bash ../run_env.sh summarize.py --output results_smoke
```

The first command copies and fingerprints sources without initializing CUDA.
The second launches measurements. Omitting `--freeze-only` on the first command
combines those steps. Use a fresh output directory for a changed case selection,
source snapshot, rounds, repeats, or GPU. `--candidate-root` and `--baseline-root`
can override the source trees.

## Interpretation

The headline combines equal-weight cold/warm measurements on common successful
cases across all three variants. Speedup is
`100 * (geomean(baseline_runtime / new_native_runtime) - 1)`. Anchor and heldout
summaries are separate. Captured direct GPU sequences exclude host dispatch;
inline graph replay is reported separately because host launch gaps can affect
its event interval. Preparation follows protocol v2 and stays outside the timed
interval. Failures remain in the reports and are excluded consistently from
both baseline comparisons.

Outputs include `raw.csv`, `per_case.csv`, `per_op.csv`, `summary.json`,
`failures.json`, `per_kernel.csv`, `per_candidate.csv`, and
`compile_tune_costs.csv`. Individual worker results retain source fingerprints,
loaded module paths, tuning diagnostics, and persistent-versus-looped MultiKernel
choices. Resource profiles include registers, spills and shared memory; the
`selected_for_execution` field distinguishes compiled MultiKernel alternatives
from the selected implementation.

`compile_validate_capture_wall_seconds` includes compilation, autotuning,
correctness checks and capture setup. Tuner objects provide initial-precompile
and overall-autotune counters for both source tiers. The latter includes neighbor
compilation, benchmarking and validation. New-source diagnostic records separate
accounted compiler and tuner phase costs, but the reported precompile counter
can omit asynchronous initial compilation. The accounted compiler value therefore
must not be presented as complete initial compilation wall time. These fields stay null for the previous source
when its instrumentation cannot separate them. None of these costs are added to
steady-state runtime measurements.

The new integration suite is
`pytorch_triton_only/test/inductor/test_amd_native_dynamic.py`. It compares eager
and native outputs while checking compilation counts and guard failures across
dynamic rows, pointwise tails, and 2D/batched transposes. Run that suite on the
separate numeric-validation GPU after the candidate source freeze.

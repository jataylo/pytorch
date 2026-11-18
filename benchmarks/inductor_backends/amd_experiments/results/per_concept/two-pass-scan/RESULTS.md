# two-pass-scan

Compute chunk totals and prefixes in FP64 across two kernels, then cast to FP32. This changes accumulation precision and is experimental.

Independent branch: `experiment/amd-lowering-two-pass-scan`. Parent: `experiment/amd-lowering-bench-base`, based on upstream `dddd90223f0`. No other optimization branch is required. The disabled arm uses stock lowering; see `concept.py` for the single enabled configuration and workload shapes.

Measured on one MI355X, physical GPU 3: 8 valid pairs, 8 exercising this feature, 0 failed cases. Warm captured GPU sequence geomean speedup: **+91.13%**, best +178.08%, worst +27.53%. Wins/regressions outside a conservative 2% band: 8/0. These are operator units, not model speedups.

| Case | Stock us | Feature us | Speedup | Paired 95% interval | Feature exercised |
|---|---:|---:|---:|---|---|
| cumsum_1x131072_bfloat16 | 15.756 | 9.430 | +67.08% | +64.36% to +69.86% | True |
| cumsum_1x131072_float32 | 15.472 | 9.506 | +62.76% | +60.80% to +66.59% | True |
| cumsum_2x262144_bfloat16 | 19.410 | 9.890 | +96.26% | +93.16% to +98.46% | True |
| cumsum_2x262144_float32 | 19.372 | 10.076 | +92.26% | +88.41% to +93.10% | True |
| cumsum_8x131073_bfloat16 | 17.938 | 14.066 | +27.53% | +26.34% to +28.69% | True |
| cumsum_8x131073_float32 | 25.614 | 14.366 | +78.30% | +77.62% to +82.02% | True |
| cumsum_8x262145_bfloat16 | 48.714 | 17.518 | +178.08% | +176.57% to +179.44% | True |
| cumsum_8x262145_float32 | 48.378 | 17.628 | +174.44% | +170.69% to +176.03% | True |

`results/comparison.csv` includes all four invocation/cache protocols, compile costs, source hashes and kernel counts. `results/summary.csv` contains aggregate statistics. Per-case `result.json` preserves every timing sample; `*.source.txt` files preserve the measured generated code. Intervals resample paired timing rounds and do not include variation between separate compilations. Small gains and candidate-selection instability need repeat trials. Failed or ineligible cases are not credited as lowering wins.

Run the commands in `results/run.sh`; validation/lint commands and outputs are in `results/validation/`. Benchmarks check three input seeds; scans use a FP64-prefix reference.

**Precision contract:** FP64 partials intentionally differ from the stock FP32 scan tree, especially for overflow and cancellation. Nonnegative performance inputs are used in both arms; signed, nonfinite, overflow, stride and exclusion behavior are covered by the unit tests. This branch requires a separate numerical-semantics decision before upstreaming.

## Upstream priority

Last for upstream readiness despite strong performance: explicit FP64 accumulation changes overflow, cancellation and rounding semantics.

All compiler options are default-off. Source-transform branches use the existing stock autotuner and do not include the kernel-selection experiment. Kernel counts count generated definitions, including unused MultiKernel alternatives; they are not runtime launch counts. Compile times are elapsed times with a reused compiler cache, not cold-build comparisons.

## Agent details

Prepared with an AI assistant for human review before upstream submission.

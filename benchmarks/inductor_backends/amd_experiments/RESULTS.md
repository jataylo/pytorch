# wide-reduction-splits

Create or refine parallel partial reductions for underfilled wide rows with coupled broadcast epilogues; retain the existing reduction arithmetic.

Independent branch: `experiment/amd-lowering-wide-reduction-splits`. Parent: `experiment/amd-lowering-bench-base`, based on upstream `dddd90223f0`. No other optimization branch is required. The disabled arm uses stock lowering; see `concept.py` for the single enabled configuration and workload shapes.

Measured on one MI355X, physical GPU 3: 8 valid pairs, 4 exercising this feature, 0 failed cases. Warm captured GPU sequence geomean speedup: **+21.62%**, best +57.08%, worst -0.93%. Wins/regressions outside a conservative 2% band: 5/0. These are operator units, not model speedups.

| Case | Stock us | Feature us | Speedup | Paired 95% interval | Feature exercised |
|---|---:|---:|---:|---|---|
| norm_backward_63x65535_bfloat16 | 24.142 | 24.026 | +0.48% | -0.33% to +1.14% | False |
| norm_backward_63x65535_float32 | 26.678 | 26.558 | +0.45% | -0.42% to +1.11% | False |
| norm_backward_8x131072_bfloat16 | 13.898 | 13.568 | +2.43% | +1.07% to +3.73% | False |
| norm_backward_8x131072_float32 | 15.088 | 15.230 | -0.93% | -2.33% to -0.17% | False |
| softmax_2x32769_bfloat16 | 15.772 | 11.800 | +33.66% | +29.14% to +37.18% | True |
| softmax_2x32769_float32 | 16.182 | 11.314 | +43.03% | +41.28% to +45.28% | True |
| softmax_8x65535_bfloat16 | 19.424 | 12.366 | +57.08% | +55.74% to +58.98% | True |
| softmax_8x65535_float32 | 20.146 | 12.944 | +55.64% | +53.46% to +57.98% | True |

`results/comparison.csv` includes all four invocation/cache protocols, compile costs, source hashes and kernel counts. `results/summary.csv` contains aggregate statistics. Per-case `result.json` preserves every timing sample; `*.source.txt` files preserve the measured generated code. Intervals resample paired timing rounds and do not include variation between separate compilations. Small gains and candidate-selection instability need repeat trials. Failed or ineligible cases are not credited as lowering wins.

Run the commands in `results/run.sh`; validation/lint commands and outputs are in `results/validation/`. Benchmarks check three input seeds; scans use a FP64-prefix reference.

## Upstream priority

Highest priority: clear gain in the four exercised cases; four unchanged controls. Broaden coupled-epilogue and cross-device coverage.

All compiler options are default-off. Source-transform branches use the existing stock autotuner and do not include the kernel-selection experiment. Kernel counts count generated definitions, including unused MultiKernel alternatives; they are not runtime launch counts. Compile times are elapsed times with a reused compiler cache, not cold-build comparisons.

## Agent details

Prepared with an AI assistant for human review before upstream submission.

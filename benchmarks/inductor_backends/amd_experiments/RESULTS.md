# kernel-selection

Expand generic reduction configurations and use bounded, mutation-safe paired tuning and confirmation; leave emitted kernel math unchanged.

Independent branch: `experiment/amd-lowering-kernel-selection`. Parent: `experiment/amd-lowering-bench-base`, based on upstream `dddd90223f0`. No other optimization branch is required. The disabled arm uses stock lowering; see `concept.py` for the single enabled configuration and workload shapes.

Measured on one MI355X, physical GPU 3: 8 valid pairs, 8 exercising this feature, 0 failed cases. Warm captured GPU sequence geomean speedup: **+0.28%**, best +2.94%, worst -2.18%. Wins/regressions outside a conservative 2% band: 2/1. These are operator units, not model speedups.

| Case | Stock us | Feature us | Speedup | Paired 95% interval | Feature exercised |
|---|---:|---:|---:|---|---|
| softmax_37x769_bfloat16 | 6.924 | 6.998 | -1.06% | -2.25% to +1.80% | True |
| softmax_37x769_float32 | 7.086 | 6.976 | +1.58% | -2.46% to +2.50% | True |
| sum_1024x128_bfloat16 | 7.102 | 6.934 | +2.42% | -0.91% to +5.27% | True |
| sum_1024x128_float32 | 6.972 | 6.992 | -0.29% | -1.93% to +4.16% | True |
| sum_64x4099_bfloat16 | 6.850 | 6.934 | -1.21% | -5.29% to +0.40% | True |
| sum_64x4099_float32 | 6.854 | 6.844 | +0.15% | -5.46% to +2.03% | True |
| swiglu_4608000_bfloat16 | 9.864 | 10.084 | -2.18% | -3.19% to -0.18% | True |
| swiglu_4608000_float32 | 12.268 | 11.918 | +2.94% | +0.90% to +4.54% | True |

`results/comparison.csv` includes all four invocation/cache protocols, compile costs, source hashes and kernel counts. `results/summary.csv` contains aggregate statistics. Per-case `result.json` preserves every timing sample; `*.source.txt` files preserve the measured generated code. Intervals resample paired timing rounds and do not include variation between separate compilations. Small gains and candidate-selection instability need repeat trials. Failed or ineligible cases are not credited as lowering wins.

Run the commands in `results/run.sh`; validation/lint commands and outputs are in `results/validation/`. Benchmarks check three input seeds; scans use a FP64-prefix reference.

## Upstream priority

Reusable infrastructure, but this stock-math-only sample is nearly flat and adds tuning cost. Evaluate broader workloads before upstreaming.

All compiler options are default-off. Source-transform branches use the existing stock autotuner and do not include the kernel-selection experiment. Kernel counts count generated definitions, including unused MultiKernel alternatives; they are not runtime launch counts. Compile times are elapsed times with a reused compiler cache, not cold-build comparisons.

## Agent details

Prepared with an AI assistant for human review before upstream submission.

# hierarchical-scan

Compare additive FP32 chunk scans plus preceding-chunk offsets with the original associative scan, keeping launch geometry fixed.

Independent branch: `experiment/amd-lowering-hierarchical-scan`. Parent: `experiment/amd-lowering-bench-base`, based on upstream `dddd90223f0`. No other optimization branch is required. The disabled arm uses stock lowering; see `concept.py` for the single enabled configuration and workload shapes.

Measured on one MI355X, physical GPU 3: 8 valid pairs, 6 exercising this feature, 0 failed cases. Warm captured GPU sequence geomean speedup: **-0.08%**, best +2.45%, worst -2.60%. Wins/regressions outside a conservative 2% band: 1/1. These are operator units, not model speedups.

| Case | Stock us | Feature us | Speedup | Paired 95% interval | Feature exercised |
|---|---:|---:|---:|---|---|
| cumsum_2x32769_bfloat16 | 13.112 | 12.868 | +1.90% | +0.11% to +3.19% | False |
| cumsum_2x32769_float32 | 18.488 | 18.748 | -1.39% | -2.78% to -0.95% | False |
| cumsum_2x4099_bfloat16 | 7.362 | 7.186 | +2.45% | -2.86% to +5.54% | True |
| cumsum_2x4099_float32 | 7.522 | 7.640 | -1.54% | -3.32% to +2.20% | True |
| cumsum_8x32768_bfloat16 | 15.354 | 15.346 | +0.05% | -0.58% to +1.37% | True |
| cumsum_8x32768_float32 | 15.160 | 15.086 | +0.49% | -0.13% to +1.85% | True |
| cumsum_8x8193_bfloat16 | 10.120 | 10.390 | -2.60% | -3.46% to -0.80% | True |
| cumsum_8x8193_float32 | 10.168 | 10.154 | +0.14% | -1.39% to +1.80% | True |

`results/comparison.csv` includes all four invocation/cache protocols, compile costs, source hashes and kernel counts. `results/summary.csv` contains aggregate statistics. Per-case `result.json` preserves every timing sample; `*.source.txt` files preserve the measured generated code. Intervals resample paired timing rounds and do not include variation between separate compilations. Small gains and candidate-selection instability need repeat trials. Failed or ineligible cases are not credited as lowering wins.

Run the commands in `results/run.sh`; validation/lint commands and outputs are in `results/validation/`. Benchmarks check three input seeds; scans use a FP64-prefix reference.

## Upstream priority

Hold: no convincing isolated speedup on these units, despite retaining original-tree candidates. Do not credit combined-branch scan gains to this pass.

All compiler options are default-off. Source-transform branches use the existing stock autotuner and do not include the kernel-selection experiment. Kernel counts count generated definitions, including unused MultiKernel alternatives; they are not runtime launch counts. Compile times are elapsed times with a reused compiler cache, not cold-build comparisons.

## Agent details

Prepared with an AI assistant for human review before upstream submission.

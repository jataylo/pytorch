# transpose-tiling

Compare two-dimensional transpose tiles and read orders with the original flat copy, proving strides and masks.

Independent branch: `experiment/amd-lowering-transpose-tiling`. Parent: `experiment/amd-lowering-bench-base`, based on upstream `dddd90223f0`. No other optimization branch is required. The disabled arm uses stock lowering; see `concept.py` for the single enabled configuration and workload shapes.

Measured on one MI355X, physical GPU 3: 16 valid pairs, 10 exercising this feature, 0 failed cases. Warm captured GPU sequence geomean speedup: **+0.64%**, best +3.25%, worst -1.63%. Wins/regressions outside a conservative 2% band: 5/0. These are operator units, not model speedups.

| Case | Stock us | Feature us | Speedup | Paired 95% interval | Feature exercised |
|---|---:|---:|---:|---|---|
| transpose_1x2048x1024x1_bfloat16 | 7.292 | 7.316 | -0.33% | -2.81% to +2.08% | False |
| transpose_1x2048x1024x1_float32 | 7.822 | 7.918 | -1.21% | -2.63% to +1.75% | False |
| transpose_1x513x257x1_bfloat16 | 7.030 | 7.066 | -0.51% | -2.12% to +2.41% | True |
| transpose_1x513x257x1_float32 | 6.978 | 7.090 | -1.58% | -2.39% to +1.22% | True |
| transpose_1x8x64x128_bfloat16 | 6.844 | 6.906 | -0.90% | -4.33% to -0.64% | False |
| transpose_1x8x64x128_float32 | 6.914 | 6.936 | -0.32% | -3.36% to +0.99% | False |
| transpose_4x1024x1024x1_bfloat16 | 7.940 | 7.780 | +2.06% | -2.89% to +3.98% | False |
| transpose_4x1024x1024x1_float32 | 9.514 | 9.522 | -0.08% | -1.39% to +1.55% | False |
| transpose_4x128x256x4_bfloat16 | 7.210 | 7.022 | +2.68% | -3.06% to +3.21% | True |
| transpose_4x128x256x4_float32 | 7.340 | 7.226 | +1.58% | -1.41% to +4.19% | True |
| transpose_4x16x65x129_bfloat16 | 6.998 | 7.114 | -1.63% | -3.47% to +2.01% | True |
| transpose_4x16x65x129_float32 | 7.296 | 7.172 | +1.73% | -0.16% to +2.99% | True |
| transpose_4x4x127x65_bfloat16 | 7.070 | 6.866 | +2.97% | -1.06% to +4.97% | True |
| transpose_4x4x127x65_float32 | 7.182 | 6.956 | +3.25% | +0.29% to +5.46% | True |
| transpose_8x32x128x64_bfloat16 | 7.314 | 7.292 | +0.30% | -0.52% to +2.32% | True |
| transpose_8x32x128x64_float32 | 7.766 | 7.580 | +2.45% | +0.52% to +4.35% | True |

`results/comparison.csv` includes all four invocation/cache protocols, compile costs, source hashes and kernel counts. `results/summary.csv` contains aggregate statistics. Per-case `result.json` preserves every timing sample; `*.source.txt` files preserve the measured generated code. Intervals resample paired timing rounds and do not include variation between separate compilations. Small gains and candidate-selection instability need repeat trials. Failed or ineligible cases are not credited as lowering wins.

Run the commands in `results/run.sh`; validation/lint commands and outputs are in `results/validation/`. Benchmarks check three input seeds; scans use a FP64-prefix reference.

## Upstream priority

Experimental: small gain, largely within the 2% band; six cases do not match the transform. Needs stronger independent evidence.

All compiler options are default-off. Source-transform branches use the existing stock autotuner and do not include the kernel-selection experiment. Kernel counts count generated definitions, including unused MultiKernel alternatives; they are not runtime launch counts. Compile times are elapsed times with a reused compiler cache, not cold-build comparisons.

## Agent details

Prepared with an AI assistant for human review before upstream submission.

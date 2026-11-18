# persistent-selection

Use live-value and hardware resource estimates to admit persistent kernels, then compare with looped MultiKernel alternatives.

Independent branch: `experiment/amd-lowering-persistent-selection`. Parent: `experiment/amd-lowering-bench-base`, based on upstream `dddd90223f0`. No other optimization branch is required. The disabled arm uses stock lowering; see `concept.py` for the single enabled configuration and workload shapes.

Measured on one MI355X, physical GPU 3: 8 valid pairs, 8 exercising this feature, 0 failed cases. Warm captured GPU sequence geomean speedup: **+9.31%**, best +32.44%, worst -1.27%. Wins/regressions outside a conservative 2% band: 4/0. These are operator units, not model speedups.

| Case | Stock us | Feature us | Speedup | Paired 95% interval | Feature exercised |
|---|---:|---:|---:|---|---|
| rms_64x32768_bfloat16 | 7.956 | 8.058 | -1.27% | -2.68% to +2.68% | True |
| rms_64x32768_float32 | 9.208 | 8.388 | +9.78% | +6.83% to +12.05% | True |
| rms_8x16384_bfloat16 | 7.020 | 6.964 | +0.80% | -2.61% to +3.29% | True |
| rms_8x16384_float32 | 6.900 | 6.886 | +0.20% | -1.57% to +4.36% | True |
| softmax_64x8192_bfloat16 | 9.448 | 7.134 | +32.44% | +30.52% to +36.70% | True |
| softmax_64x8192_float32 | 9.540 | 7.450 | +28.05% | +23.72% to +30.57% | True |
| softmax_8x4096_bfloat16 | 7.368 | 7.230 | +1.91% | -1.91% to +3.41% | True |
| softmax_8x4096_float32 | 7.384 | 6.854 | +7.73% | +5.05% to +8.87% | True |

`results/comparison.csv` includes all four invocation/cache protocols, compile costs, source hashes and kernel counts. `results/summary.csv` contains aggregate statistics. Per-case `result.json` preserves every timing sample; `*.source.txt` files preserve the measured generated code. Intervals resample paired timing rounds and do not include variation between separate compilations. Small gains and candidate-selection instability need repeat trials. Failed or ineligible cases are not credited as lowering wins.

Run the commands in `results/run.sh`; validation/lint commands and outputs are in `results/validation/`. Benchmarks check three input seeds; scans use a FP64-prefix reference.

## Upstream priority

High priority: clear selected-case benefit. Review MultiKernel compile overhead and broader memory-pressure behavior.

All compiler options are default-off. Source-transform branches use the existing stock autotuner and do not include the kernel-selection experiment. Kernel counts count generated definitions, including unused MultiKernel alternatives; they are not runtime launch counts. Compile times are elapsed times with a reused compiler cache, not cold-build comparisons.

## Agent details

Prepared with an AI assistant for human review before upstream submission.

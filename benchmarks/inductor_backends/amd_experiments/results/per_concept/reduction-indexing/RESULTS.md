# reduction-indexing

Remove an integer modulo only when the load mask proves that its nonnegative index is already below the modulus.

Independent branch: `experiment/amd-lowering-reduction-indexing`. Parent: `experiment/amd-lowering-bench-base`, based on upstream `dddd90223f0`. No other optimization branch is required. The disabled arm uses stock lowering; see `concept.py` for the single enabled configuration and workload shapes.

Measured on one MI355X, physical GPU 3: 8 valid pairs, 8 exercising this feature, 0 failed cases. Warm captured GPU sequence geomean speedup: **+7.74%**, best +23.85%, worst -14.35%. Wins/regressions outside a conservative 2% band: 5/1. These are operator units, not model speedups.

| Case | Stock us | Feature us | Speedup | Paired 95% interval | Feature exercised |
|---|---:|---:|---:|---|---|
| masked_gather_sum_4x32769_bfloat16 | 10.570 | 8.632 | +22.45% | +20.80% to +24.08% | True |
| masked_gather_sum_4x32769_float32 | 9.018 | 8.278 | +8.94% | +7.09% to +12.03% | True |
| masked_gather_sum_4x4099_bfloat16 | 8.644 | 7.012 | +23.27% | +21.41% to +26.28% | True |
| masked_gather_sum_4x4099_float32 | 6.722 | 7.848 | -14.35% | -16.34% to -12.28% | True |
| masked_gather_sum_8x32769_bfloat16 | 11.144 | 8.998 | +23.85% | +20.94% to +28.63% | True |
| masked_gather_sum_8x32769_float32 | 8.988 | 8.490 | +5.87% | +3.91% to +7.65% | True |
| masked_gather_sum_8x4099_bfloat16 | 6.786 | 6.922 | -1.96% | -4.99% to -1.02% | True |
| masked_gather_sum_8x4099_float32 | 7.008 | 6.986 | +0.32% | -1.17% to +5.36% | True |

`results/comparison.csv` includes all four invocation/cache protocols, compile costs, source hashes and kernel counts. `results/summary.csv` contains aggregate statistics. Per-case `result.json` preserves every timing sample; `*.source.txt` files preserve the measured generated code. Intervals resample paired timing rounds and do not include variation between separate compilations. Small gains and candidate-selection instability need repeat trials. Failed or ineligible cases are not credited as lowering wins.

Run the commands in `results/run.sh`; validation/lint commands and outputs are in `results/validation/`. Benchmarks check three input seeds; scans use a FP64-prefix reference.

## Upstream priority

Useful narrow proof, but one substantial regression. Investigate launch selection before upstream default enablement.

All compiler options are default-off. Source-transform branches use the existing stock autotuner and do not include the kernel-selection experiment. Kernel counts count generated definitions, including unused MultiKernel alternatives; they are not runtime launch counts. Compile times are elapsed times with a reused compiler cache, not cold-build comparisons.

## Agent details

Prepared with an AI assistant for human review before upstream submission.

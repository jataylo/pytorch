# row-packing

Prove row-tail masks before admitting larger row groups and device-bounded warp configurations to persistent reductions.

Independent branch: `experiment/amd-lowering-row-packing`. Parent: `experiment/amd-lowering-bench-base`, based on upstream `dddd90223f0`. No other optimization branch is required. The disabled arm uses stock lowering; see `concept.py` for the single enabled configuration and workload shapes.

Measured on one MI355X, physical GPU 3: 12 valid pairs, 12 exercising this feature, 0 failed cases. Warm captured GPU sequence geomean speedup: **+0.18%**, best +3.13%, worst -3.08%. Wins/regressions outside a conservative 2% band: 2/2. These are operator units, not model speedups.

| Case | Stock us | Feature us | Speedup | Paired 95% interval | Feature exercised |
|---|---:|---:|---:|---|---|
| rms_37x256_bfloat16 | 7.020 | 6.932 | +1.27% | -3.24% to +2.84% | True |
| rms_37x256_float32 | 6.962 | 6.810 | +2.23% | +0.71% to +5.36% | True |
| rms_4096x769_bfloat16 | 7.708 | 7.710 | -0.03% | -1.99% to +1.86% | True |
| rms_4096x769_float32 | 8.652 | 8.842 | -2.15% | -4.27% to +0.14% | True |
| softmax_37x128_bfloat16 | 7.042 | 6.828 | +3.13% | -2.62% to +5.15% | True |
| softmax_37x128_float32 | 6.956 | 6.912 | +0.64% | -6.78% to +1.63% | True |
| softmax_37x769_bfloat16 | 7.064 | 7.136 | -1.01% | -3.66% to +5.41% | True |
| softmax_37x769_float32 | 6.992 | 7.214 | -3.08% | -5.49% to +4.91% | True |
| softmax_4096x128_bfloat16 | 7.096 | 6.968 | +1.84% | -1.42% to +5.18% | True |
| softmax_4096x128_float32 | 7.230 | 7.260 | -0.41% | -4.35% to +2.92% | True |
| sum_37x1024_bfloat16 | 6.944 | 6.838 | +1.55% | -2.06% to +6.81% | True |
| sum_37x1024_float32 | 6.840 | 6.954 | -1.64% | -2.93% to +1.07% | True |

`results/comparison.csv` includes all four invocation/cache protocols, compile costs, source hashes and kernel counts. `results/summary.csv` contains aggregate statistics. Per-case `result.json` preserves every timing sample; `*.source.txt` files preserve the measured generated code. Intervals resample paired timing rounds and do not include variation between separate compilations. Small gains and candidate-selection instability need repeat trials. Failed or ineligible cases are not credited as lowering wins.

Run the commands in `results/run.sh`; validation/lint commands and outputs are in `results/validation/`. Benchmarks check three input seeds; scans use a FP64-prefix reference.

## Upstream priority

Hold: independently near-flat. Any synergy with the new selector requires a separate factorial experiment.

All compiler options are default-off. Source-transform branches use the existing stock autotuner and do not include the kernel-selection experiment. Kernel counts count generated definitions, including unused MultiKernel alternatives; they are not runtime launch counts. Compile times are elapsed times with a reused compiler cache, not cold-build comparisons.

## Agent details

Prepared with an AI assistant for human review before upstream submission.

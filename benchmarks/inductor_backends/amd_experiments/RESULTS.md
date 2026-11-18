# pointwise-subtiling

Process several smaller pointwise tiles per program, reusing scalar loads and arithmetic; retain flat candidates.

Independent branch: `experiment/amd-lowering-pointwise-subtiling`. Parent: `experiment/amd-lowering-bench-base`, based on upstream `dddd90223f0`. No other optimization branch is required. The disabled arm uses stock lowering; see `concept.py` for the single enabled configuration and workload shapes.

Measured on one MI355X, physical GPU 3: 12 valid pairs, 12 exercising this feature, 0 failed cases. Warm captured GPU sequence geomean speedup: **-0.07%**, best +13.70%, worst -12.64%. Wins/regressions outside a conservative 2% band: 2/2. These are operator units, not model speedups.

| Case | Stock us | Feature us | Speedup | Paired 95% interval | Feature exercised |
|---|---:|---:|---:|---|---|
| cast_add_4608000_bfloat16 | 7.992 | 9.148 | -12.64% | -13.29% to -10.87% | True |
| cast_add_4608000_float32 | 11.528 | 12.046 | -4.30% | -6.49% to -2.04% | True |
| scalar_16777216_bfloat16 | 13.468 | 13.464 | +0.03% | -1.83% to +1.69% | True |
| scalar_16777216_float32 | 22.322 | 22.510 | -0.84% | -1.13% to -0.04% | True |
| scalar_4099_bfloat16 | 6.856 | 6.890 | -0.49% | -1.81% to +2.66% | True |
| scalar_4099_float32 | 6.816 | 6.730 | +1.28% | -1.35% to +2.78% | True |
| scalar_4608000_bfloat16 | 8.008 | 7.818 | +2.43% | -1.67% to +4.75% | True |
| scalar_4608000_float32 | 9.270 | 9.344 | -0.79% | -2.87% to -0.00% | True |
| swiglu_16777216_bfloat16 | 22.044 | 19.388 | +13.70% | +9.87% to +17.20% | True |
| swiglu_16777216_float32 | 31.918 | 31.412 | +1.61% | +0.60% to +2.14% | True |
| swiglu_4608000_bfloat16 | 9.788 | 9.736 | +0.53% | -1.88% to +1.58% | True |
| swiglu_4608000_float32 | 11.880 | 11.818 | +0.52% | -0.73% to +3.17% | True |

`results/comparison.csv` includes all four invocation/cache protocols, compile costs, source hashes and kernel counts. `results/summary.csv` contains aggregate statistics. Per-case `result.json` preserves every timing sample; `*.source.txt` files preserve the measured generated code. Intervals resample paired timing rounds and do not include variation between separate compilations. Small gains and candidate-selection instability need repeat trials. Failed or ineligible cases are not credited as lowering wins.

Run the commands in `results/run.sh`; validation/lint commands and outputs are in `results/validation/`. Benchmarks check three input seeds; scans use a FP64-prefix reference.

## Upstream priority

Hold: near-flat aggregate masks both a 13.70% win and a 12.64% regression. Further geometry selection and separate recompilation trials are needed.

All compiler options are default-off. Source-transform branches use the existing stock autotuner and do not include the kernel-selection experiment. Kernel counts count generated definitions, including unused MultiKernel alternatives; they are not runtime launch counts. Compile times are elapsed times with a reused compiler cache, not cold-build comparisons.

## Agent details

Prepared with an AI assistant for human review before upstream submission.

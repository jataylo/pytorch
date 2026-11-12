> Historical initial native-Triton experiment. For the final 142-case results,
> validated source changes and PR plan, see
> [the current change description](TRITON_CHANGE_DESCRIPTION_AND_PR_PLAN.md)
> and [the complete artifact catalog](benchmark_artifacts/README.md).

# Native Triton experiment

For the per-lowering description, ranked follow-up opportunities and proposed
PR dependencies, see [Triton changes and PR plan](TRITON_CHANGE_DESCRIPTION_AND_PR_PLAN.md).

The parent commit, `108eae3a723` on `flydsl/triton-hillclimb-gluon`, preserves
Gluon specializations and the FlyDSL backend work. This branch,
`flydsl/triton-hillclimb-triton`, tests the same recognized operations using
`triton.language` and Triton's inferred layouts.

## Selecting the implementation

AMD optimization flags remain opt-in. When enabled, `triton.amd_use_gluon=False`
(the default on this branch) emits native Triton kernels. Set it to `True` to
compare against the parent Gluon implementation. Approximate math retains its
independent opt-in flag. GEMM and convolution benchmark backends remain ATen.

## What carries over

The guarded AST recognizers, scalar broadcast simplification, cast/add
specializations, flat pointwise fusion, static unrolling, transpose traversal
choices, reduction math, masked modulo proof, and optional GELU math remain
shared. A final AST projection changes the selected specializations to
`triton.language`, removes layout construction and layout-only parameters,
and uses `@triton.jit`. The code is compiled through the normal Triton ASTSource
path. Metadata records `amd_native_family` and `use_gluon=False`.

Candidate configurations retain block sizes, warps, unrolling and traversal
order. Candidates that differ only by removed Gluon layouts are deduplicated.
The bounded hill climber, alias-preserving validation, cache contexts,
confirmation rounds, hardware limits and diagnostic traces remain enabled.

Triton infers lane/vector/warp layout. This experiment cannot reproduce every
Gluon layout by deleting its annotations. It measures whether source choices
and launch tuning recover the gains, and retains the Gluon switch for cases
where explicit layout control wins.

Native Triton also supports `tl.PropagateNan.ALL`, which was missing from
Gluon and caused XGLM and MobileNetV3 compilation failures in the frozen parent
model run. Unit tests cover generated native NaN minimum/maximum code; a full
model rerun is a separate validation step.

## Experiment artifacts

The local experiment is in
`/dockerx/pytorch-flex/agent_space/simd_bench/native_triton_experiment/`.
`run.py` compares stock Triton, Gluon specializations, and native Triton
specializations on eight operation families in BF16 and FP32. It validates
multiple input seeds before timing, uses one MI355X, and measures warm/cold
cache contexts with protocol v2. Captured GPU sequences and host-inclusive
inline replay are reported separately. `summarize.py` generates comparison
CSVs, summaries, and an SVG chart from complete per-case result snapshots.

New tests live in `test/inductor/test_amd_native_triton.py`. Existing tests
whose assertions specifically require Gluon select that backend explicitly.

## Final focused results

All 182 focused tests passed (151 retained tests and 31 new native tests).
The final source fingerprint matched after measurement. All 16 operation cases
passed across three variants, producing 192 measurements and zero failures.
These are operation workloads, not full-model training results.

For captured GPU sequences, with equal weights for warm/cold contexts and each
case, native Triton achieved a 1.1431x speedup over stock Triton. It was 0.2%
slower than the Gluon specializations in aggregate. The earlier pass measured
1.1054x over stock and 0.8% faster than Gluon; small native/Gluon differences
should be treated as parity on this sample.

| Family (BF16 and FP32) | Native speedup vs stock | Native speedup vs Gluon |
| --- | ---: | ---: |
| Cast/add | 14.43% | -1.11% |
| Broadcast | 6.09% | 2.19% |
| Transpose | 7.83% | -0.28% |
| Softmax | 5.69% | 3.28% |
| Softmax backward | 15.83% | -4.15% |
| Norm backward | 4.26% | 0.64% |
| Cross-entropy backward | 12.85% | 1.85% |
| Cumsum | 54.61% | -3.77% |

Speedup is `100 * (baseline_runtime / native_runtime - 1)`. Negative values
favor the baseline. Layout inference recovers much of the gain in this focused
sample, while explicit layouts retain an advantage for softmax backward and
wide scans. Broader shapes, other hardware, and full models need separate
validation before enabling this path by default in a general release.

Final artifacts:

- `native_triton_experiment/results/per_case.csv`: each case/cache/invocation.
- `native_triton_experiment/results/per_op.csv`: operation aggregations.
- `native_triton_experiment/results/summary.json`: coverage and geometric means.
- `native_triton_experiment/results/runtime_chart.svg`: captured GPU timings.
- `native_triton_experiment/results/graph_replay_runtime_chart.svg`: replay
  timings including host launch effects.
- `native_triton_experiment/final_source_fingerprint.json`: measured sources.
- `native_triton_experiment/validation/test_results.json`: focused test results.

These paths are relative to `/dockerx/pytorch-flex/agent_space/simd_bench/`.
The earlier pass is retained as `native_triton_experiment/results_round1/`.

The reusable branch runner uses the same workloads and measurement helper:

```bash
python benchmarks/inductor_backends/amd_native_triton_hillclimb.py \
  --output agent_space/native_triton_results --rounds 5 --repeats 5
```

Run with the branch's built PyTorch or the local source overlay described above.
The result directory must be empty. The runner stores a source manifest,
per-case results, raw CSV and all failures. Its syntax, CLI imports and scoped
lint were checked; the timed results above came from the equivalent local
runner rather than an additional measurement pass.

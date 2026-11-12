# AMD Inductor Triton changes and proposed PR split

## Summary

This change updates Inductor's native Triton pointwise, transpose and reduction
paths on AMD GPUs. It adds guarded source specializations, geometry- and
device-derived launch candidates, resource-aware reduction decisions and
validated bounded hill climbing. Generated kernels use `triton.language`,
`@triton.jit` and the normal Triton compiler interfaces.

The baseline is standard Inductor Triton lowering and autotuning with these AMD
optimizations disabled in the same checkout. The contribution changes Inductor
codegen and runtime policy; it does not modify the Triton compiler's MLIR passes.
The exact upstream base should be fixed when extracting the proposed PRs.

The main AMD optimization flags remain opt-in. Unsupported expressions and
unproven transformations retain their standard lowering. Additional approximate
GELU/softmax math remains disabled for this comparison and outside the initial
submission scope. The implementation uses hardware properties and expression
proofs; model names and benchmark case identities do not select kernels.

Validation passed **358 focused tests plus 8 wide-reduction checks**.
The final 142-case MI355X operation comparison measured
**1.1243x geometric-mean speedup versus standard Triton**;
the updated policy measured **-0.54% versus
the previous native policy**. Individual regressions are included below.
Two full training-step comparisons
measured approximately **0.65%
to 0.89% speedups**. These combined-policy results
do not isolate each individual change, and the model accuracy checks have
specific limits described below.

This is an AI-generated local technical description and review plan.

## Packaged benchmark evidence

The branch contains the session's benchmark evidence in
[the artifact catalog](benchmark_artifacts/README.md): directly browsable CSVs,
charts, summaries and reports, plus compressed raw runs, frozen sources,
benchmark scripts and validation logs. The catalog documents checksum
verification and extraction. Absolute runtime paths quoted below are preserved
historical provenance; the catalog maps them to the packaged files.

## Changes relative to standard Inductor

| Lowering path | Standard behavior | Updated opt-in behavior |
| --- | --- | --- |
| Flat pointwise and broadcast | Standard block/warp choices and generated scalar/vector accesses | Proven native unrolling and scalar reuse, geometry-derived candidates plus retained standard candidates |
| Transpose copy | Flat indexing and standard pointwise launch candidates | Checked tiled traversal plus both flat directions; existing wrapper grid retained |
| Masked reduction indexing | Generated remainder may repeat a proven range restriction | Remove remainder only under the complete mask, bounds and nonoverflow proof |
| Inner row reductions | Standard persistence and reduction launch heuristics | Device/value-pressure estimate, confident persistent choices and measured persistent/looped competition when uncertain |
| Partial and dynamic rows | Restricted specialized tile eligibility | Runtime row masks admitted only after access/dependency/axis proofs |
| Codegen pass integration | Separate source transformations | One explicit parsed source shared across the guarded passes, emitted once |

The three AMD optimization tier flags default to false. The new persistence
comparison sub-option defaults to true but takes effect only inside the enabled
native AMD path. Diagnostics are optional. No model name or fixed benchmark
shape is a dispatch key.

### Pointwise, casts and scalar broadcast

Static add/cast and supported straight-line elementwise fusions gain tunable
blocks, warp counts and unrolling. The native path also accepts runtime element
counts for proven flat expressions. Every unrolled subtile computes its own
bounds mask. Scalar broadcast values can be loaded before the vector loop while
preserving the matched expression and casts.

```python
lane = tl.arange(0, XBLOCK // AMD_UNROLL)
for u in tl.static_range(AMD_UNROLL):
    i = tl.program_id(0) * XBLOCK + u * (XBLOCK // AMD_UNROLL) + lane
    a = tl.load(in_ptr0 + i, i < xnumel, other=0).to(tl.float32)
    b = tl.load(in_ptr1 + i, i < xnumel, other=0).to(tl.float32)
    tl.store(out_ptr0 + i, a + b, i < xnumel)
```

Recognizers check pointer types, index construction, stores, masks and expression
structure. Indirect addressing, unsupported functions and unrecognized forms use
the standard path. The checks do not authorize arbitrary reassociation.

### Tiled transpose with retained flat implementations

The new native transpose path recognizes exact static contiguous copy
permutations, including canonical 2D kernels and supported batched permutations
with an inner contiguous dimension. The generated two-dimensional tile reads
`[rows, columns * inner]` and writes the proven permuted output address.

The existing wrapper grid is preserved. A grid-stride tile loop covers every
tile, including partial edges. Positive tile configurations use `XBLOCK=M*N`
with a one-dimensional grid, or `XBLOCK=M, YBLOCK=N` with a two-dimensional grid.
The latter preserves parallelism rather than reducing the number of CTAs by an
extra tile-width factor.

The candidate set retains the original flat output-contiguous copy and a flat
input-contiguous alternative. Zero tile dimensions select these flat paths;
positive tile dimensions select the tiled path. Cast chains, masks and numerical
edge cases are preserved. Unsupported layouts, extra stores, altered address
expressions and unsafe padded-index ranges leave the original source unchanged.

### Redundant integer indexing

A flattened reduction address can include `% M` even when its load predicate
already establishes that the address is in `[0, M)`. The simplification removes
that remainder after proving positive extents, standard row/reduction indexing,
loop bounds, strict upper-bound masking and signed-32-bit nonoverflow. The load
mask stays in place. Missing proof leaves the expression unchanged.

This pass remains at codegen where the complete load predicate is available.
Moving the same proof into general symbolic indexing needs an equivalent way to
carry the predicate facts; the current change does not discard those conditions.

### Reduction resources and persistence

The policy uses reduction width, row count, input/reduction counts, dtype and
reported register capacity to estimate live-value pressure and desired
residency. Mixed floating-point/integer reductions estimate tile values from
floating/complex input widths and reduction output widths. Per-row int64 target
labels no longer double the FP32 reduction tile estimate. Integer-only
reductions and int64 reduction outputs still retain their wider estimate.
This remains a heuristic: mixed integer inputs are not proven row-invariant by
the width estimator itself. This is a conservative search heuristic rather than an exact
register-allocation or occupancy proof.

Clearly favorable persistent reductions retain that implementation directly.
For eligible read-only reductions where the resource estimate is uncertain,
Inductor's existing `MultiKernel` machinery compares persistent and looped
implementations. Comparison admission allows slack in the live-value estimate
because it does not model value reuse precisely. Compiler limits and candidate
validation remain active.

The new comparison is guarded by the AMD optimization flags, native mode,
`amd_persistent_reduction_autotune`, supported static widths, and the existing
mutation/cooperative/deterministic exclusions. Explicit global multi-kernel
configuration retains its normal behavior. Required kernel feature overrides
are honored.

### Native candidates and resource profiling

Native seeds are generated from geometry, dtype width and device properties.
They are merged with legal standard Inductor configurations and deduplicated.
Specialized signatures receive default unroll/traversal parameters so a
standard configuration remains callable. Stock transpose configurations retain
the original flat copy; input-contiguous flat seeds remain available too.

A device-derived per-thread pressure estimate filters newly introduced row
seeds and generic reduction extras before compilation. It accounts for the tile,
warp/wave size, value width, loads and reductions. All legal standard candidates
bypass this heuristic filter. The estimate can be conservative; it is not used
to declare a standard configuration invalid.

Profiling records candidate and selected configuration, registers, spills,
shared memory, candidate counts and measured neighbor compilation time. Reports
are emitted as the structured-trace `amd_kernel_autotune` JSON artifact and can
also be appended to the configured local diagnostics JSONL path. The
initial precompile counter can omit asynchronous compilation. Accordingly,
`reported_precompile_time_ns`, `neighbor_compile_time_ns` and
`accounted_compile_time_ns` are reported separately with that limitation.
Benchmark setup wall time includes compilation and initial execution/tuning;
it is not labeled as pure compiler time.

### Partial rows and dynamic shapes

Additional partial-row and dynamic-row tiles require a positive proof. Row-dependent memory accesses
must contain a strict `xindex < xnumel` predicate, including supported mask
aliases. Row-independent loads, such as column weights, can remain unmasked
with respect to rows after a recursive address-dependency check. All stores
must retain row masking. Reductions stay on the reduction axis.

The proof rejects reassigned bindings, unsupported control flow, aliased
function calls, shape-changing methods/properties and row slices that could
mix padded rows into valid results. Unknown dependencies decline the
transformation. Dynamic row counts use runtime masks rather than row-count
specialization. Dynamic reduction widths still follow ordinary Inductor
specialization rules; arbitrary width reuse is not claimed.

### Shared codegen representation

Native pointwise, row and tiled-copy paths emit TL directly. The source passes
share one explicit `AMDCodegenSource` AST, eliminating repeated full-source
parsing and serialization between passes. Matching passes mark the context
changed; final emission happens once. Unmatched source remains byte-for-byte
unchanged. Differential tests compare shared-AST and sequential emission for
pointwise, row, 1D-grid transpose and 2D-grid transpose cases.

The implementation reuses standard Inductor pointwise/reduction configuration
factories and multi-kernel selection interfaces. It does not introduce a second
compiler or a model-specific dispatch table.

### Validation and timing contracts

Candidate trials preserve storage aliases and views, restore mutations, apply
reset-to-zero behavior and check outputs. Graph-based tuning temporarily disables
automatic cyclic Python collection during each capture and restores the prior
state after context exit, including exceptions. This prevents discarded launcher
cycles from finalizing unrelated Triton modules inside HIP capture. Current
launchers remain strongly referenced; the guard does not prevent explicit
`gc.collect()` or arbitrary refcount-triggered destruction. No garbage
collection is forced by the tuning code. Ordinary outputs are poisoned to
expose missing stores. Seed-launcher agreement checks configuration consistency;
independent eager-reference tests validate the source transformations.

Pointwise autotuning is enabled for all compared policies; the additional AMD
flags distinguish the updated policy from the standard path.

Benchmarking distinguishes captured GPU kernel sequences from inline graph
replay. The captured protocol times actual GPU work with external events;
inline replay is reported separately because host launch gaps can contribute.
Preparation stays outside timing. GEMM and convolution are forced to ATen for
consistent focused comparisons. Source manifests and loaded-module checks
identify the exact code used by each isolated worker.

## Final performance and test evidence

The final source snapshot attempted **142 cases**: 16 reference cases and 126
extended-shape cases with BF16/FP32-origin inputs, pointwise tails, transpose layouts, reduction
widths from 127 through 32768, odd rows and different row/width ratios.
The case dtype label describes input generation; some workloads promote their
inputs or computations to FP32. In particular, the softmax-backward workloads
feed FP32 probabilities and gradients for both origin labels. Actual input dtype
metadata should be used when interpreting kernel precision.

The three compared policies are standard Inductor Triton, the previous native
policy and this updated native policy. Each has cold/warm contexts and separate
captured-GPU/inline-replay measurements: **426 workers and 1704 result rows**.
The matched comparison contains **142/142 cases**.
All 142 cases passed reference validation for all three policies. Scan validation uses FP64 accumulation rounded to the expected output dtype; all other operation references retain their established eager definitions.

On one MI355X (`gfx950`), captured GPU sequence timings show
**1.1243x speedup over standard Inductor**
and **-0.54% versus the previous native policy**
by geometric mean. Compiler artifacts were preseeded from prior immutable runs
using content-addressed source/configuration keys; Inductor and winner-selection
caches were fresh, and selections, reference validation and timings were rerun.
Consequently these setup measurements are not cold-compilation comparisons.
The existing input-cache preparation and timing/sampling policies were unchanged.
The capture-lifetime guard changes when automatic Python collection may run,
not the recorded GPU work or sampling schedule.
Reference-case speedup over standard is
1.1305x; extended-shape speedup is
1.1235x. Every valid matched case and both cache contexts
receive equal weight. Families with more cases therefore have more weight in
the overall figure. The extended-shape split is named `heldout` in the
manifest; several cases were used to diagnose regressions during development,
so it should not be interpreted as a strictly unseen evaluation set.

| Workload family | Cases | Speedup vs standard | Speedup vs previous native |
| --- | ---: | ---: | ---: |
| broadcast | 2 | +13.06% | -2.88% |
| cast_add | 6 | +8.93% | +0.29% |
| cross_entropy_backward | 24 | +18.68% | -1.43% |
| cumsum | 24 | +14.30% | +0.48% |
| norm_backward | 24 | +6.73% | +0.44% |
| silu_gate | 4 | -3.36% | -2.19% |
| softmax | 24 | +20.75% | -0.23% |
| softmax_backward | 24 | +11.97% | -1.32% |
| transpose | 10 | -1.11% | -1.40% |

Speedup is `100 * (baseline_runtime / updated_runtime - 1)`. Negative values
favor the baseline. These figures measure the combined source, persistence and
tuning policy. They do not isolate the gain from every individual rewrite.
Individual regressions remain in the CSVs, including p10/p90 observations.
Using a descriptive 1% band, **95 cases improve by more than 1%**,
**26 regress by more than 1%**, and **21 are within 1%** of
standard. These buckets summarize observed timings, not statistical significance.
The three largest gains and three weakest cases, combining cold/warm contexts
with the same geometric-mean definition, are:

| Case | Speedup vs standard |
| --- | ---: |
| heldout_softmax_8x32768_bfloat16 | +93.72% |
| heldout_softmax_8x32768_float32 | +75.46% |
| heldout_softmax_127x16384_bfloat16 | +66.36% |
| heldout_transpose_127x128_float32 | -6.87% |
| heldout_transpose_2x5x37x127_float32 | -6.48% |
| heldout_cross_entropy_backward_127x128_bfloat16 | -6.42% |

Close results can change with tuning variability. These are repeat observations
on this GPU rather than confidence intervals or cross-machine guarantees.

All **358 focused tests passed** on the final source snapshot, plus **8 targeted
wide-reduction checks** covering softmax and cross-entropy backward at
`8/257 x 32768` in FP32/BF16 with eager and graph-replay comparisons. Coverage includes
CPU proof/candidate contracts, capture-state restoration and cyclic-finalizer
negative controls, high-precision scan-reference checks, compiled numerical
checks, partial tiles,
NaNs/infinities/signed-zero inputs, dynamic graph reuse, saved statistics and
forward/backward gradient checks. CPU tests exercise wave32 candidate legality;
GPU correctness and performance validation used wave64 MI355X. The scoped
changed-file lint passed; Pyrefly dependencies and the C++ compatibility-check
build directory were unavailable. Validation used the existing built PyTorch
with a source overlay and serial kernel compilation.

These operation workloads include compound expressions. The separate full-model
measurements below assess steady training-step throughput; neither experiment
establishes convergence or performance on other GPUs. No additional approximate-math option was enabled. Floating-point
reduction schedules can still differ numerically within the tested tolerances.

## Full-model training-step check

The same frozen candidate source was also tested on one physical GPU0 with
cached, architecture-initialized DistillGPT2 and MobileNetV3 models. Inputs,
initial state, dependency versions, compiler flags and loaded source hashes
were audited. The two policies are standard Triton and the updated native path.
Each process uses a single GPU, AMP BF16 with FP32 parameters and
`SGD(lr=1e-4, momentum=0, foreach=True)`. GEMM and convolution use ATen.
No model checkpoints are read or written. The models are initialized from
cached configurations; these runs do not use pretrained weights. Inputs are
synthetic, and the loss is `output.float().square().mean()` on the first
differentiable model output. This is a full architecture/optimizer workload
with a synthetic objective. The forward/loss and AOTAutograd backward are
compiled; `zero_grad` and the SGD step run through the ordinary optimizer API
and are included in the timing.

| Model | Batch | Standard step ms | Updated step ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| mobilenetv3_large_100 | 512 | 88.1773 | 87.6119 | +0.65% |
| DistillGPT2 | 32 | 5.4910 | 5.4425 | +0.89% |

These are synchronized wall timings of full forward/backward/SGD steps,
including CPU overhead, with 5 initial warmups, 10 additional steady warmups
and 7 repeats of 20 steps. Compilation and first-step tuning are excluded from
steady throughput. Each model/policy uses one process; repeats are within that
process, so these sub-percent gains are not cross-run confidence estimates.
Compiler artifacts were preseeded by content key, but
Inductor/selection caches were fresh and the models were recompiled and tuning was rerun.
This is a two-model throughput check rather than the complete model suite.
It is separate from the captured-GPU operation timings and should not be
combined into the same geometric mean.

All parameter gradients must be present. Finiteness and eager agreement are
sampled on the first 1024 elements of the first/middle/last parameter gradient
tensors, alongside loss checks. Accuracy validation compares each compiled
variant with its own eager reference; it is not a direct full-gradient-vector
comparison between variants.
For MobileNet, BatchNorm validation uses batch statistics in eval mode; this
sampled check has a relatively loose absolute tolerance and can hide large
relative errors on small gradients. The exporter reports absolute and relative
errors alongside stock results. Passing this check is not evidence of tight
per-parameter agreement or training convergence. Timed steps use train mode.
For example, sampled MobileNet stem-gradient relative RMS error versus each variant's eager reference was 0.384 for standard and 0.394 updated; maximum absolute errors were 0.000242 and 0.000299. These are separate eager comparisons, not direct stock-versus-updated gradient-vector errors.

Model timings, repeat samples, accuracy details, setup costs and provenance are
in `/dockerx/pytorch-flex/agent_space/simd_bench/native_triton_v2/models_v5/per_model.csv` and `/dockerx/pytorch-flex/agent_space/simd_bench/native_triton_v2/models_v5/summary.json`.

## Experiments and decisions

- Retaining only the stock flat transpose direction omitted a useful contiguous
  input traversal. Both flat directions now remain candidates alongside tiles.
- A losing `32 x 1024 / 1 warp` reduction seed caused a roughly 237-second
  compilation/first-execution observation. The resource bound removed that
  newly introduced seed; a later targeted observation completed in about 4.5
  seconds. This is observed evidence rather than an isolated causal timing
  experiment. The guard also covers generic fallback reduction extras.
- Treating a live-value estimate as a hard admission limit excluded useful 32K
  persistent reductions and caused large regressions. The revised policy
  admits comparison with estimation slack and keeps confident persistent
  decisions directly. Four targeted cases recovered to approximate parity
  with the previous native policy before the full matrix was rerun.
- Mixed-dtype cross-entropy backward exposed another overly restrictive cost
  estimate: int64 target labels were charged as though every FP32 reduction
  value occupied eight bytes. The value-width helper now distinguishes this
  case while preserving wide integer reduction outputs. The metadata feeds
  both persistence admission and new-seed pressure checks; its version changes
  invalidate the old tuning policy.
- A full-model preflight exposed a capture-lifetime failure: automatic cyclic
  collection finalized an older discarded launcher and unloaded its Triton
  module during graph capture, which HIP rejects. The scoped GC guard addresses
  that runtime failure without changing the emitted kernel or sampling policy.
  The failed model attempt and preceding immutable sources remain available.
- A stock FP32 scan differed from an FP32 eager prefix sum at 15 elements for
  one seed. An independent FP64 accumulation showed both schedules within the
  unchanged error contract; stock was actually closer to FP64. Final scan
  validation therefore uses FP64 accumulation of the exact input values,
  rounded to the expected output dtype, outside timing. The original eager
  mismatch, all three seeds and both reference error reports are retained.
  The failed seed's stock-versus-FP64 maximum absolute error was about
  `1.0024e-4`; FP32 eager-versus-FP64 reached about `8.4361e-4`.
  This is a schedule/reference difference, not evidence of an incorrect stock
  result or a reason to relax the tolerance.
- The existing cache flush size, cache-context weights and sampling policy were
  retained. Resource/candidate improvements do not by themselves eliminate
  tuning variability. Those sampling/cache-policy changes were outside the
  requested follow-ups.

## Branch scope and proposed PR split

Split by behavior and evidence rather than submitting one combined experiment.
Each performance PR needs its own immediate-parent comparison, tuning-only /
source-only / combined ablations, and regression results.

| PR | Proposed scope | Dependencies and review evidence |
| --- | --- | --- |
| 1. Reproducible AMD kernel benchmarking | Captured GPU timing, separate inline replay, high-precision scan gold, numerical execution probes, raw results and source manifests. | Independent. Keep a focused reusable runner and measurement helper. |
| 2. Resource-aware validated AMD tuning | Device limits, native seeds plus stock candidates, pressure checks, alias-preserving trials, capture-lifetime safety, bounded search and resource diagnostics. | Use PR 1 to establish tuning-only performance, legal candidate retention, low-memory behavior and setup costs. |
| 3. Native pointwise, indexing and tiled copy | Guarded native source paths, unrolling, scalar reuse, redundant-modulo proof and flat/tiled transpose alternatives. | Uses PR 2 where tuning is required. Cover failed proofs, casts, tails, address ranges and old/stock alternatives. Modulo removal can be a small independent prerequisite. |
| 4. Reduction and dynamic-row policy | Resource-confidence persistence decisions, uncertain persistent/looped comparison, saved statistics and proof-gated partial/dynamic rows. | Depends on PR 2; can develop alongside PR 3. Validate gradients, graph reuse, deterministic/cooperative/mutation exclusions and full-model effects. |
| 5. Companion FlyDSL correctness | Backend tail/indexing/layout, symbolic splitting, RNG/math and fusion coverage fixes with focused reproductions. | Independent backend track against an agreed base containing that backend. Shared scheduler fixes can be separate prerequisite bugfixes. |
| 6. Optional FlyDSL versus Triton selection | Per-kernel competition, mutation-safe timing, cache separation and supported/fallback/winner diagnostics. | Depends on the relevant correctness fixes in PR 5. Works with standard Triton independently of PRs 2-4. |

The main dependency structure is `1 -> 2 -> {3,4}`, with a separate `5 -> 6`
backend track. Approximate GELU/softmax math and broader scan eligibility need
separate numerical contracts and additional evidence. Fix the upstream base,
extract each diff and rerun tests/benchmarks before submission. Broader GPU and
full-model claims require their own validation. Use the appropriate repository
PR template and identify AI-generated descriptions with the user's own review
and commentary as required by `AI_POLICY.md`.

## Implemented follow-ups

### 1) Native candidates and retained standard configurations

Implemented independent geometry/dtype/device seeds, legal stock-candidate
unions, ABI defaults and deduplication. Standard candidates remain available
when the new seed pressure estimate declines a configuration. Flat transpose
candidates retain both traversal directions.

### 2) Resource profiling and resource-aware exploration

Implemented per-candidate/selected registers, spills, shared memory, candidate
counts and explicitly scoped compilation/tuning counters. Actual resource
measurements prioritize compatible neighbor exploration. Seed and generic-extra
pressure checks avoid compiling clearly excessive new tiles without removing
legal stock candidates. Resource CSVs retain actual multi-kernel winners and
alternatives separately.

### 3) Native two-dimensional tiled transpose

Implemented guarded canonical 2D and supported batched copies, partial-edge
masks, correct 1D/2D CTA geometry, grid-stride coverage and flat fallback
alternatives. Unsupported forms retain the original lowering. Eligibility is
static; arbitrary dynamic tiled permutations are not claimed.

### 4) Resource-aware persistence selection

Implemented a row/width/dtype/live-value resource estimate, direct confident
persistent choices and existing multi-kernel comparison for uncertain eligible
reductions. Corrected overly restrictive admission using targeted regressions.
Required feature overrides and mutation/cooperative/deterministic exclusions
remain in effect. The estimate remains a heuristic rather than an occupancy
proof.

### 5) Partial-row and dynamic-shape coverage

Implemented strict row-mask and dependency proofs, including safe row-independent
weight reads. Added negative tests for unsafe masks, reassignment, aliased
calls, transpose methods/properties and row slices. Added dynamic shape reuse
and eager gradient comparisons. Unproven cases use ordinary lowering; zero/one
sizes and changing reduction widths retain normal specialization behavior.

### 7) Integration with native Inductor codegen

Implemented direct TL emission, one shared parsed source representation,
byte-preserving unmatched behavior, standard candidate-factory reuse and
standard multi-kernel interfaces. The modulo optimization keeps its complete
mask proof at codegen; a broader symbolic-indexing version must carry the same
facts rather than removing conditions.

Item 6's cache-model/adaptive-sampling changes were not introduced.

## Reproduction and artifacts

The reusable branch runner compares standard and updated native policies and
accepts either the focused default cases or an explicit workload JSON file:

```bash
python benchmarks/inductor_backends/amd_native_triton_hillclimb.py \
  --cases benchmarks/inductor_backends/benchmark_artifacts/payload/reports/native_triton_v2/results_final_v5/cases.json \
  --output agent_space/native_triton_results --rounds 5 --repeats 5
```

Use a built checkout or the established local source overlay. Output must be a
fresh directory. The isolated three-policy comparison and its frozen sources
are in:

```text
/dockerx/pytorch-flex/agent_space/simd_bench/native_triton_v2/results_final_v5
```

- `summary.json`: matched geometric means, reference/extended-shape splits and definitions.
- `per_case.csv`, `per_op.csv`: timings and all performance deltas.
- `per_kernel.csv`, `per_candidate.csv`: resources and selected configurations.
- `compile_tune_costs.csv`: setup and explicitly scoped phase observations.
- `resource_evidence.json`: winner resources, regressions and targeted observations.
- `runtime_chart.svg`, `per_op_chart.svg`: captured GPU figures.
- `graph_replay_runtime_chart.svg`, `graph_replay_per_op_chart.svg`: inline replay figures.
- `manifest.json`, source fingerprints and `VALIDATION.json`: source/coverage audit.
- `sources/candidate`: immutable measured compiler source.

Final test outcomes are recorded in:

```text
/dockerx/pytorch-flex/agent_space/simd_bench/native_triton_v2/validation_final_v5/test_results.json
```

Earlier exploratory or paused runs are preserved separately and excluded from
the final headline. In particular:

- `/dockerx/pytorch-flex/agent_space/simd_bench/native_triton_v2/results_final_v3` retains the FP32 eager scan mismatch.
- `/dockerx/pytorch-flex/agent_space/simd_bench/native_triton_v2/results_final_v4` retains 49 matched operation cases before the
  full-model capture failure paused that run.
- `/dockerx/pytorch-flex/agent_space/simd_bench/native_triton_v2/models_v4` retains the failed MobileNet attempt; its runtime is
  null rather than a successful timing.
- `/dockerx/pytorch-flex/agent_space/simd_bench/native_triton_v2/capture_diagnostic` retains the finalizer-cycle mechanism proof
  and the failing-old/passing-fixed capture regression.
- `/dockerx/pytorch-flex/agent_space/simd_bench/native_triton_v4/stock_scan_precision/precision.json`
  retains all three seeds' eager/FP64 errors and exact failing indices.

These diagnostics are not merged into the final performance aggregate. Production code and generated benchmark artifacts remain
uncommitted for review.

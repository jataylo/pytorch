# Native AMD Triton lowering and tuning

This branch adds opt-in native Triton specializations and resource-aware tuning
to upstream Inductor at `dddd90223f002504572be4070edece265279d819`. The implementation changes six compiler/runtime
files and adds focused tests and a standalone benchmark. It is independent of the
backend-selection and explicit-layout experiments in the earlier development
branch. This is an AI-generated local technical description.

## Behavior and configuration

The standard lowering remains the default. Enable the new paths with:

```python
from torch._inductor import config

with config.patch({"triton.amd_native_kernel_optimizations": True}):
    compiled = torch.compile(fn)
    result = compiled(*args)
```

`triton.amd_persistent_reduction_autotune` controls persistent/looped comparison
inside this opt-in path. `triton.amd_autotune_diagnostics_path` optionally appends
JSONL tuning reports. Structured traces also carry the `amd_kernel_autotune`
artifact. There are no experimental tier switches or additional approximate-math
path in this implementation.

## Changes from upstream

| Path | Change | Required proof or bound |
| --- | --- | --- |
| Pointwise and scalar broadcast | Native TL unrolling, scalar reuse and device-derived candidates | Recognized expression, pointer, cast and bounds structure |
| Flat elementwise fusions | Tune unrolling/warps without changing the expression | Straight-line supported calls, independent subtiles and retained masks |
| Transpose copies | Add 2D tiles and both flat traversal directions | Exact static permutation, masks, address range and wrapper-grid coverage |
| Reduction indexing | Remove redundant remainder operations | Full load predicate, positive extents and signed-index nonoverflow |
| Inner reductions | Resource-confidence persistence or existing MultiKernel comparison | Supported widths, dtype/output pressure, readonly and deterministic/cooperative exclusions |
| Partial and dynamic rows | Admit additional row tiles | All row-dependent accesses masked, row-independent addresses proven, reduction axis preserved |
| Autotuning | Native seeds plus legal standard candidates and bounded neighbors | Device limits, numerical validation, alias/mutation restoration and resource evidence |

`codegen/amd_kernel_optimizations.py` shares one explicit parsed source across the
guarded passes and emits TL directly. Unmatched source remains byte-for-byte
unchanged. The masked-modulo proof remains at codegen, where its predicate facts
are available; this branch does not remove those conditions to force a broader
symbolic simplification.

`codegen/triton.py` supplies feature/device metadata and integrates with ordinary
configuration factories and MultiKernel selection. `runtime/hints.py` supplies
wave/thread/register limits. Native seed geometry and dtype width are independent
of explicit layout APIs. Legal standard candidates bypass the new seed-pressure
estimate and remain available.

The persistence estimate distinguishes floating reduction values from auxiliary
wide integer inputs, while preserving int64 reduction output pressure. Estimates
are heuristics, not exact register-allocation proofs. They guide new exploration;
compiled candidate validation and the ordinary fallback remain active.

`runtime/amd_autotune.py` checks alias-preserving trial state, restores mutations,
poisons ordinary outputs, validates candidates, measures configured cache contexts
and confirms finalists. It reports registers, spills, shared memory and scoped
compile/tune counters. The initial precompile counter can omit asynchronous work;
setup wall time is not presented as pure compiler time.

Automatic cyclic GC is disabled only during each graph capture and the caller's
prior state is restored on every exit. This prevents discarded launcher cycles
from unloading unrelated Triton modules during HIP capture. It does not add a
global keepalive cache or force garbage collection.

## Scope of the cleanup

Only the native lowering, persistence/candidate policy, validated tuning and
measurement support are included. Shared scheduler/backend code is unchanged.
The old conversion helper, explicit-layout candidate pools, approximation emitter,
backend selectors, duplicate benchmark entrypoints and large raw archive bundle
are absent. The benchmark has two policies: standard and updated native Triton.
GEMM and convolution are forced to ATen in that comparison.

Hardware decisions use reported wave size, block limits and register capacity.
CPU tests cover device-derived legality; GPU checks used wave64 MI355X. Performance
on other devices is not established. The existing fixed cache preparation and
sampling policy were retained; this cleanup does not claim an adaptive cache model.

## Fresh validation of this extraction

All **203 focused tests passed** against the upstream-based checkout, including
35 candidate contracts, 89 transpose cases, dynamic/codegen checks, capture-state
and cyclic-finalizer regressions, mutation/alias checks and full forward/backward
gradient comparisons in BF16 and FP32. Removed legacy-only tests were replaced or
trimmed to the remaining native contracts; numerical tolerances were retained.
All loaded Inductor modules were checked to come from this checkout.

A fresh **16-case / 2-policy / 128-row** operation run passed all original seeds
with zero failures. Its captured direct cold/warm geomean was **1.1236x**
versus standard in this short run. The three-round, three-repeat dataset is a
smoke measurement, not a substitute for a full performance evaluation.
Compilation artifacts were warm; compile costs are not cold-start comparisons.

[Validation and source hashes](results/amd_native_triton/validation.json),
[test outcomes](results/amd_native_triton/test_results.json), and
[fresh benchmark rows](results/amd_native_triton/clean_branch/raw.csv)
are included. The existing built binary was reused with the checkout's complete
Inductor Python package overlaid; only Python compiler/runtime code changed.
Applicable lint passed with Pyrefly's unavailable dependencies and the C++
compatibility check's missing build directory excluded.

## Historical performance reference

The compact [V5 reference results](results/amd_native_triton/reference_v5) retain
142 cases, per-operation/case/candidate CSVs and four SVG charts from the earlier
measured implementation at `b95d6d3c9a994ca326aa72f0e906e31fdc14ab85`.
That run measured +12.43% versus standard and -0.54% versus the preceding native
policy. These are explicitly historical figures, not a new 142-case benchmark
of this extraction. Its original source hashes and measurement definitions remain
in the provenance file.

The earlier two-model training CSV is also historical. Those runs used synthetic
inputs and an output-square loss, and their sampled-gradient accuracy criteria
were loose for small MobileNet gradients. They do not establish convergence or
strict relative-gradient agreement. No full-model performance claim is transferred
to this cleaned branch without a new run.

## Suggested PR split

1. Measurement and reference correctness: captured GPU timing versus inline
   replay, precise scan gold, frozen callables and source fingerprints.
2. Device-bounded validated tuning: standard-candidate union, native seeds,
   alias/mutation validation, capture lifetime, neighbor search and diagnostics.
3. Guarded native pointwise/indexing/transpose lowering, with negative matches
   and stock/flat alternatives retained.
4. Resource-aware reduction persistence and proof-gated partial/dynamic rows,
   with saved-statistics and gradient validation.

The dependency order is `1 -> 2 -> {3,4}`. Per-PR tuning-only/source-only/combined
ablations and broader hardware/model benchmarks remain necessary for upstream
performance claims.

## Reproduce

Run the focused tests through the ordinary PyTorch test runner in a built checkout.
The operation runner accepts the packaged 142-case list or its default 16 cases:

```bash
python benchmarks/inductor_backends/amd_native_triton_hillclimb.py \
  --cases benchmarks/inductor_backends/results/amd_native_triton/reference_v5/cases.json \
  --output /tmp/amd-native-triton-results --rounds 5 --repeats 5
```

Use a new output directory. The run exports both cold/warm contexts and separates
captured GPU sequences from host-inclusive inline replay. Scan references use
FP64 accumulation rounded to the expected output dtype, outside compilation and
timing, with the established tolerances. The timed workload stays unchanged.

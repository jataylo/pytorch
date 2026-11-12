# Extended AMD Inductor lowering experiments

Enable the additional gfx950 experiments with both default-disabled flags:

```python
from torch._inductor import config

with config.patch({
    "triton.amd_kernel_optimizations": True,
    "triton.amd_extended_kernel_optimizations": True,
}):
    compiled = torch.compile(model)
    result = compiled(*inputs)
```

The extension retains the existing math functions and precision policy.
`amd_approximate_math` remains an independent, default-disabled option; its
existing GELU-backward transformation is applied before generic layout changes.

## Lowering and selection changes

- Static flat elementwise fusions can use explicit Gluon layouts. The generated
  expressions, casts, load/store ordering and math functions are retained.
  Layout vector width, workgroup size and static unrolling are tuned together.
  Targets include dropout backward, SiLU gating and SiLU backward. Pointer
  signatures, linear addressing, supported operations and masks are checked;
  unsupported sources retain the existing path. Tail masks are kept even when
  the original tile had no tail.
- Native-like vector-width-one layouts seed the search alongside wider vectors
  and unrolled variants. A bounded local search explores compatible neighbors.
- Static inner reductions above 1,024 and up to 32,768 elements can use a
  persistent kernel on gfx950. This keeps normalization/log-softmax data in
  registers across reductions and their epilogues. Explicit row layouts also
  cover these wider reductions, including saved statistics. Cooperative,
  deterministic and batch-invariant cases retain their existing rules.
- New and existing transformed kernels can tune against both modeled producer
  residency and warmed input storage. Each timing round uses the geometric
  mean across the two contexts; candidate selection uses the median round
  score. Input restoration, cache flushes and warming occur outside the GPU
  event interval. Diagnostics retain both contexts' raw samples.

The first hill-climb iteration improved cold-input cases but regressed several
warm-input cases. Balanced cache tuning improved that tradeoff. Further seeds
included vector-width-one layouts. GELU-forward layouts did not show a useful
aggregate improvement after either iteration, so erf-based elementwise kernels
retain their existing path. GELU forward remains in the benchmark as a control.
This is a bounded experimental search, not proof of a globally optimal layout.

## Reproducing the comparisons

Use a built checkout with Triton/Gluon and FlyDSL available. Compilation time is
reported separately from GPU execution time. The accepted local protocol uses
one MI355X and one compiler thread, with ATen GEMM/convolution backends.

```sh
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python benchmarks/inductor_backends/amd_extended_kernel_hillclimb.py \
  --output /tmp/amd-extended --rounds 15 --repeats 5
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python benchmarks/inductor_backends/amd_extended_kernel_hillclimb.py \
  --case-manifest benchmarks/inductor_backends/amd_kernel_reference_cases.json \
  --output /tmp/amd-extended-reference --rounds 15 --repeats 5
python benchmarks/inductor_backends/plot_amd_extended_kernel_hillclimb.py \
  /tmp/amd-extended
```

The new suite contains 62 cases across seven families and BF16/FP32: four sizes
for each elementwise family and five row geometries for each reduction family.
The reference manifest separately rechecks 18 previously recorded model/kernel
geometries. Each case compares regular Triton, previous optimized Triton
(extension disabled), extended optimized Triton and FlyDSL. FlyDSL backend
competition is disabled; unsupported FlyDSL operations may still fall back.
All four variants have approximate math disabled in this comparison.

Frozen compile_fx callables prevent subsequent Dynamo resets from replacing a
measured variant. Accuracy checks use three seeds. Direct GPU-event execution
and CUDA graph replay are separate measurements, each with cold/warm inputs,
15 rotating timing rounds and five invocations per sample. A 256 MiB flush
precedes each sample; warm samples touch input storage after the flush. Mutated
inputs are restored before each invocation outside timing. Restoring a mutated
input can warm that storage even in the cold-input context.

`comparison.csv` adds paired runtime reductions against regular Triton, the
previous optimized implementation and FlyDSL. `comparison_summary.csv` keeps
per-family geomeans, ranges, and counts of wins/regressions exceeding 3%.
Per-case outputs retain timing samples, generated code, source hashes, candidate
configurations, cache-context samples and failures. Reports keep regressions.
Use a fresh output directory after changing compiler source or timing options.

These are operation-graph measurements, not full-model training throughput or
convergence validation. Small training tests exercise forward/backward accuracy.
Additional variance probes compare ordinary and large-offset FP32 inputs against
FP64; neither FP32 implementation has FP64 accuracy for ill-conditioned inputs.
Asynchronous source-overlay compiler workers and repository lint remain outside
local validation; the existing environment lacks spin.

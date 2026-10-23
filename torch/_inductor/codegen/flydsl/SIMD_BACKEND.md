# FlyDSL SIMD backend: pointwise and reduction kernels

The FlexAttention path lowers a few FX subgraphs into a hand-written kernel. This backend
goes the other way: it replaces Triton as the code generator for Inductor's ordinary
fused kernels -- every pointwise and reduction kernel the scheduler builds -- and emits a
FlyDSL module per kernel instead of a Triton one. It is **ROCm only, experimental and off
by default**.

```bash
TORCHINDUCTOR_FLYDSL_SIMD=1 python train.py          # FlyDSL where it can, Triton elsewhere
TORCHINDUCTOR_FLYDSL_SIMD_STRICT=1 ...               # raise instead of falling back
```

or `torch._inductor.config.flydsl.simd = True` / `flydsl.simd_strict = True`.

Every kernel it refuses goes to Triton unchanged, with the reason counted in
`counters["flydsl_simd"]` (`flydsl_simd_counters()` returns them). On the models below,
FlyDSL generates 117 of 125 kernels of a GPT training step, and all of the refusals are
transposes (see [2D tiles](#2d-tiles-and-transposes)).

## Files

| File | Role |
| --- | --- |
| [`flydsl_simd.py`](flydsl_simd.py) | `FlyDSLKernel` (a `SIMDKernel`), its ops overrides and index printer, and `FlyDSLSIMDScheduling` (a `TritonScheduling` that tries FlyDSL first). |
| [`flydsl_simd_runtime.py`](flydsl_simd_runtime.py) | `fdp`, the trace-time runtime the generated code calls: typed arithmetic, buffer loads/stores, vector packing, cross-thread reductions, chunked combines. The only file here that imports `flydsl`. |
| [`cuda_combined_scheduling.py`](../cuda_combined_scheduling.py) | Swaps `FlyDSLSIMDScheduling` in for the Triton scheduling when `flydsl.simd` is set on ROCm. |
| [`test_flydsl_simd.py`](../../../../test/inductor/test_flydsl_simd.py) | Strict-mode numerics (dtypes, casts, indirect, masked, every reduction type, softmax/layernorm/cross-entropy, a transformer train step) and the fallback cases. |
| [`benchmarks/inductor_backends/flydsl_simd/`](../../../../benchmarks/inductor_backends/flydsl_simd/) | `bench_kernels.py` (per-kernel, vs Triton) and `bench_models.py` (whole models: compile time, step time, CUDA-graph step time). |

## How a kernel is generated

The scheduler hands `FlyDSLSIMDScheduling.codegen_node_schedule` a fused node schedule.
It picks a flat tiling -- `{x}` for pointwise, `{x, r0_}` for a reduction -- and drives the
normal `SIMDKernel` codegen with `FlyDSLKernel`, whose ops overrides print every op as a
call into `fdp` (`fdp.add(tmp0, tmp1)`, `fdp.vload(...)`). Any `FlyDSLUnsupported` raised
on the way rolls the attempt back and the schedule is handed to Triton.

The emitted module is a `@flyc.kernel` with a fixed 256-thread block, plus a `@flyc.jit`
launcher and a `_main` that caches `flyc.compile`. FlyDSL traces the kernel in Python, so
the generated code is ordinary Python that runs once at compile time: `range_constexpr`
loops are unrolled during tracing, and `fdp` keeps trace-time dictionaries (vector-load
and vector-store caches) that turn per-element Inductor ops into vector instructions.

**Thread layout.** A thread owns `unroll` runs of `ept` contiguous elements (`ept` is the
widest vector that divides the row and the dtype allows, up to 16 bytes).

- *Pointwise:* runs are interleaved across the block, so each vector instruction of a
  wave covers one contiguous span.
- *Row reductions* (r contiguous): `tpr` consecutive threads share a row; the row is
  combined by butterfly shuffles, and through LDS if `tpr` spans waves.
- *Column reductions* (x contiguous): 64 lanes run along x and the waves split r.

Rows of at most 16 elements a thread are *persistent*: unrolled entirely, held in
registers, and re-read from the trace-time cache by later passes (a layernorm's normalize
pass does not reload). Longer rows run an `scf.for`.

**Chunked combines.** Inductor asks for one combine per element. Since a thread holds a
whole chunk at trace time, `fdp` combines it at once: sums as a balanced tree, online
softmax with one rescale per chunk, and -- for persistent rows -- softmax and variance in
two passes over registers (max then exp-sum; sum then squared deviations), as Triton does.

Every launch parameter is written into the kernel as a comment, e.g.
`# row reduction xnumel=4096 rnumel=768 ept=4 unroll=1 tpr=64 stride=1 xunroll=1 persistent=True`,
and `TORCHINDUCTOR_FLYDSL_SIMD_TUNE="row_tpr=..,row_unroll=..,col_stride=..,col_unroll=..,xunroll=..,ept=..,force_flat=1"`
overrides any of them for experiments.

## Results

AMD Instinct MI308X (gfx942, 80 CUs), ROCm, `HIP_VISIBLE_DEVICES=0`. GPU time is per call
from a CUDA graph of 50 calls; *call* is host wall time per eager call of the compiled
function.

### Kernels

Geomean FlyDSL/Triton GPU time over the 25 cases FlyDSL compiles: **0.85** (16 faster).

| case | Triton µs | FlyDSL µs | FlyDSL/Triton |
| --- | ---: | ---: | ---: |
| add_f32_16M | 52.0 | 53.3 | 1.03 |
| bcast_add_f32 | 90.4 | 82.8 | 0.92 |
| silu_mul_bf16 | 337.5 | 329.9 | 0.98 |
| gelu_f16_64M | 297.0 | 317.0 | 1.07 |
| transpose_f32 | 52.9 | 52.9 | Triton fallback |
| small_add_4K | 12.0 | 11.7 | 0.97 |
| embedding_bf16 | 46.9 | 43.5 | 0.93 |
| sum_rows_f32 | 62.6 | 58.5 | 0.93 |
| sum_cols_f32 | 96.4 | 85.3 | 0.88 |
| sum_short_rows | 35.1 | 42.8 | 1.22 |
| amax_rows_bf16 | 116.1 | 112.0 | 0.96 |
| argmax_vocab | 236.6 | 221.3 | 0.94 |
| var_mean_f32 | 158.3 | 45.0 | 0.28 |
| softmax_f32_4K | 325.9 | 168.7 | 0.52 |
| softmax_bf16_32K | 1340.8 | 833.0 | 0.62 |
| softmax_short_rows | 89.4 | 88.7 | 0.99 |
| layernorm_bf16 | 355.8 | 160.4 | 0.45 |
| rmsnorm_bf16 | 148.2 | 167.5 | 1.13 |
| cross_entropy_32K | 755.9 | 330.8 | 0.44 |
| add_layernorm_768 | 19.1 | 20.8 | 1.09 |
| layernorm_bwd_768 | 22.5 | 23.4 | 1.04 |
| masked_softmax_512 | 145.8 | 169.6 | 1.16 |
| conv_bias_nchw | 57.5 | 46.2 | 0.80 |
| bn_stats_nchw | 108.2 | 112.9 | 1.04 |
| bn_bwd_reduce_nchw | 77.8 | 69.6 | 0.89 |
| softmax_bwd_512 | 42.2 | 48.8 | 1.16 |

The big wins are long-row normalizations, where Triton's default (non-autotuned) loop
reduction configs leave bandwidth on the table: chunked online softmax, two-pass variance
and wide vector loads reach 1.6-3 TB/s against Triton's 0.4-0.85.

The losses are short rows (`sum_short_rows`, 32-wide), the GPT-shaped 512/768-wide
normalizations, and `rmsnorm`. `gelu_f16` loses 7% to a precise divide inside `tanh`.

### Whole models

`bench_models.py`, each model and backend in a fresh process with empty Inductor, Triton
and FlyDSL caches. *Cold* is the first step including compilation; *warm* is the same in a
second process with the caches filled; *graph* is the `mode="reduce-overhead"` step.

| model | backend | cold s | warm s | step ms | graph ms | kernels |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| GPT train (d=768, 4 layers, bf16, AdamW) | Triton | 23.3 | 6.1 | 16.65 | 16.69 | |
| | FlyDSL | 26.7 | 8.1 | 16.86 | 16.88 | 117 FlyDSL, 8 Triton (transpose) |
| GPT inference (bf16) | Triton | 8.2 | 4.1 | 10.08 | 10.09 | |
| | FlyDSL | 10.8 | 5.6 | 10.30 | 10.31 | 29 FlyDSL, 4 Triton (transpose) |
| ResNet-ish train (f32, channels_last) | Triton | 13.8 | 4.8 | 26.56 | 26.59 | |
| | FlyDSL | 15.7 | 6.1 | 26.84 | 26.86 | 99 FlyDSL, 19 Triton (transpose) |

Step time is within 1-2% of Triton; graph time tracks step time, so the remaining gap is
kernel time, not launch overhead. Losses in the model are concentrated in the 512/768-wide
softmax and layernorm shapes of the table above; the long-row wins do not show up because
these models have no long rows outside matmuls.

These numbers came from profiling the models kernel by kernel (`rocprofv3
--kernel-trace`, timed steps only) and fixing what lost: a 768-wide layernorm went from
2.0x Triton to 1.09x (one wave per row, two-pass variance, vectors narrowed to tile the row
exactly), a BatchNorm backward reduction from 1.63x to 0.89x (scalar gathers with deeper
unroll), and an in-place bias add from 1.48x to 1.07x (loads of every run issued before
the first store). The model gaps went from 2.8% / 5.1% / 5.2% to 1.3% / 2.2% / 1.1%.

## Where it is complicated

These are the parts that took the most work, or that still limit it, roughly in order of
how much they matter.

### 2D tiles and transposes

A flat 1D layout cannot coalesce both the read and the write of a transpose: one side
strides by the row length. Run flat, a 4096^2 transpose was **7.9x** slower than Triton,
which uses a 2D `(YBLOCK, XBLOCK)` tile and lets its compiler stage the transpose through
LDS. FlyDSL has no such tile abstraction to target from here, so whenever Inductor's own
tiling analysis prefers a y/z tile, the kernel goes to Triton (`transposed access needs a
2D tile`). This is every refusal in the model runs. Supporting it means an explicit LDS
tile with two thread mappings -- the single biggest missing piece.

### Static shapes and 32-bit offsets only

AMD buffer descriptors carry an exact byte extent and take 32-bit offsets. The extent
comes from the buffer layout at codegen time, so dynamic shapes are refused, as are
buffers of 2 GiB or more and kernels of 2^31 elements or more. Lifting either needs the
extent passed at launch and 64-bit address arithmetic outside the descriptor.

### Choosing a layout per shape

Triton autotunes reduction configs; this backend picks one layout per kernel from a
handful of rules, each found by profiling a case that lost:

- column mode (64 lanes along x) when more loads are x-contiguous than r-contiguous;
- one wave per row whenever a wave can still hold the row in registers (no LDS round
  trip), vectors narrowed when that tiles the row exactly (768 = 64 x 3 x 4);
- two runs a thread for 32-64-run rows; two rows per group for bare reductions;
- scalar loads, one per thread, unrolled 8 deep when no load of the row vectorizes. The
  decision uses only the load/store indices, recovered per node from
  `LoopBody.memory_usage`, because `finalize_indexing` also sees `index_expr` values such
  as a split reduction's bounds check, which looked contiguous and kept the wrong layout.

Each rule has a counterexample one step away, so they are written next to the
measurements that justify them. The remaining losses (32-wide sums, 512-wide softmax,
rmsnorm) are what a small autotuning sweep over `tpr`/`unroll`/`xunroll` would target.

### Memory-ordering the compiler cannot prove

Buffer loads and stores go through descriptors the backend cannot prove disjoint, so a
store blocks every later load from being hoisted above it. With a run-at-a-time body, a
pointwise kernel with two runs a thread kept one load in flight: 141 µs against Triton's
95 on an in-place bias add. Pointwise kernels now replay their loads for every run ahead
of the body (into the trace-time vector cache; unused loads are dead-code-eliminated).

The persistent-row cache has the converse problem: a value cached before an in-place
store must not be reused after it. Stores record the buffer, and the cache is cleared for
it after the loop -- not mid-chunk, which once misaligned vector reloads and gave a
convnet a wrong BatchNorm forward.

### Narrow types and vector legalization

`<8 x i8>` / `<4 x i16>` buffer stores abort in LLVM's AMDGPU backend, so sub-dword vectors
are bitcast to i32/i16 words around every load and store. `torch.bool` is stored as a byte
but computed as i1, so storage and compute kinds are tracked separately. bf16/f16 results
are rounded through their storage type exactly where Triton rounds, so results match.

### Atomics

A masked lane's offset is pushed out of bounds (0x7FFFFFFF), which hardware drops for
loads and stores -- but buffer atomics fault on it (`APERTURE_VIOLATION`). Masked atomics
are wrapped in an explicit `scf.if`. Only f32 `atomic_add` is supported; bf16/f16 atomics
go to Triton.

### Writing control flow from helpers

FlyDSL rewrites `if` and `for` into MLIR control flow only inside the kernel function's
AST. Helpers in `fdp` are not rewritten, so a dynamic branch there (the masked atomic)
is a hand-built `scf.IfOp`, and anything else that would branch on a runtime value is
expressed as selects.

### Compile time and caching

FlyDSL's disk cache keys on the kernel function's source and helpers in the same
directory, not on runtime modules like `fdp`; a runtime change would silently reuse stale
binaries. Each kernel's source is stamped with a fingerprint of the runtime modules.

Cold compile is 2-3.5 s slower per model and warm compile 1.3-2 s slower. Most of the warm gap is
FlyDSL re-JITting each kernel's host launch stub on every process start (about 30 ms per
kernel, measured), which a cached host stub would remove. A call also costs about 1-3 µs
more host time than Triton's launcher, invisible under CUDA graphs.

### Not supported

Scan, sort, `bucketize`, random number generation (`load_seed`), bf16/f16 atomics,
dynamic shapes, cpp-wrapper and AOT Inductor. All of them fall back to Triton per kernel,
or raise under `flydsl.simd_strict`.

## Reproducing

```bash
export HIP_VISIBLE_DEVICES=0
# per-kernel table (CUDA-graph GPU time, GB/s, host call time); --only <cases>, --json out.json
python benchmarks/inductor_backends/flydsl_simd/bench_kernels.py
# whole models: cold/warm compile, step, CUDA-graph step; --only <models>, --json out.json
python benchmarks/inductor_backends/flydsl_simd/bench_models.py
# tests
python test/inductor/test_flydsl_simd.py
```

Per-kernel profiles of a model: run one training step under
`rocprofv3 --kernel-trace --output-format csv` and compare kernel durations by fused-op
name; `torch.profiler` does not work on this ROCm build.

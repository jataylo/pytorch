# FlyDSL Backend

FlyDSL is an AMD Python DSL that compiles to MLIR and ROCm. This directory holds the
Inductor codegen for it: a generic template backend (the FlyDSL analog of
[`codegen/cutedsl/`](../cutedsl/README.md)) plus a FlexAttention path that lowers user
`score_mod` / `mask_mod` functions into hand-tuned flash-attention kernels.

The FlexAttention path is **ROCm only and experimental**, and covers forward and backward.
It is never chosen by `AUTO`; it has to be asked for by name, and when it is asked for and
cannot be served it raises with the reason rather than silently falling back. Forward and
backward are chosen together and cannot be mixed: this forward writes LSE in natural log
where Triton's backward reads log2, so a fall-through between them would compute wrong
gradients rather than slow ones.

## Quick Start

```python
import torch
from torch.nn.attention.flex_attention import flex_attention

def alibi(score, b, h, q_idx, kv_idx):
    return score + 0.125 * (h + 1) * (kv_idx - q_idx)

compiled = torch.compile(flex_attention, fullgraph=True)
out = compiled(q, k, v, score_mod=alibi, kernel_options={"BACKEND": "FLYDSL"})
```

`q`/`k`/`v` must be bf16 or f16 with `head_dim` a multiple of 32 from 64 to 256 on a
supported architecture. Gradients work too, for the same mods, with one exception: a mod
may *read* a captured tensor but that tensor cannot itself require grad. See
[Supported configurations](#supported-configurations).

## Architecture

Two compilers run, one after the other:

1. **Inductor** lowers the traced `score_mod` / `mask_mod` FX subgraph to FlyDSL *source
   text* and renders it into a self-contained Python module.
2. **FlyDSL's JIT** traces that module and compiles a GPU binary.

The attention kernel itself is not generated. It is a hand-written FlyDSL kernel with two
hooks in it, and all this backend does is fill those hooks and call the launcher.

### Generic template backend

Shared with any FlyDSL template, not FlexAttention-specific:

- **[FlyDSLTemplate](flydsl_template.py#L21)** — template definition and registration;
  produces [`FlyDSLTemplateCaller`](flydsl_template.py#L150) choices for autotuning.
- **[FlyDSLTemplateKernel](flydsl_kernel.py#L43)** — argument management and the
  `def_kernel` / `gen_defines` / `get_output` render hooks.
- **[FlyDSLKernelWrapper](flydsl_kernel.py#L31)** — gives a compiled kernel a `.run()`.
- **[FlyDSLScheduling](flydsl_scheduling.py#L41)** — scheduler integration, dispatched from
  [`cuda_combined_scheduling.py`](../cuda_combined_scheduling.py#L69); compiles through
  [`async_compile.flydsl()`](../../async_compile.py#L755).
- **[FlyDSLTemplateBuffer](../../ir.py#L6519)** — the IR node, beside
  [`CuteDSLTemplateBuffer`](../../ir.py#L6485).
- **[flydsl_utils.py](flydsl_utils.py)** — availability probing. Checks the `_mlir`
  extension, the JIT runtime `.so` and that `.so`'s own shared-library dependencies,
  without importing `flydsl` as a side effect of importing torch.

### FlexAttention additions

- **[FlyDSLFlexTemplateKernel](flydsl_flex_kernel.py#L75)** — adds subgraph bodies, a CSE
  scope, and `modification()`, which renders one lowered subgraph into the template. Also
  owns the aux-slot bookkeeping and [`mod_key()`](flydsl_flex_kernel.py#L226).
- **[ModificationWrapperFlyDSL](flydsl_flex_kernel.py#L390)** — the ops handler used while
  a subgraph is lowered. Resolves subgraph placeholders (`m`, `n`) to the template's own
  variables (`q_idx`, `kv_idx`) and turns captured-tensor reads into aux-slot reads.
- **[FlyDSLFlexTemplate](flydsl_flex_kernel.py#L546)** — binds the two together and routes
  `subgraphs` to the kernel constructor.
- **[FlyDSLOpOverrides](flydsl_op_overrides.py#L166)** — the op table. Emits strings only,
  and must never import `flydsl`, because it runs during lowering on any machine.
  [`UNSUPPORTED_OPS`](flydsl_op_overrides.py#L37) is the explicit reject list.
- **[flydsl_mod_runtime.py](flydsl_mod_runtime.py)** — the runtime shim the generated code
  calls (as `fdu`). The only file here that may import `flydsl`. Handles type promotion and
  expands the ops FlyDSL cannot lower on AMDGPU: `exp`, `tanh` and `sigmoid` over `exp2`,
  and the transcendental family (`erf`, `erfc`, `atan`, `atan2`, `asin`, `acos`, and the
  four hyperbolics) over exp2/log2/sqrt. Only `erfinv`, `lgamma` and `digamma` are left on
  the reject list. Expansion accuracy is measured against eager in the test suite rather
  than assumed, since it is a property of this file.
- **[flydsl_flash_attention.py](../../kernel/flex/flydsl_flash_attention.py)** — the
  eligibility gate and kernel factory, dispatched from
  [`flex_attention.py`](../../kernel/flex/flex_attention.py#L340).
- **[flydsl_flash_attention.py.jinja](../../kernel/flex/templates/flydsl_flash_attention.py.jinja)**
  — the template that becomes the generated module.

### Vendored kernels

Under [`kernel/vendored_templates/flydsl/`](../../kernel/vendored_templates/flydsl), which
mirrors the CuteDSL precedent: the branch is self-contained and needs no sibling checkout,
only the optional FlyDSL runtime. These are derived copies of FlyDSL's flash kernels and are
excluded from `lintrunner` for the same reason the CuteDSL ones are — they are re-synced by
diffing against upstream, not restyled.

| File | Role |
| --- | --- |
| `flex_kernels/flex_flash_generic.py` | The forward kernel. gfx942-validated. Carries the score/mask hooks, LSE output, aux reads, mod vectorization, and a KV block-skip loop. |
| `flex_kernels/flex_flash_950.py` | Mechanical gfx950 port of the same hooks. Never executed on hardware; gated behind `config.flydsl.allow_unvalidated_arch`. |
| `flex_kernels/flex_mods.py` | Reference mods, plus the `@elementwise` / `@elementwise_mask` decorators that lift a scalar mod body to the vectorized ABI. Doubles as the contract our codegen has to emit against. |
| `flex_kernels/flex_interface.py` | Host side. `flex_flash_attn_bhsd()` adapts Inductor's `[B, H, S, D]` to the kernel's `[B, S, H, D]`, allocates, validates aux, and launches. |
| `kernels/kernels_common.py` | Low-level FlyDSL/MLIR helpers shared by both kernels. |

### Tests

- [`test/inductor/test_flydsl_flex_attention.py`](../../../../test/inductor/test_flydsl_flex_attention.py)
  — end-to-end numerics, captures, LSE, vectorization equivalence, and the rejection cases.
- [`test/inductor/test_flydsl_template.py`](../../../../test/inductor/test_flydsl_template.py)
  — the generic template backend.

Both skip unless FlyDSL is installed on a supported architecture.

### Benchmarks

[`benchmarks/transformer/flydsl/`](../../../../benchmarks/transformer/flydsl/) reproduces
every number quoted below: `mod_matrix.py` for the geomean against Triton, and five
narrower scripts for the questions that came up while chasing it — where a call's fixed
cost goes, which layout to index, what the kernel does per head_dim, and where it spills.

## How a score_mod becomes a kernel

1. `flex_attention` lowering reaches
   [`_use_flydsl_flash_attention()`](../../kernel/flex/flydsl_flash_attention.py#L182),
   which returns False for any other backend and raises if `FLYDSL` was asked for but
   cannot be served.
2. [`create_flex_flydsl_attention_kernel()`](../../kernel/flex/flydsl_flash_attention.py#L216)
   rejects unsupported captures, allocates the output and the LSE buffer, and appends one
   template choice per `mod_vec_size`.
3. The template renders. `{{ modification(...) }}` walks the FX subgraph through
   `ModificationWrapperFlyDSL`, emitting `fdu.*` calls into a mod body. Captured tensors are
   discovered *here*, registered as kernel arguments, and assigned aux slots.
4. Because captures are only known after the mods render, the template emits
   `NUM_AUX_TENSORS`, `AUX_SPECS` and `MOD_KEY` *after* the mod definitions and builds the
   launcher at module level.
5. Inductor compiles and caches the module; FlyDSL's JIT compiles and caches the binary; the
   autotuner picks a `mod_vec_size`.

## Mapping to CuteDSL

| FlyDSL | CuteDSL | Notes |
| --- | --- | --- |
| `flydsl_template.py` | `cutedsl_template.py` | Same shape. |
| `flydsl_kernel.py` | `cutedsl_kernel.py` | Ours is smaller; the flex parts are split out. |
| `flydsl_flex_kernel.py` | `cutedsl_kernel.py` (flex parts) | CuteDSL keeps `modification()` and its wrapper in the one kernel file; we kept the generic backend clean of flex concerns. |
| `flydsl_op_overrides.py` | `cutedsl_op_overrides.py` | Much thinner — see below. |
| `flydsl_mod_runtime.py` | *(no analog)* | CuteDSL emits `cutlass.*` expressions inline. |
| `flydsl_scheduling.py` | `cutedsl_scheduling.py` | Same shape. |
| `kernel/flex/flydsl_flash_attention.py` | `kernel/flex/flex_flash_attention.py` | Same gate-then-factory shape; we reuse `HierarchicalIndex`, `patch_fixed_layout_indexer_for_cutedsl`, `is_trivial_*_graph`, `input_buffers_require_grads` and `has_unsupported_cpu_scalar_tensor_captures` from it rather than duplicating them. |
| `templates/flydsl_flash_attention.py.jinja` | `templates/flash_attention.py.jinja` | See divergences. |
| `FlyDSLTemplateBuffer` | `CuteDSLTemplateBuffer` | Near-identical; a shared base is a TODO in `ir.py`. |

## Where it diverges from CuteDSL, and why

**Mods and launcher are built at module level, not in the kernel body.** A FlyDSL flash
launcher traces its mods at build time, so it is built once and the kernel body only
launches; CuteDSL's `_flash_attn_fwd` takes its mods as arguments on every call. This does
not change how captures work, because a mod body names only its slot in the `aux` list it is
passed, never a kernel argument.

**Scalar mods, lifted — not fragment ops.** CuteDSL's overrides carry TensorSSA machinery
because CuTe ops are whole-fragment operations needing shape and lane bookkeeping. A FlyDSL
mod is written for one score at a time, and the vectorized ABI is handled by lifting the
whole mod (`@elementwise`), so each op is a plain call and all the type promotion lives in
one shim.

**Captured tensors are addressed by per-axis stride specs.** CuteDSL materializes index
fragments and loads through them. Here an aux slot is a reader `reader(b, h, q_idx, kv_idx)`
that resolves one element offset from four strides, where `0` means "broadcast over this
axis". The reader always gets the true four coordinates; the spec says which matter. That
makes the spec depend on *which coordinate the mod indexed each axis with* rather than on the
capture's shape — an `[H]` table is `(0, 1, 0, 0)` and an `[S]` table of the same length is
`(0, 0, 1, 0)` — so it is recorded during lowering and read back by the template. Getting
this wrong reads the wrong element silently, which is why anything the reader cannot express
is rejected outright.

A slot is therefore a *(tensor, spec)* pair and not a tensor: the same table read by `q_idx`
and by `kv_idx` is the same data addressed two ways and takes two slots, over one kernel
argument that the template converts once. An axis may also be indexed by a *value* —
`offsets[document_id[q_idx]]`, or `table[q_idx + 1]` — which needs no gather entry point,
because a reader position multiplies its argument by that axis's stride and does not care
where the argument came from. Such an axis claims a spare position and the value is passed
there, lowest-first so that `stride_kv` stays free: the vectorised reader treats
`stride_kv == 1` as a promise that `kv` is contiguous, which a gather does not keep. Between
them those two rules are what document masking needs, and it reads its document-id table
both ways *and* indexes offsets by the result.

**`mod_key` has no CuteDSL analog.** FlyDSL's JIT cache keys on traced source plus
recursively collected scalar closure values, so a constant a mod reads any other way — a
module global, a generated constexpr — is invisible to it, and two mods differing only in
such a constant would silently share the first one's binary. `mod_key()` hashes the rendered
bodies to break that.

**The autotune space is one tunable, not a config sweep.** Only `mod_vec_size` (1, 2 or 4)
is exposed. All three must produce identical results; the autotuner is free to pick any.

**LSE is natural log and a mutated input.** The kernel writes `ln(sum exp)` in place, so
`FLYDSL` is listed in `_NATURAL_LOG_LSE_BACKENDS` in
[`torch/nn/attention/flex_attention.py`](../../../nn/attention/flex_attention.py) alongside
`FLASH` and the wrapper does not rescale it.

## Supported configurations

| | |
| --- | --- |
| Direction | Forward and backward (`dq` and `dk`/`dv`) |
| Architecture | gfx942; gfx950 only with `config.flydsl.allow_unvalidated_arch` |
| dtype | bf16, f16 (all of q/k/v the same) |
| head_dim | Multiples of 32 from 64 to 256, and `qk_head_dim == v_head_dim` |
| Captures | At most 4 across both mods, rank ≤ 4, on device. A mod may *read* one; a gradient with respect to one is refused |
| seq_len | Any, ragged tails included, and Q may differ from KV (cross attention). K and V must match each other |
| GQA | Yes |
| BlockMask | Blocks are skipped in all three kernels when a `mask_mod` is present; the backward additionally needs ≥2 workgroups per CU, else it walks densely |

Cross attention takes two extents rather than one, and which kernel uses which is not
uniform: the forward and `dq` tile Q and walk KV, while `dkdv` tiles KV and walks Q, so each
one's grid, resident tile, `num_records` bound and padding mask take opposite members of the
pair. The `mask_mod` grid stops being square too, and the regrid rescales each axis against
its own tile count.

It was held as a single `seq_len` until the axes were split, and the failure mode was
silence: 0.97 relative error at `sq=512, sk=256` and 1.57 at the reverse, no error raised,
because the KV walk ran to the Q length. Equal lengths make every one of those sites
indistinguishable, so the tests that hold this up all use unequal ones, in both orders, with
a ragged pair (`512` against `300`) to catch a bound that took the wrong extent and still
looked plausible.

K and V must still agree with each other — one column index reads both, so there is no
second extent to give them. FlexAttention's own batched matmul rejects that shape before we
see it, so the guard is for direct callers of the kernel interface.

`head_dim` is an allowlist rather than a rule. The kernel accepts `head_dim % 32 == 0`, and
the allowlist stops at 256 because 288 needs 73728 B of LDS against gfx942's 65536 B limit
-- which surfaces as a backend compile error, not a kernel-side check. 64 is correct only
because the K swizzle sizes its row mask to `HEAD_DIM` (`HEAD_DIM // 16 - 1`, and 0 where
that is not a contiguous mask); with the mask hardcoded to 7 the XOR walked off the end of
a 64-wide row into the next one and returned ~44% relative error while building perfectly
happily. That is the failure mode the allowlist exists to prevent: a kernel returning
plausible garbage rather than falling back to Triton.

## Template hooks

Beyond the generic `{{def_kernel(...)}}`, `{{gen_defines()}}` and `{{get_output()}}`:

- `{{ modification(subgraph_number, output_name, **fixed_inputs) }}` — render one lowered
  subgraph, mapping its placeholders onto the template's variables.
- `{{ mod_key(*extra) }}` — cache-discriminating hash of the rendered bodies. Call *after*
  the mods; mandatory whenever there is a mod.
- `{{ get_tensor_buffers() }}` — capture argument names in aux-slot order. Call after the
  mods.
- `{{ aux_specs() }}` — `(stride_b, stride_h, stride_q, stride_kv)` per slot. Call after the
  mods.

## Config flags

| Flag | Env | Default |
| --- | --- | --- |
| `flydsl.allow_unvalidated_arch` | `TORCHINDUCTOR_FLYDSL_ALLOW_UNVALIDATED_ARCH` | off |
| `flydsl.autotune_mod_vec_size` | `TORCHINDUCTOR_FLYDSL_AUTOTUNE_MOD_VEC_SIZE` | on |
| `flydsl.autotune_qk_prefetch_depth` | `TORCHINDUCTOR_FLYDSL_AUTOTUNE_QK_PREFETCH_DEPTH` | off |
| `flydsl.autotune_backward_tile` | `TORCHINDUCTOR_FLYDSL_AUTOTUNE_BACKWARD_TILE` | on |
| `flydsl.autotune_forward_block_m` | `TORCHINDUCTOR_FLYDSL_AUTOTUNE_FORWARD_BLOCK_M` | on |
| `flydsl.autotune_kv_gpfetch` | `TORCHINDUCTOR_FLYDSL_AUTOTUNE_KV_GPFETCH` | on |

`kernel_options={"MOD_VEC_SIZE": n}` pins the mod width and skips autotuning;
`kernel_options={"DKDV_TILE": (kv, q)}` pins the backward's dk/dv tile;
`kernel_options={"BLOCK_M": n}` pins the forward's Q tile;
`kernel_options={"ENABLE_KV_GPFETCH": bool}` pins the forward's K staging;
`kernel_options={"LAYOUT": "bhsd"|"bshd"}` pins which layout the kernel indexes, which is
otherwise read off the strides of q/k/v (and `do`) so that nothing has to be copied.

The forward's Q tile is swept only at 32 heads and up, which is where the old
`BLOCK_M = 256 if num_heads >= 32` heuristic engaged and the only place the two candidates
differ. Measured, that heuristic was wrong at six of seven head_dims — 19% at 64 and 2.5x
at 256 — and right only at 160, which the sweep still picks. Upstream instead dispatches
the tile on total work (`batch * seq * heads`); measured here the taller tile loses at
every shape below 32 heads we tried, including large-batch and 16k-sequence ones, so the
head count is the better gate.

## Staging KV through registers

Both the forward and the `dq` backward can stage a KV tile global → registers → LDS rather
than global → LDS, and carry the next tile's registers across the loop so its global read
overlaps the current tile's GEMMs. This is upstream's main gfx942 lever: the alternative
is a DMA straight into LDS, which needs a 16-byte `buffer_load_lds` that gfx942 does not
have, and a second LDS buffer this kernel cannot afford anyway.

The two kernels landed differently, which is why one is a sweep and the other a default:

- **Forward** — swept, and swept *jointly with the Q tile*, because the two axes interact
  rather than compose. Measured alone at `BLOCK_M=128` it wins 6–15% at head_dim 96/192/224
  and loses 7–9% at 64 and 128; but at head_dim 128 the best of the four combinations is
  `BLOCK_M=256` **with** staging (81.3 TFLOP/s against 80.8 for the best 128-row config),
  which neither axis alone would have found. Reading either axis in isolation gives the
  wrong answer, which is why `itertools.product` covers the pair.

  The interaction has a mechanism worth knowing: a 256-row tile is a 512-thread workgroup,
  which *halves* the per-thread VGPR budget to 256. Staging's carried tile fits inside that
  at head_dim 128 (250 VGPRs, no spilling, hence the win) and does not at 192, where it
  spills 79 slots and throughput collapses from 71.8 to 39.5. The autotuner rejects that
  corner on measurement, so this costs a wasted build rather than a bad choice — but it is
  the reason the sweep cannot be replaced by a rule read off the head_dim.
- **`dq` backward** — on by default. Won at all of 64/96/128/192/256 (3.8% to 24.4%, best
  at 192). The reason to have doubted it was register pressure, and that measured clear:
  329 → 363 VGPRs of 512 at head_dim 192, no spilling.

- **`dkdv` backward** — on by default at head_dim 160/192/224 only, and off elsewhere. This
  one was expected not to work at all: it is the tightest of the three for registers (452
  VGPRs of 512 at head_dim 192, four LDS orientations against `dq`'s three, two sets of
  accumulators, and `2 * NUM_BATCHES_Q` carried vectors), and its walk sits inside the GQA
  group loop, so the lookahead crosses into the next Q head at each boundary and the
  pipeline has to re-prime per head rather than inherit the last one's in-flight tile.

  Built and measured, it splits by head_dim about as sharply as a knob can, and the split
  is reproducible: over 1x8x4096 and 2x16x2048, dense and causal, it is worth 1.01–1.10x at
  160/192/224 and costs 0.84–0.94x at 64/96/128/256, with all four measurements per head_dim
  agreeing in sign. Below 160 the carried tile costs more than the overlap buys; at 256 the
  kernel already holds the whole LDS at one workgroup per CU and has nothing left to spend.
  A band that clean is a default, not a tuning axis — which also keeps it off the autotune
  bill, and that bill is the reason to care (see below).

Numerics are bit-identical wherever any of this is enabled, so it only ever trades speed.

## What autotuning costs, and why the sweeps are kept narrow

Over 84 cold cells against autotuned Triton:

| | our total | Triton total | our choices | Triton choices | our s/choice | Triton s/choice |
|---|---|---|---|---|---|---|
| forward | 800s | 461s | 490 | 2052 | 1.63s | 0.225s |
| backward | 1240s | 1377s | 694 | 3900 | 1.79s | 0.353s |

The forward is 1.7x slower in wall time while offering 4x *fewer* choices; the backward only
looks even because Triton explores 5.6x more. Per unit of search we are 5–7x more expensive,
and that is the number that matters, because it is what caps how wide a sweep we can afford.
Those numbers predate the precompile pool below, which roughly halves both totals.

A single cold kernel costs ~1.54s, and it splits 0.40s emitting MLIR from Python (the builder
executes the kernel body, rewriting its AST) against 1.14s in the MLIR pipeline, LLVM and ISA.
So three quarters is native codegen over one very large fully-unrolled kernel, and the per-choice
gap to Triton's 0.22s is mostly just how much code we hand LLVM.

What made that cost *serial* was that FlyDSL compiles on first launch and nothing precompiled.
`flydsl.precompile_workers` (default 8) now compiles choices ahead of the benchmark loop in
subprocesses, worth **1.9x** end to end on six cold shapes (95.8s → 49.9s), of which the
benchmark phase itself falls 81.1s → 15.1s. Every shape after the first gains 2.1–2.4x; the
first absorbs worker startup. Two measurements decided the design:

- **Processes, not threads.** Eight compiles take 9.5s serially, 3.0s on four processes and
  1.9s on eight — but *more* than 9.5s across eight threads. The Python quarter holds the GIL
  and the native three quarters never release it, so an earlier thread-pool `precompile()`
  measured 1.35–1.5x slower and only relocated serial work. That is why the pool blocks a
  driver thread on a subprocess rather than compiling on the thread it was called from.
- **The parent pays almost nothing afterwards.** FlyDSL's on-disk cache is shared across
  processes: the same compile costs 1.15s cold and 0.07s in a *fresh* process. So a worker
  need only warm the cache and return nothing, leaving the parent its 0.40s of Python.

The pool is built on `TuningProcess`, not a `ProcessPoolExecutor`, for reasons that are easy
to rediscover the hard way. A spawn-based executor makes each worker re-import the parent's
`__main__`, which in a library means re-running the user's script in eight subprocesses —
the first version of this did exactly that, and the pool died on every job while looking
merely slow. `TuningProcess` launches through a dedicated entry script, and its children
already hold a CUDA context, which these jobs need (a launch is the only way to trigger a
compile) and Triton's compile workers deliberately do not have.

Set `flydsl.precompile_workers=0` to compile serially; eight CUDA-holding subprocesses is the
wrong trade on a small card autotuning inside a large model.

The discipline on sweep width is unchanged, since per-choice cost is only halved and not
closed: an axis earns a sweep only if measurement says neither setting can be pinned. The
forward's staging axis clears that bar (pinning it on costs 1.121x geomean on the 20 of 84
blocks where off won, worst 1.35x; pinning it off costs 1.098x on the other 64), and `dq`'s
and `dkdv`'s do not, which is why both are defaults instead.

Use `TORCH_LOGS="output_code"` to see the generated module.

## Current limitations / TODOs

- **No gradients for captured tensors.** A mod may *read* a captured tensor in the backward
  as freely as in the forward, but a capture that itself requires grad is refused: its
  gradient needs the joint graph's `zeros_and_scatter` outputs accumulated with atomics,
  which the FlyDSL `modification()` path does not emit. So a learned bias tensor is out;
  an ALiBi slope table is fine.
- **Backward block skipping is gated on occupancy.** Both backward kernels can walk block
  lists — `dq` a KV list per Q tile, `dk`/`dv` the transpose — but only when the grid has at
  least 2 workgroups per CU. Causal work per tile is triangular, so below that they are all
  resident, the kernel finishes when the heaviest one does, and skipping the light ones buys
  nothing while still paying the index loads (measured 0.89–1.18x under the threshold
  against 1.54–2.39x over it). Small-batch causal shapes therefore still walk densely, on
  purpose.
- **BlockMask regridding is memoised, not eliminated.** The kernels want one block list on
  their own tile grid where FlexAttention supplies two on its own, so `regrid_block_mask`
  unions and converts them. The conversion is a fixed ~215 µs of tiny elementwise launches
  regardless of sequence length, which was 48% of a 2x8x1024 kernel, so it is cached against
  the identity and version of the incoming tensors: a BlockMask is normally built once and
  reused, and a hit costs ~11 µs. One entry *per grid*, since a single mask is regridded to
  three — the forward's tiles and the two backward walks — and a single slot made them evict
  each other, which cost 5–34% of the causal wall without changing any answer. What remains
  is that a workload rebuilding its mask every step pays the full conversion every step, and
  that full blocks still get `mask_mod` applied per element because the kernel walks a
  single unioned list.
  Teaching it to walk both lists would fix the second, but a `mask_mod` on a dense walk
  measures at only 1.7–6.0% of a tile, so it buys a few percent of causal in exchange for
  splitting the KV loop in two — declined on that ratio, see the migration plan §1d.
  Skipping is declined (and the dense walk used) without a `mask_mod`, since the
  builder requires one and there is nothing to skip.
- **V is XOR-swizzled rather than padded**, which removed 512 B of LDS and the
  `waves-per-eu` warning. The transposed V tile was strided `BLOCK_N + 2` to keep rows off
  one LDS bank, costing `HEAD_DIM × 2` elements and putting the head_dim 128 tile at
  33280 B; it is now `BLOCK_N` with the bank spreading done by XOR, at exactly 32768 B.
  Worth 27% at 2x8x4096 and 13% at 2x8x1024 (head_dim 128). Note the *mechanism is not
  confirmed to be occupancy*: at 244 VGPRs a SIMD holds `floor(512/244) = 2` waves either
  way, so a 512-thread workgroup fills a CU before and after, and the likelier explanation
  is the store side — padding left the cooperative V store hitting only two distinct banks
  across its lanes, where the swizzle spreads it over eight. Worth confirming with a
  profiler before anyone leans on the reasoning.
- **QK prefetch depth is an opt-in autotune knob**, defaulting to 2. It decides how many K
  packs come out of LDS before the MFMA chain starts. Upstream's gfx942 path (mainline
  #850) runs 3–4 and pairs the deeper prefetch with a
  `sched_group_barrier(mask_dsrd, depth * 2, 0)`; that hint is the substance of it, and the
  kernel emits it for any depth above 2. Without it a deep prefetch is not merely unhelpful
  but dangerous — unhinted depth 5 measured **6x** slower at 2x8x4096 (14222 µs against
  2109 µs hinted), because the scheduler is free to spread the extra `ds_read`s through
  the MFMA chain. With it the pathology is gone and depth becomes an ordinary tradeoff:
  0.6–2.1% kernel-only, and 2–5% end to end at the larger head_dim 128 shapes once the
  autotuner picks per shape. `score_mod` gains most (5.4% at 2x32x4096), which fits — a mod
  puts VALU work between the MFMAs for the grouped reads to hide behind. Still opt-in via
  `TORCHINDUCTOR_FLYDSL_AUTOTUNE_QK_PREFETCH_DEPTH=1`, since four times the builds is a
  poor trade at 2–5% for a kernel compiled per shape, but worth setting for an expensive
  mod on a large shape.
- **What is left in the forward is dense, not sparse.** Against autotuned Triton on the
  `benchmarks/transformer/score_mod.py` matrix (B=4, H=16, S=4096, bf16, 18 cells over
  head_dim 64 and 128) the forward is **1.00x** geomean and the backward **1.17x**, winning
  8 of 18 and 15 of 18. Mods carrying a mask win — `causal` 1.36x/1.14x (D64/D128),
  `prefix_lm` 1.29x/1.10x, `sliding_window` 1.10x/0.91x — and the losses are the *dense*
  ones, `noop` 0.90x/0.88x, `rel` 0.90x/0.85x, `head_bias` 0.97x/0.87x, which is the known
  dense head_dim 64/128 deficit and still unexplained. `document_mask` is the odd one out at
  0.91x/0.77x forward against 1.54x/1.28x backward.

  ~~Sparse masks lose in the forward, and only in the forward~~ — they did, at 0.83x/0.71x
  for `sliding_window` and 0.86x/0.73x for `document_mask`, and the reading that a short KV
  walk was failing to amortise some per-tile setup cost was right about the shape of the
  problem and wrong about the cause. It was not in the kernel at all: the interface was
  copying q, k, v and the output into BHSD because the benchmark's tensors are BSHD memory
  behind BHSD sizes. That is a fixed 574 µs against a 461 µs kernel here, so the sparser the
  mask the worse it read. With the copies gone the forward geomean moved 0.93x → 1.00x and
  the backward 1.12x → 1.17x. See the layout entry above.

  Note when reproducing: that benchmark pins `BLOCK_N=32` for `document_mask`, and on gfx942
  the Triton kernel built from it disagrees with eager by 5.0e-01 at head_dim 64 while
  costing 3.7x more than it does when left to autotune. Comparisons against it are worthless
  in both directions -- take Triton's options out before reading any `document_mask` number.
- **No scalar captures.** CuteDSL supports them through `aux_scalars`; there is no
  `AUX_SCALAR_SYMBOLS` machinery here. Device 0-d tensors work; CPU ones are rejected.
- **head_dim 96, 160, 192 and 224 needed two separate fixes, and are now fast.** The
  cooperative KV load gives each row `HEAD_DIM // VEC_WIDTH` lanes, and 512 is not a
  multiple of 12, 20, 24 or 28, so the last lane group is partial and the tile's rows no
  longer divide into whole load batches; the loader bounds its LDS row, which covers both.
  Underneath that, K's XOR swizzle is a permutation only over a power-of-two extent --
  `HEAD_DIM // 16` is 6, 10, 12 and 14 here, so the row mask is not contiguous and
  `col ^ 80` leaves a 96-wide row from col 32 up. The swizzle is therefore off for these
  head_dims, and for a long time nothing took its place: they ran 25-55% below the
  power-of-two dims per FLOP and head_dim 192 was the worst column in the whole table.

  The fix is to pad K's LDS row by 8 elements on exactly those head_dims (`K_PAD`). This
  was previously written off as unavailable, on the grounds that the DMA path writes LDS
  contiguously and so fixes the row stride at `HEAD_DIM` — true, but DMA needs a 16-byte
  `buffer_load_lds` that **gfx94x does not have**, so on gfx942 that path is dead code and
  the objection never applied. Padding is worth ~1.4x at 96 / 160 / 224 and 2.6x at 192,
  which was worst because its 384 B row is an exact multiple of the 32-bank rotation and so
  put every row in one bank. It is gated on `not ENABLE_DMA`, so gfx950 will need the
  rotation-based swizzle instead; both backward kernels have no DMA path at all and take the
  padding unconditionally.

  256 works as of the deferred-rescale fix — it had been excluded for wanting 66560 B of
  LDS, the V swizzle removed exactly the 1024 B it overshot by, and that exposed an
  unrelated bug where only 4 of its 8 O accumulator chunks were being rescaled.
- **fp32 q/k/v stays rejected, deliberately.** Not for want of an instruction: FlyDSL
  exposes `mfma_f32_32x32x2f32` and `mfma_f32_32x32x4_xf32`. It is that fp32 is a second
  kernel rather than a gate change. Every LDS stride, swizzle granularity, vector width
  and DMA size here is written for a 2-byte element, and the tile doubles with the
  element: head_dim 128 goes from 32768 B to exactly the 65536 B limit, leaving one
  workgroup per CU, and head_dim 256 wants 131072 B and is simply impossible. The MFMA
  is also 8x the instructions for the same K extent — `32x32x2f32` against
  `32x32x16_bf16` — and the xf32 variant that would close some of that gap is not fp32,
  so it answers a different question than the one an fp32 caller asked. Triton's flex
  path handles fp32 correctly, so the fallback is a working path rather than a hole.
- ~~A BHSD↔BSHD transpose per call~~ **Fixed, twice.** The kernel takes a `layout` of
  `"bshd"` or `"bhsd"`. Both are affine in `(batch, head, token, col)`, so supporting the
  second was a change of coefficients at six addressing sites, not new index math.

  The first fix built `"bhsd"` always, on the grounds that `[B, H, S, D]` is what
  FlexAttention hands over, and measured 3–23% of the call. That reads the *sizes*:
  q/k/v out of a projection reshaped to `[B, S, H, D]` and transposed are a BHSD-sized
  view of BSHD memory, which is the normal case and the one the whole benchmark matrix
  runs, and for it the interface went on making all four copies. So the copies were only
  removed for callers who allocated BHSD directly — which is every test in the suite, and
  nothing else. `_kernel_layout` now reads the strides and builds for the layout that is
  actually there; `kernel_options={"LAYOUT": ...}` pins it.

  It was worth 574 µs per call against a 461 µs kernel at 4x16x4096x128 bf16 — four passes
  over q/k/v-sized memory, independent of what the mask does — and it is where the
  sparse-mask forward deficit went: a short KV walk has nothing to amortise it with. See
  [Sparse masks](#sparse-masks-lose-in-the-forward-and-only-in-the-forward) below.

  Indexing BSHD in place is not free in the kernel: a BHSD `(batch, head)` slice is
  contiguous and a KV tile is one run, where BSHD strides each token row by
  `num_heads * head_dim`, and that measured 8% of the forward at a long walk, 32% at a
  short one, and 45% of the backward at 2x8x1024x64. Copying still loses to it at every
  shape measured (forward 1.01–1.10x over S=4096…32768, backward 1.03–1.10x at three
  shapes of four, 0.97x at the fourth), because the copies are four to seven passes over
  memory whatever the kernel then does. But they grow with `S` where a dense kernel grows
  with `S**2`, so the default is a judgement that held everywhere it was tried rather than
  a proof, and `LAYOUT` exists for anyone whose own shape disagrees.

  `"bshd"` is still the default for standalone callers, and `test_layouts_agree` pins the
  two together. `flex_flash_950.py` has no `layout` parameter and is not wired into the
  flex path.
- **No packed mask intervals**, the SM100+ optimization CuteDSL has.
- **No fusion**, matching CuteDSL: no epilogue or prologue support.
- **gfx950 is unvalidated.** The port is mechanical and has never been run.

[`PARITY_MIGRATION_PLAN.md`](PARITY_MIGRATION_PLAN.md) works through most of the above: it
carries the measured gfx942 baseline against autotuned Triton and aten, and a phased plan
for the BlockMask plumbing, head_dim 64, backward, and gfx950/gfx1201. It also records why
rebasing onto the upstream `parity` FMHA family does not by itself close the gap.

## Extending

**Adding an op.** Most ops need one entry in `_UNARY` or `_BINARY` in
`flydsl_op_overrides.py` and a matching function in `flydsl_mod_runtime.py`. If it has no
AMDGPU lowering, either expand it in the shim (as `tanh` is) or add it to `UNSUPPORTED_OPS`
so the failure names the op instead of surfacing as an LLVM "no libcall available" error.

**Re-syncing a vendored kernel.** Files copied verbatim carry a provenance header naming
their upstream path, branch and commit, and `vendored_templates/flydsl/refresh_vendored.py`
reads it back — `--check` for local drift, `--log` for upstream commits since the pinned
commit, `--update` to pull content forward for review. Keep the flex additions, and do not
restyle to PyTorch conventions: the files are lint-excluded so that the diff stays readable.

The flash kernels themselves are a **hard fork**, not a tracked copy: upstream dissolved
the monolith they descend from into a shared helper framework, so mainline is a source of
patches to port by hand rather than something to merge. Read the upstream log for
`kernels/attention/` when picking work up — that is how the gfx942 performance commit
`829c7b4` (#850) was found, six weeks after it landed and after we had already concluded
that no portable gfx942 schedule work existed.

**Two FlyDSL versions.** `flydsl.expr.buffer_ops` exists in 0.2.4 and was removed in 0.3.1,
where those helpers moved kernel-side; import `flex_kernels.buffer_ops`, which picks
whichever is present. Importing the library path directly makes the whole flex suite *skip*
under 0.3.1 rather than fail, which is easy to mistake for green. To test against 0.3.1:

```
PYTHONPATH=/dockerx/xinya-flydsl/build-fly/python_packages \
    /dockerx/flydsl-build/venv/bin/python -m pytest test/inductor/test_flydsl_flex_attention.py
```

Note: requires the optional `flydsl` package and a ROCm build of PyTorch.

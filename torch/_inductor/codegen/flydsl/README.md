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
| `flex_kernels/flex_flash_950.py` | A hand-scheduled gfx950 forward (dual-wave, software-pipelined, D=128 only) carrying the same hooks. Nothing dispatches to it: the template builds `flex_flash_generic` on every architecture, and on gfx950 that already selects the CDNA4 instructions this file is written around. What is unexercised here is therefore the schedule, not the architecture. |
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

That width is 1, 2 or 4, and 4 is a ceiling of the MFMA fragment rather than a tuning
choice: writing an accumulator element as `r = 4g + j`, its `kv_idx` sits at offset
`8g + j`, so runs of 4 are contiguous and the groups are 8 apart. A wider call would need
elements that are not adjacent in the score domain, which is not what the ABI promises.
CuteDSL reaching 32 or 128 is therefore not a knob we have set lower — it is a different
accumulator layout, and one gated on SM100 and up.

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
`FLASH` and the wrapper does not rescale it. A loss may differentiate that LSE: the
backward takes an optional `DLSE` plane, which is where the natural-log convention has to
be believed rather than assumed, since a log2 `lse` would need the cotangent rescaled too.

## Supported configurations

| | |
| --- | --- |
| Direction | Forward and backward (`dq` and `dk`/`dv`) |
| Architecture | gfx942 and gfx950, neither behind a flag. Declared as capabilities rather than names, so anything else is refused by naming what it is missing — see [GPU architectures](#gpu-architectures) |
| dtype | bf16, f16 (all of q/k/v the same) |
| head_dim | Multiples of 32 from 64 to 256, and `qk_head_dim != v_head_dim` in either order in both directions — see [Asymmetric head dims](#asymmetric-head-dims) |
| Captures | At most 4 across both mods, rank ≤ 4. Shaped captures must be on device; a 0-d CPU one is copied there for you. A mod may *read* one; a gradient with respect to one is refused |
| seq_len | Any, ragged tails included, and Q may differ from KV (cross attention). K and V must match each other |
| GQA | Yes |
| Batch | `Bq == Bkv`. A broadcast `Bkv=1` key/value is refused in both directions: the forward addresses k and v at the Q batch index, so it would read off the end of the allocation, and `dk`/`dv` would need summing back down to `Bkv` |
| LSE | Returned in natural log and mutated in place, and differentiable: the backward takes an optional `DLSE` plane |
| BlockMask | Blocks are skipped in all three kernels when a `mask_mod` is present; the backward additionally needs ≥2 workgroups per CU, else it walks densely |
| Dynamic shapes | Yes on `seq_len`: `dynamic=True` over 512/1024/2048 autotunes once and launches that one kernel at all three, since the extents reach the kernel as arguments and only the block sizes are `guard_int`ed. CuteDSL's own indexer patch still carries a `TODO(dynamic shapes)` |

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

## Asymmetric head dims

`qk_head_dim != v_head_dim` is served in **both directions** — any pair from the admitted
set, in either order.

### The forward

The two extents are independent because the score tile sits between the two GEMMs. GEMM1
contracts over the QK extent to *produce* the score tile; GEMM2 contracts over the **KV
axis** to consume it, so the V extent only ever appears as GEMM2's free axis. Neither loop
bound is shared, which is why `K_STEPS_QK` and `D_CHUNKS` could simply be pointed at
different dims, and why `PV_K_STEPS` did not move at all. The deferred O-accumulator rescale
keys off `si + 1 < D_CHUNKS`, so it follows the V extent for free.

What actually had to change was everything that had been sized *once* and used for both
tensors, none of which is a loop bound:

| Was one thing | Became two | Because |
|---|---|---|
| `STRIDE_TOKEN_KV` | `STRIDE_TOKEN_K`, `STRIDE_TOKEN_V` | the same token sits at different byte offsets in K and in V |
| `global_idx_kv`, `global_byte_kv` | `_k` / `_v` pairs | same, per element and per byte |
| `global_idx_q` for the O store | `global_idx_o` | O is `head_dim_v` wide, Q is not |
| `_kv_nrec_bytes` | `_k_`, `_v_`, `_q_`, `_o_` | the `num_records` bound is per tensor |
| `THREADS_PER_ROW_LOAD` and its geometry | `_K` / `_V` tuples via `_load_geometry` | the lane-to-(row, col) map is a function of the row width, so K's rows and V's need different lane groupings, batch counts and partial-lane guards |

That last one is the substantive part: the cooperative load's whole geometry — lane
grouping, `ROWS_PER_BATCH_LOAD`, the batch count, the idle-lane predicate — is derived from
how wide a row is, so two widths need two of it. `NUM_BATCHES_K` also sizes the main loop's
register-staging iter_args, which carry K vectors only.

With the two dims equal, `_load_geometry` returns identical tuples and every split constant
collapses to its old value, so **a symmetric build is unchanged** — confirmed by LDS
(D128 still exactly 32768 B, D256 still exactly 65536 B) and by kernel-only timing across
the whole ladder, which moved by less than run-to-run noise.

The forward's tests cover nine pairs, and they are chosen for the *load geometry* rather than for
plausibility, because that is the part with two states. A head_dim whose lane count divides
the workgroup loads whole rows; 96, 160, 192 and 224 do not and idle their remainder. So the
cases that matter are the combinations — partial K against exact V, exact K against partial
V, and the one nothing else reaches, two *different* partial geometries in the same kernel
(`(96, 160)`, where 96 idles 8 lanes of 512 and 160 idles 12, so K's idle-lane predicate and
V's disagree about which lanes are live). Both orders are covered for the donor's own reason:
the two extents coincide in every symmetric build, so a constant left un-split is invisible
until something differs, and shows up on only one side of `qk > v` / `qk < v`.

**Speed: the asymmetric path inherits the symmetric profile rather than paying a penalty for
being asymmetric.** Forward only, dense, B=2 H=8 S=4096, both backends `max_autotune`:

| pair | fly/tri |
|---|---|
| qk 192 / v 128 | 1.19x |
| qk 96 / v 64 | 0.94x |
| qk 128 / v 256 | 0.89x |
| qk 256 / v 128 | 0.88x |
| qk 128 / v 64 | 0.77x |

That is the same shape as the symmetric forward at the same *QK* dim — a win at 192, a
deficit at 128 and 256 — which is what you would expect given the QK extent owns the swizzle
and padding story. There is no asymmetric-specific cliff; closing these is the same
head_dim work as closing the symmetric ones.

### The backward

The backward holds the same property, spread over four GEMMs rather than two. Writing the
GEMMs out by which extent each one touches is the whole argument:

| GEMM | reduces over | free axis | bound |
|---|---|---|---|
| `dq`: `sᵀ = k qᵀ` | QK | — | `K_STEPS` |
| `dq`: `dpᵀ = v doᵀ` | **V** | — | `V_STEPS` |
| `dq`: `dq = dsᵀᵀ k` | KV | QK | `D_CHUNKS` |
| `dkdv`: `s = q kᵀ` | QK | — | `K_STEPS` |
| `dkdv`: `dp = do vᵀ` | **V** | — | `V_STEPS` |
| `dkdv`: `dv = doᵀ p` | M | **V** | `D_CHUNKS_V` |
| `dkdv`: `dk = qᵀ ds` | M | QK | `D_CHUNKS` |

Each of those pairs had been *one loop with one bound*, because the bounds coincided. So
unlike the forward — where no loop bound moved at all and the work was entirely in constants
that had been sized once — here the loops themselves split. That costs the interleaving of
the two MFMA chains, which the scheduler had been free to overlap; a symmetric build is
unaffected because the two loops keep their old trip counts.

Everything else is the forward's change again, twice over: `STRIDE_TOKEN_KV` became
`_K`/`_V`, `global_idx_q`/`global_idx_kv` became four functions (Q and dQ at the QK extent,
dO at the V extent; K and dK against V and dV), the `num_records` bounds went per tensor, and
each row-major LDS tile got its own swizzle and padding derived from its own width — `dq`'s K
and V tiles, and `dkdv`'s Q and DO tiles in both orientations. The cooperative loads, which
had read K and V (and Q and DO) through *one* geometry and one `g_idx`, are now two
independent walks; `_load_geometry` is the same helper the forward carries, for the same
reason. `dkdv`'s two accumulator banks are also no longer the same length, since `dk` is
`D_CHUNKS` wide and `dv` is `D_CHUNKS_V`, which the register-staging iter_args had to learn.

With the dims equal every split constant collapses to its old value, confirmed by LDS
footprint: `dq` at D128 is still exactly 24576 B and `dkdv` still exactly 32768 B, and the
existing 169 tests pass unchanged. The asymmetric case is checked on *gradients* rather than
on the forward output, because `dk` is written at the QK extent and `dv` at the V extent out
of those different-length banks, so a mixed-up bound lands in a gradient and nowhere else.

One knock-on: the autotuner's `dkdv` LDS filter had computed a footprint from one head dim,
so it offered a `block_m` of 64 that the builder then rejected for a wide pair. It now sums
the two extents, which is what the kernel actually spends.

This also retires a refusal that used to be load-bearing. The forward/backward mismatch the
"chosen together" rule exists to prevent was real — this forward writes LSE in natural log,
so a silent fall-through to Triton's log2 backward would have returned wrong gradients rather
than slow ones — but with both directions serving the shape there is no longer a mismatch to
guard, and `_head_dim_supported` no longer asks which direction is calling.

## Why `AUTO` cannot pick this backend, and where it would pay if it could

`AUTO` never selects FlyDSL, and the reason is structural rather than a missing feature or
an untested claim. `stats_are_log2` — whether the wrapper multiplies the returned LSE by
`ln2` — is decided in the eager frontend from the literal `kernel_options["BACKEND"]` string
at Dynamo trace time, which is *before* Inductor lowers anything and therefore before any
`AUTO` decision exists. The frontend has no way to learn what Inductor went on to choose. A
backend whose LSE base differs from Triton's therefore cannot be selected later.

This is not our constraint, it is the parity design, and we follow it rather than work
around it. The parity branch does not know `FLYDSL` at all — its `_Backend` is
`["AUTO", "TRITON", "FLASH", "TRITON_DECODE"]` — so its analogue of this backend is `FLASH`,
and `FLASH` there is gated `backend != "FLASH"` at two sites, forward and backward, exactly
as we gate on `FLYDSL`; it is natural-log, via `stats_are_log2 = BACKEND != "FLASH"`; and it
is never `AUTO`-selectable. The only thing `AUTO` picks in the parity branch is Triton
versus Triton-decode, and both are log2.

Our single change to that mechanism was widening their string comparison into
`_NATURAL_LOG_LSE_BACKENDS` so it could hold a second member. Same layer, same trace-time
timing, same shape, generalised from one natural-log backend to two. So `AUTO` declining
this backend is the parity behaviour applied to a second backend that obeys the same rule —
and making it `AUTO`-selectable would move *away* from parity, since it would break the
invariant that the LSE base is a function of the literal `BACKEND` string.

Unblocking it means one of: writing log2 LSE like Triton (which reverses the deliberate
choice recorded in the roadmap, and moves the `exp`/`exp2` and `grad_logsumexp` handling in
the backward), or plumbing the convention back out of Inductor to the wrapper. Neither is a
gate change. There is also a safety argument for the current shape: under `AUTO` a forward
that was served here and a backward that declined would pair natural-log LSE with Triton's
log2 backward and produce *wrong* gradients, and the forward gate cannot fully predict the
backward's eligibility because it does not yet know whether a capture will need a gradient.

**The performance argument, since it is the other half of the question.** Forward and
backward, bf16, both backends `max_autotune`, every cell's gradients checked against eager,
as `fly/tri` on TFLOP/s over four shapes (1x8x4096, 2x16x2048, 4x8x4096, 2x32x4096) — the
range is across those shapes (`shape_ladder.py --backward`):

| head_dim | plain | score_mod | causal | verdict |
|---|---|---|---|---|
| 64 | 0.63–0.90x | 0.79–0.98x | 0.78–1.15x | Triton |
| 96 | 1.16–1.47x | 1.19–1.50x | 1.22–1.73x | **FlyDSL** |
| 128 | 0.75–0.92x | 0.75–0.88x | 0.96–1.12x | Triton |
| 160 | 1.82–2.16x | 1.76–2.20x | 1.88–2.16x | **FlyDSL** |
| 192 | 1.51–1.83x | 1.45–1.64x | 1.78–2.00x | **FlyDSL** |
| 224 | 1.43–1.67x | 1.40–1.60x | 1.61–1.86x | **FlyDSL** |
| 256 | 0.91–1.16x | 0.95–1.07x | 1.11–1.26x | a wash, causal aside |

The rule that falls out is clean and cuts by head_dim rather than by shape or by mod: **96,
160, 192 and 224 win every single cell**, worst case 1.16x and typically 1.4–2.2x. **64 and
128 lose every dense and every score_mod cell** and only reach parity on causal. 256 is a
wash outside causal.

That split is not arbitrary, and it is the same one the deficit analysis below arrives at
from the other direction: 64 and 128 are exactly the head_dims whose granule count is a
power of two, so they already had a working XOR swizzle and gained nothing from the row
padding that lifted 96/160/192/224 — and they are the sizes Triton itself is most heavily
tuned for. So if `AUTO` is ever unblocked, `head_dim in {96, 160, 192, 224}` is the
condition to select on, and it needs no shape or variant term.

Until then the practical advice is the same rule stated the other way: ask for
`BACKEND="FLYDSL"` by name at those four head_dims, and do not bother at 64 or 128.

## Fast-math, and the two flags these kernels cannot assert

All three kernels run their fp arithmetic under a fast-math flag set, but **not** the full
one. `FastMathFlags.fast` is all seven flags, and two of them — `nnan` and `ninf` — promise
the optimizer that no operand or result is ever a NaN or an infinity. That is false here by
design. A masked element *is* `-inf`, both at the `mask_mod` site and in the causal fold,
and a fully-masked row's LSE *is* `log2(0) = -inf`, which is the value `flex_attention` is
defined to return. Asserting those two turns every guard that keeps such a value from
becoming a NaN into dead code the optimizer may delete: the finite running-max seed (because
`(-inf) - (-inf)` is NaN and would poison the row's accumulator for the rest of the walk),
and the `l == 0` check before the reciprocal (because `o * rcp(0)` is `0 * inf`).

This was **not** a latent risk. Measured on the build these kernels shipped on, a fully
masked row's LSE came back as denormal garbage — `0.0`, `1.79e-43`, `3.59e-43`, ... — rather
than `-inf`, because `log2(0)` had been folded away entirely. Any row masked in full was
returning a wrong LSE, and since the backward reads LSE to rebuild `p`, it was feeding that
into the gradients too. The mechanism is worth stating plainly, because it is the reason the
bug survived a 157-test suite: a NaN or a denormal in one row of a tensor is not a large
relative error, so every tolerance-based assertion passed.

What is kept is the part that pays — `contract` for the softmax multiply-adds, `reassoc` and
`arcp` for the scale folding and the reciprocal, `nsz`, `afn` — and the function-level
`no-nans-fp-math` is dropped for the same reason as `nnan`, while `unsafe-fp-math` stays,
since it buys reassociation without claiming a value never occurs.

**Dropping the two is free.** Kernel-only forward, B=2 H=8 S=4096 causal, against the full
set: 589.2 vs 587.5 µs at D=64, 1033.4 vs 1032.5 at 128, 1777.9 vs 1781.7 at 192, 2701.6 vs
2699.3 at 256 — a 0.4% spread in both directions, which is run-to-run. Backward register
pressure is unmoved (`dq` at D=128 still 256 VGPRs and 4 spilled, `dkdv` still 256 and 78).

`test_a_fully_masked_row_gives_zero_output_and_minus_inf_lse` pins it, with exact assertions
rather than a tolerance for the reason above. Its negative control is to put
`FastMathFlags.fast` back, which reproduces the denormal LSE.

## GPU architectures

What an architecture has is declared in
[`arch_caps.py`](../../kernel/vendored_templates/flydsl/arch_caps.py), and both sides of the
backend read that one table: the gate decides whether a graph can be served, and the kernel
builders take their instruction selection and LDS budget from the same entry.

| | gfx942 (CDNA3) | gfx950 (CDNA4) |
| --- | --- | --- |
| Matrix core | MFMA | MFMA |
| LDS per workgroup | 65536 B | 163840 B |
| DMA-to-LDS | 4 B only, so unused | 16 B (`buffer_load_dwordx4_lds`) |
| Transposing LDS read | no, Vᵀ is staged in LDS | `ds_read_tr16_b64` |
| MFMA32 K | 8 | 16 |
| Fused O store | per-lane `dwordx2` | `permlane32_swap` + `cvt_pk_bf16_f32` |

This replaced an allowlist of architecture *names*, and the reason is that the two questions
it was answering are not the same question. "Which architectures have the instructions" is a
fact about silicon; "which have we run" is a fact about our test coverage. Merging them meant
gfx950 — which has every instruction the CDNA4 paths ask for — was refused by default, while
a capability was simultaneously being handed out by *negation*: the forward selected its
DMA-to-LDS prefetch on `not gpu_arch.startswith("gfx942")`, so any architecture that merely
was not gfx942 claimed a 16 B instruction it might not have, and would have failed in the
assembler rather than at a gate.

So gfx950 is served without a flag, and what stands behind it is a codegen argument rather
than a numerical one. `test_gfx950_builds_from_a_gfx942_host` drives the whole head-dim
ladder through both directions' builders for a gfx950 target, and
`isa_stats.py --arch gfx950` takes the same configs to ISA. Every one lowers, and the forward
comes out *better*: at D128 causal, 242 VGPRs with no spill against gfx942's 256 with 22
spilled, because MFMA K=16 halves the MFMA issues and the hardware transpose removes the
staging registers. Read those numbers as "the CDNA4 paths are real and the compiler is happy
with them", not as a correctness claim — nothing here has produced a number on CDNA4 silicon.

**The backward now takes `mfma_k16`, and only there.** Its GEMM1 — `sᵀ = k qᵀ` and
`dpᵀ = v doᵀ` in `dq`, the matching pair in `dkdv` — reduces over head_dim out of a
row-major tile, which is the one operand path whose address is already parameterized on the
step width. The other three GEMMs (`ds → dq`, and `dv`/`dk`) build their packs four at a
time out of computed values and read the Kᵀ/Qᵀ tiles through a swizzle derived for that
width, so widening them is a re-derivation rather than a constant change; they stay at
32×32×8. The ISA shows both shapes in one kernel, and `dq` at D128 goes from 48 MFMA issues
to 32 (16 `v_mfma_f32_32x32x16_bf16` plus 16 `v_mfma_f32_32x32x8_bf16`), 256 VGPRs to 238
with no spill. `dkdv` at D128 keeps 256 VGPRs but drops from 76 spilled to 56. One config
moves the wrong way — `dkdv` at D64 goes 216 to 244 VGPRs — and it still does not spill.

The remaining backward gaps are `lds_transpose_read`, `permlane_o_store` and
`dma_to_lds_b128`, all of which are LDS-layout changes rather than instruction selection:
the transposing read would retire the Kᵀ and Qᵀ tiles outright, which is the largest single
win left on CDNA4 and also the one that most changes the fragment maps.

And **the K swizzle was derived against 32 LDS banks**, which CDNA4 doubles to 64. It is a
permutation either way, so answers do not change, but its conflict-freedom has not been
re-derived. This applies to the wider read too: the XOR permutes 16-element granules and
never touches the low four bits of the column, so an 8-element read stays contiguous, but
"contiguous" is not "conflict-free".

Anything outside the table is refused by naming what is missing rather than its own name.
gfx1201 has an entry for exactly that reason, and what it is missing is a kernel body, not a
gate: the flex bodies emit MFMA and RDNA4 has WMMA. That is not one instruction apart.
RDNA4's WMMA is 16×16×16 on wave32 against CDNA's 32×32×16 on wave64, so a lane holds 8
accumulator elements instead of 16 columns of one row, and every score-site index — the
softmax column mapping, the causal and window bounds, the bias and dropout offsets — is
derived from that layout. Upstream's own gfx1201 forward computes `S = K Qᵀ` rather than
`Q Kᵀ` for the same reason, because one WMMA's result has to land as the next one's operand.
None of the MFMA glue here survives that, so reaching RDNA4 is a second body.

Worth knowing before assuming upstream shortens that: upstream's gfx1201 flash attention is
mature (a dense prototype with causal, window, varlen, GQA, bias, dropout, LSE, head dims
16–512, and a split backward tested to 512) but it has **no `score_mod` or `mask_mod` hook
on any architecture** — CDNA included. Its masking is region-split loops emitting fixed
`-inf` fills at build time, so what would have to be built is the mod-callback layer itself,
which is the part this backend already has and upstream does not.

That layer would not start from nothing, though. Their forward absorbs the `K Qᵀ` transpose
into its index derivation rather than into a layout pass — it already reads scores as
`S[q_idx, kv_idx]` and already unpacks them to a flat array at the site where its causal,
KV-tail, bias and dropout logic runs, which is where a mod site would go. The sharpest
concrete mismatch is small enough to state exactly: their accumulator gives **eight**
contiguous KV columns per group, where the MFMA site here is built around four and
`build_flex_flash_generic_module` rejects any `mod_vec_size` outside `(1, 2, 4)`. Their
backward is the harder half — three kernels with no mod infrastructure at all, and a
`joint_mod` over 8-wide fragments.

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
| `flydsl.autotune_mod_vec_size` | `TORCHINDUCTOR_FLYDSL_AUTOTUNE_MOD_VEC_SIZE` | on |
| `flydsl.autotune_qk_prefetch_depth` | `TORCHINDUCTOR_FLYDSL_AUTOTUNE_QK_PREFETCH_DEPTH` | off |
| `flydsl.autotune_backward_tile` | `TORCHINDUCTOR_FLYDSL_AUTOTUNE_BACKWARD_TILE` | on |
| `flydsl.autotune_forward_block_m` | `TORCHINDUCTOR_FLYDSL_AUTOTUNE_FORWARD_BLOCK_M` | on |
| `flydsl.autotune_kv_gpfetch` | `TORCHINDUCTOR_FLYDSL_AUTOTUNE_KV_GPFETCH` | on |

`kernel_options={"MOD_VEC_SIZE": n}` pins the mod width (1, 2 or 4) and skips autotuning;
`kernel_options={"DKDV_TILE": (kv, q)}` pins the backward's dk/dv tile;
`kernel_options={"BLOCK_M": n}` pins the forward's Q tile;
`kernel_options={"ENABLE_KV_GPFETCH": bool}` pins the forward's K staging;
`kernel_options={"PACK_GQA": bool}` pins whether the GQA group shares a Q tile;
`kernel_options={"LAYOUT": "bhsd"|"bshd"}` pins which layout the kernel indexes, which is
otherwise read off the strides of q/k/v (and `do`) so that nothing has to be copied.

The forward's Q tile is swept only at 32 heads and up, which is where the old
`BLOCK_M = 256 if num_heads >= 32` heuristic engaged and the only place the two candidates
differ. Measured, that heuristic was wrong at six of seven head_dims — 19% at 64 and 2.5x
at 256 — and right only at 160, which the sweep still picks. Upstream instead dispatches
the tile on total work (`batch * seq * heads`); measured here the taller tile loses at
every shape below 32 heads we tried, including large-batch and 16k-sequence ones, so the
head count is the better gate.

A short Q sequence is the exception, and takes a 64-row tile at any head count. There is no
KV walk to amortise over rows that do not exist: a decode shape has one Q row, and because
the MFMA issue count follows the tile rather than the rows in it, the 127 padding rows *are*
the cost. This was worth 1.6x uniformly on decode shapes, and it needed one thing unlocked —
the workgroup size had been written as "256 below 128 rows, else 512", which happens to pin
`ROWS_PER_WAVE` to 32 for exactly those two heights and quietly excludes every other. Written
as the relation it always was, `BLOCK_M // 32` waves, 64 becomes expressible. 64 is the floor
rather than 32 because a wave owns 32 rows, one per lane, and FlyDSL rejects a 64-thread
workgroup, so a one-wave kernel is not a thing this body can be.

## Packing the GQA group into the Q tile

`kernel_options={"PACK_GQA": True|False}`, otherwise on whenever it is legal and `Sq <= 64`.

A 64-row tile is still 63 rows of padding for a one-row Q sequence, and that bill was being
paid once per Q *head* even though a GQA group's heads share a KV head and therefore share
the whole KV walk. Packed, one block serves a KV head and the tile's rows are `(q_token,
q_head_in_group)` pairs, so the same padded work happens `G` times less often and the grid
shrinks by `G`. The row map is `row = token * G + head_in_group` rather than the other way
round, so both halves are a shift and a mask: `G` is a compile-time constant and `Sq` is not,
and the other layout would need a runtime divide per lane.

Everything downstream follows from `q_head_idx` becoming per-lane, which the index helpers
and the mod site already accept because they close over it as a value. Two things do not:

- The **O store** had leaned on `o_rsrc`'s `num_records` to drop a partial tile's rows. A
  packed row past `seq_len` is not past the bound any more — it is a valid address in the
  *next* head of the group — so the store is predicated on the row instead. The LSE store
  already was.
- The **`num_records` bound** is a property of the buffer descriptor and so has to be
  workgroup-uniform, while `slice_q` is per-lane once packed. It is therefore taken at the
  *last* head of the group. An earlier head's out-of-range rows are then in bounds and read
  the next head's data, which is harmless — every lane owns its own row, and the only
  cross-lane reduction is the half-wave `xor` shuffle, where lanes 0–31 and 32–63 hold the
  *same* rows.

Packing is mutually exclusive with `block_mask`: a BlockMask is regridded per
`(b, h, q_tile)` and cannot describe a tile that spans heads. It also needs `G` to divide the
tile, so a group of 3 never packs at the 64/128/256 heights the forward offers.

Because it is a pure remapping the test bar is *bit-identical* output and LSE against the
unpacked kernel, not a tolerance, across seven shapes including partial and multiple packed
tiles and MQA. On `(1, 8192)` D128 it measured 1.00x at grids that already fit in the 80 CUs
and up to 4.70x where they did not, and never lost — so it is a rule rather than an autotune
choice. It is asked only of short Q sequences because at prefill the packed and unpacked
grids are the same size and there is nothing to win.

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
  as freely as in the forward, but a capture that itself requires grad is refused: the joint
  graph hands its gradient over as `zeros_and_scatter`, and the FlyDSL `modification()` path
  does not emit scatter graphs. So a learned bias tensor is out; an ALiBi slope table is
  fine to read. This one is parity with CuteDSL rather than a deficit against it -- its
  backward refuses them too, `NYI: Flex Flash Attention bwd doesn't support captured grads
  yet.` -- so Triton is the only backend that services a capture requiring grad.

  ~~Its gradient needs atomics~~ — that reason, given here for a long time, is true of only
  half the problem, and the donor survey settles which half. A dense `[B, H, Sq, Skv]`
  capture needs no atomics at all: the `dq` grid is one workgroup per
  `(batch, q_head, q_block)` — ours as much as upstream's, see `flex_flash_bwd_generic.py`
  — so every element of it has exactly one writer and a plain store suffices. Upstream's
  contract mandates that bias shape for precisely this reason — dense rank-4 only, no
  broadcast strides, and it refuses bias combined with causal or a window at build time,
  which is a restriction flex semantics could not adopt anyway. The expensive half is a
  *broadcast* capture such as an `[H]` slope table, where every `(q, kv)` sums into one
  element and atomics or a reduction pass come back. So the item splits, and the cheap half
  is tractable whenever the scatter-graph codegen is.
- ~~No gradient through the LSE~~ **Fixed, and it cost nothing per element.** A loss may
  read the LSE that `return_lse=True` returns, and a non-`None` `grad_logsumexp` used to be
  refused outright. It is now a `has_dlse` build flag on both backward kernels and an
  optional `DLSE` plane beside `LSE` and `DELTA`.

  Cheap in `dq`, and it has to be spelled carefully in `dkdv`. `isa_stats.py --backward`
  puts both head_dim 128 backward builds exactly at the 256-VGPR cap, with `dq` spilling 4
  and `dkdv` spilling 78 — asking for two waves per SIMD caps a wave at half of the 512 the
  part has, and the compiler spills rather than exceed it. Loading the `dlse` plane into
  four `v4f32` and carrying them to the `ds` site pushed that to 96 spills and cost 50% of
  the kernel (4962 → 7432 µs). Subtracting into `delta` on the load line instead — one
  vector `arith.subf`, after which the plane is dead — gives 80 spills and 5267 µs, within
  6% of the build without it. `dq` is 256/4 either way, since its fold is one row scalar.

  It is cheap because of where the term lands rather than because of any care taken. `lse`
  is a log-sum-exp over the *post*-mod scores, so `d lse / d s = p`, and the kernels already
  form `ds = p * (dp - delta)`: adding `dlse * p` gives `p * (dp - (delta - dlse))`, so the
  whole thing folds into the row scalar the softmax derivative already subtracts. One load
  and one subtract per row in `dq`, per element in `dkdv` where `delta` is already
  per-element, and the elementwise loop is untouched. The fold has to happen *before* the
  joint-graph site, since a `score_mod`'s chain rule is evaluated on that cotangent, which
  is why the tests sweep both mods rather than the plain case alone.

  CuteDSL is the reference here and refuses only head_dim 256 on SM100/SM110; we have no
  such exclusion.
- **Backward block skipping is gated on occupancy.** Both backward kernels can walk block
  lists — `dq` a KV list per Q tile, `dk`/`dv` the transpose — but only when the grid has at
  least 2 workgroups per CU. Causal work per tile is triangular, so below that they are all
  resident, the kernel finishes when the heaviest one does, and skipping the light ones buys
  nothing while still paying the index loads (measured 0.89–1.18x under the threshold
  against 1.54–2.39x over it). Small-batch causal shapes therefore still walk densely, on
  purpose. No donor for this one: upstream's kernels do bound the loop, but from causal and
  window *arithmetic*, which it can do because causal is a build flag of known shape there.
  Ours is a runtime BlockMask, so that math does not apply and the donor was our own
  forward.
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
- **The `dq_write_order` fields are ignored on purpose.** `BlockMask` carries three optional
  tensors -- `dq_write_order`, `dq_write_order_full`, `dq_kv_order` -- documented as
  deterministic dQ metadata for the block-sparse FLASH backward, and the flex lowering hands
  them to CuteDSL and not to us. They order accumulation where dQ has more than one writer.
  Ours has exactly one, a workgroup per `(batch, q_head, q_block)`, so determinism is
  structural here and there is nothing for the ordering to fix. Checked rather than assumed:
  against a mask built with `compute_dq_write_order=True` the backward runs unchanged, its
  gradients are bitwise identical across runs, and bitwise identical to the same mask built
  without the metadata.
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

  One candidate for that deficit is now closed. Upstream's XCD workgroup swizzle
  (`cf1db2f`) keeps a head's Q blocks on a single XCD so its K/V stays in that XCD's 4 MB
  of L2, and it was the last unported forward commit. Implemented both ways — Q tile on the
  fast axis, and the full remap that gives each XCD a contiguous run of heads — it measures
  **1.00–1.01x on dense** across four shapes at both head_dims, and 0.90–0.97x on causal.
  Causal moving is what makes the dense column trustworthy: placement demonstrably changed,
  and dense did not care. It cannot care, because there is no bandwidth wall to remove —
  assume the worst case where every Q tile re-reads its head's K/V from DRAM and the dense
  forward still only asks 0.60 TB/s of a part that achieves 1.90. Whatever the deficit is,
  it is not L2 locality.

  What the sparse losses *are* made of is now measured, and it is not throughput. Fitting
  time against blocks walked (`walk_cost.py`) puts our per-block rate ahead of Triton's at
  both head_dims — 1080 against 1261 ns/block at 64, and 1947 against 2664 at 128 — while
  our fixed per-call cost is 150 µs at head_dim 64 and 344 µs at 128 against Triton's 108 µs
  and −26 µs. So the cost we pay once per Q tile roughly doubles with head_dim, which is
  what a Q load plus an O and LSE write should do, and Triton hides its own behind the KV
  walk where we do not: at head_dim 128 its per-block work doubled too, giving it more to
  hide behind, and its intercept went to zero. A mask that walks few blocks has nothing to
  amortise ours against, which is exactly the shape of the remaining forward losses
  (`document_mask` 0.77x, `sliding_window` 0.91x at D128).

  Worth being precise about where that cost is *not*: at one block per tile,
  `profile_call.py` shows the whole call as a single kernel and nothing else, so this is
  neither the layout copies (gone) nor the regrid launches (cached). It is inside the
  kernel, and closing it means overlapping the prologue and epilogue with the walk rather
  than removing any work. Note also that a fit taken under a mask does not extrapolate to
  the dense column, since a dense walk evaluates no `mask_mod` at all — these numbers bound
  the sparse story, not the dense one.

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
- **Symbolic captures work, by specializing on the value.** A mod closing over a value
  derived from a dynamic shape — a window of `seq_len // 4` — arrives as a sympy
  expression rather than a tensor, and `rename_indexing` turns it into a kernel argument
  name at the use site. That name cannot resolve, because the mod bodies are built at
  *module* level while the name is a parameter of the kernel function, and the result was
  `name 'ks0' is not defined` after an earlier `'FloorDiv' object has no attribute
  'get_size'` from the capture gate counting it as a tensor. Both are fixed: such a
  capture costs no aux slot, and its value is guarded to an int and emitted as a
  module-level constant. The guard is what makes that sound — a different value
  recompiles rather than reusing a module with the old constant baked in, which the test
  checks by running three sequence lengths, each implying a different window, in one
  process. The cost is that such a mod gives up dynamic sharing on that axis; CuteDSL
  instead threads an `aux_scalars` tuple through the mod ABI, which here would mean
  changing the mod call site in all three vendored kernels.
- **Device 0-d tensor captures work; CPU ones are rejected**, in `flex_attention()`
  rather than here, since by the time this backend sees them they have been realized and
  no longer look like scalars. Worth noting the Triton path *crashes* on that shape
  (`RuntimeError: unbacked_bindings`) where this one declines it.
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
  "What is left in the forward is dense, not sparse" above.

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
- **No packed mask intervals**, the SM100+ optimization CuteDSL has. Nothing to take from
  parity here either: it has no packed or runtime block list at all, and bounds its tile
  walk arithmetically from the causal, window and varlen parameters instead
  (`decompose_causal_regions`). That works because those are build flags of known shape
  there; a `mask_mod` is not, which is the same reason its backward skipping does not port.
- **No fusion**, matching CuteDSL: no epilogue or prologue support.
- **gfx950 is unexecuted, and its backward is only half specialized.** It is served without
  a flag on codegen evidence — see [GPU architectures](#gpu-architectures) — but no number
  here has come off CDNA4 silicon, and the backward takes only `mfma_k16`, in GEMM1; the
  three LDS-layout capabilities remain forward-only.
  Whether to carry *more* unexecuted code than this is the plan's open decision D3, and it
  is a real one: its phases 3–5 would add roughly 10k lines that cannot be run here.
- **gfx1201 is refused, and not for want of a gate.** The flex bodies emit MFMA; RDNA4 has
  WMMA. Reaching it means a second kernel body, not a capability entry.

[`PARITY_MIGRATION_PLAN.md`](PARITY_MIGRATION_PLAN.md) works through most of the above: it
carries the measured gfx942 baseline against autotuned Triton and aten, and a phased plan
for the BlockMask plumbing, head_dim 64, backward, and gfx950/gfx1201. It also records why
rebasing onto the upstream `parity` FMHA family does not by itself close the gap.

Two PRs are open against `main` doing overlapping work, AMD's gfx950 FlyDSL forward and
backward. [`UPSTREAM_PR_COMPARISON.md`](UPSTREAM_PR_COMPARISON.md) is the comparison: what
each side is better at, and the four things in them worth taking. The short version is that
they accept only an identity `score_mod`, so they are a fast dense FlashAttention reached
through the FlexAttention API rather than a competing backend — and that their decode path,
which we do not have at all, is the one feature they have clearly tuned past us.

Its §8 has since answered the question it opens with, and the answer for gfx942 is **no**.
Every lead that moved the number turned out to be ours to take — the 512 B LDS overage, the
layout copies, the block-skip plumbing — and each was assumed at some point to need parity.
The one item that genuinely needs parity's schedule needs CDNA4 features this part does not
have. What parity is still for is gfx950 and gfx1201 reach, which turns on whether hardware
appears. Individual techniques are still worth porting on their own merits, and the
roadmap's donor survey tracks which ones landed, which were closed by measurement, and
which do not apply.

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

**Two FlyDSL versions, both validated.** `flydsl.expr.buffer_ops` exists in 0.2.4 and was
removed in 0.3.x, where those helpers moved kernel-side; import `flex_kernels.buffer_ops`,
which picks whichever is present. Importing the library path directly makes the whole flex
suite *skip* rather than fail, which is easy to mistake for green.

The suite passes 168/168 on gfx942 against both 0.2.4 and 0.3.2, and
`_FLYDSL_SUPPORTED_RELEASES` in `flydsl_utils.py` admits exactly those two, since a
release that has not been run is not a release we can claim. **0.3.2 is the faster of the
two**, by 10–22% on the forward — kernel-only, `B=2 H=8 S=4096` bhsd causal, run-to-run
spread under 1%:

| head_dim | 0.2.4 | 0.3.2 | 0.3.2 speedup |
| --- | --- | --- | --- |
| 64 | 593 us | 525 us | 1.13x |
| 96 | 817 us | 670 us | 1.22x |
| 128 | 1031 us | 877 us | 1.18x |
| 160 | 1502 us | 1338 us | 1.12x |
| 192 | 1773 us | 1604 us | 1.11x |
| 224 | 1984 us | 1806 us | 1.10x |
| 256 | 2694 us | 2223 us | 1.21x |

Every other performance figure in this document was measured on 0.2.4, so read them as a
floor rather than as the number a 0.3.2 user sees. The suite is *slower* in wall clock on
0.3.2 (752 s against 463 s) while every kernel is faster, so that gap is compile time, not
kernel time.

Testing against another release does not need a second interpreter — install it beside the
one in use and shadow it on the path, which leaves the working install untouched:

```
pip install --no-deps --target /tmp/fly032/site flydsl==0.3.2
PYTHONPATH=/tmp/fly032/site python -m pytest test/inductor/test_flydsl_flex_attention.py
```

Note: requires the optional `flydsl` package and a ROCm build of PyTorch.

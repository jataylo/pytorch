# What was done on this branch

A FlexAttention backend for ROCm built on FlyDSL, served through Inductor. `fly-dsl-testing`,
branched from `dddd90223f0`, 11 commits, **+19,337 lines across 45 files** in the relevant
paths.

The short version: PyTorch's FlexAttention on AMD hardware runs on Triton. This branch adds a
second backend that reaches AMD's hand-written FlyDSL flash-attention kernels instead, and
makes them serve a *real* `score_mod` and `mask_mod` — an arbitrary traced subgraph lowered
into the kernel at the score site — rather than only dense attention reached through the
FlexAttention API.

## At a glance

| | |
| --- | --- |
| Direction | Forward and backward (`dq`, `dkdv`) in one lowering, chosen together |
| Architecture | gfx942 (CDNA3) and gfx950 (CDNA4), neither behind a flag, off a capability table |
| dtype | bf16 and f16 |
| head_dim | Every multiple of 32 from 64 to 256, and `qk_head_dim != v_head_dim` in either order, forward and backward |
| Captures | 4 slots, f32, rank ≤ 4, either mod |
| seq_len | Any, ragged tails included, and cross attention (`Sq != Sk`) |
| GQA | MHA / GQA / MQA, both directions, no atomics |
| Tests | 168 passing on gfx942, against FlyDSL 0.2.4 **and** 0.3.2 |
| Performance | Forward 1.09x geomean over 84 cells against autotuned Triton, forward+backward 1.28x |

Selectable with `BACKEND="FLYDSL"`. `AUTO` cannot pick it, by design — see
[Why AUTO cannot pick this backend](#why-auto-cannot-pick-this-backend).

## Flex attention

### Ported the FlyDSL flash attention kernels

The attention kernel is **not generated**. It is a hand-written FlyDSL kernel carried in
[`kernel/vendored_templates/flydsl/`](../../kernel/vendored_templates/flydsl), following the
CuteDSL precedent so the branch is self-contained and needs no sibling checkout — only the
optional `flydsl` runtime. These are derived copies, re-synced by diffing against upstream
rather than restyled, and excluded from `lintrunner` for the same reason the CuteDSL ones are.

Three kernels: the forward (`flex_flash_generic.py`), and the two backward kernels (`dq` and
`dkdv`). Two hooks were cut into each — one at the score site, one at the mask site — and
everything else this backend does is fill those hooks and call the launcher.

What the port added to the donor kernels, beyond the hooks: an LSE output plane, aux-tensor
reads for captures, mod vectorization, a KV block-skip loop driven by `BlockMask`, and split
Q/KV sequence extents.

### Made them serve FlexAttention

This is the part that distinguishes the backend from a fast dense FlashAttention wearing the
FlexAttention API.

- **`score_mod` is a traced FX subgraph lowered into the kernel** at the score site.
  `ModificationWrapperFlyDSL` walks the graph, resolves subgraph placeholders (`m`, `n`) to
  the kernel's own `q_idx` / `kv_idx`, and emits calls into a mod body.
- **`mask_mod` likewise**, and where a mask is present all three kernels skip KV blocks
  rather than evaluating a predicate over dense work.
- **Captured tensors are discovered during rendering** — not declared up front, because you
  cannot know what a mod reads until you lower it. They are registered as kernel arguments
  mid-render and assigned aux slots, which is why `NUM_AUX_TENSORS`, `AUX_SPECS` and
  `MOD_KEY` are emitted *after* the mod definitions.
- **The backward differentiates the mod**, taking the joint graph at the `ds` site, so a
  soft-cap `score_mod` produces correct gradients rather than being silently ignored.
- **A runtime shim (`flydsl_mod_runtime.py`) expands the ops FlyDSL cannot lower on AMDGPU**:
  `exp`, `tanh` and `sigmoid` over `exp2`, and the transcendental family (`erf`, `erfc`,
  `atan`, `atan2`, `asin`, `acos`, the four hyperbolics) over exp2/log2/sqrt. Only `erfinv`,
  `lgamma` and `digamma` remain rejected. Expansion accuracy is *measured* against eager in
  the suite rather than assumed, since it is a property of that file.

### Inductor hookup

Two compilers run in sequence: Inductor lowers the mod subgraphs to FlyDSL **source text** and
renders a self-contained Python module; FlyDSL's JIT then traces that module and compiles a GPU
binary.

The generic half is shared with any FlyDSL template and is not FlexAttention-specific —
`FlyDSLTemplate` (definition, registration, choices), `FlyDSLTemplateKernel` (arguments and the
`def_kernel` / `gen_defines` / `get_output` render hooks), `FlyDSLScheduling` (dispatched from
`cuda_combined_scheduling.py`, compiling through `async_compile.flydsl()`), and
`FlyDSLTemplateBuffer` in `ir.py` beside `CuteDSLTemplateBuffer`.

The flex half sits on top: `FlyDSLFlexTemplateKernel` adds subgraph bodies, a CSE scope and
`modification()`; `FlyDSLOpOverrides` is the op table, which emits strings only and must never
import `flydsl` because it runs during lowering on any machine; and
`kernel/flex/flydsl_flash_attention.py` is the eligibility gate and kernel factory, dispatched
from `flex_attention.py`.

The gate is deliberately loud. `_use_flydsl_flash_attention()` returns False for any other
backend, and **raises** if `FLYDSL` was asked for but cannot be served, rather than falling
through to Triton. It has to: our forward writes LSE in natural log, so a silent fallback to
Triton's log2 backward would compute *wrong gradients* rather than merely slow ones.

### Autotuning

One template choice is appended per `mod_vec_size`, and five axes are swept. The discipline is
that **an axis earns a sweep only if measurement says neither setting can be pinned** — the
forward's KV-staging axis clears that bar (pinning it on costs 1.121x geomean on the 20 of 84
blocks where off won; pinning it off costs 1.098x on the other 64), while `dq`'s and `dkdv`'s
do not and are defaults instead.

That discipline exists because FlyDSL search is expensive: ~1.54s per cold kernel, of which
0.40s is emitting MLIR from Python and 1.14s is the MLIR pipeline, LLVM and ISA. Per unit of
search we are 5–7x costlier than Triton, which is what caps sweep width.

What made that cost *serial* was that FlyDSL compiles on first launch and nothing precompiled.
`flydsl.precompile_workers` (default 8) now compiles choices ahead of the benchmark loop in
subprocesses — **1.9x** end to end on six cold shapes, with the benchmark phase itself falling
81.1s → 15.1s. Two measurements decided the design: processes rather than threads (eight
compiles take 9.5s serially, 1.9s on eight processes, and *more* than 9.5s on eight threads,
because the Python quarter holds the GIL and the native three quarters never release it), and
`TuningProcess` rather than a `ProcessPoolExecutor` (a spawn-based executor makes each worker
re-import the parent's `__main__`, which in a library means re-running the user's script in
eight subprocesses).

### Architectures, and how one is picked up

Declared as **capabilities in one table**
([`arch_caps.py`](../../kernel/vendored_templates/flydsl/arch_caps.py)) read from both sides of
the backend: the gate asks whether a graph can be served, and the kernel builders take
instruction selection and LDS budget from the same entry.

| | gfx942 (CDNA3) | gfx950 (CDNA4) | gfx1201 (RDNA4) |
| --- | --- | --- | --- |
| Matrix core | MFMA | MFMA | **WMMA** |
| LDS per workgroup | 65536 B | 163840 B | 65536 B |
| DMA-to-LDS | 4 B only, so unused | 16 B (`buffer_load_dwordx4_lds`) | none at all |
| Transposing LDS read | no, Vᵀ staged in LDS | `ds_read_tr16_b64` | no |
| MFMA32 K | 8 | 16 | n/a |
| Fused O store | per-lane `dwordx2` | `permlane32_swap` + `cvt_pk_bf16_f32` | n/a |
| **Served?** | yes, on hardware | yes, build + ISA only | **no** |

This replaced an allowlist of architecture *names*, because that allowlist was answering two
different questions with one list. "Which architectures have the instructions" is a fact about
silicon; "which have we run" is a fact about test coverage. Merging them meant gfx950 — which
has every instruction the CDNA4 paths ask for — was refused by default, while a capability was
simultaneously handed out by *negation*: the forward selected its DMA-to-LDS prefetch on
`not gpu_arch.startswith("gfx942")`, so any architecture that merely was not gfx942 claimed a
16 B instruction it might not have, and would have failed in the assembler rather than at a
gate. Capabilities are now declared positively: an arch gets a feature only by naming it.

**gfx942** is the architecture everything was developed and benchmarked on, and the only one
whose numbers come from hardware.

**gfx950** is served without a flag, and what stands behind it is a codegen argument rather
than a numerical one: `test_gfx950_builds_from_a_gfx942_host` drives the whole head-dim ladder
through both directions' builders for a gfx950 target, and every one lowers. The forward comes
out *better* — at D128 causal, 242 VGPRs with no spill against gfx942's 256 with 22 spilled,
because MFMA K=16 halves the MFMA issues and the hardware transpose removes the staging
registers. Read that as "the CDNA4 paths are real and the compiler is happy with them", not as
a correctness claim: **no number here has been produced on CDNA4 silicon.** Two known gaps —
the backward reads *none* of the four CDNA4 capabilities, and the K swizzle was derived against
32 LDS banks where CDNA4 has 64 (a permutation either way, so answers do not change, but its
conflict-freedom has not been re-derived).

**gfx1201 is refused**, and it has a table entry for exactly that reason: so the refusal can
name what is missing, which is a *kernel body*, not a gate. The flex bodies emit MFMA and RDNA4
has WMMA, and that is not one instruction apart. RDNA4's WMMA is 16×16×16 on wave32 against
CDNA's 32×32×16 on wave64, so a lane holds 8 accumulator elements instead of 16 columns of one
row, and every score-site index — the softmax column mapping, the causal and window bounds, the
bias offsets — is derived from that layout. Upstream's own gfx1201 forward computes `S = K Qᵀ`
rather than `Q Kᵀ` for the same reason, because one WMMA's result has to land as the next one's
operand. None of the MFMA glue survives that, so reaching RDNA4 is a second body.

Worth knowing before assuming upstream shortens that work: upstream's gfx1201 flash attention
is mature, but it has **no `score_mod` or `mask_mod` hook on any architecture**, CDNA included.
Its masking is region-split loops emitting fixed `-inf` fills at build time. What would have to
be built is the mod-callback layer — which is precisely the part this backend has and upstream
does not.

### Why AUTO cannot pick this backend

`stats_are_log2` is decided in the eager wrapper from the literal `BACKEND` string at Dynamo
trace time, *before* Inductor picks anything, so the wrapper never learns what backend was
chosen. A natural-log backend therefore has to be opt-in by name.

This is not a gap on our side — it is the design this backend was aligned to. The parity branch
gates `FLASH` identically (`backend != "FLASH"`, at two sites, forward and backward), makes it
natural-log via `stats_are_log2 = BACKEND != "FLASH"`, and never lets AUTO select it. The only
change made to that mechanism here was widening a string comparison into
`_NATURAL_LOG_LSE_BACKENDS` so it could hold a second member. Enabling AUTO would move *away*
from parity, since it would break the invariant that the LSE base is a function of the literal
`BACKEND` string.

The per-head_dim argument for wanting it anyway is real (1.4–2.2x at head_dim 96/160/192/224,
losses at 64/128), which is why the question is documented rather than dismissed.

## Correctness work worth calling out

Four bugs found here were **silent** — plausible numbers, no error raised — which is the failure
mode this backend has to be most careful about, since the alternative to a loud refusal is not
a slow kernel but a wrong one.

| What | How it presented | Why it happened |
| --- | --- | --- |
| **Fast-math flags the kernels contradict** | A fully-masked row's LSE came back as denormal garbage (`0.0`, `1.79e-43`) instead of `-inf` | The kernels ran under `FastMathFlags.fast`, which includes `nnan` and `ninf`. These kernels *deliberately* make infinities: masked scores are `-inf`, and a fully-masked LSE is `log2(0)`. Those flags promise the optimizer no infinity occurs, making every NaN guard dead code it may delete — and it did, folding `log2(0)` away |
| **Cross attention taking one extent** | 0.97 relative error at `sq=512, sk=256`, 1.57 at the reverse | Held as a single `seq_len`, so the KV walk ran to the Q length. Equal lengths make every affected site indistinguishable, which is why the tests holding this up use unequal extents in both orders, plus a ragged pair (512 against 300) |
| **head_dim 64 K swizzle** | ~44% relative error, building perfectly happily | The row mask was hardcoded to 7 rather than sized to `HEAD_DIM`, so the XOR walked off the end of a 64-wide row into the next one |
| **Layout copies read as a sparse-kernel deficit** | `sliding_window` 0.83x, `document_mask` 0.86x, diagnosed as a per-tile setup cost | Not in the kernel at all: the interface was copying q, k, v and the output into BHSD, a fixed 574 µs against a 461 µs kernel. Removing it moved the forward geomean 0.93x → 1.00x |

The fast-math fix is the one to read closely, because a reviewer on one of the upstream PRs
raised it as a *risk* to their kernel and it turned out to be a **live bug** in ours. The flag
set is now the honest subset (`reassoc`, `nsz`, `arcp`, `contract`, `afn`) with `nnan` and
`ninf` dropped, confirmed free by measurement (0.4% spread, backward register pressure
unchanged), and there is a test with a negative control: reverting the fix reproduces the
denormal LSE. A third channel was found later — `fast_fp_math` wraps the whole traced body in
an ambient fast scope that every op without an explicit `fastmath=` inherits — and all four
compile-hint dicts now set the same subset.

Also added: **asymmetric head dims** (`qk_head_dim != v_head_dim`, any admitted pair in either
order), in the forward first and later in both backward kernels. The forward's work was
splitting the cooperative load geometry, since K and V rows have different widths and so need
separate lane groupings, batch counts and partial-lane guards; the backward's was that, twice
over, plus four GEMM loops whose bounds had coincided. Forward speed is 1.19x (qk192/v128) down
to 0.77x (qk128/v64) — the symmetric profile at the same QK dim, not an asymmetric penalty.

## FlyDSL release support

Validated against **both 0.2.4 and 0.3.2**, 168/168 on gfx942 each, and
`_FLYDSL_SUPPORTED_RELEASES` admits exactly those two, since a release that has not been run is
not one we can claim. The two differ in ways the kernels absorb —
`flydsl.expr.buffer_ops` was removed in 0.3.x, where those helpers moved kernel-side, so
`flex_kernels/buffer_ops.py` picks the library copy when present and a vendored copy otherwise.
Importing the library path directly makes the whole suite *skip* rather than fail, which is easy
to mistake for green.

**0.3.2 is faster by 10–22%** on the forward (kernel-only, `B=2 H=8 S=4096` bhsd causal,
run-to-run spread under 1%), so every other performance figure in these docs, taken on 0.2.4,
reads as a floor:

| head_dim | 64 | 96 | 128 | 160 | 192 | 224 | 256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3.2 speedup | 1.13x | 1.22x | 1.18x | 1.12x | 1.11x | 1.10x | 1.21x |

## Comparison against the two reference PRs

Two PRs are open against `pytorch:main` doing what this backend does:
[#193854](https://github.com/pytorch/pytorch/pull/193854) (backward only) and
[#194309](https://github.com/pytorch/pytorch/pull/194309) (forward only). They are **siblings,
not a stack** — both branched off `main` independently from the same fork, four days apart —
and they **cannot both merge as-is**: both add `flex_flydsl_attention.py` with entirely
different contents, and both add `flex_attn_utils.py` with *incompatible APIs behind the same
function names*.

The full treatment is in [`UPSTREAM_PR_COMPARISON.md`](UPSTREAM_PR_COMPARISON.md). The
condensed version:

### The one difference that matters most

**They accept only an identity `score_mod`.** Both gate on `is_trivial_score_graph` and refuse
anything else, and both reject captured score buffers outright. So despite the name, neither PR
is a FlexAttention backend in the sense this one is — they are fast dense FlashAttention reached
*through* the FlexAttention API, with sparsity expressed through `BlockMask` and everything else
fixed at build time. That is a coherent product, and their decode numbers show it pays, but it
means the two efforts are not competing for the same slot, and it explains most of the table
below.

Their `mask_mod` is the more interesting half and a genuinely different design: rather than
codegen, it walks the FX graph and emits a **tuple of instruction tuples over a value stack** —
a small bytecode program specialised into the kernel at build time, with a fixed dict of 16
binary ops. That buys a stable, inspectable, cacheable mask representation and a hard bound on
what a mask can cost; it costs that the opcode dict is the whole language, the int32 restriction
excludes a float bias or slope table, and recognising causal means pattern-matching an
instruction sequence rather than reading the graph.

### Where we are better

| | ours | theirs |
| --- | --- | --- |
| `score_mod` | traced graph lowered into the kernel; 11 further ops expanded in the shim | **identity only**; captured score buffers refused |
| `score_mod` backward | joint graph at the `ds` site, so a soft-cap differentiates correctly | n/a — no non-identity mod to differentiate |
| Architectures | gfx942 **and** gfx950, off a capability table | gfx950 only, `arch.split(":")[0] == "gfx950"` exact match |
| Unknown arch | refused by naming the missing capability | refused by not being the one name |
| head_dim | 7 values, every multiple of 32 from 64 to 256 | 2 pairs: (128,128) and (192,128) |
| dtype | bf16 **and** f16 | bf16 only |
| Directions | one lowering, both directions, chosen together | split across two PRs; the missing half falls to Triton **silently** |
| seq_len | ragged, and cross attention in all three kernels | `Sk % 128 == 0`; #193854 is MHA-only with `Sq == Sk` |
| GQA | both directions, no atomics | forward yes; **backward rejects GQA** |
| Captures | 4 slots, f32 | 4 slots, **int32 only** |
| Autotuning | 5 swept axes, each earning its sweep by measurement | one choice appended, then `configs = []` and the choice hook skipped entirely |
| Measured vs Triton | forward 1.09x geomean over 84 cells, forward+backward 1.28x, on hardware | forward 0.94–1.49x prefill on gfx950; backward unmeasured against Triton |
| CI | 168 tests on gfx942 | none: workflows awaiting approval |

### Where they are better

| | theirs | ours |
| --- | --- | --- |
| **Decode** | packed-GQA decode, `Sq ∈ {1,4,8}`, pipelined KV double-buffering, a `SPLIT_KV` mode for low-parallelism MHA decode. Reports **3.6–7.0x** on top-k-16 sparse decode | **no decode kernel.** Decode shapes are served correctly by the prefill kernel at 2.0–6.5x Triton decode's time (and 3.6x faster than Triton prefill) — see `UPSTREAM_PR_COMPARISON.md` |
| **Real gfx950 validation** | benchmarked on MI355X, ROCm 7.2.53211, four shape families | **build and ISA only.** No CDNA4 silicon here; correctness there is unproven |
| gfx950 schedule | hand-written: owner-wave selection, `waves_per_eu` occupancy hint, dual-wave staging | the generic builder's output, with CDNA4 instructions selected off capabilities |
| Asymmetric head dims | `(192, 128)` first-class in **both** directions | forward any admitted pair either order; **backward refuses** |
| Backward on CDNA4 | written for gfx950 throughout | takes `mfma_k16` in GEMM1; the three LDS-layout capabilities are still forward-only |
| Upstreaming | in the review queue, `albanD` / `drisspg` requested | a local branch |

The first two are the honest asymmetry. Decode is a whole feature we do not have and they have
tuned, and it is the shape where a sparse mask has the most to give, because the dense work per
token is tiny. And no amount of ISA inspection substitutes for running the kernel.

Two of their bugs are **not** ours, and why is instructive: their `_run_compiled` duplicates a
shared helper and drops `os.getpid()` from the cache key, so a fork after a warm launch reuses a
dispatcher bound to the parent's GPU context; and `configure_flydsl_cache_dir()` is never called,
so FlyDSL's disk cache lands outside Inductor's cache root. Both follow from reimplementing the
shared runtime shim rather than calling it.

### A caveat on the comparison

Both authors' most recent replies describe fixes that are **not in the pushed heads** (the
dense-mask helper moved into the lowering, an early return, `_create_empty_block_mask` always
used when `block_mask is None`). #194309's head is from 2026-08-30 and contains none of them.
Anything compared here should be read against the described-but-unpushed design as well as the
code, or it will be unfair on points already conceded.

## What is left

Roughly in value order. Two of these are closed rather than pending — they are kept here
because "we decided not to" and "we have not got to it" are different states and the
difference is easy to lose.

**1. A decode kernel.** The largest remaining gap, and now the only item here that is both
implementable and fully validatable on this hardware. It is a measured gap rather than an
asserted one: see [the decode section](UPSTREAM_PR_COMPARISON.md) and
`benchmarks/transformer/flydsl/decode_gap.py`. Decode shapes are served correctly today by
the prefill kernel, at 2.0–6.5x Triton decode's time and 3.6x faster than Triton prefill.
The work is GQA packing into the M tile first — FlyDSL's time is flat across `Sq` 1, 4 and
8 because a single query row is padded into a `BLOCK_M=128` tile — then a KV split for
parallelism. The MHA row prices the first half on its own: it is the one shape where the
group cannot be packed, and it is also where the deficit is smallest. This is a new kernel
body, not plumbing, and it is fully validatable here.

**2. ~~Asymmetric head dims in the backward.~~ Done** as of `1e8ce52f31e`. Both directions
now serve any admitted `qk_head_dim != v_head_dim` pair in either order, checked on gradients
across six pairs plus one under GQA with a `score_mod`. The backward turned out to need more
than the forward had: there the two extents were never loop bounds, while here they are four
— `dq`'s `sᵀ = k qᵀ` against `dpᵀ = v doᵀ`, and `dkdv`'s `dv` against `dk` — each of which had
been one loop with one bound. LDS was never the obstacle it might have been, as predicted:
all 49 admitted pairs fit gfx942 at `block_n=32`. What did bite was the autotuner's `dkdv`
LDS filter, which computed its footprint from one head dim and so offered a `block_m` the
builder then rejected. See the README's asymmetric section.

**3. The three remaining CDNA4 capabilities in the backward.** `mfma_k16` is taken in GEMM1
as of `158ac9c0685`. `lds_transpose_read`, `permlane_o_store` and `dma_to_lds_b128` are
still forward-only, and all three are LDS-layout changes rather than instruction selection.
The transposing read is the big one: it would retire the Kᵀ and Qᵀ tiles outright, which is
both the largest win left on CDNA4 and the change that most disturbs the fragment maps. Two
narrower pieces sit here too — widening the `ds`/`p` GEMMs to K=16 needs `_kt_swizzle`
re-derived for eight contiguous elements, and the K swizzle's conflict-freedom has never
been re-derived for CDNA4's 64 LDS banks (it stays a permutation, so answers do not change).
Build- and ISA-validatable only.

**4. RDNA4 (gfx1201).** Refused by capability, and what is missing is a kernel body rather
than a gate: `require_caps` demands MFMA, and the lowering declines WMMA before a build is
attempted. RDNA4's WMMA is 16×16×16 on wave32 against CDNA's 32×32×16 on wave64, so a lane
holds 8 accumulator elements rather than 16 columns of one row.

The donor tree has a production gfx1201 forward (`flash_attn_func_gfx1201_aiw.py`, ~2.3k
lines) plus three backward kernels (~4k), and it has no `score_mod`/`mask_mod` anywhere —
the only `mask_mod` match in it is a dropout bitmask builder. So the mod layer is new work.
It is less new than it first looks, though, and the reason is worth recording because it
changes the recommendation. The donor computes `S = K Qᵀ` rather than `Q Kᵀ`, but it absorbs
that transpose into its index derivation rather than into a layout pass: it already reads
scores as `S[q_idx, kv_idx]`, with `q_idx` from `lane16` and `kv_idx` from
`acc_elem_column(i) + klane * 8`, and it already unpacks to a flat `s_raw[]` at exactly the
site where its causal, KV-tail, bias and dropout logic runs. That is a real graft point, not
a hypothetical one — the donor's own comment notes four call sites had open-coded that
column map before it was factored out.

The sharpest concrete mismatch is small and checkable: `acc_elem_column` makes **eight**
contiguous KV columns per group, while our mod site is built around four and
`build_flex_flash_generic_module` rejects any `mod_vec_size` outside `(1, 2, 4)` — "4
contiguous KV cols per lane". A WMMA mod site wants 8. Add to that up to 2 Q row subtiles per
wave, and up to 128 scores per wave against MFMA's 32.

So the recommendation, if this is ever picked up, is to port the donor wholesale and graft
the flex mod and joint layer onto its existing score site — not to write a WMMA body in the
shape of `flex_flash_generic.py`, which would discard the donor's tuning while still owing
the same index re-derivation. The backward is where the genuine work is: the donor's three
kernels have no mod infrastructure at all, and `joint_mod` on 8-wide fragments is the
hardest single sub-problem.

Validation is the reason not to start now. There is no gfx1201 silicon here — all eight GPUs
are MI308X. FlyDSL does expose the RDNA4 WMMA intrinsics on both 0.2.4 and 0.3.2, and the
donor does cross-build under `FLYDSL_GPU_ARCH=gfx1201`, so a port would be build-checkable.
But unlike gfx950, which runs the same validated MFMA bodies with different instructions
selected, a successful RDNA4 build would check almost nothing: the index derivation is both
the new part and the part a build cannot test.

**5. gfx950 on silicon.** Hardware-blocked: every GPU in this environment is MI308X
(gfx942). Everything claimed for CDNA4 is build- and ISA-level, which is strictly weaker
than the reference PRs' MI355X numbers. Nothing else unblocks this.

**6. The dense head_dim 64/128 deficit.** `noop` at 0.90x and 0.88x, still unexplained. The
XCD workgroup swizzle was implemented both ways and measures 1.00–1.01x on dense while
moving causal 0.90–0.97x, so whatever the cause is, it is not L2 locality. Investigation,
not implementation.

**7. CPU 0-d tensor captures.** Symbolic captures work as of `9ce495f6605`, by specializing
on the value. Device 0-d tensors work. CPU 0-d tensors are declined — Triton crashes on that
shape, so declining is not obviously the worse behaviour, and closing it properly means the
`aux_scalars` threading that `9ce495f6605` deliberately did not build.

**8. `AUTO` selection — closed by decision, not pending.** A natural-log-LSE backend is
structurally ineligible: `stats_are_log2` is decided in the eager wrapper from the literal
`BACKEND` string at Dynamo trace time, before Inductor picks anything. Parity gates `FLASH`
the same way and never lets `AUTO` select it, so leaving `FLYDSL` refused keeps the two
aligned. Documented above, not forgotten.

## Commits

| | |
| --- | --- |
| `af1b9ac0fb3` | FlyDSL draft work |
| `6af95befd5e` | FlyDSL parity work updates |
| `c68a3c31a51` | Serve gfx950 natively via a capability registry |
| `4956ef55a62` | Compare the flex path against AMD's two upstream gfx950 PRs |
| `17b37392807` | Stop asserting fast-math flags the kernels contradict |
| `e7a78aa2fab` | Answer the `BACKEND=AUTO` question, both halves |
| `a2ee8a89993` | Serve `qk_head_dim != v_head_dim` in the forward |
| `d123eac7d3c` | Frame the AUTO answer as the parity design, not our gap |
| `366c3334185` | Close three gaps found reviewing the previous three commits |
| `55a4d36dc6d` | Converge shared codegen with the parity branch |
| `91765762509` | Validate the flex kernels on FlyDSL 0.3.2 |
| `7f688105e51` | Add this branch overview |
| `9ce495f6605` | Serve symbolic scalar captures, and stop refusing unproven kv pairs |
| `158ac9c0685` | Take MFMA K=16 in the backward on CDNA4 |
| `2338af8cfa6` | Price the decode gap instead of calling it nothing |
| `2762d65cd8d` | Rewrite "what is not done" as an ordered account of what is left |
| `c3f9806b140` | Ground the RDNA4 assessment in the donor's actual layout |
| `1e8ce52f31e` | Serve `qk_head_dim != v_head_dim` in the backward |

## Further reading

| | |
| --- | --- |
| [`README.md`](README.md) | The backend's own documentation: architecture, supported configurations, how a `score_mod` becomes a kernel, autotuning costs, limitations |
| [`UPSTREAM_PR_COMPARISON.md`](UPSTREAM_PR_COMPARISON.md) | The full comparison against the two reference PRs |
| [`MULTI_ARCH_ROADMAP.md`](MULTI_ARCH_ROADMAP.md) | Per-architecture reach, and what the donor tree is worth for gfx950 / gfx1201 |
| [`PARITY_MIGRATION_PLAN.md`](PARITY_MIGRATION_PLAN.md) | How this sits against the parity branch |

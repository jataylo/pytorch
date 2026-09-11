# Our FlyDSL flex path against AMD's two upstream PRs

Two PRs are open against `pytorch:main` doing what this backend does. This file records how
they differ from ours, what each side is better at, and the four things in them worth taking.
It is a comparison, not a merge plan.

| | [#193854](https://github.com/pytorch/pytorch/pull/193854) | [#194309](https://github.com/pytorch/pytorch/pull/194309) | ours |
|---|---|---|---|
| Direction | backward only | forward only | both |
| Author | `lizamd`, 9 of 11 commits by `jiacao-amd` | `jiacao-amd` | this branch |
| Head | `a4b7984` | `aa3922d` | — |
| Size | +4006 / −4, 16 files | +3712 / −32, 11 files | — |
| Review | **0** review comments | **57** review comments | — |
| CI | none: `author_association: NONE`, workflows awaiting approval | same | 157 tests on gfx942 |
| Mergeable | `dirty` | `dirty` | — |

They are **siblings, not a stack**: both branched off `main` independently from the same
fork, four days apart, with no `ghstack-source-id` and no ordering dependency. Both sit on
already-merged FlyDSL template infrastructure (`b2ba1d6124c`, #192877), which is also what we
build on, so all three agree at that layer.

They **cannot both merge as-is.** Both add `flex_flydsl_attention.py` with entirely different
contents (494 vs 659 lines), both add `flex_attn_utils.py` with *incompatible APIs behind the
same function names* (`make_global_view` differs in offset semantics; `_schedule_group` takes
a string `kind` in the forward and an integer `mask` in the backward), and both add the same
`kernels/__init__.py` with different export dicts. `flex_flydsl_mask.py` is byte-identical
across the two, so that one file would merge cleanly while being duplicated work.

---

## The one difference that matters most

**They accept only an identity `score_mod`.** Both gate on `is_trivial_score_graph` and
refuse anything else — `"supports identity score_mod only"` — and both reject captured
score buffers outright. Custom `scale` survives only because scale is a separate lowering
argument folded into a compile-time constant, not part of the score graph.

So despite the name, neither PR is a FlexAttention backend in the sense this one is. They are
fast dense FlashAttention reached *through* the FlexAttention API, with sparsity expressed
through `BlockMask` and everything else fixed at build time. That is a coherent product and
their decode numbers show it pays, but it means the two efforts are not competing for the
same slot, and it explains most of the rows in the table below.

Their `mask_mod` is the more interesting half, and it is a genuinely different design from
ours: rather than codegen, `lower_flydsl_mask_graph` walks the FX graph and emits a **tuple
of instruction tuples over a value stack** — a small bytecode program, specialised into the
kernel at FlyDSL build time. The opcode set is a fixed dict of 16 binary ops, one unary,
scalar constants, rank-0 `aten.full`, and `load_i32` for indexing a captured buffer.
Captures are capped at 4, must be **int32**, and the cap is hard-wired into the templates as
an unrolled `{% if MASK_BUFFER_COUNT >= 1..4 %}` ladder.

Two mask shapes then get pattern-matched back out of that bytecode and constant-folded:
`is_causal_mask_graph`, and a `_causal_window_size` that matches the exact 5-instruction
sliding-window program `(ge 2 3), (sub 2 3), (const_i32 W), (lt 5 6), (and 4 7)`. The
backward additionally hard-matches a batched causal-document program *literal*.

Worth being precise about the trade, because the bytecode is not simply worse than codegen.
It buys a stable, inspectable, easily-cached mask representation and a hard bound on what a
mask can cost. What it costs is that the opcode dict is the whole language: anything outside
it raises, the int32 restriction excludes a float bias or slope table, and recognising causal
requires pattern-matching an instruction sequence rather than reading the graph.

---

## What we do better

| | ours | theirs |
|---|---|---|
| `score_mod` | traced graph lowered into the kernel at the score site; 11 further mod ops expanded in the shim at bf16-baseline accuracy | **identity only**; captured score buffers refused |
| `score_mod` backward | joint graph at the `ds` site, so a soft-cap differentiates correctly | n/a — no non-identity mod to differentiate |
| Architectures | gfx942 **and** gfx950, off a capability table | gfx950 only, `arch.split(":")[0] == "gfx950"` exact match |
| Unknown arch | refused by naming the missing capability, with gfx1201 entered so its refusal is specific | refused by not being the one name |
| Head dims | 7: every multiple of 32 from 64 to 256 | 2 pairs: (128,128) and (192,128) |
| dtype | bf16 **and** f16 | bf16 only |
| Directions | one lowering, both directions, chosen together | split across two PRs; the missing half falls to Triton **silently**, with no diagnostic |
| Sequence lengths | ragged, and cross attention (`Sq != Sk`) in all three kernels | `Sk % 128 == 0`; prefill `Sq % 128 == 0`. #193854 is MHA-only with `Sq == Sk` |
| GQA | MHA/GQA/MQA both directions, no post-pass and no atomics (the group is the outer loop) | forward yes; **backward rejects GQA** |
| Captured buffers | 4 slots, f32, with the descriptor given the tensor's real size | 4 slots, **int32 only** |
| Autotuning | 5 swept axes, each earning its sweep by measurement; `mod_vec_size`, forward `BLOCK_M`, `dkdv` tile, KV staging on by default | one choice appended, then `configs = []` and `append_flex_attention_choices` skipped entirely |
| Eager frontend | `_Backend` and the natural-log LSE set only | adds `_create_dense_block_mask` and swaps it in when `BACKEND == "FLYDSL"`, because the kernels cannot consume the O(1) empty-mask sentinel |
| Measured against Triton | forward 1.09x geomean over 84 cells, forward+backward 1.28x, on hardware | forward 0.94–1.49x prefill on gfx950; backward unmeasured against Triton |

Three of those deserve a sentence each.

**The arch gate is the same instinct our old `_VALIDATED_ARCHS` had**, and it carries the
same two costs: it makes gfx942 unreachable by construction, and it gives an unknown
architecture no diagnosis. Ours went the other way for exactly this reason — see
[`arch_caps.py`](../../kernel/vendored_templates/flydsl/arch_caps.py).

**The silent fallback is the sharper problem of the two.** `BACKEND="FLYDSL"` on the
direction a given PR does not implement produces a Triton kernel with no warning. Ours raises
with the reason instead, and it has to: our forward writes LSE in natural log, so pairing it
with Triton's log2 backward would compute *wrong* gradients rather than slow ones.

**The frontend change drew three separate review objections** on #194309 — an eager-side API
encoding an Inductor template's requirement, an O(Sq/128 × Sk/128) allocation where the
sentinel was O(1), and a wasted q-side transpose. The author says it has moved into the
lowering; that is not in the pushed head.

---

## What they do better

| | theirs | ours |
|---|---|---|
| **Decode** | packed-GQA decode, `Sq ∈ {1,4,8}`, ≤256 packed query rows per KV head, pipelined KV double-buffering, and a `SPLIT_KV` mode splitting KV blocks across two worker waves for low-parallelism MHA decode. Reports **3.6–7.0x** on top-k-16 sparse decode | **no decode kernel**, but decode shapes are served correctly by the prefill kernel — 2.0–6.5x slower than Triton decode, and 3.6x *faster* than Triton prefill. See below |
| **Real gfx950 validation** | benchmarked on MI355X, ROCm 7.2.53211, FlyDSL 0.3.1, four shape families | **build and ISA only.** We have no gfx950 silicon; correctness there is unproven |
| gfx950 schedule | hand-written: owner-wave selection (1/2/4/8), a `waves_per_eu` occupancy hint, dual-wave staging | the generic builder's output, with CDNA4 instructions selected off capabilities |
| Asymmetric head dims | `(192, 128)` — `qk_head_dim != v_head_dim` is first-class, both directions | **any admitted pair, either order, in both directions** |
| Upstreaming | in the review queue with `albanD` / `drisspg` requested | a local branch |
| Backward CDNA4 | their backward is written for gfx950 throughout | ours takes `mfma_k16` in GEMM1 only; `lds_transpose_read`, `permlane_o_store` and `dma_to_lds_b128` are still forward-only |

The first two are the honest asymmetry. Decode is a whole feature we do not have and they
have tuned, and it is the shape where their sparse numbers are strongest — a decode step is
where a sparse mask has the most to give, because the dense work per token is tiny. And no
amount of ISA inspection substitutes for running the kernel: our gfx950 claim is *"every
admitted head dim lowers and the CDNA4 paths engage"*, which is strictly weaker than theirs.

### What "no decode kernel" actually costs

Worth stating precisely, because "deferred" reads as "broken" and it is not. `use_decode` is
reachable only from `BACKEND='TRITON_DECODE'` or from `AUTO`, so `BACKEND='FLYDSL'` never
routes to `flex_decoding` at all — a decode shape is served by the ordinary prefill kernel,
mods and all, and it is correct there (checked against eager at `Sq ∈ {1, 4}` with a
`score_mod`, GQA and MHA, rel err ≤ 8e-3).

bf16, D128, gfx942, microseconds per call:

| B, Hq, Hkv, Sq, Skv | Triton | Triton decode | FlyDSL | FlyDSL / decode |
|---|---|---|---|---|
| 8, 32, 8, 1, 4096 | 4059 | 228 | 1130 | 4.96x |
| 8, 32, 8, 1, 8192 | 8093 | 439 | 2227 | 5.08x |
| 8, 32, 32, 1, 8192 (MHA) | 8178 | 1101 | 2249 | 2.04x |
| 8, 32, 8, 4, 8192 | 8094 | 447 | 2227 | 4.98x |
| 8, 32, 8, 8, 8192 | 8091 | 596 | 2228 | 3.74x |
| 32, 32, 8, 1, 8192 | 26378 | 1106 | 7150 | 6.46x |

Two things fall out of that table. The FlyDSL column is *flat* at 2227 for `Sq` of 1, 4 and 8
while Triton decode rises from 439 to 596, and the autotuner's own log says why: it picks
`BLOCK_M=128`, so a single query row is padded into a 128-row tile and `Sq ≤ 128` all costs
the same. And FlyDSL is still 3.6x faster than Triton's *prefill* kernel on the same shape,
which is the part "reaches no architecture" obscured — the deficit is against a specialized
decode kernel, not against the general path.

So the missing work is a decode kernel, not decode plumbing: pack the GQA group into the M
tile so the row is not wasted, and split the KV axis for parallelism, which is what their
`SPLIT_KV` mode is for. The MHA row is the tell that the GQA packing is the larger half —
it is the one shape where the group cannot be packed, and it is also the one where our
deficit is smallest (2.04x rather than ~5x).

---

## Four things worth taking

1. ~~**The fast-math objection applies to us too.**~~ **Taken, and it was a live bug rather
   than a risk.** A reviewer flagged that their `fast_fp_math` plus `no-nans-fp-math` /
   `unsafe-fp-math` lets the optimizer discard exactly the behaviour the kernel depends on —
   the `-1.0e30` sentinel, the `final_sum > 0` guard, storing `-inf` into LSE — and that it
   working today is *"an LLVM-version-dependent guarantee for a correctness-critical path"*.
   We set the same flags. Auditing ours found the guarantee already broken: a fully-masked
   row's LSE was denormal garbage instead of `-inf`, because `nnan|ninf` let `log2(0)` fold
   away. The flag set is now the honest subset and there is a test with a negative control;
   see the README's fast-math section. **The reviewer was right, and about our kernel too.**
2. **Their `Sq != Sk` handling is a spec worth reading**, since the forward supports separate
   extents cleanly. Ours works, but it was a bug fix (we returned 0.97 and 1.57 relative error
   silently before it), so a second design to check against has value.
3. ~~**Asymmetric `qk_head_dim != v_head_dim`**~~ **Done, and now covers more of this than
   they do**: any admitted pair in either order, forward *and* backward, against their one
   `(192, 128)` pair. Their crossed loader masks were the reference for the forward. The
   backward went further than the forward had to, because there the two extents are four
   loop bounds rather than none — see the README.
4. **The precompile hook.** Both warm FlyDSL's disk cache through `FakeTensorMode` proxies
   under `FLYDSL_COMPILE_ONLY=1`. We have a precompile pool already, but note the version
   hazard a reviewer raised: the env var is `COMPILE_ONLY` rather than `FLYDSL_COMPILE_ONLY`
   in some 0.3.x builds, and `pip install flydsl` currently gives 0.3.2 while their gate
   demands `release[:2] == (0, 3)`.

Two of their bugs are **not** ours and it is worth knowing why. Their `_run_compiled`
duplicates a shared helper and drops `os.getpid()` from the cache key, so a fork after a warm
launch reuses a dispatcher bound to the parent's GPU context; and `configure_flydsl_cache_dir()`
is never called, so FlyDSL's disk cache lands outside Inductor's cache root. Both are
consequences of reimplementing the shared runtime shim rather than calling it.

---

## Where LSE puts us

All three use a compatible convention, but not the same one, and it decides what can be mixed.

Both PRs pass log2 (`output_stats_in_log2=True` / `lse_in_log2=True`), which is the standard
flex_attention HOP contract — `torch/_higher_order_ops/flex_attention.py` states logsumexp is
expected in log2 scale, and the public API rescales by `ln2` under
`stats_are_log2=BACKEND != "FLASH"`. So each of their halves interoperates with its Triton
counterpart, and with the other PR.

We are in `_NATURAL_LOG_LSE_BACKENDS` alongside FLASH, so our forward and backward read and
write natural log and pair only with each other. That is why our gate raises instead of
falling back. Their forward↔backward seam is untested in both PRs — #194309's mask tests
contain no `requires_grad` and no `.backward()` at all.

---

## A caveat on comparing against the code

Both authors' most recent replies (2026-09-09) describe fixes that are **not in the pushed
heads**: the dense-mask helper moved into the lowering, an early return through
`create_flydsl_flex_attention_kernel`, `_create_empty_block_mask` always used when
`block_mask is None`, and the typing import order restored. #194309's head is `aa3922d` from
2026-08-30 and contains none of them. Anything compared here should be compared against the
described-but-unpushed design as well as the code, or it will be unfair on points they have
already conceded.

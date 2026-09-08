# FlyDSL FlexAttention: gfx942 / gfx950 / gfx1201, forward and backward

The target is a FlyDSL flex path that is feature-complete with FlexAttention across three
architectures in both directions. This file is the plan for getting there. It supersedes
nothing: `PARITY_MIGRATION_PLAN.md` remains the record of how the gfx942 forward got built,
and its Phase 3 scoping is referenced here rather than repeated.

## First, a correction to the size estimate

An earlier summary put gfx1201 at "~12.2k lines" and gfx950 at "~17.3k". Those numbers are
real but they are raw `wc -l` across whole directory listings, and they are not the size of
the job. No single upstream file is anywhere near that; the largest in the entire parity
tree is 2,291 lines. Decomposed:

| | raw lines | files | largest file | code lines | code % |
|---|---:|---:|---:|---:|---:|
| gfx1201 forward | 6,201 | 8 | 2,291 | ~2,300 | 33–47% |
| gfx1201 backward | 5,988 | 9 | 1,451 | ~2,600 | 28–62% |
| gfx950 forward | 10,416 | 8 | 5,632 | ~7,000 | 36–83% |
| gfx950 backward | 6,881 | 6 | 2,328 | ~3,400 | 47–53% |

Three things inflate the raw counts:

1. **These files are 30–70% prose.** The gfx1201 forward body is 2,292 lines of which 1,086
   are code; `fmha_tuning_gfx1201.py` is 626 lines of which 178 are code, the rest being
   knob tables and the reasoning behind them. For comparison our own
   `flex_flash_generic.py` is 70% code, so upstream is roughly twice as documented as we are
   and the line counts are not comparable to ours at face value.

2. **Some of it is an alternative we would pick against.** gfx1201 ships both a split
   backward (`dq` + `dkdv`) and a fused one, and the fused variant is capped at head_dim 128
   by a register wall. Only its own interface and tests import it, so it drops cleanly:
   `fmha_bwd_fuse_gfx1201_kernel.py` plus its interface and tuning is 1,705 lines we do not
   need.

3. **Some of it is work we have already done.** The three `*_interface.py` host wrappers
   (728 lines) do the job `flex_interface.py` does — argument marshalling, launcher prep,
   tensor validation. They inform ours; they do not get vendored into it.

Netting those out, gfx1201 is **~9.8k raw lines, ~4.9k of code, across 12 files**. That is
still a large port, but it is a different claim from "12k lines" and the two kernel bodies
that carry the difficulty are 1,086 and 877 code lines respectively — the same order as the
gfx942 forward we already own.

**One caveat I got wrong initially.** `philox.py`, `fmha_abi_gfx1201.py` and
`gfx1201_standalone.py` look droppable — FlexAttention has no dropout, uses BlockMask rather
than varlen, and has its own import structure — but all three are imported unconditionally
at module top level by the kernel bodies. They come across (or get stubbed) whether or not
their features are compiled in. Together that is 1,208 raw / 660 code lines, and the philox
PRNG at least is self-contained and arch-agnostic.

### The number that actually matters

`flash_attn_utils.py` is 5,633 lines at 83% code density — 4,673 lines of dense shared
framework, and the single largest item in the whole plan. Our current gfx942 forward does
not use it at all; `flex_flash_generic.py` is self-contained. So vendoring the gfx950 parity
stack is not just adding a kernel, it is **adopting a framework we currently do not depend
on**, and every later parity body assumes it. That is decision D1 below, and it should be
made before any vendoring starts rather than discovered halfway through.

## Where we stand

Two of six architecture-by-direction cells work.

| | forward | backward |
|---|---|---|
| **gfx942** (CDNA3) | **working** — `flex_flash_generic.py`, autotuned | **working** — `flex_flash_bwd_generic.py`, dq + dkdv, autotuned |
| **gfx950** (CDNA4) | `flex_flash_950.py` vendored, never wired, never executed | upstream SDPA parity, not vendored — and further along than this plan assumed, see below |
| **gfx1201** (RDNA4) | upstream, not vendored; arch rejected by the gate | upstream SDPA parity, not vendored |

148 tests across both directions. The gfx942 forward covers score_mod, mask_mod, BlockMask
skipping, GQA, bf16/f16, head_dim every multiple of 32 from 64 to 256, `return_lse`, ragged
sequence lengths, four captured buffers, cross attention, and native addressing of both
BHSD and BSHD memory.
Against autotuned Triton over 84 cells it runs at **1.09x** geomean — 1.01x plain, 0.99x
score_mod, 1.28x causal — having been 0.78x before the LDS row padding and the Q-tile sweep
and 1.02x before KV register staging.

The backward covers the same set, including block skipping on both axes, with one exception:
it cannot produce a gradient *with respect to* a captured tensor, though a mod may read one.
Combined forward+backward against autotuned Triton it runs at **1.28x** geomean over 84
cells — 1.23x plain, 1.20x score_mod, 1.42x causal — from 0.97x before the padding and tile
work and 1.25x before `dq` register staging. The dense column was once the whole of the
remaining gap and was attributed to the unpipelined backward; that turned out to be wrong,
and it was bank conflicts on the head_dims whose granule count is not a power of two.

**The remaining deficit is now specifically head_dim 64 and 128**, and it is the same two
columns in both directions. Those are exactly the two head_dims whose granule count *is* a
power of two, so they already had a working XOR swizzle and gained nothing from the row
padding that lifted 96/160/192/224 — and they are the sizes Triton itself is most heavily
tuned for.

Re-measured cell by cell, the deficit is **0.88x on dense and 0.79–0.87x with a score_mod**,
uniform across head counts and unchanged by autograd. The 84-cell harness had reported
0.61–0.72x for the same cells; that was per-call harness overhead landing on us harder than
on Triton, and the autotuner's own timings for those kernels (0.657ms where the harness said
0.991ms) are the ones to believe. Two consequences: the forward geomean above is if anything
understated, and the loss is *not* concentrated at low head counts the way the harness
suggested — 2x32 is 0.84–0.85x too.

Three explanations were tested and none of them is it:

- **Wave quantisation.** The hypothesis was that 1x8x4096 at a 128-row tile is too few
  workgroups to fill the machine. It is not: this part has 80 CUs, so that shape is 3.2
  waves, and the small-grid cells are not underfilled.
- **A shorter Q tile.** `BLOCK_M=64` is dramatically worse everywhere — 30.7 against 51.3
  TFLOP/s at head_dim 64, 35.8 against 64.2 at 128 — so the 128/256 pair we offer is right.
- **A wider KV tile.** The `N128` path (KV tile 128, two subtiles) turns out to be
  *unreachable* in the flex build: it is selected on a `causal` flag the template always
  passes as `False`, causal being expressed as a block mask here instead. Forced on, it is a
  wash to a loss — 2x32x4096 head_dim 64 dense drops 9% — so it is dead code we are not
  missing, rather than a lever left unpulled.

So what remains at 64 and 128 is not a tile-shape choice, and the next forward work there is
scheduling inside the dense inner loop rather than another knob. The score_mod column is the
more tractable half: our dense-to-score_mod cost is steeper than Triton's (71.9 → 55.2
TFLOP/s against 87.5 → 78.7 at 4x8x4096 head_dim 128), which is a mod-injection cost rather
than a flash-attention one.

A fourth explanation was tested later and *was* it, for everything except these two dense
columns: the interface copied q/k/v and the output into BHSD whenever the caller's memory
was BSHD behind BHSD sizes, which is what a projection produces and what
`benchmarks/transformer/score_mod.py` hands over. The numbers on this page are unaffected —
that harness allocates BHSD directly, as does the test suite, which is why it went unseen —
but on the mod matrix it was 574 µs against a 461 µs kernel and it was the whole of the
sparse-mask forward deficit. See the layout entry in [`README.md`](README.md).

## Two pieces that are not ports

Most of this plan is moving known-good code between repos. Two items are not, and they are
the two most likely to be underestimated.

**The backward has no Inductor plumbing.** Vendoring upstream's backward kernels would not
give FlexAttention a backward pass. `flex_attention_backward` has no FlyDSL branch at all —
`_use_flex_flash_attention_backward` returns early unless `backend == "FLASH"` — so
`BACKEND=FLYDSL` silently lands on Triton today, which is what `test_falls_back_for_backward`
pins. What is missing is a backward template, choice generation, `dq`/`dk`/`dv` mutated-input
wiring, and the accumulation story for `dq`. None of it is arch-specific, all of it is
needed before any backward kernel can be reached, and it should be built once against
whichever kernel lands first.

**The mod codegen cannot differentiate a mod.** A flex backward needs the derivative of
`score_mod`, which Inductor supplies as a joint graph containing scatters. Our codegen
rejects exactly that:

```python
raise NotImplementedError("Scatter graphs are not supported for FlyDSL (backward only)")
```

So `flydsl_flex_kernel.py` needs a backward mode that can emit the joint graph, including
accumulation into captured-buffer gradients. This is fork-side work with no upstream donor,
because upstream parity has no user-defined mod at all — it has compile-time `BIAS_TYPE`,
`WINDOW` and `ENABLE_DROPOUT` flags instead. That last point generalises: **every parity
body we vendor needs the flex hooks added to it**, which is the existing plan's Phase 4 and
is per-body work, not once.

**And gfx942 backward has no donor either.** There is no `*bwd*gfx942*` anywhere in the
kernel repo. That cell is new kernel work.

## Why three architectures means three ports

| | gfx942 (CDNA3) | gfx950 (CDNA4) | gfx1201 (RDNA4) |
|---|---|---|---|
| matrix core | MFMA, K-step 8 | MFMA, K-step 16 | WMMA 16x16x16, wave32 |
| global to LDS | 4-byte dword, opt-in | 16-byte `BufferCopyLDS128b` | none; via registers |
| transpose | software | `ds_read_tr16_b64` | `global_load_tr_b128` |
| P in f32 through GEMM2 | yes | yes | impossible |
| paged attention | upstream yes | upstream yes | absent |

gfx1201 shares no staging primitive with either CDNA column, and RDNA4 WMMA has no
`F32xF32` form, so the `acc += dot(p, v)` idiom that works on CDNA has no equivalent — P
cannot stay f32 through the second GEMM. Little of the CDNA work carries over beyond the
flex-hook pattern. Conversely gfx942 and gfx950 are closer but not interchangeable: see
`PARITY_MIGRATION_PLAN.md` Phase 3 scoping for why running the gfx950 body on gfx942 means
reworking its K-pack feed for a 2-element granule and replacing its V transpose.

## Plan

Ordered so that each phase is validatable on hardware we have, for as long as that is
possible, and so the un-validatable work comes last.

### P1 — widen the gfx942 forward (validatable)

Close the fork-side feature gates. None of these are kernel-architecture work and every
later architecture inherits them.

| gap | status | note |
|---|---|---|
| head_dim 256 | **done** | was blocked twice over; see below |
| 11 of 14 rejected mod ops | **done** | expanded in the shim, all at bf16-baseline accuracy |
| head_dim 96 / 160 / 192 / 224 | **done** | load guard plus K swizzle off; correct, 25-55% off the power-of-two dims |
| `erfinv`, `lgamma`, `digamma` | won't do for now | long rational fits, implausible in a score_mod |
| fp32 q/k/v | **won't do** | doubles the tile (256 impossible, 128 at the limit) and 8x the MFMAs; Triton covers it |
| `qk_head_dim != v_head_dim` | open | upstream parity supports asymmetric `head_dim_v` |
| more than 4 captures | **done to 4** | slots widened 2 to 4; past that is signature width again |
| `BACKEND=AUTO` | open | must be asked for by name, so nobody gets it by default |

**Gate:** each gate removed comes with tests against eager, and the AUTO change comes with a
performance argument per shape class rather than being flipped on.

#### head_dim 256: two bugs stacked, and the first hid the second

The old allowlist comment said 256 "needs 66560 B of LDS against gfx942's 65536 B limit".
That was true when written and is no longer: the V swizzle removed `HEAD_DIM * 2` elements of
padding, which at head_dim 256 is exactly the 1024 B it overshot by. It now fits at exactly
65536 B — and returned **42% relative error**, because the LDS ceiling had been hiding a
second, unrelated bug.

Localising it: LSE was correct to every printed digit, so the QK feed and softmax were
fine, and the error was confined to `d >= 160` with d0–d128 matching the head_dim 128
baseline exactly. That points at the online-softmax rescale of the O accumulators. Chunk 0
is rescaled eagerly; the rest are deferred into the GEMM2 loop to hide the multiply behind
the MFMAs, and the schedule was keyed off the PV k-step:

```python
if const_expr(not USE_HW_TR and dc == 0 and pks < D_CHUNKS - 1):
    o_accs[pks + 1] = Vec(o_accs[pks + 1]) * corr_vec
```

`pks` runs to `PV_K_STEPS`, which is 4 on gfx942. head_dim 128 has `D_CHUNKS - 1 = 3`
chunks to rescale and fits; head_dim 256 has 7 and does not, so chunks 5, 6 and 7 — d160,
d192, d224 — were never scaled at all. Rekeying to the flat step index fixes it, because
`_steps` is dc-major so chunk `c` is first read at step `c * PV_K_STEPS >= c` and rescaling
chunk `si + 1` at step `si` is always in time. head_dim 64 and 128 are bit-unchanged.

head_dim 256 now matches head_dim 128's error to four digits (0.00421 against 0.00424),
across plain, score_mod and block-mask paths. It runs at 1 workgroup per CU by construction,
since 65536 B is the whole LDS, so it keeps the `waves-per-eu` warning.

#### The mod ops: 11 expanded, 3 declined

`erf`, `erfc`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `atan`, `atan2`, `asin` and `acos`
now expand in `flydsl_mod_runtime` over the exp2/log2/sqrt primitives that do lower,
following the precedent `tanh` and `sigmoid` had already set. `erf` uses Abramowitz & Stegun
7.1.26 (1.5e-7); `atan` uses Hastings' odd minimax on |x| <= 1 with the `|x| > 1` reflection
(~1e-5); the hyperbolics are closed forms folded onto |x| so the exponential cannot cancel.

Every one measures at 0.0042–0.0044 relative error, which is the plain-attention bf16
baseline — the expansions contribute nothing measurable, which is expected given a score is
bf16-rounded before it reaches the softmax anyway.

`erfinv`, `lgamma` and `digamma` stay rejected. They need multi-branch rational
approximations rather than a few terms, none is plausible in an attention score_mod, and
adding one later is transcribing coefficients next to `erf` rather than kernel work.

**A trap worth recording.** Ad-hoc harnesses that loop over mods on one `flex_attention`
call site hit Dynamo's recompile limit partway down the list and silently reuse an earlier
graph. It surfaces as a suspiciously exact `0.000000` relative error, not as a failure, and
it made six of these ops look verified when they had not run. The test suite already calls
`torch._dynamo.reset()` in `setUp` for this reason; anything outside it must too.

### P2 — backward plumbing plus a gfx942 backward kernel (validatable)

**The largest single risk in the plan.** It is new kernel work and `dq` accumulation across
KV tiles is where flash backward implementations usually go wrong.

#### Port or write? Write, and read the port

Upstream has no gfx942 backward and no portable backward at all: the only attention
backward kernels are `kernels/attention/parity/fmha_bwd_{dq,dkdv}_gfx950.py` (1741 and 2328
lines) and the gfx1201 trio. Neither gfx950 kernel depends on `flash_attn_utils.py`; they
depend on the parity framework instead — `fmha_common_gfx1201` (1657), `fmha_abi_gfx1201`
(620), `fmha_dualwave_gfx950` (1494), `fmha_traits_gfx950` (488), two tuning modules (670,
796), `gfx950_standalone` (57), `philox` (500), so 6576 lines of framework under 4069 lines
of kernel. That closure is where the "12k lines" figure actually belongs; it was attributed
to the gfx1201 forward earlier and it is the backward's.

Adopting it is decision D1 for the whole flex path, and it buys less than it looks:

- **Neither gfx950 backward has flex hooks.** No score_mod, no mask_mod, no captured-tensor
  reads. The flex-specific work — score_mod at the score site, the joint graph at the `ds`
  site, mask_mod, aux readers, `mod_key` cache discrimination — is novel either way, and it
  is the part most likely to be wrong. Our forward already has all of it working.
- **The arch deltas are the ones our forward already solved:** MFMA K-step 8 rather than 16,
  software transpose rather than `ds_read_tr16_b64`, dword DMA rather than 16-byte, plus the
  swizzles and a 65536 B LDS budget.
- **It carries features we do not want to own or validate:** dropout with Philox, bias with
  its `db` output, the varlen-bits ABI, paged-attention rejection paths.

So: build the kernel on `flex_flash_generic.py`'s scaffolding, and use the gfx950 backward
as a *reference* to read rather than a dependency to import.

#### The forward already has both GEMM shapes the backward needs

This is the finding that decides it. Our forward does not compute `s = q·kᵀ`; it computes
**`GEMM1: k·qᵀ`**, so `s` and `p` already live transposed in MFMA32 register layout, and
then **`GEMM2: vᵀ·p`**, taking `p` from registers with no LDS round trip. Every product the
backward needs is one of those two shapes with different operands:

| product | shape | as |
|---|---|---|
| `sᵀ = k·qᵀ` | GEMM1 | unchanged |
| `dpᵀ = v·doᵀ` | GEMM1 | `q → do`, `k → v` |
| `dvᵀ = doᵀ·p` | GEMM2 | `v → do` |
| `dkᵀ = qᵀ·ds` | GEMM2 | `v → q` |
| `dqᵀ = kᵀ·dsᵀ` | GEMM2 | `v → k` |

So `dq` is two GEMM1-shapes plus one GEMM2-shape, and `dkdv` is two plus two, sharing the
`sᵀ` and `dpᵀ` prologue. Nothing needs a register-fragment transpose, which is the usual
reason a flash backward is harder than its forward — the forward having chosen the
transposed score domain is what buys that. It also lowers what the gfx950 backward is worth
as a reference: the transposed-GEMM geometry was the one thing it had that we did not, and
we do have it.

The two directions still want opposite loop nesting — `dq` accumulates over `n`, `dk`/`dv`
over `m` — so they stay two kernels sharing helpers, not one fused pass.

#### Plumbing: follow FLASH, not Triton

The Triton backward is one fused kernel split by `program_id`, with `dk` produced through
`store_output` and capture grads accumulated by `tl.atomic_add`. The CuteDSL FLASH backward
is the closer precedent and much simpler: preallocate all three grads, `mutated_inputs=[
grad_key, grad_value]`, `dq` as the template's output layout. Ours writes outputs in place
through a launcher, exactly like our forward, so FLASH's shape is the one to copy.

Two things fall out of that reading and remove work from v1:

- **Captured-buffer grads are out of scope for v1,** with precedent: FLASH's backward
  rejects them outright ("NYI: Flex Flash Attention bwd doesn't support captured grads
  yet"). That means subgraph 3, `zeros_and_scatter`, and the `atomic_add` store path are all
  deferred — which is exactly the `NotImplementedError("Scatter graphs are not supported for
  FlyDSL (backward only)")` already sitting in `flydsl_flex_kernel.modification`. Captures
  that do *not* need grads still work.
- **LSE stays natural log.** `_NATURAL_LOG_LSE_BACKENDS` already holds both FLASH and
  FLYDSL, so the backward reads the base its own forward wrote and uses `exp`; Triton's
  `1/log(2)` adjustment on `grad_logsumexp` is a Triton detail we do not inherit.

#### Order

1. **P2a — plumbing against a naive kernel.** Gate (`_use_flydsl_flash_attention_backward`,
   raising rather than falling through on explicit `BACKEND=FLYDSL`), `create_..._backward_
   kernel` mirroring FLASH, the jinja template, grad allocation, GQA batch reduction, choice
   generation. Validated against a deliberately simple kernel so plumbing bugs and kernel
   bugs cannot hide in each other.
2. **P2b — the kernel.** Delta preprocess (`rowsum(do * o)`), then `dkdv` (KV tiles outer),
   then `dq`. `dq` accumulation last and on its own, since that is the known failure point.
3. **P2c — mod derivatives.** Subgraph 0 at the score site and subgraph 1 (the joint graph,
   6 placeholders: `score, b, h, m, n, grad_score_mod`) at the `ds` site, then mask_mod.
   *Done; the two sites and the joint-graph hazard are written up below.*
4. **P2d — performance.** Only once gradients match eager. Reordered after measuring
   causal — block skipping first, then the 192 cliff, then pipelining. See below.

#### What P2b actually cost

Both kernels live in `flex_flash_bwd_generic.py` and both were numerically right on the
first run — `dq` across 22 shapes, `dkdv` across 19: head_dim 64 through 256, bf16 and f16,
MHA/GQA/MQA, ragged sequence lengths, 1 to 8 waves, every tile shape offered. That is worth
recording because P2 was budgeted as the largest single risk in the plan, and the reason it
was cheap is the GEMM-shape finding above: the fragment layouts line up so exactly that
several further consequences fell out that the table did not predict.

- **The per-row quantities are scalars, not vectors.** In a `C[n, m]` accumulator
  `m = lane % 32` is constant per lane, so `lse[m]` and `delta[m]` are one register each
  and the elementwise middle is one `fma` + `exp2` + two multiplies per element. No
  cross-lane reduction anywhere in the kernel — which is also why nothing in `dq` can be
  poisoned by a bad row, and why an out-of-range Q row needs no predicate beyond the
  `num_records` bound that drops its store.
- **`dsᵀ` feeds GEMM2 with no repacking.** The accumulator's 16 slots walk
  `n = (lane//32)*4 + (s//4)*8 + s%4`, so slots `[4p:4p+4]` are 4 *consecutive* `n` — the
  exact k-contiguity an operand pack wants. `p` in the forward is already fed back this
  way; `dsᵀ` inherits it for free.
- **The backward also needs no online softmax.** The forward already wrote the final
  `lse`, so `p` is recovered directly rather than rebuilt with a running max, and there is
  no rescaling, no correction factor and no deferred-rescale trap (the one that made
  head_dim 256 wrong in P1). The backward's elementwise middle is strictly simpler than
  the forward's.

`dkdv` followed and was also right on the first run, across 19 shapes. It is the mirror
image — a KV tile resident, the Q axis walked — and needed exactly one thing `dq` did not:
its accumulator is `C[m, n]`, so `n` is the per-lane constant and `lse[m]`/`delta[m]`
spread across the 16 slots instead of being scalars. Still cheap (4 groups of 4 consecutive
`m`, so 4 vector loads) but not free. The consequence that *does* matter is that `m` is
summed over there, so an out-of-range Q row has to be masked to `p = 0`; in `dq` such a row
is harmless because nothing reduces across rows and its store is dropped anyway. Two other
things fell out for free:

- **`dk` and `dv` share one kernel** rather than being two. They share the whole
  `s`/`p`/`dp`/`ds` prologue and differ only in the last GEMM's A operand — `doᵀ` for `dv`,
  `qᵀ` for `dk` — so both come out of one pass over Q.
- **GQA needs no post-pass and no atomics.** `dk`/`dv` for a KV head are the sum over the Q
  heads sharing it, so the group is a compile-time-unrolled loop around the Q walk that
  accumulates into the same registers. One workgroup owns each `(batch, kv_head, kv_tile)`
  outright. The `repeat_interleave`-then-`sum` the reference had is gone.

#### The tile, which turned out to be the whole performance story

GEMM2 wants its A operand transposed relative to GEMM1's, so the reduction-axis tensors are
staged into LDS **twice**: `dq` stages K in both orientations plus V (three tiles), `dkdv`
stages Q and DO in both (four). That is more LDS than the forward needs, and it is what
decides occupancy — which turned out to matter more than anything else measured here.

`dq` first shipped with the forward's `BLOCK_N` of 64 and ran at 65% of the forward's
per-FLOP rate at head_dim 128 while matching it at 64. Halving the KV tile to 32 halves the
footprint from 48 KB to 24 KB, buys the second resident workgroup, and is worth:

| shape | `BLOCK_N` 64 | `BLOCK_N` 32 | forward, same shape |
|---|---|---|---|
| B4 H8 S2048 D64 | 48.8 TFLOP/s | 55.7 (+14%) | 62.2 |
| B1 H8 S4096 D128 | 37.6 | 56.8 (+51%) | — |
| B4 H8 S2048 D128 | 42.3 | 67.1 (+59%) | 58.0 |
| B2 H32 S4096 D128 | 48.5 | 84.0 (+73%) | 76.4 |

It never lost; the one shape preferring 64 was running under 2 TFLOP/s either way. At
head_dim 128 the `dq` kernel now *exceeds* the forward's per-FLOP rate, still with none of
the forward's pipelining (no DMA-to-LDS, no multi-buffer prefetch, two barriers per tile).
It also made the whole head_dim ladder fit — 160 and up had wanted more than 65536 B — so
the `_MAX_BACKWARD_HEAD_DIM` cap that existed for a few hours is gone and the backward
covers exactly what the forward does.

**This is where autotuning does and does not pay,** which is worth stating as a rule rather
than a result:

- `dq`'s `BLOCK_M` — swept 32/64/128/256: 128 won 6 of 7 shapes and the losses were large
  and one-sided (42 vs 24 at 64, vs 26 at 256). Nothing to pick. **Fixed, no knob.**
- The forward's `BLOCK_M` — the inherited `256 if num_heads >= 32 else 128` heuristic looks
  arbitrary, and is: it never sees the grid-size-against-CU-count tradeoff it stands in
  for. But swept, 128 won 6 of 7 and the heuristic left at most 0.8%. **Not worth a knob**,
  which is the useful thing to know about it.
- `dq`'s `BLOCK_N` — a 51–73% *default* fix, not a tuning axis: 32 wins everywhere.
  **Fixed default, no knob.**
- `dkdv`'s tile — the one real spread. `(kv 128, q 32)` won 4 of 6 shapes but sat 9% behind
  a 32-head shape's preferred `(256, 32)` and 21% behind a short shape's `(128, 64)`.
  **Autotuned, on by default** (`flydsl.autotune_backward_tile`), three builds; measured
  4–14% in the Inductor autotune logs. `kernel_options={"DKDV_TILE": (kv, q)}` pins it.

The general shape: knobs that move the LDS footprint move occupancy and so matter a lot,
but usually have one right answer, and belong in the default. Knobs that trade one kind of
parallelism for another are the ones with genuine per-shape spread, and those are worth
building.

#### Against Triton, and what P2d is for

The backward is behind Triton where the forward is well ahead, and that flips the combined
picture. Forward **and** backward, bf16, both backends given `max_autotune`, every cell's
gradients checked against eager before being tabulated (`bench/flydsl_vs_triton_autotune.py
--backward`), as `fly/tri` on TFLOP/s:

| head_dim | 1x8x4096 | 2x16x2048 | 4x8x4096 | 2x32x4096 |
|---|---|---|---|---|
| 64 | 0.68 | 0.80 | 0.90 | 0.84 |
| 96 | 0.84 | 0.85 | 0.95 | 1.00 |
| 128 | 0.80 | 0.78 | 0.90 | 0.90 |
| 160 | **1.26** | **1.36** | **1.49** | **1.52** |
| 192 | 0.68 | 0.75 | 0.80 | 0.76 |
| 224 | 0.88 | 0.95 | 1.01 | 0.87 |
| 256 | 0.87 | 1.00 | 1.09 | 0.86 |

Read this as the cost of an unpipelined backward diluting a pipelined forward: the forward
alone beats Triton comfortably, and combined we give most of that back. Both backward
kernels have none of the machinery the forward earns its win from — no DMA-to-LDS prefetch,
no multi-buffer LDS rotation, no `sched_group_barrier` hints, two barriers per tile. That is
P2d, and it is a known quantity rather than a mystery: the same machinery, applied to two
more kernels.

Two things in the table are worth separating from that general story:

- **head_dim 160 is a win, and not ours.** Triton collapses to 23–25 TFLOP/s there while we
  hold 29–39. We are slower in absolute terms at 160 than at 128, as expected with the K
  swizzle off; Triton is just much worse. Do not read 1.5x as a P2b achievement.
- **head_dim 192 is our worst column.** 19–24 TFLOP/s against aten's 41–46, materially
  worse than 160 or 224, which have the swizzle off too. So the swizzle is not the whole
  story and something specific to 192 — 12 granules, 6 D-chunks — is costing more than the
  pattern predicts.

  **Resolved in P2d, and it was the swizzle after all** — just not in the way the column
  suggested. Forcing the swizzle off at the head_dims that *have* it costs 46–50% (64:
  70.6→38.0 TFLOP/s, 128: 80.6→41.7, 256: 66.1→33.1), which lands them squarely in the
  32–42 band 192 was sitting in. So 192 was never anomalous; it was one of four head_dims
  that cannot express the XOR swizzle, and the reason it was the *worst* of the four is
  that its row stride is 384 B = 96 dwords, an exact multiple of the 32-bank rotation, so
  every row starts in the same bank. 96 / 160 / 224 land 16 banks apart and take a milder
  2-way conflict. Padding the row by 8 elements on exactly those head_dims fixes all four
  at once — see `K_PAD` in `flex_flash_generic.py` for the bank arithmetic. The lesson for
  the table above is that the "192 cliff" framing hid the real finding, which was worth
  ~1.4x on three more columns.

**Gate:** `BACKEND=FLYDSL` with `requires_grad=True` produces FlyDSL gradients rather than
falling back, matching eager across the full head_dim ladder, MHA/GQA/MQA and ragged
sequence lengths. It refuses — loudly, never silently — captured-buffer gradients, because
falling back would pair our natural-log LSE with Triton's log2 backward and compute wrong
gradients rather than slow ones.

#### P2c — the mods, which cost two call sites rather than one

The forward evaluates a `score_mod` at one place. The backward needs two, and the second is
the one worth being careful about:

1. **The score site.** The score has to be recomputed *through* the mod, because `lse` was
   written against the modified score. This is the forward's site verbatim — same callable,
   same signature, same score domain (`q·k * sm_scale`), same ordering of `score_mod` before
   `mask_mod` for the same reason (a soft-cap applied after a mask maps `-inf` to `-cap` and
   resurrects the position).
2. **The `ds` site.** The gradient has to be carried back through the mod by the *joint
   graph* — Inductor's AOT-traced chain rule, called as
   `joint(pre_mod_score, b, h, m, n, grad_post_mod) -> grad_pre_mod`. This has no forward
   equivalent, and skipping it is the failure mode the whole feature has to be built
   against: gradients would come back wrong by exactly a factor of the mod's derivative,
   which for an additive mod like ALiBi is 1 — so the cheapest tests pass — and for a
   soft-cap is not. Both builders therefore *refuse* a `score_mod` handed to them without a
   `joint_mod` rather than defaulting it to a passthrough, and the test that matters is the
   soft-cap one, whose negative control (joint replaced by the identity) moves dq/dk error
   from 0.4% to 4.5% while leaving dv untouched — dv depending on `p` but not on the chain
   rule, which is a useful internal check that the sites are wired where they look wired.

`mask_mod` needs no second site: a masked element gets `p = 0` and the joint graph is linear
in its cotangent, so zero in gives zero out, and both gradients fall out of the score site
alone.

Captured tensors reach the backward through the forward's aux slots unchanged — same reader
ABI, same `(stride_b, stride_h, stride_q, stride_kv)` specs — so a mod body Inductor lowers
once runs at all three call sites. What is still refused is a gradient *with respect to* a
capture, which needs the joint graph's `zeros_and_scatter` outputs accumulated with atomics.

One asymmetry worth recording, because it looks like an oversight: `mod_vec_size` applies to
`dq` but not to `dk`/`dv`. The vectorized mod ABI is defined over *contiguous `kv_idx`*, and
`dq`'s score fragment has the forward's layout (`m` per lane, `n` across slots in groups of
4) so it qualifies. `dk`/`dv` is the mirror — `kv_idx` is the per-lane constant and `q_idx`
is what runs contiguously — so it does not, and its mods are always called one element at a
time. The backward therefore does not sweep the width at all: the reachable win is at most
half the forward's measured 0.6–2.1%, against multiplying the tile sweep by three.

#### P2d block skipping — done, and it took the causal deficit to a surplus

Both backward kernels used to walk their axis densely and apply `mask_mod` per element,
consulting no block mask, so a causal workload got the forward's saving and none of the
backward's. Forward+backward, bf16, both backends on `max_autotune`, as `fly/tri` on
TFLOP/s (>1 means we win):

| shape | D | dense | causal before | causal after |
|---|---|---|---|---|
| 1x8x4096 | 64 | 0.73 | 0.68 | 0.78 |
| 2x16x2048 | 64 | 0.81 | 0.79 | 0.98 |
| 4x8x4096 | 64 | 0.89 | 0.70 | **1.07** |
| 2x32x4096 | 64 | 0.85 | 0.67 | **1.08** |
| 1x8x4096 | 128 | 0.80 | 0.66 | 0.97 |
| 2x16x2048 | 128 | 0.79 | 0.71 | **1.08** |
| 4x8x4096 | 128 | 0.90 | 0.68 | **1.09** |
| 2x32x4096 | 128 | 0.94 | 0.65 | **1.13** |

Causal went from 0.65–0.79 to 0.78–1.13, ahead of Triton on six of eight rows and ahead of
aten on seven. The dense column is untouched, as it must be — no mask, nothing to skip —
and remains the pipelining gap.

**What it is.** `dq` walks a KV list per Q tile, exactly as the forward does. `dk`/`dv`
walks the transpose, a Q list per KV tile, because it reduces over the other axis. Both
lists come from *one* occupancy, transposed, rather than from FlexAttention's own
`q_indices`: those are on its sparse grid and would need the same conversion anyway, and one
occupancy is what guarantees the two kernels agree about which blocks exist. Disagreeing
there would drop gradient contributions on one side only — a wrong `dk`/`dv` with a right
`dq`, which does not read as a transpose bug.

Per-Q-head lists cost nothing in `dk`/`dv` even though it owns a *KV* head, because the GQA
group is already the outer loop: each Q head walks its own list into shared accumulators. A
union over the group would have been the fallback.

**Three things that were not obvious.**

1. **The payoff tracks workgroups per CU, not sequence length.** Causal work per tile is
   triangular, so one workgroup walks every tile of its axis and another walks one. Below
   ~2 workgroups per CU they are all resident, the kernel finishes when the *heaviest* one
   does, and skipping the light ones moves nothing while still paying the index loads.
   Measured on 80 CUs: 0.89–1.18x under the threshold, 1.54–2.39x over it, settling near
   1.7x for `dq` and 2.0x for `dk`/`dv`. Hence the gate.

2. **The gate must not depend on the tile being swept.** It first did, since each kernel's
   grid follows its own tile — and that made autotuning choose against the optimization.
   Choices are benchmarked against *dense* fake lists (the counts must address real blocks,
   so they are generated, and generated dense), so the sweep cannot see sparsity: it picked
   the widest KV tile on dense merits, and that was the tile whose grid fell under the
   threshold, so it silently took the choice with skipping *off*. 1x8x4096 head_dim 64 ran
   its causal backward at full dense speed for exactly this reason. The decision is now one
   tile-independent answer shared by both kernels and every choice.

3. **The regrid memo held one entry per mask.** Fine while only the forward asked; with the
   two backward walks there are three grids per BlockMask, and a single slot made them evict
   each other so every step paid a full rebuild. Invisible in the answers — the lists are
   correct either way — and it cost 5–34% of the causal wall on its own. Now keyed by grid.

4. **The K row stride was bank-aligned wherever the XOR swizzle was off.** head_dim 96 /
   160 / 192 / 224 have a granule count that is not a power of two, so the swizzle degrades
   to identity and nothing was taking its place. Padding those rows by 8 elements — 16 B, 4
   banks, exactly one lane's read width — is worth 1.4x at 96 / 160 / 224 and 2.6x at 192,
   whose stride was an exact multiple of the bank rotation and so had every row landing in
   one bank. Applied to all three kernels; the two backward ones have no DMA path at all,
   so the padding is unconditional there.

5. **The forward's Q tile was a heuristic, not a measurement.** `BLOCK_M = 256 if
   num_heads >= 32` was never offered to the autotuner. Measured, the taller tile wins at
   exactly one head_dim (160) and loses at the other six, by 19% at 64 and 2.5x at 256. It
   is now a swept choice at 32 heads and up, which is the only place the two candidates
   differ.

**Where that leaves us.** Forward 0.78 → **1.02** geomean against Triton over 84 cells,
backward 0.97 → **1.25**, with zero forward regressions and five backward cells 3–10% down
(all at head_dim 64/128, which are unpadded, so run-to-run rather than caused).

**P2d, resolved for `dq` and closed for `dkdv`.** The original justification — "the whole
of the dense 6–27%" — evaporated once the padding landed, so this stopped being a deficit
to close and became a lead to extend. It was then done anyway, in the form gfx942 can
actually support: not the gfx950 donor's DMA into a second LDS buffer (no instruction for
it, no LDS to spare) but the next tile held in registers. `dq` took it and won at every
head_dim, 3.8–24.4%, at a cost of 329 → 363 VGPRs of 512 with no spilling.

`dkdv` was left out on a budget argument: it sits at 452 of 512 VGPRs, so the same carry
leaves roughly a 5% margin before spilling, and its staging is inside the GQA group loop
where a next-tile lookahead crosses into the next Q head at each boundary. That was called
a coin-flip only measurement could settle.

**Now settled, and the coin landed on its edge.** Built — the lookahead clamped two-sidedly
per Q head and the pipeline re-primed at each head — it splits by head_dim rather than
winning or losing outright: 1.01–1.10x at 160/192/224, 0.84–0.94x at 64/96/128/256, over two
shapes and both dense and causal, all four measurements per head_dim agreeing in sign. Below
160 the carried tile costs more than the overlap buys; at 256 the kernel already holds the
whole LDS at one workgroup per CU. So it ships as a default keyed on that band, not as a
sweep — which matters because a sweep here would have doubled the backward's autotune bill,
and we are 5–7x more expensive per choice than Triton already (see the README). Numerics are
bit-identical either way. P2d is now closed in both kernels.

### P3 — decide the framework question, then wire gfx950 forward (not validatable)

D1 below has to be answered first. Then either wire the already-vendored
`flex_flash_950.py` (the template currently hardcodes the generic builder) or vendor the
parity forward per Phase 3 scoping.

**Gate:** compiles and emits gfx950 ISA, stays behind `config.flydsl.allow_unvalidated_arch`,
and the file header records that it has never been executed. It cannot be stood in for on
gfx942 — the 16-byte DMA makes LLVM fail instruction selection with *"Do not know how to
expand this operator's operand"*.

### P4 — gfx950 backward (not validatable)

Reuses P2's plumbing. ~3.4k code lines plus per-body flex hooks. Same gate as P3.

### P5 — gfx1201 forward, then backward (not validatable)

~4.9k code lines across 12 files after dropping the fused backward and the host interfaces.
A different ISA throughout; budget for the f32-through-GEMM2 problem being a redesign of the
softmax-to-PV handoff rather than a substitution.

**Gate:** same as P3. Note gfx1201 has no paged-attention path upstream, so that feature
gap is permanent there rather than pending.

## Deferred: features no architecture currently reaches

Paged attention, nested/jagged tensors, and `flex_decoding` (the split-K decode path) are
not integrated for any architecture and are not sequenced above. Upstream has paged for
gfx942 and gfx950 but not gfx1201. These want their own scoping once at least one backward
exists, because decode and paged interact with the KV staging that P3–P5 are rewriting.

## What the upstream SDPA parity work actually has (donor survey)

Surveyed `xinyazhang/sdpa-gfx950-feature-bwd`, which is well past what the table above
assumed. It has a complete gfx950 backward — `dq` and `dk`/`dv`, two MFMA families,
head_dim 32 to 512, causal, windows, varlen, dropout, bias, GQA, fp16 and bf16 — plus a
gfx1201 backward including a fused variant, and roughly 500 KB of design docs.

It is SDPA/FMHA parity, so its features are build flags rather than `score_mod`/`mask_mod`
subgraphs, and its kernels are welded to `flash_attn_utils.py` ("imported, never edited.
Four production kernels import it"). That does not answer **D1** but it does sharpen it: the
files cannot be reused wholesale without adopting that framework, while the individual
techniques are each 200–400 lines embedded in 4,000+ lines of inherited helpers.

**It also found us a live bug, indirectly.** Its "mask the KV tail in window builds" fix is
about causal logic admitting columns past `seqlen_k`, which prompted the question of what we
do when Q and KV lengths differ at all. Answer: we accepted it and returned garbage. All
three kernels took one `seq_len`, the gate checked dtype/arch/head_dim but never the
sequence lengths, and cross attention came back at 0.97 and 1.57 relative error silently.

Cross attention is now implemented rather than refused: each kernel takes both extents, and
which of the two it uses is per-site, because the forward and `dq` tile Q and walk KV while
`dkdv` tiles KV and walks Q. Every grid, resident tile, `num_records` bound and padding mask
therefore takes a specific one of the pair, and the `mask_mod` grid is no longer square, so
the regrid rescales each axis against its own tile count. Equal lengths make all of those
sites indistinguishable, which is exactly why the bug was invisible and why the tests use
unequal lengths in both orders plus a ragged pair. The rest of the audit is below; this item
is the one that mattered.

Item by item against what is still open here:

- **Gradients w.r.t. a captured tensor — strong donor, for half the problem.** Its backward
  computes `dB` with **no atomics**: the `dq` grid is one workgroup per (batch, Q head, Q
  block), so each `(q,k)` element of a dense `[B,H,Sq,Skv]` bias has exactly one writer and
  a plain store suffices. Its contract mandates this outright. That covers the dense
  learnable-bias shape cheaply. It does *not* cover a broadcast capture such as an `[H]`
  ALiBi slope table, where every `(q,k)` sums into one element and atomics or a reduction
  pass return. **So that item splits into a cheap half and an expensive half.**
- **Asymmetric `qk_head_dim != v_head_dim` — strong donor as a spec.** First-class
  `hdim_qk`/`hdim_vo` with crossed loader masks, and a warning worth keeping: the two
  extents coincide in every symmetric build, so only a dedicated asymmetric test can tell
  the fix from its absence.
- **Backward block skipping — weak donor, and it did not end up mattering.** Its kernels do
  bound the loop, but from causal/window *arithmetic*, which works because causal is a build
  flag of known shape. Ours is a runtime BlockMask, so that math does not apply; the actual
  donor was our own forward.
- **head_dim 192 — no help, but it rules something out.** Its cliff is at 256 and above,
  from AGPR pressure at 32 rows per wave; 192 is healthy there at ~730–840 TFLOP/s. Its fix
  is a 16-row family on `v_mfma_f32_16x16x32`, which CDNA3 does not have. Our 192 problem is
  ours.
- **Pipelining — mixed.** Its `dq` is structurally identical to ours (one LDS buffer, two
  barriers, no prefetch), so no donor. Its `dk`/`dv` has real dual-buffer prefetch and is a
  reference for the Q/dO streaming side.

### The FA-core audit, and why most of it does not apply

Checked its recent core fixes against our kernels rather than assuming either way. Four of
its findings are real bug classes we simply do not have, and the reasons are structural
rather than luck — worth writing down so nobody re-derives them:

- **The lost-SGPR-copy miscompile** (LLVM `SplitKit` splitting a wide scalar tuple so a
  subregister is never defined, then using it as a buffer descriptor base — silent wrong
  addresses, verifiers clean). Its trigger is wide *runtime* stride tuples staged and
  spilled across regions; its own clean sibling object differed by rematerializing strides
  from the kernarg instead. Our kernels take **no runtime stride scalars at all** — just
  tensors and a single `seq_len`, with every stride a compile-time constant — so the
  precondition is largely absent. We are also on ROCm 7.2.2 / clang 22 rather than their
  24.0.0git. Their `tooling/llvm-lostcopy/mircfg.py` is the detector if this ever needs
  revisiting.
- **VALU→MFMA and `exp2` wait states** needing hand-written tied-operand `s_nop`s, because
  `GCNHazardRecognizer` does not model the gfx950 timings. It models ours:
  `checkMAIHazards` routes gfx942 through `checkMAIHazards90A`, and the extra latency is
  gated on the new part — `return NumPasses + 1 + IsGFX950`. CDNA4 needs one wait state
  more than CDNA3, which is precisely why they wrote nops and we do not.
- **The odd-`seqlen_k` bias tail**, where a tight buffer bound ending on the last valid
  element makes the hardware drop a final straddling dword, so `bias[..., sq-1, sk-1]`
  reads zero. Ours could not do that — but taking their lore's advice anyway (a
  whole-tensor norm cannot see a one-element defect, so probe the last element and assert
  it *moved*) found the mirror-image bug, below.
- **`GRID_AXIS_ORDER`**, their measured preference for head on the fast axis. All three of
  our kernels already do `block_id % NUM_HEADS`, arrived at independently.

**The mirror image of their bias bug was ours.** Their tight bound dropped a valid dword;
our `max_size=True` bound turned bounds checking *off* entirely. The mods run on the whole
score tile with the padding discarded afterwards, so the aux reader is called at q rows and
kv columns past `seq_len` — a full tile past, plus three more when `_read.vec` widens the
load — and at the last `(b, h)` those offsets leave the tensor. At head_dim 128 and
`seq_len 63` the reader addresses element `numel + 4098`, about 16 KB past a `[B, H, 63, 63]`
bias, because the last Q tile spans rows 0–127 while only 0–62 are valid.

It was a GPU memory access fault, not wrong numbers, and it hid the way those hide: whether
the overrun lands on mapped memory depends on what the caching allocator put next, so the
probe test passed run on its own and killed the whole file at test 32. `_check_aux` did not
catch it because it validates the *logical* last coordinate `(B-1, H-1, S-1, S-1)`, which is
exactly `numel - 1`; the kernel reads past that, not outside the strides.

Fixed by giving the descriptor the tensor's real size, which is a compile-time constant —
`record_aux_spec` already guards the capture's shape to ints, so the count comes free. The
hardware then returns 0 for the padding lanes, which are discarded regardless, and valid
lanes are untouched: every in-range offset is host-checked against `numel`, and aux is
always f32, so no dword holding live data can straddle the bound. That last part is why we
can take the tight bound their AOTriton fix had to work around. The size is now a *required*
companion to the strides in all three builders rather than a default, since the two
available defaults are the fault we just removed and a guess at the caller's tensor size.

Two more are safe only because of how Inductor compiles, which is worth stating as a
standing assumption rather than a fact: the `dk`/`dv` GQA loop bound and the LSE head
indexing are both **compile-time** here. Upstream got a 8.7e-01 relative error on the
first when an AOT build pinned `num_heads=1` and the runtime group was 4. We are safe
because `NUM_HEADS`/`NUM_KV_HEADS` are template defines and part of the autotune choice
key, so a shape change recompiles. Anything that ever serves one compiled kernel across
head counts breaks both.

### The second FA-core pass: what came across, and what measurement closed

A later audit against upstream's tip re-listed the flash-attention core gaps. Most of it
resolved on measurement rather than on judgement, so the outcomes are recorded here to stop
them being re-opened.

**Ported.** KV staging through registers, upstream's main gfx942 lever, is now in the
forward (swept) and in the `dq` backward (on by default, up to 24% at head_dim 192). See
the README section for the numbers and the register accounting. The gfx950 backward's
2-buffer LDS pipeline is *not* what landed: it stages by DMA, which needs an instruction
gfx942 lacks and a second LDS buffer this kernel has no room for. Carrying the tile in
registers is the same overlap within what the part can actually do.

**Closed by measurement, not adopted.**

- *Deeper QK prefetch (3–4 instead of 2).* Re-measured after the row-padding fix changed
  the bank behaviour underneath it: neutral to 5% worse at head_dim 96/128/192/224.
  Upstream's deeper default is *coupled* to its gpfetch path; depth 2 is what upstream
  itself uses without it, so our default already matched our configuration.
- *`waves_per_eu = 1` for the backward*, credited upstream with up to 4x where it stops
  spilling. Flat here — 0.999–1.000x across head_dims — because our `dq` does not spill
  (0 spill slots at 329 VGPRs). The upside was real but conditional, and the condition
  does not hold.
- *Runtime M128/M256 tile dispatch.* Superseded: we sweep the tile per shape instead of
  thresholding on `batch * seq * heads`, and the taller tile lost at every sub-32-head
  shape measured, so upstream's rule would pick wrong on our kernel.
- *`V_PERM_TR`.* Gated upstream on `vt_stride == block_n + 2` — the padded V layout we
  replaced with an XOR swizzle to reclaim LDS and hold 2 workgroups per CU. Taking it means
  giving that back.

**The MFMA operand wait state does not apply, and this was checked rather than argued.**
Upstream emits `s_nop 1` between a bf16 pack and the MFMA reading it, for documented
non-deterministic wrong answers. Its own docstring is explicit that the scope is not
established and that the general "VALU write then MFMA read" rule is wrong (~7500 such
sites across 426 mostly-passing kernels). The hazard is named on a `v_cvt_pk_*_f32` write
reaching an MFMA SrcA/SrcB; we pack bf16 *bitwise*, as `(hi & 0xFFFF0000) | (lo >> 16)`.
Dumping our backward ISA (`FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1`) and scanning for the
pair found **zero** `v_cvt_pk_*_f32` instructions in either kernel, in both bf16 and fp16 —
so there is nothing for the nop to protect. Note this is a property of our pack, not a
guarantee: anything that switches to a hardware convert re-opens the question.

**A prefetch lookahead needs a two-sided clamp, and this cost a benchmark run.** Carrying
the next tile means asking for it before knowing it exists: the last iteration reaches one
tile past the walk, and a Q block with an empty walk reaches past it before starting. Under
a block mask that indexes the tile list out of range, and the value read back is *not*
inert — `_kv_row_clamp` bounds only the top, so a stale negative int32 becomes a negative
row and the load addresses off the front of the tensor.

The failure mode is the same one the aux-buffer bug had: whether the overrun lands on
mapped memory depends on what the allocator left in that slot, so it passed a 137-test
suite and then killed the forward benchmark with a GPU memory fault at head_dim 160 on a
`USE_BLOCK_MASK` cell. Fixed by clamping the prefetch's list index to the last live entry
*and* the resulting tile to a non-negative row, so a lookahead can only re-read a tile the
walk already covers — wasted bandwidth on one iteration and nothing worse. Both the forward
and `dq` needed it. The lesson generalises: any index that deliberately runs past a live
range must be clamped at both ends, because "the buffer read is bounded" only constrains
the *value*, not its sign.

**One upstream bug found in passing.** Upstream's generic path sets
`k_swz_rowmask = head_dim // 16 - 1` with no power-of-two check and `k_pad = 4` on gfx942
non-DMA. At head_dim 96 that mask is `0b101`, so `col ^ 80` maps columns 32–47 to 112–127 —
past the end of a 100-element row. Our disabled-XOR-plus-8-pad is not just faster here, it
avoids importing that. Do not "sync" this one.

## Decisions needed

**D1 — adopt `flash_attn_utils.py` or not.** Vendoring parity means depending on a
4,673-code-line shared framework that our working kernel does not use, and every parity body
assumes it. The alternative is taking parity's *ideas* into our self-contained body, which
already has the software transpose and the 8-element granule that CDNA3 needs. This decides
the shape of P3 through P5 and should not be settled implicitly by starting to copy files.

**D2 — what "feature-complete" excludes.** fp32 attention, paged, jagged and decode are each
a phase in their own right. If they are in scope the plan is materially longer than P1–P5;
if they are not, that should be written down.

**D3 — how much unvalidated code to carry.** P3 through P5 produce roughly 10k lines of
kernel that cannot be executed on this machine, gated behind a config flag. There is a real
question whether that is worth merging before hardware exists, or whether it should live on
a branch until it can be run.

# Adopting the FlyDSL `parity` FMHA family for PyTorch FlexAttention

Status: **Phases 0-2 done and measured; 3-7 not started.** Written 2026-08-26,
executed from 2026-08-28. Outcomes are recorded under each phase; §2's baseline is
the pre-Phase-1 measurement and is left as it was taken.

Goal as stated: replace our FlexAttention FlyDSL kernel base with the `parity`
work from `xinyazhang/sdpa-gfx950-feature-bwd`, plumb in gfx950 and gfx1201,
create a gfx942 implementation derived from the gfx950 one, develop the backward
passes, and re-benchmark.

This document records what that actually requires, the five blockers standing in
front of it, and a phased plan where every phase has a verification gate. It
follows the convention of the parity repo's own plan docs: measured facts are
cited, and each phase gets an outcome section appended when it runs.

---

## 1. Executive summary

Three findings should shape the decision before any code moves.

**The premise needs adjusting.** Our current FlyDSL flex kernel is not slow
because its base is old. Measured on gfx942 it runs at 0.42x–0.78x of *autotuned*
Triton FlexAttention, and the single largest component of that gap is our own
un-plumbed BlockMask KV-skip — a feature already implemented in the kernel we
have. Adopting parity does not fix it, because parity has no block-sparse
mechanism at all.

**A parity-derived gfx942 kernel is a rewrite, not a port.** Parity's dual-wave
schedule rests on two CDNA4 memory-path features that gfx942 does not have:
16-byte DMA-to-LDS (`buffer_load_dwordx4_lds`) and the LDS transpose read
(`ds_read_b64_tr_b16`). Our existing kernel already carries the CDNA3 fallbacks
for both. So "a gfx942 implementation based on gfx950" converges on something
very close to the kernel we already ship, reached the long way round.

**gfx950 and gfx1201 cannot be validated here.** All 8 GPUs are MI308X (gfx942).
Those two targets can be written and reviewed but not run, not tested, and not
benchmarked. The last step of the request has no execution path on this machine.

The recommendation embedded in the phasing below is therefore: fix the FlyDSL
build blocker first because it gates everything and is cheap in wall-clock; then
land the two measurable gfx942 wins that need no new kernel; and treat the
parity adoption as a separately-justified project whose first deliverable is a
*compiling, unvalidated* gfx950 path, not a gfx942 replacement.

---

## 2. Measured baseline

MI308X (gfx942, 80 CUs @ 1.42 GHz), bf16, head_dim 128, forward only.
Peak dense bf16 = 80 × 1024 × 1.42e9 = **116.3 TFLOP/s**.
Both flex backends given `max_autotune=True`; backend selection verified by
reading the generated code on every row; all outputs matched eager.

| shape (BxHxS) | variant | FlyDSL | Triton | aten SDPA | fly/tri |
|---|---|---|---|---|---|
| 1x8x4096 | plain | 34.8 TF | 67.4 TF | 81.9 TF | 0.52x |
| 1x8x4096 | score_mod | 32.7 TF | 60.3 TF | — | 0.54x |
| 1x8x4096 | causal | 16.2 TF | 38.6 TF | 51.6 TF | 0.42x |
| 2x16x2048 | plain | 37.1 TF | 73.9 TF | 80.5 TF | 0.50x |
| 2x16x2048 | score_mod | 35.9 TF | 66.4 TF | — | 0.54x |
| 2x16x2048 | causal | 18.1 TF | 36.9 TF | 53.0 TF | 0.49x |
| 4x8x4096 | plain | 48.9 TF | 87.5 TF | 94.9 TF | 0.56x |
| 4x8x4096 | score_mod | 46.5 TF | 78.5 TF | — | 0.59x |
| 4x8x4096 | causal | 23.5 TF | 54.0 TF | 66.6 TF | 0.43x |
| 2x32x4096 | plain | 68.9 TF | 89.8 TF | 102.6 TF | 0.77x |
| 2x32x4096 | score_mod | 62.9 TF | 80.8 TF | — | 0.78x |
| 2x32x4096 | causal | 33.5 TF | 58.3 TF | 69.8 TF | 0.58x |

Scripts: `/dockerx/bench/flydsl_flex_bench.py` (three-way, with backend
verification), `/dockerx/bench/flydsl_vs_triton_autotune.py` (autotuned
head-to-head).

### Three facts to carry forward

1. **Causal does no block skipping.** FlyDSL spends the same wall time on causal
   as on full attention: 8198 µs vs 7976 µs at 2x32x4096. Triton goes
   6121 → 4718 µs on the same input. The `kv_indices`-driven loop exists in our
   kernel (`flex_flash_generic.py:891-968`, exposed as `launcher.kv_block_size`)
   but is not driven from Inductor. Estimated prize: **~1.7x on causal.**
2. **Occupancy target is missed on every build.** The compiler emits
   `failed to meet occupancy target given by 'amdgpu-waves-per-eu' ... desired
   occupancy was 2, final occupancy is 1`. This is consistent with FlyDSL only
   becoming competitive at large grids (0.5x at 256–1024 workgroups, 0.77x at
   2048).
3. **The hook layer is not the problem.** An ALiBi score_mod costs 9–12% over
   plain attention. The programmability design is working; the base kernel and
   the missing sparsity are what cost us.

Caveats: MI308X is a small part, so the occupancy story may differ on MI300X
where grids are several times larger. Forward only. Single dtype and head_dim,
because 128 is the only value our kernel computes correctly.

---

## 3. Lineage: what we have versus what exists upstream

Three codebases, not two.

**Ours** — `torch/_inductor/kernel/vendored_templates/flydsl/flex_kernels/`.
A fork of the monolithic `kernels/attention/flash_attn_generic.py` taken around
2026-07-07..13 (upstream was 1624 lines then; ours is 1693 with our hooks added
and paged-KV / cross-attention / dual-wave routing stripped out).

**Upstream `kernels/attention/`** — same lineage, still maintained, and it has
moved past our fork in four relevant commits:

| commit | date | what |
|---|---|---|
| `d97c20f` | 07-10 | **Add D=64 support.** Replaces the hardcoded XOR swizzle mask with `K_SWZ_ROWMASK = HEAD_DIM // 16 - 1` |
| `4be5614` | 07-17 | Refactor into `flash_attn_utils.py` helper framework (generic drops 1624 → 476 lines) |
| `829c7b4` | 07-22 | **gfx942 perf pass**: K-prefetch pipelining, env-gated DMA |
| `cc141e8` | 07-26 | Native `return_lse` (fp32 `[batch, num_heads, seq_len]`) |

**`parity/`** — a new family, 2026-08-16..25, 104 commits, 23854 non-test Python
lines. Targets bit-level ABI equivalence with AOTriton `attn_fwd` so it can
replace the Triton kernel inside AOTriton. Covers gfx950 and gfx1201, forward
and backward.

### The D=64 bug is ours and the fix is two lines

Our `_k_swizzle` hardcodes the row mask:

```python
# flex_flash_generic.py:596-598
def _k_swizzle(row_idx, col_idx):
    mask = (row_idx & fx.Index(0x7)) << fx.Index(4)
```

With `HEAD_DIM = 128` a mask of 7 keeps `col ^ mask` inside the row. With
`HEAD_DIM = 64` and `K_STRIDE = 64` it swizzles columns up to 127, so K reads
land in the neighbouring row — the ~44% relative error we measured. Upstream's
fix comments it directly: *"D=64 must avoid swizzling past col 63."* Two sites
need it: `_k_swizzle` (line 598) and the GEMM1 read mask (line 1017).

---

## 4. Blockers, with evidence

### B1 — parity does not build against our FlyDSL. **Gating.**

Attempting to build the parity gfx950 forward on this host:

```
File "kernels/attention/flash_attn_utils.py", line 3357, in init_lds
    self.lds_kv_base_ptr = lds.kv.ptr.llvm_ptr
AttributeError: 'Pointer' object has no attribute 'llvm_ptr'
```

Installed is `flydsl 0.2.4`. `llvm_ptr` is a property defined only in this
branch's source (`python/flydsl/expr/typing.py:973`), and it calls
`fly.to_llvm_ptr` — a **dialect-level op that does not exist** in 0.2.4's
compiled `fly` dialect. Its op list is `inttoptr, make_ptr, ptr_load, ptr_store,
ptrtoint`; there is no `to_llvm_ptr`. **This cannot be shimmed in Python.**

Fix: build FlyDSL from this branch. That means building LLVM/MLIR at pinned
commit `e2a39f504fee836e4def9581bed817ecc327b9dc`
(`scripts/build_llvm.sh`, then `scripts/build.sh`). Submodules under
`thirdparty/` are not checked out (cloned `--single-branch`).

Machine is well suited: 224 cores, 1007 GB RAM, 450 GB free on `/dockerx`,
cmake 3.31.6, ninja present. Estimate 20–40 min at `-j64`.

Two hazards to control:
- `build_llvm.sh` runs `pip install nanobind numpy pybind11` into the **shared**
  conda env. `numpy 1.23.2` and `pybind11 3.0.1` are already satisfied so no
  upgrade occurs without `-U`, but `nanobind` is absent and would be installed
  there. Torch is ABI-coupled to that numpy. Pre-install nanobind deliberately
  rather than letting the script's pip line run.
- Installing the new FlyDSL **must not** clobber `flydsl 0.2.4`. Both the user's
  running jobs and our working 42-test integration import it. Use a dedicated
  venv and select via `PYTHONPATH`/`VIRTUAL_ENV`, never `pip install` into the
  shared env.

### B2 — no gfx950 or gfx1201 hardware. **Unfixable here.**

8 × MI308X, all `gfx942:sramecc+:xnack-`. gfx950 and gfx1201 code can be written
and reviewed; it cannot be run, tested, or benchmarked. Note the parity repo's
own codegen gate (`tooling/codegen_fingerprint.py`) is not a cross-arch
compile-only check — it allocates `device="cuda"` tensors and actually launches,
so it validates the *local* arch only.

This is precisely how our existing unvalidated `flex_flash_950.py` (2209 lines,
never executed) came to be. Repeating it across two arches × three kernels
(forward, dq, dkdv) would produce a large body of unrunnable code.

### B3 — parity has no gfx942 kernel, and its schedule depends on CDNA4

The only gfx942 mention in `parity/` points back the other way:

```
# flash_attn_func_gfx1201_interface.py:29
The gfx950 / gfx942 equivalents live in ``flash_attn_interface.py``; this
```

Our own kernel enumerates four CDNA4-only features, and **two are structural to
parity's dual-wave schedule**:

| feature | gfx950 | gfx942 | why it matters to parity |
|---|---|---|---|
| `buffer_load_dwordx4_lds` (16 B DMA→LDS) | yes | **no**, dword (4 B) only | dual-wave staging assumes `DMA_BYTES = 16`, `VEC_KV = 8`; `staging_shape` math (`tokens per issue = 512/granule`) is derived from it |
| `ds_read_b64_tr_b16` (LDS transpose read) | yes | **no** | forward reads V as `A[d][token]` from one `[token][d]` tile; the **entire backward** reuses this ("both column-major operands are the forward's V read, unmodified") |
| MFMA K | 16 (`..._32x32x16_bf16`) | 8 (`..._32x32x8bf16_1k`) | mechanical; changes pack type v8f16→v4f16 and doubles issues per K-step |
| `permlane32_swap` + `cvt_pk_bf16_f32` | yes | **no** | 128-bit fused O store; gfx942 falls back to per-lane dwordx2 |

Plus LDS capacity: **64 KB on gfx942 vs 160 KB on gfx950**. Parity prices its
staging at `BLOCK_N * head_dim * ~8.3 B` (`fmha_traits_gfx950.py:51`), which at
head_dim 128 is ~68 KB — over gfx942's budget before anything else is placed.
Our kernel fits because it single-buffers (`BLOCK_N = 64`, `K_STRIDE = HEAD_DIM`,
≈32 KB for K+V), and that double-buffering is exactly what the 8-cluster
pipeline exists to exploit.

**Consequence:** a gfx942 kernel "based on gfx950" must replace the DMA staging
path, replace the transpose-read V path with VT-staged LDS, drop to K=8, drop the
permlane store, and re-derive the head_dim ladder against half the LDS. That is
not a `Gfx942Knobs` subclass; it is a different schedule reached by removing the
things the schedule was built for. Our current kernel is already that kernel.

### B4 — parity has no programmability, and its bias contradicts flex semantics

No `score_mod`, `mask_mod`, or `BlockMask` anywhere in the repo (the only
`kv_indices` hits are paged-KV page tables). Every feature is a compile-time
trait: `causal`, `window`, `bias`, `dropout`, `varlen`, `paged`.

Worse, bias and causal are **mutually exclusive by explicit design**:

```python
# flash_attn_func_gfx1201_aiw.py:600-613
raise ValueError(
    "bias and causal masking are mutually exclusive: bias already is "
    "an attention mask, so combining it with a positional one has no "
    "defined meaning. ..."
)
```

FlexAttention's core contract is that a score_mod and a mask_mod compose freely.
This exclusion must be resolved before parity can host flex, and resolving it
means changing parity's trait model, not just adding a hook.

### B5 — parity is a run-in-place prototype, and the vendoring closure is large

Its own README: *"these files are a self-contained prototype run in place — bare
module imports, cwd must be this directory — rather than a package imported from
elsewhere,"* and *"Nothing here is collected by `scripts/run_tests.sh`."*

Vendoring closure:

| component | lines |
|---|---|
| `parity/` non-test Python | 23 854 |
| `flash_attn_utils.py` (gfx950 parity subclasses it) | 5 632 |
| `kernels/common/` | 3 187 |
| **total** | **≈32 700** |

On top of that: hooks into six kernel bodies, plus the bias/causal resolution,
plus packaging the bare imports into a real module tree.

---

## 5. Target architecture

If we do proceed, the shape to aim for — borrowing parity's own seams, which are
designed for exactly this and currently hold one entry.

**Arch-neutral, shared across all targets:**
- The Inductor-facing template and codegen layer (`codegen/flydsl/*`). Already
  arch-neutral: it emits *source text*.
- The mod-injection ABI — our score_mod/mask_mod lowering contract. This is the
  one thing that genuinely ports, matching parity's maxim *"features port across
  architectures; schedules do not."*
- `FmhaInputMetadata`-style "what to compute": arch-neutral by contract, *"never
  by policy, and never by arch"*.
- The head_dim **ladder** + `padded_head` + `hdim_qk_floor` derivation. Compile
  the next rung up, pass the true extent as a runtime argument. This is how we
  stop being 128-only, and it is independent of any one schedule.
- Host-side ABI (`fmha_abi_gfx1201.py` is already shared by both parity arches).

**Arch-subclassed policy:**
- `FmhaKnobs` base with a per-arch subclass and a registry
  (`_BY_ARCH = {"gfx950": Gfx950Knobs}` today, keyed on arch prefix). Shared
  derivations like `_with_widths` stay on the base *"so a second arch cannot
  accidentally decide `padded_head` by another rule."*
- Replace our `_VALIDATED_ARCHS` / `_PORTED_ARCHS` allowlist with a capability
  registry (has 16 B DMA-to-LDS? has LDS transpose read? MFMA K? LDS bytes?) so
  a new arch declares what it has instead of being pattern-matched by name.

**Per-arch kernel bodies — permanently separate:**
- gfx942/CDNA3, gfx950/CDNA4, gfx1201/RDNA4. No unification, in line with
  parity's stated position: *"Merging the two would mean one abstraction serving
  two schedules that agree on almost nothing,"* and *"When gfx1250 FMHA arrives
  it gets its own `fmha_common_gfx1250.py` for the same reason."*

---

## 6. Phased plan

Each phase names its **gate** — the evidence required before the next starts.
Phases 1–2 are independent of the parity decision and worth doing regardless.

### Phase 0 — FlyDSL from source, isolated (gates everything parity)

1. `git submodule update --init` in `/dockerx/xinya-flydsl`.
2. Pre-install `nanobind` deliberately; confirm `numpy 1.23.2` and
   `pybind11 3.0.1` unchanged afterwards.
3. `LLVM_TARGETS_TO_BUILD="X86;AMDGPU"` (drop NVPTX), `nice -n 15`, `-j64`.
4. `scripts/build.sh` against that MLIR install.
5. Install into a **dedicated venv**. Verify `flydsl 0.2.4` in the shared conda
   env is byte-identical afterwards.

**Gate:** in the new venv, `Pointer.llvm_ptr` resolves and the parity gfx950
forward builder completes without `AttributeError`. Shared env untouched:
`python -c "import flydsl; print(flydsl.__version__)"` still reports 0.2.4, and
our 42 tests still pass against it.

**Outcome (2026-08-28): passed.** LLVM/MLIR built at the pinned commit in ~11 min
at `-j64` (7638 targets, 7.1 GB install at `/dockerx/llvm-project/mlir_install`),
then FlyDSL **0.3.1** into `/dockerx/flydsl-build/venv`. `Pointer.llvm_ptr` is
present and `build_flash_attn_func_gfx950_module(arch='gfx950', head_dim=128)`
returns a launcher, so **B1 is resolved**.

Two build fixes were needed, neither in the plan:
- MLIR at this commit wants **nanobind 2.9**; `build_llvm.sh`'s bare
  `pip install nanobind` gives 3.0.1 and cmake rejects it. Pinned `nanobind~=2.9.0`.
- The final install step calls `patchelf`, which is absent. Installed via pip.

Both landed in the venv rather than the shared env, which is what the isolation
was for: R2 would otherwise have downgraded nanobind under the running jobs.
Note the venv is created `--system-site-packages`, so `deactivate` alone does not
restore the shared interpreter for verification purposes — `PATH` has to be cleaned
too, or the absolute conda path used. Verified afterwards: shared `pip list` and the
`flydsl` tree are **byte-identical** to their pre-build hashes, and the suite passes
against 0.2.4.

**Correction (2026-09-01).** Two things above were recorded wrongly and matter:

- **0.3.1 was never installed into the venv.** `scripts/build.sh` builds a package tree
  and prints "Usage (no install)"; it does not install. The venv holds only nanobind,
  patchelf, pip and setuptools, and `import flydsl` inside it resolves to the shared
  **0.2.4**. 0.3.1 lives at `/dockerx/xinya-flydsl/build-fly/python_packages` and is
  reached by `PYTHONPATH`, not by activating anything:

  ```
  PYTHONPATH=/dockerx/xinya-flydsl/build-fly/python_packages \
      /dockerx/flydsl-build/venv/bin/python -m pytest test/inductor/test_flydsl_flex_attention.py
  ```

- **The suite had never actually run against 0.3.1.** The gate as written only checked
  0.2.4. Run against 0.3.1 it reported `10 passed, 38 skipped` — and the skip is the
  finding, see R1 below.

### Phase 1 — BlockMask KV-skip plumbing (no new kernel, measurable today)

The kernel-side loop and `launcher.kv_block_size` already exist. Work is
Inductor-side: pass `kv_num_blocks` / `kv_indices` through the template, and
require the BlockMask be built with `BLOCK_SIZE=(BLOCK_M, BLOCK_N_OUT)`.

**Gate:** causal wall time drops materially below plain wall time at 2x32x4096
(currently 8198 vs 7976 µs), numerics unchanged, and the causal `fly/tri` ratio
moves off 0.58x. Expected ~1.7x.

**Outcome (2026-08-28): passed, 1.51x rather than 1.7x.** At 2x32x4096 causal went
**8198 → 5436 µs** (33.5 → 50.6 TF), so causal is now well under plain (7962 µs)
where it used to match it. The `fly/tri` ratio moved **0.58x → 0.86x**.

| shape | causal before | causal after | fly/tri before → after |
|---|---|---|---|
| 1x8x4096 | 16.2 TF | 20.7 TF | 0.42x → 0.51x |
| 2x16x2048 | 18.1 TF | 20.9 TF | 0.49x → 0.57x |
| 4x8x4096 | 23.5 TF | 35.8 TF | 0.43x → 0.65x |
| 2x32x4096 | 33.5 TF | 50.6 TF | 0.58x → 0.86x |

Plain and score_mod are unchanged (69.0 vs 68.9 TF at 2x32x4096), confirming the
dense path is untouched.

Two things the plan did not anticipate:
- **The kernel's grid is not FlexAttention's.** `BLOCK_N_OUT` is 64 (the build is
  `causal=False`, so `PATH_TAG` is N32) and `BLOCK_M` is 256 for `num_heads >= 32`,
  against FlexAttention's 128x128 default. Requiring the caller to match, as the plan
  suggested, would have meant the path never engaged. Instead `regrid_block_mask`
  unions the partial and full lists and converts the grid host-side per call — a KV
  split and usually a Q merge.
- **Skipping needs a `mask_mod`**: the builder rejects `block_mask=True` without one,
  and there is nothing to skip anyway, so the gate declines to dense in that case.

The per-call regrid is why the short-sequence gains are smaller; it is the obvious
next optimisation, either hoisted or replaced by teaching the kernel to walk both
lists (which would also let it skip `mask_mod` on full blocks).

#### Phase 1b — the regrid, measured and memoised

Measured rather than assumed, and the assumption was wrong in an instructive way: the
conversion costs a **flat ~215 µs regardless of shape**, because it is a dozen tiny
op launches over a few KB, so it is launch-bound, not data-bound. It was not "small
beside the tiles it saves" at short sequences at all — it was **48% of a 2x8x1024
kernel**, 24% at 1x8x4096, and only fell under 2% at 2x32x8192.

Two attempts at making the conversion itself cheaper both made it *slower*, which is
worth recording so it is not retried: replacing the stable `argsort` compaction with a
prefix sum went 96 → 126 µs, and replacing `repeat_interleave` with expand+reshape went
17.6 → 20.9 µs. At this size one fused kernel beats six cheaper ones, so the sort stays.

What worked is not converting at all on most calls. A BlockMask is built once and reused
for every step of a run, so the result is memoised in a `WeakTensorKeyDictionary` keyed on
`kv_indices`, validated by the identity and `_version` of all four tensors plus the grid
parameters. A hit costs **~11 µs, a 19x drop**, taking worst-case overhead from 48% to
2.6%. An in-place edit or a freshly built mask misses and recomputes, so this costs time,
never correctness; the entry dies with the mask that produced it.

Folding the two lists into one scatter also surfaced a latent correctness bug. The padding
entries past `kv_num_blocks` hold *real* block ids, so once both lists share a scatter, one
list's padding entry can land on a block the other list visits — and `scatter_` gives no
order guarantee among duplicate destinations, so it could write `False` over a visited
block and silently drop it from the softmax. `scatter_reduce_(reduce="amax")` makes any
order give the union. The old code was safe only by accident, via a separate buffer per
list plus `create_block_mask` happening to pad with the unvisited blocks; a hand-built
BlockMask padded with zeros would have broken it. There is now a regression test for
exactly that collision.

Measured effect on the end-to-end causal path (FlyDSL TFLOP/s, gfx942, forward):

| shape | after Phase 1 | after 1b | change |
|---|---|---|---|
| 1x8x4096 | 20.7 TF | 26.1 TF | +26% |
| 2x16x2048 | 20.9 TF | 27.1 TF | +30% |
| 4x8x4096 | 35.8 TF | 39.0 TF | +9% |
| 2x32x4096 | 50.6 TF | 53.7 TF | +6% |

The gradient across shapes is the signature of a fixed cost being removed, which is the
confirmation that the 215 µs was what it looked like. Plain is unchanged at 68.9 TF at
2x32x4096. All 45 tests pass. The `fly/tri` ratios also rose (0.51 → 0.69x at 1x8x4096)
but part of that is Triton autotune run-to-run variance, so the FlyDSL absolute column is
the honest one.

Still open from this item: a workload that rebuilds its mask every step pays the full 215 µs
every step, and full blocks still pay `mask_mod` per element. Both want the kernel walking
two lists.

#### Phase 1c — BHSD addressing, and the end of the transposes

The standing note said fixing the layout adapter "means changing the kernel's index math,
so it is perf work rather than integration work". That was too pessimistic. Both layouts
are **affine in `(batch, head, token, col)`**; they differ only in which axis carries
`seq_len`. So this was a change of coefficients, and the addressing turned out to be
contained in six sites: two chokepoint functions (`global_idx_q`, `global_idx_kv`), two DMA
byte-offset sites, and the two `num_records` bounds.

`build_flex_flash_generic_module` now takes `layout="bshd"|"bhsd"` and the flex template
builds `"bhsd"`, so `[B, H, S, D]` from FlexAttention is indexed where it lies. Measured
first, as with 1b: the copies were **236–891 µs, 3% of the call at 2x32x8192 rising to 23%
at 2x8x1024**.

Two findings worth keeping:
- **The kernel got faster too, not just the copies removed.** Kernel-only, BHSD is 1–6%
  faster at every shape measured, because a `(batch, head)` slice is contiguous, so a KV
  tile is a contiguous run instead of one strided by `num_heads * head_dim`. The BHSD
  `num_records` bound is also tighter — the `(batch, head)` plane rather than the whole
  batch — since a row past `seq_len` would otherwise land in the next head's tokens.
- **`if BHSD:` inside the kernel silently does the wrong thing.** FlyDSL's `if`-rewriter
  turns a bare `if` into a dynamic dispatch, so bindings made inside it never escape;
  `_q_nrec_bytes` came out undefined. `const_expr(BHSD)` is the compile-time branch, which
  is the same trap the GQA code already documents for `kv_head_idx`.

This also turned up a **third hardcoded `0x7` K-swizzle row mask** that Phase 2 missed, in
the DMA K path. That path is gfx950/N128 only so it is unreachable on gfx942, but it is the
same head_dim-64 bug and now uses `K_SWZ_ROWMASK`. Untested here for want of the hardware.

Measured effect, FlyDSL TFLOP/s, gfx942 forward, against the Phase 1b column:

| shape | variant | after 1b | after 1c | change |
|---|---|---|---|---|
| 1x8x4096 | plain | 34.6 | 37.2 | +8% |
| 1x8x4096 | score_mod | 32.4 | 35.7 | +10% |
| 1x8x4096 | causal | 26.1 | 29.7 | +14% |
| 2x16x2048 | causal | 27.1 | 31.4 | +16% |
| 4x8x4096 | plain | 48.8 | 51.9 | +6% |
| 2x32x4096 | plain | 68.9 | 73.4 | +7% |
| 2x32x4096 | score_mod | 63.2 | 66.7 | +6% |
| 2x32x4096 | causal | 53.7 | 59.7 | +11% |

**This is the first item that moved the dense path**, which is the point: the transposes
were overhead on every variant, where block-skipping only ever helped the masked one.
Causal at 2x32x4096 now **beats autotuned Triton, 1.02x** (59.7 vs 58.4 TF), and against
the original Phase 1 baseline causal is up 20.7 → 29.7 TF (+43%) at 1x8x4096 and
50.6 → 59.7 TF (+18%) at 2x32x4096. All 47 tests pass, `test_layouts_agree` pins the two
addressings together, and BHSD matches eager to the same 0.0045 l2_rel as BSHD.

#### Phase 1d — R1 closed, occupancy diagnosed, two items measured and declined

**R1 was real, and it was silent.** `flydsl.expr.buffer_ops` was **deleted** in FlyDSL
0.3.1 — those helpers moved kernel-side, to `kernels/common/buffer_ops.py`, which callers
are now expected to carry. Our kernels did `from flydsl.expr import buffer_ops`, so under
0.3.1 the flex backend failed to import and all 38 flex tests **skipped**, reporting
`10 passed, 38 skipped` rather than failing. Green, and meaningless. Every parity phase
requires 0.3.1, so this would have been discovered somewhere in Phase 3.

Fixed by vendoring upstream's copy as `_vendored_buffer_ops.py` behind
`flex_kernels/buffer_ops.py`, which prefers the library module and falls back to the
vendored one. Three import sites moved onto it (`flex_flash_generic`, `flex_flash_950`,
`kernels/kernels_common`). **48/48 now pass under both 0.2.4 and 0.3.1**, so R1 is
mitigated rather than merely mitigated-on-paper.

**Occupancy is LDS, not registers, and it misses by 512 bytes.** From the emitted ISA
(`FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1`) for the `num_heads=32, head_dim=128` build:

| | |
|---|---|
| `group_segment_fixed_size` (LDS) | **33280 B** |
| `vgpr_count` / `vgpr_spill_count` | 234 / **0** |
| `sgpr_count`, `agpr_count` | 33, 0 |
| workgroup | 512 threads (8 waves) |

Registers are *not* the constraint: 234 is under the 256 that two waves per SIMD needs,
and nothing spills. Two workgroups on a 64 KB CU needs ≤ 32768 B, and we are **512 B
over** — exactly the V transpose padding, since `VT_STRIDE = BLOCK_N + 2` costs
`HEAD_DIM × 2 = 256` elements = 512 B at head_dim 128. Confirmed directly: forcing
`VT_STRIDE = BLOCK_N` drops LDS to exactly 32768 B, VGPRs to 218, and **the
`waves-per-eu` warning disappears entirely**.

It is also **2x slower** that way (2x32x4096 head_dim 128: 7378 → 14498 µs), because a
64-element stride puts every row of the transposed V in the same LDS bank. So the padding
earns its 512 bytes and must not simply be removed. The fix that gets both is an **XOR
swizzle for V**, which is conflict-free at zero space cost and is what K already does.
Upstream has a `v_swizzle` but only wires it to the DMA path, so it does not help gfx942
as-is. This is now the single highest-value gfx942 item, with an exact target: get
`group_segment_fixed_size` to 32768.

**Two items measured, then declined.** Both had been written up as "obvious next", and
the measurements say otherwise:

- *Kernel walking both block lists, to skip `mask_mod` on full blocks.* Attaching a
  `mask_mod` to a dense walk costs **1.7% (2x8x1024) to 6.0% (2x32x4096 head_dim 64)**
  of the tile. Skipping it on full blocks is therefore worth a few percent of causal, and
  it costs a KV loop split in two so each body specialises — doubling the loop code on a
  kernel whose problem is already that it does not fit. Declined on that ratio.
- *Deeper QK prefetch.* We hardcode `_QK_PREFETCH_DEPTH = 2`; upstream's gfx942 mainline
  path runs 3–4. Swept 2/3/4/5: depth 4 is +1.6% at 2x32x4096 head_dim 128, but **−4.8%
  at 2x8x4096**, and depth 5 is catastrophic there (2778 → 14222 µs). There is no single
  right constant, so the default stays at 2 and this belongs as an autotune knob beside
  `MOD_VEC_SIZE`, not as a new hardcode.

**Decision 4 acted on, and it had already cost us.** Vendored files now carry a
provenance header (upstream path, branch, commit) and `refresh_vendored.py` reads them
back: `--check` for local drift, `--log` for upstream commits since the pinned commit,
`--update` to pull content forward. Running it against the upstream log immediately turned
up what the plan predicted: **`829c7b4 [Perf]optimize flydsl flash attention kernel for
gfx942 (#850)`, 2026-07-22** — a *mainline* gfx942 performance commit, not a parity one,
carrying `ENABLE_KV_GPFETCH` (default on for gfx942), `skip_kv_pad_mask`, a vectorized KV
path, and dual `BLOCK_M` builds with runtime routing. Upstream also refactored the
monolith our fork descends from into a shared helper framework (#845, #814), so
`flash_attn_generic.py` is now 721 lines sharing almost no identifiers with our 1693 —
a straight refresh is no longer possible, which is precisely the R7 outcome.

This is the more important correction to open decision 1 than anything in 1b/1c: there
*is* portable gfx942 schedule work available, it is in mainline rather than parity, and we
have been diverging from it for six weeks.

#### Phase 1e — the V swizzle: occupancy 2, and the dense gap halves

§1d ended with an exact target — get `group_segment_fixed_size` from 33280 B to 32768 B
without reintroducing the bank conflicts that made the naive fix 2x slower. Done, by
swizzling V the way K already was.

The transposed V tile is `[HEAD_DIM][BLOCK_N]`, read one v4f16 per lane at a fixed column.
At `VT_STRIDE = BLOCK_N` consecutive lanes sit 128 B apart, which is exactly 32 banks, so
all 32 land on one bank pair. The padding to `BLOCK_N + 2` broke that by shifting each row
one bank, at a cost of `HEAD_DIM × 2` elements. The XOR does the same job for nothing:

```
def _v_swizzle_t(d_idx, n_idx):
    m = (d_idx ^ (d_idx >> V_SWZ_DSHIFT)) & V_SWZ_MASK
    return (((n_idx >> 2) ^ m) << 2) | (n_idx & 3)
```

Two details carry the design. The granule is 4 elements and `n`'s low two bits are
preserved, so a 4-aligned `n` and its next three stay in one moved granule and the reader
still gets its v4f16 in a single access. And the mask folds `d`'s high bits in via
`d ^ (d >> log2(VEC_WIDTH))` because the two access patterns step `d` differently — the
read walks lanes one `d` apart, the cooperative store `VEC_WIDTH` apart — so a mask from
`d`'s low bits alone would have been *constant* across the store's lanes and moved nothing.
The hi half is swizzled separately rather than as `lo + K_SUB_N`, since the XOR moves the
two granules independently.

**Result: LDS exactly 32768 B, the `waves-per-eu` warning gone, 244 VGPRs, no spills.**

**Correction on the mechanism.** §1d framed the 512 B as the thing standing between one
workgroup per CU and two, and the measured win does not support that reading. At 244 VGPRs
a SIMD holds `floor(512/244) = 2` waves, so 4 SIMDs hold 8 waves — exactly one 512-thread
workgroup — and at the old 234 VGPRs the answer was the same 2. Registers, not LDS, decide
it, and they decide it identically before and after. What the swizzle plausibly fixed is
the *store*: the cooperative V store writes one element per lane at a fixed `n`, and with
the `BLOCK_N + 2` stride its lanes step `d` by 16, so `(d + n/2) % 32` collapsed to two
distinct banks. The XOR folds `d`'s high bits in and spreads the same lanes over eight.
That is consistent with the win being largest at the middling shapes and absent at
head_dim 64, but it is inference from the addressing rather than a profile, and it should
be confirmed with one before being repeated.
49/49 pass on both FlyDSL versions. Kernel-only, against the padded build measured in the
same session:

| shape | D | padded | swizzled | |
|---|---|---|---|---|
| 2x8x4096 | 128 | 2778 µs | **2020 µs** | −27.3% |
| 2x8x1024 | 128 | 389 µs | **338 µs** | −13.1% |
| 2x32x4096 | 128 | 7351 µs | 7262 µs | −1.2% |
| 2x32x4096 | 64 | 4997 µs | 4809 µs | −3.8% |
| 2x8x1024 | 64 | 281 µs | 288 µs | +2.5% |

head_dim 64 is a wash, which is the expected shape of the result rather than a
disappointment: its tile was already under 32768 B, so there was no occupancy to win and
all the swizzle does is trade padding for XOR arithmetic. The win is concentrated where
occupancy actually changed, and it is largest at the middling shapes — 2x32x4096 already
has enough workgroups to fill the device, so a second per CU buys least there.

Against autotuned Triton at head_dim 128, `plain` went 0.55–0.82x → **0.72–0.85x** and
causal cleared parity at two further shapes (2x16x2048 1.06x, 4x8x4096 1.05x). The dense
gap is roughly halved rather than closed; what remains is no longer occupancy.

#### Phase 1f — #850's gfx942 work, ported: the hint, not the depth

§1d swept `_QK_PREFETCH_DEPTH` bare and found no winner, with depth 5 catastrophic at
2x8x4096 (2778 → 14222 µs). Reading what mainline #850 actually does for gfx942 explains
it. `ENABLE_GFX942_KV_GPFETCH` has exactly one behavioural site:

```python
if const_expr(traits.ENABLE_GFX942_VEC_K or traits.ENABLE_GFX942_KV_GPFETCH):
    rocdl.sched_group_barrier(rocdl.mask_dsrd, depth * 2, 0)
```

The depth is not the optimization; the hint is. Without it the scheduler spreads the extra
`ds_read`s through the MFMA chain and a "prefetch" stops being one. Our prefetch loop is
structurally identical to upstream's, so this ported directly. Re-swept with the hint:

| shape | D | d2 | d3 | d4 | d5 | best |
|---|---|---|---|---|---|---|
| 2x32x4096 | 128 | 7286 | 7320 | 7184 | **7133** | d5 +2.1% |
| 2x8x4096 | 128 | 2109 | **2081** | 2096 | 2129 | d3 +1.3% |
| 2x8x1024 | 128 | **337** | 339 | 345 | 349 | d2 |
| 2x32x4096 | 64 | 5003 | 4999 | 5025 | **4971** | d5 +0.6% |
| 2x8x1024 | 64 | **284** | 340 | 288 | 287 | d2 |
| 4x8x4096 | 128 | **3577** | 3585 | 3648 | 3688 | d2 |

The 6x cliff is gone — depth 5 at 2x8x4096 went 14222 → 2129 µs — which is the result worth
having. So `qk_prefetch_depth` became a builder parameter and an autotune dimension gated
behind `flydsl.autotune_qk_prefetch_depth`.

The kernel-only sweep above understates it. Letting the autotuner pick per shape, measured
end to end against the same script that produces the Triton comparison:

| shape | D | variant | depth 2 only | tuned | gain |
|---|---|---|---|---|---|
| 2x32x4096 | 128 | score_mod | 8109 | 7669 | 5.4% |
| 2x16x2048 | 128 | score_mod | 1686 | 1605 | 4.8% |
| 2x32x4096 | 128 | causal | 4614 | 4487 | 2.8% |
| 2x32x4096 | 128 | plain | 7380 | 7192 | 2.5% |
| 4x8x4096 | 128 | causal | 2374 | 2329 | 1.9% |
| 4x8x4096 | 128 | score_mod | 4741 | 4710 | 0.7% |

`score_mod` gains most, which fits: a mod puts VALU work between the MFMAs, leaving more
room for the grouped `ds_read`s to hide behind. head_dim 64 is flat.

It is still **off by default** — 2–5% does not pay for four times the builds when the
kernel is compiled per shape — but it is worth turning on for an expensive mod on a large
shape, which is one environment variable.

Verified end to end: with the flag off the FlyDSL path offers 3 choices (the three
`mod_vec_size`s, all at depth 2); with it on, `num_choices` is 12 and depths 2–5 all appear.
Both produce correct numbers, and the default table is unchanged from the post-swizzle
baseline.

Not taken from #850: the dual `BLOCK_M` builds with runtime routing on `B*S` (Inductor
autotunes tile choice already, and the plan's §1 reasoning for one tile shape per build
still holds), `skip_kv_pad_mask`, and `ENABLE_GFX942_VEC_K` — the last needs
`ENABLE_GFX942_DMA`, which is off by default upstream and is the same CDNA3 dword-DMA
problem Phase 3 has to solve anyway.

### Phase 2 — head_dim 64 via the swizzle backport (small, verifiable)

Apply `K_SWZ_ROWMASK = HEAD_DIM // 16 - 1` at both sites; extend
`_SUPPORTED_HEAD_DIMS` to include 64; parametrize the test suite over
head_dim ∈ {64, 128}.

**Gate:** head_dim 64 matches eager within tolerance (today: ~44% relative
error), and head_dim 128 is bit-identical to before the change.

**Outcome (2026-08-28): passed.** `K_SWZ_ROWMASK = HEAD_DIM // 16 - 1` at both
sites. head_dim 64 now lands at l2_rel 0.0044 against eager, the same accuracy as
128, where it was ~44% wrong. 128 is unaffected by construction — `128 // 16 - 1`
is the 7 that was hardcoded — and measures 69.0 vs 68.9 TF.

The ladder does not extend past 64 on gfx942, so `_SUPPORTED_HEAD_DIMS` is
`{64, 128}` and not wider: **96** fails `BLOCK_SIZE % THREADS_PER_ROW_LOAD == 0` at
build (and 96/16 = 6 is not a power of two, so the mask would not be a clean
row-confined XOR either), and **256** needs 66560 B of LDS against gfx942's 65536 B
— the same LDS ceiling that §B3 says blocks parity's peak rung.

### Phase 3 — parity gfx950 forward, vendored and compiling (unvalidated)

Package parity's bare imports into a module tree; vendor the closure from §B5;
build the gfx950 forward inside the Phase 0 venv.

**Gate:** it compiles and emits gfx950 ISA. It **cannot** be numerically
validated here — record that explicitly in the file header, as we did for
`flex_flash_950.py`, and keep it behind `config.flydsl.allow_unvalidated_arch`.

#### Phase 3 scoping — what running parity on gfx942 actually costs

The decision to validate the gfx950 body on gfx942 rests on the two being "the same CDNA
architecture with different LDS". They are not, in the two places parity leans on hardest,
and the shape of the work is now known precisely rather than guessed at.

**The DMA is one choke point, and that part is easy.** Every KV staging load in parity
funnels through one method:

```python
def _issue_kv_dma(self, src_div, lds_addr, src_elem, soffset):
    _buffer_load_lds_128(src_div, lds_addr, src_elem, soffset,
                         _dma_atom=self.dma_atom, _lds_ptr_ty=self.lds_ptr_ty)
```

`self.dma_atom` is `fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)`, which is the
instruction CDNA3 does not have and which makes LLVM fail with *"Do not know how to expand
this operator's operand"* rather than a legible error. A CDNA3 subclass overriding this one
method is a small change, and the non-parity `DmaKvGmemToLdsLoader` already shows the
shape: it issues `rocdl.raw_ptr_buffer_load_lds` with `self._dma_size = fx.Int32(4)` when
`ENABLE_GFX942_DMA` is set.

**The LDS layout follows, and that part is fine too.** Parity's traits already derive the
whole staging layout from `DMA_BYTES`, so a 4 does propagate:

```python
smem_linear_wave = WARP_SIZE * DMA_BYTES // BF16_BYTES   # 512 -> 128 elements
smem_k_pad       = DMA_BYTES // BF16_BYTES               # 8 -> 2
kv_vec_size      = DMA_BYTES // BF16_BYTES               # 8 -> 2
```

**The read path is where it stops being a knob flip.** `buffer_load_lds` writes lane `i` to
`lds_base + i * size`, so the per-lane stride *is* the DMA width — four dword issues cannot
reproduce one 16-byte issue's layout, they produce a different one. That is what the traits
above are re-deriving, and it lands in `KV_VEC_SIZE = 2`, which is not a layout constant but
the granule the MFMA feed reads with:

```python
return k_base + (ks * 2 + lane_div_32) * (traits.BLOCK_N * traits.KV_VEC_SIZE) \
     + lane_mod_32 * traits.KV_VEC_SIZE
```

At `KV_VEC_SIZE = 8` a lane's pack is one contiguous read. At 2 it is 2 bf16 — a quarter of
a `v4f16` — so every K pack read becomes four strided reads that must be reassembled, and
the bank behaviour of the existing swizzle no longer holds. **Second, `V` is read through
`_ds_read_tr_v4f16_imm`, the hardware LDS transpose, which is CDNA4-only** and needs the
software transpose our own kernel already implements.

So Phase 3 is not "vendor and flip DMA_BYTES". It is: vendor, override `_issue_kv_dma`,
then rework parity's K pack feed for a 2-element granule and replace its V transpose. The
first two are hours; the last two are the actual project, and they are exactly the two
things parity's dual-wave schedule was designed around. Worth weighing against the
alternative of taking parity's *ideas* into our working gfx942 body rather than its code —
our kernel already has the software transpose and an 8-element granule that works.

### Phase 4 — flex hooks onto the parity forward

Port the score_mod/mask_mod injection sites and aux-tensor reads onto parity's
dual-wave body. Resolve the bias/causal exclusion — most likely by routing flex
score_mods through the score site rather than parity's bias tensor, leaving
`BIAS_TYPE` unused.

**Gate:** generated FlyDSL source for a parity-hosted score_mod is inspectable
and compiles. Numerical validation still blocked by B2.

### Phase 5 — gfx1201

Vendor the aiw forward and `fmha_common_gfx1201`. Wave32/WMMA means the hook
lowering must be re-derived against a different accumulator map
(`column(i) = (i//16)*32 + ((i//8)%2)*16 + i%8 + klane*8`, versus MFMA's 16
columns per accumulator on `lane % 32`).

**Gate:** compiles for gfx1201. Unrunnable here.

### Phase 6 — backward

Two kernels per arch (dq + dkdv); no fused variant, matching parity's decision
(the gfx1201 fused kernel makes every program reserve the larger role's LDS —
52480 B vs 13568 B at head_dim 128 f16 — so dQ programs run at quarter
occupancy). `delta = rowsum(dO * O)` is an *input*; compute it in Inductor.
Wire FlexAttention autograd.

**Gate:** on gfx942 only, and only if a gfx942 backward exists — which per §B3
means writing one, since parity's backward is built on the transpose read.

### Phase 7 — re-benchmark

Re-run both scripts. Add a head_dim sweep once Phase 2 lands.

**Gate:** gfx942 numbers only. gfx950 and gfx1201 remain unmeasured until
hardware exists.

**Outcome (2026-09-01): gfx942 half done, head_dim sweep added.**
`flydsl_vs_triton_autotune.py` now sweeps `HEAD_DIMS = (64, 128)` rather than pinning 128.
Head_dim 64 is not a poor relation — on causal it is the stronger of the two relative to
Triton, clearing parity at two shapes:

| shape | variant | D=64 fly/tri | D=128 fly/tri |
|---|---|---|---|
| 1x8x4096 | plain | 0.67x | 0.55x |
| 1x8x4096 | causal | 0.86x | 0.75x |
| 2x16x2048 | causal | 0.92x | 0.86x |
| 4x8x4096 | causal | **1.04x** | 0.80x |
| 2x32x4096 | plain | 0.68x | 0.82x |
| 2x32x4096 | causal | 0.99x | **1.04x** |

Two things to read from this. Causal is at or past autotuned Triton at the large shapes on
both head_dims, so the block-skip plus layout work landed. Plain is still 0.55–0.82x
everywhere, and that residue is the 512-byte LDS overage in §1d, not anything sparsity or
addressing can reach.

**Superseded by §1e**, which fixed that overage. Re-measured after the V swizzle, head_dim
128 `plain` is 0.72–0.85x and causal clears parity at 2x16x2048 (1.06x) and 4x8x4096
(1.05x) as well. head_dim 64 is unchanged, its tile having already fit.

---

## 7. Risk register

| # | risk | severity | mitigation |
|---|---|---|---|
| R1 | New FlyDSL breaks our working integration (`ArithValue` was removed upstream; `Pointer` API changed) | high | Isolated venv; keep 0.2.4 as the default; re-run 42 tests against both |
| R2 | `pip install` in `build_llvm.sh` perturbs the shared conda env under running jobs | high | Pre-install nanobind; verify numpy/pybind11 unchanged; never `-U` |
| R3 | Thousands of lines of unvalidatable gfx950/gfx1201 code accumulate | high | Header-declare unvalidated status; keep behind the arch gate; do not let it gate gfx942 work |
| R4 | gfx942 parity port stalls on the two missing CDNA4 features | high | Recognise our current kernel already solves this; do not re-derive |
| R5 | bias⊥causal blocks flex semantics on parity | medium | Route score_mods through the score site, leave `BIAS_TYPE` unused |
| R6 | Build contends with user's jobs | medium | `nice -n 15`, `-j64`, drop NVPTX target |
| R7 | Vendored 33k lines become an unmaintainable fork, as our current 1693-line one did | medium | Vendor by subtree with a recorded upstream SHA and a refresh script |
| R8 | Parity's own known bugs inherited (split-K returns wrong answers; stale head_dim-96 comment) | low | Gate split-K off; the 96 bug was fixed in `98493cc`, the comment is stale |

---

## 8. Open decisions

1. **Is the parity adoption justified at all on gfx942?** Phases 1–2 may close
   most of the measured Triton gap without it. Suggest re-deciding after Phase 2
   with fresh numbers.

   **Answered 2026-08-28: partly, and the case for gfx942 is now weaker.** Causal
   closed most of its gap (0.58x → 0.86x of autotuned Triton at 2x32x4096) without
   touching the kernel's schedule. What did *not* move is the non-causal path:
   plain is still 0.50–0.77x and score_mod 0.48–0.78x, unchanged, because there is
   no sparsity to recover there. That residue is the schedule — the occupancy-1
   builds and the BHSD↔BSHD transposes — which is exactly what parity's dual-wave
   work addresses and what §B3 says cannot be ported to gfx942 anyway. So the
   remaining gfx942 upside sits with the two cheap local items (hoisting the
   regrid, removing the transposes), not with parity. Parity's case now rests
   almost entirely on gfx950/gfx1201 reach, which turns on decision 2.

   **Updated 2026-09-01 after Phases 1b and 1c — this now cuts against parity on
   gfx942.** Both local items are done and together they moved causal 20.7 → 29.7 TF
   (+43%) at 1x8x4096 and 50.6 → 59.7 TF (+18%) at 2x32x4096, where causal now
   *beats* autotuned Triton at 1.02x. The claim above that the dense residue was
   "the schedule — the occupancy-1 builds **and the BHSD↔BSHD transposes**" was
   half wrong about which half was reachable: the transposes were assumed to need
   parity-grade kernel work and instead took a coefficient change at six sites,
   worth +6–8% on plain and score_mod. So the dense path did move after all, and
   what is left of the gap (plain 0.55–0.82x) is *only* occupancy and the LDS
   transposes.

   That narrows parity's gfx942 case rather than strengthening it: the two items
   assumed to need parity did not, and the one that genuinely does — occupancy —
   is the one §B3 says cannot be ported for want of CDNA4 features. Parity still
   rests almost entirely on gfx950/gfx1201 reach, i.e. on decision 2.

   **Updated again 2026-09-01 after Phase 1d — the answer is now "no, and there is
   a better target".** The remaining dense gap was attributed to occupancy, and
   occupancy was attributed to register pressure, which parity's schedule would
   address and gfx942 cannot host. Measuring it says otherwise: registers are fine
   (234 of 256, no spills) and the limit is **LDS, over budget by 512 bytes** — a V
   padding choice, fixable with a swizzle, needing nothing from CDNA4. Separately,
   mainline upstream has a gfx942 performance commit we never took (#850). So the
   two live gfx942 leads are both *ours to take* and neither needs parity. Parity's
   case is now strictly decision 2 and nothing else.
2. **Is there gfx950 / gfx1201 hardware elsewhere?** This single answer decides
   whether Phases 3–6 produce validated code or a large unrunnable artifact.
3. **Do we need the backward at all near-term?** FlexAttention currently falls
   back for backward. If that fallback is acceptable, Phase 6 can be deferred
   indefinitely and the scope shrinks a great deal.
4. **Upstream-refresh or hard-fork?** Our 1693-line fork drifted far enough to
   miss four relevant fixes in three weeks. A subtree with a pinned SHA and a
   refresh script would have surfaced them.

   **Answered 2026-09-01: hard-fork for the kernel, tracked-vendor for helpers.**
   The choice is made for us on the kernel: upstream dissolved the monolith into a
   shared helper framework (#845, #814), so `flash_attn_generic.py` is 721 lines
   sharing almost no identifiers with our 1693-line fork. There is nothing left to
   refresh *from*. What is still tractable is (a) tracking helper files we copy
   verbatim, which `refresh_vendored.py` now does via provenance headers, and (b)
   reading the upstream log for that path periodically, which is how #850 surfaced.
   Treat mainline as a source of *patches to port by hand*, not merges.

---

## Appendix A — environment

| | |
|---|---|
| GPUs | 8 × AMD Instinct MI308X, `gfx942:sramecc+:xnack-`, 80 CUs @ 1.42 GHz |
| Peak bf16 dense | 116.3 TFLOP/s (80 × 1024 × 1.42e9) |
| LDS/workgroup | 64 KB (gfx942) vs 163840 B (gfx950) vs 65536 B (gfx1201) |
| torch | 2.14.0a0+gitaf1b9ac, editable from `/dockerx/pytorch` |
| flydsl | 0.2.4 (site-packages) — **lacks** `fly.to_llvm_ptr` |
| numpy / pybind11 / nanobind | 1.23.2 / 3.0.1 / absent |
| build host | 224 cores, 1007 GB RAM, 450 GB free on `/dockerx` |
| LLVM pin | `e2a39f504fee836e4def9581bed817ecc327b9dc` |

## Appendix B — key paths

Ours:
- `torch/_inductor/kernel/flex/flydsl_flash_attention.py` — eligibility gate, factory
- `torch/_inductor/kernel/flex/templates/flydsl_flash_attention.py.jinja`
- `torch/_inductor/codegen/flydsl/` — codegen, op overrides, runtime shim, README
- `.../vendored_templates/flydsl/flex_kernels/flex_flash_generic.py` — 1693 L, gfx942-validated
- `.../flex_kernels/flex_flash_950.py` — 2209 L, **never executed**
- `test/inductor/test_flydsl_flex_attention.py` — 32 tests
- `test/inductor/test_flydsl_template.py` — 10 tests

Parity (`/dockerx/xinya-flydsl/kernels/attention/parity/`):
- `flash_attn_func_gfx950.py` 1382 L, `fmha_dualwave_gfx950.py` 1494 L,
  `fmha_wide_gfx950.py` 428 L, `fmha_traits_gfx950.py` 488 L,
  `fmha_tuning_gfx950.py` 782 L
- `fmha_bwd_dq_gfx950.py` 1741 L, `fmha_bwd_dkdv_gfx950.py` 2328 L, + m16 variants
- `flash_attn_func_gfx1201_aiw.py` 2291 L, `fmha_common_gfx1201.py` 1657 L,
  `fmha_abi_gfx1201.py` 620 L
- Docs: `sdpa_lore_gfx950.md`, `sdpa_lore_gfx1201.md`, `gfx1201_fmha.md`,
  `sdpa-bwd-plan-gfx950.md` (3575 L), `sdpa-close-gap-gfx950.md` (2500 L)

## Appendix C — parity's ladder (measured on MI355X, not comparable to ours)

`B=4 H=8 S=4096` bf16 non-causal, from `fmha_tuning_gfx950.py`:

| tile | waves | BLOCK_M | gran | stages | shards | LDS | TFLOP/s |
|---|---|---|---|---|---|---|---|
| 32 | 4 | 128 | 32 | 1 | 1 | 17 KB | 618 |
| 64 | 8 | 256 | 64 | 1 | 1 | 33 KB | 889 |
| 128 | 8 | 256 | 64 | 1 | 1 | 67 KB | 1117 |
| 160 | 4 | 128 | 32 | 1 | 1 | 83 KB | 917 |
| 192 | 4 | 128 | 64 | 1 | 1 | 100 KB | 936 |
| 256 | 4 | 128 | 64 | 1 | 1 | 133 KB | 940 |
| 384 | 4 | 128 | 64 | 2 | 1 | 100 KB | 803 |
| 512 | 4 | 64 | 64 | 2 | 2 | 133 KB | 479 |

`LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)`. Note the 67 KB at
the peak rung: that is the number that does not fit gfx942.

Also from parity's docs: causal runs at **68%** of non-causal efficiency
(789.6 / 1152.7 at D=128). Ours is at ~48% (33.5 / 68.9), the difference being
its tile-cut skipping versus our none.

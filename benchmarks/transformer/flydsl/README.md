# FlyDSL flex-attention benchmarks

Measurements for the FlyDSL flex-attention backend
(`kernel_options={"BACKEND": "FLYDSL"}`), against the Triton backend on the same inputs.
They share the mods and shape generators in `../score_mod.py`, so a cell here is
comparable to a cell there.

Run them from anywhere; they find `score_mod.py` themselves.

```bash
python benchmarks/transformer/flydsl/mod_matrix.py
bash benchmarks/transformer/flydsl/run_all.sh out.log   # every table, ~1 h
```

`run_all.sh` runs them one at a time on purpose: they all measure the same GPU, so two at
once measure each other.

Every script takes `--help`. They all need a ROCm GPU and the `flydsl` package;
`kernel_only.py` and `isa_stats.py` additionally bypass Inductor and build launchers
directly, so they will not run against a build where the vendored kernels are absent.

## What each one answers

| Script | Question |
| --- | --- |
| `mod_matrix.py` | How does the backend do across the whole mod matrix, forward and backward? `--flydsl-options K=V` pins one of our knobs on the FlyDSL side only, which is how the backward's `MOD_VEC_SIZE` question was settled. |
| `shape_ladder.py` | How does it do across shapes and head_dims, against Triton *and* aten? |
| `decode_gap.py` | What does a decode shape cost with no decode kernel, bare and with a mod in the graph? Both of Triton's kernels are columns, and `--mods` picks the mod. |
| `sparse_gaps.py` | What are the two sparse gaps worth -- sparse decode under a block mask (`--decode`), and the mod site on blocks a mask already settled (`--mask-cost`)? |
| `walk_cost.py` | How much of a call is fixed per Q tile, and how much is per KV block? |
| `tile_vs_band.py` | Is that fixed cost fixed, or a tall Q tile walking the union of its rows' block lists? Prints the walk the mask asks for beside the walk each tile height takes. |
| `profile_call.py` | Which device kernels is a call actually spending its time in? |
| `layout_ab.py` | For BSHD memory, is indexing it cheaper than copying it to BHSD? |
| `kernel_only.py` | What does the kernel do per head_dim with the lowering taken out? |
| `isa_stats.py` | What is the register pressure, and where does it spill? `--backward` reads the two backward kernels, which is where the 256-VGPR cliff bites. `--arch gfx950` compiles for another GPU instead of this one, one subprocess per build, which is the only evidence a gfx942 host can produce about CDNA4. |

`mod_matrix.py` is the headline number the README in
`torch/_inductor/codegen/flydsl/` quotes, and `shape_ladder.py` is where the TFLOP/s tables
in `MULTI_ARCH_ROADMAP.md` come from. The other five exist because that headline was wrong
once in an instructive way: sparse masks were reading 0.71-0.86x in the forward, `walk_cost.py`
showed a fixed cost of 1075 us per call against Triton's 41 us while the per-block cost was
1.6x *better*, and `profile_call.py` named it -- four `aten::copy_` kernels worth 574 us,
converting BSHD inputs into the BHSD layout the launcher had been hardcoded to.
The kernel indexes either layout, so the fix was to read the strides and build the one the
caller already has. `layout_ab.py` is what keeps that decision honest as shapes change.

## Reading the numbers

Ratios are `triton / flydsl`, so above 1.00x is FlyDSL ahead. The geomean is over cells,
which weights a cheap mod the same as an expensive one -- deliberately, since the point is
coverage rather than a single workload's wall time.

Timings take the best of three runs. These boxes are usually shared and interference only
ever slows a run down, so the minimum is the closest thing to an uncontended number that a
busy machine will give up. Even so, expect a few percent of drift between sessions, and do
not read a 1.03x as a win without re-running it.

Prefill scripts time end to end; the two decode scripts read the *device timeline* instead
(`device_us` in `_common.py`). A decode kernel runs for tens of microseconds against ~350 us
of Inductor Python per call, so an end-to-end decode timing measures the host and comes out
flat in `Skv`. `decode_gap.py --wall` shows that if you want to see it.

Two things to know before quoting a decode ratio against Triton. Its decode kernel is much
faster with an all-true `BlockMask` than with no mask at all, because `block_mask=None`
hands it one sparse block of `1 << 30` and a one-block walk cannot be split across
workgroups; `decode_gap.py`'s last table takes each backend at its own cheaper spelling,
and that is the number to quote. And autotuned choices under a block mask are ranked
against flex's shared fake mask, which is fully occupied, so a sparse cell's choice was
picked on dense merits. That mostly does not matter — `tile_vs_band.py` shows the tall Q
tile stays right down to a 512-token band — but `--flydsl-options BLOCK_M=...` is how to
check a given cell.

`shape_ladder.py` is the exception: it reports a median of fifteen, which is what its
already-published rows were measured with, and changing it now would cost the comparison
against them. A median is the more pessimistic choice on a contended box, so its cells read
slightly lower than the same cell would here.

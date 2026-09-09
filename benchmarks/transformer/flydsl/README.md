# FlyDSL flex-attention benchmarks

Measurements for the FlyDSL flex-attention backend
(`kernel_options={"BACKEND": "FLYDSL"}`), against the Triton backend on the same inputs.
They share the mods and shape generators in `../score_mod.py`, so a cell here is
comparable to a cell there.

Run them from anywhere; they find `score_mod.py` themselves.

```bash
python benchmarks/transformer/flydsl/mod_matrix.py
```

Every script takes `--help`. They all need a ROCm GPU and the `flydsl` package;
`kernel_only.py` and `isa_stats.py` additionally bypass Inductor and build launchers
directly, so they will not run against a build where the vendored kernels are absent.

## What each one answers

| Script | Question |
| --- | --- |
| `mod_matrix.py` | How does the backend do across the whole mod matrix, forward and backward? `--flydsl-options K=V` pins one of our knobs on the FlyDSL side only, which is how the backward's `MOD_VEC_SIZE` question was settled. |
| `shape_ladder.py` | How does it do across shapes and head_dims, against Triton *and* aten? |
| `walk_cost.py` | How much of a call is fixed per Q tile, and how much is per KV block? |
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

`shape_ladder.py` is the exception: it reports a median of fifteen, which is what its
already-published rows were measured with, and changing it now would cost the comparison
against them. A median is the more pessimistic choice on a contended box, so its cells read
slightly lower than the same cell would here.

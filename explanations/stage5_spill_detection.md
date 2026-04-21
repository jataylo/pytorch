# Stage 5 — Spill Detection and Filtering

**Sources:**
- `_estimate_spill_risk()` in `triton_heuristics.py` (pre-compile heuristic)
- `_evict_spill_configs()` in `triton_heuristics.py` (post-compile eviction)
- `bench()` in `triton_heuristics.py` (runtime enforcement)

**Role:** Prevents benchmarking and selection of kernel configurations whose compiled
binaries exceed the register budget, causing catastrophic performance regression from
register spilling.  Two mechanisms operate at different points in the pipeline: a fast
pre-compile estimate and an exact post-compile eviction pass.

---

## Technical

### Background — VGPR budget and spills

**VGPRs (Vector General-Purpose Registers)** are per-thread scratch space used to hold
intermediate values, tensor element values, index arithmetic results, and loop variables.
AMD CDNA hardware provides **65,536 VGPRs per CU**, partitioned equally across all
resident wavefronts.  The allocation happens in **8-register granules**:

```
max_vgprs_per_thread = min(256, floor(65536 / (num_warps × 64) / 8) × 8)
```

When the Triton compiler determines the kernel needs more than `max_vgprs_per_thread`
registers, it **spills**: the excess register values are written to and read from
**scratch memory** (a per-thread local buffer backed by L2 or HBM) via extra
`scratch_store` and `scratch_load` instructions.

**Spill cost:**
- Each spilled register requires 2 extra memory operations (a store when overwritten,
  a load when needed again)
- At 16 spilled registers and 1M elements, this adds ~32M extra memory transactions
- Effective bandwidth impact: 2–5× slowdown for heavily-spilling kernels

`launcher.n_spills` (populated by Triton after compilation from the compiled binary's
metadata section) reports the number of spilled registers.

**Thresholds:**
```
spill_threshold = 32  (ROCm / AMD)
spill_threshold = 16  (CUDA / NVIDIA)
```
ROCm's threshold is higher because AMD's scratch memory is slightly faster relative to
peak HBM than CUDA's equivalent.  Both are tunable via `inductor_meta['spill_threshold']`.

### Mechanism A — Pre-compile heuristic estimate

`_estimate_spill_risk(config_dict, problem_metadata, kernel_metadata)` estimates VGPR
demand from first principles **before any compilation occurs**.

#### Step 1 — Hardware budget

```python
threads_per_block = num_warps × warp_size
raw_max           = 65536 // threads_per_block
max_vgprs         = min(256, (raw_max // 8) × 8)   # floor to 8-reg granule
```

This mirrors the exact hardware allocation formula.

#### Step 2 — Estimated demand (additive components)

```
Component       Formula                             Rationale
──────────────  ──────────────────────────────────  ──────────────────────────────────────
base            16                                  Loop-control, predicate regs, pid,
                                                    wavefront-ID registers
tensor_regs     (num_inputs + num_outputs) × 8      Per tensor: pointer (2 SGPR→VGPR
                                                    alias), loaded value, mask, offset
ops_regs        n_fast × 1                          Fast ops (FMA, add) — high reuse,
              + n_med  × 3                          1 extra temp each
              + n_slow × 6                          Slow ops (exp, log) — ~5 temp values
                                                    for polynomial approximation
dim_regs        + 12 if YBLOCK > 0                  2-D: y-index, y-stride, y-offset,
                + 18 if ZBLOCK > 0                  extra predicate; 3-D: two more
unroll_regs     min(tile_volume // warp_size, 16)   Triton unrolls all axes; capped at
                where tile = XBLOCK×YBLOCK×ZBLOCK   16 because reuse stabilises after
pipeline_regs   max(0, num_stages − 1) × 8          Software-pipelining keeps N copies
                                                    of the load buffer live simultaneously
```

```python
estimated_vgprs = base + tensor_regs + ops_regs + dim_regs + unroll_regs + pipeline_regs
estimated_vgprs = ceil(estimated_vgprs / 8) × 8    # round up to 8-reg granule
```

The estimate intentionally runs **10–20% conservative** (over-estimates demand) to avoid
suppressing valid configs.  The key design choices:

1. **`ops_regs` weights by instruction class:** `tl.exp` requires ~5 temporaries for the
   polynomial approximation steps; `tl.add` reuses the same 1–2 registers across all
   elements.  A flat multiplier would severely over-estimate FMA-heavy kernels or
   under-estimate transcendental-heavy ones.

2. **`unroll_regs` bounded at 16:** Triton unrolls loops across the full tile volume.
   For `XBLOCK=256, num_warps=4`, the unroll depth is `256×1×1 / 64 = 4` iterations.
   For `XBLOCK=256, YBLOCK=16, num_warps=4`, it's `256×16 / 64 = 64`.  The compiler
   reuses registers across unrolled iterations once the live-set stabilises, so the cap
   of 16 prevents over-estimation for large tiles.

3. **Granule rounding:** The real hardware allocates in 8-register blocks.  A kernel
   needing 65 VGPRs is allocated 72 — the "wasted" 7 registers are genuine hardware
   overhead that our estimate also produces, matching the actual allocation ceiling.

#### Step 3 — Spill risk label

```
ratio = estimated_vgprs / max_vgprs

ratio > 1.0   → "HIGH"  — very likely to spill after compilation
ratio > 0.75  → "MED"   — at risk; monitor in verbose output
ratio ≤ 0.75  → "low"   — probably safe
```

This label is shown in the verbose scoring table as the "Spill?" column.  It is
**informational only** — Mechanism A does not change config ranking or selection.

### Mechanism B — Post-compile eviction (optional)

Enabled when `TORCHINDUCTOR_HEURISTICS_SPILL_BUFFER > 0`.

#### Expanded compile pool

In heuristics-only mode (`heuristics_real_bench=False`), `_score_and_prune_heuristic_configs`
normally sets `self.configs` to the top-N scored configs.  With `spill_fallback_buffer=K`:

```python
compile_pool = top_N + K     # e.g. top-5 + buffer-3 = 8 compiled configs
self.configs = scored_configs[:compile_pool]
```

The full ranked list (all scored configs in order) is stored in
`_HEURISTICS_FULL_RANKED_HDICTS[problem_key]` for use by the eviction pass.

#### Eviction pass — `_evict_spill_configs()`

Runs **after** `_make_launchers()` (which populates `launcher.n_spills`) but **before**
`bench()` (GPU timing runs).  The pass walks the top-N pool and replaces spilling configs
with non-spilling buffer candidates:

```python
# Build lookup: effective-hdict-key → compiled launcher
hdict_to_launcher = {
    tuple(sorted(launcher_hdict(ln).items())): ln
    for ln in self.launchers
}

# Pre-filter buffer pool to only non-spilling candidates (in score order)
full_ranked  = _HEURISTICS_FULL_RANKED_HDICTS[problem_key]
buffer_pool  = full_ranked[len(top_n_selected):]   # configs ranked N+1, N+2, …

buffer_candidates = [
    hdict for hdict in buffer_pool
    if hdict_to_launcher.get(key(hdict), None) is not None
    and hdict_to_launcher[key(hdict)].n_spills <= spill_threshold
]

# Walk top-N; swap out spilling configs with buffer replacements
buf_idx = 0
new_top_n = list(top_n_selected)
for i, hdict in enumerate(new_top_n):
    launcher = hdict_to_launcher.get(key(hdict))
    if launcher and launcher.n_spills > spill_threshold:
        if buf_idx < len(buffer_candidates):
            new_top_n[i] = buffer_candidates[buf_idx]
            buf_idx += 1
```

After eviction:
- Top-N configs are passed to `bench()`.  Any remaining spilling configs (buffer
  exhausted) return `float("inf")` from `bench()`.
- If ALL top-N configs spill: walk `_HEURISTICS_FULL_RANKED_HDICTS` for any non-inf
  timing from the extended compile pool.
- Last resort: if every compiled config spills, select the config with minimum `n_spills`
  (least-bad option).

#### Spill recording in `bench()`

```python
def bench(launcher, cfg, ...):
    if launcher.n_spills > spill_threshold:
        if tracking_enabled:
            _store_actual_timing(problem_key, cfg_dict, float("inf"))   # record FIRST
        return float("inf")                                              # then exit
```

The timing is stored **before** the early return.  Without this ordering, spilled configs
would be absent from `_HEURISTICS_VALIDATION_DATA['actual_timings']`, causing the
validation summary to produce empty output for 2-D kernels where all top-N configs spill
(the silent-failure bug discovered in `tuning10_verbose.log`).

---

## Simple explanation

### What register spills are

Every GPU thread has a small number of "scratch slots" (VGPRs) — like the registers in a
CPU but one set per GPU thread (on AMD, 65,536 VGPRs shared across all threads in a CU).

If a kernel is complex enough to need more scratch slots than are available per thread
(because too many threads are sharing the total budget), the compiler has to write the
overflow values out to slow memory and read them back when needed.  This is called a
**register spill**, and it makes the kernel 2–5× slower because those extra memory
operations weren't planned for in the original design.

The tricky part: **you only find out about spills after the compiler runs**, which is
expensive (50–500 ms per config).  By the time you know a config spills, you've already
wasted compilation time.

### The two tools

**Tool A — Pre-compile estimate (fast, approximate, informational)**

Before compiling anything, the system does a quick calculation to estimate how many
registers the kernel *probably* needs:
- How many tensors does it read/write? (each needs pointer registers + value registers)
- What operations does it do? (transcendentals like `exp` need ~5 temporary registers;
  simple adds need 1)
- How big is the tile? (larger tiles need more unrolled loop variables)
- How many wavefronts? (more wavefronts = smaller register budget per thread)

The result is shown in the scoring table as `HIGH`, `MED`, or `low` under the "Spill?"
column.  A `HIGH` warning means: "this config is very likely to spill — watch out."
This estimate **doesn't change config selection** — it's a warning label only.

**Tool B — Post-compile eviction (exact, optional, requires env var)**

If you set `TORCHINDUCTOR_HEURISTICS_SPILL_BUFFER=3` (for example), the system:

1. Compiles the top-5 configs **plus 3 backup configs** (the next-best predicted configs)
2. Checks the exact spill count in each compiled binary (this is free — it's metadata
   in the compiled object, no GPU run needed)
3. **Before doing any GPU benchmarking**, swaps out any spilling top-5 configs with
   non-spilling backup configs

This prevents wasting GPU time benchmarking configs that are known to be slow (the
benchmark would return a bad time, and the config would never be selected anyway).

### Why this matters for 2-D kernels

2-D kernels have larger tile volumes (`XBLOCK × YBLOCK` elements per block).  Larger
tiles require more unrolled loop variables, which need more registers.  Combined with
high `num_warps` (which reduces the per-thread register budget), 2-D configs are the
most common cause of register spills.

Without the spill recording fix in `bench()`, if all top-5 predicted configs for a 2-D
kernel spilled, the validation summary would print nothing — the silent failure bug.
Now, spilled configs are recorded with a timing of `infinity` (meaning "effectively
unusable"), and the validation summary reports them explicitly.


# AMD Reduction Kernel Heuristics

Heuristics for Triton **reduction** kernels on AMD CDNA GPUs (MI200 / MI300 series).

Controlled by two environment variables:

| Variable | Default | Effect |
|---|---|---|
| `TORCHINDUCTOR_REDUCTION_HEURISTICS=1` | 0 | Enable AMD reduction heuristics (config generation **+** scoring) |
| `TORCHINDUCTOR_HEURISTICS_REAL_BENCH=1` | 0 | Compile & time all top-N candidates; compare predicted vs. actual winner |
| `TORCHINDUCTOR_HEURISTICS_VERBOSE=1` | 0 | Print scoring table + result summary to stdout |
| `TORCHINDUCTOR_HEURISTICS_TOP_N=5` | 5 | Size of the selection pool (benchmarked configs) |
| `TORCHINDUCTOR_POINTWISE_WAVES_PER_EU=1` | 0 | Also generate `waves_per_eu=2` variants |

The heuristics are **AMD-only** and a no-op on CUDA.  They are also disabled
when `max_autotune=True` (the autotuner already benchmarks everything).

---

## AMD Reduction Execution Model

Triton generates reduction kernels with the following structure:

```python
# INNER reduction (reduce across r, output one value per x)
accumulator = tl.full([XBLOCK, R0_BLOCK], 0, dtype)
for r0_offset in tl.range(0, r0_numel, R0_BLOCK):   # outer loop
    ... accumulate into accumulator[x_idx, r0_idx] ...
result = tl.sum(accumulator, axis=1)[:, None]         # reduce call

# OUTER reduction (reduce across r, many x outputs kept)
for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
    partial = input[x_idx, r0_idx:r0_idx+R0_BLOCK]
    output[x_idx] += tl.sum(partial, axis=1)
```

### Hardware reduction path on AMD (CDNA)

`tl.sum(accumulator, axis=1)` is compiled to:

1. **Intra-wavefront DPP butterfly** (~24 cycles, 6 rounds × 4 cycles each).
   Purely register-based — no memory traffic.  Fast and fixed-cost regardless
   of wavefront count.

2. **Cross-wavefront DS (LDS) sync** — only needed when multiple wavefronts
   share one output element (`num_warps / XBLOCK > 1`).  DS bandwidth on
   MI300X is ~100 TB/s; the barrier overhead is ~100 cycles per round.

   ```
   ds_rounds = max(0, log₂(num_warps / XBLOCK))
   ```

3. No DS sync is needed when `num_warps ≤ XBLOCK` (each output element has
   ≤ 1 wavefront, so DPP alone is sufficient).

---

## Tuning Axes

| Parameter | Meaning | Trade-off |
|---|---|---|
| `XBLOCK` | Output elements per block; `grid = ⌈xnumel / XBLOCK⌉` | Larger → fewer blocks (fewer CUs busy), but sub-wavefront parallelism per output |
| `num_warps` | Wavefronts per block (× warp_size = thread count) | More → higher HBM BW per block + more DS rounds |
| `R0_BLOCK` | r-elements processed per outer-loop iteration | Larger → fewer iterations (less loop overhead), more registers |
| `waves_per_eu` | AMD scheduler hint (co-reside N wavefronts per SIMD) | Helps HBM latency hiding when blocks are few; hurts VGPR budget |

---

## Scoring Model

Each candidate config is scored on **four factors** combined via a weighted
geometric mean:

```
score = (A^a × B^b × C^c × D^d)^(1/(a+b+c+d))
```

All factors are in `[0.65, 1.00]`; the composite score is in `[0.0, 1.00]`.

### Factor A — R-loop Efficiency (`r_efficiency`)

**What it models**: How well `R0_BLOCK` amortizes the fixed per-iteration
DPP + DS overhead.

```
optimal_r0 = min(rnumel, max_r0_block)      # 2048 normally, 1024 for register-intensive
           = min(rnumel, max_r0_block // 2)  # register_intensive=True (Welford, var, std)

if r0_block >= optimal_r0:
    score = 1.00
else:
    score = 0.65 + 0.35 × log₂(r0_block) / log₂(optimal_r0)
```

*Intuition*: Each outer-loop iteration costs ~24 cycles (DPP) + `ds_rounds × 100`
cycles (DS) regardless of how many r-elements `R0_BLOCK` covers.  Larger
`R0_BLOCK` = fewer iterations = less overhead per r-element.

### Factor B — Grid Coverage (`grid_coverage`)

**What it models**: How many Compute Units (CUs) are actively doing work.

```
num_blocks = ⌈xnumel / XBLOCK⌉
min_blocks = adaptive threshold (see below)

score = Gaussian(num_blocks, centre=min_blocks, σ = min_blocks × 0.5)
      ∈ [0.70, 1.00]
```

The adaptive `min_blocks` threshold scales with `xnumel` and `num_cus`:

| xnumel | min_blocks |
|---|---|
| ≥ 8 × num_cus (≥ 2048 on MI300X) | num_cus ÷ 2 (128) — target ≥ 50% CU coverage |
| ≥ 2 × num_cus | num_cus ÷ 4 (64) — target ≥ 25% |
| < 2 × num_cus | num_cus ÷ 8 (32) — best possible for small problems |
| any | clamped to xnumel (can't have more blocks than outputs) |

*Intuition*: For INNER reductions, hitting exactly `min_blocks` is ideal —
more blocks = smaller `XBLOCK` = sub-optimal R-reduction parallelism without
adding useful output-element parallelism.

### Factor C (INNER) — DS Sync Overhead (`sync_overhead`)

**What it models**: The cost of cross-wavefront DS barriers relative to HBM
work per outer-loop iteration.

```
wavefronts_per_output = num_warps / XBLOCK
ds_rounds = max(0, log₂(wavefronts_per_output))

ds_cost   ≈ ds_rounds × 100 cycles     (barrier + writeback)
hbm_cost  ≈ R0_BLOCK × 4B / 1.8 B/cycle  (MI300X per-CU HBM rate)

overhead_ratio = ds_cost / (ds_cost + hbm_cost)
score = 1.0 − 0.30 × overhead_ratio    ∈ [0.70, 1.00]
```

*Intuition*: DS bandwidth is ~100 TB/s on MI300X — extremely fast — but the
barrier itself has latency.  For small `R0_BLOCK` (few HBM loads per
iteration), the DS barrier is a large fraction of per-iteration time.  For
large `R0_BLOCK` (many HBM loads), DS cost is negligible.

**DPP-only configs** (`num_warps ≤ XBLOCK`) score 1.00 on this factor.

### Factor C (OUTER) — Coalescing (`outer_coalescing`)

**What it models**: Cache-line utilisation for the output writes.

```
cache_line_elems = 16   (64 bytes / 4 bytes per FP32)

if XBLOCK >= 16:  score = 1.00   (full cache-line writes)
else:             score = 0.75 + 0.25 × (XBLOCK / 16)
```

### Factor D — Warp Parallelism (`warp_parallelism`)

**What it models**: `num_warps` vs. the AMD sweet-spot for HBM latency hiding.

```
sweet_min = 4,  sweet_max = 8   (from ArchitectureConfig)
           (= 4, = 4 for register_intensive kernels)

if sweet_min ≤ num_warps ≤ sweet_max:  score = 1.00
elif num_warps < sweet_min:            score = 0.82 + 0.18 × (nw / sweet_min)
else:                                  score = 0.90 − 0.10 × excess_frac

# Idle-thread guard: if threads_per_output > rnumel, many lanes are idle
if threads_per_output > rnumel:
    score = 1.00 − 0.30 × idle_fraction   ∈ [0.70, 1.00]
```

*Intuition*: AMD CDNA Little's Law requires ~8 wavefronts per CU to hide
~300-cycle HBM latency.  4 warps (256 threads) fills the HBM pipeline well
for most reduction kernels; 8 warps is beneficial for very large reductions.

### Adaptive Weights (INNER only)

The four factor exponents are interpolated from the bottleneck regime using
the same `BottleneckAnalysis` as pointwise kernels, but with reduction-specific
`problem_metadata`:

| Factor | overhead-bound | memory-bound | compute-bound |
|---|---|---|---|
| A — r_efficiency | 0.15 | **0.40** | 0.35 |
| B — grid_coverage | **0.35** | 0.20 | 0.15 |
| C — sync_overhead | **0.30** | 0.20 | 0.15 |
| D — warp_parallelism | 0.20 | 0.20 | **0.35** |

*Why these ratios?*
- **Overhead-bound** (tiny `xnumel × rnumel`): grid coverage and DS overhead
  dominate — minimising the number of DS rounds (or CUs starved) matters most.
- **Memory-bound** (large `rnumel`): `r_efficiency` dominates — amortising loop
  overhead over large HBM loads is the key.
- **Compute-bound** (reduction of transcendentals): `warp_parallelism`
  dominates — more wavefronts keep the ALU pipeline full.

`BottleneckAnalysis.analyze_bottleneck()` is called with:
- `total_elements = xnumel`  (output elements)
- `bytes_per_element = rnumel × elem_size`  (bytes loaded per output element)
- `ops_per_element = rnumel`  (reduction ops per output)

This correctly classifies tiny reductions as overhead-bound (dispatch ≫ HBM)
and large ones as memory-bound.

**OUTER** reductions use fixed weights:

| Factor | weight |
|---|---|
| B — grid_coverage | **0.50** |
| C — outer_coalescing | 0.20 |
| A — r_efficiency | 0.15 |
| D — warp_parallelism | 0.15 |

---

## Candidate Generation

### INNER reductions

```
XBLOCK grid:  1, 2, 4, …, max_xb   (adaptive, see below)
num_warps:    {1, 2, 4, 8}           (halved cap if register_intensive)
R0_BLOCK:     {r_full, r_half}       (+ r_quarter if register_intensive)
              + r_large, r_xlarge    (if rnumel ≥ max_r0_block × 2 or × 4)
waves_per_eu: wpe=2 variants         (if TORCHINDUCTOR_POINTWISE_WAVES_PER_EU=1)
```

**Adaptive XBLOCK ceiling**:

```
min_blocks = adaptive CU-coverage floor  (see Grid Coverage above)
max_xb_raw = xnumel // min_blocks
max_xb     = floor_pow2(max_xb_raw), capped at 1024
```

Example XBLOCK grids (MI300X, 256 CUs):

| xnumel | min_blocks | max_xb | XBLOCK grid |
|---|---|---|---|
| 8 | 1 | 8 | [1, 2, 4, 8] |
| 256 | 32 | 8 | [1, 2, 4, 8] |
| 2 048 | 64 | 32 | [1, 2, 4, 8, 16, 32] |
| 65 536 | 128 | 512 | [1, 2, 4, 8, …, 512] |
| 1 M | 512 | 1 024 | [1, 2, 4, 8, …, 1 024] |

**Filters applied to each candidate**:

1. `⌈xnumel / XBLOCK⌉ ≥ min_blocks(xnumel)`  (block-count guard)
2. `rnumel / (num_warps × warp_size / XBLOCK) ≥ 1`  (work-per-thread guard)

### OUTER reductions

```
XBLOCK:   derived from target CU coverage levels (200%, 100%, 50%, 25%, min)
          xb ≈ xnumel / target_blocks, rounded up to next power of 2
num_warps: {8, 4, 2}
R0_BLOCK: {min(rnumel, 8), min(rnumel, 4)}   (tiny to keep inner loop tight)
```

---

## Top-N Selection Flow

```
[_reduction_configs()]          ─→  10–50 candidate Config objects
       │
       ▼
[CachingAutotuner.__init__]
  _score_and_prune_reduction_configs():
    • scores each config (4 factors, adaptive weights)
    • sorts descending by score (R0_BLOCK tiebreaker)
    • diversity cap: ≤ 2 configs per XBLOCK value
    • stores heuristic top-N in _TOP_N_CONFIGS_FOR_SELECTION
    │
    ├── REAL_BENCH=0  →  self.configs pruned to top-N  (only those compiled)
    └── REAL_BENCH=1  →  self.configs keeps top-10     (all benchmarked)
       │
       ▼
[CachingAutotuner.autotune_to_one_config]
  benchmark_all_configs()
  winner = best from _TOP_N_CONFIGS_FOR_SELECTION   (REAL_BENCH=1)
         = best from self.configs                   (REAL_BENCH=0)
```

---

## Verbose Output Example

```
[REDUCTION] Scoring table – 18 configs | hint=INNER | x=8 r=8192
   #   XB     R0B    NW    r_eff    grid  sync/coa    warp   score  top-N
--------------------------------------------------------------------------------
  #1     1    2048     4   1.0000  1.0000    0.9874   1.0000  0.9963  ✓
  #2     1    2048     8   1.0000  1.0000    0.9813   1.0000  0.9945  ✓
  #3     1    1024     4   0.9545  1.0000    0.9619   1.0000  0.9715  ✓
  #4     1    2048     2   1.0000  1.0000    0.9939   0.9250  0.9791  ✓
  #5     2    2048     4   1.0000  0.8308    1.0000   1.0000  0.9391  ✓
  ...
```

---

## Comparison Against max_autotune

With `REAL_BENCH=1`, the verbose output includes:

```
[REDUCTION RESULT] hint=INNER | x=8 r=8192 | 6 configs benchmarked
  #1   XBLOCK=1 R0B=2048 nw=4   1.234 µs  ← HEURISTIC CHOSEN ★ ACTUAL BEST ✓
  #2   XBLOCK=1 R0B=2048 nw=8   1.289 µs
  #3   XBLOCK=1 R0B=1024 nw=4   1.456 µs
  ...
  Heuristic chose rank #1 (score 0.9963) — optimal ✓  (Δ = 0.000 µs, +0.0%)
```

If the heuristic picks a sub-optimal config:
```
  Heuristic chose rank #3 (score 0.9715) — suboptimal  (Δ = +0.222 µs, +18.0%)
```

---

## Register-Intensive Kernels

Kernels using multi-accumulator patterns (Welford online variance, `var`,
`std`) carry more live state per thread and are flagged as
`register_intensive=True` by the codegen.  The heuristics respond with:

1. **Lower `num_warps` cap**: `nw_cap = max_nw // 2 = 4` instead of 8.
   Fewer threads per CU → more VGPRs per thread → compiler can avoid spilling
   the accumulators.

2. **Smaller optimal `R0_BLOCK`**: `optimal_r0 = max_r0_block // 2`.
   Smaller tile means fewer live accumulators per iteration.

3. **Additional `r_quarter` tier**: `R0_BLOCK = max_r0_block // 4` is also
   generated as a safety-net candidate for kernels under extreme VGPR
   pressure.

4. **`waves_per_eu` disabled** for register-intensive kernels (the tighter
   VGPR budget from `wpe > 1` would cause systematic spills).

---

## Multi-RBLOCK Reductions

When `rnumel` is very large, Inductor may decompose the reduction into
multiple nested R-loops (R0, R1, …).  The heuristics currently target the
**primary R0 axis** only.  `_get_nd_reduction_numels()` in
`triton_heuristics.py` handles the multi-dimensional decomposition and
clamps each `Rn_BLOCK` independently.

`dynamic_scale_rblock` (always-on upstream mechanism) halves the largest
`Rn_BLOCK` after compilation if the kernel exceeds the VGPR ceiling.  On AMD
this rarely triggers (131 072 VGPRs per CU vs 65 536 on NVIDIA), so the
heuristics do not pre-emptively apply it.

---

## Quick Reference

```bash
# Enable and benchmark all top-5 candidates, print scoring table:
TORCHINDUCTOR_REDUCTION_HEURISTICS=1 \
TORCHINDUCTOR_HEURISTICS_REAL_BENCH=1 \
TORCHINDUCTOR_HEURISTICS_VERBOSE=1 \
TORCHINDUCTOR_HEURISTICS_TOP_N=5 \
HIP_VISIBLE_DEVICES=0 \
python benchmark_reduction.py 2>&1 | tee reduction_bench.log

# Heuristics-only (prune to top-5 before compilation, no benchmarking overhead):
TORCHINDUCTOR_REDUCTION_HEURISTICS=1 \
TORCHINDUCTOR_HEURISTICS_TOP_N=5 \
HIP_VISIBLE_DEVICES=0 \
python benchmark_reduction.py

# Compare against max_autotune baseline:
python3 compare_logs.py --summary --no-csv max_auto.log reduction_heuristics.log
```


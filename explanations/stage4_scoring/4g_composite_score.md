# Stage 4g — Composite Score: Bringing It All Together

**Source:** `score_config()` in `triton_heuristics_pointwise.py`

At this point we have:
- **Four factor scores** from Stages 4b–4e: each in `[0.70, 1.00]`
- **Adaptive weights** from Stage 3: reflecting the kernel's bottleneck (memory, compute, or overhead-bound)

This stage combines them into a single number used to rank every candidate config.

---

## What we have so far

### From Stage 2 — problem metadata (computed once, kernel-wide)

Stage 2 extracts the physical properties of the problem that all later stages
depend on.  Its key outputs used here are:

```
total_elements      – total tensor size (drives EPB, num_blocks calculations)
num_args            – number of tensor arguments (feeds launch overhead estimate)
bytes_per_element   – dtype width (feeds time_per_element for EPB target)
peak_bandwidth      – queried from device (feeds optimal_threads, time_per_element)
num_cus             – queried from device (feeds grid target = 2 × num_CUs)
```

Without Stage 2, Stages 3 and 4 have no hardware or problem context to work with.
Stage 2 runs **once per kernel shape** — not per config.

---

### From Stage 3 — bottleneck fractions (computed once, kernel-wide)

Stage 3 uses Stage 2's metadata to estimate where time is spent.
It runs **once per kernel**, before any config is scored.

```
T_overhead  – fraction of runtime in kernel dispatch        e.g. 0.35
T_memory    – fraction of runtime waiting for HBM data      e.g. 0.60
T_compute   – fraction of runtime in ALU compute            e.g. 0.05
                                                  sum = 1.0
```

Stage 3 also produces three **fixed regime weight tables** — hand-tuned constants
that encode: *"if a kernel is purely X-bound, which scoring factor matters most?"*

```
                   BW     Launch   Grid    Occ
  OVERHEAD_W  =  [ 0.10,  0.57,   0.18,   0.15 ]   ← dispatch cost dominates → launch score matters
  MEMORY_W    =  [ 0.55,  0.10,   0.15,   0.20 ]   ← HBM dominates → bandwidth score matters
  COMPUTE_W   =  [ 0.15,  0.10,   0.30,   0.45 ]   ← ALU dominates → occupancy/grid matters
```

These tables are constants in the code — they do not change per kernel or per config.
What changes is how much of each table is used, determined by `T_overhead / T_memory / T_compute`.

---

### From Stage 4 — four factor scores (computed per config)

| Stage | Score | Measures |
|---|---|---|
| 4b | `BW` | Are threads_per_block close to the memory-optimal count? |
| 4c | `Launch` | Does each block do enough work to amortise its 3 µs startup? |
| 4d | `Grid` | Does num_blocks match the GPU's CU count? |
| 4e | `Occupancy` | Does num_warps give enough wavefronts for latency hiding? |

Each score uses Stage 2 metadata (total_elements, num_cus, etc.) as reference points.

---

## Step 1 — Blend the regime weights → W_bw, W_launch, W_grid, W_occ

The three Stage 3 fractions decide how much of each regime table to use.
For each factor, the blended weight is:

```
W_factor = T_overhead × OVERHEAD_W[factor]
         + T_memory   × MEMORY_W[factor]
         + T_compute  × COMPUTE_W[factor]
```

This is a weighted average of the three table rows — nothing more.
`T_overhead=0.60` means "use 60% of the overhead-bound row for each factor".

Example: a kernel that is 35% overhead-bound, 60% memory-bound, 5% compute-bound:

```
W_bw     = 0.35×0.10 + 0.60×0.55 + 0.05×0.15 = 0.035 + 0.330 + 0.008 = 0.373
W_launch = 0.35×0.57 + 0.60×0.10 + 0.05×0.10 = 0.200 + 0.060 + 0.005 = 0.265
W_grid   = 0.35×0.18 + 0.60×0.15 + 0.05×0.30 = 0.063 + 0.090 + 0.015 = 0.168
W_occ    = 0.35×0.15 + 0.60×0.20 + 0.05×0.45 = 0.053 + 0.120 + 0.023 = 0.196
```

The blended weights reflect which factor actually matters most for this specific kernel.

---

## Step 2 — Map blended weights to exponents → a, b, c, d

The blended weights are numbers like `0.373`.  They need to become exponents
`a, b, c, d` in the final formula.  The mapping is a simple linear scale:

```
exponent = 0.5 + (weight − 0.10) / 0.40 × 2.5
           clamped to [0.5, 3.0]
```

Anchored at:
- `weight = 0.10` → `exponent = 0.5`  (this factor barely influences ranking)
- `weight = 0.50` → `exponent = 3.0`  (this factor dominates ranking)

**This is the only step that produces a, b, c, d.**  They come entirely from the
blended weights, which come from the regime tables + Stage 3 fractions.

Using the blended weights from the example above:

| Factor | Blended weight | Exponent |
|---|---|---|
| BW (a) | 0.373 | `0.5 + (0.373−0.10)/0.40 × 2.5 = 2.21` |
| Launch (b) | 0.265 | `0.5 + (0.265−0.10)/0.40 × 2.5 = 1.53` |
| Grid (c) | 0.168 | `0.5 + (0.168−0.10)/0.40 × 2.5 = 0.93` |
| Occupancy (d) | 0.196 | `0.5 + (0.196−0.10)/0.40 × 2.5 = 1.10` |

**Why exponents rather than direct weights?**

A weighted geometric mean via exponents has a key property: a factor with a low
score *cannot* be fully compensated by other factors being perfect.  Compare:

```
Arithmetic mean (additive):  0.25×0.70 + 0.25×1.00 + 0.25×1.00 + 0.25×1.00 = 0.925
Geometric mean (multiplicative): (0.70 × 1.00 × 1.00 × 1.00)^(1/4)         = 0.915
Geometric mean with high BW exp: (0.70^2.21 × 1.00^1.53 × ...)^(1/5.77)    = 0.878
```

The higher the exponent, the more a low score on that factor drags the composite down.
This reflects physical reality: for a memory-bound kernel, getting BW wrong by 30% costs
~30% wall time regardless of how well the other factors score.

---

## Step 3 — Weighted geometric mean

```
total_exp = a + b + c + d

score = (BW^a × Launch^b × Grid^c × Occupancy^d) ^ (1 / total_exp)
```

The `^ (1/total_exp)` normalises the result back to `[0, 1]` so scores are comparable
across different kernels with different exponent sets.

Continuing the example, assume factor scores: BW=0.92, Launch=0.88, Grid=0.97, Occ=0.85:

```
total_exp = 2.21 + 1.53 + 0.93 + 1.10 = 5.77

score = (0.92^2.21 × 0.88^1.53 × 0.97^0.93 × 0.85^1.10) ^ (1/5.77)
      = (0.831 × 0.824 × 0.971 × 0.836) ^ 0.173
      = (0.556) ^ 0.173
      ≈ 0.899
```

---

## Step 4 — 2-D tie-breaker (if applicable)

For 2-D configs (`XBLOCK × YBLOCK`), two small multipliers break ties between
otherwise equal scores:

```
ratio              = max(XBLOCK, YBLOCK) / min(XBLOCK, YBLOCK)
balance_multiplier = 1.0 − 0.005 × log2(ratio)
  → prefers square tiles (elongated tiles misalign the cache line prefetcher)

innermost_multiplier = 1.0 − 0.008 × max(0, 5 − log2(XBLOCK))
  → prefers wider X (XBLOCK ≥ 32 scores 1.00; small XBLOCK hurts coalescing)

score × = max(0.95, balance_multiplier × innermost_multiplier)
```

Example: `XBLOCK=64, YBLOCK=8`:
```
ratio              = 64/8 = 8 → balance = 1.0 − 0.005×3 = 0.985
innermost          = 1.0 − 0.008×max(0, 5−6) = 1.0
final multiplier   = max(0.95, 0.985) = 0.985
```

The multiplier is capped at `0.95` minimum so it can never eliminate an otherwise
good 2-D config — it only resolves ties.

---

## Fallback — when Stage 3 is unavailable

If the bottleneck analysis fails or is disabled, fixed exponents are used:

```
bw_exp=2.5, launch_exp=1.8, grid_exp=1.2, occ_exp=0.6
```

These approximate a "mostly memory-bound" kernel, which is the most common case
for pointwise operations.

---

## Summary — the full pipeline in one view

```
Stage 2  →  total_elements, num_cus, bytes_per_element, peak_bandwidth, num_args
              │  (hardware + problem constants used by all later stages)
              │
Stage 3  →  T_overhead, T_memory, T_compute          (where does time go?)
              │
              ├─ × OVERHEAD_W table  ┐
              ├─ × MEMORY_W table    ├→  W_bw, W_launch, W_grid, W_occ
              └─ × COMPUTE_W table  ┘
                    │
                    └→  a, b, c, d    (linear map: weight → exponent)

                         ↑ sets how much each score matters
                         │
Stage 4b →  BW        ∈ [0.70, 1.00]  ┐
Stage 4c →  Launch    ∈ [0.70, 1.00]  │
Stage 4d →  Grid      ∈ [0.70, 1.00]  ├→  score = (BW^a × Launch^b × Grid^c × Occ^d)^(1/Σ)
Stage 4e →  Occupancy ∈ [0.50, 1.00]  ┘
Stage 4f →  2-D tie-breaker ×multiplier

→  composite score ∈ [0, 1]  per config
→  all configs sorted descending
→  top-N retained for compilation
```

**One-line summary:**
> Stage 2 provides the numbers. Stage 3 decides what matters. Stage 4 measures how well each config delivers it.


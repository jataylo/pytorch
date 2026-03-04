# Stage 4a — Scoring Overview: Weighted Geometric Mean

**Source:** `_score_one_config()` in `triton_heuristics_pointwise.py`, called via
`ThreadPoolExecutor` in `_score_and_prune_heuristic_configs()`

**Role:** Combines four factor scores (Bandwidth, Launch, Grid, Occupancy) into a single
composite score for each candidate config.  The combination uses a weighted geometric mean,
where the weights are the per-config adaptive weights produced by Stage 3e.  Configs are
sorted descending by this score and the top-N are retained.

---

## The Four Factors — Why Each Exists

### 4b — Bandwidth (Memory Bus Utilisation)

**What it measures:** Does this config keep the HBM bus fully loaded?

The single biggest lever for a pointwise kernel's throughput is how efficiently threads
issue memory transactions.  On MI300X HBM delivers ~5.2 TB/s peak; leaving it idle even
30% of the time costs 30% of peak throughput.

The key variable is **threads per block** (`num_warps × 64`).  Too few threads → the bus
sits idle between wavefronts.  Too many threads → threads compete for the same cache lines,
causing serialisation.  The bandwidth score is a Gaussian centred on the empirically good
thread count for the problem's element count:

```
BW_score = exp( −((threads − target_threads)² / (2σ²)) )
```

This factor is the highest-weighted for memory-bound kernels (the common case for
pointwise) because getting the thread count wrong costs the most wall time.

**Relevant for:** All kernels.  Weight is highest when Stage 3 identifies a memory-bound
bottleneck.

---

### 4c — Launch Overhead (Work-Per-Block Efficiency)

**What it measures:** Does each thread block do enough work to pay for the cost of
launching it?

Every block incurs a fixed startup cost: the SPI (Shader Processor Input) must allocate
VGPRs, LDS, and wavefront slots — roughly 2–5 μs per kernel invocation on MI300X.
If a block processes only 64 elements, that fixed cost dominates the useful work.

The scoring uses two regimes:

- **Single-block kernels** (tiny problems): score is a Gaussian over elements-per-block
  (EPB), peaked at a target EPB that matches the overhead budget.
- **Multi-block kernels**: overhead is paid **once per kernel**, not per block.  The score
  is kept near-flat across a wide EPB range (wide-sigma Gaussian) so that it does not
  incorrectly favour the largest possible XBLOCK.  Block-count optimisation is delegated
  to the Grid score (4d), which does it more accurately.

**Relevant for:** Small kernels where overhead_frac > 0.3 (see Stage 3a).  For large
streaming kernels, weight is low and the score is near-1.0 for almost all configs.

---

### 4d — Grid Granularity (CU Saturation)

**What it measures:** Does the number of blocks match the number of Compute Units?

The GPU has `num_cus` CUs (256 on MI300X).  Peak throughput requires every CU to be
busy.  The grid score penalises two failure modes:

1. **Too few blocks** (< num_cus): some CUs sit completely idle.  A kernel with 128
   blocks on 256 CUs wastes exactly 50% of the hardware.
2. **Fractional blocks** (non-integer multiple of num_cus): the last "wave" of blocks
   is smaller than the rest, causing some CUs to finish early and wait.  E.g. 257 blocks
   on 256 CUs means one CU gets 2 blocks while 255 get 1 — the total time is determined
   by the 2-block CU.

For tiny kernels (≤ 8192 elements) a discrete lookup table is used instead of a
continuous formula because the continuous model over-penalises the correct 1–4-block
configs in that regime.

**Relevant for:** All kernels.  Weight is highest in the overhead-bound regime, because
for tiny kernels the grid size is the most consequential tuning knob.

---

### 4e — Occupancy (Wavefront Latency Hiding)

**What it measures:** Are enough wavefronts resident per CU to hide HBM round-trip
latency?

HBM latency is ~300 cycles.  The GPU hides this by switching to a different wavefront
while the first one waits for its load to return (Little's Law).  To hide latency fully,
a CU needs ≥ 8 independent wavefronts resident simultaneously.

The wavefront supply comes from two sources:

- **Block-level switching:** if many blocks are dispatched (`num_blocks ≥ num_cus × 8`)
  the CU scheduler can switch between different blocks, providing latency hiding even with
  `num_warps=1` per block.
- **Intra-block switching:** if blocks are scarce (`wf_per_cu ≤ 8`), the wavefronts
  *within* each block (controlled by `num_warps`) become the only source of CU-level
  switching.  In this regime `num_warps=1` leaves the CU starved.

An ILP correction (elements-per-thread ≥ 8) applies when the compiler can software-
pipeline loads across loop iterations, providing per-thread latency hiding that partially
substitutes for extra wavefronts — but only when block-level switching is already
adequate (`need_intra_block_warps = False`).

**Relevant for:** All kernels, but the scoring formula switches regime based on
`saturation = (num_blocks × num_warps) / (num_cus × 8)`.  Weight is highest in the
memory-bound and compute-bound regimes.

---

## Technical

### Parallelism

Scoring runs in parallel across all candidate configs:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=min(len(configs), 32)) as pool:
    futures = {pool.submit(_score_one_config, cfg): cfg for cfg in configs}
    for fut in as_completed(futures):
        result = fut.result()
        if result is not None:
            scored.append(result)
```

Up to 32 worker threads run simultaneously, each scoring one config.  For 80 configs on
an 8-core machine, this takes ~1–2 ms total rather than ~10–20 ms sequential.  Each
`_score_one_config` call is CPU-bound (no GPU interaction) and GIL-releasing-safe because
it uses only pure Python math operations.

### Composite score formula — weighted geometric mean

```
score = (BW^a × Launch^b × Grid^c × Occ^d) ^ (1 / (a + b + c + d))
```

Where `{a, b, c, d}` are exponents derived from the adaptive weights via the linear map:

```
exp[k] = 0.5 + (w[k] − 0.10) / 0.40 × 2.5     clamped to [0.5, 3.0]
```

**Why this map?**

The weight range is `[0.10, 0.50]` (weights sum to 1.0 across 4 factors, so the average
is 0.25; no factor gets below ~0.10 due to the baseline contributions in the weight
tables).  The exponent range `[0.5, 3.0]` gives a 6× spread:

| w[k] | Exponent | Geometric mean contribution |
|---|---|---|
| 0.10 (factor nearly irrelevant) | 0.50 | score^0.5 — square root: barely penalising |
| 0.25 (average weight) | 1.44 | score^1.44 — moderate influence |
| 0.50 (factor dominates) | 3.00 | score^3.0 — cubic: heavy penalisation |

At exponent 3.0, a factor score of 0.80 contributes `0.80^3 = 0.512` — a 49% reduction.
The same 0.80 at exponent 0.5 contributes `0.80^0.5 = 0.894` — only an 11% reduction.
This spread ensures that configs that fail hard on the most important factor are ranked
far below configs that are merely sub-optimal.

### Normalisation

The exponent normalisation `1 / (a + b + c + d)` ensures the composite score stays in
`[0.0, 1.0]` when all factor scores are in `[0.0, 1.0]`.  Without normalisation, the
product of four values each raised to exponent ~1.5 would produce a composite near
`0.80^6 ≈ 0.26` even for excellent configs, compressing the useful range.

### Why geometric mean over arithmetic mean?

An arithmetic weighted mean `Σ w[k] × score[k]` is **compensatory** — a factor score of
0.40 can be compensated by three other factors scoring 1.0 to produce an average of 0.85.

A geometric mean `Π score[k]^w[k]` is **non-compensatory** — a score of 0.40 in one
factor pulls the total down regardless of other factors.  Specifically:
```
Geometric: 0.40^1.0 × 1.0^1.0 × 1.0^1.0 × 1.0^1.0 = 0.40
Arithmetic: 0.25×0.40 + 0.25×1.0 + 0.25×1.0 + 0.25×1.0 = 0.85
```

The geometric mean is the right choice because a config with a fundamentally broken thread
count (BW=0.40) genuinely *cannot* be saved by having an ideal block count.  The physics
of the memory bus doesn't allow compensation — poor thread count means poor bus utilisation,
period.

### Score range across the candidate pool

In practice, after pruning illegal configs in Stage 1, composite scores fall in:
- **Top-1 config:** 0.88–0.97 (near-ideal across all factors)
- **Middle of the pool:** 0.70–0.87 (one or two sub-optimal factors)
- **Bottom of pool:** 0.50–0.70 (significantly wrong in one or more factors)

The scoring table printed in verbose mode shows all ranked configs, so score distributions
can be inspected to understand which dimension is hurting each config.

### Output

`_score_one_config` returns a tuple `(composite_score, triton_cfg, effective_dict, detail_dict, bottleneck_dict)`:

- `composite_score`: float in `[0, 1]`, used for ranking
- `triton_cfg`: the original `triton.Config` object (for passing to `self.configs`)
- `effective_dict`: the resolved `{XBLOCK, YBLOCK, num_warps, …}` dict (for validation matching)
- `detail_dict`: `{memory_bandwidth, launch_overhead, grid_granularity, occupancy, num_blocks, threads_per_block}` — the four raw factor scores plus derived quantities for the verbose table
- `bottleneck_dict`: `{bottleneck, overhead_us, memory_us, compute_us, total_us, overhead_frac, memory_frac, compute_frac}` — Stage 3 output for the verbose table

---

## Simple explanation

### What this stage does

Stage 4 is the **ranking engine** — it takes all 30–200 candidate configs and assigns
each one a score between 0 and 1 representing how well it's predicted to perform.

Think of it like judging ice-skating: each skater is scored on four criteria (technical
content, presentation, artistry, execution), and the criteria are weighted based on what
kind of competition it is (some competitions weight technical higher, others artistic).

### The four criteria

| Criterion | Stage | What it measures |
|---|---|---|
| **Bandwidth** | 4b | Does this config use the right thread count to saturate the HBM bus? |
| **Launch overhead** | 4c | Does each block do enough work to amortise its startup cost? |
| **Grid granularity** | 4d | Does the number of blocks match the GPU's compute unit count? |
| **Occupancy** | 4e | Are enough wavefronts resident per CU to hide memory latency? |

### Adaptive weights

The weights on each criterion automatically adjust based on the bottleneck identified in
Stage 3.  For a tiny kernel where overhead dominates, the launch criterion gets ~4× more
weight than bandwidth.  For a large streaming kernel, bandwidth gets ~5× more weight than
launch.  This is why the same scoring formula gives sensible rankings for kernels of
wildly different sizes and operation mixes.

### Why multiply (geometric mean) not add (arithmetic mean)?

A config that's terrible at bandwidth can't be "saved" by being great at everything else.
Bad thread count means the memory bus is underutilised — that's physics, not something
you can compensate for by having a nice block count.

Geometric multiplication means: **one bad score pulls everything down**, which correctly
reflects how GPU performance works.  An arithmetic average would hide fatal flaws behind
good scores in other dimensions.

### Parallelism

All 30–200 configs are scored simultaneously using a thread pool.  On modern hardware this
takes ~1–2 ms total, making the entire heuristic cost negligible compared to even a single
Triton compilation (which takes 50–500 ms).


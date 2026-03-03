# Stage 4a — Scoring Overview: Weighted Geometric Mean

**Source:** `_score_one_config()` in `triton_heuristics_pointwise.py`, called via
`ThreadPoolExecutor` in `_score_and_prune_heuristic_configs()`

**Role:** Combines four factor scores (Bandwidth, Launch, Grid, Occupancy) into a single
composite score for each candidate config.  The combination uses a weighted geometric mean,
where the weights are the per-config adaptive weights produced by Stage 3e.  Configs are
sorted descending by this score and the top-N are retained.

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


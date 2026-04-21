# Stage 3e — Adaptive Weight Interpolation

**Source:** `BottleneckAnalysis.analyze_bottleneck()` in `triton_heuristics_adaptive.py`

**Role:** Converts the component fractions `{overhead_frac, memory_frac, compute_frac}`
into a single weight vector `{w_bandwidth, w_launch, w_grid, w_occupancy}` that controls
how much the four scoring factors matter for *this specific config*.  The weights are
per-config, not per-kernel — a large-block config and a small-block config for the same
kernel may receive completely different weight vectors.

---

## Technical

### Pure-regime weight tables

Three anchor weight vectors define the ideal weighting for each "pure" bottleneck regime:

| Factor | `W_overhead` (launch-bound) | `W_memory` (memory-bound) | `W_compute` (compute-bound) |
|---|---|---|---|
| bandwidth | 0.10 | **0.55** | 0.15 |
| launch | **0.57** | 0.10 | 0.10 |
| grid | **0.18** | 0.15 | 0.30 |
| occupancy | 0.15 | 0.20 | **0.45** |

**Interpretation of each regime:**

- **Launch-bound:** `launch` (0.57) and `grid` (0.18) share the dominant signal.
  Launch was reduced from 0.65 → 0.57 and Grid raised from 0.10 → 0.18 because
  the EPB-based Launch score is near-flat for multi-block kernels (wide sigma
  keeps all reasonable EPB in [0.88, 1.00]), so it barely discriminates configs.
  The Grid score (CU saturation) becomes the primary differentiator between e.g.
  XBLOCK=256 (high CU utilisation) and XBLOCK=1024 (low CU utilisation) for
  medium-sized overhead-bound problems.  `bandwidth` is low (0.10) because the
  memory bus is barely loaded when overhead is 70%+ of total runtime.

- **Memory-bound:** `bandwidth` dominates (0.55) because achieving the highest possible
  fraction of peak HBM bandwidth is the only lever that matters.  Grid granularity (0.15)
  and occupancy (0.20) are secondary because they affect latency hiding, which in turn
  affects bandwidth utilisation.

- **Compute-bound:** `occupancy` dominates (0.45) because sufficient wavefronts must be
  resident to pipeline the long-latency ALU instructions (especially SFU calls with
  16–64 cycle latency).  Grid granularity (0.30) is elevated because full CU saturation
  is important when the ALU is the bottleneck — a poorly-saturated grid wastes compute
  capacity.

### Linear interpolation equation

The actual weight for factor `k` is a weighted blend of the three pure-regime weights:

```
w[k] = overhead_frac × W_overhead[k]
     + memory_frac   × W_memory[k]
     + compute_frac  × W_compute[k]
```

Because `overhead_frac + memory_frac + compute_frac = 1.0`, the interpolated weights
also sum to 1.0:

```python
assert abs(sum(w.values()) - 1.0) < 1e-6   # always true by construction
```

### Worked example — mixed-regime config

A config with `overhead_frac=0.60, memory_frac=0.35, compute_frac=0.05`:

```
w_bandwidth = 0.60×0.10 + 0.35×0.55 + 0.05×0.15 = 0.06 + 0.193 + 0.008 = 0.261
w_launch    = 0.60×0.65 + 0.35×0.10 + 0.05×0.10 = 0.39 + 0.035 + 0.005 = 0.430
w_grid      = 0.60×0.10 + 0.35×0.15 + 0.05×0.30 = 0.06 + 0.053 + 0.015 = 0.128
w_occupancy = 0.60×0.15 + 0.35×0.20 + 0.05×0.45 = 0.09 + 0.070 + 0.023 = 0.183
              ────────────────────────────────────────────────────────────────────
              sum = 0.261 + 0.430 + 0.128 + 0.183 = 1.002 ≈ 1.0  ✓
```

This config is 60% launch-dominated → `w_launch = 0.43` is by far the largest weight.
The scoring formula in Stage 4 will heavily penalise configs with poor launch scores.

### Exponent mapping — from weight to geometric-mean exponent

The weighted geometric mean in Stage 4 requires exponents, not plain weights:

```
exp[k] = 0.5 + (w[k] − 0.10) / 0.40 × 2.5     clamped to [0.5, 3.0]
```

This linear map converts the weight range `[0.10, 0.50]` to the exponent range `[0.5, 3.0]`:

| Weight | Exponent | Effect in geometric mean |
|---|---|---|
| 0.10 (minimum) | 0.5 | Factor appears at 1/2 power — minimal influence |
| 0.25 (mid) | 1.4 | Moderate influence |
| 0.50 (maximum) | 3.0 | Factor appears at cube — high sensitivity |

Weights below 0.10 clamp to exp=0.5; weights above 0.50 clamp to exp=3.0.

**Why exponents instead of plain weights?**

In a weighted arithmetic mean `Σ w[k] × score[k]`, a bad score in one factor can be
compensated by exceptional scores in others.  A config with `BW=0.40, Launch=1.0,
Grid=1.0, Occ=1.0` gets arithmetic average 0.85 — nearly as high as a perfect config.

In a weighted geometric mean `Π score[k]^exp[k]`, a bad score is multiplicatively
penalising.  The same config gets geometric mean ≈ 0.40^0.5 × 1.0^... ≈ 0.63 — the poor
bandwidth score drags the total down significantly.  This reflects reality: a config with
a fundamentally wrong thread count cannot be "saved" by having good block count.

### `launch_bound` flag

```python
launch_bound = (overhead_frac > 0.50)
```

When `True`, this flag changes how Stage 4e (Occupancy) computes its score.  Specifically,
for launch-bound multi-block configs, the occupancy score uses the first-principles
overhead ratio `K_launch / T(nw)` rather than the wavefront sweet-spot Gaussian.  This
reflects that in a launch-dominated regime, extra wavefronts *hurt* (adding SPI overhead)
rather than helping (latency hiding has no value when overhead dominates).

---

## Simple explanation

### The problem this solves

Different kernels care about different things:

- A tiny kernel (512 elements) finishes in 1–2 µs.  The 3 µs startup overhead is the
  biggest problem.  We should score configs that *minimise the number of blocks* very
  highly, and care less about memory efficiency or occupancy.

- A large streaming kernel (100M elements) runs for 300+ µs.  The 3 µs startup is
  irrelevant.  We should score configs that *maximise HBM bandwidth utilisation* most
  heavily.

- A compute-heavy kernel (many `exp`, `sin`, fusions) needs lots of wavefronts to keep
  the ALU busy across multiple in-flight operations.  Occupancy is the key metric.

The adaptive weighting system makes **the same four scoring factors change their
relative importance automatically** based on what the bottleneck actually is.

### How it works

Think of it like a judge at a triathlon — they care about all three disciplines (swim,
bike, run), but how they weight them depends on the athlete's profile:

- An athlete who struggles on the swim leg → put more weight on swim improvement
- An athlete who bikes slowly → put more weight on cycling speed

The bottleneck fractions (`overhead_frac`, `memory_frac`, `compute_frac`) tell us the
athlete's profile.  The weight tables are the judge's rubric for each type.  The
interpolation blends them for athletes who struggle on multiple legs at once.

### A concrete example

| Config | overhead_frac | memory_frac | compute_frac | launch_weight | bandwidth_weight |
|---|---|---|---|---|---|
| `XBLOCK=16, nw=8` | 0.75 | 0.24 | 0.01 | **0.51** | 0.11 |
| `XBLOCK=512, nw=4` | 0.30 | 0.65 | 0.05 | 0.26 | **0.39** |

For the small-block config, `launch_weight = 0.51` dominates — the scoring stage will
heavily penalise its huge block count (16 blocks for 512-element problem).
For the large-block config, `bandwidth_weight = 0.39` dominates — bandwidth efficiency
is what matters for a well-sized streaming kernel.

The same kernel, same GPU, but the optimal weighting is completely different because the
two configs land in fundamentally different bottleneck regimes.


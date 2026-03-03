# Stage 3 — Bottleneck Analysis Overview

**Source:** `BottleneckAnalysis.analyze_bottleneck()` in `triton_heuristics_adaptive.py`

**Called:** Once per candidate config (30–120 times per kernel, in parallel via
`ThreadPoolExecutor` alongside Stage 4 scoring).

**Input:** `config` dict + `problem_metadata` dict from Stage 2
**Output:** A dict with nine fields driving Stage 4 scoring and verbose logging

---

## What Stage 3 does

Stage 2 produced a single `problem_metadata` snapshot of the kernel — its size, dtype,
op mix, broadcast/mask flags.  Stage 3 takes that snapshot and asks:

> *For this specific config (XBLOCK=256, num_warps=4, ...), how long does each component
> of execution actually take?*

It builds a **three-component time model** per config and derives the fractions
`{overhead_frac, memory_frac, compute_frac}` that tell Stage 4 which scoring factors
matter most for this particular config.

---

## Call flow inside `analyze_bottleneck()`

```
analyze_bottleneck(config, problem_metadata)
│
├─ [optional] extract_kernel_metadata(kernel_code)   ← refine metadata if source available
│
├─ get_block_dimensions(config)    → block_dims       e.g. (256,) or (64, 4)
├─ get_problem_dimensions(meta)    → problem_dims     e.g. (65536,) or (256, 256)
├─ calculate_grid_size(...)        → num_blocks       e.g. 256
│
├─ estimate_overhead_time_us(num_blocks, meta, config)   → T_overhead   [Stage 3a]
├─ estimate_memory_time_us(total_bytes, meta, num_blocks) → T_memory    [Stage 3b]
├─ estimate_compute_time_us(num_ops, threads, blocks, meta) → T_compute  [Stage 3c]
│
├─ T_total = T_overhead + max(T_memory, T_compute)         [Stage 3d — roofline]
│
├─ gross = T_overhead + T_memory + T_compute               [sum — preserves fractions]
├─ overhead_frac = T_overhead / gross
├─ memory_frac   = T_memory   / gross
├─ compute_frac  = T_compute  / gross
│
└─ → {overhead_us, memory_us, compute_us, total_us,
       overhead_frac, memory_frac, compute_frac,
       bottleneck, launch_bound}
```

---

## The five sub-stages

### 3a — `T_overhead` (kernel dispatch cost)

**What it models:** The fixed time the GPU spends *starting* the kernel before any thread
executes a single instruction.

**Key inputs:** `num_warps`, `num_blocks`, `num_tensors`, `has_mask`

```
T_overhead = 3.0 µs                          ← fixed CP dispatch cost (AMD empirical)
           + (n_args − 3) × 0.1 µs           ← extra tensor pointer SGPR loads
           + (num_warps − 1) × 0.2 µs        ← SPI wavefront bank allocation
           + grid_overhead(num_blocks)        ← CP batch dispatch when num_blocks > 1000
           + 0.5 µs  if has_mask             ← predicate setup cost
```

Config sensitivity: **high** — `num_warps` and `num_blocks` both change with config.
→ Full details: [`3a_overhead.md`](3a_overhead.md)

---

### 3b — `T_memory` (HBM transfer time)

**What it models:** How long it would take to move all data through the memory hierarchy
if compute were free.

**Key inputs:** `total_bytes`, `has_broadcast`, `has_mask`, `num_blocks` (for cache level)

```
T_memory = total_bytes / BW_effective

BW_effective = BW_peak × η_hbm × η_coalesce

  η_hbm      = 0.65   if has_mask  (partial cache-line stores break write-combining)
             = 0.80   otherwise     (clean streaming)
  η_coalesce = 1.00   for 1-D kernels
             = f(XBLOCK) for 2-D  (XBLOCK < 16 wastes cache-line fetches)
```

Cache shortcuts:
- `total_bytes ≤ L1 (32 KB)` and `num_blocks ≤ num_CUs` → L1 hit, ~0.01 µs
- Broadcast tensors that fit in L2 (16 MB) → L2 bandwidth (~1000 GB/s) instead of HBM

Config sensitivity: **medium** — `num_blocks` affects cache-level selection; `XBLOCK`
affects coalescing.
→ Full details: [`3b_memory.md`](3b_memory.md)

---

### 3c — `T_compute` (ALU throughput time)

**What it models:** How long it would take if memory bandwidth were infinite and only
the ALUs were the constraint.

**Key inputs:** `total_ops`, `ops_per_element`, `fast_ops / medium_ops / slow_ops` mix

```
T_compute = total_ops / (TFLOPS_peak × η_instr)

  total_ops = total_elements × ops_per_element

  η_instr = 0.60   if slow_frac > 0.50   (SFU stalls FMA pipeline)
           = 0.70   if medium_frac > 0.30 (mixed — some FMA stalls)
           = 0.80   otherwise              (mostly FMA/add — pipeline stays full)
```

Config sensitivity: **low** — `total_ops` is fixed; only `threads_per_block × num_blocks`
(≈ `total_elements`) changes, which cancels out in the division.
→ Full details: [`3c_compute.md`](3c_compute.md)

---

### 3d — Roofline combination

**What it models:** The actual wall-clock execution time, accounting for the fact that
memory and compute pipelines run in parallel.

```
T_total = T_overhead + max(T_memory, T_compute)
```

`max()` because the GPU's HBM controllers and SIMD ALU arrays are **independent
pipelines** — they run simultaneously.  Only the slower of the two is visible on the
wall clock; the faster is hidden.  Overhead does NOT overlap — the kernel dispatch must
finish before any wavefront starts.

Fractions are computed from the **gross sum** (not `T_total`) so they always sum to 1.0
even when memory and compute strongly overlap:

```
gross        = T_overhead + T_memory + T_compute   ← sum of all three
overhead_frac = T_overhead / gross
memory_frac   = T_memory   / gross
compute_frac  = T_compute  / gross
```

Bottleneck label:
- `overhead_frac > 0.50` → `bottleneck = 'overhead'`, `launch_bound = True`
- else `memory_us ≥ compute_us` → `bottleneck = 'memory'`
- else → `bottleneck = 'compute'`

→ Full details: [`3d_roofline.md`](3d_roofline.md)

---

### 3e — Adaptive weight interpolation

**What it models:** Converts the three fractions into a weight vector for Stage 4
scoring so that the right scoring factors dominate for each config's bottleneck.

```
w[k] = overhead_frac × W_overhead[k]
     + memory_frac   × W_memory[k]
     + compute_frac  × W_compute[k]
```

| Factor | `W_overhead` | `W_memory` | `W_compute` |
|---|---|---|---|
| bandwidth | 0.10 | **0.55** | 0.15 |
| launch | **0.57** | 0.10 | 0.10 |
| grid | **0.18** | 0.15 | 0.30 |
| occupancy | 0.15 | 0.20 | **0.45** |

Example — a kernel with `overhead_frac=0.7, memory_frac=0.3`:
```
launch_weight = 0.7 × 0.57 + 0.3 × 0.10 = 0.429   ← largest weight: favour fewer blocks
grid_weight   = 0.7 × 0.18 + 0.3 × 0.15 = 0.171   ← secondary: CU saturation
```

→ Full details: [`3e_adaptive_weights.md`](3e_adaptive_weights.md)

---

## Output dict reference

| Key | Type | Meaning |
|---|---|---|
| `overhead_us` | float | `T_overhead` in microseconds |
| `memory_us` | float | `T_memory` in microseconds |
| `compute_us` | float | `T_compute` in microseconds |
| `total_us` | float | `T_overhead + max(T_memory, T_compute)` |
| `overhead_frac` | float | `T_overhead / gross`  (0–1) |
| `memory_frac` | float | `T_memory / gross`  (0–1) |
| `compute_frac` | float | `T_compute / gross`  (0–1) |
| `bottleneck` | str | `'overhead'` \| `'memory'` \| `'compute'` |
| `launch_bound` | bool | `True` when `overhead_frac > 0.50` |

---

## Why fractions instead of raw times for weighting

Stage 4 needs to know the *relative importance* of each bottleneck, not the absolute
time.  A 10 µs kernel and a 1000 µs kernel with the same `overhead_frac=0.3` should
receive the same scoring weights — both are 30% overhead-limited regardless of their
absolute size.  Using fractions makes the weight interpolation scale-invariant.

---

## Simple explanation

### What Stage 3 is doing

Think of a kernel's execution like a relay race with three legs:

1. **Starting gun (overhead):** The time between "go" and the first runner taking off.
   Fixed cost, doesn't depend on how long the race is.

2. **Memory leg:** How long it takes to fetch all the data from slow memory (HBM).
   Depends on how much data there is and how efficiently it's accessed.

3. **Compute leg:** How long the ALUs take to crunch the numbers, assuming data is
   already in registers.

In a relay race, legs 2 and 3 **run in parallel** on the GPU — while one wavefront
waits for data, another runs ALU instructions.  Only the *slower* of the two finishes
last.  The starting gun (overhead) always happens first — it doesn't overlap with
anything.

Stage 3 times all three legs for each config, identifies which is slowest, and tells
Stage 4 scoring: *"for this config, focus on improving the slowest leg."*

### Why this runs per-config

The race length (total data, total ops) is the same for all configs — that's Stage 2.
What changes between configs is:

- **Different block count** → different overhead fraction (more blocks = longer
  overhead leg) and different cache-hit probability (memory leg)
- **Different thread count** → different memory coalescing efficiency
- **Different `num_warps`** → different SPI init overhead

So Stage 3 must run separately for every config candidate.


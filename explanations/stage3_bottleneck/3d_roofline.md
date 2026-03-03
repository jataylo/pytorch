# Stage 3d — Roofline Combination (T_total)

**Source:** `BottleneckAnalysis.analyze_bottleneck()` in `triton_heuristics_adaptive.py`

**Role:** Combines `T_overhead`, `T_memory`, and `T_compute` into a single total time
estimate per config, and identifies which component is the dominant bottleneck.  This is
the direct implementation of the **roofline model** applied to a single kernel config.

---

## Technical

### The equation

```
T_total = T_overhead + max(T_memory, T_compute)
```

### Why `max()` — not addition

The GPU has **two independent hardware pipelines running simultaneously**:

1. **Memory pipeline:** The HBM memory controllers, L2 cache, and the load/store units
   that move data between HBM and registers.

2. **Compute pipeline:** The SIMD ALU arrays (Vector ALUs) and the Special Function
   Units (SFUs) that execute arithmetic instructions.

These pipelines share the register file as an interface but are otherwise independent.
While one wavefront is waiting for a memory load to return (~300 cycles), the compute
pipeline can execute arithmetic instructions on data that other wavefronts already have
in their registers.

As a result, memory time and compute time **overlap**:

```
Wall-clock:  [─────── T_memory ──────────────]
             [── T_compute ──]
             |<─ max(T_memory, T_compute) ──>|

If T_memory=10µs and T_compute=3µs:
  Compute finishes in 3µs, then the compute pipeline sits idle for 7µs waiting for memory.
  Wall-clock duration = max(10, 3) = 10µs  (not 13µs)
```

The `max()` captures this: only the **slower** of the two is visible on the wall clock.
The faster one is hidden "for free".

**Important:** The memory pipeline and compute pipeline are only fully independent when
there are enough wavefronts to keep both busy simultaneously (sufficient occupancy).
With `num_warps=1`, there is only one wavefront — when it stalls on a memory load,
**both** pipelines go idle.  This is why occupancy is scored in Stage 4e.

### Why overhead does NOT overlap

`T_overhead` represents the time before the first wavefront begins executing.  During
this time, neither the memory pipeline nor the compute pipeline is doing any kernel work.
The GPU hardware is occupied by:
- CP processing the dispatch packet
- SPI allocating VGPR/SGPR banks for each wavefront
- DMA-ing kernel arguments into SGPRs

No kernel instructions execute during this window, so overhead stacks **additively** on
top of `max(T_memory, T_compute)`:

```
T_total = T_overhead + max(T_memory, T_compute)
```

### Computing component fractions

The fractions used by the adaptive weight interpolation (Stage 3e) are derived from the
**gross cost** (sum of all three components), not from `T_total`:

```
gross          = T_overhead + T_memory + T_compute   # sum (not max)
overhead_frac  = T_overhead / gross
memory_frac    = T_memory   / gross
compute_frac   = T_compute  / gross
```

The key difference: `gross` uses addition for all three, so the fractions always sum
to 1.0 and accurately convey each component's *relative importance* even when memory and
compute partially overlap on the wall clock.

**Example:**
```
T_overhead = 3 µs,  T_memory = 10 µs,  T_compute = 4 µs

T_total  = 3 + max(10, 4) = 13 µs         # wall-clock
gross    = 3 + 10 + 4     = 17 µs         # for fraction calculation

overhead_frac = 3 / 17 ≈ 0.18
memory_frac   = 10 / 17 ≈ 0.59
compute_frac  = 4 / 17 ≈ 0.24
```

In the verbose output (`[HEURISTICS]` scoring table), these fractions are rendered as
an ASCII bar: `[OOO|MMMMMMMMMM|CCCC]` where O=overhead, M=memory, C=compute.

### Bottleneck classification

```
if overhead_frac > memory_frac and overhead_frac > compute_frac:
    bottleneck = 'overhead'   → label: 🚀 LAUNCH-BOUND
elif memory_frac >= compute_frac:
    bottleneck = 'memory'     → label: 💾 MEMORY-BOUND
else:
    bottleneck = 'compute'    → label: ⚡ COMPUTE-BOUND
```

`launch_bound = (overhead_frac > 0.50)` is a secondary flag that triggers the
alternative occupancy scoring model in Stage 4e.

### Full worked example — config comparison

Two configs for a 1M-element FP32 `c = relu(a + b)` kernel on MI300X:

```
Problem: N=1M, bytes_per_element=12, ops_per_element=2
         T_memory_ref = 1M × 12 / (5300GB/s × 0.80) ≈ 2.83 µs
         T_compute_ref = 1M × 2 / (200 TFLOPS × 0.80) ≈ 0.013 µs

Config A: XBLOCK=256, num_warps=4, num_blocks=3906
  T_overhead = 3.0 + (3-3)×0.1 + (4-1)×0.2 + 0 = 3.6 µs
  T_memory   = 2.83 µs
  T_compute  = 0.013 µs
  T_total    = 3.6 + max(2.83, 0.013) = 3.6 + 2.83 = 6.43 µs
  overhead_frac = 3.6 / (3.6+2.83+0.013) ≈ 0.56  → LAUNCH-BOUND

Config B: XBLOCK=1024, num_warps=4, num_blocks=977
  T_overhead = 3.0 + 0 + (4-1)×0.2 = 3.6 µs    (same warp count)
  T_memory   = 2.83 µs
  T_total    = 3.6 + 2.83 = 6.43 µs              (identical!)
  overhead_frac ≈ 0.56  → also LAUNCH-BOUND
```

In this example, reducing block count via larger XBLOCK doesn't help because the overhead
is dominated by `(num_warps-1)×K_warp`, not `grid_overhead`.  Reducing `num_warps` would
help more.

---

## Simple explanation

### The roofline model — the key insight

A GPU has two "speed limits" for any kernel:

1. **The memory speed limit:** How fast the HBM memory bus can deliver data
2. **The compute speed limit:** How fast the ALU can process that data

A car analogy: imagine a factory conveyor belt that moves raw materials to a worker
assembly station.  Either the conveyor (memory) is too slow to keep the worker (ALU)
busy, or the worker is so complex that the conveyor has to wait.  Only one of these
constraints matters at a time — you hit *whichever is tighter*.

The roofline model says: **the kernel runs at the speed of whichever limit is hit first.**

### What T_total captures

```
T_total = startup overhead + max(memory speed limit, compute speed limit)
```

The `max()` is the critical insight: if the memory bus takes 10 µs and the arithmetic
takes 3 µs, they run **simultaneously** — the GPU switches between wavefronts, keeping
both pipelines busy.  The 3 µs of arithmetic is "free" — it fits inside the 10 µs memory
window.  Total runtime is 10 µs, not 13 µs.

The startup overhead (T_overhead) cannot overlap with anything — the kernel hasn't
started yet when the GPU is doing dispatch setup.  It always adds on top.

### Reading the output

In the verbose log, each config shows three numbers:

```
  Config A: Overhead=3.6µs  Memory=2.83µs  Compute=0.01µs  Effective=6.43µs
  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ OVERHEAD  ████████████ MEMORY  ▒ compute
  [OOOOOOOOOOOOOOOO|MMMMMMMMM|C]
```

The `Effective` column is `T_total = T_overhead + max(T_memory, T_compute)`.  The bar
shows what fraction of the gross cost each component contributes.  Configurations where
the bar is mostly O are launch-dominated; ones where it's mostly M are memory-dominated.

### Why this matters for config selection

Once we know which component is dominant, we know what to optimise:
- **Launch-dominated:** Use larger blocks (fewer total blocks), fewer warps.
- **Memory-dominated:** Maximise bus utilisation — right thread count, good coalescing.
- **Compute-dominated:** Maximise occupancy to hide ALU latency; reduce transcendentals.

Stage 3e translates this bottleneck classification into adaptive weights that make Stage 4
focus scoring on the right factors.


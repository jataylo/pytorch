# Pointwise Kernel Heuristics — Pipeline Deep Dive

> **Audience:** This document covers all six stages of the heuristic selection pipeline.
> Each stage has a **Technical** paragraph (for engineers who read source code) and a
> **Simple explanation** paragraph (for engineers newer to GPU programming).

---

## Stage 1 — Exhaustive Configuration Proposal

### Technical

`generate_all_candidate_configs(problem_metadata)` in `triton_heuristics_pointwise.py`
enumerates the Cartesian product of block-dimension candidates and warp counts that are
**legal for this specific problem**:

| Dimensionality | Block-size candidates | Warp candidates |
|---|---|---|
| 1-D | `{16, 32, 64, 128, 256, 512, 1024}` | `{1, 2, 4, 8, 16}` |
| 2-D | `{4, 8, 16, 32, 64, 128, 256, 512, 1024}` per axis | `{1, 2, 4, 8, 16}` |
| 3-D | `{4, 8, 16, 32, 64}` per axis | `{1, 2, 4, 8, 16}` |

Two hard constraints prune the raw Cartesian product before any scoring occurs:

1. **Dimension cap:** `XBLOCK ≤ xnumel`, `YBLOCK ≤ ynumel`, etc.  A block wider than the
   problem dimension would launch threads with no work, wasting register allocation.

2. **Warp–thread coherence:** `num_warps × warp_size ≤ total_threads_per_block`.  A config
   declaring 8 warps but launching only 64 threads would request 512 threads from the
   scheduler while only 64 exist — the hardware would silently clamp, making the declared
   `num_warps` misleading to the compiler.  Both constraints are read from
   `problem_metadata['warp_size']` and `problem_metadata['max_threads_per_block']`, which
   are populated from `torch.cuda.get_device_properties()` at problem-setup time, so the
   generator is device-agnostic (it works identically on MI300X with `warp_size=64` and
   on an A100 with `warp_size=32`).

After pruning, the surviving set is typically **30–80 configs** for a 1-D problem and
**60–120 configs** for a 2-D problem.  Every surviving config represents a valid,
compilable Triton launch — no illegal config ever reaches the scoring stage.

```python
# Simplified excerpt — 1-D case
for xblock in [16, 32, 64, 128, 256, 512, 1024]:
    if xblock > xnumel:
        continue
    for num_warps in [1, 2, 4, 8, 16]:
        if num_warps * warp_size > xblock:
            continue          # more warps than threads — illegal
        configs.append({'XBLOCK': xblock, 'num_warps': num_warps})
```

> **Why powers-of-two?**  Triton's vectoriser operates on power-of-two tile widths.
> Non-power-of-two block sizes either trigger masking on every block (wasting bandwidth)
> or require extra tile-tail handling that the compiler cannot optimise away.  Restricting
> to powers of two gives the compiler maximum freedom to unroll, vectorise, and pipeline.

---

### Simple explanation

Think of this stage as generating a menu of all the ways we could divide the work.  A
GPU processes data in rectangular tiles called **blocks**.  The "block size" (`XBLOCK`,
`YBLOCK`) determines how big each tile is, and `num_warps` controls how many groups of
threads work on it at once.

The generator creates every sensible combination of tile size and warp count, then throws
out any combination that doesn't make mathematical sense for this particular problem
(e.g. a tile bigger than the problem itself, or more thread-groups than there are threads
in the tile).  What's left is a shortlist of 30–120 valid launch configurations that
will all run correctly — we just don't know yet which one will run *fastest*.

---

## Stage 2 — Extract Problem Metadata

### Technical

Metadata is assembled in **three successive phases** before any config is scored.

#### Phase 1 — Shape & Device (in `pointwise()`, `triton_heuristics.py`)

Called at kernel-generation time when Inductor emits the Triton function:

```python
problem_metadata = {
    'dimensions':   tuple(next_power_of_2(s) for s in size_hints.values()),
    'total_elements': prod(size_hints.values()),
    'element_size': dtype_to_bytes(triton_meta['dtype']),
    'warp_size':    device_props.warp_size,          # 64 on AMD, 32 on NVIDIA
    'max_threads_per_block': device_props.max_threads_per_block,
    'num_cus':      device_props.multi_processor_count,
}
```

`size_hints` values are rounded up to the next power-of-two because Triton's codegen
emits a mask for the tail elements (`tl.load(..., mask=pid*BLOCK+offs < n)`) and the mask
arithmetic is cheapest when the tile boundary is a power-of-two-aligned address.

#### Phase 2 — Kernel operand counts (in `_score_and_prune_heuristic_configs`)

After the Triton function is generated but before compilation, a lightweight regex pass
over `fn.src` counts tensor arguments and store instructions:

```python
num_inputs  = len(re.findall(r'\btl\.load\b',  kernel_code))
num_outputs = len(re.findall(r'\btl\.store\b', kernel_code))
num_tensors = num_inputs + num_outputs
```

`bytes_per_element` defaults to `element_size × (num_inputs + num_outputs)` — i.e. every
tensor is a full read or write pass over all elements — but `extract_kernel_metadata()`
in `triton_heuristics_kernel_analysis.py` can refine this with broadcast detection (a
broadcast input contributes far fewer bytes per output element than a full-tensor read).

#### Phase 3 — Instruction-mix & op density (in `extract_kernel_metadata`)

The kernel source is scanned for instruction categories:

| Category | Instructions | Cycles (approx.) |
|---|---|---|
| Fast | `tl.add`, `tl.mul`, FMA | 1–4 |
| Medium | `tl.sqrt`, `tl.abs`, integer div | 4–16 |
| Slow | `tl.exp`, `tl.log`, `tl.sin`, `tl.div` (FP) | 16–64 |

> **Terminology:**
> - **FMA (Fused Multiply-Add):** A single hardware instruction that computes `a × b + c` in
>   one cycle with no intermediate rounding.  It is the building block of virtually all neural
>   network arithmetic (matrix multiply, layer norm, attention scale).  "Fused" means the
>   multiply and add share a single pipeline pass — effectively two FLOPs for the price of one
>   instruction.
> - **Integer div (`int div`):** Dividing two integers on a GPU has no dedicated hardware unit
>   on CDNA; the compiler emits a multi-instruction sequence (reciprocal + multiply + correction)
>   costing 4–16 cycles.  Common in index arithmetic (`pid * BLOCK + offset % stride`).
> - **FP div (`tl.div`, floating-point division):** Unlike integer div, FP division on AMD
>   CDNA goes through the Special Function Unit (SFU/Transcendental Unit), which computes a
>   reciprocal estimate (`rcp`) followed by one or two Newton–Raphson refinement steps to reach
>   full FP32 precision.  This costs 16–32 cycles per element vs. 1 cycle for FMA, making
>   expressions like `1.0 / x` or `a / b` significantly more expensive than they appear in
>   Python source.

`ops_per_element` is computed as a **weighted sum** of instruction counts divided by
`total_elements`, using the cycle costs above as weights.  This is the quantity that
feeds both the roofline arithmetic-intensity calculation and the spill-risk VGPR
estimate:

```
AI = ops_per_element / bytes_per_element        [FLOPs / byte]
```

If `AI < OI_ceiling` (peak FLOPs / peak bandwidth), the kernel is memory-bound;
otherwise compute-bound.

> **What is `OI_ceiling` and why does it determine the bound?**
>
> `OI_ceiling` (Operational Intensity ceiling) is the **ridge point** of the roofline
> model — the arithmetic intensity at which a perfect kernel would saturate both the
> memory bus and the ALU simultaneously:
>
> ```
> OI_ceiling = peak_FLOPS / peak_bandwidth
>            = (200 TFLOPS × 10¹²) / (5300 GB/s × 10⁹)   ← MI300X example
>            ≈ 37.7 FLOPs / byte
> ```
>
> A kernel with `AI = ops_per_element / bytes_per_element` below this ridge is
> **memory-bound**: the ALUs finish their work faster than HBM can supply new data,
> so the ALUs sit idle waiting.  Above the ridge the situation reverses — data arrives
> faster than the ALUs can process it, making it **compute-bound**.  For almost all
> pointwise kernels (element-wise add, relu, silu, etc.) `AI` is 0.1–2 FLOPs/byte,
> far below the ~38 FLOPs/byte ridge, which is why they are nearly always
> memory-bound.
>
> **Important — this calculation happens in Stage 2, not Stage 3.**  Stage 2 computes
> `AI` once from the kernel's instruction mix and records it in `problem_metadata`.
> Stage 3 then uses that pre-computed `AI` value (along with `ops_per_element` and
> `bytes_per_element`) to build the three time-component estimates (`T_overhead`,
> `T_memory`, `T_compute`) and determine *how much* each component contributes.
> The distinction is:
> - **Stage 2** → *what is the kernel's arithmetic intensity?*  (static, per-kernel,
>   computed once from source analysis)
> - **Stage 3** → *given AI and a specific config's tile size + warp count, how long
>   does each component take, and which one dominates for this config?*  (dynamic,
>   per-config, called 30–80 times)
>
> Stage 2's `AI` check is a fast coarse gate; Stage 3's full time model is what drives
> the adaptive weight interpolation.

`has_mask` is flagged when the source contains `tl.load(..., mask=` or `tl.store(..., mask=`,
which reduces effective HBM efficiency from 0.80 to 0.65 in `estimate_memory_time_us`
because partial cache-line stores break write-combining.

---

### Simple explanation

Before scoring anything, the system gathers facts about the kernel — like reading a recipe
before deciding how to cook it.

**What it collects:**
- **Shape and size** — how many elements there are and how they're arranged (1-D, 2-D, etc.).
- **Element size** — are we working with 32-bit floats (4 bytes) or 16-bit half-precision (2 bytes)?
- **Tensor count** — how many tensors does the kernel read from and write to?  More tensors
  means more memory traffic.
- **Operation mix** — does the kernel mostly do simple additions, or does it call expensive
  transcendentals like `exp()` or `sin()`?  Expensive ops mean the ALU is the bottleneck,
  not memory.
- **Masking** — does the problem size not divide evenly into blocks?  If so, the kernel
  must check bounds on every load/store, which reduces memory efficiency.

All of this is extracted by reading the generated Triton source code with simple pattern
matching — no compilation needed.  These facts feed every subsequent stage.

---

## Stage 3 — Bottleneck Analysis

### Technical

`BottleneckAnalysis.analyze_bottleneck(config, problem_metadata)` in
`triton_heuristics_adaptive.py` builds a three-component **time model** for each
candidate config and identifies which component is the limiting factor.

#### Component 1 — Overhead (kernel dispatch)

```
T_overhead = K_launch  +  (n_args − 3) × 0.1 µs  +  (num_warps − 1) × 0.2 µs
             + grid_overhead(num_blocks) + masking_overhead
```

> **Reading the equation:**
> Every term is an additive cost incurred before the first thread executes a single
> floating-point operation.
> - `K_launch` (3.0 µs) is the irreducible fixed cost: the driver must DMA the kernel
>   descriptor to the Command Processor, the CP must parse it and signal the Shader
>   Processor Input (SPI) unit, and the SPI must begin wavefront allocation.  This
>   happens once regardless of how much work the kernel does.
> - `(n_args − 3) × 0.1 µs`: each tensor pointer passed to the kernel must be copied
>   from CPU-visible memory into a Scalar GPR (SGPR) before dispatch.  Three pointers
>   is the baseline (two inputs, one output); every additional pointer adds ~0.1 µs.
> - `(num_warps − 1) × 0.2 µs`: declaring more wavefronts per block forces the SPI to
>   allocate VGPR/SGPR register banks for each one.  With `num_warps=16` this adds 3 µs
>   — doubling the baseline `K_launch`.  This is the core reason the occupancy model
>   penalises large `num_warps` for launch-dominated kernels.
> - `grid_overhead`: logarithmic cost for very large grids (>1000 blocks) where the CP
>   must batch-dispatch in multiple rounds.

- `K_launch = 3.0 µs` — empirically measured constant for AMD ROCm kernel dispatch.
  This is the fixed cost of the CP (Command Processor) DMA-ing the kernel descriptor,
  initialising wave schedulers, and signalling execution.
- Argument setup: each extra tensor pointer beyond the baseline of 3 adds ~0.1 µs for
  the driver to copy pointer values into scalar SGPRs before dispatch.
- Warp init: each additional wavefront declared (above nw=1) requires the SPI (Shader
  Processor Input unit) to allocate VGPR/SGPR slots and enqueue the wavefront, costing
  ~0.2 µs per extra warp.
- Grid overhead is logarithmic for large grids (>1000 blocks) since the CP batches
  dispatches.

#### Component 2 — Memory transfer

```
T_memory = total_bytes / (BW_peak × η_hbm)

where  η_hbm = 0.80  (clean streaming)
             = 0.65  (masked / partial-store kernels)
       total_bytes = total_elements × bytes_per_element
```

> **Reading the equation:**
> This is the idealised streaming time — how long it would take to move all the
> kernel's input and output data through the memory hierarchy if compute were free.
> - `total_bytes = total_elements × bytes_per_element` counts every byte that must
>   cross the HBM bus: all input tensor reads plus all output tensor writes.  For a
>   simple `c = a + b` kernel on FP32, `bytes_per_element = 12` (4 bytes read from
>   `a`, 4 from `b`, 4 written to `c`).
> - `BW_peak` is the device's peak HBM bandwidth (e.g. 5,300 GB/s on MI300X,
>   2,000 GB/s on an A100).  This is the absolute ceiling imposed by the memory bus
>   width and clock.
> - `η_hbm` is an efficiency derating factor accounting for the gap between peak
>   and achievable bandwidth.  In practice, clean sequential streaming achieves ~80%
>   of peak due to ECC overhead, refresh pauses, and row-buffer conflicts.  Masked
>   kernels drop to ~65% because predicate instructions cause partial cache-line
>   stores, breaking the write-combining path in the memory controller and forcing
>   extra read-modify-write transactions.

For problems smaller than L2 (typically 4 MB on MI300X), the model uses L2 bandwidth
(~1 TB/s) instead of HBM bandwidth.  Broadcast tensors that fit in L2 are flagged so
only the non-broadcast data is charged HBM bandwidth.

L1 cache (32 KB per CU) is explicitly **not** treated as a global cache — it is
replicated per CU, so data that "fits in L1" for a single block is fetched from HBM
by each of the 120 CUs running in parallel.

#### Component 3 — Compute throughput

```
T_compute = total_ops / (TFLOPS_peak × η_instr)

where  η_instr = 0.80  (mostly FMA/add)
              = 0.70  (mixed)
              = 0.60  (>50% transcendentals)
       total_ops = total_elements × ops_per_element
```

> **Reading the equation:**
> This is the time the kernel would take if memory bandwidth were infinite and the
> ALUs were the only constraint — the pure compute ceiling.
> - `total_ops = total_elements × ops_per_element`: the total floating-point
>   operation count.  `ops_per_element` was computed in Stage 2 from the
>   instruction-mix scan (fast/medium/slow weighted counts).
> - `TFLOPS_peak` is the device's peak FP32 throughput (e.g. 200 TFLOPS on MI300X).
>   Like `BW_peak`, it is read from `torch.cuda.get_device_properties()`.
> - `η_instr` accounts for the fact that not all instructions execute at the FMA
>   rate.  An FMA completes in 1 cycle; `tl.exp` or `tl.sin` routes through the
>   Special Function Unit (SFU) and takes 16–64 cycles.  When more than 50% of
>   instructions are transcendentals, the ALU pipeline stalls waiting for the SFU,
>   dropping effective throughput to ~60% of peak.  For typical pointwise kernels
>   (mostly adds and multiplies), η_instr = 0.80 because the FMA pipeline stays
>   full with only minor stalls from data dependencies.

#### Roofline combination

```
T_total = T_overhead + max(T_memory, T_compute)
```

> **Reading the equation:**
> The GPU's memory controller and ALU pipelines are **independent and concurrent** —
> while one wavefront is waiting for a memory load to return, another wavefront can be
> executing ALU instructions on data it already has.  This means memory time and
> compute time overlap: only the *slower* one is visible on the wall clock.  The
> `max()` captures this: if `T_memory = 10 µs` and `T_compute = 3 µs`, the compute
> finishes first and is fully hidden — the effective execution time is 10 µs, not 13 µs.
>
> Overhead does **not** overlap with either component — the kernel dispatch must fully
> complete before any wavefront starts executing.  So overhead stacks additively on top
> of whatever `max(T_memory, T_compute)` gives.
>
> The three fractions `{overhead_frac, memory_frac, compute_frac}` are computed from
> the raw individual components (not from `T_total`), so they always sum to 1.0 and
> accurately reflect each component's share of gross cost even when two overlap.

Memory and compute **overlap** on the GPU (the memory controller and ALU units are
independent pipelines), so only the slower of the two is visible; the faster one is
hidden behind it.  Overhead always stacks on top because kernel dispatch must complete
before the first wavefront executes.

#### Adaptive weight interpolation

The component fractions `{overhead_frac, memory_frac, compute_frac}` act as mixing
coefficients between three pure-regime weight vectors:

```
w[k] = overhead_frac × W_overhead[k]
     + memory_frac   × W_memory[k]
     + compute_frac  × W_compute[k]
```

| Factor | W_overhead | W_memory | W_compute |
|---|---|---|---|
| bandwidth | 0.10 | 0.55 | 0.15 |
| launch | 0.65 | 0.10 | 0.10 |
| grid | 0.10 | 0.15 | 0.30 |
| occupancy | 0.15 | 0.20 | 0.45 |

A kernel whose `overhead_frac=0.7, memory_frac=0.3, compute_frac=0.0` gets:
`launch_weight = 0.7×0.65 + 0.3×0.10 = 0.485` — nearly half the total weight goes
to the launch factor, correctly prioritising configs with fewer, larger blocks.

The `launch_bound` boolean is set when `overhead_frac > 0.50`, which triggers the
`LAUNCH-BOUND` regime label in verbose output and drives the occupancy scoring to use
the first-principles overhead ratio rather than the wavefront sweet-spot model.

---

### Simple explanation

This stage answers the question: **"What is actually slowing this kernel down?"**

Every GPU kernel has three possible bottlenecks:

1. **Launch overhead** — the fixed time the GPU driver spends just *starting* the kernel.
   For a kernel that only processes 512 numbers, this overhead can be larger than the
   actual work.  Think of it like the time to warm up a car engine vs. a 30-second drive.

   Two config parameters directly control launch overhead:

   - **Block size (`XBLOCK`, `YBLOCK`):** A *larger* block size means *fewer* total blocks
     for the same amount of work, which **lowers** launch overhead.  Each block carries a
     fixed dispatch cost, so halving the number of blocks halves the grid-setup portion of
     overhead.  For example, `XBLOCK=1024` on a 1M-element problem creates ~1000 blocks,
     while `XBLOCK=64` creates ~15,600 blocks — the smaller block config dispatches 15×
     more blocks for zero extra useful work.

   - **`num_warps`:** More warps **increases** launch overhead, even for the same block
     size and same total work.  When you declare `num_warps=16` instead of `num_warps=1`,
     the Shader Processor Input (SPI) unit must allocate 16 separate VGPR/SGPR register
     banks and enqueue 16 wavefronts before execution can start — even if the kernel is
     tiny and finishes in 2 µs.  Each extra warp adds ~0.2 µs of this SPI initialisation
     cost.  So `num_warps=16` adds `15 × 0.2 = 3 µs` on top of the base 3 µs dispatch
     cost — **doubling the total overhead** before a single thread executes.

   The key insight is that block size affects overhead *indirectly* (via block count),
   while `num_warps` affects overhead *directly* (via per-block warp initialisation).
   For a tiny 512-element kernel, choosing `XBLOCK=512, num_warps=1` gives a single
   block with minimal warp init — the best possible launch overhead.  Choosing
   `XBLOCK=16, num_warps=8` creates 32 blocks each initialising 8 wavefronts — the
   kernel spends the vast majority of its wall time just starting up.

2. **Memory bandwidth** — how fast we can stream data from HBM (the main GPU memory) to
   the compute units.  Most AI kernels are bottlenecked here — they're simple enough that
   the ALUs are always waiting for data.

   Config parameters affect how efficiently the memory bus is used:

   - **`XBLOCK` (1-D) — thread count and bus utilisation:** More threads per block means
     more outstanding memory requests in flight simultaneously.  The GPU memory controller
     can pipeline HBM requests, so having ~512 threads all issuing loads at once keeps the
     bus saturated.  Too few threads (e.g. 64) means only 64 simultaneous requests —
     the bus is underutilised and bandwidth drops.  Too many threads (e.g. 2048) starts
     competing for registers and L1 cache space, reducing effective throughput.

   - **`XBLOCK` (2-D) — memory coalescing:** In a 2-D kernel (`XBLOCK × YBLOCK`), the X
     dimension is the *contiguous* axis in memory (row-major layout).  If `XBLOCK` is
     small (e.g. 4), each row of the tile only touches 4 × 4 bytes = 16 bytes, but the
     GPU fetches a full 64-byte cache line — the remaining 48 bytes are wasted bandwidth.
     Adjacent blocks will eventually reuse that cache line, but in the meantime you're
     paying HBM bandwidth for data you didn't need yet.  `XBLOCK ≥ 16` fills a full cache
     line per row, giving 100% bus utilisation.

   - **`num_warps` — latency hiding:** Each wavefront can hide its memory latency by
     switching to another wavefront while waiting for HBM.  With `num_warps=1`, every
     load stalls the single wavefront for ~300 cycles.  With `num_warps=4–8`, while one
     wavefront waits for data, others continue executing — keeping the ALUs and memory
     bus busy simultaneously.  However, beyond ~8 wavefronts, the register budget per
     wavefront shrinks (65,536 VGPRs / 8 = 8,192 VGPRs / wavefront = 128 VGPRs/thread),
     and if the kernel needs more than that, the compiler spills registers to local memory
     — adding extra load/store traffic that *increases* the effective bytes read and
     worsens bandwidth efficiency.

   - **`YBLOCK` (2-D aspect ratio) — prefetcher alignment:** Very tall, thin tiles
     (e.g. `XBLOCK=4, YBLOCK=256`) jump between rows in memory more often than wide, flat
     tiles (`XBLOCK=64, YBLOCK=16`).  Each row-switch is a new cache-line fetch with
     potentially no spatial reuse of the previous line.  The hardware prefetcher works
     best with sequential, wide access patterns along the contiguous axis — wider
     `XBLOCK` and moderate `YBLOCK` give it the best signal to prefetch ahead.

3. **Compute throughput** — how many floating-point operations the hardware can execute per
   second.  Only kernels with heavy maths (like `exp`, `sin`, or long chains of FMAs) hit
   this limit.

The roofline model says: *memory and compute run at the same time, so only the slower one
counts*.  Overhead always adds on top.

Once we know which bottleneck dominates, we adjust *how much we care about each scoring
factor*.  If the kernel is tiny and launch-dominated, we care a lot about minimising the
number of blocks (launch factor).  If it's a large streaming kernel, we care most about
memory coalescing (bandwidth factor).

---

## Stage 4 — Parallel Scoring

### Technical

Each surviving config is scored by `_score_one_config()`, called in parallel via
`ThreadPoolExecutor` (one task per config, up to `min(len(configs), 32)` workers).
The four factors below are multiplied together using a **weighted geometric mean**:

```
score = (BW^a × Launch^b × Grid^c × Occ^d) ^ (1 / (a+b+c+d))
```

where `{a, b, c, d}` are exponents derived from adaptive weights via a linear map:

```
exp = 0.5 + (weight − 0.10) / 0.40 × 2.5    clamped to [0.5, 3.0]
```

#### Factor 1 — Bandwidth utilisation

Modelled as a Gaussian centred at `optimal_threads_bandwidth` (512 threads / 8
wavefronts on AMD MI300X, from Little's Law: `wavefronts_needed = latency_cycles /
issue_gap_cycles ≈ 300 / 40 ≈ 8`):

```
BW_score = 0.75 + 0.25 × exp(−0.5 × ((threads − optimal) / σ)²)
           floor: 0.60 for threads < 64
```

> **Reading the equation:**
> This is a bell-curve (Gaussian) centred at the empirically optimal thread count of
> 512 (8 wavefronts × 64 threads/wavefront on AMD).  The score is 1.0 at exactly 512
> threads and falls smoothly on both sides.  The floor of 0.75 means even a very
> poor thread count still scores at least 0.75 — because a bad thread count can still
> be the best available option for tiny problems.  Configs with fewer than 64 threads
> (less than one full wavefront) are hard-floored at 0.60 — they are fundamentally
> unable to fill the memory pipeline.
>
> The optimal of 512 comes from Little's Law: to hide the ~300-cycle HBM round-trip
> latency with a 40-cycle wavefront issue interval, you need `⌈300/40⌉ = 8`
> wavefronts in flight simultaneously.  At `warp_size=64`, 8 wavefronts = 512 threads.
> Too few threads → ALUs stall waiting for HBM.  Too many threads → register pressure
> increases, the compiler must spill to local memory, and cache-line reuse within a
> block drops.

For 2-D kernels a **coalescing correction** is applied because the GPU cache line holds
64 bytes = 16 FP32 elements.  If `XBLOCK < 16`, each tile row occupies less than one
cache line, leaving fetched bytes unused until a neighbouring block reuses them:

```
coalescing = min(1.0, XBLOCK / 16)
BW_score  *= (0.65 + 0.35 × coalescing)
```

`XBLOCK=4` → `×0.74`; `XBLOCK=8` → `×0.82`; `XBLOCK≥16` → `×1.00`.

#### Factor 2 — Launch overhead

Gaussian centred at `optimal_elements_per_block` (hardware-derived from launch
amortisation analysis):

```
Launch_score = 0.75 + 0.25 × exp(−0.5 × ((EPB − optimal_EPB) / σ)²)
               floor: 0.70 for EPB < 64 (overhead dominates)
```

> **Reading the equation:**
> `EPB` (elements per block) = `total_elements / num_blocks`.  It measures how much
> *useful work* each block does relative to its fixed dispatch cost.  The score peaks
> at `optimal_EPB` (hardware-derived from launch amortisation analysis) and falls
> symmetrically on both sides.
>
> **When Launch score is HIGH** (close to 1.0): each block is doing a large amount of
> work, so the fixed `K_launch` dispatch cost is well-amortised.  A config with
> `XBLOCK=1024` and a 1M-element problem creates ~1000 blocks each handling 1024
> elements — the 3 µs dispatch cost is a tiny fraction of the total.
>
> **When Launch score is LOW**: the blocks are too small or too numerous.  Classic
> failure mode is a config like `XBLOCK=16, num_warps=1` on a 1M-element problem —
> this creates 65,536 blocks, each doing only 16 elements of work.  The GPU spends
> more time dispatching blocks than executing them.  EPB = 16 < 64, so the formula
> hard-floors to 0.70 immediately.
>
> **The large-grid penalty** is a secondary correction for pathological grids — even
> when EPB looks acceptable, a grid vastly larger than 4× the number of CUs adds
> Command Processor queue pressure (the CP must batch-dispatch in multiple rounds),
> adding measurable latency that the base Gaussian doesn't capture.

A large-grid scheduler penalty applies when `num_blocks > 4 × num_CUs`:

```
grid_excess_penalty = 1.0 − 0.01 × log₂(num_blocks / (4 × num_CUs))
                      clamped: ≥ 0.88
```

This models the finite AMD Command Processor queue: grids beyond ~4× full occupancy add
measurable scheduling latency without further throughput benefit.

#### Factor 3 — Grid granularity

> **What is grid granularity?**
> The GPU is composed of many Compute Units (CUs) — MI300X has 304, MI250X has 220.
> Each CU can run blocks independently and in parallel.  The *grid* is the total set
> of blocks the kernel launches.  Grid granularity measures whether the number of
> blocks is well-matched to the number of CUs available.
>
> **Too few blocks** (under-saturated): most CUs sit idle.  A single block running on
> one CU uses less than 1% of the hardware — even if that block finishes in 10 µs,
> you've wasted 99% of your compute budget.
>
> **Too many blocks** (over-saturated): the Command Processor (CP) must dispatch blocks
> in multiple rounds.  After the first `num_CUs` blocks start, the CP must queue the
> rest and re-dispatch as CUs free up.  For very large grids this scheduling overhead
> becomes measurable, and the per-block dispatch latency cuts into useful compute time.
>
> **The sweet spot** is enough blocks to keep all CUs busy with at least one wave of
> work, without creating so many that the scheduler becomes the bottleneck.  This sweet
> spot shifts with problem size: tiny problems can't afford many blocks because dispatch
> overhead dominates; large streaming problems need at least 2× num_CUs so CUs stay
> busy while early blocks are still in flight.

Problem-size-adaptive Gaussian; the optimal block count scales with problem size:

| Problem size | Target blocks | Rationale |
|---|---|---|
| < 2 048 | 4 | Spread across 4 CUs; overhead of more blocks exceeds benefit |
| 2 K – 16 K | `num_CUs / 8` | Partial saturation, favour smaller grids |
| 16 K – 256 K | `num_CUs` | Half-saturation |
| > 256 K | `2 × num_CUs` | Full saturation target |

> **How the score is calculated:**
> For each problem-size band, a Gaussian is centred at the target block count above.
> The Gaussian has width `σ = target × 0.5`, giving a smooth score that decays as the
> actual block count moves away from the target in either direction.  The score is
> bounded to `[0.70, 1.00]` — even a badly-sized grid scores at least 0.70 because
> other factors (bandwidth, occupancy) may compensate.
>
> For the **tiny** band (< 2048 elements), the Gaussian is replaced by a discrete
> lookup table because the element count is too small to fit a meaningful Gaussian:
> 4 blocks = 1.0, 3–2 blocks = 0.93, 1 block = 0.85, >8 blocks = 0.70.  The peak at
> 4 (not 1) reflects the empirical finding that AMD can dispatch 4 small blocks to 4
> separate CUs with negligible extra overhead, delivering ~2–3% more throughput than a
> single block by avoiding the single-CU resource contention bottleneck.

For tiny kernels the grid score peaks at exactly 4 blocks (score = 1.0); single-block
configs are penalised to 0.85 because AMD can dispatch 4 blocks to 4 CUs with negligible
extra overhead while leaving 116 CUs idle with a single block.

#### Factor 4 — Occupancy

> **What is occupancy?**
> Occupancy is the number of wavefronts (groups of 64 threads) that are simultaneously
> resident on a single Compute Unit.  The GPU uses wavefront switching to hide latency:
> when one wavefront stalls waiting for a memory load (which takes ~300 cycles on HBM),
> the CU immediately switches to execute another wavefront that has ready data.  If
> enough wavefronts are resident, this switching keeps the ALUs busy 100% of the time
> even though each individual wavefront is stalling frequently.
>
> **Too few wavefronts (under-occupied):** The CU has nothing to switch to while
> waiting for memory.  Every `tl.load` causes a visible stall — the ALUs go dark for
> 10–20 ns while HBM data arrives.  A single-wavefront block (num_warps=1) on a
> memory-bound kernel can be 3–5× slower than the same kernel with 8 wavefronts.
>
> **Too many wavefronts (over-occupied):** Each wavefront needs registers (VGPRs) to
> hold its live values.  The 65,536 VGPRs per CU are split equally among all resident
> wavefronts — so 16 wavefronts get only `65536 / 16 = 4096` VGPRs each, meaning at
> most 64 VGPRs per thread.  If the kernel needs more than 64 VGPRs (common for
> complex fused operations), the compiler must spill registers to local memory, causing
> extra load/store instructions that are worse than the original stalls.  Additionally,
> each extra declared wavefront costs ~0.2 µs in launch overhead (SGPR/VGPR bank
> allocation in the SPI), which for tiny kernels dwarfs any latency-hiding benefit.

Three regimes, selected by saturation level and block count:

**Memory-bound / well-saturated (`saturation ≥ 0.25`):**  Sweet-spot model — `num_warps`
in `[sweet_min, sweet_max]` (4–8 on AMD) gives `score=1.0`.  Derived from Little's Law:
enough resident wavefronts must exist to hide ~300-cycle HBM latency given a 40-cycle
issue gap ⇒ `ceil(300/40) = 8` wavefronts, but 4 suffices when L2 hit rate is high.

**Launch-bound, single-block (`saturation < 0.25, num_blocks=1`):**  All work on one CU;
wavefront latency hiding still governs, so the same sweet-spot model applies.

**Launch-bound, multi-block (`saturation < 0.25, num_blocks > 1`):**  Each block runs on
a separate CU; latency hiding is provided by grid spread rather than per-block warp count.
The per-block score becomes a first-principles overhead ratio:

```
T(nw) = K_launch + (nw − 1) × K_warp        K_launch=3.0 µs, K_warp=0.2 µs
score  = K_launch / T(nw) = 3.0 / (3.0 + (nw−1) × 0.2)
```

> **Reading the multi-block equation:**
> When each block lands on a *separate* CU, latency hiding is already handled across
> CUs by the grid (many CUs work simultaneously).  Within a single block, declaring
> more wavefronts only adds the SPI allocation cost of `(nw−1) × 0.2 µs` without
> providing any additional latency-hiding benefit.  The ideal is therefore `nw=1`
> (minimum overhead), and the score is the ratio of that minimum time to the actual
> time for this config's `nw`.  At `nw=1` the ratio is `3.0/3.0 = 1.00`; at `nw=16`
> it's `3.0/(3.0 + 15×0.2) = 3.0/6.0 = 0.50` — the kernel spends half its wall
> time just initialising wavefronts.

This produces `score(nw=1)=1.00`, `score(nw=4)=0.83`, `score(nw=16)=0.50` — no tuned
coefficients, only the same hardware constants used by the overhead model.

#### 2-D tile tie-breaker

For 2-D configs with equal composite scores, a secondary multiplier rewards:
1. **Square-ish aspect ratio** — elongated tiles misalign with the hardware prefetcher:
   `×(1.0 − 0.005 × log₂(max/min))`
2. **Larger XBLOCK** — row-major tensors have contiguous elements along X; a wider X tile
   maximises cache-line fill:
   `×(1.0 − 0.008 × max(0, 5 − log₂(XBLOCK)))`

After all configs are scored the list is sorted descending and the top-N
(`heuristics_top_n_configs`, default 5) plus the spill buffer are placed into
`self.configs` for Triton compilation.

---

### Simple explanation

This is the main scoring stage — the engine that ranks all 30–120 configs without
compiling any of them.  Think of it like a judge at a baking competition rating entries
on four criteria: *taste, presentation, texture, and originality*.

**The four scoring factors are:**

1. **Bandwidth** — Does this config use the right number of threads to keep the memory
   pipe busy?  Too few threads and the GPU sits idle waiting for data.  Too many and you
   waste register space.  There's a sweet spot (~512 threads on AMD) and the score peaks
   there.  For 2-D problems, there's an extra check: are the threads accessing memory in
   a nice, contiguous pattern?  Jumpy access patterns waste cache-line fetches.

2. **Launch overhead** — Does each block do enough work to justify the fixed cost of
   launching it?  A config that creates 10,000 tiny blocks is slow because the GPU spends
   most of its time just *starting* blocks rather than executing them.

3. **Grid granularity** — Does the grid size match the GPU's capacity?  Too few blocks
   leave most Compute Units idle.  Too many blocks create scheduling pressure.  The sweet
   spot scales with problem size.

4. **Occupancy** — Are the right number of wavefronts (thread groups) resident per
   Compute Unit to hide memory latency?

   A **wavefront** is a group of 64 threads that the GPU schedules as a single unit.
   When one wavefront stalls waiting for memory, the Compute Unit instantly switches to
   another wavefront that is ready to run — this is how the GPU hides the ~300-cycle
   HBM latency and keeps its ALUs busy.

   - **Too few wavefronts:** The CU has nothing to switch to while one wavefront waits
     for memory.  The ALUs go dark for the full ~300-cycle latency on every load,
     making the kernel 3–5× slower than it needs to be.
   - **Too many wavefronts:** Each wavefront needs its own set of registers (VGPRs).
     The 65,536 VGPRs on a CU are split equally among all resident wavefronts.  Declare
     16 wavefronts and each thread gets only 64 VGPRs — barely enough for a simple
     kernel, and a register spill waiting to happen for a complex one.  On top of that,
     each extra wavefront declared adds ~0.2 µs of SPI initialisation overhead that is
     pure dead time for small kernels.
   - **The ideal (4–8 wavefronts on AMD)** is derived from Little's Law: you need just
     enough resident wavefronts to keep the ALUs fed, without exceeding the register
     budget or wasting launch overhead on wavefronts that don't contribute to latency
     hiding.  For small, launch-dominated kernels the balance tips toward fewer warps —
     the latency-hiding benefit is marginal when the kernel only runs for a few
     microseconds anyway.

All four scores are multiplied together (weighted geometric mean), with the weights
automatically adjusted based on the bottleneck analysis from Stage 3.  The result is a
single number per config; the top 5 move on to actual GPU compilation and benchmarking.

The scoring runs in parallel across all configs simultaneously, taking roughly 1–2 ms
total regardless of how many configs there are.

---

## Stage 5 — Spill Detection and Filtering

### Technical

Register spills occur when a Triton kernel requires more Vector General-Purpose Registers
(VGPRs) per thread than the hardware allocates given the thread-block size.  On AMD CDNA3
(MI300X) there are **65,536 VGPRs per CU**; with `num_warps × 64` threads per block, the
hardware assigns at most:

```
max_vgprs_per_thread = min(256, 65536 // (num_warps × warp_size))
```

A 16-warp block has `65536 / (16×64) = 64` VGPRs per thread.  If the compiler needs 80
VGPRs (for intermediate temporaries, loop variables, tensor pointers, and accumulator
registers), it must spill 16 VGPRs to local memory (LDS or scratch space), which adds
hundreds of extra load/store instructions and can multiply execution time by 2–5×.

Spills are **not knowable without compilation** — they depend on the compiler's register
allocation across the full kernel graph.  The system provides two mechanisms:

#### Mechanism A — Pre-compile heuristic estimate (informational only)

`_estimate_spill_risk(config_dict, problem_metadata)` in `triton_heuristics.py`
estimates VGPR demand from first principles:

```
estimated_vgprs = 16                                  # base: loop vars, predicates, pid
                + (num_inputs + num_outputs) × 8      # operand registers
                + ops_per_element × 1.5               # compute temporaries
                + 10  if YBLOCK > 0                   # 2-D stride/index overhead
                + 8   if XBLOCK ≥ 512                 # loop-unroll live-value pressure
```

The estimate is intentionally conservative (tends to over-estimate by 10–20%) to avoid
suppressing valid configs.  The ratio `estimated_vgprs / max_vgprs` is shown in the
verbose scoring table as `low`, `MED`, or `HIGH` — purely informational, does not alter
ranking:

```
ratio > 1.0  → HIGH  (very likely to spill)
ratio > 0.75 → MED   (at risk)
ratio ≤ 0.75 → low   (probably safe)
```

#### Mechanism B — Post-compile eviction (optional, env-controlled)

When `TORCHINDUCTOR_HEURISTICS_SPILL_BUFFER > 0`, the compile pool is expanded from
`top_N` to `top_N + buffer` configs.  After `_make_launchers()` runs (which populates
`launcher.n_spills` from the compiled binary metadata), a **pre-benchmark eviction pass**
replaces any spilling top-N config with the highest-ranked non-spilling buffer config:

```python
for i, hdict in enumerate(top_n_pool):
    launcher = hdict_to_launcher[tuple(sorted(hdict.items()))]
    if launcher.n_spills > spill_threshold:
        top_n_pool[i] = next_non_spilling_buffer_config()
```

The spill threshold defaults to 32 for ROCm and 16 for CUDA (configurable via
`inductor_meta['spill_threshold']`).  Evicted configs are never benchmarked — this avoids
wasting GPU time on kernels that will return `inf` from `bench()` and cannot be selected.

If all top-N configs spill even after eviction (buffer exhausted), `autotune_to_one_config`
falls back to the fastest non-spilling config across all compiled configs.

---

### Simple explanation

A GPU thread has a limited number of "scratchpad slots" called registers (VGPRs).  If a
kernel is too complex for the number of registers available — which depends on how many
threads are in a block — the compiler has to temporarily write register values to slower
memory and read them back.  This is called a **register spill**, and it makes the kernel
run many times slower.

The problem is that you only find out about spills *after* the compiler has run, which is
expensive.  The system has two tools to deal with this:

**Tool A (fast, approximate):** Before compiling anything, the system estimates how many
registers a kernel will need based on simple rules: how many tensors it touches, how many
operations it performs, and the tile size.  This estimate is shown in the scoring table
as a warning (`HIGH`, `MED`, or `low`).  It doesn't change which configs are selected —
it's just an early warning sign.

**Tool B (exact, optional):** If you set the `TORCHINDUCTOR_HEURISTICS_SPILL_BUFFER`
environment variable to a positive number (e.g. 3), the system compiles a few extra
"backup" configs beyond the top 5.  After compiling, before doing any GPU timing runs,
it checks the actual spill count in each compiled binary.  If a top-5 config spills, it
gets swapped out for the best non-spilling backup — saving the wasted GPU time of
benchmarking a kernel that is guaranteed to be slow.

---

## Stage 6 — Validation

### Technical

Controlled by `inductor_config.heuristics_real_bench` (env var
`TORCHINDUCTOR_HEURISTICS_REAL_BENCH`).  When enabled, `_score_and_prune_heuristic_configs`
does **not** prune `self.configs` — all generated configs are passed to the Triton
compiler and benchmarked.  However, the **winner is still selected from the heuristic
top-N**, not from the global minimum:

```
# Real-bench mode: compile and time all configs
self.configs = all_generated_configs          # no pruning
# ...after benchmarking:
timings = {cfg: bench(cfg) for cfg in all_configs}

# Selection restricted to heuristic top-N
top_n_timings = {cfg: t for cfg, t in timings.items() if cfg in top_n_set}
best = min(top_n_timings, key=top_n_timings.get)
```

This means real-bench mode collects ground-truth timing for every config while still
testing whether the heuristic selection would have chosen the right config.

`_print_heuristics_validation_summary` then emits a structured comparison report:

```
════════════════════ HEURISTICS VALIDATION SUMMARY ════════════════════
  Regime        : MEMORY-BOUND  (AI=0.167 vs OI=222 FLOPs/B)
  Overhead      : 10% | Memory: 85% | Compute: 5%
  PREDICTED #1  : {XBLOCK:256, num_warps:4}  score=0.9241
  ACTUAL BEST   : {XBLOCK:256, num_warps:4}  time=4.12µs  ← CORRECT ✓
  ...
  Factor  Pred   Actual  Delta  Wt    Note
  BW      0.962  0.971  -0.009  55%   on-target
  Launch  0.810  0.804  +0.006  10%   predicted HIGHER
  Grid    0.900  1.000  -0.100  15%   predicted LOWER
  Occ     1.000  0.900  +0.100  20%   predicted HIGHER
```

The per-factor delta table exposes **which scoring signal is drifting** — for example, a
consistently negative Grid delta means the grid granularity formula is under-scoring
configs with more blocks than it expects.  This is the primary tool used to identify and
fix scoring bugs iteratively (as seen across tuning log versions 4–11).

Hit-rate metrics reported across a benchmark suite:
- **Top-1 hit rate** — heuristic #1 prediction matches actual best.
- **Top-3 hit rate** — actual best is within the heuristic top-3.
- **Top-5 hit rate** — actual best is within the heuristic top-5.

A missed top-1 prediction is logged with: actual vs. predicted config diff, speedup gap
(`actual_time / predicted_time`), and the exact sub-problem (kernel name + size_hints)
so regressions can be pinpointed.

The validation log is gated behind `inductor_config.heuristics_verbose`
(`TORCHINDUCTOR_HEURISTICS_VERBOSE`) — in production (verbose=False) the summary is
suppressed and only the selection message is printed.

---

### Simple explanation

This stage is about **checking our work** — comparing what the heuristic predicted to
what actually runs fastest.

In normal production mode, the system picks the top-5 configs, compiles only those, and
benchmarks only those.  In validation mode (`REAL_BENCH`), it compiles and times *every*
config, then asks: "Would the heuristic's top-5 have included the real winner?"

The output is a **scorecard** that shows:
- Which config the heuristic ranked #1
- Which config actually ran fastest
- Whether they match (✓ or ✗)
- A factor-by-factor breakdown showing where the score model was off

This is how the heuristic is improved over time.  When we see that the heuristic
consistently *over-scores* the grid factor (predicting large-grid configs to be better
than they are), that's a signal to re-calibrate the grid scoring formula.  Real-bench
mode turns the heuristic into a self-improving system: every run generates labeled data
that shows exactly where the model is wrong.

In day-to-day use, validation mode is off.  It's only enabled when developing or
debugging the heuristic, because benchmarking all 60–120 configs adds significant
compile + GPU time.


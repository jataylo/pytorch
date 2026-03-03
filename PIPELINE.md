# Pointwise Kernel Heuristics — Pipeline Deep Dive

> **Audience:** This document covers all six stages of the heuristic selection pipeline.
> Each stage has a **Technical** paragraph (for engineers who read source code) and a
> **Simple explanation** paragraph (for engineers newer to GPU programming).

---

## System Overview

The pointwise heuristics system replaces exhaustive autotuning (benchmarking every
possible Triton config on the GPU) with a **static performance model** that predicts the
best config from first principles in 1–2 ms.  The goal is to achieve >90% top-1 accuracy
and >99% top-5 accuracy while eliminating the per-kernel GPU benchmarking cost that can
add hundreds of milliseconds to cold-start compilation.

### Architecture

```
  Inductor code-gen                    triton_heuristics.py
  ───────────────────────              ──────────────────────────────────────────────────
  pointwise() →                        CachingAutotuner.__init__()
    size_hints, dtype,                   _score_and_prune_heuristic_configs()
    device props          ──────►          │  Stage 1: generate_all_candidate_configs()
    ↓                                      │  Stage 2: extract_kernel_metadata() (regex)
  problem_metadata                         │           fn.arg_names (authoritative counts)
  injected into                            │           BottleneckAnalysis (device model)
  inductor_meta as                         │  Stage 3: analyze_bottleneck() per config
  '_heuristics_pending'                    │  Stage 4: _score_one_config() × N (parallel)
                                           │  Stage 5: _estimate_spill_risk() per config
                                           │  → prune self.configs to top-N (+buffer)
                                           │
                                         _make_launchers()   ← Triton compilation
                                           │  launcher.n_spills populated from binary
                                           │
                                         autotune_to_one_config()
                                           │  Stage 5B: _evict_spill_configs() (if buffer>0)
                                           │  bench() → float("inf") for spilled configs
                                           │  Stage 6: _print_heuristics_validation_summary()
                                           └─► select winner
```

### Global State Dictionaries

| Dict | Key | Value | Purpose |
|---|---|---|---|
| `_HEURISTICS_VALIDATION_DATA` | `problem_key` | `{problem_metadata, predicted_scores, actual_timings, kernel_code}` | Stores predictions + benchmark results for the validation summary |
| `_TOP_N_CONFIGS_FOR_SELECTION` | `problem_key` | `[list of top-N effective dicts]` | Restricts winner selection to heuristic top-N in real-bench mode |
| `_HEURISTICS_FULL_RANKED_HDICTS` | `problem_key` | `[all effective dicts in score order]` | Full ranked list used for spill-fallback in heuristics-only mode |

`problem_key` is produced by `_normalize_problem_key(size_hints)` — a canonical string
representation of the problem dimensions (e.g. `"(32768,)"` for a 1-D 32K-element problem).

### Environment Variables

| Variable | Config attribute | Default | Effect |
|---|---|---|---|
| `TORCHINDUCTOR_POINTWISE_HEURISTICS` | — | `1` | Master on/off switch; set to `0` to disable entirely and fall back to standard autotuning |
| `TORCHINDUCTOR_HEURISTICS_REAL_BENCH` | `heuristics_real_bench` | `1` | `1` = compile + benchmark all configs, select winner from top-N (validation/dev mode); `0` = heuristics-only, compile only top-N |
| `TORCHINDUCTOR_HEURISTICS_TOP_N` | `heuristics_top_n_configs` | `5` | Size of the selection pool.  In heuristics-only mode, only this many configs are compiled |
| `TORCHINDUCTOR_HEURISTICS_SPILL_BUFFER` | `heuristics_spill_fallback_buffer` | `0` | Extra configs compiled as a spill safety net (heuristics-only mode only); `0` disables |
| `TORCHINDUCTOR_HEURISTICS_VERBOSE` | `heuristics_verbose` | `1` | `1` = print full scoring table, bottleneck analysis, and validation summary; `0` = compact one-liners only |

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

After the Triton function is generated but before compilation, two complementary passes
count tensor arguments:

**Pass A — regex over `fn.src` (fast, always available):**

```python
num_inputs  = len(re.findall(r'\btl\.load\b',  kernel_code))
num_outputs = len(re.findall(r'\btl\.store\b', kernel_code))
num_tensors = num_inputs + num_outputs
```

**Pass B — authoritative from `fn.arg_names` (preferred when available):**

After `self.fn` is set, `_score_and_prune_heuristic_configs` checks if the JIT function
exposes `arg_names`.  Every `_ptr`-suffixed argument is a tensor pointer; the number of
`tl.store` calls in the source determines how many are outputs:

```python
ptr_args    = [a for a in self.fn.arg_names if a.endswith('_ptr')]
num_outputs = max(1, kernel_code.count('tl.store'))
num_inputs  = max(0, len(ptr_args) - num_outputs)
```

This pass supersedes the regex result and is more accurate because:
- `fn.arg_names` is the *compiler's* parameter list — it cannot be confused by comments
  or string literals in the source.
- Counting `tl.store` calls is exact (there is exactly one `tl.store` per output tensor
  in Inductor-generated code).

`bytes_per_element` defaults to `element_size × (num_inputs + num_outputs)` — i.e. every
tensor is a full read or write pass over all elements — but `extract_kernel_metadata()`
in `triton_heuristics_kernel_analysis.py` can refine this with broadcast detection (a
broadcast input contributes far fewer bytes per output element than a full-tensor read).

#### Phase 2.5 — Cache-hit recovery

`_score_and_prune_heuristic_configs` is called from `CachingAutotuner.__init__`.  On a
**cold path**, `pointwise()` executes and injects `_heuristics_pending` into
`inductor_meta` with the pre-built `problem_metadata`.  On a **cache hit** (the compiled
module is loaded from `PyCodeCache`, `FxGraphCache`, or any other Inductor cache layer),
`pointwise()` is *not* re-executed — the decorator is never re-applied — so
`_heuristics_pending` is absent.

In this case the system recovers by calling:

```python
problem_metadata = _convert_to_pointwise_heuristics_metadata(
    self.size_hints, self.inductor_meta, self.triton_meta
)
```

which reconstructs `problem_metadata` from `size_hints` (always available in the
constructor), re-deriving device constants and dimension info.  The reconstruction is
logged at `INFO` level as:

```
[HEURISTICS] Reconstructed problem_metadata from size_hints=…
(module was loaded from cache; pointwise() was not re-executed)
```

This ensures heuristic scoring is applied correctly even on cache hits, rather than
silently falling back to non-heuristic config selection.

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
| launch | 0.57 | 0.10 | 0.10 |
| grid | 0.18 | 0.15 | 0.30 |
| occupancy | 0.15 | 0.20 | 0.45 |

> **W_overhead rationale (launch 0.57, grid 0.18):** The Launch factor's EPB Gaussian
> is near-flat for multi-block kernels (wide sigma makes all reasonable block sizes
> score ≈ 0.88–1.00), so it barely discriminates configs in the overhead-bound regime.
> The Grid factor (CU saturation) is the *primary* signal distinguishing e.g. XBLOCK=256
> (84% CU utilisation) from XBLOCK=1024 (21% CU utilisation) for medium-sized problems —
> empirically the grid weight increase from 0.10 → 0.18 fixed an 18% speed gap in the
> conv-block benchmark.  Launch was reduced from 0.65 → 0.57 to compensate.

A kernel whose `overhead_frac=0.7, memory_frac=0.3, compute_frac=0.0` gets:
`launch_weight = 0.7×0.57 + 0.3×0.10 = 0.429` — the largest single weight, correctly
prioritising configs with well-amortised blocks; Grid gets `0.7×0.18 + 0.3×0.15 = 0.171`.

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

Two behaviours based on grid size:

**Single-block kernels** (`num_blocks = 1`): EPB directly proxies overhead amortisation.
Gaussian centred at `optimal_EPB` (hardware-derived):

```
Launch_score = 0.75 + 0.25 × exp(−0.5 × ((EPB − optimal_EPB) / σ)²)
               σ = optimal_EPB / 2
               floor: 0.70 for EPB < 64 (overhead completely dominates)
```

**Multi-block kernels** (`num_blocks > 1`): the ~3 µs kernel dispatch cost is shared
equally across ALL blocks, so it is already amortised over the whole problem regardless
of block count.  Using the narrow EPB Gaussian would incorrectly reward large-XBLOCK
configs (few, large blocks) over smaller ones that keep more CUs active.  Instead a very
wide sigma makes the score near-flat across all reasonable EPB values:

```
Launch_score = 0.88 + 0.12 × exp(−0.5 × ((EPB − optimal_EPB) / (3 × optimal_EPB))²)
               hard floor: 0.88 for EPB < 32 (sub-cache-line blocks)
               range: [0.88, 1.00]
```

> **Reading the equations:**
> `EPB` (elements per block) = `total_elements / num_blocks`.
>
> For a **single-block** kernel the sole block carries the entire 3 µs dispatch cost; a
> large EPB means the kernel does substantial work for that fixed overhead, scoring well.
> EPB < 64 is hard-floored at 0.70 — overhead completely dominates.
>
> For a **multi-block** kernel the 3 µs is spread across all blocks (e.g. 1000 blocks
> → 0.003 µs overhead per block), so EPB is no longer a meaningful predictor.  The
> wide-sigma formula (`σ = 3 × optimal_EPB`) keeps all configs in [0.88, 1.00] — only
> pathological sub-cache-line blocks (EPB < 32) stay at the 0.88 floor.  Block-count
> optimisation is fully delegated to the Grid score.  This change fixed an 18% speed
> gap where the old narrow Gaussian incorrectly preferred XBLOCK=1024 (fewer blocks)
> over XBLOCK=256 (more CUs active) for medium-sized convolution kernels on MI300X.
>
> **The large-grid penalty** applies to both cases: a grid vastly larger than 4× the
> number of CUs causes the Command Processor to batch-dispatch in multiple rounds,
> adding measurable scheduling latency.

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
> 4 blocks = 1.0, 3–2 blocks = 0.93, 1 block = 0.85, 5–8 blocks = 0.90, >8 blocks = 0.70.
> The peak at 4 (not 1) reflects the empirical finding that AMD can dispatch 4 small
> blocks to 4 separate CUs with negligible extra overhead, delivering ~2–3% more
> throughput than a single block by avoiding single-CU resource contention.  Configs
> with more than 8 blocks at < 2048 elements are penalised (0.70) because per-block
> dispatch overhead then dominates any parallelism benefit.

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

Four regimes, selected by saturation level, block count, and blocks-per-CU:

**Well-saturated (`saturation ≥ 0.25`) or single-block:**  Sweet-spot model — `num_warps`
in `[sweet_min, sweet_max]` (4–8 on AMD) gives `score=1.0`.  Derived from Little's Law:
`ceil(300/40) = 8` wavefronts needed to hide HBM latency; 4 suffices when L2 absorbs
traffic.

Within this regime, `num_warps` cases below and above the sweet-spot are handled as:

| `num_warps` range | Score | Reason |
|---|---|---|
| `[4, 8]` (sweet spot) | 1.00 | Optimal latency hiding |
| `[2, 12]` (near sweet) | 0.95 | Marginal deviation from ideal |
| `= 1`, CU-level hiding OK (`wf/CU > sweet_max`) | 0.85–1.00 | Many blocks keep CUs busy; ILP further helps |
| `= 1`, CU-level hiding insufficient (`wf/CU ≤ sweet_max`) | **0.82** | Too few wavefronts per CU; see below |
| `> 12` (over-warped) | **0.88** | Extra wavefronts cause SPI overhead; see below |

**CU-level latency hiding (`wf_per_cu` check):**  The key distinction for `num_warps=1`
is whether the CU scheduler can switch between *blocks* to hide latency, or whether
per-block warp count is the sole source of wavefronts:

```
wf_per_cu = num_blocks / num_CUs
need_intra_block_warps = (wf_per_cu ≤ sweet_max)   # e.g. ≤ 8 on AMD
```

When `need_intra_block_warps=True` (e.g. 2048 blocks / 256 CUs = 8.0 wf/CU ≤ 8),
the CU can only switch among ≤ 8 wavefronts while waiting for HBM.  Little's Law
requires ~8 resident wavefronts, so `num_warps=1` is borderline — empirically 1.4–2.8×
slower than `num_warps=8` on MI300X for the same block count.  The ILP bonus (extra
loads per thread hiding per-thread latency) does NOT compensate: the bottleneck is the
number of independent wavefronts the CU scheduler can choose from, not per-thread
instruction parallelism.  Score is capped at **0.82**.

When `need_intra_block_warps=False` (e.g. 16 384 blocks / 256 CUs = 64 wf/CU >> 8),
the CU naturally switches between 64 blocks' wavefronts; per-block warp count becomes
less critical.  The normal ILP correction applies: `num_warps=1` with ≥ 8 elements/
thread scores 1.0 (compiler-pipelined loads hide latency), and 0.85 otherwise.

**High-warp configs (`num_warps > 12`, e.g. `num_warps=16`):**  Score is **0.88**
regardless of ILP.  Rationale: 16 wavefronts per block provides ample CU-level latency
hiding (16 >> 8 needed) and is empirically faster than `num_warps=1` (0.82) for kernels
with few blocks per CU.  The 0.88 ceiling (vs the sweet-spot 1.0) reflects that extra
SPI initialisation cost and reduced VGPR budget per wavefront still impose a small
penalty relative to the ideal 4–8 range.

**Launch-bound, multi-block (`saturation < 0.25, num_blocks > 1`):**  Each block lands
on a separate CU; latency hiding is provided by grid spread.  The per-block score is a
first-principles overhead ratio penalising extra SPI warp init cost:

```
T(nw) = K_launch + (nw − 1) × K_warp        K_launch=3.0 µs, K_warp=0.2 µs
score  = K_launch / T(nw) = 3.0 / (3.0 + (nw−1) × 0.2)
```

> **Reading the multi-block equation:**
> When each block lands on a *separate* CU and the grid is small (`saturation < 0.25`),
> all latency hiding comes from CU-to-CU parallelism, not intra-block wavefront
> switching.  Extra warps per block add SPI allocation cost without hiding more latency.
> Score: `nw=1 → 1.00`, `nw=4 → 0.83`, `nw=16 → 0.50`.
>
> This path applies only for *under-saturated* grids (total wavefronts < 25% of CU
> capacity).  Kernels with enough blocks to be well-saturated use the sweet-spot model
> above — with the `wf_per_cu` correction distinguishing CU-rich from CU-poor configs.

An ILP correction within the launch-bound multi-block path is scaled by CU utilisation
(`cu_util = num_blocks / num_CUs`) to avoid over-scoring configs where the GPU is mostly
idle: `ilp_scale = min(1.0, cu_util / 0.25)`.  Full ILP benefit applies only at ≥ 25%
CU utilisation.

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
   launching it?  For **single-block** kernels, a config that does very little work per
   block is slow because the GPU spends most of its time just *starting* blocks rather
   than executing them.  For **multi-block** kernels, the ~3 µs dispatch cost is already
   shared across all blocks and is a small fraction of total runtime — so the scoring
   becomes nearly flat (all reasonable configs score 0.88–1.00) and the Grid factor takes
   over as the primary block-count discriminator.

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

   **Key subtlety — CU-level vs block-level wavefront switching:**
   The scoring model uses `num_warps` (wavefronts *per block*), but what actually
   matters for latency hiding is the total wavefronts each CU sees.  This comes from two
   sources simultaneously:

   - **Intra-block:** the `num_warps` declared in the config.  With `num_warps=8` and
     1 block on a CU, the CU has 8 wavefronts to switch between.
   - **Inter-block (CU-level switching):** if many blocks are dispatched, each CU may
     hold *multiple* blocks at once.  With 4096 blocks on 256 CUs, each CU hosts ~16
     blocks simultaneously — providing 16 independent wavefronts to switch between even
     if every block only declares `num_warps=1`.

   So `num_warps=1` is **not always bad**.  When there are many blocks (`blocks/CU >
   sweet_max ≈ 8`), the CU already has enough wavefront targets from block-level
   switching, and each thread's multiple loads (ILP) further hide per-load latency.
   But when blocks are scarce (`blocks/CU ≤ 8`), the CU may only see 4–8 wavefronts
   total — right at the threshold where every load causes a visible stall.  Empirically
   on MI300X, `num_warps=1` with 8 blocks/CU is 1.4–2.8× slower than `num_warps=8`
   for memory-bound streaming kernels.

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

`_estimate_spill_risk(config_dict, problem_metadata, kernel_metadata)` in
`triton_heuristics.py` estimates VGPR demand from first principles:

```
# ── Step 1: hardware budget ───────────────────────────────────────────────
threads_per_block = num_warps × warp_size
raw_max           = 65536 // threads_per_block          # AMD CDNA2/3: 65 536 VGPRs/CU
max_vgprs         = min(256, (raw_max // 8) × 8)        # round DOWN to 8-reg granule

# ── Step 2: estimated demand (additive components) ────────────────────────
base           = 16                         # loop-control vars, predicate regs, pid
tensor_regs    = (num_inputs + num_outputs) × 8   # pointer + value + mask per tensor
ops_regs       = n_fast × 1                # fast ops (add/mul/FMA) — high register reuse
               + n_med  × 3               # medium ops (sqrt, abs, int div) — 1-2 temps each
               + n_slow × 6               # slow ops (exp, log, sin, FP div) — ~5 temp regs
dim_regs       = 12  if YBLOCK > 0        # 2-D: y-index, y-stride, y-offset, extra predicate
               + 18  if ZBLOCK > 0        # 3-D: two more loop vars + strides + predicates
unroll_depth   = XBLOCK × max(1,YBLOCK) × max(1,ZBLOCK) // warp_size
unroll_regs    = min(unroll_depth, 16)    # capped at 16; compiler reuses beyond that
pipeline_regs  = max(0, num_stages − 1) × 8  # software-pipeline buffers two load copies

estimated_vgprs = base + tensor_regs + ops_regs + dim_regs + unroll_regs + pipeline_regs
estimated_vgprs = round_up_to_multiple_of_8(estimated_vgprs)   # mirrors hardware allocator
```

Key differences from a naïve estimate:
- `ops_regs` weights the three instruction classes separately (1 / 3 / 6 cycles of
  temporaries) rather than a flat multiplier.  This correctly gives a higher VGPR demand
  to kernels heavy in `exp`/`log` than to kernels that only do FMA chains.
- `unroll_regs` grows with the tile *volume* (`XBLOCK × YBLOCK × ZBLOCK`) not just
  `XBLOCK`, because Triton unrolls all three loop axes.  Capped at 16 because the
  compiler reuses registers across unrolled iterations once the live-set stabilises.
- `pipeline_regs` accounts for software pipelining (`num_stages > 1`): each extra stage
  effectively needs a duplicate set of load-buffer registers.
- The final `estimated_vgprs` is rounded **up** to the next multiple of 8, mirroring the
  hardware's 8-register granule allocator.  This produces the same "wasted" ceiling the
  real hardware experiences.

The estimate intentionally runs ~10–20% high (conservative) to avoid suppressing valid
configs.  The ratio `estimated_vgprs / max_vgprs` is shown in the verbose scoring table
as `low`, `MED`, or `HIGH` — purely informational, does not alter ranking:

```
ratio > 1.0  → HIGH  (very likely to spill after compile)
ratio > 0.75 → MED   (at risk — worth monitoring)
ratio ≤ 0.75 → low   (probably safe)
```

#### Mechanism B — Post-compile eviction (optional, env-controlled)

When `TORCHINDUCTOR_HEURISTICS_SPILL_BUFFER > 0`, the compile pool is expanded from
`top_N` to `top_N + buffer` configs.  After `_make_launchers()` runs (which populates
`launcher.n_spills` from the compiled binary metadata), `autotune_to_one_config` runs a
**pre-benchmark eviction pass** before any GPU timing occurs:

```python
spill_threshold = inductor_meta.get('spill_threshold',
                                    32 if torch.version.hip else 16)

# Build fast lookup: effective hdict → compiled launcher
hdict_to_launcher = {
    tuple(sorted(launcher_hdict(ln).items())): ln
    for ln in self.launchers
}

# Pre-filter buffer pool to only non-spilling candidates (in score order)
buffer_candidates = [
    hdict for hdict in full_ranked[len(top_n_sel):]   # ranked N+1, N+2, …
    if hdict_to_launcher[key(hdict)].n_spills <= spill_threshold
]

# Walk top-N in score order; swap out any that spill
buf_idx = 0
for i, hdict in enumerate(new_top_n):
    if hdict_to_launcher[key(hdict)].n_spills > spill_threshold:
        if buf_idx < len(buffer_candidates):
            new_top_n[i] = buffer_candidates[buf_idx]
            buf_idx += 1
        # else: no buffer replacement available → bench() returns inf as fallback
```

The spill threshold defaults to **32 for ROCm** and **16 for CUDA**
(overridable via `inductor_meta['spill_threshold']`).  Evicted configs are never
benchmarked — this avoids wasting GPU time on kernels that will return `inf` from
`bench()` and cannot be selected.

If all top-N configs spill even after eviction (buffer exhausted), `autotune_to_one_config`
walks `_HEURISTICS_FULL_RANKED_HDICTS` (the full score-ordered list) and selects the
fastest non-`inf` timing from the extended compile pool.  As a last resort, if *every*
compiled config spills, it picks the one with the fewest spills (minimum `inf` timing).

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

```python
# Real-bench mode: compile and time all configs
self.configs = all_generated_configs          # no pruning
# ...after benchmarking:
timings = {cfg: bench(cfg) for cfg in all_configs}

# Selection restricted to heuristic top-N (stored in _TOP_N_CONFIGS_FOR_SELECTION)
top_n_timings = {cfg: t for cfg, t in timings.items() if cfg in top_n_set}
best = min(top_n_timings, key=top_n_timings.get)
```

This means real-bench mode collects ground-truth timing for every config while still
testing whether the heuristic selection would have chosen the right config.

#### `bench()` — spill-skipped configs always recorded

A critical implementation detail: when a compiled config exceeds the register-spill
threshold (`launcher.n_spills > spill_threshold`), `bench()` short-circuits and returns
`float("inf")` — but **before** returning it records the `inf` timing in
`_HEURISTICS_VALIDATION_DATA`:

```python
if launcher.n_spills > spill_threshold:
    if _heur_track:
        _store_actual_timing(problem_key, config_dict, float("inf"))   # record FIRST
    return float("inf")                                                # then exit
```

Without this fix, spill-skipped configs would be absent from `actual_timings`, and
`_print_heuristics_validation_summary` would see an empty timing list for 2-D/3-D
kernels whose large block products (`XBLOCK × YBLOCK`) push every top-N config over the
VGPR limit — producing no summary output at all (the silent-failure bug for multi-block
kernels seen in `tuning10_verbose.log`).

#### Validation summary — rank labels and fallback handling

`_print_heuristics_validation_summary` uses the following logic to determine what to
show as the "heuristic pick":

1. **Chosen from top-N pool (normal case):** Walk `actual_timings` in ascending order
   and find the first config that is (a) in `_TOP_N_CONFIGS_FOR_SELECTION` and (b) has
   a finite timing.  This is the config that actually runs at inference time.  Label:
   `📊 HEURISTIC CHOSEN  (fastest from top-N selection pool)`.

2. **Spill-skipped top predictions:** If the highest-scored predicted configs all have
   `inf` timing (register spill), walk `predicted_sorted` to find the first config with
   a finite benchmark result.  The summary label reflects the skip:
   `📊 PREDICTED #3  (ranks 1–2 spill-skipped, using next available)`.

3. **All predicted configs spill:** If *every* predicted config has `inf` timing, fall
   back to the best finite timing across all benchmarked configs regardless of rank.

The `actual_inf_count` (number of `inf` timings in `actual_timings`) is tracked and
used to distinguish between "spilled" (expected for 2-D configs) and "compile-failed"
(unexpected) configs in diagnostic output.

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


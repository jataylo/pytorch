# ROCm Pointwise Kernel Heuristics — Design & Implementation

**Status:** V5 (Kernel-Aware)  
**Files:** `triton_heuristics_pointwise.py`, `triton_heuristics_adaptive.py`,
`triton_heuristics_hardware.py`, `triton_heuristics_kernel_analysis.py`,
`runtime/triton_heuristics.py`

---

## 1. Motivation

Triton's default autotuner benchmarks every candidate config at compile time.
For ROCm builds this is expensive: a typical pointwise kernel has 60–120
candidate (block, warp) combinations, each requiring a GPU kernel launch and
timing measurement.  The heuristic system replaces exhaustive autotuning with a
**static scoring pass** that selects the best 5 configs before a single kernel
is compiled, reducing cold-start cost by 5–10×.

Critically, scoring happens **after codegen** so the exact Triton kernel source
(`self.fn.src`) is available.  This lets us measure the actual instruction mix
rather than guessing, making the model fundamentally more accurate than any
purely structural approach.

---

## 2. Control Knobs

| Env variable | Values | Effect |
|---|---|---|
| `TORCHINDUCTOR_POINTWISE_HEURISTICS` | `0` / `1` (default on ROCm) | Enable/disable the entire heuristics system |
| `TORCHINDUCTOR_HEURISTICS_REAL_BENCH` | `0` / `1` | `1` = benchmark **all** configs but restrict winner selection to the heuristic top-5; used for validation |
| `inductor_config.heuristics_real_bench` | `bool` | Python-level mirror of the env var |

---

## 3. File Map

```
triton_heuristics_hardware.py         — Hardware abstraction layer
  └─ ArchitectureConfig               — Dataclass: CUs, warp size, cache sizes,
                                        mathematically-derived optimal values
  └─ get_architecture_config()        — Singleton factory (cached)

triton_heuristics_kernel_analysis.py  — Kernel code parser
  └─ extract_kernel_metadata()        — Counts ptr args, fast/medium/slow ops,
                                        broadcast patterns, mask usage
  └─ get_instruction_mix_efficiency() — Maps op mix → compute efficiency scalar

triton_heuristics_adaptive.py         — Time model + adaptive weights
  └─ BottleneckAnalysis
       ├─ _get_device_constants()     — BW / TFLOPS / cache from device props
       ├─ estimate_overhead_time_us() — Launch + grid setup time
       ├─ estimate_memory_time_us()   — HBM / L2 / L1 time with cache model
       ├─ estimate_compute_time_us()  — Raw compute throughput time (always >0)
       ├─ analyze_bottleneck()        — Roofline: total = overhead + max(mem,cmp)
       └─ get_adaptive_weights()      — Continuous weight interpolation

triton_heuristics_pointwise.py        — Scoring factors + config generation
  └─ PointwiseHeuristics
       ├─ estimate_memory_bandwidth() — BW score per config
       ├─ estimate_launch_overhead()  — Launch score per config
       ├─ estimate_grid_efficiency()  — Grid score per config
       ├─ estimate_occupancy()        — Occupancy score per config
       ├─ score_config()              — Weighted geometric mean of factors
       ├─ prune_configs()             — Sort + return top-N
       └─ generate_all_candidate_configs() — Enumerate valid (block, warp) pairs

runtime/triton_heuristics.py         — Inductor integration
  ├─ _apply_pointwise_heuristics()   — Generates & validates candidate configs
  ├─ pointwise()                     — Decorator: stashes pending data, creates
  │                                    direct Config objects (no scaling)
  └─ CachingAutotuner
       ├─ __init__()                 — Entry point; calls scoring when pending
       └─ _score_and_prune_heuristic_configs()
            — Extracts fn.src, runs prune_configs(), prints analysis box +
              scoring table, applies REAL_BENCH / heuristics-only logic
```

---

## 4. End-to-End Data Flow

```
PyTorch Inductor codegen
        │
        ▼
@triton_heuristics.pointwise(size_hints, …)   ← called at module load time
        │
        ├─ _apply_pointwise_heuristics()
        │    ├─ _convert_to_pointwise_heuristics_metadata()
        │    │    └─ builds problem_metadata from size_hints + inductor_meta
        │    ├─ PointwiseHeuristics.generate_all_candidate_configs()
        │    │    └─ enumerate (XBLOCK[×YBLOCK[×ZBLOCK]], num_warps) pairs
        │    │       filtered to: threads ∈ [64,1024], dims ≤ size_hint, pow2
        │    └─ structural validation (dim count, thread bounds)
        │
        ├─ Stash in inductor_meta['_heuristics_pending']
        │    └─ {problem_metadata, size_hints}
        │
        ├─ Convert each heuristic dict → triton.Config DIRECTLY
        │    (bypass triton_config() scaling — see §6.1)
        │
        └─ If REAL_BENCH: set inductor_meta['_heuristics_skip_cache_read']=True
                          (skip on-disk cache read — see §6.2)

cached_autotune(configs, …)
        │
        ├─ unique_configs()         ← dedup by (kwargs, num_warps)
        ├─ check_autotune_cache()   ← on-disk cache; skipped if skip_cache_read
        └─ CachingAutotuner.__init__(fn, configs, …)
                │
                ├─ self.fn = fn        ← JITFunction with .src and .arg_names
                ├─ self.configs = configs
                │
                └─ _score_and_prune_heuristic_configs()   [if pending & no mem-cache hit]
                        │
                        ├─ kernel_code = str(self.fn.src)
                        ├─ Refine num_inputs/num_outputs from self.fn.arg_names
                        │
                        ├─ PointwiseHeuristics.prune_configs(configs, meta, kernel_code)
                        │    └─ for each config:
                        │         score_config() → weighted geometric mean of 4 factors
                        │              ├─ get_adaptive_weights()  ← calls analyze_bottleneck()
                        │              │    ├─ extract_kernel_metadata(kernel_code)
                        │              │    ├─ estimate_overhead_time_us()
                        │              │    ├─ estimate_memory_time_us()   [cache-aware]
                        │              │    ├─ estimate_compute_time_us()  [mix-aware]
                        │              │    └─ roofline: total=overhead+max(mem,cmp)
                        │              └─ exponent = f(weight) → score factor
                        │
                        ├─ Print "Kernel & Problem Analysis" box
                        ├─ Print scoring table (rank, score, times, factor scores)
                        │
                        ├─ REAL_BENCH mode:
                        │    └─ keep ALL self.configs; store top-5 in
                        │       _TOP_N_CONFIGS_FOR_SELECTION[problem_key]
                        └─ Heuristics-only mode:
                             └─ self.configs = top_5_triton  (prune before compile)

Autotuner benchmarks self.configs
        │
        ├─ Heuristics-only: benchmarks top-5 only; picks fastest
        └─ REAL_BENCH: benchmarks all; picks fastest FROM top-5 only
                └─ _print_heuristics_validation_summary() shows
                   predicted rank vs actual rank + perf comparison
```

---

## 5. Scoring Model

### 5.1 Four Factors

Each config gets an individual score per factor (all ∈ [0, 1]):

| Factor | Key insight | Shape |
|---|---|---|
| **Memory BW** | Gaussian peak at `arch.optimal_threads_bandwidth` (~1536 threads / 24 wavefronts on MI300X from a self-consistent Little's Law), σ = 1.5 × optimal.  The wider σ reduces the score gap between XBLOCK=1024 (≈0.994) and XBLOCK=512 (≈0.977) from 3.7 % to 1.7 %, keeping smaller blocks competitive.  **1-D and 2-D kernels**: XBLOCK ≥ 16 → factor 1.00; XBLOCK < 16 → factor 0.75–0.98.  **3-D kernels** use shape-aware coalescing: for *cube-like* problems (max/min dim ratio < 2, e.g. 64×64×64) both X and Y accesses are symmetric so `warp_align = 0.75 + 0.25 × avg(cl_x, cl_y)` (0.75–1.00) is applied; for *non-cube* problems (ratio ≥ 2) only the XBLOCK sub-cache-line rule applies (XBLOCK ≥ 16 → 1.00). | Smooth decay; warp-align 0.75–1.00 |
| **Launch overhead** | Amortize dispatch cost: elements/block should exceed `arch.optimal_elements_per_block`.  One secondary penalty: large-grid batch-dispatch when `num_blocks > 4×num_CUs`.  Per-block warp count is not penalised here — the occupancy model owns that signal. | Step-up with soft threshold |
| **Grid efficiency** | Grid should saturate all CUs.  Target: **1-D/2-D** → `2× arch.num_cus`; **3-D cube-like** (ratio < 2) → `arch.num_cus` (L2 locality prefers fewer, larger tiles); **3-D non-cube** (ratio ≥ 2) → `4× arch.num_cus` (mixed-stride traffic needs more wavefronts). | Peak near target, penalise over/under |
| **Occupancy** | Wavefronts/CU in sweet spot `[arch.occupancy_sweetspot_min, arch.occupancy_sweetspot_max]`.  ILP correction applied in **both** the well-saturated path and the launch-bound path: `num_warps=1` with ept ≥ 8 scores 1.00 (compiler-pipelined loads fully hide HBM latency for the single wavefront), ept ≥ 4 scores 0.93. | Plateau in sweet spot; ILP-aware in all paths |

### 5.2 Weighted Geometric Mean

```
score = (BW^a × Launch^b × Grid^c × Occupancy^d) ^ (1/(a+b+c+d))
```

Exponents `a,b,c,d` are mapped from weights via:
```
exponent = 0.5 + (weight − 0.10) / 0.40 × 2.5    [clamped to (0.5, 3.0)]
```

### 5.3 Adaptive Weights

Weights are computed **per-config** from the bottleneck analysis, not globally.
Pure-regime anchor vectors (each sums to 1.0):

| Factor     | Overhead-bound | Memory-bound | Compute-bound |
|------------|----------------|--------------|---------------|
| bandwidth  | 0.10           | 0.55         | 0.15          |
| launch     | 0.65           | 0.10         | 0.10          |
| grid       | 0.10           | 0.15         | 0.30          |
| occupancy  | 0.15           | 0.20         | 0.45          |

Interpolation (continuous, not switch-based):
```
w[k] = overhead_frac × OVERHEAD_W[k]
     + memory_frac   × MEMORY_W[k]
     + compute_frac  × COMPUTE_W[k]
```

This means a kernel with AI=1 (deep in memory territory) gets BW weight ~0.55,
while a kernel with AI=180 (barely memory-bound) gets a much more balanced mix.

---

## 6. Key Engineering Decisions

### 6.1 Config Creation: Direct `triton.Config`, not `triton_config()`

`triton_config()` applies a scale-up heuristic that can collapse 85+ distinct
(XBLOCK, YBLOCK, num_warps) triples into a single unique `triton.Config` after
`unique_configs()` deduplication.  The heuristics already pick block sizes
appropriate for the problem, so they are now wrapped directly:

```python
# Old (collapsed 85 → 1 after unique_configs)
triton_config_with_settings(size_hints, xblock, yblock, num_warps=nw)

# New (preserves all distinct configs)
Config({'XBLOCK': min(xblock, size_hints['x']),
        'YBLOCK': min(yblock, size_hints['y'])},
       num_warps=nw, num_stages=1)
```

### 6.2 REAL_BENCH Mode and On-Disk Cache

Without intervention, `check_autotune_cache()` reads the best config from a
prior run and returns `configs=[best_config]` before `__init__` runs.  In
REAL_BENCH mode every config must be benchmarked, so:

- `pointwise()` sets `inductor_meta['_heuristics_skip_cache_read'] = True`
- `check_autotune_cache()` skips the `read_best()` call when this flag is set
- The `autotune_cache` object is still created so the winner is **written** after the run

### 6.3 Scoring After Codegen via `self.fn.src`

Scoring is deliberately deferred to `CachingAutotuner.__init__` so that
`self.fn.src` (the compiled Triton kernel source) is available.  This avoids
file I/O and gives access to `self.fn.arg_names` for precise tensor-count
detection.

### 6.4 Roofline Model in `analyze_bottleneck`

```
total_us = overhead_us + max(memory_us, compute_us)
```

`memory_us` and `compute_us` overlap on the GPU (memory controller and ALUs
run in parallel).  `estimate_compute_time_us()` always returns the raw
compute-throughput-limited time regardless of regime; the caller decides which
side of the roofline dominates.

Fractions for adaptive weights are computed from the **gross** sum
`overhead + memory + compute`, not from `total_us`.  This preserves information
about how large each component is even when memory and compute strongly overlap.

### 6.5 Device Constants with Three-Level Fallback

`_get_device_constants()` tries in order:
1. Custom ROCm properties: `props.memory_bandwidth_gb_s`, `props.compute_throughput_tflops`
2. Calculated from standard CUDA properties: `memoryClockRate × memoryBusWidth` and `multi_processor_count × clockRate × ops_per_cu`
3. Conservative hard-coded defaults (900 GB/s, 200 TFLOPS)

This ensures the function **never raises**, so a missing property never silently
zeros out scores.

---

## 7. Debug Output

### Analysis Box (printed once per kernel)
```
[HEURISTICS] ┌─ Kernel & Problem Analysis ──────────────────────────────────────────┐
[HEURISTICS] │  Source         :  fn.src (890 chars)
[HEURISTICS] │  Tensor args    :  4 ptr args  →  3 inputs  /  1 output(s)
[HEURISTICS] │  Instruction mix:  16 fast (add/mul/fma)  ·  0 medium  ·  0 slow
[HEURISTICS] │  Ops / element  :  16 (weighted)   ·   Broadcast: ✗   ·   Masking: ✓
[HEURISTICS] ├─ Problem ───────────────────────────────────────────────────────────┤
[HEURISTICS] │  2,097,152 elements  ·  4 B/scalar  ·  16.0 B/elem (r+w)  →  32768.0 KB
[HEURISTICS] │  Total FLOPs    :  ~33.55M   ·   Arithmetic intensity: 1.00 ops/byte
[HEURISTICS] ├─ Device / Roofline ──────────────────────────────────────────────────┤
[HEURISTICS] │  Peak HW        :  200.0 TFLOPS  ·  900.0 GB/s  →  peak OI ceiling 222.2
[HEURISTICS] │  Effective       :  160.0 TFLOPS (×0.8)  ·  720.0 GB/s (×0.8)  →  eff. OI ridge 222.2
[HEURISTICS] │  Regime          :  MEMORY-BOUND  (AI 1.00 < ridge 222.2)
[HEURISTICS] └────────────────────────────────────────────────────────────────────────┘
```

### Scoring Table (printed once per kernel)
```
Rank  Score  Bottleneck    Ohd_µs  Mem_µs  Cmp_µs  Tot_µs │  BW   Lnch  Grid  Occup │  Blks  Thr/blk  Config
  #1  0.937  💾 MEMORY      4.32   23.30    0.10   27.62  │ 0.90  0.83  0.75  0.93  │  2048      512  {…}  ◄ top-5
```

### Validation Summary (REAL_BENCH mode only)
Printed after all benchmarks complete; shows predicted vs actual ranking
and a bottleneck breakdown for the heuristic pick and the true winner.

---

## 8. Known Limitations & Potential Oversights

### 8.1 `triton_config()` for non-heuristic paths still uses scale-up logic

The fix (§6.1) only applies to configs generated *by the heuristics*.  The
fallback code paths for non-ROCm or disabled heuristics still call
`triton_config_with_settings()`.  This is intentional (the original configs
were hand-tuned for that code path) but means the two code paths are
asymmetric.

### 8.2 `num_stages=1` hard-coded for all heuristic configs

`Config({…}, num_warps=nw, num_stages=1)` always uses 1 pipeline stage.
Software pipelining (num_stages=2) can hide memory latency for some patterns
but is never explored by the heuristics.

### 8.3 L1 cache model uses a fixed 32 KB per CU

`_get_device_constants()` hard-codes `l1_cache_size = 32 * 1024` for AMD
regardless of the actual GPU (CDNA2 / CDNA3 / RDNA3 values differ; CDNA3 uses
256 KB LDS that can partly act as L1).  The NVIDIA path also uses `128 * 2048`
without checking the actual shared-memory configuration.

### 8.4 `estimate_overhead_time_us()` uses a fixed 3 µs launch time

The base launch overhead is hard-coded.  Real ROCm dispatch latency varies
by ~2–6 µs depending on kernel complexity and driver state.  This constant
has the highest impact on tiny-kernel scoring.

### 8.5 Instruction counting is line-count-based, not AST-based

`extract_kernel_metadata()` uses `str.count('tl.exp')` etc.  A line like
`# tl.exp is slow` or a string literal inside the kernel would inflate counts.
Pattern-matching on Python arithmetic operators (`+`, `-`, `*`) also over-counts
index arithmetic, not just element-wise math operations.

### 8.6 `ops_per_element` is not scaled by vector width

The parser counts *instruction occurrences* not *element throughput*.  Triton
often vectorises loads/stores across a block, so one `tl.exp` processes many
elements.  `ops_per_element` should ideally reflect the per-element cost.

### 8.7 Grid efficiency score ignores multi-dimensional grids

The grid-efficiency factor uses `num_blocks = prod(grid_dims)` as a flat
count.  2D and 3D kernels can have very different scheduling characteristics
(scheduler uses X-dimension for block distribution) but the score doesn't
distinguish.

### 8.8 No waves_per_eu tuning

On ROCm, `waves_per_eu` is a config knob that controls the number of
wavefronts resident per CU beyond the VGPR-limited maximum.  The heuristics
never explore this dimension.

### 8.9 `_arch_config` is process-level singleton, not device-indexed

`PointwiseHeuristics._arch_config` is cached on the class the first time it is
read.  In a multi-GPU setup where GPU 0 and GPU 1 have different architectures
(e.g., MI300 + MI350) the wrong architecture constants will be used for the
second device.

### 8.10 Validation summary only covers REAL_BENCH mode

There is no lightweight accuracy tracking in production
(`TORCHINDUCTOR_POINTWISE_HEURISTICS=1` without `REAL_BENCH`).  A rolling
accuracy counter sampled at low rate would let you detect regressions without
the full benchmarking cost.

---

## 9. Suggested Improvements (roughly highest-impact first)

### 9.1 Use Triton's own IR / AST for instruction counting (High impact)

Instead of grepping the Python source, parse the kernel after Triton lowers to
its `triton.language` IR or TTIR.  This gives exact instruction counts, types,
and vectorisation factors with zero false positives.  Alternatively, intercept
Triton's codegen phase to annotate the `JITFunction` with pre-computed metadata
at emit time.

**Benefit:** Eliminates all the operator over-counting bugs in §8.5/§8.6;
enables accurate `ops_per_element` (currently the biggest source of compute-time
estimation error).

### 9.2 Calibrate launch overhead from real measurements (High impact)

The 3 µs constant (§8.4) dominates tiny-kernel scores.  Run a one-time
calibration pass at process startup (a null kernel dispatch) and cache the
result.  This is already possible inside `CachingAutotuner` before the first
`bench()` call.

**Benefit:** Directly improves accuracy for all kernels ≤ 16 K elements, which
are otherwise hardest to classify.

### 9.3 Add `waves_per_eu` as a search dimension (Medium-High impact)

Extend `generate_all_candidate_configs()` to also enumerate `waves_per_eu ∈
{0, 1, 2, 4}` (0 = auto).  Add a `waves_per_eu` scoring term to the occupancy
factor.  On MI300/MI350, `waves_per_eu=2` has empirically shown ~20%
improvement on bandwidth-bound kernels.

**Benefit:** Expands the search space to cover a dimension known to matter on
ROCm without adding compile cost (still only top-5 are compiled).

### 9.4 Top-N XBLOCK diversity pass (Implemented)

`prune_configs()` now limits the returned top-N list to at most 2 configs per
distinct XBLOCK value.  Without this, the scoring loop was observed to fill all
5 top-N slots with XBLOCK=1024 variants (differing only in `num_warps`),
preventing XBLOCK=512 and XBLOCK=256 candidates from ever being compiled and
benchmarked.  Empirically, 42 % of 1-D full-misses were caused by this effect.

Algorithm: after sorting by composite score, a primary pass greedily takes up
to 2 configs per XBLOCK bucket; overflow configs (those that hit the cap) fill
any remaining top-N slots in score order, ensuring the list is always full.

### 9.4a 3-D coalescing: shape-aware (cube symmetric, elongated XBLOCK-only)

For 3-D kernels the coalescing penalty is **problem-shape-aware**:

**Cube-like problems** (max/min dimension ratio < 2, e.g. 64×64×64):  
All three loop axes are nearly the same length and each can be stride-1 in
permute/broadcast kernels.  Wide YBLOCK tiles genuinely improve spatial reuse
because both X and Y accesses are symmetric.  The model averages X and Y
coalescing:

```
cl_x = min(XBLOCK, 16) / 16
cl_y = min(YBLOCK, 16) / 16
warp_align = 0.75 + 0.25 × (cl_x + cl_y) / 2    # range [0.75, 1.00]
```

This correctly scores `{X16,Y16}` higher than `{X16,Y8}` for 64×64×64, where
the empirically best configs use wide, balanced tiles.

**Non-cube problems** (ratio ≥ 2, e.g. 256×64×32):  
The YBLOCK axis is often mixed-stride: stride-1 for one tensor, stride-N for
another.  Applying a YBLOCK penalty is empirically harmful here — `{X32,Y8}`
beats `{X16,Y16}` by 3–5 % on MI300X.  Only the XBLOCK sub-cache-line rule
applies:

```
warp_align = 1.00  if XBLOCK ≥ 16
warp_align = 0.75 + 0.23 × (XBLOCK / 16)  otherwise   # range [0.75, 0.98]
```

The threshold of ratio < 2 (not < 4) ensures the cube path is used only for
genuinely symmetric workloads; moderately elongated shapes like 128×64×64
(ratio=2) take the XBLOCK-only path.

### 9.4b ILP boost in launch-bound occupancy path (Implemented)

The ILP correction (high `elements_per_thread` ≥ 8 → score 1.00) was previously
only applied in the well-saturated occupancy branch.  Configs in the
launch-bound path (grid saturation < 0.25) with high ept were scored by the
`natural_warps` ratio alone, systematically under-estimating their capability.
The ILP check is now applied in both branches.

### 9.4c 3-D tile balance tie-breaker (Problem-shape aware)

For **cube-like problems** (max/min dimension ratio < 2, e.g. 64×64×64) a mild
multiplier prefers cube-like tiles and a minimum YBLOCK:

```
balance_mult = 1 − 0.003 × log2(max_tile / min_tile)
yblock_mult  = 1 − 0.008 × max(0, 4 − log2(YBLOCK))
score       *= max(0.97, balance_mult × yblock_mult)
```

Examples: `{16,16,8}` (tile ratio 2) → ×0.997; `{64,4,4}` (tile ratio 16)
→ ×0.988.  For cube problems the optimal tile is empirically cube-like, so
this mild bias is correct.

The threshold of ratio < 2 (tightened from the earlier < 4) ensures the
tie-breaker only fires on genuinely symmetric workloads.  Moderately elongated
shapes like 128×64×64 (ratio=2) **skip** the tie-breaker; their optimal tile
may itself be elongated (e.g. {X32,Y4,Z4}).

For **non-cube problems** (ratio ≥ 2) the tie-breaker is **skipped**.  These
benefit from tile shapes that align with the problem dimensions, and a
cube-shape bias would penalise those configs incorrectly.

### 9.4d 3-D grid target: shape-aware block count (Implemented)

The optimal number of thread-blocks depends on the 3-D problem shape:

| Shape | Target | Rationale |
|---|---|---|
| Cube-like (ratio < 2) | `arch.num_cus / 2` (~228 on MI300X) | Symmetric regular-stride access → excellent L2 spatial reuse. Fewer, larger blocks keep their working set in L2 across wavefront-switches. Empirically, 256-block configs beat 512-block ones for 64×64×64. |
| Non-cube (ratio ≥ 2) | `2 × arch.num_cus` (×2 vs 1-D/2-D) | Mixed-stride / permute access → irregular HBM traffic. More wavefronts needed per CU to hide round-trip latency; 1024-block configs beat 512-block ones by 2–5 % on MI300X. |

Using the standard 1-D/2-D target (2×CUs = 456) for cube problems caused
512-block configs to score Grid≈0.993 while the empirically best 256-block
configs scored only Grid≈0.920 — a 7 % gap that outweighed the BW and
coalescing advantages of balanced tiles.

### 9.5 Kernel-aware minimum-latency BW threshold (Medium impact)

The BW Gaussian is centred at `arch.optimal_threads_bandwidth` with σ = 1.5 ×
optimal, computed from a self-consistent Little's Law using effective
(L2-blended) latency.  The wider σ reduces the score gap between XBLOCK=1024
and XBLOCK=512 from 3.7 % to 1.7 %, working in concert with the diversity
pass to ensure a variety of block widths reach the compile-and-benchmark stage.

A further refinement would make the threshold per-kernel by using the actual
instruction mix from kernel metadata:

```
min_threads_for_latency_hiding = ceil(latency_cycles / (instructions_per_load × IPC))
```

Configs below this threshold would receive a near-zero BW score instead of a
smooth Gaussian decay, providing a sharper signal for highly memory-bound
kernels with few arithmetic instructions.

### 9.6 Fix L1 cache model to use real per-GPU values (Medium impact)

Query `props.l1_cache_size` or derive from `totalConstMem` / `sharedMemPerBlock`
for NVIDIA, and from `lds_size` for ROCm where available.  This fixes the
conservative model that currently treats everything ≤ 32 KB as "possible L1
hit" on MI300 which has 64 KB LDS per CU in some configurations.

### 9.7 Score configs in parallel (Medium impact)

`prune_configs()` calls `score_config()` sequentially.  With 85+ candidates
and per-config `analyze_bottleneck()` calls (each involving device-property
lookups), the scoring loop is the single longest-running part of the startup
path.  Scoring is embarrassingly parallel (no shared state); a
`concurrent.futures.ThreadPoolExecutor` with 4 workers would give ~3× speed-up
on the scoring step.

### 9.8 Per-problem-type weight calibration from REAL_BENCH data (Medium impact)

The pure-regime anchor weights (§5.3) were designed by reasoning about hardware.
With the REAL_BENCH validation summaries now being logged, it is possible to
collect `(predicted_rank, actual_rank)` pairs over a large model suite and
run a simple linear regression to find optimal anchor weights.  This closes the
loop between the analytical model and empirical data without changing the model
structure.

### 9.8 Per-device `_arch_config` caching (Low-Medium impact)

Replace the class-level singleton with a `dict` keyed by `device_index`.
Trivial to implement; prevents wrong-arch scoring in multi-GPU machines.

### 9.9 Explore `num_stages=2` for bandwidth-bound kernels (Low-Medium impact)

Add `num_stages=2` variants to the top few high-BW-scoring configs.  Triton's
software pipelining can prefetch the next tile while computing the current one,
which helps on kernels near the BW/compute crossover.  The scoring model could
predict benefit from `stages>1` when `memory_frac > 0.6`.

### 9.10 Lightweight production accuracy tracking (Low impact, but essential for maintenance)

Outside REAL_BENCH mode, randomly sample ~5% of kernels and benchmark the top
heuristic pick vs the second-best pick.  Log the ratio.  This gives a
continuous "heuristic quality" metric without the 10× benchmarking cost, making
regressions visible in production.

---

## 10. Quick Reference — Accuracy vs Cost Trade-offs

| Mode | Configs compiled | Configs benchmarked | Selection pool |
|---|---|---|---|
| Heuristics-only (default) | Top-5 scored | Top-5 | All top-5 |
| REAL_BENCH validation | All N | All N | Top-5 scored only |
| No heuristics (baseline) | All N | All N | All N |

REAL_BENCH mode gives full ground truth but is as slow as the old autotuner.
Production use runs heuristics-only; REAL_BENCH is for accuracy measurement.


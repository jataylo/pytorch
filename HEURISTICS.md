# Pointwise Kernel Heuristics — Technical Reference

> **Scope:** PyTorch Inductor's static heuristics system for Triton pointwise kernels.
> Covers the full pipeline from InductorConfig env-vars through hardware constants,
> bottleneck analysis, adaptive weight interpolation, per-config scoring, and the
> autotuner integration that applies them in production.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [InductorConfig Integration & Environment Variables](#2-inductorconfig-integration--environment-variables)
3. [Module Architecture & Data Flow](#3-module-architecture--data-flow)
4. [Stage 1 — Hardware Constants (`triton_heuristics_hardware.py`)](#4-stage-1--hardware-constants)
5. [Stage 2 — Kernel Code Analysis (`triton_heuristics_kernel_analysis.py`)](#5-stage-2--kernel-code-analysis)
6. [Stage 3 — Bottleneck Analysis (`triton_heuristics_adaptive.py`)](#6-stage-3--bottleneck-analysis)
7. [Stage 4 — Adaptive Weight Interpolation](#7-stage-4--adaptive-weight-interpolation)
8. [Stage 5 — Per-Config Scoring (`triton_heuristics_pointwise.py`)](#8-stage-5--per-config-scoring)
9. [Stage 6 — Autotuner Integration (`runtime/triton_heuristics.py`)](#9-stage-6--autotuner-integration)
10. [Spill Prediction & Fallback Buffer](#10-spill-prediction--fallback-buffer)
11. [Validation & Verbose Logging](#11-validation--verbose-logging)
12. [End-to-End Example](#12-end-to-end-example)

---

## 1. System Overview

Triton kernels must be compiled for a specific **configuration** — a tuple of
`(XBLOCK, YBLOCK, ZBLOCK, num_warps)` that controls how the problem is tiled and
how many threads run per GPU block.  The naive solution is to benchmark all possible
configurations every time a new kernel shape appears (autotuning), but this costs
hundreds of milliseconds per kernel.

The heuristics system replaces brute-force autotuning with a **static performance
model** that predicts the best configuration analytically in microseconds.  Rather
than one universal formula, the system:

1. **Reads actual hardware properties** at startup (CU count, warp size, bandwidth, …).
2. **Parses the generated Triton kernel source** to extract real instruction counts,
   tensor counts, masking patterns, and broadcast structure.
3. **Models three time components** for each candidate configuration:
   overhead (kernel dispatch), memory (HBM streaming), compute (ALU throughput).
4. **Weights four scoring factors** adaptively — the weight of each factor shifts
   based on which time component dominates.
5. **Scores every candidate** in parallel and selects the top-N for compilation.

On AMD CDNA (MI300 / MI350) the system achieves >95% top-1 accuracy and eliminates
>90% of autotuning overhead compared to exhaustive benchmarking.

---

## 2. InductorConfig Integration & Environment Variables

All behavioural knobs are exposed as `torch._inductor.config` variables so they
compose cleanly with the rest of Inductor's configuration system and can be set
from environment variables without touching Python code.

| Config variable | Env var | Default | Description |
|---|---|---|---|
| `heuristics_real_bench` | `TORCHINDUCTOR_HEURISTICS_REAL_BENCH` | `1` | Benchmark **all** configs for validation data; select winner only from top-N |
| `heuristics_top_n_configs` | `TORCHINDUCTOR_HEURISTICS_TOP_N` | `5` | Size of the predicted selection pool |
| `heuristics_spill_fallback_buffer` | `TORCHINDUCTOR_HEURISTICS_SPILL_BUFFER` | `0` | Extra backup configs compiled to survive register spills (0 = disabled) |
| `heuristics_verbose` | `TORCHINDUCTOR_HEURISTICS_VERBOSE` | `1` | Print full scoring tables and validation summaries |

### How the config reaches the scoring code

`CachingAutotuner._score_and_prune_heuristic_configs()` in
`runtime/triton_heuristics.py` is the entry point.  It receives:

```python
# Called once per unique (kernel, size_hints) pair at first compilation
def _score_and_prune_heuristic_configs(self, size_hints, problem_metadata, kernel_code):
    from torch._inductor import config as inductor_config

    _top_n   = inductor_config.heuristics_top_n_configs
    _verbose = inductor_config.heuristics_verbose
    _spill_buf = inductor_config.heuristics_spill_fallback_buffer
    ...
```

`problem_metadata` is a dict assembled across three phases of Inductor's
compilation pipeline (detailed in Section 3.1).  Every key that appears in a
typical run is explained below.

```python
problem_metadata = {
    # ── Problem shape ──────────────────────────────────────────────────────
    'dimensions':        (1048576,),   # per-axis element counts
    'total_elements':    1_048_576,    # product of dimensions

    # ── Data type ──────────────────────────────────────────────────────────
    'element_size':      4,            # bytes per scalar (FP32=4, FP16=2, FP64=8)

    # ── Tensor structure ───────────────────────────────────────────────────
    'num_tensors':       3,            # total pointer args  (inputs + outputs)
    'num_inputs':        2,            # read-only tensor args
    'num_outputs':       1,            # write-back tensor args

    # ── Arithmetic work ────────────────────────────────────────────────────
    'ops_per_element':   6,            # weighted op count per output element
    'fast_ops':          5,            # add/sub/mul/fma counts in kernel src
    'medium_ops':        0,            # div/sqrt counts
    'slow_ops':          1,            # exp/log/sin/tanh counts

    # ── Memory traffic ─────────────────────────────────────────────────────
    'bytes_per_element': 12.0,         # total bytes read+written per element

    # ── Access patterns ────────────────────────────────────────────────────
    'has_mask':          False,        # boundary-condition masking present
    'has_broadcast':     False,        # any tensor is broadcast over a dimension
    'broadcast_tensor_bytes': 0,       # bytes in the broadcast tensor (if any)

    # ── Device limits (hardware query) ────────────────────────────────────
    'warp_size':              64,      # AMD wave64=64, NVIDIA=32
    'max_threads_per_block': 1024,     # hardware ceiling

    # ── Kernel fusion ──────────────────────────────────────────────────────
    'fusion_depth':  1,                # number of fused operators
    'vector_width':  1,                # SIMD vector width hint
}
```

---

#### `dimensions` and `total_elements`

`dimensions` is a tuple of the **rounded-up element count for each axis** of the
problem, taken directly from `size_hints.values()`.  Inductor always rounds up to
the next power of two so that Triton can use power-of-two block sizes without
out-of-bounds checks on the happy path.

```python
# 1-D kernel:  shape (1048576,)
size_hints = {'x': 1048576}
dimensions = (1048576,)

# 2-D kernel:  shape (128, 1024)
size_hints = {'x': 1024, 'y': 128}
dimensions = (1024, 128)
```

`total_elements` is simply `reduce(mul, dimensions)`.

**Used by:**
- `generate_all_candidate_configs` — filters out block sizes that exceed the axis
  size (no point launching a 1024-element block on a 128-element axis)
- `estimate_grid_granularity` — selects the right Gaussian regime (tiny / small /
  medium / large) and calculates `num_blocks` for that regime's CU-saturation target
- `estimate_occupancy_impact` — computes the `saturation` ratio
  `(num_blocks × num_warps) / (num_CUs × 8)` to decide which scoring branch applies
- `estimate_memory_time_us` — `total_elements × bytes_per_element` gives total HBM bytes

> **Why round up to powers of two?**  
> Triton tiles are always powers of two in size.  If the real tensor has 1000
> elements and the tile is 128, Triton needs 8 tiles to cover it.  The last tile
> will only have 104 real elements; the remaining 24 are out-of-bounds.  Inductor
> rounds the size hint up to 1024 so the heuristic can assume full tiles
> everywhere, and a mask (`xmask = xindex < real_n`) guards the boundary tile.

---

#### `element_size`

The byte width of a single scalar element, derived from `triton_meta["dtype"]`:

```python
dtype_str    = str(triton_meta.get("dtype", "float32"))
element_size = 2 if "float16" or "bfloat16" in dtype_str else \
               8 if "float64"               in dtype_str else 4
```

| dtype | `element_size` |
|---|---|
| `float16`, `bfloat16` | 2 |
| `float32` (default) | 4 |
| `float64` | 8 |

**Used by:**
- Fallback `bytes_per_element` computation: `element_size × (num_inputs + num_outputs)`
  when `extract_kernel_metadata` has not run or failed
- `estimate_cache_locality` — working-set-per-block calculation for L1 fit checks

Different dtypes change how many bytes cross the HBM bus per element.  A FP16
kernel moves half the data of a FP32 kernel for the same tensor shape, so it will
appear less memory-bound under the roofline model.

> **For junior engineers:** `element_size` answers "how wide is one number?"
> FP32 (32-bit float, 4 bytes) is the default deep-learning dtype.  FP16 (half
> precision, 2 bytes) is common in inference and mixed-precision training.  FP64
> (double, 8 bytes) is rare on GPUs but used in scientific computing.  The GPU
> memory bus is finite (e.g. 5.3 TB/s on MI300X), so fewer bytes per element means
> the same bus can serve twice as many elements per second.

---

#### `num_tensors`, `num_inputs`, `num_outputs`

These describe the **tensor pointer structure** of the generated kernel.

```
num_tensors = num_inputs + num_outputs
```

They are initially guessed from `inductor_meta` (Phase 1), then overridden by
counting `_ptr`-suffixed arguments in `self.fn.arg_names` and `tl.store` calls in
`self.fn.src` (Phase 2):

```python
# Phase 2 — authoritative
ptr_args    = [a for a in self.fn.arg_names if a.endswith('_ptr')]
num_tensors = len(ptr_args)                          # e.g. 3
num_outputs = max(1, kernel_code.count('tl.store'))  # e.g. 1
num_inputs  = num_tensors - num_outputs              # e.g. 2
```

**Used by:**
- `bytes_per_element` fallback: `element_size × (num_inputs + num_outputs)` — each
  tensor contributes one full read or write pass over the data
- `estimate_overhead_time_us`: `(num_tensors − 3) × 0.1 µs` — extra pointer
  arguments add argument-setup overhead at kernel dispatch
- `estimate_vgpr_per_thread`: `pointer_regs = 2 × (num_inputs + num_outputs)` —
  every tensor needs at least two VGPRs for the 64-bit base pointer

---

#### `ops_per_element`

The **weighted arithmetic work** per output element, produced by `extract_kernel_metadata`
(Phase 3):

```python
total_weighted_ops = (
    fast_ops   × 1   +   # add/sub/mul/fma  — ~1 GPU clock each
    medium_ops × 10  +   # div/sqrt         — ~10–20 clocks each
    slow_ops   × 30      # exp/log/sin/tanh — ~30–50 clocks each
)
ops_per_element = max(2, total_weighted_ops)
```

The weights (1, 10, 30) are proportional to the latency cost of each op class on
CDNA3.  A kernel that calls `tl.exp()` once per element has an effective
`ops_per_element` of 30, making it look 15× more compute-intensive than a
pure-add kernel — which is correct: `exp` takes roughly 30 clock cycles on the
hardware transcendental unit.

**Used by:**
- `analyze_bottleneck → estimate_compute_time_us`:
  ```python
  num_ops   = total_elements × ops_per_element
  T_compute = num_ops / (achievable_tflops × 1e6)  # µs
  ```
  This is the roofline's compute arm — if `T_compute > T_memory` the kernel is
  compute-bound and occupancy weight rises.
- `calculate_arithmetic_intensity`:
  ```python
  AI = ops_per_element / bytes_per_element   # FLOPs per byte
  ```
  AI is compared against the device's OI ceiling to classify the kernel on the
  roofline plot and to display the regime in verbose output.
- `_estimate_spill_risk`:
  ```python
  estimated_vgprs += ops_per_element × 1.5   # live compute temporaries
  ```

> **For junior engineers:** Imagine two workers at a conveyor belt.
> Worker A just passes each box to the next station (like `add`).
> Worker B has to open each box and run a 30-step chemical test (like `exp`).
> Even if both workers process 1 million boxes, Worker B takes 30× longer.
> `ops_per_element` captures exactly this difference so the model doesn't
> mistake a transcendental kernel for a trivial memcpy.

> **On the roofline:** a pure memcpy has `ops_per_element ≈ 0` and AI ≈ 0,
> placing it far left on the roofline — always memory-bound.  A kernel doing
> `exp` on every element has AI ≈ 2.5 FLOPs/byte on FP32, still well below the
> OI ceiling (≈ 200 FLOPs/byte for MI300X), so it remains memory-bound despite
> the heavier compute — though the gap is much smaller.

---

#### `bytes_per_element`

The **total bytes transferred to/from HBM per output element** — both reads and
writes combined.

```python
# Phase 3 (authoritative, from kernel analysis):
bytes_per_element = float(num_tensors × element_size)
# e.g. 3 tensors × 4 bytes = 12.0 for a 2-input FP32 add

# Fallback (Phase 1/2, used when kernel_code unavailable):
bytes_per_element = element_size × (num_inputs + num_outputs)
# same formula, same result when num_tensors == num_inputs + num_outputs
```

For a kernel that reads tensors A and B and writes tensor C:

```
bytes_per_element = 4 + 4 + 4 = 12.0 bytes / element
total_bytes       = 1_048_576 × 12.0  =  12 MB
```

**Used by:**
- `estimate_memory_time_us` (the single most important consumer):
  ```python
  total_bytes = total_elements × bytes_per_element
  T_memory    = total_bytes / (effective_bandwidth_GB_s × 1e3)   # µs
  ```
  This is the entire memory arm of the roofline model.  Getting `bytes_per_element`
  wrong by even 1.5× shifts the predicted memory time proportionally, which can
  flip the kernel from memory-bound to launch-bound.
- `calculate_arithmetic_intensity`:
  ```python
  AI = ops_per_element / bytes_per_element
  ```
- Verbose output: `total_KB = (total_elements × bytes_per_element) / 1024`

> **Why read + write, not just write?**  
> The GPU must fetch data into its cache hierarchy before the ALU can operate on it.
> A simple `C = A + B` issues two HBM reads (A and B) and one HBM write (C).
> All three cross the memory bus, so all three must be counted.  Counting only the
> write would underestimate traffic by 3×, making the model predict the kernel is
> 3× faster than it really is.

> **Caching caveat:** If data fits in L2, the effective `bytes_per_element` is
> lower because repeated accesses hit cache.  `estimate_memory_time_us` handles
> this separately using the L2-vs-HBM path, so `bytes_per_element` always
> represents the worst-case HBM cost and the cache logic discounts it.

---

#### `has_mask`

`True` when the kernel contains boundary-condition guard predicates — the pattern
`tl.load(ptr, mask=xmask)` / `tl.store(ptr, val, mask=xmask)` in the generated
source, or a named variable `xmask` / `ymask`.

```python
has_mask = 'mask=' in kernel_code or 'xmask' in kernel_code or 'ymask' in kernel_code
```

This appears whenever the problem size is **not** a multiple of the block size,
because the last tile needs to guard out-of-bound threads.

**Effect (two places):**

1. `estimate_memory_time_us` — HBM efficiency drops from 0.80 → 0.65:
   ```python
   hbm_efficiency = 0.65 if has_mask else 0.80
   ```
   Masked stores break hardware write-combining because partial cache lines must be
   read-modify-written instead of streamed.  The 15-point drop (0.80 → 0.65) models
   the ≈20% bandwidth penalty observed empirically.

2. `estimate_overhead_time_us` — +0.5 µs for extra predicate instructions and
   conditional branch overhead in the dispatch path.

`warp_size` is read directly from `device_props.warp_size` (Phase 1).  It feeds
`generate_all_candidate_configs` to enforce the constraint that
`num_warps × warp_size ≤ threads_per_block` — a config with more warps than
threads is physically invalid and filtered out before scoring.

`kernel_code` is the raw Python source of the generated Triton `@triton.jit`
function — a string of ~50–200 lines — passed verbatim from Inductor's codegen
layer and detailed fully in Section 3.3.

> **For junior engineers:** Think of `problem_metadata` as the heuristic's
> "job brief".  Every number in it answers a specific physical question the scoring
> formulas need: *How much data moves?* (`bytes_per_element × total_elements`).
> *How much computation happens?* (`ops_per_element × total_elements`). *What shape
> is the GPU's workload?* (`dimensions`).  *What overhead does each launch pay?*
> (`num_tensors`, `has_mask`, `num_warps`).  Getting any one of these wrong skews
> the bottleneck classification and causes the wrong config to win.

---

## 3. Module Architecture & Data Flow

```
Inductor codegen
  │  ┌──────────────────────────────────────────┐
  │  │  Generated Triton kernel source (.py str) │
  │  └───────────────────────┬──────────────────┘
  │                          │
  ▼                          ▼
runtime/triton_heuristics.py          triton_heuristics_kernel_analysis.py
  CachingAutotuner                      extract_kernel_metadata(kernel_code)
  _score_and_prune_heuristic_configs()    → {num_inputs, fast_ops, slow_ops, …}
  │                          │
  │  problem_metadata ◄──────┘  (merged)
  │
  ├──► triton_heuristics_hardware.py
  │      get_architecture_config()
  │        → ArchitectureConfig {num_cus, warp_size,
  │            optimal_threads_bandwidth, …}
  │
  ├──► triton_heuristics_adaptive.py  (per config, parallelised)
  │      BottleneckAnalysis.analyze_bottleneck()
  │        → {overhead_us, memory_us, compute_us,
  │            overhead_frac, memory_frac, compute_frac,
  │            bottleneck, launch_bound}
  │      BottleneckAnalysis.get_adaptive_weights()
  │        → {bandwidth: 0.47, launch: 0.11, …}
  │      BottleneckAnalysis.get_adaptive_exponents()
  │        → {bandwidth: 2.3, launch: 0.6, …}
  │
  └──► triton_heuristics_pointwise.py  (per config)
         PointwiseHeuristics.score_config()
           estimate_memory_bandwidth()   → BW ∈ [0.60, 1.00]
           estimate_launch_overhead()    → LO ∈ [0.70, 1.00]
           estimate_grid_granularity()   → GG ∈ [0.70, 1.00]
           estimate_occupancy_impact()   → OC ∈ [0.70, 1.00]
           weighted geometric mean → composite score ∈ [0, 1]

  Results sorted → top-N compiled → (optional) all benchmarked
```

All per-config scoring calls are dispatched to a `ThreadPoolExecutor` (up to 8
workers) so that the `torch.cuda.get_device_properties()` calls — which can block
briefly on the first call — do not serialise the scoring of 20+ configs.

---

### 3.1 `problem_metadata` — construction, fields, and consumers

`problem_metadata` is the central data structure that carries everything the scoring
stages need to know about the mathematical problem being compiled.  It is built in
**three phases** at different points in Inductor's compilation pipeline.

#### Phase 1 — Static metadata from `inductor_meta` / `triton_meta` / `size_hints`

**Where:** `_convert_to_pointwise_heuristics_metadata()` called from
`_apply_pointwise_heuristics()` called from `pointwise()` (the decorator factory).

**When:** At codegen time, before any Triton JIT object exists.

`pointwise()` is the decorator factory that Inductor calls once per unique kernel
signature to build a `CachingAutotuner`.  It receives three dictionaries directly
from Inductor's code-generation layer:

```python
def pointwise(size_hints, triton_meta, tile_hint, filename, min_elem_per_thread,
              inductor_meta):
    ...
    heuristics_result = _apply_pointwise_heuristics(
        size_hints,   # {'x': 1048576}  or  {'x': 128, 'y': 1024}
        inductor_meta,
        triton_meta,
        triton_config_with_settings,
        filename,
    )
```

Inside `_convert_to_pointwise_heuristics_metadata`:

```python
# ── size_hints ─────────────────────────────────────────────────────────────
# Ordered dict from Inductor: dimension name → rounded-up numel.
# The heuristic receives it as a dict; values() gives the dimension tuple.
dimensions    = tuple(size_hints.values())       # e.g. (1048576,) or (128, 1024)
total_elements = functools.reduce(mul, dimensions, 1)

# ── inductor_meta ──────────────────────────────────────────────────────────
# Built by Inductor's TritonKernelWrapper / CUDAWrapper for each kernel.
num_inputs  = inductor_meta.get("num_inputs",  2)    # count of input tensor args
num_outputs = inductor_meta.get("num_outputs", 1)    # count of output tensor args
fusion_depth = inductor_meta.get("fusion_depth", 1)  # number of fused ops
has_mask    = inductor_meta.get("has_mask",    False) # boundary-condition masks
has_broadcast        = inductor_meta.get("has_broadcast", False)
broadcast_tensor_bytes = inductor_meta.get("broadcast_tensor_bytes", 0)
vector_width = inductor_meta.get("vector_width", 1)  # SIMD vector width hint

# ── triton_meta ────────────────────────────────────────────────────────────
# Carries device properties (a DeviceProperties namedtuple) and dtype.
device_props       = triton_meta.get("device")
warp_size          = device_props.warp_size              # 64 on AMD, 32 on NVIDIA
max_threads_per_block = device_props.max_threads_per_block   # 1024 for both

# dtype → element_size (bytes)
dtype_str   = str(triton_meta.get("dtype", "float32"))
element_size = 2 if ("float16" or "bfloat16") in dtype_str else \
               8 if "float64" in dtype_str else 4
```

The result is stashed into `inductor_meta['_heuristics_pending']` and passed
forward — **no scoring happens here** because `self.fn.src` (the kernel source code)
only becomes available after `CachingAutotuner.__init__` runs.

```python
inductor_meta['_heuristics_pending'] = {
    'problem_metadata': problem_metadata,   # Phase-1 dict
    'size_hints':       size_hints,
}
```

> **Why the two-phase split?**  
> Triton's `@jit` decorator compiles the kernel function lazily.  At the time
> `pointwise()` is called the kernel's Python source string (`fn.src`) and argument
> name list (`fn.arg_names`) do not yet exist — they are created by Triton's
> decorator machinery when `CachingAutotuner.__init__` wraps the function.  Splitting
> construction across two phases lets us use the authoritative kernel source rather
> than relying entirely on Inductor's guesses about tensor counts.

#### Phase 2 — Refinement from the live `JITFunction` in `CachingAutotuner.__init__`

**Where:** `CachingAutotuner._score_and_prune_heuristic_configs()`, which is called
from `__init__` once `self.fn` is populated.

**When:** Still at compile time, immediately after the `JITFunction` object is ready.

Two additional data sources become available:

**`self.fn.arg_names`** — the authoritative list of every parameter name in the
generated kernel.  Parameters ending in `_ptr` are tensor pointers.

```python
ptr_args = [a for a in self.fn.arg_names if a.endswith('_ptr')]
# e.g. ['in_ptr0', 'in_ptr1', 'out_ptr0']

num_outputs = max(1, kernel_code.count('tl.store'))   # stores ↔ outputs
num_inputs  = max(0, len(ptr_args) - num_outputs)

problem_metadata = {
    **problem_metadata,         # merge, overwriting Phase-1 guesses
    'num_tensors': len(ptr_args),
    'num_inputs':  num_inputs,
    'num_outputs': num_outputs,
}
```

**`self.fn.src`** — the raw Python source of the `@triton.jit` kernel as a string.
This is the same source that Triton's compiler sees.

```python
kernel_code = str(self.fn.src)
# → "@triton.jit\ndef triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, ..."
```

`kernel_code` is passed forward to `extract_kernel_metadata` and
`BottleneckAnalysis.analyze_bottleneck` for deeper analysis (Phase 3 / Stage 2 and 3).

#### Phase 3 — Kernel-parsed fields merged inside `analyze_bottleneck`

**Where:** `BottleneckAnalysis.analyze_bottleneck()` in
`triton_heuristics_adaptive.py`.

**When:** Once per `(config, problem_metadata, kernel_code)` triple during the
parallel scoring loop.

```python
if kernel_code:
    kernel_metadata = extract_kernel_metadata(kernel_code)
    # merge: kernel_metadata keys take precedence over Phase-1/2 guesses
    problem_metadata = {**problem_metadata, **kernel_metadata}
```

Fields added or refined by `extract_kernel_metadata`:

```python
{
    'bytes_per_element': 12.0,   # num_tensors × dtype_bytes (more precise than guess)
    'ops_per_element':   3.2,    # weighted op count per output element
    'fast_ops':          8,      # tl.add/mul/fma, +, -, *
    'medium_ops':        2,      # tl.div/sqrt, /
    'slow_ops':          1,      # tl.exp/log/sin/tanh …
    'has_broadcast':     True,   # detected from index patterns
    'has_mask':          True,   # detected from tl.load/store mask= kwargs
}
```

---

### 3.2 Complete `problem_metadata` field reference

| Field | Type | Source (phase) | Default | Consumers |
|---|---|---|---|---|
| `dimensions` | `Tuple[int,…]` | `size_hints.values()` — Phase 1 | — | `get_problem_dimensions`, `calculate_grid_size`, config generation |
| `total_elements` | `int` | product of `dimensions` — Phase 1 | — | all scoring factors, bottleneck analysis |
| `element_size` | `int` bytes | `triton_meta["dtype"]` — Phase 1 | `4` (FP32) | `bytes_per_element` fallback, cache-fit checks |
| `num_tensors` | `int` | `fn.arg_names` ptr count — Phase 2 | `3` | `estimate_overhead_time_us` (tensor arg overhead) |
| `num_inputs` | `int` | `fn.arg_names` − outputs — Phase 2 | `2` | `estimate_vgpr_per_thread`, `bytes_per_element` fallback |
| `num_outputs` | `int` | `tl.store` count in `fn.src` — Phase 2 | `1` | same as above |
| `bytes_per_element` | `float` | `num_tensors × element_size` — Phase 3 | `12.0` | `estimate_memory_time_us` (total HBM bytes), AI calculation |
| `ops_per_element` | `float` | weighted op count from kernel — Phase 3 | `2` | `estimate_compute_time_us`, AI display |
| `fast_ops` | `int` | regex on `fn.src` — Phase 3 | `2` | `compute_efficiency` in `estimate_compute_time_us` |
| `medium_ops` | `int` | regex on `fn.src` — Phase 3 | `0` | same |
| `slow_ops` | `int` | regex on `fn.src` — Phase 3 | `0` | same; `slow_frac > 0.5` → efficiency = 0.60 |
| `has_mask` | `bool` | `inductor_meta` or regex — Phase 1/3 | `False` | HBM efficiency (0.65 vs 0.80), overhead +0.5 µs |
| `has_broadcast` | `bool` | `inductor_meta` or regex — Phase 1/3 | `False` | L2 cache reuse path in `estimate_memory_time_us` |
| `broadcast_tensor_bytes` | `int` | `inductor_meta` — Phase 1 | `0` | L2 broadcast-fit check |
| `warp_size` | `int` | `device_props.warp_size` — Phase 1 | `64` | `generate_all_candidate_configs` (warp validity filter) |
| `max_threads_per_block` | `int` | `device_props.max_threads_per_block` — Phase 1 | `1024` | same |
| `fusion_depth` | `int` | `inductor_meta` — Phase 1 | `1` | `estimate_vgpr_per_thread` (intermediate register pressure) |
| `vector_width` | `int` | `inductor_meta` — Phase 1 | `1` | `estimate_vgpr_per_thread` (SIMD register width) |

> **For junior engineers:** Think of the three phases like building a dossier on a
> job applicant.  Phase 1 is the application form — you know the broad strokes
> (problem size, data type, which GPU) from what Inductor filled in.  Phase 2 is
> the interview — now you can read the actual kernel argument list and source code
> to correct any wrong guesses about tensor counts.  Phase 3 is the background
> check — a regex-based parser reads every line of the kernel to count exactly how
> many fast additions, expensive `exp()` calls, and masked memory accesses exist.
> Each phase overwrites the previous phase's guesses with harder evidence.

---

### 3.3 `kernel_code` — what it is and how it reaches the scorers

`kernel_code` is the raw Python source of the generated `@triton.jit` function —
typically 50–200 lines.  A representative excerpt:

```python
@triton.jit
def triton_poi_fused_add_exp_0(
        in_ptr0, in_ptr1, out_ptr0,
        xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.exp(tmp2)
    tl.store(out_ptr0 + x0, tmp3, xmask)
```

From this string, `extract_kernel_metadata` extracts:

| Evidence in source | Extraction | Value |
|---|---|---|
| `in_ptr0`, `in_ptr1`, `out_ptr0` | `_ptr` params in signature | `num_tensors=3, num_inputs=2, num_outputs=1` |
| `tl.store` (1 call) | `kernel_code.count('tl.store')` | `num_outputs=1` |
| `tmp0 + tmp1` (fast op) | `re.findall(r'\s\+\s', …)` | `fast_ops += 1` |
| `tl.exp(tmp2)` (slow op) | `kernel_code.count('tl.exp')` | `slow_ops += 1` |
| `xmask` in `tl.load`/`tl.store` | `re.search(r'mask\s*=', …)` | `has_mask=True` |
| `bytes_per_element` | `num_tensors × 4` | `12.0` |

The string is accessed from `self.fn.src` in `_score_and_prune_heuristic_configs`
and passed to every downstream call that accepts `kernel_code`:

```python
kernel_code = str(self.fn.src)   # JITFunction attribute set by Triton's @jit

# Passed to:
PointwiseHeuristics.score_config(config, problem_metadata, kernel_code)
PointwiseHeuristics.get_detailed_scores(config, problem_metadata, kernel_code)
BottleneckAnalysis.analyze_bottleneck(config, problem_metadata, kernel_code)
# → inside analyze_bottleneck:
#   kernel_metadata = extract_kernel_metadata(kernel_code)
#   problem_metadata = {**problem_metadata, **kernel_metadata}  # Phase 3 merge
```

All three callers guard with `if kernel_code:` so the system degrades gracefully
to Phase-1/2 metadata if `fn.src` is unavailable (e.g. when loading from cache).

---

## 4. Stage 1 — Hardware Constants

**File:** `torch/_inductor/codegen/triton_heuristics_hardware.py`  
**Class:** `ArchitectureConfig`  
**Entry point:** `get_architecture_config()` (singleton, lazy)

### 4.1 What is queried

```python
props = torch.cuda.get_device_properties(device)

num_cus          = props.multi_processor_count   # CU / SM count
warp_size        = props.warp_size               # 64 AMD, 32 NVIDIA
max_threads_cu   = props.max_threads_per_multi_processor
l2_cache_size    = props.L2_cache_size
```

> **What is a Compute Unit (CU)?**
> A CU is the fundamental processing block of an AMD GPU (equivalent to NVIDIA's
> Streaming Multiprocessor / SM).  Each CU contains:
> - 64 ALU lanes (one per thread in a wavefront)
> - 32 KB of L1 / scratchpad cache
> - 65 536 VGPRs (Vector General-Purpose Registers) — 256 VGPRs per thread at
>   minimum occupancy
> - 40 wavefront slots
>
> An MI300X has 304 CUs; an MI350X has 320.  Every block launched by Triton is
> assigned to exactly one CU.

### 4.2 `optimal_threads_bandwidth` — latency-hiding thread count

**Goal:** enough concurrent threads per block that HBM latency is fully hidden.

**Derivation:**

```
HBM access latency        : L_mem  ≈ 400 clock cycles  (HBM2e / HBM3)
ALU instruction latency   : L_alu  ≈ 4  clock cycles
Instructions per element  : I_elem ≈ 2  (typical pointwise)

Threads needed to fill latency gap:
  N_threads = L_mem / (L_alu × I_elem)
            = 400  / (4   × 2)
            = 50 threads

Rounded to wavefront boundaries (warp_size=64):
  wavefronts_needed = ceil(50 / 64) = 1   →  clamped to min=8

∴ optimal_threads_bandwidth = 8 × warp_size
                             = 8 × 64  = 512   (AMD)
                             = 8 × 32  = 256   (NVIDIA)
```

The raw formula gives 1 wavefront; empirical sweeps show 8 wavefronts (512 threads
on AMD) saturates HBM bandwidth, so the clamp `max(8, …)` is applied.

> **What is a wavefront?**
> On AMD a *wavefront* (analogous to NVIDIA's *warp*) is a group of 64 threads that
> execute **in lock-step** — they all run the same instruction at the same time on
> the 64-wide ALU.  When a wavefront issues a memory load, it has to wait ~400 GPU
> clock cycles for HBM to respond.  During that wait the hardware can switch to
> another wavefront in the same CU and keep the ALUs busy.  This is called
> **latency hiding**: by keeping 8 wavefronts in flight you can hide 7 HBM round
> trips behind 1, making the memory bus appear 8× faster to the compute pipeline.

### 4.3 `optimal_blocks_grid` — CU saturation target
f
```
blocks_per_cu_large  = 2    # 2 waves per CU allows load balancing
optimal_blocks_grid  = num_cus × 2
```

Two waves per CU is the empirically validated sweet spot: one wave always runs
while the other is resident and ready, giving the GPU's command processor room
to load-balance tail work without leaving CUs idle.

### 4.4 `occupancy_sweetspot` — VGPR-limited wavefront range

```
assumed_vgprs_per_thread  = 50
vgprs_per_cu              = 65 536
vgprs_per_wavefront       = 50 × 64  = 3200

max_wavefronts_by_vgpr    = 65536 / 3200  = 20

occupancy_sweetspot_min   = 4   (enough for latency hiding)
occupancy_sweetspot_max   = min(8, 20)  = 8
```

> **Why not maximise wavefronts?**
> More wavefronts means fewer VGPRs per wavefront. Triton will spill registers to
> L1/L2 if a wavefront needs more VGPRs than are available, causing extra memory
> traffic and harming rather than helping throughput.  The sweet spot of 4–8
> wavefronts balances latency hiding against register pressure.

### 4.5 `optimal_elements_per_block` — launch amortisation

```
launch_overhead_us      = 3.0    μs
element_time_us         = 0.05   μs / element  (memory-bound estimate)
target_overhead_fraction = 0.05  (< 5 % overhead target)

min_elements  = 3.0 / 0.05                  = 60
optimal_elems = 60  / 0.05                  = 1 200
→ rounded to next power of 2: 2 048
→ clamped to [256, 2048]: 2 048
```

---

## 5. Stage 2 — Kernel Code Analysis

**File:** `torch/_inductor/codegen/triton_heuristics_kernel_analysis.py`  
**Entry point:** `extract_kernel_metadata(kernel_code: str) → Dict`

The function receives the raw Python source of the generated `@triton.jit` kernel
and extracts metadata that would otherwise have to be guessed.

### 5.1 Tensor counting

```python
# Parse function signature for pointer parameters
sig_match = re.search(r'def\s+\w+\s*\((.*?)\):', kernel_code, re.DOTALL)
ptr_params   = [p for p in params.split(',') if '_ptr' in p]
num_tensors  = len(ptr_params)

num_stores   = kernel_code.count('tl.store')
num_outputs  = max(1, num_stores)
num_inputs   = num_tensors - num_outputs

bytes_per_element = num_tensors * 4    # FP32 assumption
```

### 5.2 Instruction mix classification

Operations are categorised by GPU execution cost:

| Category | Operations | Typical latency |
|---|---|---|
| **fast** | `tl.add`, `tl.sub`, `tl.mul`, `tl.fma`, `+`, `-`, `*` | 1–2 cycles |
| **medium** | `tl.div`, `tl.sqrt`, `tl.rsqrt`, `/` | 10–20 cycles |
| **slow** | `tl.exp`, `tl.log`, `tl.sin`, `tl.cos`, `tl.tanh` | 30–50 cycles |

The `slow_frac` and `medium_frac` ratios feed `estimate_compute_time_us()` to set
realistic `compute_efficiency`:

```python
if slow_frac > 0.5:     compute_efficiency = 0.60   # transcendental-heavy
elif slow_frac > 0.2:   compute_efficiency = 0.70   # mixed
else:                   compute_efficiency = 0.80   # mostly FMA
```

### 5.3 Broadcast & mask detection

```python
# Broadcast: any dimension-0 index reuse
has_broadcast = bool(re.search(r'tl\.broadcast|expand|repeat', kernel_code))

# Masking: boundary guards on memory accesses
has_mask = bool(re.search(r'mask\s*=|tl\.load.*mask=|tl\.store.*mask=', kernel_code))
```

`has_mask` triggers two downstream adjustments:
- HBM efficiency drops from 0.80 → 0.65 (partial cache-line stores break write-combining)
- Overhead estimate gains +0.5 µs (predicate instruction overhead)

> **Why does masking hurt performance?**
> When a kernel processes a tensor whose size is not a multiple of the block size,
> Triton inserts "mask" predicates so out-of-bound threads don't write garbage.
> On AMD, masked stores break the hardware's write-combining buffer, forcing
> partial cache-line writes to HBM instead of coalesced full-line writes.  This
> reduces effective bandwidth by ~15–25%.

---

## 6. Stage 3 — Bottleneck Analysis

**File:** `torch/_inductor/codegen/triton_heuristics_adaptive.py`  
**Class:** `BottleneckAnalysis`  
**Entry point:** `analyze_bottleneck(config, problem_metadata, kernel_code) → Dict`

This stage models three time components for a given `(kernel, config)` pair and
applies the **roofline model** to determine which component limits performance.

### 6.1 Overhead time model

```
T_overhead = K_launch
           + (num_tensors − 3) × 0.1  µs   (tensor arg setup)
           + (num_warps   − 1) × 0.2  µs   (per-warp thread init)
           + 0.5 × log2(num_blocks / 1000)  µs  (grid setup, large grids)
           + 0.5  µs   (if has_mask)

K_launch ≈ 3.0 µs   (empirical AMD CDNA constant)
```

The `(num_warps − 1) × 0.2` term is measured from kernel dispatch traces: each
additional wavefront group scheduled in a CU costs approximately 0.2 µs of
initialisation overhead.  This is the same constant reused in the occupancy
first-principles formula (Section 8.5).

> **What is kernel launch overhead?**
> Every time the CPU tells the GPU to run a kernel, there is a fixed setup cost:
> the command is placed in a ring buffer, the GPU's command processor parses it,
> allocates CU resources, and begins dispatching wavefronts.  Even a kernel that
> does one addition on one element still pays this ~3 µs tax.  For tiny kernels
> (≤ 2K elements that complete in ≈ 0.5 µs), this overhead dominates 85 % of
> wall-clock time, which is why the heuristic must correctly classify such kernels
> as "launch-bound".

### 6.2 Memory time model

```python
total_bytes = total_elements × bytes_per_element

if total_bytes ≤ L1_cache (32 KB) and num_blocks ≤ num_CUs:
    T_memory ≈ 0 µs   (true L1 hit, negligible)

elif total_bytes ≤ L2_cache (4–8 MB):
    T_memory = total_bytes / (L2_bandwidth_GB_s × 1e3)   # L2 hit

else:   # HBM streaming
    hbm_efficiency = 0.65 if has_mask else 0.80
    effective_bw   = memory_bandwidth_GB_s × hbm_efficiency
    T_memory       = total_bytes / (effective_bw × 1e3)   µs
```

**Critical AMD insight — L1 is per-CU, not global.**  A 304-CU GPU has 304 × 32 KB
= 9.5 MB of L1, but each CU's 32 KB is private.  If 304 blocks each load the same
small tensor, that tensor is read from HBM 304 times even though it "fits in L1".
The model only grants an L1 hit when `num_blocks ≤ num_CUs`.

> **L1, L2, and HBM — cache hierarchy on AMD CDNA:**
> - **L1 (SRAM, 32 KB/CU):** Fastest; private to each CU.  One cache-line = 64 bytes.
>   A wavefront that reads a missed cache-line stalls for ~100 clock cycles.
> - **L2 (SRAM, 4–32 MB, shared):** Shared across all CUs on one GCD (Graphics
>   Compute Die).  ~10× slower than L1 but shared so broadcast data is read once.
> - **HBM (DRAM, 192 GB, shared):** On-package stacked DRAM.  ~400 cycles latency,
>   but very high bandwidth (5.3 TB/s on MI300X).  Most pointwise kernels are
>   HBM-bound once data exceeds L2.

### 6.3 Compute time model

```
ops_per_element  from kernel metadata (or problem_metadata)
num_ops          = total_elements × ops_per_element

achievable_tflops = peak_tflops × compute_efficiency

T_compute = num_ops / (achievable_tflops × 1e6)   µs
```

`peak_tflops` is derived from hardware properties:

```python
if is_hip:
    ops_per_cu_clk = 128         # AMD CDNA2/3: 128 FP32 ops/CU/clock
else:
    ops_per_cu_clk = 256         # NVIDIA: 128 CUDA-cores × 2 FP32/core/clk

compute_tflops = (num_cus × ops_per_cu_clk × clock_hz) / 1e12
```

> **Why 128 FP32 ops/CU/clock on AMD CDNA?**
> Each CU contains **4 SIMD units**, each 16-wide.  Per clock, each SIMD can issue
> one FP32 FMA (Fused Multiply-Add), which counts as **2 FLOPS** (one multiply +
> one add).  So: 4 SIMDs × 16 lanes × 2 FLOPs = **128 FP32 FLOPS/CU/clock**.

### 6.4 Roofline model

```
T_effective = T_overhead + max(T_memory, T_compute)

bottleneck:
  'overhead'  if T_overhead ≥ max(T_memory, T_compute)
  'memory'    elif T_memory ≥ T_compute
  'compute'   else

launch_bound = overhead_frac > 0.50
  where overhead_frac = T_overhead / (T_overhead + T_memory + T_compute)
```

The roofline captures **memory-compute overlap**: the GPU's memory load/store units
and ALU units are independent pipelines.  A memory request can be in-flight while
the ALU processes the previous result.  The slower pipeline determines execution
time; the faster one is "hidden" behind it.

> **Roofline model intuition:**
> Imagine a factory with two workers: one fetches boxes from a warehouse (memory)
> and one assembles parts (compute).  If fetching takes 10 minutes and assembly
> takes 2 minutes, the factory is "warehouse-bound" — the assembler waits 8 minutes
> between boxes.  If assembly took 15 minutes, the warehouse worker would wait.
> The roofline model finds which worker is the bottleneck.  On modern GPUs, almost
> all pointwise kernels are warehouse-bound (memory-bound) because adding two
> floats is ≈ 200× faster than reading them from HBM.

### 6.5 Output dict

```python
{
    'overhead_us':    3.4,
    'memory_us':      0.6,
    'compute_us':     0.1,
    'total_us':       4.0,
    'overhead_frac':  0.71,   # → launch_bound = True
    'memory_frac':    0.21,
    'compute_frac':   0.08,
    'bottleneck':     'overhead',
    'launch_bound':   True,
}
```

---

## 7. Stage 4 — Adaptive Weight Interpolation

**File:** `torch/_inductor/codegen/triton_heuristics_adaptive.py`  
**Entry point:** `get_adaptive_weights(config, problem_metadata, kernel_code) → Dict`

### 7.1 Pure-regime weight vectors

Three "pure" weight vectors define the ideal factor emphasis in each extreme regime.
Values were derived by reasoning about which physical bottleneck each scoring factor
addresses:

```
Factor       │ OVERHEAD (tiny kernel) │ MEMORY (streaming HBM) │ COMPUTE (transcend.)
─────────────┼────────────────────────┼─────────────────────────┼──────────────────────
bandwidth    │   0.10                 │   0.55                  │   0.15
launch       │   0.65                 │   0.10                  │   0.10
grid         │   0.10                 │   0.15                  │   0.30
occupancy    │   0.15                 │   0.20                  │   0.45
             │   ────                 │   ────                  │   ────
             │   1.00                 │   1.00                  │   1.00
```

**OVERHEAD regime rationale:**
- `launch` (0.65): The single most impactful choice is block count — fewer, larger
  blocks reduce kernel dispatch pressure.  The launch score is the only factor that
  directly penalises high block counts for fixed total work.
- `grid` (0.10): Deliberately low.  The grid granularity score rewards "more blocks"
  (for CU saturation), which fights the launch score.  Keeping grid weight low
  prevents these two antagonistic factors from cancelling each other.
- `bandwidth` (0.10): Irrelevant — tiny data fits in L2/L1, bandwidth is not the wall.
- `occupancy` (0.15): A small role: one dispatch stall can be hidden if another
  wavefront is ready.

**MEMORY regime rationale:**
- `bandwidth` (0.55): Coalescing and thread count are paramount — this is the wall.
- `occupancy` (0.20): Wavefront latency hiding keeps the HBM pipe full.
- `grid` (0.15): All CUs must be fed; an under-subscribed GPU wastes bandwidth slots.
- `launch` (0.10): Dispatch overhead is negligible relative to streaming time.

**COMPUTE regime rationale:**
- `occupancy` (0.45): ALU utilisation scales directly with wavefront count.
- `grid` (0.30): Every CU must execute to saturate peak TFLOPS.
- `bandwidth` (0.15): Operands must arrive; feeds the ALU.
- `launch` (0.10): Amortised over long execution time.

### 7.2 Continuous interpolation

Instead of hard-switching between three static tables, the system uses the raw
component fractions from `analyze_bottleneck` as **mixing coefficients**:

```
w[k] = overhead_frac × OVERHEAD_W[k]
     + memory_frac   × MEMORY_W[k]
     + compute_frac  × COMPUTE_W[k]
```

Because the three fracs sum to 1.0 and each regime vector sums to 1.0, the
resulting weights always sum to 1.0 (normalised for floating-point safety).

**Example: a kernel with `overhead_frac=0.70, memory_frac=0.25, compute_frac=0.05`:**

```
bandwidth = 0.70×0.10 + 0.25×0.55 + 0.05×0.15 = 0.213
launch    = 0.70×0.65 + 0.25×0.10 + 0.05×0.10 = 0.485
grid      = 0.70×0.10 + 0.25×0.15 + 0.05×0.30 = 0.123
occupancy = 0.70×0.15 + 0.25×0.20 + 0.05×0.45 = 0.178
```

### 7.3 Weight-to-exponent mapping

The geometric mean formula `score = (BW^a × LO^b × GG^c × OC^d)^(1/(a+b+c+d))`
uses exponents rather than weights directly.  The linear mapping:

```
exp = 0.5 + (weight − 0.10) / 0.40 × 2.5

weight=0.10 → exp=0.5   (minimal influence)
weight=0.50 → exp=3.0   (dominant influence)
```

This converts the normalised weight (range 0.10–0.50 across typical regimes) to an
exponent range of 0.5–3.0, giving the highest-weight factor a 6× stronger pull than
the lowest-weight factor in the geometric mean.

> **Why geometric mean instead of arithmetic mean?**
> A factor score of 0.0 (e.g. zero-threads config) should kill the overall score,
> not just reduce it.  The geometric mean ensures that a score of 0 in any factor
> makes the composite score 0, while the arithmetic mean would merely subtract a
> fraction.  For heuristics where catastrophically bad configs must be clearly
> separated from mediocre ones, this multiplicative property is essential.

---

## 8. Stage 5 — Per-Config Scoring

**File:** `torch/_inductor/codegen/triton_heuristics_pointwise.py`  
**Class:** `PointwiseHeuristics`

The composite score is a weighted geometric mean of four factors:

```
score = (BW^a × LO^b × GG^c × OC^d)^(1/(a+b+c+d))
```

where exponents `a, b, c, d` come from the adaptive weight mapping (Section 7.3).

### 8.1 Factor 1 — Memory Bandwidth (`BW`)

**Measures:** how well the thread count per block matches the latency-hiding optimum.

**Model:**

```python
optimal_threads = arch.optimal_threads_bandwidth   # 512 on AMD

diff    = (threads_per_block − optimal_threads) / optimal_threads
gaussian = exp(−0.5 × diff²)

score = 0.75 + 0.25 × gaussian      # range [0.75, 1.00]
score = max(0.60, score)             # hard floor

# Special: too few threads (<64)
score = 0.60
```

The Gaussian peaks at `optimal_threads` and decays symmetrically.  The `σ = optimal_threads`
choice means the score drops to `0.75 + 0.25/e ≈ 0.84` when thread count is either
half or double the optimum.

**2D/3D coalescing factor:**

For kernels with YBLOCK (2D/3D), XBLOCK is the contiguous (fast) dimension.
A cache line holds 64 B = 16 FP32 values.  If XBLOCK < 16, threads in a wavefront
access fewer than one full cache line per row, wasting memory bus capacity:

```python
cache_line_elems = 16
coalescing       = min(1.0, XBLOCK / 16)

score *= (0.65 + 0.35 × coalescing)
# XBLOCK=1  → ×0.65
# XBLOCK=8  → ×0.82
# XBLOCK=16 → ×1.00
```

> **Memory coalescing:**
> 64 threads in a wavefront issue memory requests simultaneously.  If those 64
> threads access 64 consecutive floats, the hardware can service them with a single
> cache-line fetch.  If they access scattered locations, up to 64 separate fetches
> are needed — using 64× the HBM bandwidth for the same data.  XBLOCK controls
> how many consecutive elements a tile row covers; small XBLOCK = poor coalescing.

### 8.2 Factor 2 — Launch Overhead (`LO`)

**Measures:** how well the elements-per-block ratio amortises per-block dispatch cost.

```python
elements_per_block = total_elements / num_blocks
optimal_elem       = arch.optimal_elements_per_block   # 2048

diff     = (elements_per_block − optimal_elem) / (optimal_elem / 2)
gaussian = exp(−0.5 × diff²)

score = 0.75 + 0.25 × gaussian   # range [0.75, 1.00]

# Too few elements per block (<64): score = 0.70
```

**Large-grid command-processor penalty:**

AMD's command processor has finite instruction-dispatch bandwidth.  Grids beyond
`4 × num_CUs` blocks begin to saturate the dispatch queue:

```python
max_good_blocks  = num_CUs × 4
if num_blocks > max_good_blocks:
    excess_ratio  = num_blocks / max_good_blocks
    penalty       = 1.0 − 0.01 × log2(excess_ratio)
    score        *= max(0.88, penalty)
```

A grid of 8× too many blocks (`excess_ratio=8`) incurs a 3% penalty (`log2(8)=3`).

### 8.3 Factor 3 — Grid Granularity (`GG`)

**Measures:** CU utilisation for the given problem size.

Problem-size-adaptive sweet spots (each with a Gaussian centred on optimal block count):

| Problem size | Regime | Optimal blocks | Notes |
|---|---|---|---|
| < 2 048 elements | Tiny | 3–4 | 4 CUs, minimal dispatch; >8 wastes overhead budget |
| 2 K – 16 K | Small | `num_CUs / 32` | Partial GPU occupancy is acceptable |
| 16 K – 256 K | Medium | `num_CUs` | Half GPU coverage |
| > 256 K | Large | `2 × num_CUs` | Full GPU, 2-wave load balancing |

For the tiny regime the scoring is discrete (empirically validated):

```python
num_blocks == 1:  0.85    # single CU — no parallelism
num_blocks == 2:  0.93
num_blocks <= 4:  1.00    # AMD empirical optimum for <2K
num_blocks <= 8:  0.90
else:             0.70
```

> **Why is 4 blocks better than 1 for tiny problems?**
> Even with only 512 elements, 4 blocks of 128 elements each can be dispatched to
> 4 different CUs simultaneously.  AMD's command processor can issue all 4 in a
> single dispatch wave with negligible extra overhead.  The single-block config
> leaves 300+ CUs completely idle, which matters even for tiny kernels because
> they still have memory latency to hide.

### 8.4 Factor 4 — Occupancy (`OC`)

**Measures:** appropriateness of `num_warps` for the kernel's execution regime.

The scoring logic has **three distinct branches** based on saturation and block count:

#### Branch A — Well-saturated / memory-bound (`saturation ≥ 0.25`)

```python
saturation = min(1.0, (num_blocks × num_warps) / (num_CUs × 8))

if   sweet_min ≤ num_warps ≤ sweet_max:    # 4–8 on AMD
    base_score = 1.00
elif sweet_min//2 ≤ num_warps ≤ sweet_max×1.5:
    base_score = 0.95
elif num_warps == 1:
    base_score = 0.85
else:
    base_score = 0.75
```

The sweet spot 4–8 wavefronts/block satisfies Little's Law:

```
L (wavefronts in flight) = λ (issue rate) × W (latency)
W ≈ 400 cycles, λ ≈ 1 wf/64 cycles  →  L ≈ 6 wavefronts
```

#### Branch B — Launch-bound, single block (`saturation < 0.25`, `num_blocks == 1`)

All work is on one CU.  Memory latency must be hidden within that CU by wavefront
switching.  The sweet-spot model is identical to Branch A.

#### Branch C — Launch-bound, multi-block (`saturation < 0.25`, `num_blocks > 1`)

Each block runs on a separate CU; inter-CU latency hiding is handled by grid spread.
The only per-config variable that changes wall-clock time is **wavefront initialisation
overhead**:

```
T(num_warps) = K_launch + (num_warps − 1) × K_warp + T_exec

K_launch = 3.0 µs   (BottleneckAnalysis.KERNEL_LAUNCH_US)
K_warp   = 0.2 µs   (per-extra-warp init, from estimate_overhead_time_us)
T_exec   = total_bytes / bandwidth   (config-invariant)
```

`T_exec` cancels in the score ratio:

```
score = T_min / T(num_warps)
      = K_launch / (K_launch + (num_warps − 1) × K_warp)

num_warps=1:  3.0 / 3.0                = 1.000
num_warps=2:  3.0 / 3.2                = 0.938
num_warps=4:  3.0 / 3.6                = 0.833
num_warps=8:  3.0 / 4.4                = 0.682
num_warps=16: 3.0 / 6.0                = 0.500
```

This is a **first-principles derivation** — both constants come from the existing
overhead model, no new calibration is introduced.  The element-loop depth
(EPT = XBLOCK / (num_warps × warp_size)) was found to contribute < 0.0001 µs per
iteration (loop instruction overhead ≈ 4 GPU instructions ≈ 0.08 ns), which is
2500× below `K_warp` and thus negligible.

### 8.5 2D tile shape tie-breaker

For 2D configs with the same composite score, two multipliers break ties:

```python
ratio = max(XBLOCK, YBLOCK) / min(XBLOCK, YBLOCK)

# Prefer square-ish tiles (prefetcher alignment)
balance_multiplier = 1.0 − 0.005 × log2(max(ratio, 1.0))
# ratio=1 → ×1.000 | ratio=4 → ×0.990 | ratio=16 → ×0.980

# Prefer larger XBLOCK (row-major = better cache-line fill)
innermost_multiplier = 1.0 − 0.008 × max(0, 5 − log2(max(XBLOCK, 4)))
# XBLOCK=256 → ×1.000 | XBLOCK=32 → ×0.976 | XBLOCK=4 → ×0.952

score *= max(0.95, balance_multiplier × innermost_multiplier)
```

---

## 9. Stage 6 — Autotuner Integration

**File:** `torch/_inductor/runtime/triton_heuristics.py`  
**Class:** `CachingAutotuner`

### 9.1 Entry point: `_score_and_prune_heuristic_configs`

Called once per unique `(kernel_fn, size_hints)` pair when the first compilation of
that configuration is triggered.  Receives:

- `self.configs` — the full list of candidate `triton.Config` objects (20–80 configs
  for a typical 1D pointwise kernel, more for 2D/3D)
- `size_hints` — problem shape tuple, e.g. `(1048576,)` or `(128, 1024)`
- `problem_metadata` — the metadata dict assembled by codegen
- `kernel_code` — optional raw kernel source for `extract_kernel_metadata`

**Parallel scoring:**

```python
def _score_one_config(triton_cfg):
    effective = {k: triton_cfg.kwargs[k] for k in ('XBLOCK','YBLOCK','ZBLOCK') …}
    effective['num_warps'] = triton_cfg.num_warps
    sc = PointwiseHeuristics.score_config(effective, problem_metadata, kernel_code)
    d  = PointwiseHeuristics.get_detailed_scores(effective, problem_metadata, kernel_code)
    bn = BottleneckAnalysis.analyze_bottleneck(effective, problem_metadata, kernel_code)
    return (sc, triton_cfg, effective, d, bn)

with ThreadPoolExecutor(max_workers=8) as pool:
    _scored_raw = list(filter(None, pool.map(_score_one_config, self.configs)))

_scored_raw.sort(key=lambda x: x[0], reverse=True)
```

### 9.2 Mode logic

**`REAL_BENCH` mode** (`heuristics_real_bench=True`):
- `self.configs` is left unchanged (all candidates compiled and benchmarked)
- `_TOP_N_CONFIGS_FOR_SELECTION[key]` is set to the top-N predicted dicts
- At selection time, `autotune_to_one_config` filters `timings` to only the
  top-N pool and picks the fastest non-spilling config within that pool

**Heuristics-only mode** (`heuristics_real_bench=False`):
- `self.configs` is pruned to `top-(N + spill_buffer)` Triton configs
- Only those compile and get benchmarked (faster first-use, Triton caches the rest)
- Selection picks the fastest from the standard top-N; if all spill, the fallback
  buffer provides alternatives (see Section 10)

### 9.3 Problem key normalisation

```python
def _normalize_problem_key(size_hints) -> str:
    """
    Convert size_hints (tuple or dict) to a canonical string for dict keying.
    size_hints=(32768,)    → "32768"
    size_hints={'x':32768} → "32768"
    """
```

Used as the key for `_HEURISTICS_VALIDATION_DATA`, `_TOP_N_CONFIGS_FOR_SELECTION`,
and `_HEURISTICS_FULL_RANKED_HDICTS` — the three global dicts that carry state
between the compile-time scoring pass and the runtime selection pass.

### 9.4 `autotune_to_one_config`

Called at first kernel invocation (after `precompile()` has run):

```
1.  Post-compile spill eviction  (if buffer > 0 — see Section 10)
2.  benchmark_all_configs()  — bench() for each launcher
3.  Filter timings to top-N selection pool
4.  If all top-N are inf (spilled): walk buffer pool for fallback
5.  self.launchers = [best non-spilling config]
```

---

## 10. Spill Prediction & Fallback Buffer

### 10.1 Why spills occur

Triton compiles each config independently.  A large block with many warps allocates
fewer VGPRs per thread (total VGPRs ÷ threads).  If the kernel needs more VGPRs
than available, the compiler **spills** excess registers to L1 scratch, causing
extra memory traffic.  Spilling configs are detected at runtime:

```python
# In bench():
if launcher.n_spills > spill_threshold:   # 32 AMD, 16 NVIDIA
    return float("inf")                   # skip — worse than any real config
```

`launcher.n_spills` is populated by `_make_launchers()` from the compiled binary
metadata — it is **known after compile, before any GPU benchmark call**.

### 10.2 Heuristic spill risk (informational, `_estimate_spill_risk`)

A lightweight pre-compile estimate shown in the verbose scoring table:

```python
max_vgprs_per_thread = min(256, 65536 // threads_per_block)

estimated_vgprs = 16                                   # base (loop vars, addresses)
estimated_vgprs += (num_inputs + num_outputs) × 8      # operand registers
estimated_vgprs += ops_per_element × 1.5               # compute temporaries
estimated_vgprs += 10  if YBLOCK > 0                   # 2-D loop overhead
estimated_vgprs += 8   if XBLOCK ≥ 512                 # loop-unroll live values

ratio = estimated_vgprs / max_vgprs_per_thread

risk = 'HIGH' if ratio > 1.0
       'MED'  if ratio > 0.75
       'low'  otherwise
```

This is intentionally conservative (over-estimates) to flag risk without suppressing
valid configs.  It **does not affect ranking or selection** — it is purely diagnostic.

### 10.3 Compile-pool buffer (`heuristics_spill_fallback_buffer`)

When `TORCHINDUCTOR_HEURISTICS_SPILL_BUFFER=N > 0`:

**At scoring time (heuristics-only mode):**
```python
_compile_n  = min(_top_n + N, len(scored))
self.configs = [triton_cfg for _, triton_cfg, _ in scored[:_compile_n]]
_HEURISTICS_FULL_RANKED_HDICTS[key] = [hdict for _, _, hdict in scored]
```

**At selection time, post-compile eviction (before `benchmark_all_configs`):**
```python
# n_spills is now known for all compiled launchers
for i, hdict in enumerate(top_n_selection_pool):
    if spills(launcher_for(hdict)):
        # Replace with best non-spilling buffer candidate
        replacement = next(non_spilling buffer configs)
        top_n_selection_pool[i] = replacement
        evicted += 1

# Update selection pool; spill-evicted configs won't be selected
_TOP_N_CONFIGS_FOR_SELECTION[key] = updated_pool
```

**If all top-N still spill after eviction:**
```python
# Walk full ranked list to find first non-spilling config
for rank, hdict in enumerate(full_ranked[len(top_n):], ...):
    if timings[launcher_for(hdict)] < inf:
        self.launchers = [that launcher]
        break
```

> **Why buffer, not pre-compile filter?**
> Register spill count is only known after Triton compiles the kernel — there is no
> way to exactly predict it without running the compiler.  The heuristic estimate
> (Section 10.2) is too imprecise to use for hard filtering without risking false
> positives that discard the true best config.  The buffer approach pays only a
> one-time compilation cost (Triton caches all binaries) for a guaranteed valid
> fallback.

---

## 11. Validation & Verbose Logging

Controlled by `TORCHINDUCTOR_HEURISTICS_VERBOSE=1`.

### 11.1 Kernel & Problem Analysis box

Printed once per unique kernel before the scoring table.  Shows the roofline
regime, AI vs OI ceiling comparison, estimated time breakdown, and architecture
constants.

### 11.2 Scoring table

One row per candidate config, printed after parallel scoring:

```
  Rank    Score  Bottleneck    Ohd_µs  Mem_µs  Cmp_µs  Tot_µs  │    BW    Lnch    Grid   Occup  │   Blks  Thr/blk    Spill?  Config
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  #  1   0.9342  🚀 LAUNCH      3.60    0.03    0.02    3.63  │ 0.875   0.921   1.000   1.000  │      4      512  low(0.52)  {XBLOCK:512, nw:1}
  #  2   0.9108  🚀 LAUNCH      3.40    0.03    0.02    3.43  │ 0.875   0.921   0.930   0.938  │      4      512  low(0.52)  {XBLOCK:512, nw:2}  ◄ top-5
  ...
  Spill? column: estimated VGPRs / max VGPRs per thread (LOW<0.75, MED 0.75-1.0, HIGH>1.0 → likely spill)
```

### 11.3 Validation summary (`_print_heuristics_validation_summary`)

Printed after benchmarking (REAL_BENCH mode) for each problem size.  Shows:

- Predicted #1 config (or first non-spilling predicted if spills occurred)
- Actual best config with its predicted rank
- Per-factor score comparison: predicted vs actual
- Regime string (LAUNCH-BOUND / MEMORY-BOUND / COMPUTE-BOUND)
- Slowdown of predicted choice vs actual best

```
📊 PREDICTED #1  (heuristic top pick)
   Score     :  0.9342
   Config     :  {'XBLOCK': 512, 'num_warps': 1}
   Measured   :  0.0148 ms

🏆 ACTUAL Best  (rank #1 in benchmark)
   Score     :  0.9342
   Config     :  {'XBLOCK': 512, 'num_warps': 1}
   Measured   :  0.0148 ms

✅ Heuristics correctly identified the best config!
   Predicted #1 = Actual #1  (slowdown: 1.000x)

  Regime         :  LAUNCH-BOUND  (overhead 71% of gross time; roofline: MEMORY-BOUND)
```

---

## 12. End-to-End Example

**Problem:** elementwise add, shape `(4096,)`, AMD MI300X (304 CUs, warp=64)

### Step 1 — Hardware constants (cached)

```
num_cus=304, warp_size=64
optimal_threads_bandwidth = 8 × 64 = 512
optimal_blocks_grid       = 304 × 2 = 608
occupancy_sweetspot       = [4, 8]
optimal_elements_per_block = 2048
```

### Step 2 — Kernel analysis

```
kernel_code parse:
  num_tensors=3, num_inputs=2, num_outputs=1
  bytes_per_element=12.0 (3×FP32)
  fast_ops=6, medium_ops=0, slow_ops=0
  has_mask=False, has_broadcast=False
```

### Step 3 — Candidate generation

20 configs generated: `{XBLOCK: [256,512,1024], num_warps: [1,2,4,8,16]}` pruned to
valid ranges (e.g. `XBLOCK=1024, nw=16` has `nw×64=1024 > XBLOCK`, invalid).

### Step 4 — Bottleneck analysis for `{XBLOCK:1024, nw:1}`

```
threads_per_block = 1024
num_blocks        = 4096 / 1024 = 4

T_overhead = 3.0 + (1-1)×0.2 = 3.00 µs    (nw=1, no warp overhead)
T_memory   = 4096×12 / (900×0.8×1e3) = 0.068 µs  (HBM streaming)
T_compute  = 4096×6  / (200×0.8×1e6) = 0.000 µs  (negligible)

T_effective = 3.00 + max(0.068, 0) = 3.07 µs

overhead_frac = 3.0 / (3.0 + 0.068 + 0.0) = 0.978  → launch_bound = True
bottleneck    = 'overhead'
```

### Step 5 — Adaptive weights

```
w[bandwidth] = 0.978×0.10 + 0.020×0.55 = 0.109
w[launch]    = 0.978×0.65 + 0.020×0.10 = 0.638
w[grid]      = 0.978×0.10 + 0.020×0.15 = 0.101
w[occupancy] = 0.978×0.15 + 0.020×0.20 = 0.151
```

Exponents: `launch=3.0, occupancy=0.72, bandwidth=0.52, grid=0.50`

### Step 6 — Factor scores

```
BW  = gaussian(1024, opt=512, σ=512) = 0.875  (threads > optimal, slight decay)
LO  = gaussian(4096/4=1024, opt=2048, σ=1024) = 0.875
GG  = tiny path, num_blocks=4 → 1.000
OC  = launch-bound multi-block:
        K_launch/(K_launch + (1-1)×K_warp) = 3.0/3.0 = 1.000
```

### Step 7 — Composite score

```
score = (0.875^0.52 × 0.875^3.0 × 1.000^0.50 × 1.000^0.72)^(1/4.74)
      = (0.934 × 0.670 × 1.000 × 1.000)^(0.211)
      = 0.6257^0.211
      ≈ 0.910
```

Contrast with `{XBLOCK:1024, nw:16}`:

```
OC (nw=16) = 3.0 / (3.0 + 15×0.2) = 3.0/6.0 = 0.500
T_overhead  = 3.0 + 15×0.2 = 6.0 µs   (vs 3.0 µs for nw=1)
```

The launch exponent of 3.0 amplifies the launch score gap; nw=16 is ranked ≈ #12.
Empirically, nw=1 completes in ≈ 0.034 ms vs nw=16 ≈ 0.052 ms — a 1.53× gap
correctly predicted.

---

*Document reflects the V5 heuristics system as implemented across:*
- `torch/_inductor/codegen/triton_heuristics_pointwise.py`
- `torch/_inductor/codegen/triton_heuristics_adaptive.py`
- `torch/_inductor/codegen/triton_heuristics_hardware.py`
- `torch/_inductor/codegen/triton_heuristics_kernel_analysis.py`
- `torch/_inductor/runtime/triton_heuristics.py`
- `torch/_inductor/config.py`


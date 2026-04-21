# Stage 2 — Extract Problem Metadata

**Sources:**
- Phase 1: `pointwise()` in `triton_heuristics.py` (at kernel-generation time)
- Phase 2: `_score_and_prune_heuristic_configs()` in `triton_heuristics.py`
- Phase 3: `extract_kernel_metadata()` in `triton_heuristics_kernel_analysis.py`

**Role in pipeline:** Assembles a `problem_metadata` dict describing the kernel's problem
shape, data types, tensor counts, instruction mix, and device limits.  Every scoring
formula in Stages 3–5 reads from this dict — it is computed once and reused 30–200 times.

---

## Technical

### Phase 1 — Shape & Device constants

Called at kernel-generation time, when Inductor emits the Triton function.  This is the
earliest possible point — the Triton source has not yet been written to disk:

```python
problem_metadata = {
    # Tensor shape
    'dimensions':       tuple(next_power_of_2(s) for s in size_hints.values()),
    'total_elements':   prod(size_hints.values()),
    'element_size':     dtype_to_bytes(triton_meta['dtype']),   # 4 for fp32, 2 for fp16

    # Device limits (from torch.cuda.get_device_properties)
    'warp_size':                device_props.warp_size,          # 64 AMD, 32 NVIDIA
    'max_threads_per_block':    device_props.max_threads_per_block,
    'num_cus':                  device_props.multi_processor_count,
    'peak_bandwidth_gbs':       device_props.memory_bandwidth_gb_per_s,
    'peak_tflops':              device_props.peak_fp32_tflops,
}
```

`size_hints` values are rounded up to the next power-of-two.  Triton emits a bounds mask
for tail elements:
```triton
offs = tl.arange(0, XBLOCK) + pid * XBLOCK
mask = offs < n                                # tail guard
x    = tl.load(ptr + offs, mask=mask)
```
The mask arithmetic `offs < n` is cheapest when `n` is aligned to a power-of-two because
the compiler can fold it into a bitwise compare rather than a general integer comparison.

### Phase 2 — Tensor counts

After `self.fn` (the `JITFunction` object) is available in `CachingAutotuner.__init__`,
two complementary passes determine `num_inputs` and `num_outputs`.

#### Pass A — regex over `fn.src` (always available, rough)

```python
num_inputs  = len(re.findall(r'\btl\.load\b',  kernel_code))
num_outputs = len(re.findall(r'\btl\.store\b', kernel_code))
```

Fast but can be confused by string literals or comments containing the pattern.

#### Pass B — from `fn.arg_names` (preferred, authoritative)

```python
ptr_args    = [a for a in self.fn.arg_names if a.endswith('_ptr')]
num_outputs = max(1, kernel_code.count('tl.store'))
num_inputs  = max(0, len(ptr_args) - num_outputs)
```

`fn.arg_names` is the compiler's own parameter list — it cannot contain comments or string
literals.  Inductor generates exactly one `tl.store` per output tensor, so counting them
in `fn.src` is exact.  Pass B supersedes Pass A and updates `problem_metadata` in-place.

The result feeds `bytes_per_element`:
```
bytes_per_element = element_size × (num_inputs + num_outputs)
```
This represents the total bytes the kernel reads and writes per output element.  For
`c = a + b` in FP32: 4 bytes read from `a` + 4 from `b` + 4 written to `c` = 12 bytes.

Broadcast tensors (where a tensor contributes far fewer bytes because the same value is
reused across many output elements) are detected in Phase 3 and can reduce this estimate.

### Phase 2.5 — Cache-hit recovery

`_score_and_prune_heuristic_configs` is called from `CachingAutotuner.__init__`.  On the
**cold path**, `pointwise()` executes, builds `problem_metadata`, and injects it into
`inductor_meta` under the key `_heuristics_pending`:

```python
# Inside pointwise() — cold path
inductor_meta['_heuristics_pending'] = {
    'problem_metadata': problem_metadata,
    'size_hints': size_hints,
}
```

On a **cache hit** (module loaded from `PyCodeCache`, `FxGraphCache`, etc.), `pointwise()`
is not re-executed — the `@pointwise` decorator is never re-applied to the cached function
object.  As a result, `_heuristics_pending` is absent from `inductor_meta`.

Recovery path:

```python
pending = self.inductor_meta.pop('_heuristics_pending', None)
if pending is None:
    # Cache hit: reconstruct from always-available size_hints
    problem_metadata = _convert_to_pointwise_heuristics_metadata(
        self.size_hints, self.inductor_meta, self.triton_meta
    )
    pending = {'problem_metadata': problem_metadata, 'size_hints': self.size_hints}
    log.info(
        "[HEURISTICS] Reconstructed problem_metadata from size_hints=%s "
        "(module was loaded from cache; pointwise() was not re-executed)",
        self.size_hints,
    )
```

`self.size_hints` is always available (it is a constructor argument), so recovery is
always possible.  The reconstructed metadata is equivalent to what `pointwise()` would
have produced, because both call the same device-property queries.

### Phase 3 — Instruction mix and op density

`extract_kernel_metadata(kernel_code)` in `triton_heuristics_kernel_analysis.py` scans
the Triton source with regex patterns to classify instructions:

| Category | Regex patterns | Approximate GPU cycles | Registers needed per op |
|---|---|---|---|
| **Fast** | `tl.add`, `tl.mul`, `tl.maximum`, `tl.where`, FMA forms | 1–4 | 1 (high reuse) |
| **Medium** | `tl.sqrt`, `tl.abs`, `tl.floor`, `tl.ceil`, `%` (int div/mod) | 4–16 | 3 (1-2 temporaries) |
| **Slow** | `tl.exp`, `tl.log`, `tl.sin`, `tl.cos`, `tl.div` (FP) | 16–64 | 6 (~5 temp regs) |

**FMA (Fused Multiply-Add):** The hardware instruction `fma a, b, c → a×b+c` executes in
a single pipeline pass at 1 cycle, counting as two FLOPs.  It is the building block of
virtually all neural network math — matrix multiply, layer norm scale/shift, attention
dot-product, gating activations.  The compiler fuses separate multiply and add operations
into FMA automatically.

**Integer div/mod:** CDNA has no integer division hardware unit.  The compiler emits a
multi-instruction sequence: compute reciprocal (FP), multiply, round, correction step.
This costs 4–16 cycles and is common in index arithmetic (`pid * BLOCK + offset % stride`).

**FP division (`tl.div`):** Routes through the Special Function Unit (SFU/Transcendental
Unit).  The SFU computes a reciprocal estimate via table lookup, then applies one or two
Newton–Raphson refinement steps to reach full FP32 precision.  Cost: 16–32 cycles vs.
1 cycle for FMA.  Expressions like `1.0 / normaliser` or `a / b` in Python source look
cheap but are 16–32× slower than multiplication.

**`ops_per_element`** is the weighted-average instruction cost per output element:

```
ops_per_element = (n_fast × 1 + n_med × 4 + n_slow × 32) / total_elements
```

This feeds the **Arithmetic Intensity** calculation:

```
AI = ops_per_element / bytes_per_element        [FLOPs / byte]
```

And the **OI ceiling** (ridge point of the roofline model):

```
OI_ceiling = peak_FLOPS / peak_bandwidth
           = (200 × 10¹²) / (5300 × 10⁹)       ← MI300X example
           ≈ 37.7 FLOPs / byte
```

If `AI < OI_ceiling` → **memory-bound**: ALUs finish faster than HBM delivers data,
ALUs sit idle.  For almost all pointwise kernels (relu, add, silu, etc.) `AI` is
0.1–2 FLOPs/byte — far below the ~38 FLOPs/byte ridge.

If `AI ≥ OI_ceiling` → **compute-bound**: data arrives faster than the ALUs can process
it.  Only kernels with heavy transcendentals (chains of `exp`, `log`, `sin`) approach
this regime.

> **Note:** This `AI < OI_ceiling` check in Stage 2 is a **fast coarse gate** — it
> classifies the kernel once from source analysis.  Stage 3 builds the full time model
> per-config (`T_overhead`, `T_memory`, `T_compute`) using this pre-computed `AI`.

**`has_broadcast`** is flagged when a tensor argument appears in `tl.load` but not with
the full output index range — detected by comparing the load pointer stride patterns.
A broadcast tensor contributes far fewer bytes per output element.

**`has_mask`** is flagged when the source contains `tl.load(..., mask=` or
`tl.store(..., mask=`.  This reduces the HBM efficiency factor η from 0.80 to 0.65 in
the memory time model, because:
- Predicated stores force **partial cache-line stores** — the memory controller must first
  read the existing cache line, merge the predicated bytes, then write back.  This adds
  one extra HBM round-trip per cache line with a partial write.
- On CDNA hardware, write-combining (merging multiple stores into one HBM burst) is
  disabled for masked stores.

---

## Simple explanation

### What this stage does

Before scoring any config, the system reads the kernel's "recipe card" — facts about what
the kernel does and how much data it moves.  Think of a chef checking the recipe
before deciding on the right pan size and cooking time.

### What information is collected and why

**Shape and total elements:** How many numbers does this kernel process?  A kernel with
1,000 elements behaves very differently from one with 100 million.  Small kernels are
dominated by the cost of *starting* on the GPU; large kernels are dominated by how fast
data can stream through memory.

**Element size (dtype):** FP32 (4 bytes) vs. FP16 (2 bytes) directly determines how many
bytes must cross the memory bus.  Halving the dtype doubles the effective bandwidth.

**Tensor count (inputs + outputs):** Each tensor is one full read or write pass over all
elements.  More tensors → more memory traffic → more likely the kernel is memory-bound.
The system gets this count from two sources and takes the more reliable one:
- **Regex scan** — quick but can be fooled by comments in the source
- **`fn.arg_names`** — the compiler's own list of function arguments; not foolable

**Instruction mix (fast / medium / slow):** Does the kernel mostly do cheap additions
(FMA), or does it call expensive transcendentals like `exp()` or `sin()`?
- Cheap ops (FMA): 1 cycle → lots of work per unit time → ALU probably not the limit
- Expensive ops (`exp`, `sin`): 16–64 cycles → ALU is likely the bottleneck even for
  memory-bound-looking kernels

**Masking:** If the problem size doesn't divide evenly by the block size, every load and
store must check bounds.  This breaks an optimisation called "write-combining" in the
memory controller, reducing effective bandwidth by ~20%.

### Where these facts go

All of this is stored in a `problem_metadata` dictionary that every subsequent stage reads.
It is computed once and reused for all 30–200 candidate configs, keeping total scoring
time in the millisecond range.

### Cache hits

If the kernel was already compiled and cached (from a previous run), Inductor loads it
directly without re-running the code that builds `problem_metadata`.  The system detects
this and reconstructs the metadata from the problem dimensions that are always available
in the cached module, so heuristic scoring works correctly even on warm cache hits.


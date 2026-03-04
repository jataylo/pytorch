# Heuristics TODO

## TODO-1 — Replace hardcoded `instructions_per_load` with kernel instruction mix

**File:** `torch/_inductor/codegen/triton_heuristics_hardware.py`

**Current code:**
```python
instructions_per_load = 2  # arithmetic ops between consecutive memory ops
```

**Problem:**
`instructions_per_load = 2` is a fixed assumption for a simple elementwise kernel.
Real kernels vary significantly:

| Kernel type | Actual I | Effect on computed N_min (optimal wavefronts) |
|---|---|---|
| Simple copy / cast | ~1 | N_min rises — more wavefronts needed |
| Elementwise add/mul (current assumption) | ~2 | N_min = 24 (baseline) |
| Gelu / layer norm | ~8–15 | N_min falls to ~8–12 wavefronts |
| Fused attention patterns | ~20+ | N_min ≈ 4–8 wavefronts |

**Fix:**
Extract `instructions_per_load` from `kernel_metadata['instruction_mix']` (already
computed in Stage 2) and pass it into `get_architecture_config()` or
`estimate_memory_bandwidth()` so the Gaussian peak shifts per-kernel.

`kernel_metadata` is available at scoring time via `problem_metadata`.  The relevant
field is something like `kernel_metadata.get('flops_per_element', 2)` or a ratio of
arithmetic to memory instructions from the instruction mix.

**Impact:** Affects `optimal_threads` and `σ_bw` (= 1.5 × optimal), which shifts the
Gaussian peak in `estimate_memory_bandwidth()`.  Compute-heavy kernels (high I) would
correctly prefer fewer warps; memory-thin kernels (low I) would correctly prefer more.

---

## TODO-3 — Factor `num_stages` into the ILP correction in `estimate_occupancy_impact`

**File:** `torch/_inductor/codegen/triton_heuristics_pointwise.py`

**Current code:**
```python
if elems_per_thread >= 8:
    return 1.00   # ILP: full benefit
elif elems_per_thread >= 4:
    return 0.93   # Partial ILP benefit
```

**Problem:**
`elems_per_thread` (EPT) is a proxy for software pipelining depth.  The actual mechanism
is `num_stages` in the `triton.Config` — the number of loop iterations the Triton
compiler prefetches simultaneously.  With `num_stages=S`, a single wavefront has `S`
loads in-flight at once, hiding `S × compute_time_per_iter` cycles of latency without
needing a second wavefront.

The correct ILP capacity is:
```
cycles_hidden = num_stages × elems_per_thread × issue_gap_per_element
```

The threshold for full single-wavefront latency hiding:
```
num_stages × I × issue_gap ≥ effective_latency
```

**Current state:**
On AMD (HIP, Triton > 3.2), `triton.py` codegen injects `num_stages=2` into
`tl.range()` for all pointwise element loops.  So currently `num_stages` is always 2
and does not vary per config — EPT alone is sufficient as a proxy.  If `num_stages`
ever becomes a tuned parameter for pointwise kernels, the ILP correction should use:

```python
pipeline_depth = config.get('num_stages', 1) * elems_per_thread
if pipeline_depth >= 8:
    # full ILP benefit
elif pipeline_depth >= 4:
    # partial benefit
```

**Impact:** Currently zero (num_stages is not a tuned pointwise config parameter).
Becomes important if num_stages tuning is added for pointwise kernels.

---

## TODO-2 — Replace `(n_args - 3) × K_arg` with `n_args × K_arg` in overhead calc

**File:** `torch/_inductor/codegen/triton_heuristics_adaptive.py`
(or wherever `T_overhead` / kernel launch overhead is computed)

**Current formula:**
```python
T_overhead = K_launch + (n_args - 3) * K_arg
```

**Problem:**
Subtracting 3 from `n_args` before multiplying by `K_arg` has no clear theoretical
justification.  The `- 3` was likely intended to exclude a fixed set of "standard"
arguments (e.g. output pointer, grid dims, stream) that are always present and whose
cost is already captured in `K_launch`.  However:
- The boundary between "fixed" and "variable" args is architecture-dependent
- For kernels with very few args, `n_args - 3` can go negative
- The correct model is: every additional argument adds `K_arg` overhead; the baseline
  cost (zero extra args) is `K_launch`

**Fix:**
```python
T_overhead = K_launch + n_args * K_arg
```
And re-calibrate `K_arg` empirically if needed, since the old `K_arg` was tuned
against the `- 3` form.

**Impact:** Affects `overhead_frac` in `analyze_bottleneck()`, which feeds into the
adaptive weight interpolation in Stage 3e.  Small kernels with many args would see
slightly higher predicted overhead, shifting their weights toward the overhead-bound
regime and increasing the Launch score's influence.


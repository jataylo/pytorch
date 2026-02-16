# Pointwise Heuristics V2 - First Principles Redesign

## Summary

Redesigned the pointwise kernel heuristics from first principles based on:
1. Analysis of validation data showing `load_balance` was always 1.0 (non-discriminating)
2. Physical understanding of memory-bound kernel behavior
3. Proper modeling of CDNA4 architecture characteristics

## Key Changes

### 1. Removed: Load Balance (was 40% weight)

**Why removed:**
- For power-of-2 problem sizes (most common), `load_balance` was always 1.000
- Example: (32, 1024) → all configs scored 1.000 for balance
- Non-discriminating factor provides no value in config selection
- Was artificially inflating scores without differentiating configs

### 2. New Factor: Memory Bandwidth (40% weight)

**First principles:**
Memory-bound pointwise kernels are limited by HBM bandwidth, not compute.

**Key insights:**
1. **Latency Hiding**: HBM latency ~300-500ns → need many threads in flight
2. **Coalescing**: Multiple threads accessing consecutive memory (cache line = 128B)
3. **Wavefront Alignment**: CDNA4 has 64-thread wavefronts → prefer multiples of 64
4. **Sweet Spot**: 256-512 threads per block (4-8 wavefronts)
   - Enough parallelism to hide latency
   - Good coalescing (multiple warps together)
   - Not so large that we hit resource limits

**Scoring:**
- Optimal: 256-512 threads → score = 1.0
- Good: 128-256 threads → score = 0.90-1.0 (linear)
- Acceptable: 512-1024 threads → score = 0.85-1.0 (diminishing returns)
- Poor: 64-128 threads → score = 0.70-0.90 (insufficient parallelism)
- Very poor: <64 threads → score = 0.60 (less than one wavefront)

### 3. Redesigned: Launch Overhead (30% weight, was 15%)

**Old logic:** Penalized high block counts uniformly
**Problem:** Didn't account for work per block or problem size

**New first principles:**
1. **Fixed Cost**: ~5μs per kernel launch
2. **Per-Block Cost**: ~100ns per block scheduling
3. **Amortization**: Want enough work per block to hide overhead
4. **Optimal**: 512-2048 elements per block

**Scoring:**
```python
elements_per_block = total_elements / num_blocks

Optimal: 512-2048 elem/blk → 1.0
Good:    256-512 elem/blk   → 0.90-1.0
Accept:  2048-4096 elem/blk → 0.95-1.0
Poor:    128-256 elem/blk   → 0.80-0.90
V.Poor:  <128 elem/blk      → 0.70
```

**Why this is better:**
- Accounts for actual work done per block
- Scales with problem size (small problems naturally have fewer blocks, that's OK)
- Models true amortization economics

### 4. Redesigned: Grid Granularity (20% weight, was 5%)

**Old logic:** Fixed target (3-8 blocks/CU regardless of problem size)
**Problem:** Small problems (<16K elements) were unfairly penalized

**New first principles:**
1. **GPU Saturation**: MI350 has 304 CUs → need enough blocks
2. **CU Capability**: Each CU handles 1-4 blocks optimally
3. **Problem Size Matters**: Can't create blocks out of nothing!

**Adaptive targets:**
```python
if total_elements < 16K:    # Small problem
    ideal_range = 16-64 blocks
elif total_elements < 256K: # Medium problem
    ideal_range = 64-512 blocks
else:                       # Large problem
    ideal_range = 256-1216 blocks (0.85-4 per CU)
```

**Scoring:**
- In ideal range → 1.0
- Below ideal (under-saturated) → 0.70-1.0 based on ratio
- Above ideal (excessive overhead) → 0.70-1.0 based on ratio

**Why this is better:**
- Adapts to problem size (small problems naturally have fewer blocks)
- Recognizes that 32 blocks for 4K elements is GOOD, not bad
- Balances GPU saturation with launch overhead

### 5. Redesigned: Occupancy (10% weight, same)

**Old logic:** Based on wavefront count and alignment
**Keep:** Core logic is sound

**Refinements:**
- Still checks wavefront alignment (threads % 64 == 0)
- Still prefers 2-8 wavefronts per block
- Minor penalty adjustments

**Scoring:**
- Optimal: 2-8 wavefronts, aligned → 1.0
- Good: 2-8 wavefronts, not aligned → 0.95
- Acceptable: 1 wavefront → 0.80-0.85
- Acceptable: 9-16 wavefronts → 0.85-0.90
- Poor: >16 wavefronts → 0.70

**Why 10% weight:**
- For pointwise: low registers, no LDS → occupancy usually good anyway
- Less discriminating than bandwidth or launch overhead
- But alignment still matters for efficiency

## New Weight Distribution

```
OLD V1:                              NEW V2:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Load Balance:      40% ❌           Memory Bandwidth:  40% ✅ NEW
Launch Overhead:   30% ⚠️           Launch Overhead:   30% ✅ IMPROVED
Occupancy:         20% ✅           Grid Granularity:  20% ✅ IMPROVED
Grid Granularity:  10% ⚠️           Occupancy:         10% ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Expected Improvements

### 1. Better Config Discrimination
- **Before**: All configs had balance=1.0, scores clustered tightly
- **After**: Bandwidth varies 0.60-1.0 based on thread count → better discrimination

### 2. More Accurate Predictions
- **Before**: (32, 1024) predicted XBLOCK=4,YBLOCK=16 (64 threads) as #1
  - Actual best: XBLOCK=8,YBLOCK=16 (128 threads)
  - Why wrong: Didn't model bandwidth needs
- **After**: Should favor 256-512 thread configs → better alignment

### 3. Problem-Size Awareness
- **Before**: Small problems penalized for having few blocks
- **After**: Adaptive targets → small problems not unfairly penalized

### 4. Physics-Based Scoring
- **Before**: Some heuristic rules without clear physical basis
- **After**: Every factor derived from first principles:
  - Bandwidth → memory latency hiding
  - Launch → work amortization
  - Grid → GPU saturation vs overhead trade-off
  - Occupancy → wavefront scheduling

## Implementation Details

### Files Modified:
1. **`triton_heuristics_pointwise.py`**:
   - Removed `estimate_load_balance()`
   - Added `estimate_memory_bandwidth()`
   - Updated `estimate_launch_overhead()` to take `problem_metadata`
   - Updated `estimate_grid_granularity()` to take `problem_metadata` and use adaptive targets
   - Updated `score_config()` to use new weights
   - Updated `get_detailed_scores()` to return new factor names

2. **`triton_heuristics.py`**:
   - Updated all logging to use new factor names
   - Updated validation summary to compare new factors
   - Updated root cause analysis to use new factors

### Backward Compatibility:
- Config generation logic unchanged
- Validation system unchanged
- Only scoring logic modified

## Testing Plan

```bash
# 1. Quick smoke test
python /root/benchmark_pointwise.py

# 2. Check for new patterns
# Look for:
# - bandwidth scores varying (0.6-1.0)
# - launch scores reflecting work per block
# - grid scores adapting to problem size

# 3. Validation
# Compare predicted best vs actual best
# Should see improved alignment
```

## Success Metrics

1. **Config scores are more spread out** (not all clustered near 1.0)
2. **Predicted #1 matches actual #1 more often** (accuracy >70%)
3. **When wrong, predicted is within 1.15x of actual** (acceptable)
4. **Heuristics favor 256-512 thread configs** (aligned with bandwidth theory)

## Next Steps

1. Run comprehensive benchmark suite
2. Analyze validation summaries
3. If needed, fine-tune weight distribution (40/30/20/10)
4. Consider adding memory transaction modeling for multi-dimensional problems


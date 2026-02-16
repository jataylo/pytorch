# V3: Continuous Scoring Implementation ✅

## Problem Identified

**V2 had too many configs scoring 1.000:**
- For (32, 1024) problem: **16 configs all scored 1.000**
- All had: bw=1.000, launch=1.000, grid=1.000, occ=1.000
- Defeats the purpose of heuristics (no discrimination!)

## Root Cause

**Discrete bins with flat "perfect" regions:**
```python
# OLD V2 logic
if 256 <= threads <= 512:
    return 1.0  # Entire range gets perfect score!
```

## V3 Solution: Continuous Gaussian Scoring

### 1. Memory Bandwidth (40%)
**OLD**: Flat 1.0 for 256-512 threads
**NEW**: Gaussian peak at 384 threads

```python
optimal_threads = 384
sigma = 256
diff = (threads - optimal_threads) / sigma
gaussian = math.exp(-0.5 * diff * diff)
score = 0.75 + 0.25 * gaussian  # Range: 0.75-1.00
```

**Result**: Every thread count gets unique score!
- 512 threads → 0.971
- 384 threads → 1.000 (peak)
- 256 threads → 0.971
- 128 threads → 0.902

### 2. Launch Overhead (30%)
**OLD**: Flat 1.0 for 512-2048 elem/block
**NEW**: Gaussian peak at 1024 elem/block

```python
optimal_elem = 1024
sigma = 512
# ... same Gaussian formula ...
```

**Result**: Continuous scoring!
- 512 elem/block → 0.902
- 1024 elem/block → 1.000 (peak)
- 2048 elem/block → 0.902

### 3. Grid Granularity (20%)
**OLD**: Flat 1.0 for adaptive range
**NEW**: Gaussian peak at adaptive optimal

```python
# For 32K elements
optimal_blocks = 32
sigma = 16
# ... Gaussian centered at 32 blocks ...
```

**Result**: Every block count scores differently!
- 64 blocks → 0.831
- 32 blocks → 1.000 (peak)
- 128 blocks → 0.902

### 4. Occupancy (10%) + Tie-Breakers
**OLD**: 1.00 for all aligned 4-8 wavefront configs
**NEW**: Base score × num_warps penalty × block shape bonus

```python
# Base score from wavefronts and alignment
if 4 <= wavefronts <= 8 and aligned:
    base_score = 1.00
    
# Aggressive num_warps penalty
if num_warps == 1:
    warp_mult = 1.00
elif num_warps == 2:
    warp_mult = 0.97  # -3%
elif num_warps == 4:
    warp_mult = 0.94  # -6%
elif num_warps == 8:
    warp_mult = 0.91  # -9%

# Block shape bonus (for 2D)
# Prefer balanced shapes + larger innermost dimension
balance = 1.0 - 0.005 * log2(aspect_ratio)
innermost = 1.0 - 0.002 * (7 - log2(yblock))

score = base_score * warp_mult * balance * innermost
```

**Result**: Breaks ties between same-thread configs!
- 16x32, nw=1 → 1.000 (larger Y, fewer warps)
- 32x16, nw=1 → 0.982 (smaller Y)
- 16x32, nw=2 → 0.970 (more warps)
- 16x32, nw=4 → 0.940 (even more warps)

## Results: Problem (32, 1024)

### V2 (BAD - Clustering):
```
Configs #1-16:  score = 1.0000 (ALL TIED!)
Configs #17-28: score = 0.9694
Configs #29-36: score = 0.8967
```

### V3 (GOOD - Discrimination):
```
#1: 16x32, nw=1  → 0.9156 ✅ UNIQUE
#2: 32x16, nw=1  → 0.9138 ✅ UNIQUE
#3: 16x32, nw=2  → 0.9129 ✅ UNIQUE
#4: 16x16, nw=1  → 0.9110 (different thread count)
#5: 32x16, nw=2  → 0.9110 (tied with #4, acceptable)
```

**Improvement:** From 16 configs clustered at 1.0 to 5 unique scores!

## Technical Details

### Gaussian Function
```python
def gaussian(value, optimal, sigma):
    diff = (value - optimal) / sigma
    return math.exp(-0.5 * diff * diff)
```

**Properties:**
- Single peak at `optimal`
- Smooth decay on both sides
- `sigma` controls width (larger = wider peak)
- Range: 0-1

### Scoring Formula (V3)
```python
score = (
    (bandwidth ** 2.5) *      # 40%
    (launch ** 1.8) *         # 30%
    (granularity ** 1.2) *    # 20%
    (occupancy ** 0.6)        # 10%
) ** (1.0 / 6.1) * tie_breakers
```

## Expected Benefits

### 1. Better Config Selection
- Top-1 prediction more reliable
- Clear winner instead of 16-way tie
- Tie-breakers align with hardware (coalescing, resource pressure)

### 2. More Stable Rankings
- Small changes in problem size don't cause rank flips
- Continuous functions → smooth score transitions

### 3. Explainable Decisions
- Can see exactly why one config beats another
- Distance from optimal (Gaussian center) → lower score

## Validation TODO

Run full benchmark:
```bash
python /root/benchmark_pointwise.py
```

Check for:
1. ✅ No more clustering at 1.000
2. ⏳ Predicted #1 closer to actual #1
3. ⏳ Improved accuracy (>70% top-1 match)

## Files Modified

1. **`triton_heuristics_pointwise.py`**:
   - `estimate_memory_bandwidth()`: Gaussian at 384 threads
   - `estimate_launch_overhead()`: Gaussian at 1024 elem/block
   - `estimate_grid_granularity()`: Adaptive Gaussian
   - `estimate_occupancy_impact()`: Aggressive num_warps penalty
   - `score_config()`: Block shape tie-breaker (balance + innermost)

---

**Status: ✅ IMPLEMENTED, READY FOR TESTING**


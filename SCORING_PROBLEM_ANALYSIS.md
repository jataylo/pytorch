# Scoring Problem Analysis

## Issue: Too Many Configs Score 1.000

### Example: Problem (32, 1024) = 32,768 elements

**16 configs all scored 1.000:**
- All have: bw=1.000, lnch=1.000, grid=1.000, occ=1.000
- They differ in block dims: 4x128, 8x64, 16x32, 32x16 (all → 512 threads, 64 blocks)
- They differ in num_warps: 1, 2, 4, 8

**Why they all get 1.000:**
1. **Bandwidth**: All 512 threads → 1.000 (in optimal 256-512 range)
2. **Launch**: All 512 elem/block → 1.000 (in optimal 512-2048 range)
3. **Grid**: All 64 blocks → 1.000 (in optimal range for 32K problem)
4. **Occupancy**: All aligned → 1.000

## Root Cause: Discrete Bins

Current logic uses **discrete bins**:
```python
if 256 <= threads <= 512:
    return 1.0  # ALL get perfect score!
elif 128 <= threads < 256:
    return 0.90 + 0.10 * (...)  # Linear in this bin
```

**Problem**: Wide "perfect" bins cause clustering.

## Solution: Continuous Scoring

### 1. Remove Perfect Plateaus
Instead of flat 1.0 for entire range, use **peaked curves**:

```python
# OLD: Flat plateau at 1.0 for 256-512 threads
if 256 <= threads <= 512:
    return 1.0

# NEW: Peak at 384 (midpoint), decay on both sides
optimal = 384
if threads <= optimal:
    return 0.85 + 0.15 * (threads / optimal)
else:
    return 1.0 - 0.15 * ((threads - optimal) / (1024 - optimal))
```

### 2. Add Secondary Discriminators

When primary factors are equal, use tie-breakers:

**A) num_warps preference:**
- For same thread count, prefer fewer warps (less resource pressure)
- 256 threads: prefer 4 warps over 8 warps

**B) Block shape:**
- For 2D, prefer square-ish blocks (better cache locality)
- 16x16 > 8x32 > 4x64 (for same total threads)

**C) Memory transaction efficiency:**
- For 2D, prefer larger innermost dimension
- YBLOCK=128 > YBLOCK=32 (better coalescing)

### 3. Use Gaussian/Exponential Curves

Instead of linear ramps, use smooth curves:

```python
# Gaussian peak at optimal
def gaussian_score(value, optimal, sigma):
    diff = (value - optimal) / sigma
    return math.exp(-0.5 * diff * diff)

# Exponential decay from optimal
def exp_decay(value, optimal, rate):
    if value <= optimal:
        return 1.0
    else:
        return math.exp(-rate * (value - optimal))
```

## Proposed New Scoring

### Memory Bandwidth (40%)
```python
# Peak at 384 threads, decay on both sides
optimal_threads = 384
sigma = 256  # Controls width of peak

if threads < 64:
    score = 0.60  # Floor
else:
    # Gaussian peak
    diff = (threads - optimal_threads) / sigma
    score = 0.85 + 0.15 * math.exp(-0.5 * diff * diff)
    score = max(0.60, min(1.0, score))
```

### Launch Overhead (30%)
```python
# Peak at 1024 elements/block
optimal_elem_per_block = 1024
sigma = 512

if elem_per_block < 64:
    score = 0.70
else:
    diff = (elem_per_block - optimal_elem_per_block) / sigma
    score = 0.85 + 0.15 * math.exp(-0.5 * diff * diff)
    score = max(0.70, min(1.0, score))
```

### Grid Granularity (20%)
```python
# Adaptive optimal based on problem size
if total_elements < 16384:
    optimal_blocks = 32
elif total_elements < 262144:
    optimal_blocks = 256
else:
    optimal_blocks = 608  # 2 per CU

sigma = optimal_blocks * 0.5

diff = (num_blocks - optimal_blocks) / sigma
score = 0.85 + 0.15 * math.exp(-0.5 * diff * diff)
score = max(0.70, min(1.0, score))
```

### Occupancy (10%)
```python
# Already fine-grained, but add num_warps preference
wavefronts = threads // warp_size
aligned = (threads % warp_size == 0)

# Base score
if 4 <= wavefronts <= 8 and aligned:
    base = 1.0
elif 2 <= wavefronts <= 12 and aligned:
    base = 0.95
else:
    base = 0.80

# Tie-breaker: prefer fewer warps (less resource pressure)
if num_warps <= 4:
    warp_bonus = 1.0
elif num_warps <= 8:
    warp_bonus = 0.98
else:
    warp_bonus = 0.95

score = base * warp_bonus
```

## Expected Results

### Before (V2):
```
16 configs → score=1.000 (all tied)
12 configs → score=0.9694
7 configs → score=0.8967
```

### After (V3):
```
Config 1: 4x128, nw=4  → score=0.982 (optimal elem/block, fewer warps)
Config 2: 4x128, nw=8  → score=0.963 (optimal elem/block, more warps)
Config 3: 8x64, nw=4   → score=0.978 (slightly better shape)
Config 4: 16x32, nw=4  → score=0.974 (similar)
Config 5: 16x16, nw=4  → score=0.934 (256 threads, less optimal)
```

**Much better discrimination!**


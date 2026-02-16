# Heuristics Problems Found from log.log Analysis

## Critical Findings

### 1. ❌ Balance is ALWAYS 1.000 - Remove It!

**Evidence from log**:
```
Balance : Predicted=1.000 vs Actual=1.000 (≈ same)  # Every single case!
```

**Why**: 
- Benchmark uses power-of-2 problem sizes: 512, 4096, 16384, etc.
- All block sizes are also power-of-2: 64, 128, 256, 512
- **Perfect division**: 512 ÷ 64 = 8 (no remainder!)
- Load balance calculates wasted threads from remainders
- No remainders = no waste = always 1.000

**Impact**: 
- 40% of scoring weight is wasted on a constant!
- Contributes ZERO information to ranking

**Fix**: Remove balance completely, redistribute weight to other factors

---

### 2. ❌ Grid Granularity Logic is BACKWARDS

**Evidence from log**:
```
Problem: (512,)
Predicted Best: XBLOCK=64,  Grid: 8 blocks   → grid=0.703 (rank #1)
Actual Best:    XBLOCK=512, Grid: 1 block    → grid=0.700 (rank #8)

Problem: (4096,)
Predicted Best: XBLOCK=64,  Grid: 64 blocks  → grid=0.721 (rank #1)
Actual Best:    XBLOCK=128, Grid: 32 blocks  → grid=0.711 (rank #2)

Problem: (16384,)
Predicted Best: XBLOCK=64,  Grid: 256 blocks → grid=0.784 (rank #1)
Actual Best:    XBLOCK=512, Grid: 32 blocks  → grid=0.711 (rank #6)
```

**Pattern**: 
- Predicted best ALWAYS has MORE blocks → HIGHER grid score
- Actual best ALWAYS has FEWER blocks → LOWER grid score
- But actual best is FASTER!

**Current Logic** (`estimate_grid_granularity`):
```python
ideal_min = NUM_CUS * 3   # ~900 blocks
ideal_max = NUM_CUS * 8   # ~2400 blocks

if num_blocks < ideal_min:  # Too few blocks
    ratio = num_blocks / ideal_min
    return 0.7 + 0.3 * ratio  # Penalty!
```

For 8 blocks: `0.7 + 0.3 * (8/900) = 0.703`
For 256 blocks: `0.7 + 0.3 * (256/900) = 0.785`

**Problem**: 
- Logic assumes we need 900-2400 blocks for good utilization
- But for SMALL problem sizes (512-16K elements), this is WRONG!
- Actual best uses 1-64 blocks, not 900+
- **The penalty is too harsh for small grids**

**Why Current Logic Fails**:
1. For pointwise kernels (memory-bound, not compute-bound):
   - Fewer, larger blocks = **better memory bandwidth**
   - More work per block = **amortizes launch overhead**
   - Better **cache locality** within block

2. The 900-2400 blocks target is for **compute-intensive** kernels
   - Pointwise is **memory-intensive**!
   - Different optimization strategy needed

---

### 3. ⚠️ Occupancy is MOSTLY 1.000 - Low Discrimination

**Evidence**:
```
Occupancy: Predicted=1.000 vs Actual=1.000  # 90% of cases
Occupancy: Predicted=1.000 vs Actual=0.850  # Only for 1024 threads/block
```

**Why**:
- Only penalizes configs with 1024 threads/block (drops to 0.850)
- All other configs (64-512 threads) score 1.000
- **Not discriminating** between 64, 128, 256, 512 thread configs

**Impact**: 20% weight, but only matters for 1024-thread configs

---

### 4. ❌ Missing Critical Factors

#### A. Memory Bandwidth Utilization
**Why it matters**: Pointwise kernels are **memory-bound**!

- Larger blocks process more elements per launch
- Better amortization of memory transaction overhead
- Current heuristics don't model this AT ALL

**Evidence**:
- XBLOCK=512 (faster) vs XBLOCK=64 (predicted best)
- 512 threads process 8x more data per transaction
- But heuristics prefer 64 (more blocks)

#### B. Warp Efficiency
- 64 threads = 1 warp (perfect)
- 128 threads = 2 warps (perfect)
- 512 threads = 8 warps (perfect)
- All should score similarly, but launch overhead differs

#### C. Cache Utilization
- Larger blocks = more data stays in L1/L2 cache
- Better temporal locality
- Not modeled

---

## Corrected Understanding

### For Pointwise Kernels (Memory-Bound):

**WANT**:
- ✅ Fewer, larger blocks (1-128 blocks)
- ✅ Larger thread counts (256-512 threads)
- ✅ Better memory bandwidth utilization
- ✅ Amortized launch overhead

**DON'T WANT**:
- ❌ Many small blocks (256+ blocks)
- ❌ Tiny thread counts (64 threads)
- ❌ High launch overhead from many blocks

**Current heuristics do the OPPOSITE**!

---

## Recommendations

### 1. REMOVE Balance (40% weight freed)

```python
# Remove this completely
# balance = estimate_load_balance(...)
```

### 2. FIX Grid Granularity (or replace)

**Option A**: Reverse the logic for pointwise
```python
def estimate_grid_granularity(grid_size, problem_metadata):
    """For pointwise: prefer FEWER blocks with more work each"""
    num_blocks = prod(grid_size)
    total_elements = problem_metadata['total_elements']
    work_per_block = total_elements / num_blocks
    
    # Prefer 256-2048 elements per block (sweet spot for memory bandwidth)
    if 256 <= work_per_block <= 2048:
        return 1.0
    elif work_per_block < 256:
        return 0.7 + 0.3 * (work_per_block / 256)  # Too little work
    else:
        return 0.9  # Large work is OK for pointwise
```

**Option B**: Use block count relative to problem size
```python
# For small problems (<32K elements): prefer 1-64 blocks
# For medium problems (32K-1M): prefer 64-256 blocks  
# For large problems (>1M): prefer 256-2048 blocks
```

### 3. ADD Memory Bandwidth Factor (30% weight)

```python
@staticmethod
def estimate_memory_bandwidth(config, problem_metadata):
    """
    Larger blocks = better bandwidth utilization for memory-bound kernels.
    """
    block_dims = get_block_dimensions(config)
    threads_per_block = prod(block_dims)
    
    # Sweet spot: 256-512 threads for memory bandwidth
    if 256 <= threads_per_block <= 512:
        return 1.0
    elif threads_per_block < 256:
        # Too small - poor bandwidth
        return 0.75 + 0.25 * (threads_per_block / 256)
    else:
        # 1024 threads OK but may hurt occupancy
        return 0.95
```

### 4. REDUCE Occupancy Weight (20% → 10%)

Since it's only discriminating for 1024-thread configs

### 5. New Weight Distribution

```python
score = (
    # balance ** 2.5,        # REMOVED (was 40%)
    memory_bandwidth ** 2.5,  # NEW! 40% - most important for pointwise
    launch ** 1.5,            # 25% (was 30%)
    occupancy ** 0.9,         # 10% (was 20%, reduced)
    grid ** 1.5,              # 25% (was 10%, but FIXED logic)
)
```

---

## Validation

After fixes, expect to see:
- ✅ XBLOCK=128-512 ranked higher
- ✅ XBLOCK=64 ranked lower
- ✅ Fewer blocks preferred for small problems
- ✅ Better alignment with actual performance

**Current**:
```
Predicted: XBLOCK=64, 256 blocks  (rank #1, slow)
Actual:    XBLOCK=512, 32 blocks  (rank #6, fast!)
```

**After fixes**:
```
Predicted: XBLOCK=512, 32 blocks  (rank #1-2)
Actual:    XBLOCK=512, 32 blocks  (rank #1, fast!)
```

---

## Summary

| Factor | Current | Problem | Fix |
|--------|---------|---------|-----|
| Balance | 40% | Always 1.0, useless | REMOVE |
| Grid | 10% | Logic backwards, prefers many blocks | FIX or REPLACE (25%) |
| Launch | 30% | OK but could be better | Keep (25%) |
| Occupancy | 20% | Rarely discriminates | Reduce (10%) |
| Memory BW | 0% | **MISSING** - most important! | ADD (40%) |

**Net effect**: Going from 2 discriminating factors to 4, with correct logic!


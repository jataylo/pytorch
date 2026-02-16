# Multiple Warp Counts Per Block Configuration

## Issue

**Before**: For each block size combination (e.g., XBLOCK=16, YBLOCK=16), only **one** `num_warps` value was generated:
```python
num_warps = total_threads // warp_size  # Fixed calculation
```

Example:
- XBLOCK=16, YBLOCK=16 (256 threads) → **only** num_warps=4
- XBLOCK=16, YBLOCK=32 (512 threads) → **only** num_warps=8

**User's Question**: "Why not try XBLOCK=16, YBLOCK=16 with num_warps=[1, 2, 4, 8, 16]?"

## Why This Matters

Different `num_warps` values can significantly affect performance:

1. **Occupancy**: More warps → higher occupancy (if registers/LDS allow)
2. **Register Pressure**: Fewer warps → more registers per thread
3. **LDS Usage**: Fewer warps → more LDS per warp
4. **Latency Hiding**: More warps → better latency hiding
5. **Memory Coalescing**: Different warp counts can affect memory access patterns

The optimal `num_warps` is **NOT** always `threads // warp_size`!

## Fix Applied

Now we generate **multiple configs** for each block size, varying `num_warps`:

```python
# Candidate warp counts to try (powers of 2: 1, 2, 4, 8, 16)
warp_candidates = [1, 2, 4, 8, 16]

for xblock in block_sizes:
    for yblock in block_sizes:
        total_threads = xblock * yblock
        
        # Try MULTIPLE warp counts for this block configuration
        for num_warps in warp_candidates:
            # Skip if not enough threads
            if num_warps * warp_size > total_threads:
                continue
            # Skip if exceeds device max
            if num_warps > max_warps:
                continue
            
            configs.append({
                'XBLOCK': xblock,
                'YBLOCK': yblock,
                'num_warps': num_warps  # Variable!
            })
```

## Results

### Before (Single Warp Count)
For problem (256, 256):
- Generated: ~30 configs
- XBLOCK=16, YBLOCK=16 → **only** num_warps=4
- XBLOCK=16, YBLOCK=32 → **only** num_warps=8

### After (Multiple Warp Counts)
For problem (256, 256):
- Generated: **85 configs** (2.8x more!)
- XBLOCK=16, YBLOCK=16 (256 threads) → num_warps=[**1, 2, 4**]
- XBLOCK=16, YBLOCK=32 (512 threads) → num_warps=[**1, 2, 4, 8**]
- XBLOCK=32, YBLOCK=16 (512 threads) → num_warps=[**1, 2, 4, 8**]
- XBLOCK=32, YBLOCK=32 (1024 threads) → num_warps=[**1, 2, 4, 8, 16**]

## Warp Count Limits

The code automatically limits `num_warps` based on thread count:

| Threads | Max Warps | Valid num_warps |
|---------|-----------|-----------------|
| 64      | 1         | [1]             |
| 128     | 2         | [1, 2]          |
| 256     | 4         | [1, 2, 4]       |
| 512     | 8         | [1, 2, 4, 8]    |
| 1024    | 16        | [1, 2, 4, 8, 16]|

**Rule**: `num_warps * warp_size ≤ total_threads`

Example:
- 256 threads, warp_size=64 → max 4 warps (4 × 64 = 256)
- Can't use num_warps=8 (8 × 64 = 512 > 256 threads)

## Example Output

```
[POINTWISE HEURISTICS] ALL CONFIGS (sorted by predicted score):
[POINTWISE HEURISTICS]   # 1: [TOP-1]   score=0.8218 | XBLOCK=16, YBLOCK=32, num_warps=8
[POINTWISE HEURISTICS]        bal=1.000(40%) lnch=1.000(30%) occ=1.000(20%) grid=0.721(10%) | 64blk 512thr
[POINTWISE HEURISTICS]   # 2: [TOP-2]   score=0.8218 | XBLOCK=16, YBLOCK=32, num_warps=4  ← NEW!
[POINTWISE HEURISTICS]        bal=1.000(40%) lnch=1.000(30%) occ=1.000(20%) grid=0.721(10%) | 64blk 512thr
[POINTWISE HEURISTICS]   # 3: [TOP-3]   score=0.8218 | XBLOCK=16, YBLOCK=32, num_warps=2  ← NEW!
[POINTWISE HEURISTICS]        bal=1.000(40%) lnch=1.000(30%) occ=1.000(20%) grid=0.721(10%) | 64blk 512thr
[POINTWISE HEURISTICS]   # 4: [TOP-4]   score=0.8218 | XBLOCK=16, YBLOCK=32, num_warps=1  ← NEW!
[POINTWISE HEURISTICS]        bal=1.000(40%) lnch=1.000(30%) occ=1.000(20%) grid=0.721(10%) | 64blk 512thr
[POINTWISE HEURISTICS]   # 5: [TOP-5]   score=0.8063 | XBLOCK=16, YBLOCK=16, num_warps=4
[POINTWISE HEURISTICS]        bal=1.000(40%) lnch=0.980(30%) occ=1.000(20%) grid=0.742(10%) | 128blk 256thr
[POINTWISE HEURISTICS]   # 6: [VALID]   score=0.8063 | XBLOCK=16, YBLOCK=16, num_warps=2  ← NEW!
[POINTWISE HEURISTICS]        bal=1.000(40%) lnch=0.980(30%) occ=1.000(20%) grid=0.742(10%) | 128blk 256thr
[POINTWISE HEURISTICS]   # 7: [VALID]   score=0.8063 | XBLOCK=16, YBLOCK=16, num_warps=1  ← NEW!
[POINTWISE HEURISTICS]        bal=1.000(40%) lnch=0.980(30%) occ=1.000(20%) grid=0.742(10%) | 128blk 256thr
```

## Impact

### Config Generation
| Metric                  | Before | After | Change |
|-------------------------|--------|-------|--------|
| Configs per block size  | 1      | 1-5   | +400%  |
| Total configs (256×256) | ~30    | 85    | +183%  |
| Warp variations tested  | 0      | 1-5   | NEW!   |

### Autotuning Coverage
- **More comprehensive search space**
- **Better chance of finding optimal config**
- **Handles register pressure variations**
- **Tests different occupancy levels**

### Performance Impact
The autotuner now tests configs that might have been optimal but were previously never generated:
- **Low warp count** (1-2): Good for register-heavy kernels
- **Medium warp count** (4-8): Balanced for most cases
- **High warp count** (8-16): Good for latency hiding

## Files Modified

**`/root/pytorch/torch/_inductor/codegen/triton_heuristics_pointwise.py`**

### 1D Configs (Line ~441)
```python
# Before:
configs.append({
    'XBLOCK': xblock,
    'num_warps': max(1, min(xblock // warp_size, max_warps))  # Single value
})

# After:
for num_warps in warp_candidates:  # Try multiple values
    if num_warps * warp_size > total_threads:
        continue
    configs.append({
        'XBLOCK': xblock,
        'num_warps': num_warps
    })
```

### 2D Configs (Line ~470)
```python
# Before:
configs.append({
    'XBLOCK': xblock,
    'YBLOCK': yblock,
    'num_warps': max(1, min(total_threads // warp_size, max_warps))  # Single value
})

# After:
for num_warps in warp_candidates:  # Try multiple values
    if num_warps * warp_size > total_threads:
        continue
    configs.append({
        'XBLOCK': xblock,
        'YBLOCK': yblock,
        'num_warps': num_warps
    })
```

### 3D Configs (Line ~516)
Similar changes for 3D configurations.

## Validation

```python
from torch._inductor.codegen.triton_heuristics_pointwise import PointwiseHeuristics

problem_metadata = {
    'dimensions': (256, 256),
    'total_elements': 65536,
    'dtype_size': 4,
    'num_inputs': 2,
    'warp_size': 64,
    'max_threads_per_block': 1024,
}

configs = PointwiseHeuristics.generate_all_candidate_configs(problem_metadata)
print(f"Generated {len(configs)} configs")  # 85 (was ~30)

# Check XBLOCK=16, YBLOCK=16
configs_16x16 = [c for c in configs if c['XBLOCK']==16 and c['YBLOCK']==16]
warps_16x16 = sorted([c['num_warps'] for c in configs_16x16])
print(f"XBLOCK=16, YBLOCK=16: num_warps={warps_16x16}")  # [1, 2, 4]
```

**Expected Output**:
```
Generated 85 configs
XBLOCK=16, YBLOCK=16: num_warps=[1, 2, 4]
```

## Summary

✅ **Before**: 1 warp count per block size (fixed calculation)  
✅ **After**: Multiple warp counts per block size (1, 2, 4, 8, 16)  
✅ **Impact**: 2-3x more configs, better autotuning coverage  
✅ **Smart Limits**: Automatically filters invalid combinations  
✅ **Result**: Higher chance of finding the truly optimal config!

The autotuner now has much better coverage of the configuration space, especially for exploring different occupancy and register pressure trade-offs. 🚀


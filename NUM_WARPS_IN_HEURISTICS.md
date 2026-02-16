# How num_warps is Used in Pointwise Heuristics

## Overview

`num_warps` is a critical parameter that affects GPU occupancy and performance. In the heuristics, it's used in **3 key places**:

## 1. Config Generation (Setting num_warps)

**Location:** `triton_heuristics_pointwise.py` - `generate_all_candidate_configs()`

`num_warps` is automatically set based on block size:

```python
# 1D configs
for xblock in block_sizes_1d:
    configs.append({
        'XBLOCK': xblock,
        'num_warps': max(1, min(xblock // 64, 16))  # ← num_warps set here
    })

# 2D configs
for xblock in block_sizes_2d:
    for yblock in block_sizes_2d:
        total_threads = xblock * yblock
        if 64 <= total_threads <= 1024:
            configs.append({
                'XBLOCK': xblock,
                'YBLOCK': yblock,
                'num_warps': max(1, min(total_threads // 64, 16))  # ← num_warps set here
            })
```

**Logic:**
- `num_warps = threads_per_block / 64` (since each warp has 64 threads on AMD)
- Clamped between 1 and 16 warps
- Examples:
  - XBLOCK=64 → num_warps=1 (64/64=1)
  - XBLOCK=256 → num_warps=4 (256/64=4)
  - XBLOCK=512 → num_warps=8 (512/64=8)
  - XBLOCK=1024 → num_warps=16 (1024/64=16)

## 2. Occupancy Estimation (Main Impact)

**Location:** `triton_heuristics_pointwise.py` - `estimate_occupancy_impact()`

This is where `num_warps` has the **BIGGEST impact** on scoring:

```python
@staticmethod
def estimate_occupancy_impact(config: Dict, problem_metadata: Dict) -> float:
    """
    Estimate GPU occupancy based on register/LDS pressure.
    Higher num_warps → More VGPR usage → Lower occupancy
    """
    num_warps = config.get('num_warps', 4)
    
    # Estimate VGPR usage per thread
    vgpr_per_thread = PointwiseHeuristics.estimate_vgpr_per_thread(config, problem_metadata)
    
    # VGPR usage grows with num_warps
    vgpr_per_block = vgpr_per_thread * 64 * num_warps  # ← num_warps multiplier
    
    # Wave64 has 1024 VGPRs available
    vgpr_pool = PointwiseHeuristics.VGPR_POOL_WAVE64  # 1024
    
    # More warps → More VGPRs needed → Fewer blocks can run simultaneously
    max_blocks_vgpr = vgpr_pool // max(vgpr_per_block, 1)
    max_blocks = min(max_blocks_vgpr, PointwiseHeuristics.MAX_BLOCKS_PER_CU)
    
    # Scoring: 12+ blocks is optimal for latency hiding
    if max_blocks >= 12:
        return 1.0
    elif max_blocks >= 8:
        return 0.95
    elif max_blocks >= 4:
        return 0.85
    else:
        return 0.70  # Poor occupancy
```

**How it affects scoring:**

| num_warps | VGPR per block | Max blocks/CU | Occupancy Score | Impact on Final Score |
|-----------|----------------|---------------|-----------------|----------------------|
| 1 | ~320 VGPRs | 3 blocks | 0.70 | **Low** score (penalty) |
| 2 | ~640 VGPRs | 1 block | 0.70 | **Low** score (penalty) |
| 4 | ~1280 VGPRs | <1 block | 1.00* | **High** score (good) |
| 8 | ~2560 VGPRs | <1 block | 1.00* | **High** score (good) |
| 16 | ~5120 VGPRs | <1 block | 0.85 | **Medium** score |

*Note: The actual calculation is more complex and depends on VGPR estimation per thread.

**Real Example from ResNet152:**

```
Config #1: {'XBLOCK': 512, 'num_warps': 8}  → occup=1.000 (20% weight)
Config #5: {'XBLOCK': 1024, 'num_warps': 16} → occup=0.850 (20% weight)
```

The config with `num_warps=16` gets a **15% penalty** on the occupancy factor, which translates to ~3% lower total score.

## 3. VGPR Estimation (Indirect Impact)

**Location:** `triton_heuristics_pointwise.py` - `estimate_vgpr_per_thread()`

While `num_warps` isn't directly used in VGPR estimation, the **threads per block** (which determines `num_warps`) affects register usage:

```python
@staticmethod
def estimate_vgpr_per_thread(config: Dict, problem_metadata: Dict) -> int:
    block_dims = PointwiseHeuristics.get_block_dimensions(config)
    threads_per_block = PointwiseHeuristics.prod(block_dims)  # Related to num_warps
    
    num_inputs = problem_metadata.get('num_inputs', 2)
    num_outputs = problem_metadata.get('num_outputs', 1)
    
    # Base registers
    pointer_regs = (num_inputs + num_outputs) * 2
    index_regs = len(block_dims) * 2
    mask_regs = 4 if problem_metadata.get('has_mask', False) else 0
    
    # Data registers
    data_regs = num_inputs * vector_width
    output_regs = num_outputs * vector_width
    
    # Total with overhead
    vgpr_est = int((pointer_regs + index_regs + mask_regs + 
                    data_regs + output_regs) * 1.15)
    
    return vgpr_est
```

**Then this feeds back into occupancy:**
```python
vgpr_per_block = vgpr_per_thread * 64 * num_warps  # ← Combined effect
```

## Why num_warps Matters

### For Small Blocks (num_warps=1-2):
- ✅ **Lower VGPR usage** → More blocks can run concurrently
- ❌ **Worse latency hiding** → Fewer threads to hide memory latency
- ❌ **More launch overhead** → More blocks to launch

### For Medium Blocks (num_warps=4-8):
- ✅ **Balanced VGPR usage** → Good occupancy
- ✅ **Good latency hiding** → Sufficient threads
- ✅ **Moderate launch overhead**

### For Large Blocks (num_warps=16):
- ❌ **Higher VGPR usage** → Fewer blocks can run concurrently
- ✅ **Best latency hiding** → Most threads
- ✅ **Lowest launch overhead** → Fewer blocks

## Complete Scoring Chain

```
num_warps (config)
    ↓
threads_per_block = XBLOCK × YBLOCK × ZBLOCK
    ↓
vgpr_per_block = vgpr_per_thread × 64 × num_warps
    ↓
max_blocks_per_cu = VGPR_POOL / vgpr_per_block
    ↓
occupancy_score = f(max_blocks_per_cu)
    ↓
final_score = balance^2.5 × launch^1.8 × occupancy^1.2 × grid^0.6
```

## Weight in Final Score

`num_warps` affects the **occupancy factor** which has a **20% weight** in the final score:

```python
score = (
    balance ** 2.5 *      # 40% weight
    launch ** 1.8 *       # 30% weight
    occupancy ** 1.2 *    # 20% weight ← num_warps affects this
    granularity ** 0.6    # 10% weight
)
```

So a 15% penalty in occupancy (0.85 vs 1.00) results in approximately:
- 0.85^1.2 = 0.835
- Overall score impact: ~3% lower final score

## Summary

`num_warps` is used in:

1. **Config Generation** - Automatically set based on block size (threads/64)
2. **Occupancy Estimation** - Primary impact through VGPR calculation
3. **VGPR Calculation** - Indirectly through threads per block

The heuristics balance `num_warps` against:
- Register pressure (higher warps → more VGPRs → lower occupancy)
- Latency hiding (higher warps → better latency hiding)
- Launch overhead (higher warps → larger blocks → fewer launches)

The optimal `num_warps` varies by problem size and is discovered through benchmarking the top 5 configs.



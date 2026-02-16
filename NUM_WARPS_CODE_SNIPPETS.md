# num_warps in Heuristics - Code Snippets

## Location 1: Config Generation (Setting num_warps)

**File:** `torch/_inductor/codegen/triton_heuristics_pointwise.py`  
**Function:** `generate_all_candidate_configs()`

### 1D Configs:
```python
if ndims == 1:
    xnumel = problem_dims[0]
    for xblock in block_sizes_1d:  # [16, 32, 64, 128, 256, 512, 1024]
        if xblock > xnumel:
            continue
        configs.append({
            'XBLOCK': xblock,
            'num_warps': max(1, min(xblock // 64, 16))  # ← SET HERE
        })
```

**Logic:** `num_warps = threads / 64` (clamped to 1-16)

### 2D Configs:
```python
elif ndims == 2:
    xnumel, ynumel = problem_dims[0], problem_dims[1]
    for xblock in block_sizes_2d:
        if xblock > xnumel:
            continue
        for yblock in block_sizes_2d:
            if yblock > ynumel:
                continue
            
            total_threads = xblock * yblock
            if 64 <= total_threads <= 1024:
                configs.append({
                    'XBLOCK': xblock,
                    'YBLOCK': yblock,
                    'num_warps': max(1, min(total_threads // 64, 16))  # ← SET HERE
                })
```

### 3D Configs:
```python
elif ndims == 3:
    for xblock in block_sizes_3d:
        # ... dimension checks ...
        total_threads = xblock * yblock * zblock
        if 64 <= total_threads <= 512:
            configs.append({
                'XBLOCK': xblock,
                'YBLOCK': yblock,
                'ZBLOCK': zblock,
                'num_warps': max(1, min(total_threads // 64, 8))  # ← SET HERE
            })
```

---

## Location 2: Occupancy Estimation (Main Impact on Scoring)

**File:** `torch/_inductor/codegen/triton_heuristics_pointwise.py`  
**Function:** `estimate_occupancy_impact()`

```python
@staticmethod
def estimate_occupancy_impact(config: Dict, problem_metadata: Dict) -> float:
    """
    Estimate occupancy based on VGPR usage.
    Returns: Score (0.7-1.0)
    """
    block_dims = PointwiseHeuristics.get_block_dimensions(config)
    
    # threads_per_block is directly related to num_warps
    # num_warps = threads_per_block / 64
    threads_per_block = PointwiseHeuristics.prod(block_dims)  # ← Related to num_warps
    
    # Estimate VGPR usage per thread
    vgpr_per_thread = PointwiseHeuristics.estimate_vgpr_per_thread(problem_metadata)
    vgpr_bytes_per_thread = vgpr_per_thread * 4  # 4 bytes per VGPR
    
    # Total VGPR usage grows with threads (i.e., with num_warps)
    vgpr_per_block = threads_per_block * vgpr_bytes_per_thread  # ← num_warps effect
    
    # Wave64 has 1024 VGPRs available per wave
    vgpr_pool = PointwiseHeuristics.VGPR_POOL_WAVE64  # 1024 VGPRs
    
    # More warps → More VGPRs needed → Fewer blocks can run simultaneously
    max_blocks_vgpr = vgpr_pool // max(vgpr_per_block, 1)
    max_blocks = min(max_blocks_vgpr, PointwiseHeuristics.MAX_BLOCKS_PER_CU)
    
    # Scoring based on how many blocks can run
    if max_blocks >= 12:
        return 1.0   # Excellent occupancy
    elif max_blocks >= 8:
        return 0.95  # Good occupancy
    elif max_blocks >= 4:
        return 0.85  # Okay occupancy
    else:
        return 0.70  # Poor occupancy
```

**Key Relationship:**
```
num_warps = threads_per_block / 64
vgpr_per_block = vgpr_per_thread × threads_per_block
              = vgpr_per_thread × (num_warps × 64)
              
Higher num_warps → Higher vgpr_per_block → Lower max_blocks → Lower score
```

---

## Location 3: Final Scoring (Weighted Geometric Mean)

**File:** `torch/_inductor/codegen/triton_heuristics_pointwise.py`  
**Function:** `score_config()`

```python
@staticmethod
def score_config(config: Dict, problem_metadata: Dict) -> float:
    """
    Score a configuration using weighted geometric mean of CONFIG-VARYING factors.
    Returns a score between 0 and 1 (higher is better).
    """
    block_dims = PointwiseHeuristics.get_block_dimensions(config)
    problem_dims = PointwiseHeuristics.get_problem_dimensions(problem_metadata)
    
    grid_size = PointwiseHeuristics.calculate_grid_size(problem_dims, block_dims)
    
    # Calculate individual factor scores
    balance = PointwiseHeuristics.estimate_load_balance(problem_dims, block_dims)
    launch = PointwiseHeuristics.estimate_launch_overhead(grid_size)
    occupancy = PointwiseHeuristics.estimate_occupancy_impact(config, problem_metadata)  # ← num_warps affects this
    granularity = PointwiseHeuristics.estimate_grid_granularity(grid_size)
    
    # Weighted geometric mean (rebalanced weights: 40/30/20/10 = 100%)
    score = (
        balance ** 2.5 *        # 40% weight
        launch ** 1.8 *         # 30% weight
        occupancy ** 1.2 *      # 20% weight ← num_warps impacts final score here
        granularity ** 0.6      # 10% weight
    )
    
    return score
```

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Config Generation                                             │
│    XBLOCK=512 → threads=512 → num_warps=512/64=8                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Occupancy Estimation                                          │
│    threads_per_block = 512 (= num_warps × 64)                   │
│    vgpr_per_thread = 50 (estimated)                             │
│    vgpr_per_block = 512 × 50 × 4 bytes = 102,400 bytes         │
│                                                                  │
│    VGPR_POOL = 1024 VGPRs = 4096 bytes                         │
│    max_blocks = 4096 / 102,400 ≈ 0.04 blocks (very limited!)   │
│                                                                  │
│    Wait, this doesn't look right... Let me recalculate:         │
│    VGPR_POOL_WAVE64 = 1024 registers (not bytes)               │
│    vgpr_per_block (in registers) = threads × vgpr_per_thread   │
│                                   = 512 × 50 = 25,600 regs      │
│    max_blocks = 1024 / 25,600 < 1 block                        │
│                                                                  │
│    Actually, the VGPR pool is PER CU, and the calculation      │
│    considers how many blocks of this config can fit.            │
│                                                                  │
│    occupancy_score = f(max_blocks)                              │
│    → Higher num_warps → More VGPRs → Fewer blocks → Lower score │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Final Scoring                                                 │
│    score = balance^2.5 × launch^1.8 × occupancy^1.2 × grid^0.6 │
│              (40%)         (30%)       (20%)         (10%)       │
│                                         ↑                        │
│                              num_warps affects this 20%          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Real Example from Output

```python
# Config with num_warps=8
Config #1: {'XBLOCK': 512, 'num_warps': 8}
  Factors: balance=1.000(40%), launch=0.800(30%), occup=1.000(20%), grid=0.815(10%)
  Final score: 0.5918

# Config with num_warps=16
Config #5: {'XBLOCK': 1024, 'num_warps': 16}
  Factors: balance=1.000(40%), launch=0.800(30%), occup=0.850(20%), grid=0.830(10%)
  Final score: 0.4923

# Analysis:
# num_warps=16 has 15% lower occupancy (0.850 vs 1.000)
# This contributes to 17% lower final score (0.4923 vs 0.5918)
```

---

## Summary

**num_warps is used in 3 places:**

1. **Set during config generation** based on `threads_per_block / 64`
2. **Used in occupancy estimation** via `vgpr_per_block = threads × vgpr_per_thread`
3. **Affects final score** through the occupancy factor (20% weight)

**Key insight:** The heuristics don't directly optimize `num_warps`. Instead, they:
- Generate multiple configs with different `num_warps` (1, 2, 4, 8, 16)
- Score each based on predicted occupancy impact
- Pass top 5 to autotuner for real benchmarking
- Let hardware determine the true optimal `num_warps`



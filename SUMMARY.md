# Complete Summary: Environment Toggle + num_warps Explanation

## 1. Environment Variable Toggle ✅

### Implementation
Added `TORCHINDUCTOR_POINTWISE_HEURISTICS` environment variable to toggle between new heuristics and original behavior.

**File Modified:** `torch/_inductor/runtime/triton_heuristics.py`

```python
# Line ~40
POINTWISE_HEURISTICS_ENABLED = os.environ.get("TORCHINDUCTOR_POINTWISE_HEURISTICS", "1") == "1"

# Line ~2850
use_heuristics = torch.version.hip and POINTWISE_HEURISTICS_AVAILABLE and POINTWISE_HEURISTICS_ENABLED
```

### Usage

```bash
# Use NEW heuristics (default) - benchmarks 5 configs
export TORCHINDUCTOR_POINTWISE_HEURISTICS=1
python your_script.py

# Use ORIGINAL behavior - single default config, no heuristics
export TORCHINDUCTOR_POINTWISE_HEURISTICS=0
python your_script.py
```

### Performance Comparison

| Mode | Env Var | Compilation Time | Runtime Performance | Use Case |
|------|---------|------------------|---------------------|----------|
| **Original** | `=0` | Fast (~8s) | Good | Quick testing, baseline comparison |
| **New Heuristics** | `=1` (default) | Slower (~20s) | Better | Production, optimal performance |

**Test Script:** `/root/test_heuristics_toggle.py`

---

## 2. How num_warps Works in Heuristics

### Overview
`num_warps` is set based on block size and affects GPU occupancy scoring.

### Three Key Locations

#### Location 1: Config Generation (Setting)
```python
# File: triton_heuristics_pointwise.py
# Line ~443
configs.append({
    'XBLOCK': xblock,
    'num_warps': max(1, min(xblock // 64, 16))  # Set here
})
```

**Formula:** `num_warps = threads_per_block / 64` (clamped to 1-16)

**Examples:**
- XBLOCK=64 → num_warps=1
- XBLOCK=256 → num_warps=4
- XBLOCK=512 → num_warps=8
- XBLOCK=1024 → num_warps=16

#### Location 2: Occupancy Estimation (Main Impact)
```python
# File: triton_heuristics_pointwise.py
# Line ~286-315
def estimate_occupancy_impact(config, problem_metadata):
    threads_per_block = prod(block_dims)  # = num_warps × 64
    vgpr_per_block = threads_per_block × vgpr_per_thread
    
    max_blocks = VGPR_POOL / vgpr_per_block
    
    # Score based on max_blocks
    if max_blocks >= 12: return 1.0
    elif max_blocks >= 8: return 0.95
    elif max_blocks >= 4: return 0.85
    else: return 0.70
```

**Key Relationship:**
```
Higher num_warps → More threads → More VGPR usage → Fewer blocks → Lower occupancy score
```

#### Location 3: Final Scoring
```python
# File: triton_heuristics_pointwise.py
# Line ~360-400
score = (
    balance ** 2.5 *      # 40% weight
    launch ** 1.8 *       # 30% weight
    occupancy ** 1.2 *    # 20% weight ← num_warps affects this
    granularity ** 0.6    # 10% weight
)
```

### Real Example

From ResNet152 (16M element kernel):

```
Config #1: {'XBLOCK': 512, 'num_warps': 8}
  → occupancy=1.000 → Final score: 0.5918 ✅ BEST

Config #5: {'XBLOCK': 1024, 'num_warps': 16}
  → occupancy=0.850 → Final score: 0.4923 ❌ 17% lower
```

The higher `num_warps=16` causes 15% occupancy penalty, resulting in 17% lower final score.

### Trade-offs

| num_warps | VGPR Usage | Occupancy | Latency Hiding | Launch Overhead | Best For |
|-----------|------------|-----------|----------------|-----------------|----------|
| 1-2 | ✅ Low | ✅ High | ❌ Poor | ❌ High | Very small kernels |
| 4-8 | ✅ Balanced | ✅ Good | ✅ Good | ✅ Moderate | **Most kernels** |
| 16 | ❌ High | ❌ Lower | ✅ Best | ✅ Low | Large, compute-heavy |

---

## 3. How Heuristics Work Together

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Generate Configs                                 │
│   For each block size [16,32,64,128,256,512,1024]:     │
│     num_warps = block_size / 64                         │
│   → Generates 7 configs per problem                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Score Each Config                                │
│   balance(40%) × launch(30%) × occupancy(20%) × grid(10%)│
│                                    ↑                     │
│                          num_warps affects this          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: Prune to Top 5                                   │
│   Keep 5 highest scoring configs                        │
│   (Includes various num_warps: 1, 2, 4, 8, 16)         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 4: Benchmark (Triton Autotuner)                    │
│   Compile all 5 configs                                 │
│   Run each multiple times                               │
│   Select fastest based on REAL performance              │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Documentation Files

1. **`/root/test_heuristics_toggle.py`** - Test script for env var toggle
2. **`/root/NUM_WARPS_IN_HEURISTICS.md`** - Detailed num_warps explanation
3. **`/root/NUM_WARPS_CODE_SNIPPETS.md`** - Code snippets showing usage
4. **`/root/IMPROVED_CONFIG_GENERATION.md`** - All recent improvements
5. **`/root/SUMMARY.md`** - This file

---

## 5. Quick Reference

### Toggle Heuristics
```bash
# Disable (use original)
TORCHINDUCTOR_POINTWISE_HEURISTICS=0 python script.py

# Enable (use new)
TORCHINDUCTOR_POINTWISE_HEURISTICS=1 python script.py
```

### Test Toggle
```bash
python /root/test_heuristics_toggle.py
```

### View num_warps Impact
```bash
# Run with logging
rm -rf /tmp/torchinductor_root/
python your_script.py 2>&1 | grep "num_warps"

# Check log file
cat /tmp/pointwise_heuristics_calls.log | grep "num_warps"
```

---

## 6. Key Insights

1. **num_warps is automatically calculated** - Not manually tuned, derived from block size
2. **Affects occupancy via VGPR usage** - More warps → more VGPRs → lower occupancy
3. **Has 20% weight in final score** - Significant but not dominant
4. **Optimal value varies by problem** - That's why we benchmark 5 configs
5. **Environment variable allows easy A/B testing** - Compare new vs original anytime


# Detailed Factor Analysis in Validation Summary

## Overview

The validation summary now includes a **detailed factor-by-factor breakdown** and **root cause analysis**, showing exactly why heuristics predictions fail.

## New Output Format

```
================================================================================
[HEURISTICS VALIDATION] Predicted vs Actual Performance
================================================================================
Problem: (32768,), Total elements: 32,768
Generated: 15 configs, Benchmarked: 5 configs

📊 PREDICTED Best Config (rank #1, score=0.8378):
  Config: {'XBLOCK': 64, 'num_warps': 1}
  Actual time: 0.006800ms
  Factors: balance=1.000(40%), launch=0.950(30%), occup=1.000(20%), grid=0.868(10%)
  Grid: 512 blocks, 64 threads/block

🏆 ACTUAL Best Config:
  Config: {'XBLOCK': 1024, 'num_warps': 4}
  Actual time: 0.006480ms (fastest)
  Predicted score: 0.6703 (rank #13)
  Factors: balance=1.000(40%), launch=1.000(30%), occup=0.850(20%), grid=0.711(10%)
  Grid: 32 blocks, 1024 threads/block

🔍 ANALYSIS:
  ⚠️  Heuristics ranked actual best as #13 (off by 12 positions)
  Score gap: 0.1675 (20.0% difference)
  Real speedup: 1.049x (actual best vs predicted best)

  📋 Factor Comparison (Predicted #1 vs Actual Best #13):
       Balance   : Predicted=1.000 vs Actual=1.000 (≈ same)
    🟢 Launch    : Predicted=0.950 vs Actual=1.000 (Δ=-0.050, weight=30%)
    🔴 Occupancy : Predicted=1.000 vs Actual=0.850 (Δ=+0.150, weight=20%)
    🔴 Grid      : Predicted=0.868 vs Actual=0.711 (Δ=+0.158, weight=10%)

  💡 Root Cause Analysis:
    • Occupancy scored HIGHER for predicted config (+0.150)
      But actual best performed 1.049x better despite lower Occupancy
      → Suggests Occupancy weight (20%) may be too high

📈 ACCURACY: Predicted config is 1.05x vs actual best
  ✅ EXCELLENT: Within 5% of optimal
================================================================================
```

## Breaking Down the Analysis

### 1. Config Details (NEW)

**Predicted Best:**
```
Config: {'XBLOCK': 64, 'num_warps': 1}
Factors: balance=1.000(40%), launch=0.950(30%), occup=1.000(20%), grid=0.868(10%)
Grid: 512 blocks, 64 threads/block
```

**Actual Best:**
```
Config: {'XBLOCK': 1024, 'num_warps': 4}
Factors: balance=1.000(40%), launch=1.000(30%), occup=0.850(20%), grid=0.711(10%)
Grid: 32 blocks, 1024 threads/block
```

Shows all 4 scoring factors with their weights for both configs.

### 2. Factor Comparison (NEW)

```
📋 Factor Comparison (Predicted #1 vs Actual Best #13):
     Balance   : Predicted=1.000 vs Actual=1.000 (≈ same)
  🟢 Launch    : Predicted=0.950 vs Actual=1.000 (Δ=-0.050, weight=30%)
  🔴 Occupancy : Predicted=1.000 vs Actual=0.850 (Δ=+0.150, weight=20%)
  🔴 Grid      : Predicted=0.868 vs Actual=0.711 (Δ=+0.158, weight=10%)
```

**Symbols**:
- `🔴` Red: Predicted config scored **higher** on this factor
- `🟢` Green: Actual best scored **higher** on this factor
- No symbol: Essentially the same (<1% difference)

**Reading the Comparison**:
- `Balance`: Both 1.000 → Not discriminating
- `Launch`: Actual best 5% higher → Helped actual best
- `Occupancy`: Predicted 15% higher → **Hurt actual best (-0.150 × 20% = -3% score)**
- `Grid`: Predicted 18% higher → **Hurt actual best (-0.158 × 10% = -1.6% score)**

### 3. Root Cause Analysis (NEW)

```
💡 Root Cause Analysis:
  • Occupancy scored HIGHER for predicted config (+0.150)
    But actual best performed 1.049x better despite lower Occupancy
    → Suggests Occupancy weight (20%) may be too high
```

Automatically identifies:
1. **Which factor** caused the misprediction
2. **Direction** of the bias (too high/too low weight)
3. **Magnitude** of the performance mismatch
4. **Actionable recommendation** for weight tuning

## Use Cases

### Case 1: Overweighted Factor

**Symptom**:
```
🔴 Occupancy : Predicted=1.000 vs Actual=0.850 (Δ=+0.150, weight=20%)
→ Suggests Occupancy weight (20%) may be too high
```

**Diagnosis**: Factor is penalizing good configs too harshly

**Action**: Reduce weight
```python
# Before
occupancy ** 1.2  # 20% weight

# After
occupancy ** 0.9  # ~12% weight
```

### Case 2: Underweighted Factor

**Symptom**:
```
🟢 Launch : Predicted=0.700 vs Actual=0.950 (Δ=-0.250, weight=30%)
→ Suggests Launch weight (30%) may be too low
```

**Diagnosis**: Good factor not getting enough credit

**Action**: Increase weight
```python
# Before
launch ** 1.8  # 30% weight

# After
launch ** 2.2  # ~38% weight
```

### Case 3: Non-Discriminating Factors

**Symptom**:
```
Balance : Predicted=1.000 vs Actual=1.000 (≈ same)
Grid    : Predicted=0.850 vs Actual=0.853 (≈ same)
```

**Diagnosis**: Factors aren't helping distinguish configs

**Action**: 
- Add more discriminating factors
- Or adjust problem-specific logic (e.g., balance only matters for certain shapes)

### Case 4: Contradictory Signals

**Symptom**:
```
🟢 Launch    : Predicted=0.900 vs Actual=1.000 (Δ=-0.100, weight=30%)
🔴 Occupancy : Predicted=1.000 vs Actual=0.800 (Δ=+0.200, weight=20%)
```

**Diagnosis**: Factors pulling in opposite directions

**Action**:
- One factor may be wrong for this problem type
- Consider interaction effects (e.g., high occupancy more important when launch is low)

## Example: Tuning Based on Analysis

### Initial Run
```
📋 Factor Comparison:
  🔴 Occupancy : Predicted=1.000 vs Actual=0.850 (Δ=+0.150, weight=20%)
  🔴 Grid      : Predicted=0.868 vs Actual=0.711 (Δ=+0.158, weight=10%)

💡 Occupancy weight (20%) may be too high
📈 ACCURACY: 1.05x (EXCELLENT)
```

**Problem**: Configs with large threads/block (1024) are penalized too much for occupancy.

### Hypothesis
For large problem sizes (>16K elements), occupancy matters less because there's enough parallelism regardless.

### Change Weights
```python
# /root/pytorch/torch/_inductor/codegen/triton_heuristics_pointwise.py

# Before
def estimate_occupancy_impact(config, problem_metadata):
    # ... existing logic ...
    occupancy_score = threads_per_block / 1024.0  # Linear penalty

# After (adaptive)
def estimate_occupancy_impact(config, problem_metadata):
    # ... existing logic ...
    occupancy_score = threads_per_block / 1024.0
    
    # Relax occupancy penalty for large problems
    total_elements = problem_metadata['total_elements']
    if total_elements > 16384:  # >16K elements
        # Less harsh penalty for large threads/block
        occupancy_score = max(0.85, occupancy_score)  # Floor at 0.85
    
    return occupancy_score
```

Or adjust weight:
```python
# In score_config(), reduce occupancy exponent
score = (
    balance ** 2.5,      # 40%
    launch ** 1.8,       # 30%
    occupancy ** 0.9,    # 12% (was 1.2 = 20%)
    granularity ** 0.8   # 13% (was 0.6 = 10%)
)
```

### Re-run Benchmark
```
📋 Factor Comparison:
  🔴 Occupancy : Predicted=1.000 vs Actual=0.900 (Δ=+0.100, weight=12%)
       Grid      : Predicted=0.868 vs Actual=0.711 (≈ less impact)

💡 All factors reasonably balanced
📈 ACCURACY: 1.02x (EXCELLENT)
Rank: #2 (was #13)
```

**Result**: Actual best moved from rank #13 → #2! Much better!

## Statistics to Track

After many benchmarks, aggregate:

### 1. Factor Bias Analysis
```python
# Which factors consistently wrong?
occupancy_overweight_cases = 0
launch_underweight_cases = 0

for case in validation_results:
    if case['root_cause'] == 'Occupancy weight too high':
        occupancy_overweight_cases += 1
    elif case['root_cause'] == 'Launch weight too low':
        launch_underweight_cases += 1

# If occupancy_overweight_cases > 50% of failures:
# → Strong signal to reduce occupancy weight
```

### 2. Factor Correlation with Accuracy
```python
# Do certain factor patterns predict failure?
import numpy as np

# Collect data
rank_errors = []
occupancy_diffs = []

for case in validation_results:
    rank_errors.append(case['actual_rank'] - 1)
    occupancy_diffs.append(
        case['predicted_factors']['occupancy'] - 
        case['actual_factors']['occupancy']
    )

# Correlation
corr = np.corrcoef(rank_errors, occupancy_diffs)[0, 1]

# If corr > 0.7: Occupancy differences strongly predict rank error
# → Need to fix occupancy scoring or weight
```

### 3. Weight Sensitivity
```python
# How much does each factor contribute to mispredictions?
factor_impacts = {
    'balance': 0.0,
    'launch': 0.0,
    'occupancy': 0.0,
    'grid': 0.0
}

for case in validation_results:
    for factor, diff in case['factor_diffs'].items():
        weight = case['factor_weights'][factor]
        impact = abs(diff) * weight
        factor_impacts[factor] += impact

# Normalize
total = sum(factor_impacts.values())
for factor in factor_impacts:
    factor_impacts[factor] /= total

# Result: {'occupancy': 0.45, 'grid': 0.30, 'launch': 0.15, 'balance': 0.10}
# → Occupancy contributes 45% of total scoring variance
# → Consider rebalancing
```

## Files Modified

- `/root/pytorch/torch/_inductor/runtime/triton_heuristics.py`
  - Added detailed factor retrieval for both configs
  - Added factor-by-factor comparison
  - Added root cause analysis with weight recommendations
  - Added visual indicators (🔴/🟢) for significant differences

## Summary

✅ **Shows detailed factors** for predicted and actual best configs
✅ **Highlights differences** with visual indicators
✅ **Identifies root cause** (which factor, which direction)
✅ **Provides recommendations** for weight tuning
✅ **Makes heuristic tuning data-driven** instead of guesswork

Now you can see **exactly** which factors are causing mispredictions and how to fix them! 🎯


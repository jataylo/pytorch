# Enhanced Validation Summary

## New Feature: Score Analysis for Actual Best Config

The validation summary now shows the predicted score for the actual best config, helping identify **where and why** heuristics predictions fail.

## New Output Format

```
================================================================================
[HEURISTICS VALIDATION] Predicted vs Actual Performance
================================================================================
Problem: (32768,), Total elements: 32,768
Generated: 15 configs, Benchmarked: 5 configs

📊 PREDICTED Best Config (rank #1, score=0.8378):
  {'XBLOCK': 64, 'num_warps': 1}
  Actual time: 0.006640ms

🏆 ACTUAL Best Config:
  {'XBLOCK': 512, 'num_warps': 2}
  Actual time: 0.006480ms (fastest)
  Predicted score: 0.8218 (rank #5)

🔍 ANALYSIS:
  ⚠️  Heuristics ranked actual best as #5 (off by 4 positions)
  Score gap: 0.0160 (1.9% difference)
  Real speedup: 1.025x (actual best vs predicted best)
  Heuristic ranking: Predicted #1 > #5 (score: 0.8378 > 0.8218)

📈 ACCURACY: Predicted config is 1.02x vs actual best
  ✅ EXCELLENT: Within 5% of optimal
================================================================================
```

## What Each Section Shows

### 📊 PREDICTED Best Config
- **What heuristics predicted** would be fastest
- Includes its predicted score and actual measured time
- Shows rank #1 position

### 🏆 ACTUAL Best Config
- **What actually performed best** in benchmarking
- Shows fastest measured time
- **NEW**: Shows what score heuristics gave it
- **NEW**: Shows what rank heuristics gave it

### 🔍 ANALYSIS
Detailed breakdown of why prediction failed:

#### Case 1: Config Was Generated But Ranked Wrong
```
⚠️  Heuristics ranked actual best as #5 (off by 4 positions)
Score gap: 0.0160 (1.9% difference)
Real speedup: 1.025x (actual best vs predicted best)
Heuristic ranking: Predicted #1 > #5 (score: 0.8378 > 0.8218)
```

**Insights**:
- Actual best **was** in generated configs ✅
- But heuristics **underscored** it (0.8218 vs 0.8378 for predicted best)
- Only 1.9% score difference led to 4-position rank drop
- **Action**: May need to tune weights (launch_overhead, grid_granularity, etc.)

#### Case 2: Config Was Not Generated
```
❌ Predicted score: N/A (config was NOT in generated configs!)

🔍 ANALYSIS:
❌ CRITICAL: Heuristics didn't generate the actual best config!
This indicates a gap in config generation logic.
```

**Insights**:
- Actual best **wasn't even generated** ❌
- **Action**: Expand config generation (try more block sizes, num_warps, etc.)

#### Case 3: Perfect Prediction
```
✅ Heuristics correctly identified the best config!
```

**Insights**:
- Predicted best == Actual best ✅
- Heuristics working perfectly for this case

### 📈 ACCURACY
Overall performance rating:
- ✅ **EXCELLENT**: <5% slowdown
- ✓ **GOOD**: <15% slowdown
- ⚠ **ACCEPTABLE**: <30% slowdown
- ❌ **POOR**: >30% slowdown

## Use Cases

### 1. Identifying Weight Tuning Issues

If you see many cases like:
```
Score gap: 0.0160 (1.9% difference)
Real speedup: 1.025x
```

**Diagnosis**: Score differences are small but lead to wrong choices
**Action**: Fine-tune factor weights (balance, launch_overhead, occupancy, grid_granularity)

### 2. Identifying Config Generation Gaps

If you see:
```
❌ CRITICAL: Heuristics didn't generate the actual best config!
```

**Diagnosis**: Missing configs in generation logic
**Action**: 
- Add more block sizes to `block_sizes_1d/2d/3d`
- Try more `num_warps` values
- Relax validation rules if too restrictive

### 3. Identifying Factor Issues

If predicted best consistently has:
```
Grid: 512 blocks, 64 threads/block
```

But actual best has:
```
Grid: 64 blocks, 512 threads/block
```

**Diagnosis**: Grid granularity or launch overhead weights may be off
**Action**: Increase/decrease respective factor weights

### 4. Tracking Improvement Over Time

Run benchmarks before and after heuristic changes:

**Before** (many wrong predictions):
```
⚠️  Heuristics ranked actual best as #8 (off by 7 positions)
Score gap: 0.0450 (5.4% difference)
📈 ACCURACY: 1.15x (GOOD)
```

**After** (better predictions):
```
⚠️  Heuristics ranked actual best as #2 (off by 1 position)
Score gap: 0.0080 (1.0% difference)
📈 ACCURACY: 1.03x (EXCELLENT)
```

## Example Analysis Workflow

### Step 1: Run Benchmark
```bash
rm -rf ~/.triton/cache/ /tmp/torchinductor_root/ && \
TORCHINDUCTOR_POINTWISE_HEURISTICS=1 \
python /root/benchmark_pointwise.py --warmup 100 --iters 100
```

### Step 2: Analyze Results

Collect all validation summaries and look for patterns:

**Pattern 1**: Actual best consistently ranked #4-6
- **Diagnosis**: Scoring is close but slightly off
- **Fix**: Minor weight adjustments (±5-10%)

**Pattern 2**: Actual best often not generated
- **Diagnosis**: Config generation too conservative
- **Fix**: Expand block size ranges, try more num_warps

**Pattern 3**: Large score gaps (>0.05) but small perf differences
- **Diagnosis**: One factor is over-weighted
- **Fix**: Reduce dominant factor's weight

### Step 3: Tune Heuristics

Edit `/root/pytorch/torch/_inductor/codegen/triton_heuristics_pointwise.py`:

```python
# In score_config(), adjust weights:
score = (
    balance ** 2.5,  # ← Try 2.3 or 2.7
    launch ** 1.8,   # ← Try 1.6 or 2.0
    occupancy ** 1.2,  # ← Try 1.0 or 1.4
    granularity ** 0.6  # ← Try 0.5 or 0.7
)
```

### Step 4: Re-benchmark and Compare

Look for:
- ✅ Reduced rank differences
- ✅ Smaller score gaps
- ✅ Better accuracy ratings
- ✅ More "✅ Heuristics correctly identified" messages

## Statistics to Track

For a large benchmark run, calculate:

1. **Top-1 Accuracy**: How often predicted #1 == actual best
   - `count(predicted_best == actual_best) / total_cases`

2. **Top-3 Accuracy**: How often actual best is in top 3
   - `count(actual_best_rank <= 3) / total_cases`

3. **Average Rank Error**: How far off predictions are
   - `avg(abs(predicted_rank - actual_rank))`

4. **Average Score Gap**: Score difference magnitude
   - `avg(abs(predicted_best_score - actual_best_score))`

5. **Generation Coverage**: How often actual best was generated
   - `count(actual_best_in_generated) / total_cases`

## Files Modified

- `/root/pytorch/torch/_inductor/runtime/triton_heuristics.py`
  - Enhanced `_print_heuristics_validation_summary()` function
  - Added actual best config scoring analysis
  - Added rank difference and score gap calculations

## Testing

```bash
python /root/test_validation_debug.py
```

Look for enhanced output showing predicted scores for actual best configs.

## Summary

✅ **Added**: Predicted score for actual best config
✅ **Added**: Rank analysis (off by N positions)
✅ **Added**: Score gap analysis (absolute and percentage)
✅ **Added**: Real speedup comparison
✅ **Added**: Detection of non-generated configs
✅ **Added**: Detailed diagnostic information

Now you can **pinpoint exactly where and why** heuristics fail! 🎯


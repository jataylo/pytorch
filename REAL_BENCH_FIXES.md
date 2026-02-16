# REAL_BENCH Configuration Fixes

## Issues Reported

### Issue 1: `AttributeError: torch._inductor.config.heuristics_real_bench does not exist`

**Error**:
```
AttributeError: torch._inductor.config.heuristics_real_bench does not exist
```

**Root Cause**: The config option was added inside the `triton` class instead of at the module level, making it inaccessible as `config.heuristics_real_bench`.

**Fix**: Moved the config definition to the module level (around line 481) where other autotuning configs are defined.

### Issue 2: Confusing Output Messages

**Problem**: Output showed contradictory messages:
```
[POINTWISE HEURISTICS] Passing top 5 configs to autotuner for benchmarking...
```

But then later should say:
```
[POINTWISE HEURISTICS] REAL_BENCH mode: Benchmarking ALL 15 valid configs...
```

**Root Cause**: Old message was printing before new logic checked the config option.

**Fix**: Removed the misleading old message and clarified that "TOP CONFIGS" refers to predictions, not what will be benchmarked.

## Changes Made

### 1. Config Definition (Module Level)

**File**: `/root/pytorch/torch/_inductor/config.py`

**Location**: Lines 481-486 (module level, NOT inside a class)

```python
# pass ALL valid heuristic configs to autotuner (no pruning to top-N)
# enables full benchmarking to validate heuristic predictions vs reality
# when True: benchmarks ALL configs, compares predicted vs actual performance
# when False: only benchmarks top-5 predicted configs (faster but less validation)
heuristics_real_bench = (
    os.environ.get("TORCHINDUCTOR_HEURISTICS_REAL_BENCH", "1") == "1"
)
```

**Key Points**:
- At **module level** (like `max_autotune`, `max_autotune_pointwise`, etc.)
- Accessible as `config.heuristics_real_bench` ✅
- Defaults to `"1"` (enabled by default)
- Uses standard pattern: `os.environ.get(...) == "1"`

### 2. Removed Duplicate Definition

**File**: `/root/pytorch/torch/_inductor/config.py`

**Removed** the duplicate definition that was inside the `triton` class (was at line ~1499).

### 3. Improved Output Messages

**File**: `/root/pytorch/torch/_inductor/runtime/triton_heuristics.py`

**Before** (lines 2933-2935):
```python
msg = f"[POINTWISE HEURISTICS] Passing top {len(top_configs)} configs to autotuner for benchmarking..."
log.info(msg)
print(msg, flush=True)
```

**After** (line 2931):
```python
msg = "[POINTWISE HEURISTICS] TOP CONFIGS (by predicted score):"
log.info(msg)
print(msg, flush=True)
```

**Result**: Clearer messaging flow:
1. Shows TOP 5 **predicted** configs with scores
2. Then shows actual benchmarking mode and count:
   - `REAL_BENCH mode: Benchmarking ALL 15 valid configs...` (if enabled)
   - `Fast mode: Benchmarking top 5 configs only...` (if disabled)
3. After autotuning, shows validation summary

## Verification

### Test 1: Config Exists

```bash
python -c "from torch._inductor import config; print(config.heuristics_real_bench)"
# Output: True ✅
```

### Test 2: Environment Variable Works

```bash
TORCHINDUCTOR_HEURISTICS_REAL_BENCH=0 python -c "from torch._inductor import config; print(config.heuristics_real_bench)"
# Output: False ✅
```

### Test 3: Default Behavior

```bash
# Default (REAL_BENCH=1): Benchmarks ALL valid configs
python /root/benchmark_pointwise.py

# Fast mode (REAL_BENCH=0): Benchmarks only top 5
TORCHINDUCTOR_HEURISTICS_REAL_BENCH=0 python /root/benchmark_pointwise.py
```

## Expected Output (Fixed)

### With REAL_BENCH=1 (Default)

```
[POINTWISE HEURISTICS] ROCm detected - Using advanced heuristics for problem: (4096,)
[POINTWISE HEURISTICS] ======================================================================
[POINTWISE HEURISTICS] Problem: (4096,), Elements: 4,096
[POINTWISE HEURISTICS] Generated: 15 configs, Valid: 15, Top-N: 5
[POINTWISE HEURISTICS] ======================================================================
[POINTWISE HEURISTICS] ALL CONFIGS (sorted by predicted score):
[POINTWISE HEURISTICS]   # 1: [TOP-1]   score=0.8218 | XBLOCK=64, num_warps=1
[POINTWISE HEURISTICS]        bal=1.000(40%) lnch=1.000(30%) occ=1.000(20%) grid=0.721(10%) | 64blk 64thr
[... 14 more configs ...]
[POINTWISE HEURISTICS] ======================================================================
[POINTWISE HEURISTICS] TOP CONFIGS (by predicted score):
[POINTWISE HEURISTICS]   #1: {'XBLOCK': 64, 'num_warps': 1} (score=0.8218)
[POINTWISE HEURISTICS]       Factors: balance=1.000(40%), launch=1.000(30%), occup=1.000(20%), grid=0.721(10%)
[POINTWISE HEURISTICS]       Grid: 64 blocks, 64 threads/block
[... 4 more top configs ...]
[POINTWISE HEURISTICS] REAL_BENCH mode: Benchmarking ALL 15 valid configs...
[... autotuning happens ...]

================================================================================
[HEURISTICS VALIDATION] Predicted vs Actual Performance
================================================================================
Problem: (4096,)

Predicted Best Config (score=0.8218):
  {'XBLOCK': 64, 'num_warps': 1}
  Actual time: 0.000156ms

Actual Best Config:
  {'XBLOCK': 128, 'num_warps': 2}
  Actual time: 0.000142ms

Accuracy: Predicted config is 1.10x vs actual best
  ✓ GOOD: Within 15% of optimal
================================================================================
```

### With REAL_BENCH=0 (Fast Mode)

```
[POINTWISE HEURISTICS] ROCm detected - Using advanced heuristics for problem: (4096,)
[... same as above until ...]
[POINTWISE HEURISTICS] TOP CONFIGS (by predicted score):
[... shows top 5 ...]
[POINTWISE HEURISTICS] Fast mode: Benchmarking top 5 configs only...
[... autotuning happens ...]
(No validation summary)
```

## Files Modified

1. **`/root/pytorch/torch/_inductor/config.py`**
   - Added `heuristics_real_bench` at module level (lines 481-486)
   - Removed duplicate from `triton` class

2. **`/root/pytorch/torch/_inductor/runtime/triton_heuristics.py`**
   - Updated message at line 2931 to clarify "by predicted score"
   - Removed old misleading message

## Testing

Run the test script:
```bash
python /root/test_real_bench_fix.py
```

Or test manually:
```bash
# Full validation (default)
python /root/benchmark_pointwise.py --quick

# Fast mode
TORCHINDUCTOR_HEURISTICS_REAL_BENCH=0 python /root/benchmark_pointwise.py --quick
```

## Summary

✅ **Fixed**: Config attribute error
✅ **Fixed**: Confusing output messages
✅ **Verified**: Environment variable toggle works
✅ **Tested**: No linter errors

The implementation is now complete and functional!


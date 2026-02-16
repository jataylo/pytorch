# REAL_BENCH Implementation Summary

## Overview

Implemented comprehensive heuristics validation system that benchmarks ALL valid configurations and compares predicted scores against actual performance. This allows us to measure and improve the accuracy of the pointwise heuristics.

## Changes Made

### 1. Relaxed Dimension Validation

**File**: `/root/pytorch/torch/_inductor/codegen/triton_heuristics_pointwise.py`

**Problem**: Configs like `XBLOCK=8, YBLOCK=8` for problem `(32, 1024)` were being filtered even though they produce the same grid (512 blocks, 64 threads) as the top config `XBLOCK=4, YBLOCK=16`.

**Solution**: Modified validation to allow smaller block sizes when overall grid is reasonable:

```python
# Calculate grid first to check if it's reasonable
grid_size = PointwiseHeuristics.calculate_grid_size(problem_dims, block_dims)
num_blocks = PointwiseHeuristics.prod(grid_size)

for block_dim, problem_dim in zip(block_dims, problem_dims):
    if ndims == 3 or problem_dim <= 32:
        min_for_this_dim = 4
    elif num_blocks <= 1024:  # NEW: Allow small blocks if grid is reasonable
        min_for_this_dim = 4
    else:
        min_for_this_dim = 16  # Only enforce 16 for large dims + large grid
```

**Impact**: More configs pass validation, especially for problems with mixed small/large dimensions.

### 2. New Public API for Scoring

**File**: `/root/pytorch/torch/_inductor/codegen/triton_heuristics_pointwise.py`

Added `score_specific_config()` method for external benchmarking and validation:

```python
@staticmethod
def score_specific_config(config, problem_metadata):
    """
    Score a specific Triton config for a problem and return detailed breakdown.
    
    Returns:
        Dict with 'score', 'details', 'valid', 'config', and optional 'error'
    """
```

**Usage Example**:
```python
from torch._inductor.codegen.triton_heuristics_pointwise import PointwiseHeuristics

result = PointwiseHeuristics.score_specific_config(
    {'XBLOCK': 256, 'num_warps': 4},
    problem_metadata
)
print(f"Score: {result['score']}, Valid: {result['valid']}")
```

### 3. New Config Option

**File**: `/root/pytorch/torch/_inductor/config.py`

Added `heuristics_real_bench` option:

```python
# pass ALL valid heuristic configs to autotuner (no pruning to top-N)
# enables full benchmarking to validate heuristic predictions vs reality
heuristics_real_bench = os.environ.get("TORCHINDUCTOR_HEURISTICS_REAL_BENCH", "1") == "1"
```

**Environment Variable**: `TORCHINDUCTOR_HEURISTICS_REAL_BENCH`
- Default: `1` (enabled)
- `1`: Benchmark ALL valid configs, print validation
- `0`: Benchmark only top 5 (fast mode)

### 4. Validation Infrastructure

**File**: `/root/pytorch/torch/_inductor/runtime/triton_heuristics.py`

Added global storage and helper functions:

```python
# Storage for validation data
_HEURISTICS_VALIDATION_DATA = {}

def _store_heuristics_predictions(problem_key, problem_metadata, predicted_scores):
    """Store predicted scores for later comparison"""

def _store_actual_timing(problem_key, config, timing_ms):
    """Store actual benchmark timing for a config"""

def _print_heuristics_validation_summary(problem_key):
    """Print comparison of predicted vs actual performance"""
```

### 5. Hook into Config Selection

**File**: `/root/pytorch/torch/_inductor/runtime/triton_heuristics.py`

Modified `_apply_pointwise_heuristics()` to respect new config:

```python
from torch._inductor import config as inductor_config

if inductor_config.heuristics_real_bench:
    # Benchmark ALL valid configs
    configs_to_benchmark = all_valid_configs
    msg = f"REAL_BENCH mode: Benchmarking ALL {len(configs_to_benchmark)} valid configs..."
else:
    # Only benchmark top 5
    configs_to_benchmark = top_configs
    msg = f"Fast mode: Benchmarking top {len(configs_to_benchmark)} configs only..."
```

Also stores predictions before returning:

```python
if inductor_config.heuristics_real_bench and scored_all_configs:
    problem_key = str(tuple(size_hints))
    _store_heuristics_predictions(problem_key, problem_metadata, scored_all_configs)
```

### 6. Hook into Benchmarking

**File**: `/root/pytorch/torch/_inductor/runtime/triton_heuristics.py`

Modified `bench()` method to capture actual timings:

```python
# Measure timing
timing = benchmarker.benchmark(...)

# Store timing for validation
if (inductor_config.heuristics_real_bench and 
    self.heuristic_type == "pointwise" and 
    self.size_hints is not None):
    problem_key = str(tuple(self.size_hints))
    config_dict = {
        'XBLOCK': launcher.config.kwargs.get('XBLOCK'),
        'YBLOCK': launcher.config.kwargs.get('YBLOCK'),
        'ZBLOCK': launcher.config.kwargs.get('ZBLOCK'),
        'num_warps': launcher.config.num_warps,
    }
    config_dict = {k: v for k, v in config_dict.items() if v is not None}
    _store_actual_timing(problem_key, config_dict, timing)

return timing
```

### 7. Print Validation Summary

**File**: `/root/pytorch/torch/_inductor/runtime/triton_heuristics.py`

Modified `autotune_to_one_config()` to print summary after selecting best:

```python
if self.save_cache_hook:
    self.save_cache_hook(...)

# Print heuristics validation summary
if (inductor_config.heuristics_real_bench and 
    self.heuristic_type == "pointwise" and 
    self.size_hints is not None):
    problem_key = str(tuple(self.size_hints))
    _print_heuristics_validation_summary(problem_key)
```

## Output Format

### Phase 1: Heuristics Generation

```
[POINTWISE HEURISTICS] REAL_BENCH mode: Benchmarking ALL 59 valid configs...
[POINTWISE HEURISTICS] Problem: (32, 1024), Elements: 32,768
[POINTWISE HEURISTICS] Generated: 59 configs, Valid: 59, Top-N: 5
[POINTWISE HEURISTICS] ALL CONFIGS (sorted by predicted score):
[POINTWISE HEURISTICS]   # 1: [TOP-1]   score=0.8378 | XBLOCK=4, YBLOCK=16, num_warps=1
[POINTWISE HEURISTICS]        bal=1.000(40%) lnch=0.950(30%) occ=1.000(20%) grid=0.868(10%) | 512blk 64thr
[POINTWISE HEURISTICS]   # 2: [FILTERED] score=0.8378 | XBLOCK=8, YBLOCK=8, num_warps=1
[POINTWISE HEURISTICS]        bal=1.000(40%) lnch=0.950(30%) occ=1.000(20%) grid=0.868(10%) | 512blk 64thr
...
```

### Phase 2: Benchmarking

(Internal - no visible output during benchmarking)

### Phase 3: Validation Summary

```
================================================================================
[HEURISTICS VALIDATION] Predicted vs Actual Performance
================================================================================
Problem: (32, 1024)

Predicted Best Config (score=0.8378):
  {'XBLOCK': 4, 'YBLOCK': 16, 'num_warps': 1}
  Actual time: 0.000156ms

Actual Best Config:
  {'XBLOCK': 8, 'YBLOCK': 16, 'num_warps': 2}
  Actual time: 0.000142ms

Accuracy: Predicted config is 1.10x vs actual best
  ✓ GOOD: Within 15% of optimal
================================================================================
```

### Accuracy Ratings

- **EXCELLENT**: <5% slowdown (✅)
- **GOOD**: <15% slowdown (✓)
- **ACCEPTABLE**: <30% slowdown (⚠)
- **POOR**: >30% slowdown (❌)

## Example: Why Config #2 & #3 Were Filtered

For problem `(32, 1024)`:

**Config #1** (TOP-1): `XBLOCK=4, YBLOCK=16`
- `xnumel=32`: Uses min=4 ✅ (dim ≤ 32)
- `ynumel=1024`: Uses min=16 ✅ (YBLOCK=16 ≥ 16)
- Grid: 512 blocks, 64 threads
- **Score: 0.8378**

**Config #2** (FILTERED): `XBLOCK=8, YBLOCK=8`
- `xnumel=32`: Uses min=4 ✅ (dim ≤ 32)
- `ynumel=1024`: Uses min=4 now ✅ (num_blocks=512 ≤ 1024, NEW RULE!)
- Grid: 512 blocks, 64 threads
- **Score: 0.8378** (identical to #1)
- **Status**: Now PASSES validation (before: would be FILTERED)

**Config #3** (FILTERED): `XBLOCK=16, YBLOCK=4`
- `xnumel=32`: Uses min=4 ✅ (dim ≤ 32)
- `ynumel=1024`: Uses min=4 now ✅ (num_blocks=512 ≤ 1024, NEW RULE!)
- Grid: 512 blocks, 64 threads
- **Score: 0.8378** (identical to #1)
- **Status**: Now PASSES validation (before: would be FILTERED)

## Usage

### Default Mode (Full Validation)

```bash
# Default: benchmark all configs, print validation
python my_model.py
```

### Fast Mode

```bash
# Only benchmark top 5, skip validation
export TORCHINDUCTOR_HEURISTICS_REAL_BENCH=0
python my_model.py
```

### Run Test

```bash
# Test the feature
python /root/test_real_bench.py

# Or with fast mode
TORCHINDUCTOR_HEURISTICS_REAL_BENCH=0 python /root/test_real_bench.py
```

## Files Created/Modified

### Created
- `/root/HEURISTICS_REAL_BENCH.md` - Full documentation
- `/root/test_real_bench.py` - Test script
- `/root/REAL_BENCH_IMPLEMENTATION_SUMMARY.md` - This file

### Modified
- `/root/pytorch/torch/_inductor/config.py` - Added `heuristics_real_bench` option
- `/root/pytorch/torch/_inductor/codegen/triton_heuristics_pointwise.py`:
  - Relaxed dimension validation
  - Added `score_specific_config()` API
- `/root/pytorch/torch/_inductor/runtime/triton_heuristics.py`:
  - Added validation storage (`_HEURISTICS_VALIDATION_DATA`)
  - Added helper functions for storing/printing validation
  - Modified `_apply_pointwise_heuristics()` to store predictions
  - Modified `bench()` to capture timings
  - Modified `autotune_to_one_config()` to print summary

## Performance Impact

### REAL_BENCH=1 (Default)
- **First compilation**: Longer (benchmarks all configs)
- **Inference**: No impact (same as before)
- **Value**: Complete validation, identifies heuristic weaknesses

### REAL_BENCH=0 (Fast)
- **First compilation**: ~9x faster (5 configs vs 45)
- **Inference**: No impact
- **Value**: Production-ready, minimal overhead

## Next Steps

1. **Run validation on diverse workloads** to collect accuracy data
2. **Analyze validation results** to identify patterns in mispredictions
3. **Tune heuristic weights** based on validation feedback
4. **Add JSON export** of validation data for offline analysis
5. **Build ML model** trained on validation data for config selection

## See Also

- `/root/HEURISTICS_REAL_BENCH.md` - Full documentation
- `/root/POINTWISE_HEURISTICS_OPTIMIZATION.md` - Heuristics overview
- `/root/BENCHMARK_README.md` - Benchmark suite
- `/root/test_real_bench.py` - Test script


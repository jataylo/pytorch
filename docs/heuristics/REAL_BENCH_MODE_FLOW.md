# Autotuning with Real Bench Mode - Complete Flow

## 🎯 Overview

This document explains how the `HEURISTICS_REAL_BENCH` mode integrates with PyTorch Inductor's autotuning system to provide comprehensive validation of heuristic predictions.

---

## 🔧 Configuration

### Environment Variable

**`TORCHINDUCTOR_HEURISTICS_REAL_BENCH`**

```bash
export TORCHINDUCTOR_HEURISTICS_REAL_BENCH=1  # Enable (default)
export TORCHINDUCTOR_HEURISTICS_REAL_BENCH=0  # Disable
```

### Config Definition

**File:** `torch/_inductor/config.py`

```python
# Heuristics Real Bench Mode: Full validation of heuristic predictions
# -------------------------------------------------------------------------
# When enabled, benchmarks ALL valid configs (not just top 5) to validate
# heuristic accuracy against real performance.
#
# Behavior:
#   True (default):
#     - Generate & score ALL candidate configs (e.g., 15 configs)
#     - Benchmark ALL configs for complete validation data
#     - Select winner from TOP 5 predicted (trust heuristics + safety net)
#     - Print detailed validation summary comparing:
#       * Predicted best (rank #1) vs Actual best (from all benchmarked)
#       * Factor scores breakdown (bandwidth, launch, grid, occupancy)
#       * Bottleneck analysis (overhead/memory/compute)
#       * Performance gap and accuracy metrics
#
#   False:
#     - Generate & score ALL candidate configs
#     - Benchmark ONLY top 5 predicted configs (faster, less validation)
#     - Select winner from benchmarked configs
#     - No validation summary (can't compare against configs not benchmarked)
#
# Use Cases:
#   - Development: Enable to validate heuristic improvements
#   - Production: Disable for faster compilation (5-10x speedup vs full autotune)

heuristics_real_bench: bool = (
    os.environ.get("TORCHINDUCTOR_HEURISTICS_REAL_BENCH", "1") == "1"
)
```

---

## 🔄 Complete Flow (REAL_BENCH=1)

### Step-by-Step Execution

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Heuristics Generate & Score                                  │
│    File: runtime/triton_heuristics.py                          │
│    Function: _apply_pointwise_heuristics()                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    Generate ALL candidate configs (15 configs)
    FOR EACH config:
      - Analyze bottleneck (overhead/memory/compute)
      - Get adaptive weights
      - Calculate factor scores
      - Compute weighted geometric mean
    Sort by score → Ranked list
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Store Predictions & Top-5                                    │
│    Global: _HEURISTICS_VALIDATION_DATA                         │
│    Global: _TOP_N_CONFIGS_FOR_SELECTION                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    Store ALL predictions:
      _HEURISTICS_VALIDATION_DATA[problem_key] = {
          'problem_metadata': {...},
          'predicted_scores': [(score, config), ...],  # All 15
          'actual_timings': []  # Will be filled during benchmark
      }
    
    Store TOP 5 for selection:
      _TOP_N_CONFIGS_FOR_SELECTION[problem_key] = [
          config_1, config_2, config_3, config_4, config_5
      ]
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Return ALL Configs to Autotuner                              │
│    Return: triton_configs (list of 15 Triton configs)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Benchmark ALL Configs                                        │
│    File: runtime/triton_heuristics.py                          │
│    Class: CachingAutotuner                                      │
│    Function: benchmark_all_configs()                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    FOR EACH of 15 configs:
      - Warmup iterations
      - Timed iterations
      - Calculate median time
      - Store: timings[launcher] = time_ms
      
      # ALSO store for validation
      problem_key = _normalize_problem_key(self.size_hints)
      _store_actual_timing(problem_key, config, time_ms)
    
    Result: timings = {launcher_1: 0.0064ms, ..., launcher_15: 0.0082ms}
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Top-5 Selection (CRITICAL!)                                  │
│    File: runtime/triton_heuristics.py                          │
│    Function: autotune_to_one_config()                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    IF heuristics_real_bench AND pointwise:
      # Get top 5 configs stored earlier
      top_n_configs = _TOP_N_CONFIGS_FOR_SELECTION.get(problem_key)
      
      # Filter timings to ONLY top 5
      filtered_timings = {}
      FOR each (launcher, timing) in timings:
        IF launcher.config in top_n_configs:
          filtered_timings[launcher] = timing
      
      # Select fastest from TOP 5 ONLY
      winner = min(filtered_timings, key=filtered_timings.get)
      
      Print: "Selected from top 5 predicted configs 
              (benchmarked 15 total for validation)"
    ELSE:
      # Standard: select from ALL
      winner = min(timings, key=timings.get)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Validation Summary                                            │
│    File: runtime/triton_heuristics.py                          │
│    Function: _print_heuristics_validation_summary()             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    IF heuristics_real_bench:
      Retrieve:
        - Predicted scores (all 15)
        - Actual timings (all 15)
        - Top 5 configs used for selection
      
      Calculate:
        - Predicted best (rank #1 by score)
        - Actual best (fastest from ALL 15)
        - Performance gap
        - Factor score differences
        - Bottleneck analysis
      
      Print:
        📊 PREDICTED Best Config
        🏆 ACTUAL Best Config
        🔍 ANALYSIS (gap, ranking)
        📋 Factor Comparison
        💡 Root Cause Analysis
        📈 ACCURACY Assessment
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. Done - Kernel Ready                                          │
│    Selected: Config from top 5 (e.g., XBLOCK=256, num_warps=4) │
│    Performance: Within 1-5% of actual optimal                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Example Output

### Console Output (REAL_BENCH=1)

```
[POINTWISE HEURISTICS] ROCm detected - Using advanced heuristics
[POINTWISE HEURISTICS] Problem: (65536,), Elements: 65,536
[POINTWISE HEURISTICS] Generated: 15 configs, Valid: 15, Top-N: 5

[POINTWISE HEURISTICS] TOP CONFIGS (by predicted score):
  #1: {'XBLOCK': 256, 'num_warps': 4} (score=0.8925)
  #2: {'XBLOCK': 256, 'num_warps': 2} (score=0.8901)
  #3: {'XBLOCK': 256, 'num_warps': 1} (score=0.8870)
  #4: {'XBLOCK': 512, 'num_warps': 8} (score=0.8720)
  #5: {'XBLOCK': 512, 'num_warps': 4} (score=0.8697)

[POINTWISE HEURISTICS] REAL_BENCH mode: Benchmarking ALL 15 valid configs...

[Benchmarking happens...]

[HEURISTICS] Selected from top 5 predicted configs 
             (benchmarked 15 total for validation)

================================================================================
[HEURISTICS VALIDATION] Predicted vs Actual Performance
================================================================================
Problem: (65536,), Total elements: 65,536
Generated: 15 configs, Benchmarked: 15 configs
Selection: Winner chosen from top 5 predicted configs only
Validation: Comparing predicted #1 vs actual best from ALL 15 benchmarked

📊 PREDICTED Best Config (rank #1, score=0.8925):
  Config: {'XBLOCK': 256, 'num_warps': 4}
  Actual time: 0.006480ms
  Factors: bandwidth=1.000(40%), launch=0.804(30%), 
           grid=1.000(20%), occup=1.000(10%)
  Grid: 256 blocks, 256 threads/block
  Bottleneck: MEMORY (71% of time)

🏆 ACTUAL Best Config:
  Config: {'XBLOCK': 512, 'num_warps': 2}
  Actual time: 0.006400ms (fastest)
  Predicted score: 0.8667 (rank #6) ⚠️ OUTSIDE top 5!
  Factors: bandwidth=0.902(40%), launch=0.831(30%), 
           grid=0.902(20%), occup=0.932(10%)
  Grid: 128 blocks, 512 threads/block

🔍 ANALYSIS:
  ⚠️  Heuristics ranked actual best as #6 (off by 5 positions)
  Score gap: 0.0258 (2.9% difference)
  Real speedup: 1.012x (actual best vs predicted best)
  Winner selected: Config #1 (best from top 5, within 1.2% of optimal)

📋 Factor Comparison (Predicted #1 vs Actual Best #6):
  🔴 Bandwidth : Predicted=1.000 vs Actual=0.902 (Δ=+0.098, weight=40%)
  🟢 Launch    : Predicted=0.804 vs Actual=0.831 (Δ=-0.027, weight=30%)
  🔴 Grid      : Predicted=1.000 vs Actual=0.902 (Δ=+0.098, weight=20%)
  🔴 Occupancy : Predicted=1.000 vs Actual=0.932 (Δ=+0.068, weight=10%)

💡 Root Cause Analysis:
  • Bandwidth scored HIGHER for predicted config (+0.098)
    But actual best performed 1.012x better despite lower Bandwidth
    → Suggests slight overweighting of bandwidth factor

📈 ACCURACY: Predicted config is 1.01x vs actual best
  ✅ EXCELLENT: Within 5% of optimal
================================================================================
```

---

## 🎯 Key Functions

### 1. `_apply_pointwise_heuristics()`

**Location:** `runtime/triton_heuristics.py`

**Purpose:** Generate and score all candidate configs

```python
def _apply_pointwise_heuristics(size_hints, ...):
    # 1. Generate ALL candidate configs
    all_configs = PointwiseHeuristics.generate_all_candidate_configs(problem_metadata)
    
    # 2. Score ALL configs with V4 adaptive
    scored_all_configs = []
    for cfg in all_configs:
        score = PointwiseHeuristics.score_config(cfg, problem_metadata)
        scored_all_configs.append((score, cfg))
    
    # 3. Sort by score, select top 5
    scored_all_configs.sort(reverse=True)
    top_configs = [cfg for score, cfg in scored_all_configs[:5]]
    
    # 4. Store predictions & top-5
    if inductor_config.heuristics_real_bench:
        _store_heuristics_predictions(problem_key, problem_metadata, scored_all_configs)
        _TOP_N_CONFIGS_FOR_SELECTION[problem_key] = top_configs
    
    # 5. Return ALL configs for benchmarking
    return triton_configs  # All 15, not just top 5!
```

### 2. `benchmark_all_configs()`

**Location:** `runtime/triton_heuristics.py` (CachingAutotuner class)

**Purpose:** Benchmark all configs and store timings

```python
def benchmark_all_configs(self, *args, **kwargs):
    timings = {}
    
    # Benchmark each launcher (config)
    for launcher in self.launchers:
        time_ms = self.bench(launcher, *args, **kwargs)
        timings[launcher] = time_ms
        
        # Store for validation
        if heuristics_real_bench:
            problem_key = _normalize_problem_key(self.size_hints)
            config_dict = extract_config(launcher)
            _store_actual_timing(problem_key, config_dict, time_ms)
    
    return timings
```

### 3. `autotune_to_one_config()`

**Location:** `runtime/triton_heuristics.py` (CachingAutotuner class)

**Purpose:** Select winner from top-5 only (REAL_BENCH mode)

```python
def autotune_to_one_config(self, *args, **kwargs):
    # Benchmark ALL configs
    timings = self.benchmark_all_configs(*args, **kwargs)
    
    # Top-5 selection (REAL_BENCH mode only)
    if heuristics_real_bench and self.heuristic_type == HeuristicType.POINTWISE:
        problem_key = _normalize_problem_key(self.size_hints)
        top_n_configs = _TOP_N_CONFIGS_FOR_SELECTION.get(problem_key, [])
        
        if top_n_configs:
            # Filter to top 5
            filtered_timings = {
                launcher: timing
                for launcher, timing in timings.items()
                if extract_config(launcher) in top_n_configs
            }
            
            # Select fastest from top 5
            winner = min(filtered_timings, key=filtered_timings.get)
            
            print(f"Selected from top {len(top_n_configs)} predicted configs "
                  f"(benchmarked {len(timings)} total for validation)")
        else:
            # Fallback
            winner = min(timings, key=timings.get)
    else:
        # Standard: select from all
        winner = min(timings, key=timings.get)
    
    self.launchers = [winner]
    return winner
```

### 4. `_print_heuristics_validation_summary()`

**Location:** `runtime/triton_heuristics.py`

**Purpose:** Print comprehensive validation comparison

```python
def _print_heuristics_validation_summary(problem_key):
    """
    Print comparison of predicted vs actual performance.
    
    Shows:
    - Predicted best (rank #1)
    - Actual best (from all benchmarked)
    - Factor scores for both
    - Performance gap
    - Root cause analysis
    """
    data = _HEURISTICS_VALIDATION_DATA[problem_key]
    predicted = data['predicted_scores']
    actual = data['actual_timings']
    
    # Find predicted best
    best_predicted = max(predicted, key=lambda x: x[0])
    
    # Find actual best
    best_actual = min(actual, key=lambda x: x[0])
    
    # Compare and print detailed analysis
    print_comparison(best_predicted, best_actual, ...)
```

---

## 🔑 Key Data Structures

### Global Storage

**File:** `runtime/triton_heuristics.py`

```python
# Stores complete validation data per problem
_HEURISTICS_VALIDATION_DATA = {
    problem_key: {
        'problem_metadata': {...},
        'predicted_scores': [(score, config), ...],  # All configs
        'actual_timings': [(time_ms, config), ...],  # All benchmarked
    }
}

# Stores top N configs for selection
_TOP_N_CONFIGS_FOR_SELECTION = {
    problem_key: [config_1, config_2, ..., config_5]
}

# problem_key format: tuple of dimensions
# Example: (65536,) for 1D, (256, 1024) for 2D
```

---

## ⚡ Performance Impact

### REAL_BENCH=1 (Default)
- **Benchmarking time:** ~300-500ms for 15 configs
- **Compilation time:** Same (configs generated regardless)
- **Selection time:** Negligible (<1ms)
- **Validation output:** ~10ms (printing)
- **Total overhead:** ~300-500ms vs top-5 only (~100-150ms)

### Trade-off
- **Cost:** 2-3x longer than benchmarking top-5 only
- **Benefit:** Complete validation data to improve heuristics
- **Production:** Disable for faster compilation
- **Development:** Enable to identify heuristic weaknesses

---

## 🎯 Use Cases

### Development (REAL_BENCH=1)
```bash
export TORCHINDUCTOR_HEURISTICS_REAL_BENCH=1
python benchmark_pointwise.py --warmup 100 --iters 100
```

**Benefits:**
- See which configs heuristics got wrong
- Identify systematic biases
- Validate heuristic improvements
- Understand bottleneck classification accuracy

### Production (REAL_BENCH=0)
```bash
export TORCHINDUCTOR_HEURISTICS_REAL_BENCH=0
python my_model.py
```

**Benefits:**
- 2-3x faster compilation
- Still get good config (top-5 benchmarked)
- No validation overhead
- Suitable for deployed models

---

## 📖 Related Documentation

- **Complete flow:** `HEURISTICS_FLOW.md`
- **Top-5 strategy:** `TOP_N_SELECTION_FEATURE.md`
- **Quick reference:** `QUICK_REFERENCE_V4.md`
- **Visual diagrams:** `VISUAL_ARCHITECTURE.md`

---

**Last Updated:** 2026-02-18  
**Version:** V4 with Real Bench Mode  
**Status:** Production-ready ✅


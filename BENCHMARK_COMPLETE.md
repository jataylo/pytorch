# ✅ Pointwise Benchmark Suite - Complete

## Overview

A comprehensive benchmark suite to validate the new pointwise heuristics on real-world workloads.

## What's Included

### 1. Benchmark Script (`benchmark_pointwise.py`)

**Tests 5 operations across 13 shapes = 65 kernels total:**

| # | Operation | Description | Why It Matters |
|---|-----------|-------------|----------------|
| 1 | `add` | `z = x + y` | Basic 2-input pointwise |
| 2 | `mul` | `z = x * y` | Basic 2-input pointwise |
| 3 | `relu_sigmoid` | `z = sigmoid(relu(x))` | Chained activations |
| 4 | `gelu` | `z = gelu(x)` | Complex activation |
| 5 | **`fused`** | `z = gelu((x + y) * w + b)` | **Most important - tests fusion** |

**Test shapes:**
- 1D: 1K, 64K, 1M, 16M elements
- 2D: 256², 1024², 4096², wide, tall
- 3D: 32³, 64³, 128³, batched

### 2. Comparison Script (`compare_heuristics.sh`)

Runs benchmark twice:
- `TORCHINDUCTOR_POINTWISE_HEURISTICS=0` (original behavior)
- `TORCHINDUCTOR_POINTWISE_HEURISTICS=1` (new heuristics)

Then compares geomean speedups.

### 3. Visualization Script (`visualize_benchmark.py`)

Creates ASCII visualizations:
- Speedup by operation
- Speedup by problem size
- Top/bottom performers
- Overall assessment

### 4. Quick Test (`test_benchmark_quick.py`)

Fast validation that everything works before running full suite.

## Quick Start

```bash
# 1. Quick test (verify it works)
python /root/test_benchmark_quick.py

# 2. Run full benchmark with new heuristics
python /root/benchmark_pointwise.py

# 3. Visualize results
python /root/visualize_benchmark.py pointwise_benchmark_results.csv

# 4. A/B comparison (new vs original)
bash /root/compare_heuristics.sh

# 5. Compare the two results
python /root/visualize_benchmark.py /tmp/results_original.csv /tmp/results_new.csv
```

## Expected Results

### Target Performance (MI350/CDNA4)

| Metric | Target | Notes |
|--------|--------|-------|
| **Overall Geomean** | **2.0-2.5x** | Main success metric |
| Simple ops (add, mul) | 1.3-1.5x | Memory bound |
| Activations (relu, gelu) | 1.8-2.2x | More compute |
| **Fused** | **4.0-6.0x** | **Validates fusion works** |

### Sample Output

```
================================================================================
  SUMMARY
================================================================================

Geometric Mean Speedup by Operation:
--------------------------------------------------
  add                 :  1.456x
  mul                 :  1.423x
  relu_sigmoid        :  2.134x
  gelu                :  1.987x
  fused               :  5.234x  ← Most important!
--------------------------------------------------
  OVERALL             :  2.123x  ← Main metric
================================================================================

Speedup by Problem Size:
--------------------------------------------------
  small (<100K)       :  1.234x (20 kernels)
  medium (100K-1M)    :  1.823x (24 kernels)
  large (1M-10M)      :  2.234x (15 kernels)
  huge (>10M)         :  2.456x (6 kernels)
================================================================================
```

## Understanding the Results

### ✅ Good Result

```
OVERALL: 2.234x
  fused: 5.123x
```

**Interpretation:**
- Compile is 2.23x faster than eager overall
- Fusion is working (5.12x speedup means 4 kernels → 1 kernel)
- Heuristics are providing value ✅

### ⚠️ Needs Investigation

```
OVERALL: 1.123x
  fused: 1.456x
```

**Possible issues:**
- Fusion may not be happening
- Configs not optimal for hardware
- Launch overhead dominating
- Check logs for warnings

### ❌ Problem

```
OVERALL: 0.856x
  fused: 0.923x
```

**Action items:**
- Compile is slower than eager
- Check for errors in logs
- Verify heuristics are enabled
- May need to tune weights/factors

## Why the Fused Benchmark Matters Most

### Eager Mode (4 separate kernels):

```python
z = gelu((x + y) * w + b)
```

Executes as:
1. Kernel 1: `t1 = x + y` → write to memory
2. Kernel 2: `t2 = t1 * w` → read from memory, write to memory
3. Kernel 3: `t3 = t2 + b` → read from memory, write to memory
4. Kernel 4: `z = gelu(t3)` → read from memory, write to memory

**Total: 4 kernel launches + 6 memory round-trips**

### Compiled Mode (1 fused kernel):

```python
z = gelu((x + y) * w + b)
```

Executes as:
1. Kernel 1: `z = gelu((x + y) * w + b)` → all ops in registers

**Total: 1 kernel launch + 0 intermediate memory**

### Expected Speedup:

- **Memory savings:** 6 round-trips → 0 round-trips
- **Launch overhead:** 4 launches → 1 launch
- **Register reuse:** All intermediates stay in registers
- **Target speedup:** ~4-6x (theoretical limit ~4x from kernel reduction)

**If fused speedup < 3x:** Fusion may not be working correctly!

## Validating Heuristics Impact

To prove the new heuristics help:

```bash
# Run A/B comparison
bash /root/compare_heuristics.sh

# Compare results
python /root/visualize_benchmark.py \
    /tmp/results_original.csv \
    /tmp/results_new.csv
```

**Look for:**
- New heuristics should have **higher overall geomean**
- New heuristics should have **better large problem performance**
- Both should show good fusion (4-6x), but new may be faster

## Files Summary

| File | Purpose | When to Use |
|------|---------|-------------|
| `test_benchmark_quick.py` | Quick validation | Before running full suite |
| `benchmark_pointwise.py` | Main benchmark | Primary performance validation |
| `compare_heuristics.sh` | A/B comparison | Prove heuristics help |
| `visualize_benchmark.py` | Visualize results | Understand results |
| `BENCHMARK_README.md` | Detailed docs | Reference guide |
| `BENCHMARK_COMPLETE.md` | This file | Quick overview |

## Integration with Heuristics

The benchmark uses these environment variables:

| Variable | Values | Effect |
|----------|--------|--------|
| `TORCHINDUCTOR_POINTWISE_HEURISTICS` | `0` or `1` | Enable/disable new heuristics |
| `TORCH_LOGS` | `+graph_code` | Show generated kernels |
| `TORCHINDUCTOR_CACHE_DIR` | path | Custom cache location |

## Example Workflow

### 1. Initial Validation

```bash
# Test it works
python test_benchmark_quick.py

# Run full benchmark with new heuristics
python benchmark_pointwise.py --warmup 10 --iters 50
```

### 2. Check Results

```bash
# Visualize
python visualize_benchmark.py pointwise_benchmark_results.csv
```

**Expected output:**
```
📊 KEY METRICS:
  Overall Geomean:     2.123x
  Fused Geomean:       5.234x  ✅ GOOD

🎯 ASSESSMENT:
  ✅ EXCELLENT - Overall speedup >= 2.0x
  ✅ FUSION WORKING - Fused kernels >= 4.0x speedup
```

### 3. A/B Comparison

```bash
# Compare with original behavior
bash compare_heuristics.sh

# Visualize comparison
python visualize_benchmark.py /tmp/results_original.csv /tmp/results_new.csv
```

**Expected output:**
```
COMPARISON
  A: /tmp/results_original.csv  (1.987x overall)
  B: /tmp/results_new.csv        (2.123x overall)
  → B is +6.8% faster ✅
```

### 4. Share Results

```bash
# Upload CSV to GitHub/internal tools
cat pointwise_benchmark_results.csv

# Or share visualization
python visualize_benchmark.py pointwise_benchmark_results.csv > results.txt
```

## Success Criteria

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Overall geomean | > 1.0x | > 1.5x | > 2.0x |
| Fused speedup | > 2.0x | > 4.0x | > 5.0x |
| Large problems | > 1.5x | > 2.0x | > 2.5x |
| vs Original | > 1.0x | > 1.05x | > 1.10x |

**If all criteria met:** Heuristics validated ✅

## Troubleshooting

### Q: Speedups are low (<1.5x)

**A:** Check:
1. Is max-autotune enabled? (`mode='max-autotune'`)
2. Are heuristics enabled? (`TORCHINDUCTOR_POINTWISE_HEURISTICS=1`)
3. Is cache cleared? (`rm -rf /tmp/torchinductor_*`)
4. Enough warmup? (`--warmup 10`)

### Q: Fused speedup is low (<3x)

**A:** Check:
1. Are kernels actually fusing? (`TORCH_LOGS=+graph_code`)
2. Look for 1 `poi*` kernel (fused) vs 4 (unfused)
3. May need more aggressive fusion heuristics

### Q: Results vary between runs

**A:** Normal! Try:
1. Increase iterations (`--iters 100`)
2. Increase warmup (`--warmup 20`)
3. Run on idle system (no other GPU workloads)
4. Use geometric mean (less sensitive to outliers)

## Next Steps

1. ✅ Run `python test_benchmark_quick.py` - verify it works
2. ✅ Run `python benchmark_pointwise.py` - get baseline
3. ✅ Check overall geomean > 2.0x - validate performance
4. ✅ Check fused speedup > 4.0x - validate fusion
5. ✅ Run `bash compare_heuristics.sh` - prove heuristics help
6. ✅ Share results - show the team

**If all checks pass:** The heuristics are production-ready! 🎉


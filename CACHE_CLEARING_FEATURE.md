# Cache Clearing Per Shape Feature

## Overview

Added `--clear-per-shape` flag to the pointwise benchmark to clear the compilation cache after EACH shape/sub-problem. This ensures you see fresh heuristics output for every single test case.

## Usage

```bash
# Clear cache after each shape (most verbose, shows all heuristics)
python benchmark_pointwise.py --clear-per-shape

# Old behavior: cache persists, only see heuristics for first compilation of each operation
python benchmark_pointwise.py
```

## Why This Matters

### Problem
PyTorch's compilation cache stores kernels by shape. Once a shape is compiled:
- Subsequent runs reuse the cached kernel
- No heuristics output (already compiled!)
- You only see heuristics for the FIRST shape of each operation

### Example (Without --clear-per-shape)
```
Benchmark 1: Add
  Shape 1: (512,)        ← Shows heuristics ✅
  Shape 2: (4096,)       ← No heuristics (cached) ❌
  Shape 3: (65536,)      ← No heuristics (cached) ❌
  ...

Benchmark 2: Mul
  Shape 1: (512,)        ← Shows heuristics ✅
  Shape 2: (4096,)       ← No heuristics (cached) ❌
  ...
```

### Solution (With --clear-per-shape)
```
Benchmark 1: Add
  Shape 1: (512,)        ← Shows heuristics ✅
  [Cache cleared]
  Shape 2: (4096,)       ← Shows heuristics ✅
  [Cache cleared]
  Shape 3: (65536,)      ← Shows heuristics ✅
  [Cache cleared]
  ...
```

## Implementation

### 1. Added `clear_cache_per_shape` Parameter
```python
class PointwiseBenchmark:
    def __init__(self, ..., clear_cache_per_shape=False):
        self.clear_cache_per_shape = clear_cache_per_shape
```

### 2. Cache Clearing After Each Shape
Added to all 16 benchmark methods:
```python
for shape_name, shape in self.shapes:
    # ... benchmark the shape ...
    print(f"  {shape_name} | ... | Speedup: {speedup}x")
    
    # NEW: Clear cache after each shape
    if self.clear_cache_per_shape:
        self.clear_compilation_cache()
```

### 3. Command-Line Flag
```python
parser.add_argument('--clear-per-shape', action='store_true',
                    help='Clear cache after EACH shape')
```

## Performance Impact

**With --clear-per-shape:**
- ⚠️ SLOWER: Every shape requires full compilation
- ⚠️ DISK I/O: Clears `/tmp/torchinductor_root/` and `~/.triton/cache/`
- ✅ VERBOSE: See heuristics for ALL 38 shapes × 16 operations = 608 compilations!

**Without --clear-per-shape:**
- ✅ FASTER: Compilation cache reused
- ✅ Less disk I/O
- ❌ Only see heuristics for first occurrence of each shape

## Use Cases

### Development & Testing
```bash
# See heuristics for every single problem
python benchmark_pointwise.py --clear-per-shape
```

**Perfect for:**
- Validating heuristics across ALL shapes
- Debugging config selection for specific sizes
- Comparing predicted vs actual for every case

### Production Benchmarking
```bash
# Fast, realistic performance measurement
python benchmark_pointwise.py
```

**Perfect for:**
- Measuring actual runtime with warm cache
- Comparing eager vs compile speedup
- Performance regression testing

## Other Cache Clearing Options

### Option 1: Clear Every N Benchmarks
```bash
python benchmark_pointwise.py --show-all-heuristics --clear-cache-every 5
```
- Clears cache every 5 complete benchmarks
- Shows heuristics for multiple shapes, but not all

### Option 2: Clear Per Shape (NEW!)
```bash
python benchmark_pointwise.py --clear-per-shape
```
- Clears cache after EVERY shape
- Most verbose, see all heuristics

### Option 3: No Clearing (Default)
```bash
python benchmark_pointwise.py
```
- Cache persists across entire run
- Only see heuristics once per unique operation+shape combo

## Example Output

```
================================================================================
  POINTWISE KERNEL BENCHMARK SUITE
================================================================================
  Device: cuda
  Warmup iterations: 10
  Benchmark iterations: 50
  Pointwise heuristics: 1
  🔄 Cache clearing: After EACH shape (to show heuristics for every sub-problem)
  GPU: AMD Radeon Graphics
  ROCm version: 6.2.41133-dd7f95766
================================================================================

================================================================================
  Benchmark 1: Elementwise Add (z = x + y)
================================================================================
[POINTWISE HEURISTICS] ROCm detected - problem: (512,)
[POINTWISE HEURISTICS] Generated: 7 configs, Top-N: 5
  #1: XBLOCK=512, nw=8 → 0.9239 ✅
  #2: XBLOCK=512, nw=4 → 0.9183
  ...
  1D_tiny              | Eager:   0.0123ms | Compile:   0.0098ms | Speedup:  1.255x

[POINTWISE HEURISTICS] ROCm detected - problem: (4096,)  ← Fresh heuristics!
[POINTWISE HEURISTICS] Generated: 10 configs, Top-N: 5
  #1: XBLOCK=256, nw=4 → 0.9165 ✅
  ...
  1D_small             | Eager:   0.0145ms | Compile:   0.0112ms | Speedup:  1.295x

... and so on for all 608 shape×operation combinations!
```

## Technical Details

### What Gets Cleared
1. **Inductor Cache**: `/tmp/torchinductor_root/`
   - Compiled Triton kernels
   - Code generation artifacts

2. **Triton Cache**: `~/.triton/cache/`
   - PTX/assembly cache
   - Compiled GPU binaries

### Timing
- Cache clearing happens AFTER benchmarking each shape
- Does NOT affect benchmark timing (happens after measurement)
- Next shape gets fresh compilation

---

**Status: ✅ Implemented and tested**

**Usage:** `python benchmark_pointwise.py --clear-per-shape`


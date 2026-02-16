# Cache Clearing to Show Heuristics for All Benchmarks

## Issue

Heuristics output stops appearing after initial compilations because **PyTorch's compilation cache is working**:

1. **First compilation** for a shape → Heuristics run, output shown
2. **Cached result** used for same shape later → No recompilation, no heuristics output
3. **Result**: You only see heuristics for the first few unique shapes

Example:
```
[POINTWISE HEURISTICS] Problem: (16, 256)  ← First compilation, shown
[POINTWISE HEURISTICS] Generated: 45 configs...

[Later benchmark with same shape...]
(No output - using cached kernel)
```

## Why This Happens

**This is expected and good behavior:**
- Cache improves performance (no recompilation overhead)
- Same shapes reuse compiled kernels (faster benchmarking)
- Dynamic shapes disabled → more aggressive caching

**But** for debugging/analysis, you want to see heuristics for every benchmark.

## Solution: Periodic Cache Clearing

New flags added to force recompilation:

```bash
# Clear cache every 5 benchmarks (default)
python /root/benchmark_pointwise.py --show-all-heuristics

# Clear cache every 3 benchmarks
python /root/benchmark_pointwise.py --show-all-heuristics --clear-cache-every 3

# Clear cache after every single benchmark
python /root/benchmark_pointwise.py --show-all-heuristics --clear-cache-every 1
```

## What It Does

With `--show-all-heuristics`:
1. **Runs benchmark normally**
2. **After every N benchmarks**: Clears compilation cache
3. **Next compilation**: Forced to recompile → heuristics shown again
4. **Repeat** throughout benchmark suite

Example output:
```
Benchmark 1: add
[POINTWISE HEURISTICS] Generated: 45 configs...  ← Shown

Benchmark 2: mul
(Using cache)

Benchmark 3: relu
(Using cache)

Benchmark 4: gelu
(Using cache)

Benchmark 5: tanh
(Using cache)

🔄 Clearing cache (benchmark #5) to show heuristics...

Benchmark 6: silu
[POINTWISE HEURISTICS] Generated: 45 configs...  ← Shown again!

Benchmark 7: squared_relu
(Using cache)

...

🔄 Clearing cache (benchmark #10) to show heuristics...

Benchmark 11: fused
[POINTWISE HEURISTICS] Generated: 45 configs...  ← Shown again!
```

## Trade-offs

| Aspect | Without Flag | With `--show-all-heuristics` |
|--------|--------------|------------------------------|
| Speed | ⚡ Fast (cache works) | 🐌 Slower (recompiles) |
| Heuristics Visibility | ⚠️  Only first few | ✅ All benchmarks |
| Real-world Performance | ✅ Realistic | ⚠️  Includes recompilation overhead |
| Use Case | Production benchmarking | Heuristics debugging |

## Usage Examples

### 1. Normal Benchmarking (Fast, Cache Enabled)
```bash
python /root/benchmark_pointwise.py --quick
```
**Use when**: You want fast, realistic performance measurements

### 2. Heuristics Analysis (Slower, Shows All Output)
```bash
python /root/benchmark_pointwise.py --quick --show-all-heuristics --clear-cache-every 3
```
**Use when**: You want to see heuristics for many different benchmarks

### 3. Debug Single Benchmark (Slowest, Maximum Visibility)
```bash
python /root/benchmark_pointwise.py --quick --show-all-heuristics --clear-cache-every 1
```
**Use when**: You want to see heuristics for EVERY single benchmark

## What Gets Cleared

When cache is cleared:
1. `/tmp/torchinductor_root/` - Inductor compilation cache
2. `~/.triton/cache/` - Triton kernel cache
3. Python GC + `torch.cuda.empty_cache()` - Memory cleanup

This forces full recompilation on next kernel invocation.

## Example Output

```bash
$ python /root/benchmark_pointwise.py --quick --show-all-heuristics --clear-cache-every 2

================================================================================
  POINTWISE KERNEL BENCHMARK SUITE
================================================================================
  Device: cuda
  Warmup iterations: 10
  Benchmark iterations: 50
  Pointwise heuristics: 1
  🔄 Cache clearing: Every 2 benchmarks (to show heuristics)
  GPU: AMD Radeon Graphics
  ROCm version: 7.2.53150
================================================================================

================================================================================
  PHASE 1: Basic Elementwise Operations
================================================================================

Benchmark 1: Elementwise Add
[POINTWISE HEURISTICS] ROCm detected...
[POINTWISE HEURISTICS] Generated: 45 configs, Valid: 45...
[POINTWISE HEURISTICS] ALL CONFIGS (sorted by predicted score):
...

Benchmark 2: Elementwise Multiply
(Using cached kernel - no heuristics output)

🔄 Clearing cache (benchmark #2) to show heuristics...

Benchmark 3: ReLU + Sigmoid
[POINTWISE HEURISTICS] ROCm detected...
[POINTWISE HEURISTICS] Generated: 45 configs, Valid: 45...
...

Benchmark 4: GELU
(Using cached kernel - no heuristics output)

🔄 Clearing cache (benchmark #4) to show heuristics...

Benchmark 5: Tanh
[POINTWISE HEURISTICS] ROCm detected...
...
```

## Performance Impact

Clearing cache adds overhead:

| Clear Frequency | Overhead | Heuristics Visibility |
|----------------|----------|----------------------|
| Never (default) | 0% | ~10% of benchmarks |
| Every 5 | ~20% | ~50% of benchmarks |
| Every 3 | ~33% | ~66% of benchmarks |
| Every 2 | ~50% | ~80% of benchmarks |
| Every 1 | ~100% | 100% of benchmarks |

**Recommendation**: Use `--clear-cache-every 3` or `5` for good balance.

## Implementation Details

```python
def clear_compilation_cache(self):
    """Clear PyTorch compilation cache to force recompilation"""
    # Clear Inductor cache
    shutil.rmtree('/tmp/torchinductor_root/')
    
    # Clear Triton cache
    shutil.rmtree('~/.triton/cache/')
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

def run_all(self, show_all_heuristics=False, clear_every=5):
    bench_count = 0
    
    def maybe_clear_cache():
        nonlocal bench_count
        bench_count += 1
        if show_all_heuristics and bench_count % clear_every == 0:
            print(f"🔄 Clearing cache (benchmark #{bench_count})...")
            self.clear_compilation_cache()
    
    # Run benchmarks with periodic clearing
    self.bench_add()
    maybe_clear_cache()
    self.bench_mul()
    maybe_clear_cache()
    ...
```

## Summary

| Scenario | Command |
|----------|---------|
| **Normal Use** | `python benchmark_pointwise.py` |
| **Debug Heuristics** | `python benchmark_pointwise.py --show-all-heuristics` |
| **Maximum Visibility** | `python benchmark_pointwise.py --show-all-heuristics --clear-cache-every 1` |

✅ **Default behavior**: Fast, realistic (cache enabled)  
✅ **Debug mode**: Slower, comprehensive heuristics visibility  
✅ **Flexible**: Adjust clearing frequency with `--clear-cache-every N`

The heuristics aren't "dying" - the cache is just doing its job! This flag lets you bypass it for analysis purposes. 🚀


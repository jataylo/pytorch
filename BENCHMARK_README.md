# Pointwise Kernel Benchmark Suite

Comprehensive benchmark to validate the new pointwise heuristics on real workloads.

## What It Tests

### 5 Benchmark Categories:

1. **Elementwise Add** - `z = x + y`
   - Simple 2-input pointwise operation
   - Tests basic memory bandwidth

2. **Elementwise Multiply** - `z = x * y`
   - Another basic 2-input operation
   - Similar characteristics to add

3. **ReLU + Sigmoid** - `z = sigmoid(relu(x))`
   - Chained activations
   - Tests operator chaining

4. **GELU Activation** - `z = gelu(x)`
   - Complex activation function
   - Multiple operations per element

5. **Fused Pointwise Model** - `z = gelu((x + y) * w + b)`
   - Multiple operations with shared inputs
   - **Tests kernel fusion** (should compile to 1 kernel vs 4 in eager)
   - Most important benchmark for validating heuristics

### Test Shapes (13 per operation = 65 total):

| Category | Shapes | Elements |
|----------|--------|----------|
| **1D** | small, medium, large, huge | 1K - 16M |
| **2D** | square (256²-4096²), wide, tall | 64K - 16M |
| **3D** | cube (32³-128³), batch | 32K - 2M |

## Usage

### Quick Test (verify it works)

```bash
python /root/test_benchmark_quick.py
```

### Full Benchmark (with new heuristics)

```bash
python /root/benchmark_pointwise.py
```

**Options:**
- `--warmup N` - Warmup iterations (default: 10)
- `--iters N` - Benchmark iterations (default: 50)
- `--csv FILE` - Output CSV file (default: pointwise_benchmark_results.csv)
- `--no-heuristics` - Disable new heuristics (use original behavior)

### A/B Comparison (new vs original)

```bash
bash /root/compare_heuristics.sh
```

This runs the benchmark twice:
1. With `TORCHINDUCTOR_POINTWISE_HEURISTICS=0` (original)
2. With `TORCHINDUCTOR_POINTWISE_HEURISTICS=1` (new)

Then compare:
```bash
grep 'OVERALL' /tmp/benchmark_original.log
grep 'OVERALL' /tmp/benchmark_new.log
```

## Expected Output

```
================================================================================
  POINTWISE KERNEL BENCHMARK SUITE
================================================================================
  Device: cuda
  Warmup iterations: 10
  Benchmark iterations: 50
  Pointwise heuristics: 1
  GPU: AMD Instinct MI350
================================================================================

================================================================================
  Benchmark 1: Elementwise Add (z = x + y)
================================================================================
  1D_small             | Eager:   0.0123ms | Compile:   0.0089ms | Speedup:  1.382x
  1D_medium            | Eager:   0.0234ms | Compile:   0.0145ms | Speedup:  1.614x
  1D_large             | Eager:   0.1234ms | Compile:   0.0823ms | Speedup:  1.499x
  ...

================================================================================
  Benchmark 5: Fused Pointwise Model
  z = gelu((x + y) * w + b)
================================================================================
  1D_small             | Eager:   0.0456ms | Compile:   0.0089ms | Speedup:  5.123x
  1D_medium            | Eager:   0.0823ms | Compile:   0.0145ms | Speedup:  5.676x
  ...

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
  OVERALL             :  2.123x
================================================================================

Best speedup:  fused_1D_huge                              5.823x
Worst speedup: add_1D_small                               1.123x

Speedup by Problem Size:
--------------------------------------------------
  small (<100K)       :  1.234x (20 kernels)
  medium (100K-1M)    :  1.823x (24 kernels)
  large (1M-10M)      :  2.234x (15 kernels)
  huge (>10M)         :  2.456x (6 kernels)
================================================================================

✅ Results saved to pointwise_benchmark_results.csv
```

## Key Metrics

### What to Look For:

1. **Overall Geomean > 1.0** ✅
   - Anything above 1.0 means compile is faster than eager

2. **Fused Benchmark Speedup > 3.0x** ✅
   - Eager runs 4 separate kernels
   - Compile should fuse into 1 kernel
   - ~4x speedup validates fusion is working

3. **Large Problems > Small Problems** ✅
   - Heuristics should shine on larger problems
   - Small problems have more launch overhead

4. **2D/3D Configs Work** ✅
   - Validates multi-dimensional heuristics
   - Ensures XBLOCK/YBLOCK/ZBLOCK logic is correct

## Files

| File | Description |
|------|-------------|
| `benchmark_pointwise.py` | Main benchmark script |
| `compare_heuristics.sh` | A/B comparison script |
| `test_benchmark_quick.py` | Quick validation test |
| `pointwise_benchmark_results.csv` | Output results (CSV) |

## Troubleshooting

### No speedup on small problems?

This is expected! Small problems (<100K elements) have high launch overhead and may not benefit as much from optimized configs.

### Speedup < 1.0 (compile slower)?

1. Check if heuristics are enabled:
   ```bash
   grep "POINTWISE HEURISTICS" output
   ```

2. Try with heuristics off for comparison:
   ```bash
   python benchmark_pointwise.py --no-heuristics
   ```

3. Check compilation overhead:
   - First run is slower (JIT compilation)
   - Subsequent runs use cached kernels

### Fused benchmark not showing big speedup?

1. Verify fusion is happening:
   ```bash
   TORCH_LOGS="+graph_code" python benchmark_pointwise.py 2>&1 | grep "poi"
   ```
   You should see 1 `poi*` kernel for compile vs 4 for eager

2. Check if autotuning is enabled:
   - Use `mode='max-autotune'` in `torch.compile()`
   - Without max-autotune, only top config is used

## Advanced Usage

### Custom Operation

Add to `benchmark_pointwise.py`:

```python
def bench_my_op(self):
    """Benchmark: z = my_custom_operation(x, y)"""
    def eager_fn(x, y):
        # Your operation here
        return torch.tanh(x * y + x)
    
    compile_fn = torch.compile(eager_fn, mode='max-autotune')
    
    for shape_name, shape in self.shapes:
        x = torch.randn(shape, device=self.device)
        y = torch.randn(shape, device=self.device)
        
        result = self.benchmark_op(f"my_op_{shape_name}", eager_fn, compile_fn, [x, y])
        # ... record results ...
```

### Custom Shapes

```python
self.shapes = [
    ("my_shape", (1024, 2048, 4)),
    # Add more...
]
```

## Expected Performance Targets

Based on MI350/CDNA4:

| Operation | Target Geomean | Notes |
|-----------|----------------|-------|
| Simple (add, mul) | 1.3-1.5x | Memory bound, limited optimization |
| Activations (relu, gelu) | 1.8-2.2x | More compute, better config matters |
| **Fused** | **4.0-6.0x** | **Most important - validates fusion** |
| **Overall** | **2.0-2.5x** | **Target metric** |

## Interpreting Results

### Good Result ✅

```
OVERALL: 2.234x
  fused: 5.123x
```
→ Heuristics are working! 🎉

### Needs Investigation ⚠️

```
OVERALL: 1.123x
  fused: 1.456x
```
→ Fusion may not be happening or configs not optimal

### Problem ❌

```
OVERALL: 0.856x
  fused: 0.923x
```
→ Compile is slower than eager - check logs for issues

## Next Steps

1. Run benchmark: `python /root/benchmark_pointwise.py`
2. Check overall geomean - should be > 2.0x
3. Verify fused speedup - should be > 4.0x
4. Compare with original: `bash /root/compare_heuristics.sh`
5. If results are good, the heuristics are validated! ✅


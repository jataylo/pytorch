# Autotuner Results Limitation

## Current Behavior

When you see:
```
[POINTWISE HEURISTICS] Benchmarking ALL 9 valid configs to validate heuristics...
[POINTWISE HEURISTICS] Generated 9 configs for autotuning/benchmarking
```

The configs are passed to PyTorch Inductor's autotuner, which:
1. Benchmarks each config
2. Selects the best one
3. Uses it for code generation

**However, we don't see the actual benchmark timings** for each config.

## Why?

The autotuning happens deep inside PyTorch Inductor's `Autotuner` class (`torch/_inductor/select_algorithm.py`), which:
- Runs benchmarks in a subprocess/separate process
- Only returns the **best config** (not timing details)
- Doesn't expose per-config timing results in a way we can easily capture

## Workaround

To see actual performance, you can:

### 1. Use the Standalone Validation Script
```bash
python /root/validate_heuristics.py
```

This manually benchmarks each config using Triton directly and shows:
```
Pred   Actual XBLOCK   Warps  Pred Score   Runtime (ms)   Status
1      2      64       1      0.8218       0.011065       ⚠️  PREDICTED #1
2      3      128      2      0.8146       0.010747       ✅ Close
3      1      256      4      0.8110       0.010652       🏆 ACTUAL BEST
4      4      512      8      0.8092       0.011029       ✅ Close
```

### 2. Enable Inductor Debug Logging
```bash
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_POINTWISE_HEURISTICS=1 python your_script.py
```

This generates debug output in `/tmp/torchinductor_*/` including:
- Generated Triton code for each config
- Autotuner benchmark results (in `output_code.txt`)
- Which config was selected

### 3. Check Triton Cache
```bash
# After running, check the Triton cache
ls -lth ~/.triton/cache/ | head -20
```

The most recently modified kernels are the ones selected by the autotuner.

## What We CAN Show

Currently, we show:
1. ✅ **ALL configs generated** with predicted scores
2. ✅ **Which configs pass validation** ([TOP-N], [VALID], [FILTERED])
3. ✅ **Detailed factor breakdowns** (balance, launch, occupancy, grid)
4. ✅ **Grid configuration** (blocks, threads per block)

## What We CANNOT Show (Yet)

1. ❌ **Actual timing** for each config from autotuner
2. ❌ **Which config won** the autotuner race
3. ❌ **Performance comparison** between configs in real-time

## Future Improvements

To show actual autotuner results, we would need to:

1. **Hook into Autotuner class**:
   - Modify `torch/_inductor/select_algorithm.py`
   - Capture timing results before they're discarded
   - Return full benchmark data (not just best config)

2. **Add logging callback**:
   ```python
   class Autotuner:
       def benchmark_all_configs(self, configs):
           results = []
           for cfg in configs:
               time = self.benchmark_one(cfg)
               results.append((cfg, time))
               # LOG HERE: config + timing
           return min(results, key=lambda x: x[1])
   ```

3. **Store results globally**:
   - Save timing results to a shared dict/file
   - Read them in heuristics output
   - Display in comparison table

This would require modifying PyTorch Inductor core, which is beyond the scope of the current heuristics system.

## Recommendation

For now, use `validate_heuristics.py` to:
- See predicted vs actual performance
- Validate heuristics accuracy
- Compare configs side-by-side

This gives you the insight you need without requiring deep Inductor modifications.

## Example Workflow

```bash
# 1. Run your model with heuristics
TORCHINDUCTOR_POINTWISE_HEURISTICS=1 python your_model.py

# You see:
# [POINTWISE HEURISTICS] ALL CONFIGS (sorted by predicted score):
#   # 1: [TOP-1]   score=0.8218 | XBLOCK=4, YBLOCK=16
#   # 2: [TOP-2]   score=0.8146 | XBLOCK=4, YBLOCK=32
#   ...
# [POINTWISE HEURISTICS] Benchmarking ALL 9 valid configs...
# (Autotuner runs internally, picks best)

# 2. Validate predictions against actual performance
python /root/validate_heuristics.py

# You see:
# Predicted Best: XBLOCK=4, YBLOCK=16 (score=0.8218)
# Actual Best: XBLOCK=8, YBLOCK=32 (runtime=0.0106ms)
# Slowdown: 1.2%
```

This gives you full visibility into heuristics accuracy.

## Summary

| What                    | Available? | How                              |
|-------------------------|------------|----------------------------------|
| Predicted scores        | ✅ Yes      | Heuristics output                |
| Config generation       | ✅ Yes      | Heuristics output                |
| Validation filtering    | ✅ Yes      | [TOP-N], [VALID], [FILTERED]     |
| **Actual timing**       | ❌ No       | Autotuner internal               |
| **Which config won**    | ❌ No       | Autotuner internal               |
| Manual benchmarking     | ✅ Yes      | `validate_heuristics.py`         |
| Debug output            | ⚠️  Partial | `TORCH_COMPILE_DEBUG=1`          |

The limitation is architectural, not a bug. The validation script provides the insight you need. 🚀


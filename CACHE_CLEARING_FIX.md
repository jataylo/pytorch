# Cache Clearing Fix - Missing Heuristics Output

## Problem Identified

Even with `--clear-per-shape` flag, **heuristics were NOT showing for every sub-problem!**

### Root Cause

**Two separate caching issues:**

1. **In-Memory Compiled Function Cache**
   - `compile_fn = torch.compile(eager_fn)` created ONCE per operation
   - Used for ALL shapes in the loop
   - PyTorch's `torch._dynamo` keeps compiled functions in memory
   - Clearing disk cache (`/tmp/torchinductor_root/`) didn't help!

2. **Missing Validation Summaries**
   - `TORCHINDUCTOR_HEURISTICS_REAL_BENCH` environment variable not set
   - Validation summary (predicted vs actual) never printed

## The Fix

### 1. Clear In-Memory Cache with `torch._dynamo.reset()`

**Before (BROKEN):**
```python
def clear_compilation_cache(self):
    # Only cleared disk cache
    shutil.rmtree('/tmp/torchinductor_root/')
    shutil.rmtree('~/.triton/cache/')
```

**After (FIXED):**
```python
def clear_compilation_cache(self):
    # CRITICAL: Clear in-memory cache FIRST
    torch._dynamo.reset()  # ← Forces torch.compile to recompile
    torch._inductor.metrics.reset()
    
    # Then clear disk caches
    shutil.rmtree('/tmp/torchinductor_root/')
    shutil.rmtree('~/.triton/cache/')
```

### 2. Enable Validation Summaries by Default

```python
# NEW: Enable real bench mode
os.environ.setdefault("TORCHINDUCTOR_HEURISTICS_REAL_BENCH", "1")
```

## Why `torch._dynamo.reset()` is Critical

### What Happens Without It:

```
Benchmark 1: Add
  compile_fn = torch.compile(eager_fn)  ← Created ONCE
  
  for shape in shapes:
      Shape 1: (512,)
        → torch.compile sees new shape
        → Compiles kernel
        → Shows heuristics ✅
        
      clear_cache()  # Clears disk only!
      
      Shape 2: (4096,)
        → torch.compile recognizes same function object
        → Finds cached trace in memory ❌
        → No recompilation!
        → No heuristics! ❌
```

### What Happens With `torch._dynamo.reset()`:

```
Benchmark 1: Add
  compile_fn = torch.compile(eager_fn)  ← Created ONCE
  
  for shape in shapes:
      Shape 1: (512,)
        → Compiles, shows heuristics ✅
        
      torch._dynamo.reset()  # ← Invalidates in-memory cache
      clear_disk_caches()
      
      Shape 2: (4096,)
        → torch.compile forced to recompile ✅
        → Fresh compilation!
        → Shows heuristics! ✅
```

## Expected Behavior Now

### With `--clear-per-shape`:

```bash
python benchmark_pointwise.py --clear-per-shape
```

**You should see:**

```
Benchmark 1: Add
  Shape 1: (512,)
  [POINTWISE HEURISTICS] Problem: (512,)
    #1: XBLOCK=512, nw=8 → 0.8932 ✅
    ...
  ================================================================================
  [HEURISTICS VALIDATION] Predicted vs Actual Performance
  ================================================================================
  📊 PREDICTED Best: XBLOCK=512, nw=8
  🏆 ACTUAL Best: XBLOCK=256, nw=4
  📈 ACCURACY: Within 10% ✅
  
  Shape 2: (4096,)
  [POINTWISE HEURISTICS] Problem: (4096,)  ← Fresh heuristics! ✅
    #1: XBLOCK=256, nw=4 → 0.9165 ✅
    ...
  [HEURISTICS VALIDATION] ...  ← Validation for THIS shape ✅
  
  Shape 3: (65536,)
  [POINTWISE HEURISTICS] Problem: (65536,)  ← Fresh heuristics! ✅
    ...
  [HEURISTICS VALIDATION] ...  ← Validation for THIS shape ✅
  
... ALL 38 shapes get fresh heuristics + validation! ✅
```

### Total Output Expected:

- **38 shapes × 16 operations = 608 compilations**
- **608 heuristics outputs** (one per compilation)
- **608 validation summaries** (predicted vs actual for each)

## Technical Details

### `torch._dynamo.reset()` What It Does:

1. **Clears FX graph cache**: Invalidates traced computational graphs
2. **Clears guard state**: Resets tensor shape guards
3. **Clears compiled code**: Forces recompilation on next call

### Why Disk Cache Clearing Alone Fails:

```python
compile_fn = torch.compile(eager_fn)

# First call: (512,) shape
compile_fn(x, y)  # → Compiles, caches in memory + disk

# Clear disk cache
shutil.rmtree('/tmp/torchinductor_root/')

# Second call: (4096,) shape
compile_fn(x, y)  # → Finds in-memory trace, skips compilation!
                  # Even though disk cache is empty!
```

### Proper Order:

```python
1. torch._dynamo.reset()        # Clear in-memory first
2. torch._inductor.metrics.reset()  # Reset metrics
3. shutil.rmtree(disk_caches)   # Then clear disk
```

## Verification

To verify the fix is working:

```bash
# Run with cache clearing
python benchmark_pointwise.py --clear-per-shape 2>&1 | tee output.log

# Count heuristics outputs (should be 608)
grep -c "POINTWISE HEURISTICS" output.log

# Count validation summaries (should be 608)
grep -c "HEURISTICS VALIDATION" output.log

# Check if each shape gets heuristics
grep "1D_tiny" output.log | head -5
# Should see heuristics BEFORE each 1D_tiny benchmark
```

## Performance Impact

**Before fix:**
- First shape: Full compilation + heuristics
- Remaining shapes: Fast (cached), but NO heuristics ❌

**After fix:**
- ALL shapes: Full compilation + heuristics ✅
- Much slower (608 full compilations)
- But complete validation data for analysis!

---

**Status: ✅ FIXED**

**Files Modified:**
- `/root/benchmark_pointwise.py`: Added `torch._dynamo.reset()` and `HEURISTICS_REAL_BENCH=1`

**Usage:** `python benchmark_pointwise.py --clear-per-shape`


# Dynamic Warp Size Detection (No Hardcoding)

## Problem

Previously, `num_warps` calculation assumed a hardcoded warp/wave size of 64:

```python
# OLD (hardcoded)
num_warps = max(1, min(threads_per_block // 64, 16))
```

This had several issues:
1. ❌ Assumes warp_size=64 (only correct for AMD GPUs)
2. ❌ Assumes max_warps=16 (arbitrary limit)
3. ❌ Not portable to NVIDIA GPUs (warp_size=32)
4. ❌ Ignores device-specific thread limits

## Solution

Now queries device properties dynamically:

```python
# NEW (dynamic)
warp_size = problem_metadata['warp_size']  # From device
max_warps = max_threads_per_block // warp_size  # Device limit
num_warps = max(1, min(threads_per_block // warp_size, max_warps))
```

## Implementation

### 1. Pass Device Properties to Heuristics

**File:** `torch/_inductor/runtime/triton_heuristics.py`

```python
def _convert_to_pointwise_heuristics_metadata(size_hints, inductor_meta, triton_meta):
    # ... existing code ...
    
    # Get device properties for hardware-specific parameters
    device_props = triton_meta.get("device")
    warp_size = device_props.warp_size if device_props and device_props.warp_size else 64
    max_threads_per_block = device_props.max_threads_per_block if device_props and device_props.max_threads_per_block else 1024
    
    return {
        # ... existing fields ...
        'warp_size': warp_size,
        'max_threads_per_block': max_threads_per_block,
    }
```

### 2. Use Dynamic Values in Config Generation

**File:** `torch/_inductor/codegen/triton_heuristics_pointwise.py`

```python
def generate_all_candidate_configs(problem_metadata: Dict) -> List[Dict]:
    # Get device-specific parameters (don't hardcode warp size)
    warp_size = problem_metadata.get('warp_size', 64)  # Query from device
    max_threads = problem_metadata.get('max_threads_per_block', 1024)
    max_warps = max_threads // warp_size  # Calculate max warps from device limits
    
    if ndims == 1:
        for xblock in block_sizes_1d:
            configs.append({
                'XBLOCK': xblock,
                'num_warps': max(1, min(xblock // warp_size, max_warps))
                #                         ↑            ↑
                #                    dynamic      dynamic
            })
```

## Device Properties Available

From `DeviceProperties` class:

| Property | Type | Example (MI350) | Example (A100) |
|----------|------|-----------------|----------------|
| `warp_size` | int | 64 (wave size) | 32 (warp size) |
| `max_threads_per_block` | int | 1024 | 1024 |
| `multi_processor_count` | int | 256 (CUs) | 108 (SMs) |
| `regs_per_multiprocessor` | int | 131072 | 65536 |

## Results

### MI350/CDNA4 (warp_size=64)

```python
XBLOCK=16   → num_warps=1  (16 / 64 = 0.25 → max(1, 0) = 1)
XBLOCK=64   → num_warps=1  (64 / 64 = 1)
XBLOCK=256  → num_warps=4  (256 / 64 = 4)
XBLOCK=512  → num_warps=8  (512 / 64 = 8)
XBLOCK=1024 → num_warps=16 (1024 / 64 = 16)
```

### Hypothetical NVIDIA (warp_size=32)

```python
XBLOCK=16   → num_warps=1  (16 / 32 = 0.5 → max(1, 0) = 1)
XBLOCK=64   → num_warps=2  (64 / 32 = 2)
XBLOCK=256  → num_warps=8  (256 / 32 = 8)
XBLOCK=512  → num_warps=16 (512 / 32 = 16)
XBLOCK=1024 → num_warps=32 (1024 / 32 = 32, capped by max_warps)
```

### With Device Limits (max_threads_per_block=512)

```python
max_warps = 512 / 64 = 8

XBLOCK=64   → num_warps=1  (64 / 64 = 1, within limit)
XBLOCK=256  → num_warps=4  (256 / 64 = 4, within limit)
XBLOCK=512  → num_warps=8  (512 / 64 = 8, at limit)
XBLOCK=1024 → num_warps=8  (would be 16, capped to 8)
```

## Benefits

✅ **Portable** - Works on AMD (64) and NVIDIA (32) GPUs  
✅ **Correct** - Respects device-specific thread limits  
✅ **Future-proof** - Adapts to new hardware automatically  
✅ **No assumptions** - Queries actual hardware capabilities  

## Testing

Run the test suite:

```bash
python /root/test_dynamic_warp_size.py
```

Expected output:
```
✅ All tests passed!

Key findings:
  • num_warps is now calculated from device warp_size (not hardcoded to 64)
  • Respects max_threads_per_block device limit
  • Formula: num_warps = min(threads_per_block // warp_size, max_warps)
  • Works for different hardware (AMD: 64, NVIDIA: 32, etc.)
```

## Verification with Real Models

```bash
# Run with new dynamic code
TORCHINDUCTOR_POINTWISE_HEURISTICS=1 \
  python micro_benchmarking_pytorch.py --network resnet152 --compile

# Check the log
cat /tmp/pointwise_heuristics_calls.log | grep "num_warps"
```

You should see `num_warps` values that match your device's actual warp size.

## Code Locations

**Modified files:**
1. `/root/pytorch/torch/_inductor/runtime/triton_heuristics.py`
   - `_convert_to_pointwise_heuristics_metadata()` - Added `warp_size` and `max_threads_per_block` to metadata

2. `/root/pytorch/torch/_inductor/codegen/triton_heuristics_pointwise.py`
   - `generate_all_candidate_configs()` - Use dynamic `warp_size` and `max_warps`

**Test file:**
- `/root/test_dynamic_warp_size.py` - Comprehensive test suite

## Summary

**Before:**
```python
num_warps = max(1, min(xblock // 64, 16))  # Hardcoded
```

**After:**
```python
warp_size = device_props.warp_size  # From hardware
max_warps = max_threads_per_block // warp_size  # From hardware
num_warps = max(1, min(xblock // warp_size, max_warps))  # Dynamic
```

This ensures the heuristics work correctly across all GPU architectures without hardcoded assumptions about warp/wave size or thread limits.



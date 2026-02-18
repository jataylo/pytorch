# Fixing the Critical Gaps - Practical Solutions

## 🎯 Can We Actually Fix These?

### 1. No Actual Kernel Information + 2. bytes_per_element

**YES! We can look at the FXGraph AND the generated Triton kernel!**

#### Option A: FXGraph Analysis (BETTER - Earlier in pipeline)

The FXGraph contains the actual operations BEFORE Triton code generation:

```python
# In runtime/triton_heuristics.py, we have access to the FXGraph node
def pointwise(node, ...):
    # node is the FX graph node
    # node.args contains input tensors
    # node.target contains the operation
    
    # Extract actual info:
    num_inputs = len([arg for arg in node.args if isinstance(arg, torch.fx.Node)])
    num_outputs = 1  # or count outputs if multiple
    
    # Detect broadcasts:
    input_shapes = [arg.meta['val'].shape for arg in node.args if ...]
    has_broadcast = any(s1 != s2 for s1, s2 in zip(input_shapes[0], input_shapes[1]))
    
    # Count operations:
    if node.target == operator.add:
        ops_per_element = 1
    elif node.target == operator.mul:
        ops_per_element = 1
    elif node.target == torch.exp:
        ops_per_element = 50  # expensive!
    # etc...
```

**Where to do this:** In `runtime/triton_heuristics.py` at the `pointwise()` function

#### Option B: Parse Generated Triton Kernel (EASIER - Kernel already exists)

```python
# After kernel generation, parse the Triton code
def extract_kernel_metadata(kernel_code: str) -> Dict:
    metadata = {}
    
    # Count parameters from signature
    # def kernel(x_ptr, y_ptr, z_ptr, ...):
    import re
    params = re.findall(r'def \w+\((.*?)\):', kernel_code, re.DOTALL)
    if params:
        param_list = params[0].split(',')
        ptr_params = [p for p in param_list if '_ptr' in p]
        metadata['num_tensors'] = len(ptr_params)
        metadata['num_inputs'] = len(ptr_params) - 1  # assume last is output
        metadata['num_outputs'] = 1
    
    # Count operations and their types
    op_counts = {
        'fast': kernel_code.count('tl.add') + kernel_code.count('tl.mul'),
        'medium': kernel_code.count('tl.div') + kernel_code.count('tl.sqrt'),
        'slow': (kernel_code.count('tl.exp') + kernel_code.count('tl.sin') + 
                 kernel_code.count('tl.log') + kernel_code.count('tl.cos')),
    }
    
    # Weighted ops per element
    metadata['ops_per_element'] = (
        op_counts['fast'] * 1 +
        op_counts['medium'] * 10 +
        op_counts['slow'] * 30
    )
    
    # Calculate actual bytes per element
    metadata['bytes_per_element'] = metadata['num_tensors'] * 4  # FP32
    
    # Detect broadcasts by looking for dimension handling
    if 'xindex < xnumel' in kernel_code and 'yindex < ynumel' in kernel_code:
        # Check if one dimension is broadcast
        if kernel_code.count('yindex') < kernel_code.count('xindex') / 2:
            metadata['has_broadcast_y'] = True
    
    return metadata
```

**Where to hook this in:**

```python
# In runtime/triton_heuristics.py
class TritonKernelWrapper:
    def __init__(self, kernel_code, ...):
        self.kernel_code = kernel_code
        self.metadata = extract_kernel_metadata(kernel_code)
        
    def get_heuristics(self):
        # Use self.metadata instead of defaults!
        problem_metadata = {
            'num_inputs': self.metadata['num_inputs'],
            'num_outputs': self.metadata['num_outputs'],
            'ops_per_element': self.metadata['ops_per_element'],
            'bytes_per_element': self.metadata['bytes_per_element'],
            'has_broadcast': self.metadata.get('has_broadcast_y', False),
            # ... other fields
        }
        return score_configs(problem_metadata)
```

---

### 3. Fix L1/L2 Cache Model

**YES! Easy fix - account for per-CU replication**

```python
@staticmethod
def estimate_memory_time_us(total_bytes: int, problem_metadata: Dict, 
                            num_blocks: int) -> float:  # ADD num_blocks param
    """Fixed cache model accounting for per-CU replication."""
    
    device_consts = BottleneckAnalysis._get_device_constants()
    l1_cache_size = device_consts['l1_cache_size']
    l2_cache_size = device_consts['l2_cache_size']
    memory_bandwidth_gb_s = device_consts['memory_bandwidth_gb_s']
    num_cus = device_consts.get('num_cus', 304)
    
    # NEW: Account for per-CU replication
    # If many blocks run in parallel, each CU gets its own copy
    blocks_per_cu = num_blocks / num_cus
    
    if total_bytes <= l1_cache_size:
        # L1 hit ONLY if few blocks (< CUs)
        if num_blocks <= num_cus:
            # True L1 hit - data fits in each CU's L1
            return 0.01 + 0.05 * (total_bytes / l1_cache_size)
        else:
            # FALSE L1 "hit" - data replicated across CUs!
            # Actual: each CU loads from HBM/L2
            # Use L2 or HBM depending on broadcast
            has_broadcast = problem_metadata.get('has_broadcast', False)
            if has_broadcast and total_bytes * blocks_per_cu <= l2_cache_size:
                # Broadcast can fit in L2, reused across blocks
                l2_bandwidth_gb_s = 1000.0
                return total_bytes / (l2_bandwidth_gb_s * 1e3)
            else:
                # Go to HBM
                effective_bandwidth = memory_bandwidth_gb_s * 0.8
                return total_bytes / (effective_bandwidth * 1e3)
    
    elif total_bytes <= l2_cache_size:
        # L2 cache - check for broadcast reuse
        has_broadcast = problem_metadata.get('has_broadcast', False)
        if has_broadcast:
            # Broadcast tensor stays in L2, reused across all blocks
            # Only pay L2 bandwidth cost once
            l2_bandwidth_gb_s = 1000.0
            return total_bytes / (l2_bandwidth_gb_s * 1e3)
        else:
            # Normal L2 access
            l2_bandwidth_gb_s = 1000.0
            return total_bytes / (l2_bandwidth_gb_s * 1e3)
    
    else:
        # HBM access
        has_broadcast = problem_metadata.get('has_broadcast', False)
        broadcast_size = problem_metadata.get('broadcast_tensor_bytes', 0)
        
        if has_broadcast and broadcast_size > 0:
            # Separate broadcast tensor from other I/O
            # Broadcast: one-time load, cached in L2/L1
            # Other: full HBM bandwidth
            non_broadcast_bytes = total_bytes - broadcast_size
            
            if broadcast_size <= l2_cache_size:
                # Broadcast cached, only pay for non-broadcast
                effective_bandwidth = memory_bandwidth_gb_s * 0.8
                return non_broadcast_bytes / (effective_bandwidth * 1e3)
        
        # Normal HBM access
        effective_bandwidth = memory_bandwidth_gb_s * 0.8
        return total_bytes / (effective_bandwidth * 1e3)
```

**Also update the call site to pass num_blocks:**

```python
# In analyze_bottleneck()
memory_us = BottleneckAnalysis.estimate_memory_time_us(
    total_bytes, problem_metadata, num_blocks  # ADD THIS
)
```

---

### 4. Overhead - Can we improve?

**YES! Multiple improvements possible:**

```python
@staticmethod
def estimate_overhead_time_us(num_blocks: int, problem_metadata: Dict,
                              config: Dict) -> float:  # ADD params
    """
    Improved overhead estimate accounting for kernel complexity.
    """
    # Base kernel launch
    base_overhead_us = 3.0
    
    # Factor 1: Scale by number of kernel arguments
    num_tensors = problem_metadata.get('num_tensors', 3)
    # Each tensor pointer adds ~0.1μs setup overhead
    arg_overhead = num_tensors * 0.1
    
    # Factor 2: Scale by num_warps (resource allocation overhead)
    num_warps = config.get('num_warps', 4)
    # More warps = more resource allocation overhead
    warp_overhead = (num_warps - 1) * 0.2  # +0.2μs per extra warp
    
    # Factor 3: Grid setup (logarithmic for large grids)
    if num_blocks > 1000:
        grid_overhead = 0.5 * math.log2(num_blocks / 1000)
    elif num_blocks > 100:
        grid_overhead = 0.2 * math.log2(num_blocks / 100)
    else:
        grid_overhead = 0.0
    
    # Factor 4: Complexity overhead (masks, branches)
    has_mask = problem_metadata.get('has_mask', False)
    complexity_overhead = 0.5 if has_mask else 0.0
    
    total_overhead = (base_overhead_us + arg_overhead + 
                     warp_overhead + grid_overhead + complexity_overhead)
    
    return total_overhead
```

**Update call sites:**

```python
# In analyze_bottleneck()
overhead_us = BottleneckAnalysis.estimate_overhead_time_us(
    num_blocks, problem_metadata, config  # ADD THESE
)
```

---

### 5. Instruction Mix - Can we get from kernel?

**YES! Parse the Triton kernel code:**

```python
@staticmethod
def estimate_compute_time_us_improved(num_ops: int, threads_per_block: int, 
                                      num_blocks: int, 
                                      kernel_code: str = None) -> float:
    """
    Improved compute estimation using actual instruction mix from kernel.
    """
    device_consts = BottleneckAnalysis._get_device_constants()
    peak_tflops = device_consts['compute_tflops']
    memory_bandwidth_gb_s = device_consts['memory_bandwidth_gb_s']
    
    oi_ceiling = BottleneckAnalysis.get_oi_ceiling()
    
    total_elements = threads_per_block * num_blocks
    
    # NEW: Get instruction mix from kernel code
    if kernel_code:
        op_mix = parse_instruction_mix(kernel_code)
        # op_mix = {'fast': 2, 'medium': 0, 'slow': 1}
        
        # Weighted ops per element
        weighted_ops = (
            op_mix.get('fast', 0) * 1 +      # add/mul: 1 cycle
            op_mix.get('medium', 0) * 10 +   # div/sqrt: 10 cycles
            op_mix.get('slow', 0) * 30       # exp/sin: 30 cycles
        )
        ops_per_element = weighted_ops / max(total_elements, 1)
        
        # Use actual bytes from kernel (counted tensors)
        bytes_per_element = op_mix.get('bytes_per_element', 12.0)
    else:
        # Fallback to defaults
        ops_per_element = num_ops / max(total_elements, 1)
        bytes_per_element = 12.0
    
    arithmetic_intensity = ops_per_element / bytes_per_element
    
    if arithmetic_intensity < oi_ceiling:
        return 0.0  # Memory-bound
    else:
        # Compute-bound - use efficiency based on instruction mix
        if kernel_code:
            # Adjust efficiency based on slow ops
            slow_fraction = op_mix.get('slow', 0) / max(sum(op_mix.values()), 1)
            if slow_fraction > 0.5:
                efficiency = 0.6  # Lots of slow ops
            elif slow_fraction > 0.2:
                efficiency = 0.7  # Some slow ops
            else:
                efficiency = 0.8  # Mostly fast ops
        else:
            efficiency = 0.7  # Default
        
        achievable_tflops = peak_tflops * efficiency
        ops_per_us = (achievable_tflops * 1e12) / 1e6
        
        return num_ops / ops_per_us


def parse_instruction_mix(kernel_code: str) -> Dict[str, int]:
    """
    Parse Triton kernel code to extract instruction mix.
    """
    import re
    
    op_mix = {
        'fast': 0,
        'medium': 0,
        'slow': 0,
    }
    
    # Count fast ops (1-2 cycles)
    op_mix['fast'] = (
        kernel_code.count('tl.add') +
        kernel_code.count('tl.sub') +
        kernel_code.count('tl.mul') +
        kernel_code.count(' + ') +  # Python ops
        kernel_code.count(' * ')
    )
    
    # Count medium ops (10-20 cycles)
    op_mix['medium'] = (
        kernel_code.count('tl.div') +
        kernel_code.count('tl.fdiv') +
        kernel_code.count('tl.sqrt') +
        kernel_code.count(' / ')
    )
    
    # Count slow ops (30-50 cycles)
    op_mix['slow'] = (
        kernel_code.count('tl.exp') +
        kernel_code.count('tl.log') +
        kernel_code.count('tl.sin') +
        kernel_code.count('tl.cos') +
        kernel_code.count('tl.tanh')
    )
    
    # Count actual tensor parameters for bytes_per_element
    ptr_count = kernel_code.count('_ptr')
    op_mix['bytes_per_element'] = ptr_count * 4  # FP32
    
    return op_mix
```

---

## 🎯 Integration Plan

### Step 1: Add kernel_code parameter to bottleneck analysis

```python
# In triton_heuristics_adaptive.py
@staticmethod
def analyze_bottleneck(config: Dict, problem_metadata: Dict, 
                       kernel_code: str = None) -> Dict[str, float]:
    """
    Now accepts optional kernel_code for better estimates.
    """
    # ... existing code ...
    
    # Parse kernel if available
    if kernel_code:
        kernel_metadata = extract_kernel_metadata(kernel_code)
        # Override defaults with actual values
        problem_metadata.update(kernel_metadata)
    
    # ... rest of analysis uses actual data ...
```

### Step 2: Thread kernel_code through the call chain

```python
# In runtime/triton_heuristics.py
def score_configs_with_kernel(configs, problem_metadata, kernel_code):
    for config in configs:
        # Pass kernel_code to bottleneck analysis
        analysis = BottleneckAnalysis.analyze_bottleneck(
            config, problem_metadata, kernel_code=kernel_code
        )
        weights = BottleneckAnalysis.get_adaptive_weights(...)
        score = score_config(config, problem_metadata, weights)
```

### Step 3: Extract kernel_code at the right place

```python
# In runtime/triton_heuristics.py, after kernel generation:
class TritonKernelOverrides:
    def __call__(self, *args, **kwargs):
        # After autotune, we have access to kernel.src (Triton kernel code)
        kernel_code = self.kernel.src
        
        # Use for future heuristics
        metadata = extract_kernel_metadata(kernel_code)
        # Cache this for similar kernels
```

---

## 📊 Expected Impact

| Fix | Accuracy Gain | Implementation Time |
|-----|---------------|---------------------|
| Extract num_inputs/outputs from kernel | +10-15% | 1-2 days |
| Parse instruction mix | +10-15% | 2-3 days |
| Fix L1 per-CU model | +5-10% | 1 day |
| Improve overhead estimate | +3-5% | 1 day |
| Detect broadcasts | +10-20% (when present) | 2-3 days |

**Total: +38-65% accuracy improvement in 7-11 days!**

---

## 🚀 Quick Start (Today!):

1. **Add kernel_code extraction** (30 min):
   - Find where kernel.src is available
   - Pass it to heuristics

2. **Parse tensor count** (1 hour):
   - Count `_ptr` parameters
   - Use for bytes_per_element

3. **Fix L1 model** (2 hours):
   - Add num_blocks check
   - Account for replication

4. **Parse instruction mix** (3 hours):
   - Count tl.exp, tl.div, etc.
   - Weight ops by latency

**In 1 day, we could have 20-30% better accuracy!**


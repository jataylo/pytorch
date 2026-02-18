"""
Kernel Analysis Module - Extract metadata from Triton kernels

Parses generated Triton kernel code to extract:
- Number of inputs/outputs
- Instruction mix (fast/medium/slow ops)
- Bytes per element
- Broadcast detection
"""

import re
from typing import Dict


def extract_kernel_metadata(kernel_code: str) -> Dict:
    """
    Parse Triton kernel code to extract metadata for heuristics.
    
    Returns:
        Dict with:
        - 'num_tensors': Total tensor parameters
        - 'num_inputs': Number of input tensors
        - 'num_outputs': Number of output tensors
        - 'bytes_per_element': Bytes read+written per element
        - 'ops_per_element': Weighted operations per element
        - 'fast_ops': Count of fast operations (add, mul)
        - 'medium_ops': Count of medium operations (div, sqrt)
        - 'slow_ops': Count of slow operations (exp, sin, log)
        - 'has_broadcast': Whether broadcast pattern detected
        - 'has_mask': Whether masking is used
    """
    metadata = {
        'num_tensors': 3,
        'num_inputs': 2,
        'num_outputs': 1,
        'bytes_per_element': 12.0,
        'ops_per_element': 2,
        'fast_ops': 2,
        'medium_ops': 0,
        'slow_ops': 0,
        'has_broadcast': False,
        'has_mask': False,
    }
    
    if not kernel_code:
        return metadata
    
    try:
        # Extract tensor parameters from function signature
        # Pattern: def kernel_name(arg1_ptr, arg2_ptr, ..., output_ptr, ...)
        sig_match = re.search(r'def\s+\w+\s*\((.*?)\):', kernel_code, re.DOTALL)
        if sig_match:
            params = sig_match.group(1)
            # Count pointer parameters (tensors)
            ptr_params = [p.strip() for p in params.split(',') if '_ptr' in p]
            metadata['num_tensors'] = len(ptr_params)
            
            # Heuristic: Usually last 1-2 are outputs, rest are inputs
            # Look for store operations to confirm
            num_stores = kernel_code.count('tl.store')
            metadata['num_outputs'] = max(1, num_stores)
            metadata['num_inputs'] = metadata['num_tensors'] - metadata['num_outputs']
            
            # bytes_per_element = num_tensors * 4 bytes (FP32)
            metadata['bytes_per_element'] = float(metadata['num_tensors'] * 4)
        
        # Count operations by type
        # Fast ops (1-2 cycles): add, sub, mul, fma
        fast_ops = (
            kernel_code.count('tl.add') +
            kernel_code.count('tl.sub') +
            kernel_code.count('tl.mul') +
            kernel_code.count('tl.fma') +
            # Python ops in kernel
            len(re.findall(r'\s\+\s', kernel_code)) +
            len(re.findall(r'\s-\s', kernel_code)) +
            len(re.findall(r'\s\*\s', kernel_code))
        )
        
        # Medium ops (10-20 cycles): div, fdiv, sqrt, rsqrt
        medium_ops = (
            kernel_code.count('tl.div') +
            kernel_code.count('tl.fdiv') +
            kernel_code.count('tl.sqrt') +
            kernel_code.count('tl.rsqrt') +
            len(re.findall(r'\s/\s', kernel_code))
        )
        
        # Slow ops (30-50 cycles): exp, log, sin, cos, tanh, sigmoid
        slow_ops = (
            kernel_code.count('tl.exp') +
            kernel_code.count('tl.exp2') +
            kernel_code.count('tl.log') +
            kernel_code.count('tl.log2') +
            kernel_code.count('tl.sin') +
            kernel_code.count('tl.cos') +
            kernel_code.count('tl.tanh') +
            kernel_code.count('tl.sigmoid')
        )
        
        metadata['fast_ops'] = fast_ops
        metadata['medium_ops'] = medium_ops
        metadata['slow_ops'] = slow_ops
        
        # Weighted ops per element (assuming per-iteration counts)
        # Normalize by number of elements (heuristic: 1 per kernel invocation)
        total_weighted_ops = (
            fast_ops * 1 +      # 1 cycle
            medium_ops * 10 +   # 10 cycles
            slow_ops * 30       # 30 cycles
        )
        metadata['ops_per_element'] = max(2, total_weighted_ops) if total_weighted_ops > 0 else 2
        
        # Detect broadcasts
        # Look for dimension handling patterns
        if 'xindex' in kernel_code and 'yindex' in kernel_code:
            # Count usage of each index
            x_count = kernel_code.count('xindex')
            y_count = kernel_code.count('yindex')
            
            # If one index is used much less, likely broadcast
            if x_count > 0 and y_count > 0:
                ratio = max(x_count, y_count) / min(x_count, y_count)
                if ratio > 3:
                    metadata['has_broadcast'] = True
        
        # Single dimension with conditional → likely broadcast
        if kernel_code.count('xindex < xnumel') > 0:
            if kernel_code.count('yindex') == 0 and kernel_code.count('ynumel') == 0:
                # 1D kernel, check for multiple loads from same base
                if kernel_code.count('tl.load') > metadata['num_tensors']:
                    metadata['has_broadcast'] = True
        
        # Detect masking
        if 'tl.load' in kernel_code and 'mask=' in kernel_code:
            metadata['has_mask'] = True
        elif 'xmask' in kernel_code or 'ymask' in kernel_code:
            metadata['has_mask'] = True
        
    except Exception:
        # If parsing fails, return defaults
        pass
    
    return metadata


def get_instruction_mix_efficiency(metadata: Dict) -> float:
    """
    Calculate compute efficiency based on instruction mix.
    
    Args:
        metadata: Dict from extract_kernel_metadata
    
    Returns:
        Efficiency factor (0.5-0.9) for compute-bound kernels
    """
    fast_ops = metadata.get('fast_ops', 0)
    medium_ops = metadata.get('medium_ops', 0)
    slow_ops = metadata.get('slow_ops', 0)
    
    total_ops = fast_ops + medium_ops + slow_ops
    if total_ops == 0:
        return 0.7  # Default
    
    # Calculate weighted average efficiency
    # Fast ops: 90% efficiency (good ILP, low latency)
    # Medium ops: 70% efficiency (moderate latency)
    # Slow ops: 60% efficiency (high latency, poor ILP)
    
    fast_frac = fast_ops / total_ops
    medium_frac = medium_ops / total_ops
    slow_frac = slow_ops / total_ops
    
    efficiency = (
        fast_frac * 0.9 +
        medium_frac * 0.7 +
        slow_frac * 0.6
    )
    
    return max(0.5, min(0.9, efficiency))


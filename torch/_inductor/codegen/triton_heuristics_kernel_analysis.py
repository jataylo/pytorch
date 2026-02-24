"""Triton kernel source analysis for pointwise heuristics.

Parses the generated Triton kernel source to extract metadata that the
scoring model needs but cannot derive from the problem shape alone:
tensor argument counts, instruction-mix breakdown, broadcast patterns,
and masking usage.  All parsing is done with simple string operations
on the already-generated source; no compilation is needed.
"""

import re
from typing import Dict


def extract_kernel_metadata(kernel_code: str) -> Dict:
    """Parse Triton kernel source and return metadata for the heuristics.

    The returned dict supplements problem_metadata in analyze_bottleneck()
    and estimate_compute_time_us().  If parsing fails the function returns
    safe defaults rather than raising, so callers never need to guard.

    Keys in the returned dict:
        num_tensors        -- total pointer arguments in the kernel signature
        num_inputs         -- load sites (tl.load count)
        num_outputs        -- store sites (tl.store count)
        bytes_per_element  -- num_tensors * 4 bytes (assumes FP32)
        ops_per_element    -- weighted instruction count (fast=1, med=10, slow=30)
        fast_ops           -- add / sub / mul / fma instructions
        medium_ops         -- div / sqrt / rsqrt instructions
        slow_ops           -- transcendentals: exp / log / sin / cos / tanh / sigmoid
        has_broadcast      -- True when index-usage asymmetry suggests a broadcast input
        has_mask           -- True when predicated loads or stores are present
    """
    metadata: Dict = {
        'num_tensors':       3,
        'num_inputs':        2,
        'num_outputs':       1,
        'bytes_per_element': 12.0,
        'ops_per_element':   2,
        'fast_ops':          2,
        'medium_ops':        0,
        'slow_ops':          0,
        'has_broadcast':     False,
        'has_mask':          False,
    }

    if not kernel_code:
        return metadata

    try:
        # --- Tensor argument counts ---
        # Pointer parameters in the function signature correspond to tensors.
        sig_match = re.search(r'def\s+\w+\s*\((.*?)\):', kernel_code, re.DOTALL)
        if sig_match:
            params = sig_match.group(1)
            ptr_params = [p.strip() for p in params.split(',') if '_ptr' in p]
            metadata['num_tensors'] = len(ptr_params)

            num_stores = kernel_code.count('tl.store')
            metadata['num_outputs'] = max(1, num_stores)
            metadata['num_inputs']  = metadata['num_tensors'] - metadata['num_outputs']

            # Bytes per element = all pointer args * 4 bytes (FP32 assumed).
            metadata['bytes_per_element'] = float(metadata['num_tensors'] * 4)

        # --- Instruction mix ---
        # Python arithmetic operators (+, -, *) are counted alongside explicit
        # tl.* calls because Triton emits them as element-wise tile operations.
        fast_ops = (
            kernel_code.count('tl.add') +
            kernel_code.count('tl.sub') +
            kernel_code.count('tl.mul') +
            kernel_code.count('tl.fma') +
            len(re.findall(r'\s\+\s', kernel_code)) +
            len(re.findall(r'\s-\s',  kernel_code)) +
            len(re.findall(r'\s\*\s', kernel_code))
        )

        medium_ops = (
            kernel_code.count('tl.div')   +
            kernel_code.count('tl.fdiv')  +
            kernel_code.count('tl.sqrt')  +
            kernel_code.count('tl.rsqrt') +
            len(re.findall(r'\s/\s', kernel_code))
        )

        # Transcendentals route through the Special Function Unit (SFU) and cost
        # 16–64 cycles each versus 1 cycle for FMA, so they dominate compute time.
        slow_ops = (
            kernel_code.count('tl.exp')     +
            kernel_code.count('tl.exp2')    +
            kernel_code.count('tl.log')     +
            kernel_code.count('tl.log2')    +
            kernel_code.count('tl.sin')     +
            kernel_code.count('tl.cos')     +
            kernel_code.count('tl.tanh')    +
            kernel_code.count('tl.sigmoid')
        )

        metadata['fast_ops']   = fast_ops
        metadata['medium_ops'] = medium_ops
        metadata['slow_ops']   = slow_ops

        # Weighted op count: 1 / 10 / 30 for fast / medium / slow.
        # This feeds ops_per_element in the compute-time estimate.
        total_weighted = fast_ops * 1 + medium_ops * 10 + slow_ops * 30
        metadata['ops_per_element'] = max(2, total_weighted) if total_weighted > 0 else 2

        # --- Broadcast detection ---
        # When one index (xindex vs yindex) is referenced far more than the
        # other, one tensor is likely broadcast along the less-used dimension.
        if 'xindex' in kernel_code and 'yindex' in kernel_code:
            x_count = kernel_code.count('xindex')
            y_count = kernel_code.count('yindex')
            if x_count > 0 and y_count > 0:
                ratio = max(x_count, y_count) / min(x_count, y_count)
                if ratio > 3:
                    metadata['has_broadcast'] = True

        # A 1-D kernel with more loads than pointer args also suggests a broadcast
        # (the same base pointer is loaded at multiple offsets).
        if (kernel_code.count('xindex < xnumel') > 0
                and 'yindex' not in kernel_code
                and kernel_code.count('tl.load') > metadata['num_tensors']):
            metadata['has_broadcast'] = True

        # --- Masking detection ---
        # Predicated stores break write-combining, reducing effective HBM
        # streaming efficiency from ~80 % to ~65 % of peak bandwidth.
        if ('tl.load' in kernel_code and 'mask=' in kernel_code
                or 'xmask' in kernel_code or 'ymask' in kernel_code):
            metadata['has_mask'] = True

    except Exception:
        pass  # Safe defaults already populated above

    return metadata


def get_instruction_mix_efficiency(metadata: Dict) -> float:
    """Compute an efficiency scalar from the instruction-mix breakdown.

    Fast ops execute at FMA throughput (~90% pipeline efficiency).
    Medium ops (div/sqrt) introduce pipeline bubbles (~70%).
    Slow transcendentals block the FMA pipeline while the SFU is busy (~60%).

    Returns a value in [0.5, 0.9].
    """
    fast_ops   = metadata.get('fast_ops',   0)
    medium_ops = metadata.get('medium_ops', 0)
    slow_ops   = metadata.get('slow_ops',   0)

    total = fast_ops + medium_ops + slow_ops
    if total == 0:
        return 0.7

    efficiency = (
        (fast_ops   / total) * 0.9 +
        (medium_ops / total) * 0.7 +
        (slow_ops   / total) * 0.6
    )
    return max(0.5, min(0.9, efficiency))

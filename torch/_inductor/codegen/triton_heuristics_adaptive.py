"""
Adaptive Bottleneck Analysis for Pointwise Kernel Heuristics

Instead of hardcoded size thresholds, we calculate:
1. Overhead time (kernel launch + grid scheduling)
2. Memory time (data transfer bandwidth)
3. Compute time (arithmetic operations)

Then adaptively weight scoring factors based on the actual bottleneck.
"""

from typing import Dict, Tuple
import math


class BottleneckAnalysis:
    """Analyze kernel bottlenecks to adaptively weight heuristic factors."""
    
    # Hardware constants (from architecture detection)
    KERNEL_LAUNCH_US = 3.0           # Kernel dispatch overhead
    BLOCK_DISPATCH_US = 0.02         # Per-block scheduling overhead (~20ns)
    MEMORY_BANDWIDTH_GB_S = 3500.0   # HBM bandwidth (e.g., MI350: 3.5 TB/s)
    COMPUTE_TFLOPS = 1300.0          # Peak compute (e.g., MI350: 1.3 PFLOPS FP32)
    L1_CACHE_SIZE = 32 * 1024        # 32 KB per CU
    L2_CACHE_SIZE = 4 * 1024 * 1024  # 4 MB per XCD
    
    @staticmethod
    def estimate_overhead_time_us(num_blocks: int) -> float:
        """
        Estimate total overhead time in microseconds.
        
        Includes:
        - Kernel launch/dispatch: ~3μs (fixed)
        - Grid setup overhead: scales with log(num_blocks) for large grids
        
        Once launched, blocks execute in parallel, so no per-block overhead.
        However, very large grids (>1000 blocks) have some setup cost.
        
        Returns:
            Overhead time in microseconds
        """
        launch_time = BottleneckAnalysis.KERNEL_LAUNCH_US
        
        # Grid setup scales logarithmically for large grids
        if num_blocks > 1000:
            grid_overhead = 0.5 * math.log2(num_blocks / 1000)
        else:
            grid_overhead = 0.0
        
        return launch_time + grid_overhead
    
    @staticmethod
    def estimate_memory_time_us(total_bytes: int, problem_metadata: Dict) -> float:
        """
        Estimate memory transfer time in microseconds.
        
        Considers:
        - L1 cache: 32 KB (essentially free)
        - L2 cache: 4 MB (very fast)
        - HBM: Full bandwidth cost
        
        Returns:
            Memory transfer time in microseconds
        """
        # Check if data fits in cache
        if total_bytes <= BottleneckAnalysis.L1_CACHE_SIZE:
            # L1 hit: ~4 cycles @ 2GHz = ~0.002μs (essentially free)
            # But still need to move data, scale with size
            l1_fraction = total_bytes / BottleneckAnalysis.L1_CACHE_SIZE
            return 0.01 + 0.05 * l1_fraction
        elif total_bytes <= BottleneckAnalysis.L2_CACHE_SIZE:
            # L2 hit: ~40-100 cycles, ~0.02-0.05μs per access
            # L2 bandwidth: ~1000 GB/s (much lower than HBM)
            l2_bandwidth_gb_s = 1000.0
            l2_bandwidth_bytes_per_us = l2_bandwidth_gb_s * 1e3
            return total_bytes / l2_bandwidth_bytes_per_us
        else:
            # HBM access: bandwidth-limited
            # Convert GB/s to μs: time = bytes / (GB_s * 1e9) * 1e6
            # Achievable bandwidth is typically 70-90% of peak
            effective_bandwidth_gb_s = BottleneckAnalysis.MEMORY_BANDWIDTH_GB_S * 0.8
            bandwidth_bytes_per_us = effective_bandwidth_gb_s * 1e3
            return total_bytes / bandwidth_bytes_per_us
    
    @staticmethod
    def estimate_compute_time_us(num_ops: int, threads_per_block: int, num_blocks: int) -> float:
        """
        Estimate compute time in microseconds.
        
        For pointwise kernels:
        - Simple ops (add/mul): 1-2 cycles
        - Complex ops (div/exp): 10-20 cycles
        - GPU can sustain ~1000 GFLOPS for simple ops when memory-fed
        
        Returns:
            Compute time in microseconds
        """
        # Achievable GFLOPS for simple pointwise: ~1000 GFLOPS = 1M ops/μs
        # This is much lower than peak (1300 TFLOPS) because memory-bound
        gflops_achievable = 1000.0  # GFLOPS
        ops_per_us = gflops_achievable * 1000  # Convert to ops/μs
        
        return num_ops / ops_per_us
    
    @staticmethod
    def analyze_bottleneck(config: Dict, problem_metadata: Dict) -> Dict[str, float]:
        """
        Analyze kernel bottleneck and return time breakdown.
        
        Returns:
            Dict with:
            - 'overhead_us': Overhead time
            - 'memory_us': Memory transfer time
            - 'compute_us': Compute time
            - 'total_us': Total estimated time
            - 'overhead_frac': Overhead as fraction of total
            - 'memory_frac': Memory as fraction of total
            - 'compute_frac': Compute as fraction of total
            - 'bottleneck': 'overhead', 'memory', or 'compute'
        """
        # Extract config parameters
        from torch._inductor.codegen.triton_heuristics_pointwise import PointwiseHeuristics
        
        block_dims = PointwiseHeuristics.get_block_dimensions(config)
        problem_dims = PointwiseHeuristics.get_problem_dimensions(problem_metadata)
        threads_per_block = PointwiseHeuristics.prod(block_dims)
        
        try:
            grid_size = PointwiseHeuristics.calculate_grid_size(problem_dims, block_dims)
            num_blocks = PointwiseHeuristics.prod(grid_size)
        except:
            num_blocks = 1
        
        total_elements = problem_metadata.get('total_elements', 1)
        element_size = problem_metadata.get('element_size', 4)  # bytes
        num_inputs = problem_metadata.get('num_inputs', 2)
        num_outputs = problem_metadata.get('num_outputs', 1)
        ops_per_element = problem_metadata.get('ops_per_element', 2)  # e.g., add=1, mul=1
        
        # Calculate times
        overhead_us = BottleneckAnalysis.estimate_overhead_time_us(num_blocks)
        
        # Memory: read all inputs + write all outputs
        total_bytes = total_elements * element_size * (num_inputs + num_outputs)
        memory_us = BottleneckAnalysis.estimate_memory_time_us(total_bytes, problem_metadata)
        
        # Compute: total FLOPs
        num_ops = total_elements * ops_per_element
        compute_us = BottleneckAnalysis.estimate_compute_time_us(
            num_ops, threads_per_block, num_blocks
        )
        
        # Total time
        total_us = overhead_us + memory_us + compute_us
        
        # Fractions
        overhead_frac = overhead_us / total_us if total_us > 0 else 0
        memory_frac = memory_us / total_us if total_us > 0 else 0
        compute_frac = compute_us / total_us if total_us > 0 else 0
        
        # Determine bottleneck (what takes >40% of time)
        if overhead_frac > 0.4:
            bottleneck = 'overhead'
        elif memory_frac > 0.4:
            bottleneck = 'memory'
        elif compute_frac > 0.4:
            bottleneck = 'compute'
        else:
            # Mixed - use largest component
            bottleneck = max(
                [('overhead', overhead_frac), ('memory', memory_frac), ('compute', compute_frac)],
                key=lambda x: x[1]
            )[0]
        
        return {
            'overhead_us': overhead_us,
            'memory_us': memory_us,
            'compute_us': compute_us,
            'total_us': total_us,
            'overhead_frac': overhead_frac,
            'memory_frac': memory_frac,
            'compute_frac': compute_frac,
            'bottleneck': bottleneck,
        }
    
    @staticmethod
    def get_adaptive_weights(config: Dict, problem_metadata: Dict) -> Dict[str, float]:
        """
        Get adaptive factor weights based on bottleneck analysis.
        
        Returns:
            Dict with weights for: 'bandwidth', 'launch', 'grid', 'occupancy'
        """
        analysis = BottleneckAnalysis.analyze_bottleneck(config, problem_metadata)
        
        bottleneck = analysis['bottleneck']
        overhead_frac = analysis['overhead_frac']
        memory_frac = analysis['memory_frac']
        compute_frac = analysis['compute_frac']
        
        # Base weights (normalized to sum to 1.0)
        if bottleneck == 'overhead':
            # Overhead-dominated (tiny kernels)
            # Minimize launch overhead and block count
            weights = {
                'launch': 0.50,      # Launch overhead most critical
                'grid': 0.30,        # Grid overhead (per-block) critical
                'bandwidth': 0.10,   # Memory less important (cached)
                'occupancy': 0.10,   # Latency hiding less important
            }
            
        elif bottleneck == 'memory':
            # Memory-dominated (typical pointwise)
            # Maximize bandwidth utilization and latency hiding
            weights = {
                'bandwidth': 0.40,   # Memory bandwidth most critical
                'launch': 0.25,      # Launch overhead still matters
                'grid': 0.20,        # GPU saturation important
                'occupancy': 0.15,   # Latency hiding important
            }
            
        elif bottleneck == 'compute':
            # Compute-dominated (heavy ops like exp, div)
            # Maximize occupancy and GPU saturation
            weights = {
                'occupancy': 0.35,   # Maximize parallel compute
                'grid': 0.30,        # GPU saturation critical
                'bandwidth': 0.20,   # Memory less critical
                'launch': 0.15,      # Launch overhead less important
            }
        
        else:
            # Balanced (no clear bottleneck)
            weights = {
                'bandwidth': 0.30,
                'launch': 0.30,
                'grid': 0.25,
                'occupancy': 0.15,
            }
        
        # Fine-tune based on actual fractions
        # If overhead is significant (>20%) even if not dominant, boost launch weight
        if overhead_frac > 0.2 and bottleneck != 'overhead':
            boost = min(0.15, overhead_frac - 0.2)  # Up to +15%
            weights['launch'] += boost
            # Reduce others proportionally
            other_total = 1.0 - weights['launch']
            for k in ['bandwidth', 'grid', 'occupancy']:
                weights[k] *= (1.0 - weights['launch']) / other_total
        
        # Normalize to ensure sum = 1.0
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    @staticmethod
    def get_adaptive_exponents(weights: Dict[str, float]) -> Dict[str, float]:
        """
        Convert normalized weights (0-1, sum=1) to exponents for geometric mean.
        
        We use weighted geometric mean:
        score = (bandwidth^a * launch^b * grid^c * occupancy^d)^(1/(a+b+c+d))
        
        To get desired percentage weights:
        - Map weight to exponent range [0.5, 3.0]
        - Higher weight -> higher exponent
        
        Returns:
            Dict with exponents for: 'bandwidth', 'launch', 'grid', 'occupancy'
        """
        # Map weights (0-1) to exponents (0.5-3.0)
        # weight=0.10 -> exp=0.5
        # weight=0.50 -> exp=3.0
        # Linear interpolation
        exponents = {}
        for factor, weight in weights.items():
            # Map [0.10, 0.50] -> [0.5, 3.0]
            exp = 0.5 + (weight - 0.10) / 0.40 * 2.5
            exp = max(0.5, min(3.0, exp))  # Clamp
            exponents[factor] = exp
        
        return exponents


# Example usage
if __name__ == "__main__":
    # Test case 1: Tiny kernel (512 elements)
    config_tiny = {'XBLOCK': 512, 'num_warps': 1}
    problem_tiny = {
        'dimensions': (512,),
        'total_elements': 512,
        'element_size': 4,
        'num_inputs': 2,
        'num_outputs': 1,
        'ops_per_element': 2,
        'warp_size': 64,
    }
    
    print("="*80)
    print("TINY KERNEL (512 elements)")
    print("="*80)
    analysis = BottleneckAnalysis.analyze_bottleneck(config_tiny, problem_tiny)
    print(f"Overhead: {analysis['overhead_us']:.3f}μs ({analysis['overhead_frac']*100:.1f}%)")
    print(f"Memory:   {analysis['memory_us']:.3f}μs ({analysis['memory_frac']*100:.1f}%)")
    print(f"Compute:  {analysis['compute_us']:.3f}μs ({analysis['compute_frac']*100:.1f}%)")
    print(f"Total:    {analysis['total_us']:.3f}μs")
    print(f"Bottleneck: {analysis['bottleneck'].upper()}")
    print()
    weights = BottleneckAnalysis.get_adaptive_weights(config_tiny, problem_tiny)
    print("Adaptive Weights:")
    for factor, weight in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {factor:12s}: {weight:.1%}")
    print()
    
    # Test case 2: Large kernel (1M elements)
    config_large = {'XBLOCK': 256, 'num_warps': 4}
    problem_large = {
        'dimensions': (1048576,),
        'total_elements': 1048576,
        'element_size': 4,
        'num_inputs': 2,
        'num_outputs': 1,
        'ops_per_element': 2,
        'warp_size': 64,
    }
    
    print("="*80)
    print("LARGE KERNEL (1M elements)")
    print("="*80)
    analysis = BottleneckAnalysis.analyze_bottleneck(config_large, problem_large)
    print(f"Overhead: {analysis['overhead_us']:.3f}μs ({analysis['overhead_frac']*100:.1f}%)")
    print(f"Memory:   {analysis['memory_us']:.3f}μs ({analysis['memory_frac']*100:.1f}%)")
    print(f"Compute:  {analysis['compute_us']:.3f}μs ({analysis['compute_frac']*100:.1f}%)")
    print(f"Total:    {analysis['total_us']:.3f}μs")
    print(f"Bottleneck: {analysis['bottleneck'].upper()}")
    print()
    weights = BottleneckAnalysis.get_adaptive_weights(config_large, problem_large)
    print("Adaptive Weights:")
    for factor, weight in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {factor:12s}: {weight:.1%}")
    print()
    
    # Test case 3: Medium kernel (64K elements)
    config_medium = {'XBLOCK': 256, 'num_warps': 4}
    problem_medium = {
        'dimensions': (65536,),
        'total_elements': 65536,
        'element_size': 4,
        'num_inputs': 2,
        'num_outputs': 1,
        'ops_per_element': 2,
        'warp_size': 64,
    }
    
    print("="*80)
    print("MEDIUM KERNEL (64K elements)")
    print("="*80)
    analysis = BottleneckAnalysis.analyze_bottleneck(config_medium, problem_medium)
    print(f"Overhead: {analysis['overhead_us']:.3f}μs ({analysis['overhead_frac']*100:.1f}%)")
    print(f"Memory:   {analysis['memory_us']:.3f}μs ({analysis['memory_frac']*100:.1f}%)")
    print(f"Compute:  {analysis['compute_us']:.3f}μs ({analysis['compute_frac']*100:.1f}%)")
    print(f"Total:    {analysis['total_us']:.3f}μs")
    print(f"Bottleneck: {analysis['bottleneck'].upper()}")
    print()
    weights = BottleneckAnalysis.get_adaptive_weights(config_medium, problem_medium)
    print("Adaptive Weights:")
    for factor, weight in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {factor:12s}: {weight:.1%}")


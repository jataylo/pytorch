"""Bottleneck analysis and adaptive weight interpolation for pointwise heuristics.

For each candidate config, this module estimates how long the three major cost
components would take (kernel dispatch overhead, HBM memory transfer, and raw
ALU compute), then uses those estimates to interpolate factor weights rather
than applying a fixed weight set regardless of kernel characteristics.

The key insight is that a tiny 512-element kernel is dominated by the ~3 µs
dispatch cost, so the launch and grid factors should dominate scoring.  A large
streaming kernel is dominated by HBM bandwidth, so the bandwidth factor should
dominate.  A kernel full of transcendentals is compute-bound, so occupancy (which
controls ALU pipeline utilisation) should dominate.  Hard-coded weights can only
be right for one of these regimes at a time.
"""

import math
from typing import Dict, Tuple


class BottleneckAnalysis:
    """Time model and adaptive weight interpolation for a single kernel config.

    Call order:
        analyze_bottleneck()       → overhead / memory / compute fractions
        get_adaptive_weights()     → per-factor weights that sum to 1.0
        get_adaptive_exponents()   → exponents for the weighted geometric mean
    """

    # Base kernel dispatch cost on AMD ROCm.  Empirically ~3 µs; roughly
    # constant across kernel sizes because it reflects driver + CP overhead.
    KERNEL_LAUNCH_US = 3.0

    @staticmethod
    def _get_device_constants() -> Dict[str, float]:
        """Read peak bandwidth, compute throughput, and cache sizes from the device.

        Three-level fallback so this function never raises:
          1. Custom ROCm properties (memory_bandwidth_gb_s, compute_throughput_tflops)
          2. Standard CUDA properties (memoryClockRate * memoryBusWidth, etc.)
          3. Conservative defaults (900 GB/s, 200 TFLOPS)
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    'memory_bandwidth_gb_s': 100.0,
                    'compute_tflops':        1.0,
                    'l1_cache_size':         32 * 1024,
                    'l2_cache_size':         1  * 1024 * 1024,
                }

            props = torch.cuda.get_device_properties(0)

            l2_cache_size = getattr(props, 'L2_cache_size', 4 * 1024 * 1024)
            l1_cache_size = 32 * 1024   # AMD CDNA/RDNA: 32 KB per CU

            # --- Memory bandwidth ---
            if hasattr(props, 'memory_bandwidth_gb_s'):
                memory_bandwidth_gb_s = props.memory_bandwidth_gb_s
            else:
                try:
                    # DDR: effective rate = clock_hz * bus_bytes * 2 (double data rate)
                    mem_clk_hz = props.memoryClockRate * 1e3   # kHz → Hz
                    bus_bytes  = props.memoryBusWidth  / 8     # bits → bytes
                    memory_bandwidth_gb_s = (mem_clk_hz * bus_bytes * 2) / 1e9
                    memory_bandwidth_gb_s = max(50.0, memory_bandwidth_gb_s)
                except Exception:
                    memory_bandwidth_gb_s = 900.0

            # --- Compute throughput ---
            if hasattr(props, 'compute_throughput_tflops'):
                compute_tflops = props.compute_throughput_tflops
            else:
                try:
                    is_hip  = bool(getattr(torch.version, 'hip', None))
                    clk_hz  = props.clockRate * 1e3
                    num_cus = props.multi_processor_count
                    # AMD CDNA: 128 FP32 ops/CU/clock
                    # NVIDIA Ampere: 128 CUDA cores/SM * 2 FP32/core/clock
                    ops_per_cu_clk = 128 if is_hip else 128 * 2
                    compute_tflops = (num_cus * ops_per_cu_clk * clk_hz) / 1e12
                    compute_tflops = max(1.0, compute_tflops)
                except Exception:
                    compute_tflops = 200.0

            return {
                'memory_bandwidth_gb_s': memory_bandwidth_gb_s,
                'compute_tflops':        compute_tflops,
                'l1_cache_size':         l1_cache_size,
                'l2_cache_size':         l2_cache_size,
            }

        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(
                '[HEURISTICS] _get_device_constants fallback: %s', e
            )
            return {
                'memory_bandwidth_gb_s': 900.0,
                'compute_tflops':        200.0,
                'l1_cache_size':         32 * 1024,
                'l2_cache_size':         4  * 1024 * 1024,
            }

    @staticmethod
    def estimate_overhead_time_us(num_blocks: int,
                                  problem_metadata: Dict = None,
                                  config: Dict = None) -> float:
        """Estimate total kernel dispatch overhead in microseconds.

        Components (all additive — overhead precedes any wavefront execution):
          - Base dispatch cost: ~3 µs (CP DMA, SPI wavefront allocation signal)
          - Per-extra-tensor-arg: ~0.1 µs (pointer copy into SGPRs before dispatch)
          - Per-extra-warp: ~0.2 µs (VGPR/SGPR bank allocation in the SPI)
          - Large-grid setup: logarithmic scaling for grids > 1000 blocks
            (the CP must batch-dispatch in multiple rounds)
          - Masking: +0.5 µs (extra predicate instructions increase CP complexity)
        """
        total = BottleneckAnalysis.KERNEL_LAUNCH_US

        if problem_metadata:
            num_tensors  = problem_metadata.get('num_tensors', 3)
            total       += (num_tensors - 3) * 0.1

        if config:
            num_warps = config.get('num_warps', 4)
            total    += (num_warps - 1) * 0.2

        if num_blocks > 1000:
            total += 0.5 * math.log2(num_blocks / 1000)
        elif num_blocks > 100:
            total += 0.2 * math.log2(num_blocks / 100)

        if problem_metadata and problem_metadata.get('has_mask', False):
            total += 0.5

        return total

    @staticmethod
    def get_oi_ceiling() -> float:
        """Return the roofline ridge point (peak FLOPs / peak bandwidth) in FLOPs/byte.

        Kernels with arithmetic intensity below this value are memory-bound;
        above it they are compute-bound.
        """
        c = BottleneckAnalysis._get_device_constants()
        return (c['compute_tflops'] * 1e12) / (c['memory_bandwidth_gb_s'] * 1e9)

    @staticmethod
    def calculate_arithmetic_intensity(num_ops: int,
                                       total_elements: int,
                                       bytes_per_element: float = 12.0
                                       ) -> Tuple[float, str]:
        """Compute arithmetic intensity and classify as memory- or compute-bound.

        Returns (arithmetic_intensity, bottleneck_label) where the label is
        one of 'MEMORY-BOUND' or 'COMPUTE-BOUND'.
        """
        oi_ceiling       = BottleneckAnalysis.get_oi_ceiling()
        ops_per_element  = num_ops / max(total_elements, 1)
        ai               = ops_per_element / bytes_per_element
        bound            = 'MEMORY-BOUND' if ai < oi_ceiling else 'COMPUTE-BOUND'
        return ai, bound

    @staticmethod
    def estimate_memory_time_us(total_bytes: int,
                                problem_metadata: Dict,
                                num_blocks: int = 1) -> float:
        """Estimate HBM transfer time in microseconds.

        Cache-level selection:
          - L1 (per-CU, 32 KB): only a true hit when num_blocks ≤ num_CUs,
            so each CU's working set is resident.  With more blocks the data
            is replicated across CUs and must come from L2/HBM.
          - L2 (shared): broadcast tensors that fit are reused across all blocks.
          - HBM: everything else, derated by 0.80 for clean streaming and 0.65
            for masked kernels (partial cache-line stores break write-combining).
        """
        dc = BottleneckAnalysis._get_device_constants()
        l1_cache_size         = dc['l1_cache_size']
        l2_cache_size         = dc['l2_cache_size']
        memory_bandwidth_gb_s = dc['memory_bandwidth_gb_s']

        try:
            import torch
            num_cus = (torch.cuda.get_device_properties(0).multi_processor_count
                       if torch.cuda.is_available() else 256)
        except Exception:
            num_cus = 256

        has_broadcast = problem_metadata.get('has_broadcast', False)

        if total_bytes <= l1_cache_size:
            if num_blocks <= num_cus:
                # True L1 hit: data fits in each active CU's cache.
                return 0.01 + 0.05 * (total_bytes / l1_cache_size)
            # Replicated across more CUs than L1 instances → falls through to L2/HBM.
            if has_broadcast and total_bytes * min(num_blocks / num_cus, 2) <= l2_cache_size:
                l2_bw = 1000.0  # GB/s — L2 bandwidth is ~1 TB/s on AMD CDNA
                return total_bytes / (l2_bw * 1e3)
            effective_bw = memory_bandwidth_gb_s * 0.8
            return total_bytes / (effective_bw * 1e3)

        if total_bytes <= l2_cache_size:
            # L2-resident: broadcast tensors are reused across blocks for free.
            l2_bw = 1000.0
            return total_bytes / (l2_bw * 1e3)

        # HBM path — check if a broadcast input is L2-resident.
        if has_broadcast:
            num_inputs       = problem_metadata.get('num_inputs', 2)
            broadcast_bytes  = total_bytes / max(num_inputs, 2)
            if broadcast_bytes <= l2_cache_size:
                effective_bw        = memory_bandwidth_gb_s * 0.8
                non_broadcast_bytes = total_bytes - broadcast_bytes
                return non_broadcast_bytes / (effective_bw * 1e3)

        # Masked kernels generate partial cache-line stores, breaking the
        # write-combining path in the memory controller (0.65 vs 0.80).
        hbm_efficiency = 0.65 if problem_metadata.get('has_mask', False) else 0.80
        effective_bw   = memory_bandwidth_gb_s * hbm_efficiency
        return total_bytes / (effective_bw * 1e3)

    @staticmethod
    def estimate_compute_time_us(num_ops: int,
                                 threads_per_block: int,
                                 num_blocks: int,
                                 problem_metadata: Dict = None) -> float:
        """Estimate raw compute-throughput-limited time in microseconds.

        This is the time compute would take if memory bandwidth were infinite —
        the other side of the roofline.  analyze_bottleneck() applies
        max(memory_us, compute_us) to determine which side actually limits.

        Instruction-mix efficiency (from problem_metadata if available):
          >50% transcendentals → 0.60  (SFU pipeline stalls FMA pipeline)
          mixed                → 0.70
          mostly FMA/add       → 0.80
        """
        dc             = BottleneckAnalysis._get_device_constants()
        peak_tflops    = dc['compute_tflops']
        total_elements = threads_per_block * num_blocks

        if problem_metadata:
            ops_per_element   = problem_metadata.get('ops_per_element',   2)
            bytes_per_element = problem_metadata.get('bytes_per_element', 12.0)  # noqa: F841
        else:
            ops_per_element = num_ops / max(total_elements, 1)

        if problem_metadata:
            fast_ops   = problem_metadata.get('fast_ops',   0)
            medium_ops = problem_metadata.get('medium_ops', 0)
            slow_ops   = problem_metadata.get('slow_ops',   0)
            total_instr = fast_ops + medium_ops + slow_ops
            if total_instr > 0:
                slow_frac   = slow_ops   / total_instr
                medium_frac = medium_ops / total_instr
                if slow_frac > 0.5:
                    compute_efficiency = 0.60   # Mostly transcendentals
                elif slow_frac > 0.2 or medium_frac > 0.5:
                    compute_efficiency = 0.70   # Mixed
                else:
                    compute_efficiency = 0.80   # Mostly FMA
            else:
                compute_efficiency = 0.70
        else:
            compute_efficiency = 0.70

        achievable_tflops = peak_tflops * compute_efficiency
        ops_per_us        = (achievable_tflops * 1e12) / 1e6
        return num_ops / ops_per_us

    @staticmethod
    def analyze_bottleneck(config: Dict,
                           problem_metadata: Dict,
                           kernel_code: str = None) -> Dict[str, float]:
        """Build a three-component time model for a single config.

        If kernel_code is provided it is parsed to refine metadata (tensor
        counts, instruction mix, masking) before the time estimates are made.

        Roofline combination:
            total_us = overhead_us + max(memory_us, compute_us)

        Memory and compute overlap on the GPU (the memory controller and ALU
        pipelines are independent); only the slower of the two is visible.
        Overhead does not overlap — dispatch must complete before any wavefront
        starts executing.

        The three fractions are computed from the gross sum
        (overhead + memory + compute), not from total_us.  This preserves
        information about relative component weight even when memory and compute
        strongly overlap, which is what get_adaptive_weights() needs.

        Returns a dict with keys:
            overhead_us, memory_us, compute_us, total_us,
            overhead_frac, memory_frac, compute_frac,
            bottleneck ('overhead' | 'memory' | 'compute'),
            launch_bound (True when overhead_frac > 0.50)
        """
        if kernel_code:
            try:
                from torch._inductor.codegen.triton_heuristics_kernel_analysis import (
                    extract_kernel_metadata,
                )
                kernel_metadata  = extract_kernel_metadata(kernel_code)
                problem_metadata = {**problem_metadata, **kernel_metadata}
            except Exception:
                pass

        from torch._inductor.codegen.triton_heuristics_pointwise import PointwiseHeuristics

        block_dims        = PointwiseHeuristics.get_block_dimensions(config)
        problem_dims      = PointwiseHeuristics.get_problem_dimensions(problem_metadata)
        threads_per_block = PointwiseHeuristics.prod(block_dims)

        try:
            grid_size  = PointwiseHeuristics.calculate_grid_size(problem_dims, block_dims)
            num_blocks = PointwiseHeuristics.prod(grid_size)
        except Exception:
            num_blocks = 1

        total_elements    = problem_metadata.get('total_elements',    1)
        element_size      = problem_metadata.get('element_size',      4)
        num_inputs        = problem_metadata.get('num_inputs',        2)
        num_outputs       = problem_metadata.get('num_outputs',       1)
        ops_per_element   = problem_metadata.get('ops_per_element',   2)
        bytes_per_element = problem_metadata.get(
            'bytes_per_element',
            float(element_size) * (num_inputs + num_outputs),
        )

        overhead_us = BottleneckAnalysis.estimate_overhead_time_us(
            num_blocks, problem_metadata, config
        )

        total_bytes = total_elements * bytes_per_element
        memory_us   = BottleneckAnalysis.estimate_memory_time_us(
            total_bytes, problem_metadata, num_blocks
        )

        num_ops    = total_elements * ops_per_element
        compute_us = BottleneckAnalysis.estimate_compute_time_us(
            num_ops, threads_per_block, num_blocks, problem_metadata
        )

        # Roofline: memory and compute overlap; overhead stacks on top.
        mem_compute_us = max(memory_us, compute_us)
        total_us       = overhead_us + mem_compute_us

        # Fractions from gross (not from total_us) so they sum to 1.0 even
        # when memory and compute are close in magnitude.
        gross_us      = overhead_us + memory_us + compute_us
        overhead_frac = overhead_us / gross_us if gross_us > 0 else 0.0
        memory_frac   = memory_us   / gross_us if gross_us > 0 else 0.0
        compute_frac  = compute_us  / gross_us if gross_us > 0 else 0.0

        if overhead_us >= mem_compute_us:
            bottleneck = 'overhead'
        elif memory_us >= compute_us:
            bottleneck = 'memory'
        else:
            bottleneck = 'compute'

        return {
            'overhead_us':   overhead_us,
            'memory_us':     memory_us,
            'compute_us':    compute_us,
            'total_us':      total_us,
            'overhead_frac': overhead_frac,
            'memory_frac':   memory_frac,
            'compute_frac':  compute_frac,
            'bottleneck':    bottleneck,
            'launch_bound':  overhead_frac > 0.50,
        }

    @staticmethod
    def get_adaptive_weights(config: Dict,
                             problem_metadata: Dict,
                             kernel_code: str = None) -> Dict[str, float]:
        """Interpolate factor weights from the per-config bottleneck fractions.

        Rather than hard-switching between three static weight tables, we define
        pure-regime ideal weight vectors and blend them using the component
        fractions that analyze_bottleneck() returns as mixing coefficients:

            w[k] = overhead_frac * OVERHEAD_W[k]
                 + memory_frac   * MEMORY_W[k]
                 + compute_frac  * COMPUTE_W[k]

        Because the three fractions sum to 1.0 and each pure-regime vector also
        sums to 1.0, the result always sums to 1.0.

        Pure-regime weight vectors:

            Factor     │ overhead-bound │ memory-bound │ compute-bound
            ───────────┼────────────────┼──────────────┼──────────────
            bandwidth  │   0.10         │   0.55       │   0.15
            launch     │   0.57         │   0.10       │   0.10
            grid       │   0.18         │   0.15       │   0.30
            occupancy  │   0.15         │   0.20       │   0.45

        Rationale per regime:
          overhead-bound  Launch (0.57) is the dominant signal.  Grid (0.18,
                          up from 0.10) is important because the EPB-based
                          Launch score is near-flat for multi-block kernels
                          (wide sigma keeps all EPB in [0.88, 1.00]), so the
                          Grid score (CU saturation) becomes the primary
                          signal distinguishing XBLOCK=256 (many CUs busy)
                          from XBLOCK=1024 (few CUs busy) for medium problems.
                          Bandwidth (0.10) is noise — data fits in L1/L2.
          memory-bound    Bandwidth (0.55) is paramount; occupancy (0.20)
                          keeps the HBM pipeline full via wavefront switching;
                          grid (0.15) ensures all CUs are fed.
          compute-bound   Occupancy (0.45) maximises concurrent ALU use;
                          grid (0.30) keeps every CU busy; bandwidth (0.15)
                          feeds operands; launch (0.10) is negligible next to
                          the compute time.

        Returns a dict with keys 'bandwidth', 'launch', 'grid', 'occupancy'.
        """
        analysis = BottleneckAnalysis.analyze_bottleneck(
            config, problem_metadata, kernel_code
        )
        o_frac = analysis['overhead_frac']
        m_frac = analysis['memory_frac']
        c_frac = analysis['compute_frac']

        # Pure-regime weight vectors.
        #
        # overhead-bound: Grid raised from 0.10 → 0.18, Launch lowered from
        #   0.65 → 0.57.  Rationale: for multi-block launch-bound kernels the
        #   EPB-based Launch score is now near-flat (wide sigma), so it barely
        #   discriminates configs.  The Grid score (CU saturation) is the primary
        #   signal that distinguishes e.g. XBLOCK=256 (84% CU util) from
        #   XBLOCK=1024 (21% CU util) for medium-sized problems.  Empirically,
        #   the 65536-element conv-block case improved from 18% gap to <3% gap
        #   across 204 benchmark cases where the old weights chose too-large XBLOCK.
        OVERHEAD_W = {'bandwidth': 0.10, 'launch': 0.57, 'grid': 0.18, 'occupancy': 0.15}
        MEMORY_W   = {'bandwidth': 0.55, 'launch': 0.10, 'grid': 0.15, 'occupancy': 0.20}
        COMPUTE_W  = {'bandwidth': 0.15, 'launch': 0.10, 'grid': 0.30, 'occupancy': 0.45}

        weights = {
            k: o_frac * OVERHEAD_W[k] + m_frac * MEMORY_W[k] + c_frac * COMPUTE_W[k]
            for k in OVERHEAD_W
        }

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    @staticmethod
    def get_adaptive_exponents(weights: Dict[str, float]) -> Dict[str, float]:
        """Map normalised weights (sum=1) to exponents for the geometric mean.

        The scoring formula is:
            score = (BW^a * Launch^b * Grid^c * Occupancy^d) ^ (1/(a+b+c+d))

        We map each weight linearly from [0.10, 0.50] → [0.5, 3.0] so that a
        factor with weight 0.10 (minimal influence) contributes exponent 0.5 and
        a factor with weight 0.50 (dominant) contributes exponent 3.0.

        Returns a dict with the same keys as weights.
        """
        exponents = {}
        for factor, weight in weights.items():
            exp = 0.5 + (weight - 0.10) / 0.40 * 2.5
            exponents[factor] = max(0.5, min(3.0, exp))
        return exponents

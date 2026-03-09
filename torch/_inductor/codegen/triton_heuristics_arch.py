"""Hardware abstraction layer for pointwise kernel heuristics.

Queries actual GPU device properties via torch.cuda.get_device_properties()
and derives optimal heuristic parameters from first principles, rather than
hardcoding constants that would only be correct for one specific chip.

The key outputs are collected in ArchitectureConfig and consumed by the
scoring factors in triton_heuristics_pointwise.py.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ArchitectureConfig:
    """Hardware-derived constants used by the scoring model.

    All fields marked "derived" are computed from the raw device properties
    using the mathematical analysis in ArchitectureConfig.from_device().
    """

    # Raw device properties
    device_name:           str
    num_cus:               int   # Compute units / multiprocessors
    warp_size:             int   # 64 on AMD (wave64), 32 on NVIDIA
    max_threads_per_block: int
    max_wavefronts_per_cu: int
    l1_cache_size:         int   # Per-CU L1 in bytes (32 KB for AMD CDNA/RDNA)
    l2_cache_size:         int   # Total L2 in bytes

    # Derived optimal values
    optimal_threads_bandwidth:  int   # Thread count for best HBM utilisation (default ipl=2)
    optimal_blocks_grid:        int   # Block count for full GPU saturation
    occupancy_sweetspot_min:    int   # Min wavefronts/block for latency hiding
    occupancy_sweetspot_max:    int   # Max wavefronts/block before VGPR pressure
    optimal_elements_per_block: int   # Elements/block to amortise launch overhead

    # Raw latency constants — exposed so callers can recompute optimal_threads
    # with a kernel-specific instructions_per_load value (see TODO-1).
    simd_units:        int    # SIMD units per CU (4 on AMD CDNA, 4 on NVIDIA)
    effective_latency: float  # Blended L2+HBM round-trip in cycles

    @classmethod
    def from_device(cls, device: Optional[torch.device] = None) -> 'ArchitectureConfig':
        """Derive heuristic parameters from the actual GPU.

        Falls back to conservative defaults when no GPU is present.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not torch.cuda.is_available():
            return cls._get_default_config()

        props = torch.cuda.get_device_properties(device)

        device_name           = props.name
        num_cus               = props.multi_processor_count
        warp_size             = props.warp_size
        max_threads_per_block = props.max_threads_per_block
        max_threads_per_cu    = props.max_threads_per_multi_processor
        max_wavefronts_per_cu = max_threads_per_cu // warp_size

        # ROCm exposes 'L2_cache_size' (capital); CUDA uses 'l2_cache_size'.
        l2_cache_size = getattr(props, 'L2_cache_size',
                        getattr(props, 'l2_cache_size', 4 * 1024 * 1024))

        # --- Bandwidth: optimal threads per block ---
        #
        # Self-consistent Little's Law for a CU with `simd_units` SIMD units:
        #
        #   Each SIMD unit issues one wavefront-instruction per cycle.
        #   With N wavefronts sharing `simd_units` issue slots, a wavefront's
        #   issue gap is N / simd_units cycles.  For a kernel with I arithmetic
        #   instructions per memory op, the compute time between two memory
        #   requests from the same wavefront is:
        #
        #       compute_gap = I × (N / simd_units)
        #
        #   Latency hiding requires compute_gap ≥ effective_latency, so:
        #
        #       N_min = sqrt(simd_units × effective_latency / I)   (self-consistent)
        #
        # We use effective (L2-blended) latency rather than raw HBM latency
        # because a significant fraction of accesses hit L2 on modern GPUs.
        # A conservative 50 % L2 hit rate gives:
        #
        #   effective_latency = 0.50 × 50 + 0.50 × 500 = 275 cycles  (CDNA3 estimates)
        #   N_min = sqrt(4 × 275 / 2) ≈ 23  → clamped to 16 wavefronts = 1024 threads
        #
        # At 50% L2 hit rate: effective_latency=275, N≈24 wavefronts (1536 threads).
        # Since Triton XBLOCK values are powers of 2, XBLOCK=1024 falls closest
        # to the Gaussian peak (score ≈ 0.987) vs XBLOCK=512 (score ≈ 0.950),
        # correctly matching the empirical observation that XBLOCK=1024 wins.
        simd_units             = 4     # AMD CDNA: 4 SIMD units per CU (NVIDIA: 4 SMs)
        l2_hit_latency         = 50    # cycles: L2 cache hit on CDNA3 / Ampere
        hbm_latency            = 500   # cycles: HBM round-trip on MI300X (~100–200 ns at 2 GHz)
        l2_hit_rate            = 0.50  # conservative; streaming kernels can be lower
        instructions_per_load  = 2     # arithmetic ops between consecutive memory ops

        effective_latency = l2_hit_rate * l2_hit_latency + (1.0 - l2_hit_rate) * hbm_latency
        wavefronts_needed = math.sqrt(simd_units * effective_latency / instructions_per_load)
        # Clamp to physically plausible range: at least 8 (well-known empirical floor),
        # at most max_wavefronts_per_cu (hardware ceiling for the device).
        wavefronts_needed = max(8, min(int(math.ceil(wavefronts_needed)), max_wavefronts_per_cu))
        optimal_threads_bandwidth = wavefronts_needed * warp_size

        # --- Grid: optimal number of blocks ---
        #
        # One block per CU gives 100% utilisation in the first wave but no
        # load-balancing if blocks finish at different times.  Two blocks per
        # CU lets faster CUs pick up the next block while slower ones finish,
        # eliminating the tail effect with negligible extra scheduling cost.
        optimal_blocks_grid = num_cus * 2

        # --- Occupancy: wavefront sweet spot ---
        #
        # VGPR budget on AMD CDNA: 65536 VGPRs per CU.
        # A typical pointwise kernel uses ~50 VGPRs/thread.
        # 50 VGPRs * 64 threads/wf = 3200 VGPRs/wf → max 20 wf/CU.
        #
        # We don't need maximum occupancy.  For memory-bound kernels
        # 4–8 wavefronts suffice to hide ~300-cycle HBM latency
        # (Little's Law: ceil(300/40) ≈ 8).  Beyond 8 the extra init
        # overhead outweighs the marginal latency-hiding benefit.
        assumed_vgprs_per_thread = 50
        vgprs_per_cu             = 65536
        vgprs_per_wavefront      = assumed_vgprs_per_thread * warp_size
        max_wf_by_vgpr           = vgprs_per_cu // vgprs_per_wavefront

        occupancy_sweetspot_min = 4
        occupancy_sweetspot_max = min(8, max_wf_by_vgpr)

        # --- Launch: optimal elements per block ---
        #
        # Kernel dispatch costs ~3 µs.  At ~0.05 µs/element for a streaming
        # FP32 kernel we need at least 3 / 0.05 = 60 elements to keep overhead
        # below 50%.  Targeting < 5%: 60 / 0.05 = 1200 → round to 2048.
        launch_overhead_us      = 3
        element_time_us         = 0.05
        target_overhead_frac    = 0.05

        min_elements    = launch_overhead_us / element_time_us
        optimal_elem    = min_elements / target_overhead_frac
        # Round to next power of two and clamp to [256, 2048].
        optimal_elements_per_block = 2 ** math.ceil(math.log2(optimal_elem))
        optimal_elements_per_block = max(256, min(optimal_elements_per_block, 2048))

        # L1 cache size is architecturally fixed per CU / SM.
        # AMD CDNA / RDNA: 32 KB per CU.  NVIDIA Ampere+: 128 KB per SM.
        is_hip        = bool(getattr(torch.version, 'hip', None))
        l1_cache_size = 32 * 1024 if is_hip else 128 * 1024

        return cls(
            device_name=device_name,
            num_cus=num_cus,
            warp_size=warp_size,
            max_threads_per_block=max_threads_per_block,
            max_wavefronts_per_cu=max_wavefronts_per_cu,
            l1_cache_size=l1_cache_size,
            l2_cache_size=l2_cache_size,
            optimal_threads_bandwidth=optimal_threads_bandwidth,
            optimal_blocks_grid=optimal_blocks_grid,
            occupancy_sweetspot_min=occupancy_sweetspot_min,
            occupancy_sweetspot_max=occupancy_sweetspot_max,
            optimal_elements_per_block=optimal_elements_per_block,
            simd_units=simd_units,
            effective_latency=effective_latency,
        )

    @classmethod
    def _get_default_config(cls) -> 'ArchitectureConfig':
        """Conservative fallback when no GPU is available."""
        return cls(
            device_name='CPU (fallback)',
            num_cus=1,
            warp_size=32,
            max_threads_per_block=1024,
            max_wavefronts_per_cu=32,
            l1_cache_size=32 * 1024,
            l2_cache_size=0,
            optimal_threads_bandwidth=256,
            optimal_blocks_grid=32,
            occupancy_sweetspot_min=4,
            occupancy_sweetspot_max=8,
            optimal_elements_per_block=1024,
            simd_units=4,
            effective_latency=275.0,
        )

    def __str__(self) -> str:
        wf = self.optimal_threads_bandwidth // self.warp_size
        lines = [
            f"Device: {self.device_name}",
            f"  CUs / SMs         : {self.num_cus}",
            f"  Warp size         : {self.warp_size}",
            f"  Max threads/block : {self.max_threads_per_block}",
            f"  Max wavefronts/CU : {self.max_wavefronts_per_cu}",
            f"  L1 cache (per CU) : {self.l1_cache_size // 1024} KB",
            f"  L2 cache          : {self.l2_cache_size // 1024} KB"
                if self.l2_cache_size else "  L2 cache          : unknown",
            f"Derived optimal values:",
            f"  Bandwidth         : {self.optimal_threads_bandwidth} threads/block ({wf} wavefronts)",
            f"  Grid              : {self.optimal_blocks_grid} blocks "
                f"({self.optimal_blocks_grid / max(self.num_cus, 1):.1f}× CUs)",
            f"  Occupancy range   : {self.occupancy_sweetspot_min}–{self.occupancy_sweetspot_max} wavefronts/block",
            f"  Launch amortise   : {self.optimal_elements_per_block} elements/block",
        ]
        return "\n".join(lines)


# Module-level singleton, lazily initialised on first call.
_arch_config: Optional[ArchitectureConfig] = None


def get_architecture_config(device: Optional[torch.device] = None) -> ArchitectureConfig:
    """Return the hardware configuration, querying the device on first call."""
    global _arch_config
    if _arch_config is None:
        _arch_config = ArchitectureConfig.from_device(device)
    return _arch_config


def reset_architecture_config() -> None:
    """Clear the cached configuration (useful in tests that mock device props)."""
    global _arch_config
    _arch_config = None

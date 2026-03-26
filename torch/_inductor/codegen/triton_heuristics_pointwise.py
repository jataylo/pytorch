"""Scoring factors and config generation for pointwise kernel heuristics.

Provides PointwiseHeuristics, which:
  - enumerates every legal (XBLOCK[×YBLOCK[×ZBLOCK]], num_warps) combination
    for a given problem shape (generate_all_candidate_configs),
  - scores each combination on four factors and ranks them (prune_configs), and
  - exposes individual factor functions for the verbose debug table.

The four scoring factors are:
  1. Memory bandwidth utilisation — Gaussian peak at the thread count that
     saturates the HBM pipeline (derived from Little's Law).
  2. Launch overhead amortisation — Gaussian peak at the elements-per-block
     value that keeps dispatch cost below ~5 % of wall time.
  3. Grid granularity — problem-size-adaptive target block count that keeps
     all CUs busy without saturating the Command Processor queue.
  4. Occupancy — wavefront count per CU for latency hiding; regime-dependent
     (sweet-spot model for memory-bound, overhead-ratio model for tiny multi-
     block kernels where grid spread already provides latency hiding).

Weights are computed per-config by BottleneckAnalysis (adaptive.py) based on
which of the three cost components (overhead, memory, compute) dominates.
"""

import math
import operator
from functools import reduce
from typing import Dict, List, Optional, Tuple

try:
    from .triton_heuristics_arch import get_architecture_config
except ImportError:
    get_architecture_config = None  # type: ignore[assignment]

try:
    from .triton_heuristics_bottleneck import BottleneckAnalysis
except ImportError:
    BottleneckAnalysis = None  # type: ignore[assignment]

__all__ = ['PointwiseHeuristics']


class PointwiseHeuristics:
    """Static scoring and config-generation heuristics for pointwise kernels.
    
    All methods are @staticmethod or @classmethod; the class is never
    instantiated.  The class-level _arch_config cache is a process-level
    singleton (one GPU per process assumed).
    """
    
    _arch_config = None
    _is_hip: Optional[bool] = None  # lazily populated; None = not yet detected

    # Hard limits on per-dimension block size.
    MIN_BLOCK_SIZE = 1
    MAX_BLOCK_SIZE = 4096

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @classmethod
    def _get_is_hip(cls) -> bool:
        """Return True when running on a ROCm/HIP (AMD GPU) build.

        Cached after the first call.  Returns False when torch is unavailable
        or when running on a CUDA (NVIDIA) build.  The result is used to gate
        AMD-specific config dimensions such as waves_per_eu.
        """
        if cls._is_hip is None:
            try:
                import torch as _torch
                cls._is_hip = bool(_torch.version.hip)
            except Exception:
                cls._is_hip = False
        return cls._is_hip

    @staticmethod
    def _is_waves_per_eu_enabled() -> bool:
        """Return True when waves_per_eu config generation is enabled.

        Controlled by the TORCHINDUCTOR_POINTWISE_WAVES_PER_EU environment
        variable (default: 0 = disabled).  Must also be on an AMD/HIP build —
        waves_per_eu is an AMD-only Triton parameter and is ignored on CUDA.

        Enable with:
            TORCHINDUCTOR_POINTWISE_WAVES_PER_EU=1
        """
        import os as _os
        return _os.environ.get("TORCHINDUCTOR_POINTWISE_WAVES_PER_EU", "0") == "1"
    
    @classmethod
    def _get_arch(cls):
        """Return the cached ArchitectureConfig, initialising on first call.

        Normally pre-warmed by pre_fork_setup() before workers are forked.
        Falls back to conservative SimpleNamespace defaults if no GPU is
        available or if the CUDA context has not been initialised yet.
        """
        if cls._arch_config is None:
            if get_architecture_config is not None:
                try:
                    cls._arch_config = get_architecture_config()
                except Exception:
                    pass
            if cls._arch_config is None:
                from types import SimpleNamespace
                cls._arch_config = SimpleNamespace(
                    num_cus=256,
                    warp_size=64,
                    max_wavefronts_per_cu=40,
                    optimal_threads_bandwidth=512,
                    optimal_blocks_grid=512,
                    occupancy_sweetspot_min=4,
                    occupancy_sweetspot_max=8,
                    optimal_elements_per_block=1024,
                    simd_units=4,
                    effective_latency=275.0,
                )
        return cls._arch_config
    
    @staticmethod
    def prod(dims: Tuple[int, ...]) -> int:
        return reduce(operator.mul, dims, 1)
    
    @staticmethod
    def get_block_dimensions(config: Dict) -> Tuple[int, ...]:
        """Extract (XBLOCK[, YBLOCK[, ZBLOCK]]) from a config dict."""
        dims = []
        for name in ('XBLOCK', 'YBLOCK', 'ZBLOCK'):
            if name in config:
                dims.append(config[name])
            else:
                break
        return tuple(dims) if dims else (config.get('BLOCK_SIZE', 256),)
    
    @staticmethod
    def get_problem_dimensions(problem_metadata: Dict) -> Tuple[int, ...]:
        if 'dimensions' in problem_metadata:
            return tuple(problem_metadata['dimensions'])
        return (problem_metadata.get('total_elements', 1),)
    
    @staticmethod
    def calculate_grid_size(problem_dims: Tuple[int, ...],
                            block_dims:   Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(problem_dims) == len(block_dims), (
            f'Dimension mismatch: problem={problem_dims}, block={block_dims}'
        )
        return tuple(
            (p + b - 1) // b
            for p, b in zip(problem_dims, block_dims)
        )

    # -------------------------------------------------------------------------
    # Factor 1 — Memory bandwidth utilisation
    # -------------------------------------------------------------------------
    
    @staticmethod
    def estimate_memory_bandwidth(config: Dict, problem_metadata: Dict) -> float:
        """Score how well this config utilises the HBM pipeline.

        Modelled as a Gaussian centred at optimal_threads with σ = 1.5 ×
        optimal_threads.  The wider sigma keeps XBLOCK=512 and XBLOCK=256
        competitive (gap vs XBLOCK=1024 shrinks from 6 % to 3 %), preventing
        the top-N pool from being flooded by XBLOCK=1024 variants.

        optimal_threads is recomputed per-kernel from the Little's Law formula:

            N_min = sqrt(simd_units × effective_latency / instructions_per_load)

        where instructions_per_load is the ratio of arithmetic ops to load ops,
        derived from the instruction mix in problem_metadata when available.
        Compute-heavy kernels (high ipl) need fewer wavefronts; memory-thin
        kernels (low ipl) need more.  Falls back to arch.optimal_threads_bandwidth
        (derived with ipl=2) when the instruction mix is absent.

        On MI300X with ipl=2: optimal_threads ≈ 1536 (24 wavefronts).

        Score is in [0.60, 1.00]:
          - floor 0.75 for threads_per_block ≥ 64  (Gaussian decay)
          - hard floor 0.60 for threads_per_block < 64  (< one wavefront)

        For 2-D/3-D kernels a cache-line-utilisation correction is applied on
        top.  On AMD (warp_size=64), the X access within each row is always
        contiguous regardless of XBLOCK, so XBLOCK=32 is NOT penalised for
        "spanning two rows" — on HBM systems those two rows can be served by
        different memory channels simultaneously (measured: XBLOCK=32 is up to
        15 % faster than XBLOCK=64 on MI300X for large 2-D tensors).  Only
        XBLOCK < 16 (sub-cache-line) is penalised, because the row access then
        wastes part of every 64-byte cache line:
            XBLOCK ≥ 16  →  factor 1.00  (full cache line per row; no penalty)
            XBLOCK <  16 →  factor 0.75–0.98  (partial cache-line utilisation)
        """
        arch      = PointwiseHeuristics._get_arch()
        warp_size = problem_metadata.get('warp_size', arch.warp_size
                                         if hasattr(arch, 'warp_size') else 64)

        # Actual hardware thread count: num_warps × warp_size.
        # Do NOT use prod(block_dims): for 1-D kernels that equals XBLOCK (tile
        # size), which can legitimately exceed 1024 while the real thread count
        # stays within hardware limits.
        threads_per_block = config.get('num_warps', 1) * warp_size

        # Derive instructions_per_load from the instruction mix when available.
        # Shifts the Gaussian peak per-kernel: compute-heavy kernels (high ipl)
        # need fewer wavefronts; memory-thin kernels (low ipl) need more.
        fast_ops   = problem_metadata.get('fast_ops',   0)
        medium_ops = problem_metadata.get('medium_ops', 0)
        slow_ops   = problem_metadata.get('slow_ops',   0)
        load_ops   = problem_metadata.get('load_ops',   0)
        total_arith = fast_ops + medium_ops + slow_ops
        if load_ops > 0 and total_arith > 0:
            # Weighted arithmetic count normalised to FP32-FMA equivalents.
            weighted_arith  = fast_ops + medium_ops * 4 + slow_ops * 32
            instructions_per_load = max(1, weighted_arith / load_ops)
        else:
            instructions_per_load = None  # fall back to arch default

        if (instructions_per_load is not None
                and hasattr(arch, 'simd_units')
                and hasattr(arch, 'effective_latency')):
            wavefronts_needed = math.sqrt(
                arch.simd_units * arch.effective_latency / instructions_per_load
            )
            max_wf = arch.max_wavefronts_per_cu if hasattr(arch, 'max_wavefronts_per_cu') else 40
            wavefronts_needed = max(8, min(int(math.ceil(wavefronts_needed)), max_wf))
            optimal_threads = wavefronts_needed * warp_size
        else:
            optimal_threads = arch.optimal_threads_bandwidth

        xblock = config.get('XBLOCK', 256)
        yblock = config.get('YBLOCK', 0)

        # ── Well-saturated-grid early exit ─────────────────────────────────────
        # When there are already ≥ 2048 blocks (≈ 8 per CU on 256-CU AMD), the
        # CU scheduler hides HBM latency entirely through block-switching.
        # Adding more warps per block (nw=8 vs nw=4) does NOT improve bandwidth
        # utilisation in this regime — blocks switch fast enough that wavefronts
        # are never stalled waiting for outstanding loads.
        #
        # Empirically (MI300X): for 8 M-element kernels with 4096+ blocks,
        # nw=4 configs run 15 % *faster* than nw=8 despite the Gaussian
        # predicting nw=8 should be 1–2 % better.  The Gaussian is calibrated
        # for single-block or few-block latency hiding; it overcounts the value
        # of additional warps once the grid already saturates the GPU.
        #
        # Return a flat 0.97 for any nw ≥ 4 (256 threads) in this regime so
        # that nw=4 and nw=8 are treated equally.  The grid score and secondary
        # XBLOCK tiebreaker then rank configs by block count — which IS a
        # meaningful discriminator for large kernels.  The real bench (TOP_N>1)
        # subsequently picks the empirical winner.
        _bw_xblock      = max(1, xblock)
        _bw_yblock      = max(1, yblock) if yblock else 1
        _bw_total       = problem_metadata.get('total_elements', 1)
        _bw_num_blocks  = max(1, _bw_total // (_bw_xblock * _bw_yblock))
        _SAT_THRESHOLD  = getattr(arch, 'num_cus', 256) * 8  # 2048 for 256-CU AMD
        if _bw_num_blocks >= _SAT_THRESHOLD or _bw_num_blocks == 1:
            # Two regimes where per-block thread count stops driving bandwidth:
            #
            # 1. Well-saturated grid (≥ 2048 blocks): block-switching hides HBM
            #    latency; adding more warps per block doesn't improve utilisation.
            #    nw=4 and nw=8 are empirically equivalent (or nw=4 faster due to
            #    lower SPI cost) for simple kernels at large problem sizes.
            #
            # 2. Single-block (1 block): there is no *other* block to switch to,
            #    so the wavefront latency-hiding argument for nw=8 doesn't apply.
            #    Empirically (MI300X), single-block 2D permute kernels run 25 %
            #    faster with nw=4 (more VGPRs per thread, better vectorisation).
            #
            # Return a flat 0.97 for any nw ≥ 4 so the grid factor and secondary
            # XBLOCK tiebreaker rank configs instead of the BW Gaussian.  The
            # real bench (TOP_N > 1) selects the empirical winner.
            if threads_per_block >= 256:  # nw ≥ 4 on AMD (4 × 64-thread wavefronts)
                return 0.97
            # nw < 4: fall through to the Gaussian (correct penalty for under-wavefronting)

        # Wider sigma (1.5× optimal) so XBLOCK=512 and XBLOCK=256 remain
        # competitive with XBLOCK=1024.  The 1× width caused a ≈6 % gap between
        # 256-element and 1024-element blocks, systematically biasing top-N
        # selection toward XBLOCK=1024 variants.
        sigma = optimal_threads * 1.5
        
        if threads_per_block < 64:
            score = 0.60
        else:
            diff     = (threads_per_block - optimal_threads) / sigma
            gaussian = math.exp(-0.5 * diff * diff)
            score    = max(0.60, min(1.0, 0.75 + 0.25 * gaussian))

        # 2-D/3-D coalescing correction — only XBLOCK < 16 is penalised.
        #
        # On AMD (warp_size=64) each wavefront always issues cache-line-aligned
        # X accesses, so XBLOCK ≥ 16 (one full 64-byte FP32 cache line) incurs
        # no penalty.  Sub-cache-line XBLOCK wastes bandwidth filling partial
        # cache lines.  XBLOCK spanning multiple rows is NEUTRAL-to-BENEFICIAL
        # on HBM systems with multiple independent memory channels.
        #
        # 3-D kernels further distinguish:
        #   cube-like  (max/min dim < 2): average X+Y coalescing factors.
        #   elongated  (max/min dim ≥ 2): XBLOCK-only (Y axis may be strided).
        if yblock > 0:
            cache_line_elems = 16   # 64 B / 4 B per FP32
            zblock = config.get('ZBLOCK', 0)

            if zblock > 0:
                dims_3d = problem_metadata.get('dimensions', ())
                is_cube_3d = False
                if len(dims_3d) == 3:
                    _max_d = max(dims_3d)
                    _min_d = max(min(dims_3d), 1)
                    is_cube_3d = (_max_d / _min_d) < 2

                if is_cube_3d:
                    coalescing_x = min(1.0, xblock / cache_line_elems)
                    coalescing_y = min(1.0, yblock / cache_line_elems)
                    avg_coalescing = (coalescing_x + coalescing_y) / 2.0
                    warp_align = 0.75 + 0.25 * avg_coalescing   # 0.75 → 1.00
                else:
                    if xblock >= cache_line_elems:
                        warp_align = 1.00
                    else:
                        coalescing = xblock / cache_line_elems
                        warp_align = 0.75 + 0.23 * coalescing   # 0.75 → 0.98
            else:
                if xblock >= cache_line_elems:
                    warp_align = 1.00
                else:
                    coalescing = xblock / cache_line_elems
                    warp_align = 0.75 + 0.23 * coalescing       # 0.75 → 0.98

            score *= warp_align
        
        return score
    
    # -------------------------------------------------------------------------
    # Factor 2 — Launch overhead amortisation
    # -------------------------------------------------------------------------
    
    @staticmethod
    def estimate_launch_overhead(grid_size: Tuple[int, ...],
                                 problem_metadata: Dict) -> float:
        """Score how well this config amortises the fixed kernel dispatch cost.

        **Single-block kernels** (num_blocks = 1):
            EPB IS the overhead-amortisation metric.  Modelled as a Gaussian
            centred at arch.optimal_elements_per_block with σ = optimal_epb/2.
            Score floor is 0.70 for EPB < 64 (overhead completely dominates).

        **Multi-block kernels** (num_blocks > 1):
            The ~3 µs kernel dispatch cost is shared equally across ALL blocks
            and is therefore already amortised over the whole problem regardless
            of block count.  The EPB-based Gaussian is therefore NOT a useful
            differentiator for multi-block configs — it incorrectly rewarded
            large-XBLOCK configs (e.g. XBLOCK=1024 → 64 blocks) over smaller
            ones (e.g. XBLOCK=256 → 256 blocks) even though the latter keeps
            4× more CUs active and is empirically 15–20% faster on MI300X for
            medium-sized problems.

            For multi-block configs we use a very wide sigma (3 × optimal_epb)
            so the score is near-flat in [0.88, 1.00] for any reasonable EPB.
            Only pathological configs with < 32 elements/block (sub-cache-line)
            are penalised.  Block-count optimisation is handled by the Grid score.

        One secondary penalty applies in both cases:

        Large-grid penalty (num_blocks > 4 × num_CUs): the Command Processor
        must batch-dispatch excess blocks, adding measurable scheduling latency:
            penalty = 1.0 - 0.01 × log2(num_blocks / (4 × num_CUs))
            clamped to ≥ 0.88

        Note: per-block warp count is intentionally NOT penalised here.  The
        occupancy model already handles over-warping via the natural_warps
        calculation; adding a redundant penalty here caused double-counting and
        incorrectly penalised XBLOCK=1024 + num_warps=16 (a natural fit where
        natural_warps = 1024/64 = 16) even though that config is empirically
        optimal.
        """
        num_blocks     = PointwiseHeuristics.prod(grid_size)
        total_elements = problem_metadata.get('total_elements', 1)
        
        if total_elements == 0 or num_blocks == 0:
            return 1.0
        
        elements_per_block = total_elements / num_blocks
        
        arch         = PointwiseHeuristics._get_arch()
        optimal_epb  = arch.optimal_elements_per_block

        if num_blocks == 1:
            # Single-block: the ~3 µs dispatch cost is not shared, so EPB
            # directly determines the overhead fraction.
            #
            # Special case: when the tile covers the ENTIRE problem (EPB ≈
            # total_elements) there is no way to reduce EPB without using a
            # different XBLOCK — e.g. for a 4096-element problem with
            # XBLOCK=256×YBLOCK=16, 1 block IS the minimum-block config.
            # Penalising high EPB here just selects multi-block configs that
            # happen to hit EPB=optimal_epb but run slower due to higher nw.
            # For problems that fit naturally in one block (≤ 2×optimal_epb
            # elements), return the flat multi-block score (0.97) so the Grid
            # factor becomes the sole differentiator between single- and multi-
            # block choices.
            if elements_per_block >= total_elements * 0.95:
                score = 0.97  # whole-problem tile: EPB penalty is not meaningful
            elif elements_per_block < 64:
                score = 0.70
            else:
                sigma    = optimal_epb // 2
                diff     = (elements_per_block - optimal_epb) / sigma
                gaussian = math.exp(-0.5 * diff * diff)
                score    = max(0.70, min(1.0, 0.75 + 0.25 * gaussian))
        else:
            # Multi-block: the kernel dispatch cost is shared equally across all
            # blocks, so EPB no longer determines the overhead fraction —
            # the overhead is already amortised by the multi-block launch.
            #
            # A Gaussian peaked at optimal_epb incorrectly rewards large-XBLOCK
            # configs (fewer blocks → high EPB → score 1.0) over small-XBLOCK
            # configs (many blocks → low EPB → penalised).  Empirically on
            # MI300X, XBLOCK=256–512 outperforms XBLOCK=2048–4096 by 25–40%
            # for large memory-bound streaming kernels, the opposite of what the
            # Gaussian predicts.  Block-count optimisation is handled by the
            # Grid score; Launch is flat once overhead is adequately amortised.
            if elements_per_block < 32:
                score = 0.88   # sub-cache-line: vectorisation broken
            elif elements_per_block < 64:
                score = 0.93   # marginal — some overhead exposure
            else:
                score = 0.97   # flat for any reasonable EPB
        
        return score
    
    # -------------------------------------------------------------------------
    # Factor 3 — Grid granularity
    # -------------------------------------------------------------------------
    
    @staticmethod
    def estimate_grid_granularity(grid_size: Tuple[int, ...],
                                  problem_metadata: Dict) -> float:
        """Score how well the grid size matches available hardware parallelism.

        The optimal block count scales with problem size:
          < 2 048   → 4 blocks (discrete lookup — Gaussian is meaningless at
                      this scale; empirically 4 CUs beats 1 by ~2–3 %)
          2 K–16 K  → num_CUs / 8   (partial saturation)
          16 K–256 K → num_CUs      (half-saturation target)
          > 256 K   → 2 × num_CUs   (full saturation)

        Each band uses a Gaussian of width σ = target × 0.5 bounded to [0.70, 1.00].
        """
        num_blocks     = PointwiseHeuristics.prod(grid_size)
        total_elements = problem_metadata.get('total_elements', 1)

        arch             = PointwiseHeuristics._get_arch()
        hardware_optimal = arch.optimal_blocks_grid  # = 2 × num_CUs
        ndims            = len(problem_metadata.get('dimensions', (1,)))

        if total_elements < 2048:
            # Tiny: discrete lookup because the element count is too small for
            # a Gaussian to carry meaningful signal.
            if num_blocks <= 1:
                return 0.85
            elif num_blocks <= 2:
                return 0.93
            elif num_blocks <= 4:
                return 1.00
            elif num_blocks <= 8:
                return 0.90
            else:
                return 0.70

        # Compute-intensity scale: compute-heavy kernels need MORE blocks because
        # high register pressure (VGPR) limits the number of concurrent wavefronts
        # per CU, requiring more blocks in flight to keep all CUs busy.
        #   ops_pe < 8  → memory-bound  → standard targets (divisor=512)
        #   ops_pe ≥ 8  → medium/heavy compute → 2× more blocks  (divisor=256)
        # Note: a 4× scale (divisor=128, targeting x=128 for 65K elements) was tried
        # but proved catastrophically wrong for variance/normalization kernels (var_6)
        # where fewer blocks (larger XBLOCK) are empirically optimal.  Capped at 2×.
        ops_pe      = max(1, int(problem_metadata.get('ops_per_element', 4)))
        comp_scale  = 1 if ops_pe < 8 else 2
        divisor_1d  = max(32, 512 // comp_scale)

        if total_elements < 16384:
            # Single-block: the tile covers the entire problem.  High ept (e.g.
            # 16 elements/thread for 4096-element × nw=4 config) gives ILP that
            # compensates for the lack of multi-CU parallelism.  0.95 reflects
            # that the whole-problem single-tile is the most coherent choice for
            # tiny 2D problems (e.g. y=16, x=256 = 4096 elements) where launching
            # many small blocks adds SPI overhead without improving GPU utilisation.
            if num_blocks == 1:
                return 0.95
            # For small problems (4K–16K) the GPU is heavily launch-bound but
            # empirical data shows that targeting more blocks with fewer warps
            # per block (nw=2, 32 blocks) consistently outperforms the "fewer,
            # larger blocks" strategy (nw=8, 8 blocks) — even though total
            # wavefront count is the same (64).  The per-block warp overhead on
            # AMD SPI is proportional to num_warps; with nw=8 that overhead can
            # double measured kernel time relative to nw=2 on small problems.
            # Target hw_opt//16 (≈32 blocks for 256-CU AMD) so XBLOCK=128+nw=2
            # configs hit the Gaussian peak.
            optimal = max(4, hardware_optimal // 16)
        elif total_elements < 262144:
            # Adaptive target scaled by compute intensity.  Memory-bound kernels
            # do well with 512-element tiles (divisor=512 → ~128 blocks for 64K
            # elements).  Compute-heavy kernels need 4× more blocks (divisor=128)
            # because VGPR pressure limits per-CU wavefront count, requiring more
            # concurrent blocks to saturate all CUs.
            optimal = max(hardware_optimal // 8, total_elements // divisor_1d)
        else:
            if ndims == 1:
                # For large 1-D kernels the optimal block count scales with
                # problem size AND compute intensity:
                #
                #   Memory-bound (few/no slow ops):
                #     Target = total_elements // 512.  XBLOCK=512 (one
                #     cache-line per warp load) hits the Gaussian peak, which
                #     is the best empirical sweet spot for streaming kernels.
                #
                #   Compute-moderate (slow_ops ≥ 2, e.g. gelu+sqrt, exp+div):
                #     Target = total_elements // 1024.  XBLOCK=1024 hits the
                #     peak, reducing the block-switching overhead when each
                #     thread needs more registers for SFU-op temporaries.
                #
                #   Compute-heavy (slow_ops ≥ 4, e.g. tanh+sin+cos+exp fused):
                #     Target = total_elements // 2048.  XBLOCK=2048 hits the
                #     peak.  Heavy transcendental kernels are VGPR-pressure
                #     bound: fewer blocks with more elements/thread gives the
                #     LLVM backend room to schedule SFU instructions without
                #     live-range conflicts.  On MI300X, x2048-nw8 runs ~15 %
                #     faster than x512-nw8 for fused tanh/sin/cos/exp kernels
                #     at 4+ M elements (matches max-autotune winner x2048+wpe1).
                #
                _slow_ops = problem_metadata.get('slow_ops', 0)
                _ops_pe   = problem_metadata.get('ops_per_element', 1)
                # _grid_div=1024 (targeting x=1024 as grid-optimal) only activates for
                # kernels that are BOTH compute-heavy AND large (≥4M elements).  At
                # ≥4M elements the `_is_large_sat_1d` flag turns on the EPT bell-curve
                # penalty in estimate_occupancy_impact, which rewards x=1024 nw=4
                # (EPT=4) and penalises x=512 (EPT=2).  Without the EPT bell-curve
                # (total < 4M), shifting the grid peak to x=1024 only hurts: it places
                # x=256 at 4×optimal (Grid≈0.75) instead of 2×optimal (Grid=1.0 via
                # the asymmetric Gaussian), excluding the empirically-fastest config
                # from the pool.  For 1M–4M slow-op kernels, keep divisor=512 so
                # both x=512 (at optimal) and x=256 (at 2×optimal, asymmetric bonus)
                # score Grid=1.0 and the benchmark can find the empirical winner.
                _large_sat_1d = (total_elements >= 4_000_000)
                if _slow_ops >= 4 or _ops_pe >= 20:
                    _grid_div = 2048   # heavy transcendental → x2048 is ILP peak
                elif _slow_ops >= 1 and _large_sat_1d:
                    # Large (≥4M) moderate transcendental (tanh, gelu, silu, exp, log):
                    # divisor=1024 targets x=1024 as the grid-optimal XBLOCK.
                    # The EPT bell-curve (active for ≥4M slow-op kernels) then rewards
                    # x=1024 nw=4 (EPT=4) over x=512 nw=4 (EPT=2).
                    _grid_div = 1024   # large SFU-saturated → x1024 (EPT=4 w/ nw=4) is peak
                else:
                    _grid_div = 512    # memory-bound, small, or simple arithmetic → x512 is peak
                optimal = total_elements // _grid_div
                # Minimum-blocks floor: at large element counts (≥1M) even heavy
                # transcendental kernels (div=2048 → 512 blocks at x=1M) need MORE
                # blocks for SFU latency hiding.  MI300X has 228 CUs; transcendental
                # instructions take ~100 cycles.  With only 512 blocks (2 blocks/CU)
                # there aren't enough wavefronts in flight to hide that latency.
                #
                # Setting _min_blocks = total // 1024 ensures we target at least
                # one block per CU at XBLOCK=1024 for any problem size.  Combined
                # with the asymmetric Gaussian (slow_ops≥1, below), this also pulls
                # XBLOCK=512 (2×optimal) into the pool, matching the empirical winner
                # for kernels like fused_abs_add_cos_exp_gelu…tanh at x=1M.
                #
                # Example: x=1M slow_ops=8 (tanh+cos+exp+sin…):
                #   raw optimal = 1M/2048 = 512 blocks → x=2048 is peak (wrong!)
                #   min_blocks  = 1M/1024 = 1024 → optimal=1024 blocks → x=1024 peak
                #   asymmetric: x=512 (2048 blocks = 2×1024) → Grid=1.0 ✓
                #   empirical winner: x=512 nw=8 (7.64 µs vs 11.44 µs for x=2048)
                _min_blocks = max(hardware_optimal, total_elements // 1024)
                optimal = max(optimal, _min_blocks)

                # Asymmetric Gaussian for SFU kernels: having MORE blocks than the
                # bandwidth-optimal target is NOT worse — smaller XBLOCK reduces
                # elements-per-thread and VGPR pressure, which often improves actual
                # hardware occupancy for kernels with transcendental operations.
                #
                # Rule: for blocks in (optimal, 2×optimal] and slow_ops ≥ 1, score
                # identically to the at-optimal case (return 1.0 immediately, before
                # the symmetric Gaussian below which would penalise these configs).
                # Beyond 2×optimal the penalty reapplies to avoid rewarding very tiny
                # XBLOCKs (x=64/128) that cause excessive launch overhead.
                #
                # Example – silu_sqrt_sub_0 x=2M (slow_ops=2):
                #   optimal = 4096 blocks (divisor=512)
                #   x=512 nw=8:  4096 blocks → Grid=1.0 (at optimal)
                #   x=256 nw=4:  8192 blocks → Grid=1.0 (asymmetric, no penalty)
                #   x=1024 nw=4: 2048 blocks → Grid=0.90 (below optimal, penalised)
                # → x=512 and x=256 both enter the top-5 pool; real bench picks x=256.
                if _slow_ops >= 1 and optimal < num_blocks <= 2 * optimal:
                    return 1.0
            else:
                # For large 2-D / 3-D kernels, target enough blocks so that
                # tiles fit within the AMD L1 cache (32 KB per CU).  For
                # FP32 (4 B/element), a 4 096-element tile = 16 KB input +
                # 16 KB output = 32 KB total, exactly fitting L1.
                #
                # The older value of max_tile_elems=16384 (EPT=32 for 512
                # threads) targeted only 1 024 blocks for a 16 M-element
                # problem, which happened to be the block count produced by
                # ultra-wide 4096×4 tiles.  Those tiles have YBLOCK=4, which
                # eliminates Y-dimension cache reuse and makes writes to the
                # transposed output scatter across thousands of cache lines —
                # empirically 2–4× slower than the smaller-tile alternatives
                # (128×32 or 64×32) that give ~4 096 blocks and fit in L1.
                #
                # Using max_tile_elems=4096 shifts the grid target to:
                #   16 M (4096²):  optimal = max(512, 4096) = 4096 blocks
                #   67 M (8192²):  optimal = max(512, 16384) = 16384 blocks
                #   2 M (128×16K): optimal = max(512, 512) = 512 blocks
                # This correctly rewards the empirical winners in all cases.
                max_tile_elems = 4096  # L1-cache-fitting 2-D tile
                optimal = max(hardware_optimal, total_elements // max_tile_elems)
                optimal = min(optimal, hardware_optimal * 8)

        # 3-D: shape-specific grid target.
        #   Cube-like (max/min < 2): regular-stride access → good L2 reuse;
        #     fewer, larger blocks are better (target CUs/2).
        #   Elongated (max/min ≥ 2): standard target (already sized by formula).
        # Note: no adjustment for 1-D / 2-D problems; the compute-intensity
        # scale above already accounts for the extra blocks needed.
        if ndims >= 3:
            dims_3d = problem_metadata.get('dimensions', (1, 1, 1))
            max_dim = max(dims_3d) if dims_3d else 1
            min_dim = max(min(dims_3d), 1) if dims_3d else 1
            if (max_dim / min_dim) < 2:
                # Cube-like: smaller block count needed for L2 locality.
                optimal = max(4, hardware_optimal // 2)

        if num_blocks < 4 and total_elements >= 262144:
            return 0.70

        sigma    = optimal * 0.5
        diff     = (num_blocks - optimal) / sigma
        gaussian = math.exp(-0.5 * diff * diff)
        return max(0.70, min(1.0, 0.75 + 0.25 * gaussian))

    # -------------------------------------------------------------------------
    # Factor 4 — Occupancy
    # -------------------------------------------------------------------------
    
    @staticmethod
    def estimate_occupancy_impact(config: Dict,
                                  problem_metadata: Dict,
                                  launch_bound: Optional[bool] = None) -> float:
        """Score whether the wavefront count per CU hides latency without wasting resources.

        Two regimes selected by grid saturation, block count, and the
        `launch_bound` flag from Stage 3 bottleneck analysis:

        Well-saturated (saturation ≥ 0.25) or single-block, or NOT launch-bound:
            Classic sweet-spot model — num_warps in [sweet_min, sweet_max] (4–8
            on AMD) scores 1.0.  Derived from Little's Law: need ceil(300/40) ≈ 8
            wavefronts to hide ~300-cycle HBM latency with a 40-cycle issue gap.
            4 suffices when L2 absorbs enough traffic to cut effective latency.

            Memory-bound kernels always use this regime even when saturation < 0.25:
            they still need multiple wavefronts per CU for HBM latency hiding.

        Launch-bound multi-block (saturation < 0.25, num_blocks > 1,
                                   and launch_bound is True):
            Few blocks → few total wavefronts → low GPU occupancy.
            Within each block the "natural" wavefront count is:
                natural_warps = ceil(XBLOCK / warp_size)
            e.g. XBLOCK=1024, warp_size=64 → natural_warps=16.

            • num_warps < natural_warps: each warp must iterate over multiple
              element chunks serially.  Score scales with thread utilisation:
                  score = 0.75 + 0.25 × (num_warps / natural_warps)
            • num_warps = natural_warps: one thread per element, optimal fit → 1.0.
            • num_warps > natural_warps: extra warps add SPI init cost with no
              additional data coverage:
                  T(extra) = K_launch + extra × K_warp
                  score    = K_launch / T(extra)
              where K_launch = 3.0 µs and K_warp = 0.2 µs.

        ILP correction — when each physical thread processes many tile elements
        (elements_per_thread = xblock_prod / threads_per_block) the compiler
        unrolls the element loop and issues all loads as independent instructions.
        At ept ≥ 8 this is equivalent to having 8+ in-flight memory requests per
        thread, which hides HBM round-trip latency without extra wavefronts.

        HOWEVER, in the launch-bound multi-block regime the ILP correction is
        scaled by CU utilisation:
            ilp_scale = min(1.0, cu_util / 0.25)
            cu_util   = num_blocks / num_CUs
        Rationale: ILP benefits only the currently-active blocks.  When far
        fewer blocks than CUs are active (e.g. 64 blocks on 304 CUs → 21% CU
        util), the HBM bandwidth from the idle 240 CUs is completely wasted.
        Applying the full ILP bonus in that case overestimates the config's
        actual performance — empirically 15–20% slower on MI300X for medium
        kernels.  At ≥ 25% CU utilisation (e.g. 76+ blocks on 304 CUs) the
        system-level bandwidth is well-utilised and the full ILP benefit applies.

        MEMORY-BOUND CORRECTION (well-saturated regime):
        The ILP bonus is disabled for num_warps < sweet_min when the kernel is
        memory-bound (arithmetic intensity < OI_ceiling).  Rationale: ILP hides
        per-thread memory LATENCY, but does NOT substitute for the per-CU
        wavefront count needed to saturate HBM BANDWIDTH.  On MI300X (OI_ceiling
        ≈ 222 FLOPS/byte), virtually all pointwise kernels are memory-bound.
        With num_warps=1 and 2048 blocks the CU has only ≈7 resident wavefronts
        — far below the ≈30 needed for full latency hiding.  Empirically,
        num_warps=1 with elems_per_thread=16 was measured 1.4–2.8× slower than
        num_warps=8 on MI300X, yet the ILP correction was scoring both identically
        at 1.0.  The cap at 0.82 (between the baseline of 0.85 and the sweet
        spot of 1.0) reflects this real performance gap without over-penalising.

        Returns a value in [0.70, 1.00].
        """
        num_warps      = config.get('num_warps', 4)
        total_elements = problem_metadata.get('total_elements', 1)
        warp_size      = problem_metadata.get('warp_size', 64)

        arch      = PointwiseHeuristics._get_arch()
        sweet_min = arch.occupancy_sweetspot_min   # 4 on AMD
        sweet_max = arch.occupancy_sweetspot_max   # 8 on AMD

        block_dims     = PointwiseHeuristics.get_block_dimensions(config)
        xblock_prod    = PointwiseHeuristics.prod(block_dims)
        num_blocks_est = max(1, total_elements // max(1, xblock_prod))

        # Large-saturated-1D flag: for 1D kernels ≥4M elements with slow_ops≥1,
        # the GPU is already fully saturated at much lower block counts.  In this
        # regime, EPT (elements-per-thread / ILP) dominates, not warp count.
        # Used by _sweet_spot_score and the EPT-ILP penalty below.
        _ndims_occ     = len(problem_metadata.get('dimensions', (1,)))
        _slow_ops_occ  = problem_metadata.get('slow_ops', 0)
        _is_large_sat_1d = (
            _ndims_occ == 1
            and _slow_ops_occ >= 1
            and total_elements >= 4_000_000
        )

        try:
            import torch as _torch
            num_cus = (_torch.cuda.get_device_properties(0).multi_processor_count
                       if _torch.cuda.is_available() else 120)
        except Exception:
            num_cus = 120

        # Saturation: fraction of peak wavefront capacity occupied.
        max_wf     = num_cus * 8   # AMD: ~8 resident wavefronts per CU
        saturation = min(1.0, (num_blocks_est * num_warps) / max_wf)

        # ILP-aware occupancy: when each thread handles many elements the compiler
        # can software-pipeline memory loads across iterations, hiding the same
        # latency that would normally require multiple wavefronts.
        # ept ≥ 8  → 8+ in-flight loads per thread; effectively immune to stalls.
        # ept ≥ 4  → partial benefit; soften the under-warp penalty.
        threads_per_block = num_warps * warp_size
        elems_per_thread  = max(1, xblock_prod // threads_per_block)

        # Determine whether this block configuration has enough CU-level wavefront
        # switching to hide HBM round-trip latency WITHOUT extra intra-block warps.
        #
        # Rule: a CU needs ≥ sweet_max (≈8) independent wavefronts to hide the
        # ~300-cycle HBM latency at a ~40-cycle issue rate (Little's Law).
        # When many blocks are dispatched (num_blocks ≥ num_cus × sweet_max) the
        # CU scheduler switches between residet blocks, providing CU-level latency
        # hiding even with only 1 wavefront per block.  When there are fewer blocks
        # per CU, the intra-block warp count becomes the sole source of wavefronts
        # for the CU to switch between — so num_warps=1 leaves the CU idle.
        #
        # This condition is computed purely from block geometry, not from the
        # compute/memory roofline, because on MI300X the OI_ceiling (≈222 FLOP/B)
        # is so high that virtually all pointwise kernels are "memory-bound" by
        # the traditional definition — making that check uninformative.
        wf_per_cu = num_blocks_est / max(1, num_cus)
        # True  → each CU sees ≤ sweet_max wavefronts from block-level switching
        # alone; num_warps=1 starves the CU of latency-hiding opportunities.
        # Use ≤ (not <) to handle the exact boundary (e.g. 2048 blocks / 256 CUs
        # = 8.0 wf/CU — borderline but empirically still insufficient on MI300X).
        need_intra_block_warps = wf_per_cu <= sweet_max

        def _sweet_spot_score(nw: int) -> float:
            if sweet_min <= nw <= sweet_max:
                return 1.00
            elif sweet_min // 2 <= nw <= int(sweet_max * 1.5):
                # For large saturated 1D slow_ops kernels, nw=2 with EPT≥4 is
                # fully equivalent to the sweet spot: the GPU is already saturated
                # via block-switching alone, and EPT=4 enables full pipelining
                # of transcendental instructions (tanh, gelu, exp, etc.).
                if nw == 2 and _is_large_sat_1d and elems_per_thread >= 4:
                    return 1.00
                return 0.95
            elif nw == 1:
                # ILP (software pipelining) can substitute for extra wavefronts
                # when the CU already has enough block-switching opportunities.
                # When blocks-per-CU is too low (need_intra_block_warps=True),
                # the CU scheduler has nothing to switch to — ILP doesn't help.
                if not need_intra_block_warps:
                    if elems_per_thread >= 8:
                        return 1.00   # CU-level block-switching + full ILP
                    elif elems_per_thread >= 4:
                        # For large saturated 1D slow_ops kernels: EPT=4 is
                        # sufficient for full transcendental-op pipelining.
                        # Empirically: gelu/tanh x=67M — nw=1 EPT=4 (102µs)
                        # equals nw=1 EPT=8; the bottleneck is SFU throughput,
                        # not load/store overlap.  Raising from 0.93→1.00 lets
                        # these low-nw configs compete with nw=4/8 in scoring.
                        if _is_large_sat_1d:
                            return 1.00
                        return 0.93   # Partial ILP benefit
                return 0.82   # Too few wavefronts; stall-bound
            # nw > sweet_max * 1.5: high warp count provides ample latency hiding
            # but scores below the sweet spot because of VGPR pressure.
            return 0.88

        # Natural-warps regime: few total wavefronts, multi-block, launch-bound.
        # Memory/compute-bound kernels stay in the sweet-spot regime even at
        # low saturation — they still need intra-block wavefronts for HBM hiding.
        use_natural_warps = (
            saturation < 0.25
            and num_blocks_est > 1
            and launch_bound is not False
        )

        if not use_natural_warps:
            # Well-saturated or confirmed not launch-bound: latency hiding governs.
            base_score = _sweet_spot_score(num_warps)

            # EPT-ILP penalty for large saturated 1D slow_ops kernels:
            #
            # When the GPU is fully saturated (many blocks/CU) and the kernel has
            # transcendental ops, a high warp count with EPT=1 provides NO occupancy
            # benefit but PREVENTS the compiler from unrolling/pipelining the slow
            # ops (tanh, gelu, erf) across multiple iterations per thread.
            #
            # Empirical: gelu_0 x=67M — x=512 nw=8 (EPT=1) takes 190µs, but
            # x=256 nw=1 (EPT=4) and x=512 nw=2 (EPT=4) both take ~103µs.
            # The 1.85× gap is entirely explained by the compiler's ability to
            # pipeline 4 gelu computations when EPT=4 vs zero overlap at EPT=1.
            #
            # Bell-curve EPT correction for large saturated 1D slow_ops kernels:
            #
            #  EPT < 4  → too few elements/thread: insufficient ILP to pipeline
            #             transcendental ops (tanh, gelu, exp).
            #             Penalty: 10% at EPT=1, ramping to 0% at EPT=4.
            #
            #  EPT 4–8  → sweet spot: optimal instruction-level pipelining of
            #             SFU instructions with low register pressure.
            #
            #  EPT > 8  → register pressure from unrolled transcendental loops.
            #             Empirically, x=1024 nw=1 (EPT=16) runs 5–10% slower
            #             than x=1024 nw=4 (EPT=4) for tanh/gelu at large sizes.
            #             Penalty: 10% at EPT=16+, ramping from 0% at EPT=8.
            #
            # Applied only in the well-saturated regime (not natural_warps path).
            if _is_large_sat_1d:
                if elems_per_thread < 4:
                    _ept_ratio   = min(1.0, elems_per_thread / 4.0)
                    _ept_penalty = 1.0 - 0.10 * (1.0 - _ept_ratio)
                elif elems_per_thread <= 8:
                    _ept_penalty = 1.0   # sweet spot
                else:
                    # EPT > 8: ramp from 0% at EPT=8 to 10% at EPT≥16
                    _over        = min(1.0, (elems_per_thread - 8) / 8.0)
                    _ept_penalty = 1.0 - 0.10 * _over
                base_score *= _ept_penalty
            elif total_elements >= 16384:
                # General mild EPT < 2 penalty for medium/large non-slow-ops kernels.
                #
                # x=512 nw=8 (EPT=1) consistently scores well by the BW/Occupancy
                # factors but runs 3-4× slower than nw=4 or nw=1 alternatives for
                # kernels like leaky_relu x=65K and mul_relu x=262K on MI300X.
                # Root cause: with exactly 1 element/thread, scheduling overhead
                # (warp dispatch, register setup for 8 wavefronts) dominates over the
                # tiny per-thread compute for kernels with >= 64K elements (enough
                # blocks that occupancy is not the bottleneck).
                #
                # Empirical: leaky_relu x=65K — x=512 nw=8 (EPT=1) takes 55µs,
                # x=512 nw=4 (EPT=2) takes ~19µs.  mul_relu x=262K — x=512 nw=8
                # takes 80µs vs 19µs for x=1024 nw=8 (EPT=2) or x=512 nw=4 (EPT=2).
                #
                # Does NOT apply to small kernels (< 16K elements) where nw=8 with
                # EPT=1 IS optimal (e.g. add_gelu x=4096: x=512 nw=8 wins at 18.9µs).
                # Does NOT double-penalise _is_large_sat_1d kernels (handled above).
                if elems_per_thread < 2.0:
                    _ept_ratio   = min(1.0, elems_per_thread / 2.0)
                    _ept_penalty = 1.0 - 0.10 * (1.0 - _ept_ratio)
                    base_score  *= _ept_penalty
        else:
            # Low-saturation launch-bound multi-block. Score relative to the
            # natural warp count — the number of wavefronts needed for one
            # thread per element.  Going below natural forces serial element
            # iteration (slower); going above adds SPI init overhead (also slower).
            natural_warps = max(1, xblock_prod // warp_size)

            k_launch = (BottleneckAnalysis.KERNEL_LAUNCH_US
                        if BottleneckAnalysis is not None else 3.0)

            k_warp = 0.2   # µs per extra wavefront beyond natural (SPI alloc)

            if num_warps <= natural_warps:
                # Under-warped: each warp iterates over multiple element chunks.
                base_score = 0.75 + 0.25 * (num_warps / natural_warps)
            else:
                # Over-warped: purely extra SPI init with no data benefit.
                extra = num_warps - natural_warps
                t_config = k_launch + extra * k_warp
                base_score = k_launch / t_config

            # ILP correction: software-pipelining hides per-thread latency, but
            # not the system-level cost of idle CUs.  Scale the ILP ceiling by
            # CU utilisation so under-subscribed grids don't get a free pass.
            #   ilp_scale = 0 at 0% CU util → full benefit at ≥ 25% CU util
            cu_util   = min(1.0, num_blocks_est / max(1, num_cus))
            ilp_scale = min(1.0, cu_util / 0.25)

            if elems_per_thread >= 8:
                # Full ILP benefit when CUs are well-used; partial otherwise.
                ilp_ceiling = 0.766 + 0.234 * ilp_scale  # 0.766 → 1.000
                base_score = max(base_score, ilp_ceiling)
            elif elems_per_thread >= 4:
                # Partial software-pipelining benefit.
                ilp_ceiling = 0.766 + 0.164 * ilp_scale  # 0.766 → 0.930
                base_score = max(base_score, ilp_ceiling)

        # ── AMD-only: waves_per_eu correction (only when feature is enabled) ──
        # waves_per_eu > 0 means the LLVM backend plans for N concurrent
        # wavefronts per EU, which has two countervailing effects:
        #
        # 1. Occupancy boost  (positive): sibling blocks co-reside on the same
        #    CU, increasing the total wavefronts available to the scheduler
        #    for latency hiding.  Effective wavefronts ≈ num_warps × waves_per_eu.
        #    Applied only in the well-saturated regime (not natural-warps) and
        #    only when the grid is large enough to actually fill the extra slots.
        #
        # 2. VGPR pressure penalty (negative): the register file is partitioned
        #    across N concurrent blocks, reducing the per-thread VGPR budget
        #    from  65536 / threads  to  65536 / (threads × waves_per_eu).
        #    If the kernel's estimated VGPR need exceeds the tighter budget,
        #    register spills occur → score penalty proportional to the excess.
        #
        # On CUDA or when the env var is off, waves_per_eu is always 0 (no
        # variants were generated), so this block is a no-op in both cases.
        waves_per_eu = config.get('waves_per_eu', 0)
        if waves_per_eu > 0 and PointwiseHeuristics._is_waves_per_eu_enabled():
            # wpe=1: max VGPR per thread (1 wave per EU); no occupancy penalty or
            # boost relative to wpe=0.  For compute-heavy kernels (slow_ops ≥ 4)
            # this lets LLVM use all 256 VGPRs freely.  Give a tiny bonus so the
            # real bench gets the chance to compare wpe=1 against wpe=0 variants.
            if waves_per_eu == 1:
                _slow_ops_wpe = problem_metadata.get('slow_ops', 0)
                if _slow_ops_wpe >= 4 and not use_natural_warps:
                    base_score = min(1.0, base_score * 1.003)  # 0.3% tie-breaker
                # No VGPR pressure change (budget identical to wpe=0).
            else:
                # --- Occupancy boost (wpe ≥ 2) ---
                # Only when under the sweet spot and not in the natural-warps regime.
                # Require enough blocks for the concurrent-block scheduling to engage:
                #   num_blocks ≥ num_cus × waves_per_eu
                if not use_natural_warps and num_warps < sweet_min:
                    if num_blocks_est >= num_cus * waves_per_eu:
                        effective_warps = min(sweet_max, num_warps * waves_per_eu)
                        base_score = max(base_score, _sweet_spot_score(effective_warps))
                else:
                        # Grid too small: wpe tightened the VGPR budget but the extra
                        # wavefront slots will sit empty — pure downside, no gain.
                        # Penalise proportionally to how far below the fill threshold
                        # we are.  Range: 0.95 (completely empty) → 1.00 (just filled).
                        _fill = num_blocks_est / max(1, num_cus * waves_per_eu)
                        base_score *= 0.95 + 0.05 * min(1.0, _fill)

            # --- VGPR pressure penalty (only for wpe ≥ 2) ---
            # wpe=1 does NOT reduce the per-thread VGPR budget (budget = 256,
            # identical to wpe=0), so no spill risk and no penalty needed.
            # For wpe ≥ 2 the budget is halved/quartered and may cause spills.
            if waves_per_eu >= 2:
                # Budget when waves_per_eu blocks share the CU register file:
                #   budget = floor(65536 / (threads_per_block × waves_per_eu))
                # rounded down to the 8-register hardware granule.
                _threads_wpe    = (num_warps * warp_size) * waves_per_eu
                _raw_budget     = 65536 // max(1, _threads_wpe)   # AMD CDNA2/3 per CU
                _vgpr_budget    = min(256, (_raw_budget // 8) * 8)

                if _vgpr_budget > 0:
                    ops_pe     = problem_metadata.get('ops_per_element', 3)
                    # Lightweight VGPR estimate: base overhead + tensor arg overhead
                    # (pointers, masks, values) + op temporaries.
                    # load_ops ≈ n_input_tensors; +1 for the output tensor.
                    # The full model lives in _estimate_spill_risk (runtime);
                    # this is a cheap scoring proxy.
                    _load_ops_s       = problem_metadata.get('load_ops', 0)
                    _n_tensors_s      = max(3, _load_ops_s + 1)
                    # Use ×4 per op (not ×2) to better capture actual VGPR usage
                    # in fused kernels with loop-carried temporaries.
                    # Base raised from 16→24 to match the generation pre-filter.
                    _rough_vgpr_need  = 24 + _n_tensors_s * 4 + min(ops_pe * 4, 128)
                    _spill_ratio      = _rough_vgpr_need / _vgpr_budget
                    if _spill_ratio > 1.0:
                        # Over budget: penalise proportionally to the excess.
                        # e.g. ratio=1.5 → ×0.667, ratio=2.0 → ×0.500.
                        base_score *= 1.0 / _spill_ratio
                    elif _spill_ratio > 0.75:
                        # Marginal (within 25% of budget): small tie-breaker penalty.
                        # Threshold tightened from 0.80→0.75 to match generation gate.
                        base_score *= 0.97

        return max(0.70, min(1.0, base_score))

    # -------------------------------------------------------------------------
    # Composite scoring
    # -------------------------------------------------------------------------
    
    @staticmethod
    def score_config(config: Dict,
                     problem_metadata: Dict,
                     kernel_code: Optional[str] = None) -> float:
        """Return a composite score in [0.0, 1.0] for a single config.

        The four factor scores are combined via a weighted geometric mean:

            score = (BW^a * Launch^b * Grid^c * Occupancy^d) ^ (1/(a+b+c+d))

        Exponents a–d are derived from adaptive weights (see BottleneckAnalysis).
        When BottleneckAnalysis is unavailable a fixed exponent set is used as
        a fallback.

        For 2-D configs a secondary tie-breaker multiplier is applied to
        distinguish otherwise equal scores:
          1. Square-ish aspect ratio (elongated tiles misalign the prefetcher):
                ×(1.0 − 0.005 × log2(max/min))
          2. Larger XBLOCK (wide tiles fill cache lines for row-major tensors):
                ×(1.0 − 0.008 × max(0, 5 − log2(XBLOCK)))
        """
        try:
            # Enrich problem_metadata with kernel-derived instruction mix when
            # kernel_code is available and the caller has not already supplied
            # ops_per_element (i.e. it is still at the uninformative default of
            # ≤ 2).  This ensures the compute-heavy EPT bonus below uses the
            # real slow-op count extracted from the compiled Triton source, not
            # just the generic problem-size estimate.
            if kernel_code and problem_metadata.get('ops_per_element', 0) <= 2:
                try:
                    from torch._inductor.codegen.triton_heuristics_analysis import (
                        extract_kernel_metadata,
                    )
                    _km = extract_kernel_metadata(kernel_code)
                    if _km.get('ops_per_element', 0) > 2:
                        problem_metadata = {**problem_metadata, **_km}
                except Exception:
                    pass

            block_dims   = PointwiseHeuristics.get_block_dimensions(config)
            problem_dims = PointwiseHeuristics.get_problem_dimensions(problem_metadata)
            
            if len(block_dims) != len(problem_dims):
                return 0.0
            
            grid_size = PointwiseHeuristics.calculate_grid_size(problem_dims, block_dims)
            
            # Run bottleneck analysis once; results feed both adaptive weights
            # and the occupancy regime switch.
            launch_bound: Optional[bool] = None
            bw_exp, launch_exp, grid_exp, occ_exp = 2.5, 1.8, 1.2, 0.6
            if BottleneckAnalysis is not None:
                try:
                    analysis  = BottleneckAnalysis.analyze_bottleneck(
                        config, problem_metadata, kernel_code
                    )
                    launch_bound = analysis.get('launch_bound')
                    weights   = BottleneckAnalysis.get_adaptive_weights(
                        config, problem_metadata, kernel_code, analysis=analysis
                    )
                    exponents = BottleneckAnalysis.get_adaptive_exponents(weights)
                    bw_exp    = exponents['bandwidth']
                    launch_exp = exponents['launch']
                    grid_exp  = exponents['grid']
                    occ_exp   = exponents['occupancy']
                except Exception:
                    pass

            bandwidth   = PointwiseHeuristics.estimate_memory_bandwidth(config, problem_metadata)
            launch      = PointwiseHeuristics.estimate_launch_overhead(grid_size, problem_metadata)
            granularity = PointwiseHeuristics.estimate_grid_granularity(grid_size, problem_metadata)
            occupancy   = PointwiseHeuristics.estimate_occupancy_impact(
                config, problem_metadata, launch_bound=launch_bound
            )

            total_exp = bw_exp + launch_exp + grid_exp + occ_exp
            score = (
                (bandwidth   ** bw_exp)    *
                (launch      ** launch_exp) *
                (granularity ** grid_exp)  *
                (occupancy   ** occ_exp)
            ) ** (1.0 / total_exp)

            # 2-D tie-breaker: mildly penalise extreme tile-shape asymmetry
            # and sub-cache-line XBLOCK.
            #
            # balance_multiplier: penalises tiles with very high aspect ratio
            # (XBLOCK >> YBLOCK, e.g. 4096×4 or 4096×1).  Empirical data
            # consistently shows that such ultra-wide tiles are 2–4× SLOWER
            # than moderate-aspect tiles (64×32, 128×16) for AMD permute and
            # transpose kernels because:
            #   1. YBLOCK=1–4 eliminates Y-direction cache reuse.
            #   2. Transposed writes become maximally scattered across output
            #      columns, breaking HBM write-combining.
            # The formula applies a soft 5% penalty at ratio=1024 (e.g. 4096×4)
            # and ~6% at ratio=4096 (4096×1), letting grid and BW scores break
            # the tie for moderate asymmetry (64×32 ratio=2 → 0.5% penalty).
            #
            # innermost_multiplier: penalises XBLOCK < 32 (sub-cache-line X
            # loads waste bandwidth).
            #
            # Skip entirely for single-block configs (tile covers whole problem,
            # shape has no effect on memory access patterns).
            num_blocks_2d = PointwiseHeuristics.prod(grid_size)
            if len(block_dims) == 2 and num_blocks_2d > 1:
                xblock, yblock = block_dims
                ratio              = max(xblock, yblock) / max(min(xblock, yblock), 1)
                balance_multiplier = 1.0 - 0.005 * math.log2(max(ratio, 1.0))
                # Sub-cache-line penalty: on AMD MI300X a cache line is 64 FP32
                # elements (256 bytes).  When XBLOCK < 64 each warp reads a partial
                # cache line, wasting L2 bandwidth and hurting permute/transpose
                # kernels especially hard (misaligned reads AND scattered writes).
                #
                # Old formula gave only 0.8 % penalty at XBLOCK=16 — far too mild.
                # Empirically, 2D permute kernels run 20-30 % faster with XBLOCK=64
                # (full cache line) than XBLOCK=16 (quarter cache line), so we need
                # a 5 % penalty per halving below 64:
                #   XBLOCK ≥ 64: 0 % (full cache line, no penalty)
                #   XBLOCK  = 32: 5 % penalty
                #   XBLOCK  = 16: 10 % penalty
                #   XBLOCK  =  8: 15 % penalty
                #   XBLOCK  =  4: 20 % penalty
                innermost_multiplier = 1.0 - 0.05 * max(0, 6 - math.log2(max(xblock, 4)))
                # Lower floor (0.90 vs old 0.95) lets the penalty actually bite for
                # XBLOCK ≤ 16 rather than being clipped at 5 % cap.
                score *= max(0.90, balance_multiplier * innermost_multiplier)

            # 3-D tie-breaker: for cube-like problems mildly prefer balanced
            # tile shapes and a minimum YBLOCK (elongated tiles coalesce poorly
            # when YBLOCK is the stride-1 axis of a permuted tensor).
            # balance_mult = 1 − 0.003 × log2(max/min tile dim)
            # yblock_mult  = 1 − 0.008 × max(0, 4 − log2(yblock))
            elif len(block_dims) == 3:
                xblock, yblock, zblock = block_dims
                problem_dims = PointwiseHeuristics.get_problem_dimensions(problem_metadata)

                is_cube_problem = False
                if len(problem_dims) == 3:
                    max_p = max(problem_dims)
                    min_p = max(min(problem_dims), 1)
                    is_cube_problem = (max_p / min_p) < 2

                if is_cube_problem:
                    max_dim = max(xblock, yblock, zblock)
                    min_dim = max(min(xblock, yblock, zblock), 1)
                    ratio   = max_dim / min_dim
                    balance_multiplier = 1.0 - 0.003 * math.log2(max(ratio, 1.0))
                    yblock_multiplier  = 1.0 - 0.008 * max(0, 4 - math.log2(max(yblock, 1)))
                    score *= max(0.97, balance_multiplier * yblock_multiplier)

            # ─── Compute-heavy EPT bonus ───────────────────────────────────────────
            # For kernels with many ops/element (≥ 12) AND multiple slow (SFU)
            # instructions (slow_ops ≥ 2), processing more elements per thread
            # (higher EPT) improves instruction-level parallelism (ILP) within
            # each wavefront and amortises register-setup overhead.  Max-autotune
            # empirically finds XBLOCK=2048–4096 configs run ~1.5× faster for
            # heavy-fusion kernels, yet grid_granularity penalises those larger
            # XBLOCKs (fewer blocks → below the "optimal" target) — understating
            # their value.
            #
            # The slow_ops ≥ 2 guard is critical: kernels with only 1 SFU op
            # (e.g. silu, gelu) do NOT suffer VGPR pressure at nw=8 and often run
            # faster with more wavefronts (nw=8) for HBM latency hiding.  Applying
            # the ILP bonus to those single-transcendental kernels would incorrectly
            # bias the scorer toward nw=4 and cause regressions.
            #
            # The total_elements < 524288 (512 K) guard restricts the bonus to the
            # overhead/mixed regime.  For large problems (≥ 512K elements) HBM
            # bandwidth dominates even for compute-heavy kernels; in that regime
            # more wavefronts (nw=8) improve HBM latency hiding and outperform the
            # ILP benefit of higher EPT.  Empirically (MI300X): fused transcendental
            # kernels at x=1048576 run ~1.5× faster with x512/nw8 (more wavefronts)
            # than x1024/nw4 despite the lower EPT.
            #
            # Guard: only apply when there are enough blocks to keep the GPU
            # reasonably busy (≥ num_cus//3 blocks); otherwise the GPU utilisation
            # is too low for the ILP benefit to matter.
            ops      = problem_metadata.get('ops_per_element', 1)
            slow_ops = problem_metadata.get('slow_ops', 0)
            _total_el_ept = problem_metadata.get('total_elements', 1)
            if (ops >= 12 and slow_ops >= 2 and _total_el_ept < 524288
                    and config.get('num_warps', 4) >= 2 and len(block_dims) >= 1):
                _xblock_ept = block_dims[0]
                _warp_size  = problem_metadata.get('warp_size', 64)
                _threads    = config.get('num_warps', 4) * _warp_size
                _ept        = _xblock_ept / max(_threads, 1)
                if _ept > 1.0:
                    _num_blks  = max(1, _total_el_ept // _xblock_ept)
                    _arch      = PointwiseHeuristics._get_arch()
                    _min_blks  = max(8, _arch.num_cus // 3)   # ~101 for MI300X
                    if _num_blks >= _min_blks:
                        # ops_factor: 0 at 12 ops, 1.0 at 27+ ops
                        _ops_f  = min(1.0, max(0.0, (ops - 12) / 15.0))
                        # ept_factor: log2(2)=1, log2(4)=2, log2(8)=3 → cap at 3
                        _ept_f  = min(3.0, math.log2(_ept))
                        # max bonus ≈ 0.045 (at ops ≥ 27, EPT ≥ 8, GPU ≥ ⅓ full)
                        score = min(1.0, score + _ops_f * _ept_f * 0.015)
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.0
    
    # -------------------------------------------------------------------------
    # Config generation
    # -------------------------------------------------------------------------
    
    # Triton's kernel compiler rejects `tl.arange` when the total number of
    # elements in the resulting tensor exceeds this constant.  Configs whose
    # tile size (XBLOCK × YBLOCK × ZBLOCK) exceeds this limit will always fail
    # to compile with ``ValueError("numel (...) exceeds triton maximum tensor
    # numel (1048576)")``.  Filter them out proactively.
    TRITON_MAX_TILE_NUMEL: int = 1048576   # 2^20

    # Hardware register-pressure limit for multi-dimensional tile kernels.
    #
    # On AMD MI300X each thread has 256 VGPRs (32-bit slots).  A 2-D/3-D
    # pointwise kernel must hold `tile_size / threads_per_block` data elements
    # simultaneously in registers while the inner loops are live.  Profiling of
    # tuning66 shows that, without exception, every config with
    #   elems_per_thread = tile_size / (num_warps × warp_size) ≥ 64
    # produces `inf µs` (register spill / PTXASError / OutOfResources).
    # No config at elems_per_thread ≤ 32 ever spills.  The range 33–63 is
    # safe for simple kernels but complex fused ops (many transcendentals) can
    # still spill there – we cap at 32 for conservatism and zero spill risk.
    #
    # Mathematically: 32 data elements × 2 VGPRs each = 64 VGPRs for data;
    # leaving 192 VGPRs for address registers, accumulators, loop variables,
    # and compiler temps – sufficient for even heavily-fused kernels.
    MAX_ELEMS_PER_THREAD_MULTIDIM: int = 32  # reject if > this (avoids VGPR spills)

    @staticmethod
    def generate_all_candidate_configs(problem_metadata: Dict) -> List[Dict]:
        """Enumerate every legal (XBLOCK[×YBLOCK[×ZBLOCK]], num_warps) pair.

        Three hard constraints are applied before any config is accepted:
          1. Dimension cap:  XBLOCK ≤ xnumel, YBLOCK ≤ ynumel, etc.
          2. Warp–thread coherence: num_warps × warp_size ≤ total_threads_per_block.
             (Declaring more warps than threads would be silently clamped by the
             hardware, making the declared num_warps misleading to the compiler.)
          3. Triton tile-numel limit: XBLOCK × YBLOCK × ZBLOCK ≤ 1 048 576.
             Configs exceeding this would fail to compile with a Triton
             ``ValueError: numel exceeds triton maximum tensor numel``.

        Block size candidates are powers of two.  Non-power-of-two sizes trigger
        tail-masking in Triton's codegen and prevent full unroll/vectorisation.

        Returns 30–80 configs for 1-D problems, 60–120 for 2-D.
        """
        problem_dims = PointwiseHeuristics.get_problem_dimensions(problem_metadata)
        ndims        = len(problem_dims)
        warp_size    = problem_metadata.get('warp_size', 64)
        max_threads  = problem_metadata.get('max_threads_per_block', 1024)
        max_warps    = max_threads // warp_size

        warp_candidates    = [1, 2, 4, 8]
        block_sizes_1d     = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        block_sizes_2d     = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        block_sizes_3d     = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

        configs: List[Dict] = []
        
        if ndims == 1:
            xnumel = problem_dims[0]
            for xblock in block_sizes_1d:
                if xblock > xnumel:
                    continue
                for nw in warp_candidates:
                    if nw > max_warps or nw * warp_size > xblock:
                        continue
                    # Register-pressure guard for 1-D kernels.
                    #
                    # The AMD AMDGPU LLVM backend fully unrolls inner loops with
                    # ≤ 16 vectorized iterations (4-wide FP32 SIMD).  At ept > 32
                    # (ept=64 → 16 iterations), 16 copies of the kernel body are
                    # live simultaneously.  For kernels with many transcendental
                    # ops (gelu, exp, tanh, …) or Welford accumulators, this
                    # pushes VGPR usage past the 256-VGPR MI300X limit, causing
                    # n_spills > 32 → bench() returns inf.
                    #
                    # Empirical evidence (tuning68):
                    #   XBLOCK=4096, nw=1 (ept=64): always inf for complex kernels
                    #   XBLOCK=4096, nw=2 (ept=32): always valid (≤ 8 loop iters)
                    #
                    # Performance cost of filtering: ≤ 4% for the 3 simple-kernel
                    # cases where ept=64 was the empirical winner — the second-best
                    # config at ept=32 is within measurement noise.
                    elems_per_thread = xblock / (nw * warp_size)
                    if elems_per_thread > PointwiseHeuristics.MAX_ELEMS_PER_THREAD_MULTIDIM:
                        continue
                    configs.append({'XBLOCK': xblock, 'num_warps': nw})
        
        elif ndims == 2:
            xnumel, ynumel = problem_dims[0], problem_dims[1]
            for xblock in block_sizes_2d:
                if xblock > xnumel:
                    continue
                for yblock in block_sizes_2d:
                    if yblock > ynumel:
                        continue
                    # tile_size is the number of *elements* processed by one block,
                    # not the hardware thread count.  For 2-D kernels each thread can
                    # handle multiple elements (ILP).  Hard constraints:
                    #   num_warps × warp_size  ≤  tile_size        (warp-coherence)
                    #   tile_size ≤ TRITON_MAX_TILE_NUMEL           (Triton limit)
                    #   tile_size / threads ≤ MAX_ELEMS_PER_THREAD  (VGPR budget)
                    tile_size = xblock * yblock
                    if tile_size < warp_size:
                        continue
                    if tile_size > PointwiseHeuristics.TRITON_MAX_TILE_NUMEL:
                        continue
                    for nw in warp_candidates:
                        if nw > max_warps or nw * warp_size > tile_size:
                            continue
                        # Register-pressure guard: empirically, every 2-D config with
                        # elems_per_thread ≥ 64 spills on AMD MI300X (100% inf rate).
                        # Cap at MAX_ELEMS_PER_THREAD_MULTIDIM to pre-filter safely.
                        threads_per_block = nw * warp_size
                        elems_per_thread  = tile_size / threads_per_block
                        if elems_per_thread > PointwiseHeuristics.MAX_ELEMS_PER_THREAD_MULTIDIM:
                            continue
                        configs.append({'XBLOCK': xblock, 'YBLOCK': yblock, 'num_warps': nw})
        
        elif ndims == 3:
            xnumel, ynumel, znumel = problem_dims
            for xblock in block_sizes_3d:
                if xblock > xnumel:
                    continue
                for yblock in block_sizes_3d:
                    if yblock > ynumel:
                        continue
                    for zblock in block_sizes_3d:
                        if zblock > znumel:
                            continue
                        # Same constraints as 2-D plus VGPR register-pressure guard.
                        tile_size = xblock * yblock * zblock
                        if tile_size < warp_size:
                            continue
                        if tile_size > PointwiseHeuristics.TRITON_MAX_TILE_NUMEL:
                            continue
                        for nw in warp_candidates:
                            if nw > max_warps or nw * warp_size > tile_size:
                                continue
                            threads_per_block = nw * warp_size
                            elems_per_thread  = tile_size / threads_per_block
                            if elems_per_thread > PointwiseHeuristics.MAX_ELEMS_PER_THREAD_MULTIDIM:
                                continue
                            configs.append({
                                'XBLOCK': xblock, 'YBLOCK': yblock,
                                'ZBLOCK': zblock, 'num_warps': nw,
                            })

        # ── AMD-only: waves_per_eu variants ──────────────────────────────────
        # waves_per_eu instructs the LLVM AMDGPU backend to reserve register
        # space for N concurrent wavefronts per Execution Unit (≈ per SIMD).
        # This trades VGPR budget for inter-block occupancy:
        #
        #   waves_per_eu=2 → VGPR budget per thread = 65536 / (threads × 2)
        #   waves_per_eu=4 → VGPR budget per thread = 65536 / (threads × 4)
        #
        # Effect on performance:
        #   • Memory-bound (simple ops):    lower VGPR need anyway; extra
        #     co-resident blocks improve HBM pipeline utilisation → FASTER.
        #   • Compute-heavy (fused kernels): tight budget causes register
        #     spills → SLOWER.
        #
        # Two pre-generation gates prevent creating configs that can never win:
        #
        #   Gate 1 — VGPR budget: the tighter per-thread VGPR budget must not
        #     exceed the estimated kernel VGPR need (same formula as the scorer).
        #     budget = floor(65536 / (threads × wpe)) rounded to 8-reg granule.
        #     Avoids generating e.g. XBLOCK=512, num_warps=8, wpe=4 (32 VGPRs)
        #     which virtually every non-trivial kernel will spill.
        #
        #   Gate 2 — Grid size: the problem must have enough grid blocks for the
        #     hardware scheduler to actually co-reside N wavefronts per SIMD.
        #     Requires num_blocks_est ≥ num_cus × wpe.  Without this, the extra
        #     wavefront slots sit empty — pure VGPR cost, zero occupancy gain.
        #
        # Only generated on HIP builds when explicitly enabled via env var.
        # Double guard: _get_is_hip() ensures CUDA is never affected;
        # _is_waves_per_eu_enabled() requires the opt-in flag so existing
        # benchmarks are not disrupted until the feature is validated.
        if PointwiseHeuristics._get_is_hip() and PointwiseHeuristics._is_waves_per_eu_enabled():
            _arch          = PointwiseHeuristics._get_arch()
            _num_cus       = getattr(_arch, 'num_cus', 110)
            _total_el      = problem_metadata.get('total_elements', 1)
            _ops_pe        = problem_metadata.get('ops_per_element', 3)
            _load_ops      = problem_metadata.get('load_ops', 0)
            # VGPR estimate: base overhead + tensor arg overhead + op temporaries.
            # Use load_ops as proxy for n_input_tensors (+1 for the output).
            # Multiply by 4 per op (was 2) to better capture actual register usage
            # in fused kernels with loop-carried state (temporaries across iterations).
            # Base raised from 16→24 to better capture untracked overhead (index
            # arithmetic, loop counters, predicate registers, etc.) which is
            # consistently under-counted, leading to OutOfResources failures.
            _n_tensors_est = max(3, _load_ops + 1)
            _vgpr_need     = 24 + _n_tensors_est * 4 + min(int(_ops_pe) * 4, 128)

            # Gate 3: Compute intensity — very simple ops (launch/memory-bound)
            # don't benefit from the occupancy boost that waves_per_eu provides.
            # Generating wpe variants for them only adds compilation overhead in
            # REAL_BENCH mode and can cause measurement noise.
            # Threshold of 8 weighted ops/element excludes add/mul/leaky_relu/silu
            # (≤6 ops) while keeping gelu (14), bias_relu, fused, heavy_* kernels.
            _skip_wpe  = (_ops_pe < 8)
            _slow_ops  = problem_metadata.get('slow_ops', 0)

            # Gate 4: Dimensionality — skip waves_per_eu for 2D/3D problems.
            # Multi-dimensional kernels (YBLOCK/ZBLOCK tiling) already leverage
            # the XBLOCK × YBLOCK × ... grid for occupancy.  Adding waves_per_eu
            # triples an already-large config space (70 → 200-860+ for 3D shapes)
            # without empirical benefit (wpe is never chosen for multi-dim shapes
            # in practice), while REAL_BENCH GPU benchmarking of the expanded set
            # thermally throttles the GPU, degrading the last few shapes in a
            # multi-shape benchmark run (3D_batch_video, 3D_odd_*).
            _problem_ndims = len(PointwiseHeuristics.get_problem_dimensions(problem_metadata))
            _wpe_allowed_by_dims = (_problem_ndims == 1)

            for _cfg in list(configs):   # snapshot — iterate base configs only
                if _skip_wpe or not _wpe_allowed_by_dims:
                    continue
                nw   = _cfg['num_warps']
                tile = (  _cfg.get('XBLOCK', 1)
                        * _cfg.get('YBLOCK', 1)
                        * _cfg.get('ZBLOCK', 1))
                _num_blocks_est = max(1, math.ceil(_total_el / max(1, tile)))
                _threads        = nw * warp_size

                for _wpe in (1, 2, 4):
                    # wpe=1: signal to LLVM to allocate max VGPRs (1 concurrent
                    # wave per EU).  Only beneficial for compute-heavy kernels
                    # dominated by transcendental ops (tanh/sin/cos/exp fused)
                    # where VGPR pressure — not occupancy — is the bottleneck.
                    # The VGPR budget at wpe=1 is identical to wpe=0 (256), so
                    # there is no risk of spilling; the grid gate still applies.
                    if _wpe == 1:
                        if _slow_ops < 4:
                            continue  # only for heavy transcendental fusions
                        # Grid must have at least num_cus blocks to keep all CUs
                        # busy (since each EU runs only 1 wave at a time).
                        if _num_blocks_est < _num_cus:
                            continue
                        _new = dict(_cfg)
                        _new['waves_per_eu'] = 1
                        configs.append(_new)
                        continue

                    # Gate 1: VGPR budget — must leave headroom for the compiler.
                    _raw_budget  = 65536 // max(1, _threads * _wpe)
                    _wpe_budget  = min(256, (_raw_budget // 8) * 8)
                    # Hard minimum: a budget below 96 VGPRs is too tight for any
                    # non-trivial fused kernel.  High-warp configs (num_warps=8)
                    # with wpe=2 land at exactly 64 VGPRs, which causes systematic
                    # OutOfResources failures on complex kernels (heavy_mlp,
                    # heavy_attn, etc.).  Skip these entirely.
                    if _wpe_budget < 96:
                        continue   # absolute budget floor; skip
                    if _vgpr_need >= _wpe_budget * 0.75:
                        continue   # would spill; skip (tightened from 0.80)
                    # Gate 2: grid must be large enough to fill the extra slots.
                    if _num_blocks_est < _num_cus * _wpe:
                        continue   # slots would be empty; skip
                    _new = dict(_cfg)
                    _new['waves_per_eu'] = _wpe
                    configs.append(_new)

        return configs
    
    # -------------------------------------------------------------------------
    # Pruning / ranking
    # -------------------------------------------------------------------------
    
    @staticmethod
    def prune_configs(configs: List[Dict], 
                     problem_metadata: Dict,
                      top_n: int = 12,
                      kernel_code: Optional[str] = None) -> List[Dict]:
        """Validate, score, and return the top-N configs by score.

        Validation rejects configs that would produce illegal Triton launches:
          - dimension count mismatch (config dims ≠ problem dims)
          - block size out of range
          - threads_per_block < 64 (can't fill a full wavefront)
          - threads_per_block > 1024 (hardware limit)
          - excessively large grids (> 10 M blocks)

        A diversity pass limits the returned list to at most 2 configs per
        distinct XBLOCK value, ensuring the top-N pool covers multiple block
        widths.  Without this, XBLOCK=1024 variants (varying only in num_warps)
        can fill all top-N slots and crowd out XBLOCK=512/256 candidates that
        the scoring model rates only slightly lower but often win empirically.
        If the diversity cap would leave fewer than top_n total entries, the
        remaining slots are filled with overflow configs (uncapped, best-first).
        """
        problem_dims   = PointwiseHeuristics.get_problem_dimensions(problem_metadata)
        total_elements = PointwiseHeuristics.prod(problem_dims)
        ndims          = len(problem_dims)
        
        scored: List[Tuple[float, Dict]] = []
        
        for cfg in configs:
            try:
                block_dims = PointwiseHeuristics.get_block_dimensions(cfg)
                
                if len(block_dims) != len(problem_dims):
                    continue
                
                # Per-dimension block-size bounds.
                invalid = False
                grid_size  = PointwiseHeuristics.calculate_grid_size(problem_dims, block_dims)
                num_blocks = PointwiseHeuristics.prod(grid_size)
                
                for block_dim, problem_dim in zip(block_dims, problem_dims):
                    # MIN_BLOCK_SIZE=1 applies uniformly.  The warp-coherence
                    # check below (hw_threads ≤ 1024 and tile ≥ hw_threads) is
                    # the real safeguard against degenerate configs; forcing a
                    # separate per-dim floor of 4 for 3-D kernels incorrectly
                    # rejects valid configs such as ZBLOCK=2 when ZBLOCK ≤ Z/2.
                    min_for_dim = PointwiseHeuristics.MIN_BLOCK_SIZE
                    if not (min_for_dim <= block_dim <= PointwiseHeuristics.MAX_BLOCK_SIZE):
                        invalid = True
                        break
                if invalid:
                    continue
                
                # Use actual hardware thread count (num_warps × warp_size), not the
                # tile size.  For 1-D kernels prod(block_dims) == XBLOCK, which can
                # exceed 1024 while the real thread count is perfectly legal.
                hw_threads  = cfg.get('num_warps', 1) * problem_metadata.get('warp_size', 64)
                min_threads = 16 if total_elements <= 64 else 64
                if not (min_threads <= hw_threads <= 1024):
                    continue
                
                # Avoid huge blocks on very small problems (most threads would be idle).
                if total_elements < 10000 and hw_threads > 512:
                    continue

                if num_blocks > 10_000_000:
                    continue
                
                s = PointwiseHeuristics.score_config(cfg, problem_metadata, kernel_code)
                if s > 0:
                    scored.append((s, cfg))
                
            except Exception:
                continue
        
        # Primary sort: score descending.  Secondary sort: XBLOCK descending as a
        # tiebreaker for near-equal scores.
        #
        # Why rounding? When Grid is at its floor (0.75) for all configs — which
        # happens for very large problems where every valid block count far exceeds
        # the target — all XBLOCK values in a warp-group score identically on Grid,
        # BW, Launch, and Occ.  However, the bottleneck analysis computes a
        # different T_overhead per config (more blocks → more overhead), which
        # shifts the exponents in the geometric mean by ~1e-7.  This tiny
        # difference makes the sort pick the smallest XBLOCK (first generated)
        # arbitrarily rather than the more ILP-efficient larger tile.
        #
        # Rounding to 4 decimal places collapses configs that differ by < 5e-5 into
        # a single score bucket, and within a bucket the XBLOCK tiebreaker picks
        # the largest tile — which empirically tends to win on large streaming
        # workloads (higher elements-per-thread → better pipelining).
        scored.sort(
            reverse=True,
            key=lambda x: (round(x[0], 4), x[1].get('XBLOCK', 0)),
        )

        # Diversity pass: ensure the top-N pool covers a wide range of tile widths
        # AND warp counts.
        #
        # Diversity cap rationale:
        #
        # Diversity cap rationale (updated):
        #
        # For all XBLOCK sizes:  cap = 2 (both nw=4 and nw=8 variants included)
        #
        # Previously, large XBLOCK (>256) used cap=1 on the assumption that the
        # scoring model reliably picked the better warp count.  After the BW
        # saturation fix (BW is flat at 0.97 for any nw≥4 when the grid has ≥2048
        # blocks) the scoring intentionally makes nw=4 and nw=8 equivalent — grid
        # score and XBLOCK size differentiate configs, but warp count is left to
        # the real bench.  Keeping cap=1 would mean only one of {nw=4, nw=8} per
        # XBLOCK reaches the bench pool, making the other permanently invisible to
        # the real-bench stage regardless of how many candidates (TOP_N) are used.
        #
        # cap=2 for all XBLOCK ensures that both the 4-warp and 8-warp variants
        # of the best XBLOCK values are bench-tested.  Empirically:
        #   - silu / leaky_relu (simple ops):    nw=8 is 15-28 % faster  → bench picks it
        #   - mul_0 / add_gelu_mul (large tiles): nw=4 is 15 % faster    → bench picks it
        #   - tanh / multi-SFU kernels (small):   EPT bonus already distinguishes
        #
        # The trade-off: top_n=5 now covers ~2-3 distinct XBLOCK values (with both
        # nw variants each) instead of ~4-5 values (one nw variant each).  Because
        # the grid-target formula now correctly ranks XBLOCK values, the 2-3
        # XBLOCK values admitted are the right ones; sacrificing one extra XBLOCK
        # width to gain the nw discrimination is empirically worthwhile.
        #
        # Overflow configs fill any remaining slots so the pool always reaches
        # exactly top_n entries when enough valid configs exist.
        xblock_counts: Dict[int, int] = {}
        primary:  List[Tuple[float, Dict]] = []
        overflow: List[Tuple[float, Dict]] = []
        for s, cfg in scored:
            xb  = cfg.get('XBLOCK', 0)
            cap = 2  # allow both nw=4 and nw=8 variants per XBLOCK
            if xblock_counts.get(xb, 0) < cap:
                primary.append((s, cfg))
                xblock_counts[xb] = xblock_counts.get(xb, 0) + 1
            else:
                overflow.append((s, cfg))

        selected = primary[:top_n]
        if len(selected) < top_n:
            selected.extend(overflow[:top_n - len(selected)])

        # Four-warp guarantee: ensure at least one num_warps=4 config reaches the
        # benchmark pool.  Without this, 3-D kernels with many YBLOCK×ZBLOCK shape
        # variants can fill every diversity-cap slot with 8-warp configs, completely
        # evicting the 4-warp variants.  Empirically, for compute-heavy fused
        # kernels (many ops per element), the 4-warp variant often wins because
        # fewer threads → more VGPRs per thread → the compiler avoids register
        # spills.  Guaranteeing one slot costs at most one suboptimal benchmark
        # invocation per problem shape; the runtime overhead is negligible.
        #
        # Guard: only apply when top_n ≥ 3.  For top_n=1 (get_optimal_config)
        # we want the single highest-scoring config unconditionally.
        #
        # Implementation: if the selected pool contains no num_warps=4 config, find
        # the highest-scoring 4-warp config not already selected and swap it in for
        # the lowest-scored entry currently in the pool.
        if top_n >= 3:
            selected_ids = {id(cfg) for _, cfg in selected}
            has_four_warp = any(cfg.get('num_warps', 8) == 4 for _, cfg in selected)
            if not has_four_warp:
                best_4w = next(
                    ((s, cfg) for s, cfg in scored
                     if cfg.get('num_warps', 8) == 4 and id(cfg) not in selected_ids),
                    None,
                )
                if best_4w is not None:
                    selected.sort(key=lambda x: x[0])   # ascending → [0] is lowest
                    # Only inject if the guarantee config scores ≥ the pool member it
                    # would replace.  Without this guard the guarantee can displace a
                    # higher-scoring config with a weaker one, actively degrading pool
                    # quality (e.g. replacing rank-3 x=512 nw=1 with rank-11 x=4096 nw=4).
                    if best_4w[0] >= selected[0][0]:
                        selected[0] = best_4w
                    selected.sort(key=lambda x: x[0], reverse=True)  # restore descending

        # Large-XBLOCK guarantee — for 1-D kernels with total_elements ≥ 16384,
        # ensure the pool contains at least one config with XBLOCK=4096 nw=4.
        #
        # Rationale: for large memory-bound kernels (leaky_relu, add_0, etc.) at
        # 4 M–16 M elements, AMD MI300X consistently runs 15 %+ faster with
        # XBLOCK=4096 nw=4 (ept=16) than with the scoring model's top choices
        # (XBLOCK=512–1024).  The higher EPT enables 4-wide vectorised loads
        # (128-bit) per thread, cutting instruction count and improving HBM
        # streaming efficiency.  For complex compute kernels (gelu, tanh) the
        # large XBLOCK is slower — but the real bench then prefers the scoring
        # model's choice, so the cost is just one extra benchmark invocation.
        #
        # Similarly, for smaller 1-D kernels (16 K–512 K elements), ILP-heavy
        # kernels (e.g. add_div_exp_var_7 x=32 K) also benefit from XBLOCK=4096
        # when the compiler can batch multiple SFU ops per wavefront instruction.
        #
        # Only applied to 1-D problems (ndims==1) at top_n ≥ 5 so that 2-D / 3-D
        # kernels and N=1 selection are not affected.
        if top_n >= 5 and ndims == 1 and total_elements >= 16384:
            selected_ids = {id(cfg) for _, cfg in selected}
            has_large_xb = any(cfg.get('XBLOCK', 0) >= 4096 for _, cfg in selected)
            if not has_large_xb:
                best_large = next(
                    ((s, cfg) for s, cfg in scored
                     if cfg.get('XBLOCK', 0) >= 4096
                     and cfg.get('num_warps', 8) == 4
                     and id(cfg) not in selected_ids),
                    None,
                )
                if best_large is not None:
                    selected.sort(key=lambda x: x[0])   # ascending → [0] is lowest
                    # Protected injection: only replace if the guarantee config
                    # scores ≥ the pool member it displaces.
                    if best_large[0] >= selected[0][0]:
                        selected[0] = best_large
                    selected.sort(key=lambda x: x[0], reverse=True)

        # Low-warp large-XBLOCK guarantee — for compute-heavy 1-D kernels (slow_ops ≥ 2),
        # also ensure the pool contains at least one config with XBLOCK≥4096 nw=2.
        #
        # Rationale: for kernels with transcendental ops (exp, sqrt, var) nw=2 provides
        # 2× the VGPR budget vs nw=4 (128 vs 64 VGPRs/thread on MI300X at max occupancy).
        # With large XBLOCK (EPT=32) this allows the compiler to hold all element-level
        # intermediates in registers without spilling, often outperforming nw=4 by 15-30%.
        # Example: var_7 x=262K — x=4096 nw=2 (9.68µs) >> x=4096 nw=4 (12.56µs).
        #
        # Gated on slow_ops ≥ 2 and top_n ≥ 5 to avoid interfering with pure-streaming
        # kernels (slow_ops=0) where nw=8 dominates and for single-config N=1 selection.
        if top_n >= 5 and ndims == 1 and total_elements >= 16384:
            _slow_ops_nw2 = problem_metadata.get('slow_ops', 0)
            if _slow_ops_nw2 >= 2:
                selected_ids_nw2 = {id(cfg) for _, cfg in selected}
                has_large_xb_nw2 = any(
                    cfg.get('XBLOCK', 0) >= 4096 and cfg.get('num_warps', 8) == 2
                    for _, cfg in selected
                )
                if not has_large_xb_nw2:
                    best_large_nw2 = next(
                        ((s, cfg) for s, cfg in scored
                         if cfg.get('XBLOCK', 0) >= 4096
                         and cfg.get('num_warps', 8) == 2
                         and id(cfg) not in selected_ids_nw2),
                        None,
                    )
                    if best_large_nw2 is not None:
                        selected.sort(key=lambda x: x[0])
                        # Protected injection.
                        if best_large_nw2[0] >= selected[0][0]:
                            selected[0] = best_large_nw2
                        selected.sort(key=lambda x: x[0], reverse=True)

        # Low-nw guarantee — for large 1-D kernels ensure the pool contains at
        # least one config with num_warps ≤ 2.  For streaming/transcendental
        # kernels (gelu, tanh, mul_relu) at ≥64K elements, the nw=1/2 regime
        # provides 2–8× the per-thread VGPR budget vs nw=8, allowing the
        # compiler to keep more transcendental-op intermediates in registers and
        # pipeline loads/stores across multiple element iterations (EPT ≥ 4).
        # Empirically this outperforms nw=8 by 1.1–1.9× at ≥1M elements.
        #
        # Protected by the score-threshold guard: only injects when the best
        # nw≤2 candidate scores AT LEAST as high as the pool member it replaces,
        # so pool quality never degrades.
        if top_n >= 5 and ndims == 1 and total_elements >= 65536:
            _sel_ids_lownw = {id(cfg) for _, cfg in selected}
            _has_low_nw    = any(cfg.get('num_warps', 8) <= 2 for _, cfg in selected)
            if not _has_low_nw:
                _best_lownw = next(
                    ((s, cfg) for s, cfg in scored
                     if cfg.get('num_warps', 8) <= 2
                     and id(cfg) not in _sel_ids_lownw),
                    None,
                )
                if _best_lownw is not None:
                    selected.sort(key=lambda x: x[0])  # [0] = lowest score
                    # Protected injection: preserve pool quality.
                    if _best_lownw[0] >= selected[0][0]:
                        selected[0] = _best_lownw
                    selected.sort(key=lambda x: x[0], reverse=True)

        return [cfg for _, cfg in selected]

    @staticmethod
    def get_optimal_config(problem_metadata: Dict) -> Dict:
        """Return the single highest-scoring config for a problem."""
        all_configs = PointwiseHeuristics.generate_all_candidate_configs(problem_metadata)
        top         = PointwiseHeuristics.prune_configs(all_configs, problem_metadata, top_n=1)
        
        if top:
            return top[0]
        
        # Safe fallback if all configs were pruned (shouldn't happen in practice).
        ndims = len(PointwiseHeuristics.get_problem_dimensions(problem_metadata))
        if ndims == 1:
            return {'XBLOCK': 256, 'num_warps': 4}
        elif ndims == 2:
            return {'XBLOCK': 128, 'YBLOCK': 64, 'num_warps': 8}
            return {'XBLOCK': 32, 'YBLOCK': 32, 'ZBLOCK': 8, 'num_warps': 8}
    
    # -------------------------------------------------------------------------
    # Debug / introspection
    # -------------------------------------------------------------------------
    
    @staticmethod
    def get_detailed_scores(config: Dict,
                            problem_metadata: Dict,
                            kernel_code: Optional[str] = None) -> Dict:
        """Return a per-factor breakdown for debugging and the verbose log table.

        Returns a dict with keys:
            memory_bandwidth, launch_overhead, grid_granularity, occupancy,
            composite, num_blocks, threads_per_block.
        All float values are in [0.0, 1.0]; num_blocks and threads_per_block are int.
        """
        _zero = {
                    'memory_bandwidth': 0.0,
            'launch_overhead':  0.0,
                    'grid_granularity': 0.0,
            'occupancy':        0.0,
            'composite':        0.0,
            'num_blocks':       0,
                    'threads_per_block': 0,
                }
            
        try:
            block_dims   = PointwiseHeuristics.get_block_dimensions(config)
            problem_dims = PointwiseHeuristics.get_problem_dimensions(problem_metadata)

            if len(block_dims) != len(problem_dims):
                return _zero

            grid_size = PointwiseHeuristics.calculate_grid_size(problem_dims, block_dims)

            return {
                'memory_bandwidth': PointwiseHeuristics.estimate_memory_bandwidth(
                    config, problem_metadata),
                'launch_overhead':  PointwiseHeuristics.estimate_launch_overhead(
                    grid_size, problem_metadata),
                'grid_granularity': PointwiseHeuristics.estimate_grid_granularity(
                    grid_size, problem_metadata),
                'occupancy':        PointwiseHeuristics.estimate_occupancy_impact(
                    config, problem_metadata),
                'composite':        PointwiseHeuristics.score_config(
                    config, problem_metadata, kernel_code),
                'num_blocks':       PointwiseHeuristics.prod(grid_size),
                'threads_per_block': PointwiseHeuristics.prod(block_dims),
            }

        except Exception:
            return _zero

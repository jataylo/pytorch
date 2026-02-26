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
from functools import lru_cache, reduce
from typing import Dict, List, Optional, Tuple

try:
    from .triton_heuristics_hardware import get_architecture_config
except ImportError:
    get_architecture_config = None  # type: ignore[assignment]

try:
    from .triton_heuristics_adaptive import BottleneckAnalysis
except ImportError:
    BottleneckAnalysis = None  # type: ignore[assignment]

__all__ = ['PointwiseHeuristics']


class PointwiseHeuristics:
    """Static scoring and config-generation heuristics for pointwise kernels.

    All methods are @staticmethod or @classmethod; the class is never
    instantiated.  The class-level _arch_config cache is a process-level
    singleton (one GPU assumed; see HEURISTICS_DESIGN.md §8.9).
    """

    _arch_config = None

    # Hard limits on per-dimension block size.
    MIN_BLOCK_SIZE = 16
    MAX_BLOCK_SIZE = 2048

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @classmethod
    def _get_arch(cls):
        """Return the cached ArchitectureConfig, initialising on first call.

        Under normal operation this is pre-warmed by pre_fork_setup() before
        any worker process is forked, so the value is inherited and no device
        query is needed here.  The try/except is a safety net for unit-test
        environments or non-standard pool configurations where pre_fork_setup()
        was not called (e.g. spawn-mode workers or direct in-process calls).
        """
        if cls._arch_config is None:
            if get_architecture_config is not None:
                try:
                    cls._arch_config = get_architecture_config()
                except Exception:
                    # pre_fork_setup() should have pre-warmed _arch_config via
                    # PointwiseHeuristics._get_arch() before workers were forked.
                    # Reaching here means something unusual happened (e.g. a
                    # unit test bypassed pre_fork_setup, or the CUDA context was
                    # initialised after the fork point).  Fall through to the
                    # SimpleNamespace defaults so scoring stays functional.
                    pass
            if cls._arch_config is None:
                from types import SimpleNamespace
                cls._arch_config = SimpleNamespace(
                    num_cus=256,
                    warp_size=64,
                    optimal_threads_bandwidth=512,
                    optimal_blocks_grid=512,
                    occupancy_sweetspot_min=4,
                    occupancy_sweetspot_max=8,
                    optimal_elements_per_block=1024,
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

        Modelled as a Gaussian centred at arch.optimal_threads_bandwidth with
        σ = 1.5 × optimal_threads.  The wider sigma keeps XBLOCK=512 and
        XBLOCK=256 competitive (gap vs XBLOCK=1024 shrinks from 6 % to 3 %),
        preventing the top-N pool from being flooded by XBLOCK=1024 variants.
        On MI300X, optimal_threads_bandwidth ≈ 1536 threads (24 wavefronts).

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
        block_dims        = PointwiseHeuristics.get_block_dimensions(config)
        threads_per_block = PointwiseHeuristics.prod(block_dims)

        arch            = PointwiseHeuristics._get_arch()
        optimal_threads = arch.optimal_threads_bandwidth

        xblock = config.get('XBLOCK', 256)
        yblock = config.get('YBLOCK', 0)

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

        # 2-D coalescing correction for X tile width.
        #
        # On AMD (warp_size=64), Triton maps the fast thread index to X.
        # Within each row the X access is always contiguous (cache-line-aligned),
        # so "coalescing" in the classic sense is fine for any XBLOCK ≥ 1.
        #
        # The legacy concern — that XBLOCK < warp_size causes a wavefront to
        # span multiple rows — is actually NEUTRAL-to-BENEFICIAL on large HBM
        # systems (e.g. MI300X with 8 independent HBM stacks):
        #   • Two rows in one wavefront → two independent 128-byte accesses that
        #     can target different HBM channels simultaneously.
        #   • One wide row in one wavefront → one 256-byte burst on a single channel.
        # Empirically, XBLOCK=32 with num_warps=1 is up to 15 % faster than
        # XBLOCK=64 on 8192×8192 tensors on MI300X, confirming the multi-channel
        # benefit outweighs any sequencing overhead.
        #
        # We therefore only penalise XBLOCK values that are sub-cache-line
        # (XBLOCK < 16 = 64 B / 4 B), where each row access wastes cache-line
        # bandwidth by not filling a full 64-byte line:
        #
        #   XBLOCK ≥ 16 →  factor 1.00  (full or multi-CL per row; no penalty)
        #   XBLOCK <  16 →  factor 0.75–0.98  (partial cache-line utilisation)
        if yblock > 0:
            cache_line_elems = 16                  # 64 B / 4 B per FP32

            if xblock >= cache_line_elems:         # ≥ 16 elements = full cache line
                # Each row access is at least one full cache line.  No penalty
                # regardless of whether the wavefront spans multiple rows.
                warp_align = 1.00
            else:
                # Sub-cache-line X width: row access doesn't fill a 64-byte line.
                # Penalise proportionally to unused bandwidth per cache line.
                coalescing = xblock / cache_line_elems   # fraction of CL used
                warp_align = 0.75 + 0.23 * coalescing   # 0.75 (XBLOCK=1) → 0.98 (XBLOCK=15)

            score *= warp_align

        return score

    # -------------------------------------------------------------------------
    # Factor 2 — Launch overhead amortisation
    # -------------------------------------------------------------------------

    @staticmethod
    def estimate_launch_overhead(grid_size: Tuple[int, ...],
                                 problem_metadata: Dict) -> float:
        """Score how well this config amortises the fixed kernel dispatch cost.

        Modelled as a Gaussian centred at arch.optimal_elements_per_block (the
        elements-per-block value at which the ~3 µs dispatch cost is amortised
        to < 5 % of wall time).  Score floor is 0.70 for EPB < 64 (overhead
        completely dominates at that point).

        One secondary penalty applies:

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
        sigma        = optimal_epb // 2

        if elements_per_block < 64:
            score = 0.70
        else:
            diff     = (elements_per_block - optimal_epb) / sigma
            gaussian = math.exp(-0.5 * diff * diff)
            score    = max(0.70, min(1.0, 0.75 + 0.25 * gaussian))

        # Large-grid scheduler overhead: Command Processor must batch-dispatch
        # blocks beyond 4×CUs, adding measurable latency.
        num_cus = arch.num_cus
        max_good_blocks = num_cus * 4
        if num_blocks > max_good_blocks:
            excess       = num_blocks / max_good_blocks
            grid_penalty = 1.0 - 0.01 * math.log2(excess)
            score       *= max(0.88, grid_penalty)

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

        if total_elements < 16384:
            optimal = max(4, hardware_optimal // 32)
        elif total_elements < 262144:
            optimal = hardware_optimal // 2
        else:
            optimal = hardware_optimal

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
    def estimate_occupancy_impact(config: Dict, problem_metadata: Dict) -> float:
        """Score whether the wavefront count per CU hides latency without wasting resources.

        Two regimes selected by grid saturation level and block count:

        Well-saturated (saturation ≥ 0.25) or single-block:
            Classic sweet-spot model — num_warps in [sweet_min, sweet_max] (4–8
            on AMD) scores 1.0.  Derived from Little's Law: need ceil(300/40) ≈ 8
            wavefronts to hide ~300-cycle HBM latency with a 40-cycle issue gap.
            4 suffices when L2 absorbs enough traffic to cut effective latency.

        Launch-bound multi-block (saturation < 0.25, num_blocks > 1):
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
        thread, which fully hides HBM round-trip latency without needing extra
        wavefronts.  The "single wavefront stalls" concern therefore disappears
        and num_warps = 1 is treated the same as being in the sweet spot.

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

        def _sweet_spot_score(nw: int) -> float:
            if sweet_min <= nw <= sweet_max:
                return 1.00
            elif sweet_min // 2 <= nw <= int(sweet_max * 1.5):
                return 0.95
            elif nw == 1:
                # Single wavefront: normally stalls on every memory access.
                # But if each thread processes ≥ 8 elements the compiler issues
                # all of them as independent loads → latency is fully hidden.
                if elems_per_thread >= 8:
                    return 1.00   # ILP compensates entirely
                elif elems_per_thread >= 4:
                    return 0.93   # Partial compensation
                return 0.85
            # nw == 2 or 3: between sweet_min//2 and sweet_min
            if elems_per_thread >= 8:
                return 1.00
            return 0.75

        if saturation >= 0.25 or num_blocks_est == 1:
            # Well-saturated or single-block: latency hiding governs.
            base_score = _sweet_spot_score(num_warps)
        else:
            # Launch-bound multi-block. Score relative to the natural warp count
            # for this block — the number of wavefronts needed for one thread
            # per element.  Going below natural forces serial element iteration
            # (slower); going above adds SPI init overhead (also slower).
            natural_warps = max(1, xblock_prod // warp_size)

            try:
                from torch._inductor.codegen.triton_heuristics_adaptive import (
                    BottleneckAnalysis as _BA,
                )
                k_launch = _BA.KERNEL_LAUNCH_US
            except Exception:
                k_launch = 3.0

            k_warp = 0.2   # µs per extra wavefront beyond natural (SPI alloc)

            if num_warps <= natural_warps:
                # Under-warped: each warp iterates over multiple element chunks.
                base_score = 0.75 + 0.25 * (num_warps / natural_warps)
            else:
                # Over-warped: purely extra SPI init with no data benefit.
                extra = num_warps - natural_warps
                t_config = k_launch + extra * k_warp
                base_score = k_launch / t_config

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
            block_dims   = PointwiseHeuristics.get_block_dimensions(config)
            problem_dims = PointwiseHeuristics.get_problem_dimensions(problem_metadata)

            if len(block_dims) != len(problem_dims):
                return 0.0

            grid_size = PointwiseHeuristics.calculate_grid_size(problem_dims, block_dims)

            bandwidth   = PointwiseHeuristics.estimate_memory_bandwidth(config, problem_metadata)
            launch      = PointwiseHeuristics.estimate_launch_overhead(grid_size, problem_metadata)
            granularity = PointwiseHeuristics.estimate_grid_granularity(grid_size, problem_metadata)
            occupancy   = PointwiseHeuristics.estimate_occupancy_impact(config, problem_metadata)

            if BottleneckAnalysis is not None:
                try:
                    weights   = BottleneckAnalysis.get_adaptive_weights(
                        config, problem_metadata, kernel_code
                    )
                    exponents = BottleneckAnalysis.get_adaptive_exponents(weights)
                    bw_exp    = exponents['bandwidth']
                    launch_exp = exponents['launch']
                    grid_exp  = exponents['grid']
                    occ_exp   = exponents['occupancy']
                except Exception:
                    bw_exp, launch_exp, grid_exp, occ_exp = 2.5, 1.8, 1.2, 0.6
            else:
                bw_exp, launch_exp, grid_exp, occ_exp = 2.5, 1.8, 1.2, 0.6

            total_exp = bw_exp + launch_exp + grid_exp + occ_exp
            score = (
                (bandwidth   ** bw_exp)    *
                (launch      ** launch_exp) *
                (granularity ** grid_exp)  *
                (occupancy   ** occ_exp)
            ) ** (1.0 / total_exp)

            # 2-D tie-breaker: mildly prefer square, wide-X tiles.
            if len(block_dims) == 2:
                xblock, yblock = block_dims

                ratio              = max(xblock, yblock) / max(min(xblock, yblock), 1)
                balance_multiplier = 1.0 - 0.005 * math.log2(max(ratio, 1.0))

                # xblock=256→1.00, xblock=128→0.992, xblock=8→0.960, xblock=4→0.952
                innermost_multiplier = 1.0 - 0.008 * max(0, 5 - math.log2(max(xblock, 4)))

                score *= max(0.95, balance_multiplier * innermost_multiplier)

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.0

    # -------------------------------------------------------------------------
    # Config generation
    # -------------------------------------------------------------------------

    @staticmethod
    def generate_all_candidate_configs(problem_metadata: Dict) -> List[Dict]:
        """Enumerate every legal (XBLOCK[×YBLOCK[×ZBLOCK]], num_warps) pair.

        Two hard constraints are applied before any config is accepted:
          1. Dimension cap: XBLOCK ≤ xnumel, YBLOCK ≤ ynumel, etc.
          2. Warp–thread coherence: num_warps × warp_size ≤ total_threads_per_block.
             (Declaring more warps than threads would be silently clamped by the
             hardware, making the declared num_warps misleading to the compiler.)

        Block size candidates are powers of two.  Non-power-of-two sizes trigger
        tail-masking in Triton's codegen and prevent full unroll/vectorisation.

        Returns 30–80 configs for 1-D problems, 60–120 for 2-D.
        """
        problem_dims = PointwiseHeuristics.get_problem_dimensions(problem_metadata)
        ndims        = len(problem_dims)
        warp_size    = problem_metadata.get('warp_size', 64)
        max_threads  = problem_metadata.get('max_threads_per_block', 1024)
        max_warps    = max_threads // warp_size

        warp_candidates    = [1, 2, 4, 8, 16]
        block_sizes_1d     = [16, 32, 64, 128, 256, 512, 1024]
        block_sizes_2d     = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        block_sizes_3d     = [4, 8, 16, 32, 64]

        configs: List[Dict] = []

        if ndims == 1:
            xnumel = problem_dims[0]
            for xblock in block_sizes_1d:
                if xblock > xnumel:
                    continue
                for nw in warp_candidates:
                    if nw > max_warps or nw * warp_size > xblock:
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
                    total_threads = xblock * yblock
                    if not (64 <= total_threads <= max_threads):
                        continue
                    for nw in warp_candidates:
                        if nw > max_warps or nw * warp_size > total_threads:
                            continue
                        configs.append({'XBLOCK': xblock, 'YBLOCK': yblock, 'num_warps': nw})

        elif ndims == 3:
            xnumel, ynumel, znumel = problem_dims
            max_threads_3d = min(max_threads, 1024)
            for xblock in block_sizes_3d:
                if xblock > xnumel:
                    continue
                for yblock in block_sizes_3d:
                    if yblock > ynumel:
                        continue
                    for zblock in block_sizes_3d:
                        if zblock > znumel:
                            continue
                        total_threads = xblock * yblock * zblock
                        if not (64 <= total_threads <= max_threads_3d):
                            continue
                        for nw in warp_candidates:
                            if nw > max_warps or nw * warp_size > total_threads:
                                continue
                            configs.append({
                                'XBLOCK': xblock, 'YBLOCK': yblock,
                                'ZBLOCK': zblock, 'num_warps': nw,
                            })

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
                    # Allow blocks as small as 4 for 3-D problems or small dimensions.
                    if ndims == 3 or problem_dim <= 32 or num_blocks <= 1024:
                        min_for_dim = 4
                    else:
                        min_for_dim = PointwiseHeuristics.MIN_BLOCK_SIZE
                    if not (min_for_dim <= block_dim <= PointwiseHeuristics.MAX_BLOCK_SIZE):
                        invalid = True
                        break
                if invalid:
                    continue

                threads_per_block = PointwiseHeuristics.prod(block_dims)
                min_threads       = 16 if total_elements <= 64 else 64
                if not (min_threads <= threads_per_block <= 1024):
                    continue

                # Avoid huge blocks on very small problems (most threads would be idle).
                if total_elements < 10000 and threads_per_block > 512:
                    continue

                if num_blocks > 10_000_000:
                    continue

                s = PointwiseHeuristics.score_config(cfg, problem_metadata, kernel_code)
                if s > 0:
                    scored.append((s, cfg))

            except Exception:
                continue

        scored.sort(reverse=True, key=lambda x: x[0])

        # Diversity pass: allow at most 2 configs per distinct XBLOCK value so
        # that the top-N pool covers a range of block widths rather than being
        # flooded by XBLOCK=1024 variants with different num_warps.  Overflow
        # configs (those that hit the per-bucket cap) fill remaining slots so
        # the returned list always has exactly top_n entries when enough valid
        # configs exist.
        max_per_xblock = 2
        xblock_counts: Dict[int, int] = {}
        primary:  List[Tuple[float, Dict]] = []
        overflow: List[Tuple[float, Dict]] = []
        for s, cfg in scored:
            xb = cfg.get('XBLOCK', 0)
            if xblock_counts.get(xb, 0) < max_per_xblock:
                primary.append((s, cfg))
                xblock_counts[xb] = xblock_counts.get(xb, 0) + 1
            else:
                overflow.append((s, cfg))

        selected = primary[:top_n]
        if len(selected) < top_n:
            selected.extend(overflow[:top_n - len(selected)])

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

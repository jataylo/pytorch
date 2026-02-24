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
        """Return the cached ArchitectureConfig, initialising on first call."""
        if cls._arch_config is None:
            if get_architecture_config is not None:
                cls._arch_config = get_architecture_config()
            else:
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

        Modelled as a Gaussian centred at arch.optimal_threads_bandwidth (512
        threads / 8 wavefronts on AMD MI300X, from Little's Law: wavefronts
        needed = ceil(300 cycles / 40 cycles) ≈ 8).  Score is in [0.60, 1.00]:
          - floor 0.75 for threads ≥ 64 (symmetric decay on both sides)
          - hard floor 0.60 for threads < 64 (less than one full wavefront)

        For 2-D kernels a coalescing correction is applied.  The X dimension
        is contiguous in memory (row-major); a cache line holds 64 B = 16 FP32
        elements.  When XBLOCK < 16 each tile row spans less than one cache
        line, so fetched bytes go partly to waste until a neighbouring block
        reuses the line from L2:
            coalescing = min(1.0, XBLOCK / 16)
            score     *= (0.65 + 0.35 * coalescing)
        giving XBLOCK=4 → ×0.74, XBLOCK=8 → ×0.82, XBLOCK≥16 → ×1.00.
        """
        block_dims        = PointwiseHeuristics.get_block_dimensions(config)
        threads_per_block = PointwiseHeuristics.prod(block_dims)

        arch            = PointwiseHeuristics._get_arch()
        optimal_threads = arch.optimal_threads_bandwidth
        sigma           = optimal_threads  # Gaussian width = optimal

        if threads_per_block < 64:
            score = 0.60
        else:
            diff     = (threads_per_block - optimal_threads) / sigma
            gaussian = math.exp(-0.5 * diff * diff)
            score    = max(0.60, min(1.0, 0.75 + 0.25 * gaussian))

        # 2-D coalescing penalty.
        xblock = config.get('XBLOCK', 256)
        yblock = config.get('YBLOCK', 0)
        if yblock > 0:
            cache_line_elems = 16   # 64 B / 4 B per FP32
            coalescing = min(1.0, xblock / cache_line_elems)
            score     *= (0.65 + 0.35 * coalescing)

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

        A secondary log-space penalty applies when the grid exceeds 4× the CU
        count (the AMD Command Processor must batch-dispatch the remainder,
        adding measurable scheduling latency):
            penalty = 1.0 - 0.01 * log2(num_blocks / (4 * num_CUs))
            clamped to ≥ 0.88
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

        # Large-grid scheduler penalty.
        try:
            import torch as _torch
            num_cus = (_torch.cuda.get_device_properties(0).multi_processor_count
                       if _torch.cuda.is_available() else 120)
        except Exception:
            num_cus = 120

        max_good_blocks = num_cus * 4
        if num_blocks > max_good_blocks:
            excess         = num_blocks / max_good_blocks
            grid_penalty   = 1.0 - 0.01 * math.log2(excess)
            score         *= max(0.88, grid_penalty)

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

        Three regimes selected by grid saturation level and block count:

        Well-saturated (saturation ≥ 0.25) and single-block (all work on one CU):
            Classic sweet-spot model — num_warps in [sweet_min, sweet_max] (4–8
            on AMD) scores 1.0.  Derived from Little's Law: need ceil(300/40) = 8
            wavefronts to hide ~300-cycle HBM latency with a 40-cycle issue gap.
            4 suffices when L2 absorbs enough traffic to cut effective latency.

        Launch-bound multi-block (saturation < 0.25, num_blocks > 1):
            Each block lands on a separate CU, so latency hiding is already
            provided across CUs by the grid.  Within a single block, extra
            wavefronts add only SPI initialisation cost with no benefit:

                T(nw) = K_launch + (nw − 1) × K_warp
                score  = K_launch / T(nw)  ∈ (0, 1]

            where K_launch = 3.0 µs and K_warp = 0.2 µs.  This gives
            score(nw=1) = 1.00, score(nw=4) = 0.83, score(nw=16) = 0.50.
            No tuning parameters — both constants are from the overhead model.

        Returns a value in [0.70, 1.00].
        """
        num_warps      = config.get('num_warps', 4)
        total_elements = problem_metadata.get('total_elements', 1)

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
        max_wf    = num_cus * 8   # AMD: ~8 resident wavefronts per CU
        saturation = min(1.0, (num_blocks_est * num_warps) / max_wf)

        def _sweet_spot_score(nw: int) -> float:
            if sweet_min <= nw <= sweet_max:
                return 1.00
            elif sweet_min // 2 <= nw <= int(sweet_max * 1.5):
                return 0.95
            elif nw == 1:
                return 0.85   # Single wavefront stalls on every memory access
            return 0.75

        if saturation >= 0.25 or num_blocks_est == 1:
            # Well-saturated or single-block: latency hiding governs.
            base_score = _sweet_spot_score(num_warps)
        else:
            # Launch-bound multi-block: extra warps add only init overhead.
            try:
                from torch._inductor.codegen.triton_heuristics_adaptive import (
                    BottleneckAnalysis as _BA,
                )
                k_launch = _BA.KERNEL_LAUNCH_US
            except Exception:
                k_launch = 3.0

            k_warp     = 0.2   # µs per extra wavefront (SPI allocation)
            t_config   = k_launch + (num_warps - 1) * k_warp
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
        return [cfg for _, cfg in scored[:top_n]]

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

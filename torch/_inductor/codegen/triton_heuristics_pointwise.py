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
            cache_line_elems = 16   # 64 B / 4 B per FP32
            zblock = config.get('ZBLOCK', 0)

            if zblock > 0:
                # ── 3-D kernel: shape-aware coalescing ────────────────────────
                #
                # The correct model depends on the problem aspect ratio:
                #
                # CUBE-LIKE problems (max_dim / min_dim < 2, e.g. 64×64×64):
                #   Both X and Y loop axes are nearly the same length and both
                #   can be stride-1 in permute/broadcast kernels.  Using wider
                #   YBLOCK tiles genuinely improves spatial reuse, so we average
                #   the X and Y coalescing factors.  This correctly prefers
                #   {X16,Y16} over {X16,Y8} for symmetric cube workloads.
                #
                # ELONGATED problems (max_dim / min_dim ≥ 2, e.g. 256×64×32):
                #   The YBLOCK axis is often mismatched — stride-1 for one tensor
                #   but stride-N for the other (typical in permute kernels).  A
                #   YBLOCK penalty empirically hurts these cases: {X32,Y8}
                #   outperforms {X16,Y16} by 3–5 % on MI300X.  We therefore
                #   apply XBLOCK-only coalescing and leave YBLOCK unpenalised.
                dims_3d = problem_metadata.get('dimensions', ())
                is_cube_3d = False
                if len(dims_3d) == 3:
                    _max_d = max(dims_3d)
                    _min_d = max(min(dims_3d), 1)
                    is_cube_3d = (_max_d / _min_d) < 2

                if is_cube_3d:
                    # Symmetric (X+Y averaged) coalescing for cube problems.
                    coalescing_x = min(1.0, xblock / cache_line_elems)
                    coalescing_y = min(1.0, yblock / cache_line_elems)
                    avg_coalescing = (coalescing_x + coalescing_y) / 2.0
                    warp_align = 0.75 + 0.25 * avg_coalescing   # 0.75 → 1.00
                else:
                    # XBLOCK-only coalescing for elongated/mixed-stride problems.
                    if xblock >= cache_line_elems:
                        warp_align = 1.00
                    else:
                        coalescing = xblock / cache_line_elems
                        warp_align = 0.75 + 0.23 * coalescing   # 0.75 → 0.98
            else:
                # ── 2-D kernel: XBLOCK-only coalescing rule (unchanged) ──────
                #
                # For 2-D transpose/strided kernels the fast thread index always
                # maps to X, so only XBLOCK determines coalescing.
                if xblock >= cache_line_elems:
                    warp_align = 1.00
                else:
                    coalescing = xblock / cache_line_elems
                    warp_align = 0.75 + 0.23 * coalescing  # 0.75→0.98

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
            # Single-block kernel: EPB IS the overhead-amortisation metric.
            # The ~3 µs dispatch cost is not shared across multiple blocks, so
            # larger EPB directly reduces the overhead fraction.
            sigma = optimal_epb // 2
            if elements_per_block < 64:
                score = 0.70
            else:
                diff     = (elements_per_block - optimal_epb) / sigma
                gaussian = math.exp(-0.5 * diff * diff)
                score    = max(0.70, min(1.0, 0.75 + 0.25 * gaussian))
        else:
            # Multi-block kernel: the kernel dispatch cost is shared across ALL
            # blocks and is already amortised over the entire problem.  EPB is
            # therefore NOT a meaningful proxy for overhead amortisation here —
            # that is already captured by the bottleneck analysis (overhead_frac)
            # and the Grid score (CU saturation).  Using a narrow Gaussian
            # centred at optimal_epb=2048 incorrectly rewarded XBLOCK=1024 (64
            # blocks) over XBLOCK=256 (256 blocks) even though the latter keeps
            # 4× more CUs busy and is empirically 15–20% faster on MI300X.
            #
            # For multi-block configs we therefore use a very wide sigma (3×
            # optimal) so only extreme EPB values (<32 elements/block) are
            # penalised.  The Grid score handles block-count optimisation.
            sigma = optimal_epb * 3
            if elements_per_block < 32:
                score = 0.88   # sub-cache-line blocks — always bad
            else:
                diff     = (elements_per_block - optimal_epb) / sigma
                gaussian = math.exp(-0.5 * diff * diff)
                score    = max(0.88, min(1.0, 0.88 + 0.12 * gaussian))

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

        # 3-D kernels require a shape-specific grid target:
        #
        # CUBE-LIKE problems (all dims within 2× of each other, e.g. 64×64×64):
        #   Symmetric, regular-stride access → excellent L2 spatial reuse.
        #   Fewer, larger blocks exploit this reuse (each block touches a
        #   contiguous 3-D tile).  The sweet spot is ≈ CUs/2 blocks, not 2×CUs.
        #   Using hardware_optimal//2 (~228 on MI300X) makes the empirically best
        #   256-block configs score Grid≈0.993 rather than 0.920.
        #
        # NON-CUBE problems (max/min dim ratio ≥ 2, e.g. 256×128×32):
        #   Mixed-stride access (e.g. permute kernels) → irregular HBM traffic.
        #   More wavefronts per CU are needed to hide latency; 1024-block configs
        #   outperform 512-block ones by 2–5 % on MI300X.  Double the target.
        ndims = len(problem_metadata.get('dimensions', (1,)))
        if ndims >= 3:
            dims_3d = problem_metadata.get('dimensions', (1, 1, 1))
            max_dim = max(dims_3d) if dims_3d else 1
            min_dim = max(min(dims_3d), 1) if dims_3d else 1
            if (max_dim / min_dim) < 2:
                # Cube-like: smaller block count needed for L2 locality.
                optimal = max(4, hardware_optimal // 2)
            else:
                # Non-cube: more blocks needed for latency hiding.
                optimal = min(optimal * 2, hardware_optimal * 4)

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
                return 0.95
            elif nw == 1:
                # Single wavefront per block.
                #
                # When the CU already has enough wavefronts from block-level
                # switching (need_intra_block_warps=False) the ILP correction
                # applies normally — extra loads per thread hide per-thread
                # latency while the CU hides inter-thread stalls via block
                # switching.
                #
                # When there are too few blocks per CU (need_intra_block_warps=
                # True), ILP within a thread does NOT compensate: the bottleneck
                # is the number of independent wavefronts available to the CU
                # scheduler, not per-thread instruction parallelism.  Empirically,
                # num_warps=1 with 6–7 blocks/CU is 1.4–2.8× slower than
                # num_warps=8 on MI300X even when elems_per_thread=16 (full ILP).
                if not need_intra_block_warps:
                    if elems_per_thread >= 8:
                        return 1.00   # CU-level hiding + ILP: full benefit
                    elif elems_per_thread >= 4:
                        return 0.93   # Partial ILP benefit
                return 0.82   # Not enough wavefronts per CU; ILP doesn't compensate
            # nw > int(sweet_max * 1.5): very high warp count — e.g. num_warps=16
            # with XBLOCK=1024 means 16 wavefronts per block, providing ample CU-
            # level latency hiding even for memory-bound kernels.  Return 0.88 for
            # all high-warp configs regardless of ILP so they score above the
            # penalised num_warps=1 path (0.82) — empirically num_warps=16 is
            # 1.2–1.4× faster than num_warps=1 for memory-bound kernels with few
            # blocks per CU, because 16 wavefronts / block >> the 8 needed to hide
            # 300-cycle HBM round-trip latency.
            return 0.88

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

            # ILP correction — software-pipelining many element loads per thread
            # hides HBM latency without extra wavefronts.  HOWEVER, this only
            # helps at the per-block level.  When CU utilisation is very low
            # (num_blocks << num_CUs) the system-level bottleneck is CU
            # idleness, not per-block memory latency.  Applying the full ILP
            # bonus to e.g. XBLOCK=1024 with only 64 active CUs out of 304
            # incorrectly scores it 1.0 while the config leaves 79% of the GPU
            # idle — empirically 15–20% slower than XBLOCK=256 (256 CUs active).
            #
            # We therefore scale the ILP ceiling by CU utilisation:
            #   ilp_scale = min(1.0, cu_util / 0.25)
            #   → full ILP benefit at ≥ 25 % CU utilisation
            #   → zero ILP boost at 0 % CU utilisation
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

            # 3-D tie-breaker: for cube-like problem shapes, mildly prefer
            # cube-like tile shapes and a minimum YBLOCK.
            #
            # For x.permute(0,2,1)+y style kernels the YBLOCK axis is the
            # contiguous direction for the permuted tensor.  Very elongated tiles
            # like {XBLOCK:64, YBLOCK:4} coalesce poorly for that tensor even if
            # the total element count is the same as a balanced tile.
            #
            # The tie-breaker is problem-shape aware: for elongated problems
            # (max/min dimension ratio > 4) the optimal tile shape may itself be
            # elongated (matching the problem), so we skip the penalty there.
            #
            #   balance_mult = 1 − 0.003 × log2(max_dim / min_dim)
            #     e.g. {16,16,8} ratio=2 → ×0.997   {64,4,4} ratio=16 → ×0.988
            #   yblock_mult  = 1 − 0.008 × max(0, 4 − log2(yblock))
            #     yblock=16 → ×1.000   yblock=8 → ×0.992   yblock=4 → ×0.984
            elif len(block_dims) == 3:
                xblock, yblock, zblock = block_dims
                problem_dims = PointwiseHeuristics.get_problem_dimensions(problem_metadata)

                # Determine if the problem is strictly cube-like (all dims within 2× of each other).
                # A ratio threshold of 2 (not 4) ensures we only apply cube-preferring
                # tie-breakers to genuinely symmetric workloads (e.g. 64×64×64, 96×96×96).
                # Moderately elongated shapes like 128×64×64 (ratio=2) are NOT cube-like
                # and get XBLOCK-only coalescing + grid doubling instead.
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

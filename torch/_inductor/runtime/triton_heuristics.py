# mypy: allow-untyped-defs
from __future__ import annotations

import builtins
import copy
import dataclasses
import functools
import hashlib
import inspect
import itertools
import logging
import math
import operator
import os
import os.path
import re
import sys
import threading
import time
from collections import namedtuple
from typing import Any, Generic, Literal, TYPE_CHECKING, TypeVar, Union

import torch
from torch._dynamo.utils import counters, set_feature_use
from torch._inductor import metrics
from torch._prims_common import compute_required_storage_length
from torch.utils._debug_mode import get_active_debug_mode
from torch.utils._ordered_set import OrderedSet

from ..triton_bundler import TritonBundler
from ..utils import prefix_is_reduction, triton_version_uses_attrs_dict
from . import triton_helpers
from .autotune_cache import AutotuneCache
from .benchmarking import benchmarker
from .coordinate_descent_tuner import CoordescTuner

# Import pointwise heuristics for intelligent config selection
# Can be disabled via environment variable for testing
def _is_pointwise_heuristics_enabled():
    """Check if pointwise heuristics are enabled at runtime (not import time)"""
    return os.environ.get("TORCHINDUCTOR_POINTWISE_HEURISTICS", "1") == "1"


# Global storage for heuristics validation data
# Maps problem_key -> {problem_metadata, predicted_scores: [(score, config)], actual_timings: [(time, config)], top_n_configs: [configs]}
_HEURISTICS_VALIDATION_DATA = {}

# Global storage for top-N predicted configs per problem
# Maps problem_key -> [list of top N configs to use for selection]
_TOP_N_CONFIGS_FOR_SELECTION = {}


def _normalize_problem_key(size_hints):
    """
    Normalize size_hints to a consistent problem key string.
    size_hints can be:
    - tuple: (32768,) or (32, 1024)
    - dict: {'x': 32768} or {'x': 32, 'y': 1024}
    """
    if isinstance(size_hints, dict):
        # Sort by key to ensure consistent ordering
        values = tuple(size_hints[k] for k in sorted(size_hints.keys()))
        return str(values)
    elif isinstance(size_hints, (tuple, list)):
        return str(tuple(size_hints))
    else:
        return str((size_hints,))


def _store_heuristics_predictions(problem_key, problem_metadata, predicted_scores, kernel_code=None):
    """Store predicted scores for later comparison with actual performance
    
    V5: Now stores kernel_code for improved re-scoring
    """
    if problem_key not in _HEURISTICS_VALIDATION_DATA:
        _HEURISTICS_VALIDATION_DATA[problem_key] = {
            'problem_metadata': problem_metadata,
            'predicted_scores': predicted_scores,
            'actual_timings': [],
            'kernel_code': kernel_code,  # V5: Store kernel code
        }


def _store_actual_timing(problem_key, config, timing_ms):
    """Store actual benchmark timing for a config"""
    if problem_key in _HEURISTICS_VALIDATION_DATA:
        _HEURISTICS_VALIDATION_DATA[problem_key]['actual_timings'].append((timing_ms, config))


def _bottleneck_bar(ohd_frac, mem_frac, cmp_frac, width=30):
    """Render a fixed-width ASCII bar showing time fractions: [OOO|MMMMMMMMMMMM|CCC]."""
    o = max(1, round(ohd_frac * width))
    m = max(1, round(mem_frac * width))
    c = max(1, round(cmp_frac * width))
    # Clamp total to width
    total = o + m + c
    if total != width:
        m += width - total
        m = max(1, m)
    return f"[{'O'*o}|{'M'*m}|{'C'*c}]"


def _format_bottleneck_block(cfg, prob_meta, kernel_code, indent="  "):
    """
    Return a list of formatted lines describing the bottleneck analysis for cfg.

    Shows estimated overhead / memory / compute times with a roofline bar and
    bottleneck label.  Returns [] if analysis is unavailable.

    Roofline semantics:
      total_us = overhead_us + max(memory_us, compute_us)
      The slower of memory/compute is the bottleneck; the faster is "hidden".
      Fractions are from the raw gross (overhead+memory+compute) so they convey
      relative weight of each component even when two overlap.
    """
    if not POINTWISE_HEURISTICS_AVAILABLE:
        return []
    try:
        bn = BottleneckAnalysis.analyze_bottleneck(cfg, prob_meta, kernel_code)
    except Exception:
        return []

    ohd_us     = bn.get('overhead_us',  0.0)
    mem_us     = bn.get('memory_us',    0.0)
    cmp_us     = bn.get('compute_us',   0.0)
    tot_us     = bn.get('total_us',     0.0)   # overhead + max(mem, cmp)
    ohd_pct    = bn.get('overhead_frac', 0.0) * 100
    mem_pct    = bn.get('memory_frac',   0.0) * 100
    cmp_pct    = bn.get('compute_frac',  0.0) * 100
    bottleneck = bn.get('bottleneck', '?')

    _BN_ICON = {'overhead': '🚀 LAUNCH', 'memory': '💾 MEMORY', 'compute': '⚡ COMPUTE'}
    icon = _BN_ICON.get(bottleneck, '?')

    bar = _bottleneck_bar(ohd_pct / 100, mem_pct / 100, cmp_pct / 100)

    # Annotate which of memory/compute is hidden by the other
    hidden_note = ""
    if bottleneck == 'memory':
        hidden_note = f"  (compute hidden by memory, effective total = {tot_us:.2f}µs)"
    elif bottleneck == 'compute':
        hidden_note = f"  (memory hidden by compute, effective total = {tot_us:.2f}µs)"
    else:
        hidden_note = f"  (effective total = {tot_us:.2f}µs)"

    lines = [
        f"{indent}Bottleneck : {icon}   {bar}  (O=launch M=memory C=compute){hidden_note}",
        f"{indent}             {'Overhead':>9}  {'Memory':>9}  {'Compute':>9}  {'Effective':>10}",
        f"{indent}  Time (µs): {ohd_us:9.2f}  {mem_us:9.2f}  {cmp_us:9.2f}  {tot_us:10.2f}",
        f"{indent}  Share (%): {ohd_pct:9.1f}  {mem_pct:9.1f}  {cmp_pct:9.1f}  {'(roofline)':>10}",
    ]
    return lines


def _print_heuristics_validation_summary(problem_key):
    """
    Print comprehensive validation summary comparing predicted vs actual performance.

    Called after autotune_to_one_config completes in REAL_BENCH mode.  The
    summary shows:
      • Problem dimensions and kernel metadata
      • 📊 PREDICTED #1 – the config the heuristic ranked top, with:
          - factor scores (BW / Launch / Grid / Occupancy)
          - bottleneck analysis (overhead / memory / compute µs + %)
          - actual measured time from benchmarking
      • 🏆 ACTUAL Best – the truly fastest benchmarked config, same detail
      • 🔍 ANALYSIS – rank error, real speedup, per-factor diff table
      • 💡 Root Cause – which factor drove the misprediction
      • 📈 ACCURACY – Excellent / Good / Acceptable / Poor verdict

    Gated by ``inductor_config.heuristics_verbose``.  Set
    ``TORCHINDUCTOR_HEURISTICS_VERBOSE=0`` to suppress.
    """
    from torch._inductor import config as inductor_config
    if not inductor_config.heuristics_verbose:
        return

    if problem_key not in _HEURISTICS_VALIDATION_DATA:
        return

    data     = _HEURISTICS_VALIDATION_DATA[problem_key]
    predicted = data['predicted_scores']
    actual    = data['actual_timings']

    if not actual:
        return  # No timing data collected yet

    kernel_code  = data.get('kernel_code', None)
    prob_meta    = data['problem_metadata']

    # ── sort actual timings (lowest = fastest) ────────────────────────────
    actual_sorted   = sorted(actual, key=lambda x: x[0])
    best_actual_ms, best_actual_cfg = actual_sorted[0]

    # ── sort predicted scores (highest = best heuristic pick) ────────────
    predicted_sorted = sorted(predicted, reverse=True, key=lambda x: x[0])
    if not predicted_sorted:
        return

    best_predicted_score, best_predicted_cfg = predicted_sorted[0]

    # find measured time for what we predicted was #1
    predicted_actual_ms = None
    for t, cfg in actual:
        if cfg == best_predicted_cfg:
            predicted_actual_ms = t
            break
    if predicted_actual_ms is None:
        return  # our top pick wasn't benchmarked (shouldn't happen in REAL_BENCH)

    slowdown = predicted_actual_ms / best_actual_ms if best_actual_ms > 0 else 1.0

    # find where the actual best landed in heuristic ranking
    actual_best_rank  = None
    actual_best_score = None
    for rank, (score, cfg) in enumerate(predicted_sorted, 1):
        if cfg == best_actual_cfg:
            actual_best_rank  = rank
            actual_best_score = score
            break

    # ── detailed factor scores ─────────────────────────────────────────────
    if POINTWISE_HEURISTICS_AVAILABLE:
        try:
            pred_details = PointwiseHeuristics.get_detailed_scores(
                best_predicted_cfg, prob_meta, kernel_code
            )
        except Exception:
            pred_details = None
        try:
            actual_details = (
                PointwiseHeuristics.get_detailed_scores(
                    best_actual_cfg, prob_meta, kernel_code
                ) if actual_best_rank is not None else None
            )
        except Exception:
            actual_details = None
    else:
        pred_details = actual_details = None

    top_n_cfgs = _TOP_N_CONFIGS_FOR_SELECTION.get(problem_key, [])

    W = 90
    print("\n" + "═" * W, flush=True)
    print("  [HEURISTICS VALIDATION]  Predicted vs Actual Performance", flush=True)
    print("═" * W, flush=True)

    # ── problem header ─────────────────────────────────────────────────────
    dims_str  = prob_meta.get('dimensions', 'N/A')
    total_el  = prob_meta.get('total_elements', 0)
    el_size   = prob_meta.get('element_size', 4)
    src_note  = f"kernel src: {len(kernel_code)} chars" if kernel_code else "no kernel src"
    print(f"  Problem    : {dims_str}  |  {total_el:,} elements  |  {el_size}B dtype  |  {src_note}", flush=True)
    print(f"  Configs    : {len(predicted)} scored  |  {len(actual)} benchmarked", flush=True)
    if top_n_cfgs:
        print(
            f"  Selection  : winner chosen from heuristic top-{len(top_n_cfgs)}  "
            f"| validation uses all {len(actual)} benchmarked configs",
            flush=True,
        )
    print("─" * W, flush=True)

    # ── helper: print one config block ────────────────────────────────────
    def _print_config_block(label, score, rank_str, cfg, measured_ms, details, indent="  "):
        print(f"\n{indent}{label}", flush=True)
        print(f"{indent}  Config     : {cfg}", flush=True)
        print(f"{indent}  Score      : {score:.4f}  (rank {rank_str})", flush=True)
        print(f"{indent}  Measured   : {measured_ms:.6f} ms", flush=True)
        if details:
            bw   = details.get('memory_bandwidth', 0)
            lnch = details.get('launch_overhead',  0)
            grid = details.get('grid_granularity', 0)
            occ  = details.get('occupancy',        0)
            nblk = details.get('num_blocks',       0)
            tpb  = details.get('threads_per_block',0)
            print(
                f"{indent}  Factors    : "
                f"BW={bw:.3f}(40%)  Launch={lnch:.3f}(30%)  "
                f"Grid={grid:.3f}(20%)  Occup={occ:.3f}(10%)",
                flush=True,
            )
            print(f"{indent}  Grid       : {nblk:,} blocks  |  {tpb} threads/block", flush=True)
        # Bottleneck block
        for line in _format_bottleneck_block(cfg, prob_meta, kernel_code, indent + "  "):
            print(line, flush=True)

    # ── predicted best ────────────────────────────────────────────────────
    _print_config_block(
        "📊 PREDICTED #1  (heuristic top pick)",
        best_predicted_score, "#1",
        best_predicted_cfg, predicted_actual_ms, pred_details,
    )

    # ── actual best ───────────────────────────────────────────────────────
    rank_note = f"#{actual_best_rank}" if actual_best_rank else "NOT in scored set"
    _print_config_block(
        "🏆 ACTUAL Best  (fastest benchmarked)",
        actual_best_score if actual_best_score is not None else 0.0,
        rank_note,
        best_actual_cfg, best_actual_ms, actual_details,
    )

    # ── analysis ──────────────────────────────────────────────────────────
    print(f"\n  🔍 ANALYSIS", flush=True)
    print("  " + "─" * (W - 2), flush=True)

    if actual_best_rank is None:
        print("  ❌ CRITICAL: actual best config was NOT generated by the heuristic!", flush=True)
        print("     This indicates a gap in the config generation logic.", flush=True)
    elif actual_best_rank == 1:
        print(f"  ✅ Heuristics correctly identified the best config!", flush=True)
        print(f"     Predicted #1 = Actual #1  (slowdown: {slowdown:.3f}x)", flush=True)
    else:
        rank_diff    = actual_best_rank - 1
        speedup_real = predicted_actual_ms / best_actual_ms
        score_gap    = best_predicted_score - actual_best_score
        print(
            f"  ⚠️  Heuristics ranked actual best as #{actual_best_rank} "
            f"(off by {rank_diff} positions)",
            flush=True,
        )
        print(f"     Score gap  : {score_gap:+.4f}  "
              f"({abs(score_gap)/best_predicted_score*100:.1f}% of predicted score)", flush=True)
        print(f"     Real speed : {speedup_real:.3f}x  (actual best vs heuristic pick)", flush=True)

        # factor comparison table
        if pred_details and actual_details:
            factors = [
                ('memory_bandwidth', 'Bandwidth', 40),
                ('launch_overhead',  'Launch',    30),
                ('grid_granularity', 'Grid',      20),
                ('occupancy',        'Occupancy', 10),
            ]
            print(f"\n  📋 Factor comparison  (Predicted #1  vs  Actual Best #{actual_best_rank}):", flush=True)
            print(f"     {'Factor':10s}  {'Pred':>7}  {'Actual':>7}  {'Delta':>7}  {'Wt':>4}  Note", flush=True)
            print(f"     {'-'*60}", flush=True)
            weighted_diffs = []
            for fk, fname, wt in factors:
                pv  = pred_details.get(fk, 0)
                av  = actual_details.get(fk, 0)
                d   = pv - av
                wd  = abs(d) * (wt / 100.0)
                icon = "🔴" if abs(d) > 0.01 and d > 0 else ("🟢" if abs(d) > 0.01 else "  ")
                note = f"predicted scored {'HIGHER' if d>0 else 'LOWER'}" if abs(d) > 0.01 else "≈ same"
                print(f"     {icon}{fname:10s}  {pv:7.3f}  {av:7.3f}  {d:+7.3f}  {wt:3d}%  {note}", flush=True)
                weighted_diffs.append((wd, fname, d, wt))
            print(f"     {'-'*60}", flush=True)

            # root-cause
            weighted_diffs.sort(reverse=True)
            biggest_wd, biggest_fname, biggest_d, biggest_wt = weighted_diffs[0]
            print(f"\n  💡 Root Cause:", flush=True)
            if biggest_wd > 0.01:
                direction = "over-scored" if biggest_d > 0 else "under-scored"
                print(
                    f"     {biggest_fname} was {direction} for predicted config "
                    f"(Δ={biggest_d:+.3f}, wt={biggest_wt}%)",
                    flush=True,
                )
                print(
                    f"     Actual best ran {speedup_real:.3f}x faster despite "
                    f"{'lower' if biggest_d>0 else 'higher'} {biggest_fname}",
                    flush=True,
                )
                adj = "too high" if biggest_d > 0 else "too low"
                print(f"     → {biggest_fname} weight ({biggest_wt}%) may be {adj}", flush=True)
            else:
                print("     All factors very similar – ranking gap is likely scoring noise.", flush=True)
                print("     Consider: more discriminating factors or tighter weight calibration.", flush=True)

    # ── benchmark ranking table (top-10 actual) ───────────────────────────
    print(f"\n  📊 Benchmark Results (top 10 of {len(actual_sorted)} configs):", flush=True)
    print(f"     {'Rank':>4}  {'Time (ms)':>10}  {'Pred rank':>9}  {'Score':>7}  Config", flush=True)
    print(f"     {'-'*70}", flush=True)
    for i, (t, cfg) in enumerate(actual_sorted[:10], 1):
        pred_rank_str = "—"
        pred_score_str = "—"
        for pr, (ps, pc) in enumerate(predicted_sorted, 1):
            if pc == cfg:
                pred_rank_str  = f"#{pr}"
                pred_score_str = f"{ps:.4f}"
                break
        slowdown_str = f"  (×{t/best_actual_ms:.3f})" if i > 1 else "  ← best"
        top5_mark    = " ★" if cfg in top_n_cfgs else ""
        print(
            f"     #{i:3d}  {t:10.6f}  {pred_rank_str:>9}  {pred_score_str:>7}  {cfg}{slowdown_str}{top5_mark}",
            flush=True,
        )
    print(f"     {'-'*70}", flush=True)
    if top_n_cfgs:
        print(f"     ★ = was in heuristic top-{len(top_n_cfgs)} selection pool", flush=True)

    # ── overall verdict ───────────────────────────────────────────────────
    if slowdown < 1.05:
        verdict = "✅ EXCELLENT  – within 5% of optimal"
    elif slowdown < 1.15:
        verdict = "✓  GOOD       – within 15% of optimal"
    elif slowdown < 1.30:
        verdict = "⚠  ACCEPTABLE – within 30% of optimal"
    else:
        verdict = "❌ POOR       – >30% slower than optimal"

    print(f"\n  📈 ACCURACY : heuristic pick is {slowdown:.3f}x vs actual best", flush=True)
    print(f"     {verdict}", flush=True)
    print("═" * W + "\n", flush=True)

try:
    from torch._inductor.codegen.triton_heuristics_pointwise import PointwiseHeuristics
    from torch._inductor.codegen.triton_heuristics_adaptive import BottleneckAnalysis
    POINTWISE_HEURISTICS_AVAILABLE = True
except ImportError:
    POINTWISE_HEURISTICS_AVAILABLE = False
    logging.warning("Pointwise heuristics not available, falling back to default configs")
from .hints import (
    _NUM_THREADS_PER_WARP,
    AutotuneHint,
    DeviceProperties,
    HeuristicType,
    ReductionHint,
    TileHint,
    TRITON_MAX_BLOCK,
    TRITON_MAX_RSPLIT,
)
from .runtime_utils import (
    ceildiv,
    conditional_product,
    create_bandwidth_info_str,
    dynamo_timed,
    get_first_attr,
    get_max_y_grid,
    get_num_bytes,
    next_power_of_2,
    triton_cache_dir,
    triton_config_to_hashable,
    triton_hash_to_path_key,
    validate_triton_config,
)
from .static_cuda_launcher import StaticallyLaunchedCudaKernel
from .triton_compat import (
    ASTSource,
    autograd_profiler,
    cc_warp_size,
    CompiledKernel,
    Config,
    GPUTarget,
    HAS_WARP_SPEC,
    KernelInterface,
    knobs,
    OutOfResources,
    PTXASError,
    triton,
)
from .triton_helpers import get_constexprs


class InductorConfig(Config):
    """Inductor-specific Triton config with additional control flags"""

    def __init__(self, *args, dynamic_scale_rblock=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_scale_rblock = dynamic_scale_rblock


class NoTritonConfigsError(RuntimeError):
    pass


if TYPE_CHECKING:
    from collections.abc import Callable, Container, Hashable

    from torch._guards import CompileId

    LauncherType = Any

_KernelType = Union[CompiledKernel, StaticallyLaunchedCudaKernel]
_T = TypeVar("_T", bound=_KernelType)

log = logging.getLogger(__name__)

triton_name_sub = re.compile(r"^def [^(]+\(")


def generate_lookup_hash_from_source_code(size_hints_str: str, source_code: str) -> str:
    # Name agnostic + strip white space
    fn_strip_name = re.sub(triton_name_sub, "(", source_code.strip(), count=1)
    hash_str = size_hints_str + fn_strip_name
    fn_hash = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()

    return fn_hash


def lookup_autotune_config(size_hints, fn) -> Config | None:
    lookup_table = torch._inductor.config.autotune_lookup_table
    cached_config = None
    if len(lookup_table) > 0 and "_fused_" in fn.src:
        fn_hash = generate_lookup_hash_from_source_code(str(size_hints), fn.src)
        if fn_hash in lookup_table:
            config_dict = lookup_table[fn_hash]
            block_configs = {k: v for k, v in config_dict.items() if "BLOCK" in k}
            cached_config = Config(
                block_configs,
                num_warps=config_dict["num_warps"],
                num_stages=config_dict["num_stages"],
            )

    return cached_config


def get_total_reduction_numel(numels: dict[str, int]) -> int:
    return conditional_product(
        *[numel for prefix, numel in numels.items() if prefix_is_reduction(prefix)]
    )


def autotune_hints_to_configs(
    hints: OrderedSet[AutotuneHint],
    size_hints,
    block_size: int,
    device_props: DeviceProperties,
) -> list[Config]:
    """
    AutotuneHints can be attached to the metadata of triton kernels for providing
    suggestions about what to try for autotuning. One reason to do this is if there are
    some configs that are only useful in specific scenarios, in which case we can avoid
    wasting compile time on autotuning unless we know we are in one of those scenarios.

    Based on those hints, this function will generate a list of additional autotuning
    configs to try.
    """
    xyz_options: tuple[tuple[int, int | None, int | None], ...]
    configs: list[Config] = []
    for hint in hints:
        if hint == AutotuneHint.ONE_ELEMENT_PER_THREAD:
            if len(size_hints) == 1:
                xyz_options = ((block_size // 4, None, None),)
            elif len(size_hints) == 2:
                xyz_options = ((block_size // 4, 1, None), (1, block_size // 4, None))
            elif len(size_hints) == 3:
                xyz_options = (
                    (block_size // 4, 1, 1),
                    (1, block_size // 4, 1),
                    (1, 1, block_size // 4),
                )
            configs.extend(
                triton_config(
                    size_hints,
                    *xyz,
                    num_elements_per_warp=(
                        device_props.warp_size if device_props.warp_size else 32
                    ),
                )
                for xyz in xyz_options
            )

    return configs


def _dump_launch_params(args, kwargs, launcher, kernel_name, grid):
    call_args = []
    call_kwargs = {}
    for arg in args:
        if isinstance(arg, (int, bool)):
            call_args.append(str(arg))
        else:
            call_args.append("T")
    for k, v in kwargs.items():
        if isinstance(arg, (int, bool)):
            call_kwargs[k] = v
        else:
            call_kwargs[k] = v
    call_kwargs.update(launcher.config.kwargs)
    call_kwargs["num_warps"] = launcher.config.num_warps
    call_kwargs["num_stages"] = launcher.config.num_stages
    if HAS_WARP_SPEC:
        call_kwargs["num_consumer_groups"] = getattr(
            launcher.config, "num_consumer_groups", 0
        )
        call_kwargs["num_buffers_warp_spec"] = getattr(
            launcher.config, "num_buffers_warp_spec", 0
        )
    args_str = [*call_args]
    args_str.extend(f"{k}={v}" for k, v in call_kwargs.items())
    args_str = ", ".join(args_str)
    abs_path = os.path.abspath(sys.argv[0])
    with open(f"{abs_path}.launch_params", "a") as f:
        f.write(f"{kernel_name} | {args_str} | {grid!r}\n")


def check_autotune_cache(
    configs: list[Config], filename: str | None, inductor_meta: dict[str, Any]
) -> tuple[list[Config], AutotuneCache | None, dict[str, Any]]:
    """
    Given a list of configs, checks autotune cache and return metadata
    """
    autotune_cache = None
    autotune_cache_info = {}
    disabled = inductor_meta.get("force_disable_caches", False)
    # In heuristics REAL_BENCH mode we want to benchmark ALL candidate configs so
    # that the predicted-vs-actual validation summary is meaningful.  Skip the
    # cache READ (which would collapse configs to [best_config]) but still create
    # the AutotuneCache object so the winner is written back at the end of the run.
    skip_cache_read = inductor_meta.get("_heuristics_skip_cache_read", False)
    if (
        not disabled
        and filename is not None
        and (len(configs) > 1 or inductor_meta.get("coordinate_descent_tuning"))
        and os.environ.get("TRITON_INTERPRET", "0") != "1"
    ):
        configs_hash = hash_configs(configs)

        autotune_cache = AutotuneCache.create(inductor_meta, filename, configs_hash)
        if autotune_cache:
            if not skip_cache_read and (best_config := autotune_cache.read_best(inductor_meta, configs)):
                configs = [best_config]
                autotune_cache_info["best_config"] = triton_config_to_hashable(
                    best_config
                )
                autotune_cache_info["autotune_cache_state"] = "hit"

            else:
                autotune_cache_info["autotune_cache_state"] = "miss"
                autotune_cache_info["num_configs"] = len(configs)
                if inductor_meta.get("coordinate_descent_tuning"):
                    autotune_cache_info["coordesc_tuning"] = True
                    if len(configs) == 1:
                        # This is the config that coordinate descent tuning started at, which
                        # is not the same as the final config chosen (i.e. only_config, best_config)
                        autotune_cache_info["coordesc_tuning_start_config"] = (
                            triton_config_to_hashable(configs[0])
                        )
    else:
        if len(configs) == 1:
            autotune_cache_info["autotune_cache_state"] = "only 1 config"
            autotune_cache_info["only_config"] = triton_config_to_hashable(configs[0])

        if disabled:
            autotune_cache_info["autotune_cache_state"] = "force_disabled"
            log.debug("autotune caching is disabled by config.force_disable_caches")

    return configs, autotune_cache, autotune_cache_info


class CachingAutotuner(KernelInterface):
    """
    Simplified version of Triton autotuner that has no invalidation
    key and caches the best config to disk to improve cold start times.
    Unlike the main triton Autotuner, this version can precompile all
    configs, and does not rely on the Triton JIT.
    """

    def __init__(
        self,
        fn,
        triton_meta,  # passed directly to triton
        configs,
        save_cache_hook,
        mutated_arg_names: list[str],  # see [Note: clone mutated buffers]
        optimize_mem,
        heuristic_type,
        size_hints=None,
        inductor_meta=None,  # metadata not relevant to triton
        custom_kernel=False,  # whether the kernel is inductor-generated or custom
        filename: str | None = None,
        reset_to_zero_arg_names: list[str] | None = None,
        autotune_cache_info: dict[str, Any] | None = None,
    ):
        super().__init__()

        assert len(configs) > 0, "Non-empty TritonConfig list required for compiling"
        # makes sure there are no pre-hooks on any of the triton configs
        for cfg in configs:
            validate_triton_config(cfg)

        self.fn = fn
        self.device_props: DeviceProperties = triton_meta["device"]
        self.triton_meta = {
            **triton_meta,
            "device": self.device_props.index,
            "device_type": self.device_props.type,
        }
        self.inductor_meta = {} if inductor_meta is None else inductor_meta
        # Add device properties to inductor_meta for use by coordinate descent tuner
        self.inductor_meta["warp_size"] = self.device_props.warp_size
        self.inductor_meta["max_threads_per_block"] = (
            self.device_props.max_threads_per_block
        )
        self.deterministic_mode = self.inductor_meta.get("deterministic", False)

        self.save_cache_hook = save_cache_hook
        self.mutated_arg_names = mutated_arg_names
        self.reset_to_zero_arg_names = (
            [] if reset_to_zero_arg_names is None else reset_to_zero_arg_names
        )
        self.optimize_mem = optimize_mem
        cached_config = lookup_autotune_config(size_hints, fn)
        self.configs = [cached_config] if cached_config else configs

        # ── Heuristic scoring (pointwise kernels, ROCm only) ──────────────
        # Now that self.fn is set we have access to self.fn.src (the exact
        # Triton kernel source) and self.fn.arg_names (ordered parameter list).
        # These give us accurate kernel-level metadata for scoring:
        #   • self.fn.src   → instruction mix (tl.exp, tl.store, etc.)
        #   • self.fn.arg_names → exact num_inputs / num_outputs via _ptr suffix
        #
        # Scoring happens here rather than in _apply_pointwise_heuristics so
        # that we never need to read the kernel off disk at all.
        #
        # We skip scoring when:
        #   • There was an autotune cache hit (cached_config is set) – a best
        #     config is already known, no need to re-score.
        #   • _heuristics_pending is absent – heuristics were not enabled or
        #     the kernel type is not pointwise.
        if (
            not cached_config
            and '_heuristics_pending' in self.inductor_meta
            and POINTWISE_HEURISTICS_AVAILABLE
        ):
            self._score_and_prune_heuristic_configs()

        self.heuristic_type = heuristic_type
        self.custom_kernel = custom_kernel
        self.cuda_kernel_saved = False
        self.autotune_cache_info = autotune_cache_info
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "CachingAutotuner gets %d configs for %s",
                len(self.configs),
                self.fn.__name__,
            )
            for c in self.configs:
                log.debug(c)

        self.compile_results: list[CompileResult[_KernelType]] = []
        self.launchers: list[LauncherType] = []
        self.lock = threading.Lock()
        if os.getenv("TRITON_CACHE_DIR") is None:
            os.environ["TRITON_CACHE_DIR"] = triton_cache_dir(
                self.triton_meta.get("device", 0)
            )
        log.debug("Triton cache dir: %s", os.environ["TRITON_CACHE_DIR"])

        self.size_hints = size_hints
        self.is_mix_order_reduction = self.inductor_meta.get("RSPLIT_SIZE") is not None
        self.coordesc_tuner = CoordescTuner(
            is_mm=False,
            is_native_matmul=triton_meta.get("native_matmul", False),
            is_mix_order_reduction=self.is_mix_order_reduction,
            name=self.fn.__name__,
            size_hints=size_hints,
            inductor_meta=self.inductor_meta,
        )
        self.filename = filename

        # used for profiling
        self.kernel_hash: str = ""

        # Kernels are stored in the codecache with the filename as a hash of the code.
        # We rely on this to obtain the kernel hash
        if self.filename is not None:
            base_name = os.path.basename(self.filename)
            if ".py" in base_name:
                self.kernel_hash = os.path.splitext(base_name)[0]

        self.precompile_time_taken_ns = 0
        self.autotune_time_taken_ns = 0
        # Dumps the launch configs after autotuning.
        self.dump_launch_params = (
            os.environ.get("TORCHINDUCTOR_DUMP_LAUNCH_PARAMS", "0") == "1"
        )

        self.triton_interpret = os.environ.get("TRITON_INTERPRET", "0") == "1"

        # Compile-time info included in runtime logginging
        self.compile_id: CompileId | None = None
        self.is_backward = False

        # Mode for launch grid calculation
        self.grid_mode: Literal["python", "cpp"] = "python"

    def _score_and_prune_heuristic_configs(self) -> None:
        """
        Score all candidate pointwise configs using ``self.fn.src`` and prune
        ``self.configs`` to the top-5 before any Triton compilation occurs.

        Called from ``__init__`` immediately after ``self.fn`` is set, giving
        access to:
          • ``self.fn.src``       – exact Triton kernel source (instruction mix,
                                    broadcast patterns, op counts)
          • ``self.fn.arg_names`` – ordered parameter list; ``_ptr``-suffixed
                                    names identify tensor args, giving an exact
                                    num_inputs / num_outputs count without regex.

        Key design: configs are scored directly from the Triton ``Config``
        objects already in ``self.configs`` (by extracting their effective
        kwargs).  This avoids the matching problem where ``triton_config()``
        adjusts block sizes based on ``size_hints``, making the raw heuristic
        dicts inconsistent with the actual Triton Config values.

        Behaviour:
          TORCHINDUCTOR_HEURISTICS_REAL_BENCH=1  (default)
            All configs in ``self.configs`` are kept so every one is compiled
            and benchmarked.  The heuristic top-5 is stored in
            ``_TOP_N_CONFIGS_FOR_SELECTION`` so ``autotune_to_one_config``
            selects the winner only from that shortlist.
            Full predicted-vs-actual comparison is printed after benchmarking
            via ``_print_heuristics_validation_summary``.

          TORCHINDUCTOR_POINTWISE_HEURISTICS=1  (heuristics-only, no REAL_BENCH)
            ``self.configs`` is pruned in-place to the top-5 scoring configs.
            Only those 5 are ever compiled or benchmarked.
        """
        pending = self.inductor_meta.pop('_heuristics_pending', None)
        if pending is None:
            return

        problem_metadata = pending['problem_metadata']
        size_hints = pending['size_hints']

        # ── 1. Extract kernel_code from the live JITFunction ──────────────
        kernel_code = None
        try:
            if self.fn is not None and hasattr(self.fn, 'src') and self.fn.src is not None:
                kernel_code = str(self.fn.src)
                log.info(
                    "[HEURISTICS] Extracted kernel source from fn.src (%d chars)",
                    len(kernel_code),
                )
        except Exception as e:
            log.warning(
                "[HEURISTICS] Failed to read fn.src: %s – scoring without kernel metadata", e
            )

        # ── 2. Refine tensor counts from fn.arg_names ─────────────────────
        # arg_names is the authoritative list; _ptr-suffixed = tensor pointer.
        # tl.store calls in the source determine how many are outputs.
        try:
            if self.fn is not None and hasattr(self.fn, 'arg_names'):
                ptr_args = [a for a in self.fn.arg_names if a.endswith('_ptr')]
                if ptr_args:
                    num_outputs = max(1, kernel_code.count('tl.store') if kernel_code else 1)
                    num_inputs = max(0, len(ptr_args) - num_outputs)
                    problem_metadata = {
                        **problem_metadata,
                        'num_tensors': len(ptr_args),
                        'num_inputs': num_inputs,
                        'num_outputs': num_outputs,
                    }
                    log.info(
                        "[HEURISTICS] Tensor counts from arg_names: "
                        "%d ptr args → %d inputs, %d outputs",
                        len(ptr_args), num_inputs, num_outputs,
                    )
        except Exception as e:
            log.warning(
                "[HEURISTICS] Failed to refine tensor counts from arg_names: %s", e
            )

        # ── 2.5  Verbosity flag ─────────────────────────────────────────────────
        # All detailed print() blocks (analysis box, scoring table, validation
        # summary) are gated by this flag.  Set
        # TORCHINDUCTOR_HEURISTICS_VERBOSE=0 to suppress them in production.
        from torch._inductor import config as inductor_config
        _verbose = inductor_config.heuristics_verbose

        # ── 2.6  Print kernel + problem + device analysis block ───────────────
        if _verbose:
            try:
                from torch._inductor.codegen.triton_heuristics_kernel_analysis import (
                    extract_kernel_metadata,
                )
                from torch._inductor.codegen.triton_heuristics_adaptive import (
                    BottleneckAnalysis,
                )

                km  = extract_kernel_metadata(kernel_code) if kernel_code else {}
                dc  = BottleneckAnalysis._get_device_constants()

                n_fast  = km.get('fast_ops',   0)
                n_med   = km.get('medium_ops',  0)
                n_slow  = km.get('slow_ops',    0)
                n_in    = km.get('num_inputs',  problem_metadata.get('num_inputs',  '?'))
                n_out   = km.get('num_outputs', problem_metadata.get('num_outputs', '?'))
                n_tens  = km.get('num_tensors', '?')
                ops_el  = km.get('ops_per_element', problem_metadata.get('ops_per_element', '?'))
                bcast   = '✓' if km.get('has_broadcast', False) else '✗'
                masking = '✓' if km.get('has_mask',      False) else '✗'

                n_elem   = problem_metadata.get('total_elements', 1)
                elem_sz  = problem_metadata.get('element_size', 4)
                bpe      = km.get('bytes_per_element',
                                  problem_metadata.get('bytes_per_element', float(elem_sz) * 3))
                total_kb = (n_elem * bpe) / 1024
                total_op = n_elem * (ops_el if isinstance(ops_el, (int, float)) else 2)
                ai       = total_op / max(n_elem * bpe, 1)

                _ti = n_fast + n_med + n_slow
                if _ti > 0:
                    _sf, _mf = n_slow / _ti, n_med / _ti
                    compute_eff = 0.6 if _sf > 0.5 else (0.7 if _sf > 0.2 or _mf > 0.5 else 0.8)
                else:
                    compute_eff = 0.7

                peak_bw     = dc['memory_bandwidth_gb_s']
                peak_tflops = dc['compute_tflops']
                oi_peak = (peak_tflops * 1e12) / (peak_bw * 1e9)
                eff_bw      = peak_bw * 0.8
                eff_tflops  = peak_tflops * compute_eff
                oi_eff      = (eff_tflops * 1e12) / (eff_bw * 1e9)
                regime   = 'MEMORY-BOUND' if ai < oi_eff else 'COMPUTE-BOUND'
                ai_cmp   = f'AI {ai:.2f} {"<" if ai < oi_eff else "≥"} eff. OI {oi_eff:.1f}'
                src_info = f'fn.src ({len(kernel_code)} chars)' if kernel_code else 'no source'

                _W = 72
                def _box_line(text=''):
                    return f"[HEURISTICS] │  {text}"
                def _box_sep(label=''):
                    pad = _W - len(label) - 4
                    return f"[HEURISTICS] ├─ {label} {'─'*max(pad,0)}┤"

                print(f"[HEURISTICS] ┌─ Kernel & Problem Analysis {'─'*(_W-27)}┐", flush=True)
                print(_box_line(f"Source         :  {src_info}"), flush=True)
                print(_box_line(f"Tensor args    :  {n_tens} ptr args  →  {n_in} inputs  /  {n_out} output(s)"), flush=True)
                print(_box_line(
                    f"Instruction mix:  {n_fast} fast (add/mul/fma)  ·  {n_med} medium (div/sqrt)  ·  {n_slow} slow (exp/sin/log)"
                ), flush=True)
                print(_box_line(
                    f"Ops / element  :  {ops_el}  (weighted)   ·   Compute eff: {compute_eff:.1f}"
                    f"   ·   Broadcast: {bcast}   ·   Masking: {masking}"
                ), flush=True)
                print(_box_sep('Problem'), flush=True)
                print(_box_line(
                    f"{n_elem:,} elements  ·  {elem_sz} B/scalar  ·  {bpe:.1f} B/elem (r+w)"
                    f"  →  {total_kb:.1f} KB total"
                ), flush=True)
                top_str = (
                    f"{total_op/1e9:.2f}G" if total_op >= 1e9 else
                    f"{total_op/1e6:.2f}M" if total_op >= 1e6 else
                    f"{total_op/1e3:.1f}K"
                )
                print(_box_line(f"Total FLOPs    :  ~{top_str}   ·   Arithmetic intensity: {ai:.2f} ops/byte"), flush=True)
                print(_box_sep('Device / Roofline'), flush=True)
                print(_box_line(
                    f"Peak HW        :  {peak_tflops:.1f} TFLOPS  ·  {peak_bw:.1f} GB/s BW"
                    f"  →  peak OI ceiling {oi_peak:.1f} ops/byte"
                ), flush=True)
                print(_box_line(
                    f"Effective      :  {eff_tflops:.1f} TFLOPS (×{compute_eff})  ·  {eff_bw:.1f} GB/s (×0.8 stream)"
                    f"  →  eff. OI ridge {oi_eff:.1f} ops/byte"
                ), flush=True)
                print(_box_line(
                    f"Regime         :  {regime}  ({ai_cmp})"
                    f"  ← time model: max(memory_us, compute_us)"
                ), flush=True)
                print(f"[HEURISTICS] └{'─'*(_W+2)}┘", flush=True)
                print(flush=True)

            except Exception as _analysis_err:
                log.debug("[HEURISTICS] Analysis summary failed: %s", _analysis_err)

        # ── 3. Score all configs in ONE parallel pass ──────────────────────
        # Worker computes score + detailed factor breakdown + bottleneck for
        # each config simultaneously.  This eliminates the old two-pass design
        # (score loop then a second get_detailed_scores+analyze_bottleneck loop
        # during printing) and gives real parallelism on calls that release the
        # GIL (torch.cuda.get_device_properties in _get_device_constants).
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from torch._inductor.codegen.triton_heuristics_adaptive import (
            BottleneckAnalysis,
        )

        def _score_one_config(triton_cfg):
            try:
                effective = {
                    k: triton_cfg.kwargs[k]
                    for k in ('XBLOCK', 'YBLOCK', 'ZBLOCK')
                    if k in triton_cfg.kwargs
                }
                effective['num_warps'] = triton_cfg.num_warps
                sc = PointwiseHeuristics.score_config(effective, problem_metadata, kernel_code)
                d  = PointwiseHeuristics.get_detailed_scores(effective, problem_metadata, kernel_code)
                bn = BottleneckAnalysis.analyze_bottleneck(effective, problem_metadata, kernel_code)
                return (sc, triton_cfg, effective, d, bn)
            except Exception:
                return None

        _max_workers = min(8, max(1, len(self.configs)))
        _scored_raw = []
        with ThreadPoolExecutor(max_workers=_max_workers) as _pool:
            _futs = {_pool.submit(_score_one_config, cfg): cfg for cfg in self.configs}
            for _fut in as_completed(_futs):
                _res = _fut.result()
                if _res is not None:
                    _scored_raw.append(_res)

        if not _scored_raw:
            log.warning("[HEURISTICS] No configs could be scored – keeping all configs")
            return

        _scored_raw.sort(key=lambda x: x[0], reverse=True)

        # ── 4. Print full scoring table (verbose only) ─────────────────────
        if _verbose:
            _BN_ICON = {'overhead': '🚀 LAUNCH ', 'memory': '💾 MEMORY ', 'compute': '⚡ COMPUTE'}
            src_tag = f"fn.src ({len(kernel_code)} chars)" if kernel_code else "no kernel src"
            hdr1 = (
                f"  {'Rank':>4}  {'Score':>7}  {'Bottleneck':13s}  "
                f"{'Ohd_µs':>7}  {'Mem_µs':>7}  {'Cmp_µs':>7}  {'Tot_µs':>7}  │  "
                f"{'BW':>6}  {'Lnch':>6}  {'Grid':>6}  {'Occup':>6}  │  "
                f"{'Blks':>6}  {'Thr/blk':>7}  Config"
            )
            sep = f"  {'-'*len(hdr1)}"
            print(
                f"\n[HEURISTICS] Scoring table – {len(_scored_raw)} configs | {src_tag} | "
                f"problem: {problem_metadata.get('total_elements', '?'):,} elements",
                flush=True,
            )
            print(hdr1, flush=True)
            print(sep, flush=True)
            _top_n = inductor_config.heuristics_top_n_configs
            for _rank, (_sc, _tcfg, _eff, _d, _bn) in enumerate(_scored_raw, 1):
                _bn_label = _BN_ICON.get(_bn.get('bottleneck', ''), '?         ')
                _marker = f"  ◄ top-{_top_n}" if _rank <= _top_n else ""
                print(
                    f"  #{_rank:3d}  {_sc:7.4f}  {_bn_label}  "
                    f"{_bn.get('overhead_us', 0):7.2f}  "
                    f"{_bn.get('memory_us',  0):7.2f}  "
                    f"{_bn.get('compute_us', 0):7.2f}  "
                    f"{_bn.get('total_us',   0):7.2f}  │  "
                    f"{_d.get('memory_bandwidth', 0):6.3f}  "
                    f"{_d.get('launch_overhead',  0):6.3f}  "
                    f"{_d.get('grid_granularity', 0):6.3f}  "
                    f"{_d.get('occupancy',        0):6.3f}  │  "
                    f"{_d.get('num_blocks',        0):6d}  "
                    f"{_d.get('threads_per_block', 0):7d}  "
                    f"{_eff}{_marker}",
                    flush=True,
                )
            print(flush=True)

        # Collapse to (score, triton_cfg, effective) for downstream use
        scored = [(_sc, _tc, _eff) for _sc, _tc, _eff, _d, _bn in _scored_raw]

        # ── How many configs form the selection pool? ──────────────────────
        # Controlled by TORCHINDUCTOR_HEURISTICS_TOP_N (default 5).
        # Clamp to [1, len(scored)] so the value is always valid.
        _top_n = max(1, min(inductor_config.heuristics_top_n_configs, len(scored)))

        top_n_scored    = scored[:_top_n]
        top_n_triton    = [tcfg  for _, tcfg, _    in top_n_scored]
        top_n_heuristic = [hdict for _, _,    hdict in top_n_scored]

        log.info(
            "[HEURISTICS] Top config (N=%d): score=%.4f  %s",
            _top_n, top_n_scored[0][0], top_n_scored[0][2],
        )

        # ── 5. Store predictions for _print_heuristics_validation_summary ──
        # predicted_scores is a list of (score, effective_dict) pairs in
        # descending score order, matching the format _store_actual_timing uses.
        problem_key = _normalize_problem_key(size_hints)
        predicted_scores = [(score, hdict) for score, _, hdict in scored]
        _store_heuristics_predictions(problem_key, problem_metadata, predicted_scores, kernel_code)

        # ── 6. Apply mode logic ────────────────────────────────────────────
        if inductor_config.heuristics_real_bench:
            # REAL_BENCH: benchmark ALL configs for full validation data.
            # Winner is chosen only from the heuristic top-N (stored below).
            # self.configs is left unchanged – every config gets compiled.
            _TOP_N_CONFIGS_FOR_SELECTION[problem_key] = top_n_heuristic
            log.info(
                "[HEURISTICS] REAL_BENCH mode: all %d configs will be benchmarked, "
                "winner selected from top-%d predicted",
                len(self.configs), _top_n,
            )
            print(
                f"[HEURISTICS] REAL_BENCH mode: benchmarking all {len(self.configs)} configs, "
                f"selecting winner from heuristic top-{_top_n}",
                flush=True,
            )
        else:
            # Heuristics-only: prune self.configs to top-N before compilation.
            self.configs = top_n_triton
            log.info(
                "[HEURISTICS] Heuristics-only mode: pruned to top-%d configs",
                _top_n,
            )
            print(
                f"[HEURISTICS] Heuristics-only mode: pruned to top-{_top_n} configs"
                f" ({len(self.configs)} total)",
                flush=True,
            )

    def is_statically_launchable(self):
        """
        Checks if every compiled kernel is statically launchable, which
        allows us to efficiently cache it in FXGraphCache
        """
        if not self.compile_results:
            return False
        return all(
            isinstance(x, StaticTritonCompileResult) for x in self.compile_results
        )

    def recheck_autotune_cache(
        self, reload_kernel_from_src: Callable[[], CachingAutotuner]
    ) -> None:
        """
        On cache load on static autotuner, we need to recheck the autotune cache, since
        a best config could have been found from a previous run
        """
        assert self.is_statically_launchable()

        configs = [result.config for result in self.compile_results]

        (cached_configs, _, autotune_cache_info) = check_autotune_cache(
            configs, self.filename, self.inductor_meta
        )
        self.autotune_cache_info = autotune_cache_info
        # I.e. there was an autotune cache hit
        if len(cached_configs) == 1 and len(configs) > 1:
            best_config = cached_configs[0]
            # Grab the best compiled config, if it's in the list of available ones
            best_config_hash = triton_config_to_hashable(best_config)

            for compile_result in self.compile_results:
                if triton_config_to_hashable(compile_result.config) == best_config_hash:
                    self.compile_results = [compile_result]
                    return

            # If the best config isn't in our list of compile results,
            # it's likely because it was found by coordesc after the cache
            # already saved
            if best_config.found_by_coordesc:
                with dynamo_timed("CachingAutotuner.slow_precompile_config"):
                    if self.fn.fn is None:
                        self.fn = reload_kernel_from_src().fn
                    self.compile_results = [self._precompile_config(best_config)]

    def set_compile_info(self, compile_id: CompileId | None, is_backward: bool) -> None:
        self.compile_id = compile_id
        self.is_backward = is_backward

    def precompile(
        self,
        warm_cache_only=False,
        reload_kernel: Callable[[], CachingAutotuner] | None = None,
        static_triton_bundle_key: str | None = None,
    ):
        if warm_cache_only:
            self._precompile_worker()
            return
        with self.lock:
            # Helper function for reloading a kernel generated in a worker
            # in the parent class. Normally we don't need to reload the kernel
            # in the parent process, but in certain cases (coordesc tuning, dynamic_scale_rblock),
            # we need to actually run compilation on the parent process
            if reload_kernel is not None:
                self._reload_kernel = reload_kernel
            self._precompile_worker()
            if static_triton_bundle_key is not None and self.is_statically_launchable():
                TritonBundler.put_static_autotuner(static_triton_bundle_key, self)
            self._make_launchers()
            self._dynamic_scale_rblock()

    def _precompile_worker(self):
        if self.compile_results:
            for result in self.compile_results:
                TritonBundler.put(
                    triton_hash_to_path_key(result.kernel.hash),  # type: ignore[attr-defined]
                    self.triton_meta.get("device", 0),
                )
            return
        assert not self.launchers
        if not self.configs:
            raise NoTritonConfigsError("No triton configs are available")

        compile_results = []
        exc = None
        for c in self.configs:
            try:
                compile_results.append(self._precompile_config(c))
            except (OutOfResources, PTXASError) as e:
                exc = e
        if len(compile_results) == 0:
            raise NoTritonConfigsError(
                f"No valid triton configs. {type(exc).__name__}: {exc}"
            )
        self.compile_results = compile_results
        self.configs = None

    def _dynamic_scale_rblock(self):
        # TODO(jansel): we should find a way to move this extra compile into the worker process
        # Currently it relies on _make_launchers(), which requires a cuda context, to populate nreg.
        device_prop = self.device_props
        if (
            not self.deterministic_mode
            and self.inductor_meta.get("dynamic_scale_rblock", True)
            and not self.inductor_meta.get("persistent_reduction")
            and self.heuristic_type == HeuristicType.REDUCTION
            and self.size_hints is not None
            # Disable for Intel as Triton is not ready to return n_regs for a compiled_binary.
            and device_prop.type in ["cuda", "hip"]
            and device_prop.major
            and (device_prop.major >= 8 or torch.version.hip)
            and device_prop.regs_per_multiprocessor is not None
        ):
            assert device_prop.regs_per_multiprocessor
            assert device_prop.max_threads_per_multi_processor
            assert device_prop.multi_processor_count
            seen_config_hashes: OrderedSet[Hashable] | None = None
            warp_size = device_prop.warp_size or 32
            for result in self.compile_results:
                triton_config = result.config
                compiled_binary = result.kernel
                assert len(self.size_hints) >= 2
                xblock = triton_config.kwargs.get("XBLOCK", 1)
                reduction_kwargs = [
                    kwarg for kwarg in triton_config.kwargs if kwarg.startswith("R")
                ]
                rblocks = [triton_config.kwargs[kwarg] for kwarg in reduction_kwargs]
                total_block = (self.size_hints["x"] + xblock - 1) // xblock
                nreg = getattr(compiled_binary, "n_regs", None)
                if nreg is None:
                    continue

                # make sure rblocks are not too small
                if conditional_product(*rblocks) <= 64:
                    continue

                # each SM of A100 has 65536 32-bit registers. To maximize
                # the theoretical occupancy, we need run 2048 threads on each
                # SM. So each thread should use no more than 65536 / 2048
                # = 32 registers. In cases where occupancy matters, and each
                # thread uses too many registers, reduce R0_BLOCK to reduce
                # the register usage.
                # For kernel https://gist.github.com/shunting314/e4cccc031fe30d378b9b23c08c238cbd
                # from PLBartForCausalLM, latency improve from
                # 7.795ms to 4.883ms.
                #
                if (
                    nreg
                    <= device_prop.regs_per_multiprocessor
                    // device_prop.max_threads_per_multi_processor
                ):
                    continue

                nreg_per_warp = nreg * warp_size
                nreg_per_block = nreg_per_warp * triton_config.num_warps

                # Previously we set max_blocks_per_sm to 'max_threads_per_multi_processo / (32 * num_warps)'
                # The formula below is a tighter upper bound since we have the assumption that
                #   nreg > device_prop.regs_per_multiprocessor // device_prop.max_threads_per_multi_processor
                # due to the if condition above and:
                #   regs_per_multiprocessor / nreg_per_block
                #   = regs_per_multiprocessor / (nreg * 32 * num_warps)
                #   < regs_per_multiprocessor / ((regs_per_multiprocessor / max_threads_per_multi_processor) * 32 * num_warps)
                #   = max_threads_per_multi_processor / (32 * num_warps)
                # Using a tighter upper bound can reveal more optimization opportunities.
                max_blocks_per_sm = max(
                    device_prop.regs_per_multiprocessor // nreg_per_block, 1
                )

                if total_block <= max_blocks_per_sm * device_prop.multi_processor_count:
                    # no need to improve occupancy
                    continue
                new_config = copy.deepcopy(triton_config)

                # Reduce the largest Rn_BLOCK by a factor of 2.
                largest_rkwarg: str = max(
                    reduction_kwargs, key=triton_config.kwargs.__getitem__
                )
                new_config.kwargs[largest_rkwarg] //= 2

                if seen_config_hashes is None:
                    seen_config_hashes = OrderedSet(
                        [
                            triton_config_to_hashable(x.config)
                            for x in self.compile_results
                        ]
                    )
                new_config_hash = triton_config_to_hashable(new_config)
                if new_config_hash in seen_config_hashes:
                    continue
                seen_config_hashes.add(new_config_hash)
                log.debug(
                    "Dynamically scale down %s from TritonConfig(%s) and get a new TritonConfig(%s)",
                    largest_rkwarg,
                    triton_config,
                    new_config,
                )
                if self.fn.fn is None:
                    """
                    We are in the parent process, while this program was compiled in a worker
                    and the fn was dropped in prepare_for_pickle().  We haven't loaded the module
                    containing the real fn yet.
                    """
                    assert hasattr(self, "_reload_kernel")
                    assert callable(self._reload_kernel)
                    self.fn = self._reload_kernel().fn
                self.compile_results.append(self._precompile_config(new_config))  # noqa: B909

            self._make_launchers()

    def _make_launchers(self):
        if len(self.launchers) == len(self.compile_results):
            return

        from torch._dynamo.device_interface import DeviceGuard

        device_interface = self.get_device_interface()

        # load binary to the correct device
        with DeviceGuard(device_interface, self.triton_meta["device"]):
            # need to initialize context
            with dynamo_timed(
                "CachingAutotuner.synchronize",
                # Deliberately avoid overloading pt2_compile_events:
                log_pt2_compile_event=False,
            ):
                device_interface.synchronize(device_interface.current_device())

            launchers = []
            exc = None
            for result in self.compile_results:
                try:
                    launchers.append(result.make_launcher())

                except (OutOfResources, PTXASError, torch.cuda.OutOfMemoryError) as e:
                    exc = e
        if len(launchers) == 0:
            raise RuntimeError(f"No valid triton configs. {type(exc).__name__}: {exc}")
        self.launchers = launchers

    def prepare_for_pickle(self) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Drop stuff from triton.JITFunction that does not pickle.
        This must be called after precompile so that these things are no longer needed.
        Returns a tuple of old values
        """
        old_values = (
            self.fn.fn,
            self.fn.__globals__,
            self.fn.used_global_vals,
            self.fn.repr,
            self.launchers,
            getattr(self.fn, "_hash_lock", None),
        )
        self.fn.fn = None
        self.fn.__globals__ = None
        self.fn.used_global_vals = None
        self.fn.repr = _ConstRepr(self.fn.repr(self.fn))
        self.launchers = []
        self.fn._hash_lock = None
        return old_values

    def restore_after_unpickle(
        self, old_values: tuple[Any, Any, Any, Any, Any, Any] | None
    ) -> None:
        if old_values:
            (
                self.fn.fn,
                self.fn.__globals__,
                self.fn.used_global_vals,
                self.fn.repr,
                self.launchers,
                self.fn._hash_lock,
            ) = old_values
        else:
            # even if we don't need/have specific values, we do need the
            # _hash_lock to be a valid RLock
            self.fn._hash_lock = threading.RLock()

    def prepare_for_caching(self) -> None:
        """
        Statically Launched CUDA Kernels have a raw cubin on them
        that we don't need to store in the cache(since TritonBundler handles the collection for us)
        """
        for result in self.compile_results:
            if isinstance(result, StaticTritonCompileResult):
                # Don't save this in the inductor cache, as it is very large
                result.kernel.cubin_raw = None

    def __getstate__(self) -> dict[str, Any]:
        assert not self.launchers, (
            "pickle should not be called with after make_launchers()"
        )
        return {
            **self.__dict__,
            "lock": None,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.lock = threading.Lock()

    def get_device_interface(self):
        # this code cannot run in compile workers, because it imports from torch
        from torch._dynamo.device_interface import get_interface_for_device

        return get_interface_for_device(self.device_props.type.replace("hip", "cuda"))

    def _create_compile_meta(self, cfg: Config) -> dict[str, Any]:
        """
        Create compilation metadata for a given autotuner config. This involves
        processing the Config kwargs so that the kwargs that are not part
        of the triton signature are passed in as options to triton.compile
        instead
        """
        compile_meta = copy.deepcopy(self.triton_meta)
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages

        cfg_kwargs = cfg.kwargs
        if self.device_props.type == "hip":
            cfg_kwargs = {**cfg_kwargs}
            for k in ("matrix_instr_nonkdim", "waves_per_eu", "kpack"):
                if k in cfg_kwargs:
                    compile_meta[k] = cfg_kwargs.pop(k)
        compile_meta["constants"].update(cfg_kwargs)

        for i in get_constexprs(self.fn):
            arg_name = self.fn.arg_names[i]
            if arg_name not in compile_meta["constants"] and (
                arg_name == "num_warps" or arg_name == "num_stages"
            ):
                compile_meta["constants"][arg_name] = getattr(cfg, arg_name)
        if HAS_WARP_SPEC:
            compile_meta["num_consumer_groups"] = getattr(cfg, "num_consumer_groups", 0)
            compile_meta["num_buffers_warp_spec"] = getattr(
                cfg, "num_buffers_warp_spec", 0
            )
        compile_meta["debug"] = self.inductor_meta.get(
            "assert_indirect_indexing", True
        ) and not self.inductor_meta.get("is_hip", False)

        # device type will be "hip" rather than "cuda" here
        compile_meta["device_type"] = self.device_props.type
        compile_meta["cc"] = self.device_props.cc

        return compile_meta

    def _create_compile_options(
        self, cfg: Config, compile_meta: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create options to pass to triton.compile based on the compile metadata
        and the given config.
        """
        options = {
            "num_warps": compile_meta["num_warps"],
            "num_stages": compile_meta["num_stages"],
            "debug": compile_meta["debug"],
            "sanitize_overflow": False,  # turn off additional asserts added for overflow checks
        }
        if "enable_fp_fusion" in compile_meta:
            options["enable_fp_fusion"] = compile_meta["enable_fp_fusion"]
        if HAS_WARP_SPEC:
            options.update(
                {
                    "num_consumer_groups": compile_meta.get("num_consumer_groups", 0),
                    "num_buffers_warp_spec": compile_meta.get(
                        "num_buffers_warp_spec", 0
                    ),
                }
            )
        if self.device_props.type == "cuda":
            options.update(
                {
                    "launch_cooperative_grid": compile_meta.get(
                        "launch_cooperative_grid", False
                    ),
                    "launch_pdl": compile_meta.get("launch_pdl", False),  # True
                }
            )
        if self.device_props.type == "hip":
            if "waves_per_eu" in compile_meta:
                options["waves_per_eu"] = compile_meta["waves_per_eu"]
            if "matrix_instr_nonkdim" in compile_meta:
                options["matrix_instr_nonkdim"] = compile_meta["matrix_instr_nonkdim"]

        return options

    def _precompile_config(self, cfg: Config) -> CompileResult[_KernelType]:
        """Ahead of time compile a given autotuner config."""
        compile_meta = self._create_compile_meta(cfg)

        if self.device_props.type == "cpu":
            triton_helpers.set_driver_to_cpu()
        else:
            triton_helpers.set_driver_to_gpu()

        if not ASTSource:
            raise RuntimeError("Installed triton version too old, please upgrade")

        compile_args = (
            ASTSource(
                self.fn,
                compile_meta["signature"],
                compile_meta["constants"],
                compile_meta["configs"][0],
            ),
        )

        if self.device_props.type == "mtia":
            from mtia.host_runtime.torch_mtia.acc_flags import (  # type: ignore[import-not-found]
                build_codename,
            )

            arch = build_codename()
        else:
            arch = compile_meta["cc"]

        target = GPUTarget(
            compile_meta["device_type"],
            arch,
            cc_warp_size(compile_meta["cc"]),
        )

        options = self._create_compile_options(cfg, compile_meta)

        compile_kwargs = {
            "target": target,
            "options": options,
        }

        try:
            binary = triton.compile(*compile_args, **compile_kwargs)
        except Exception:
            log.exception(
                "Triton compilation failed: %s\n%s\nmetadata: %s",
                self.inductor_meta.get("kernel_name", "triton_"),
                self.fn.src,
                compile_meta,
            )
            raise

        # Simulate JIT Hook call
        if (
            torch._inductor.config.run_jit_post_compile_hook
            and knobs
            and getattr(knobs.runtime, "jit_post_compile_hook", None)
        ):
            try:
                hook = knobs.runtime.jit_post_compile_hook

                # base args everyone should get
                call_kwargs = dict(
                    key=getattr(self.fn, "cache_key", self.kernel_hash or str(self.fn)),
                    repr=getattr(self.fn, "src", None),
                    fn=self.fn,
                    compile=binary,
                    is_manual_warmup=False,
                    already_compiled=True,
                )

                # only add inductor_args if the hook takes it
                sig = inspect.signature(hook)
                params = sig.parameters
                if "inductor_args" in params and "config_args" in self.inductor_meta:
                    call_kwargs["inductor_args"] = self.inductor_meta["config_args"]

                hook(**call_kwargs)
            except Exception:
                log.exception("jit_post_compile_hook failed")

        TritonBundler.put(
            triton_hash_to_path_key(binary.hash), self.triton_meta.get("device", 0)
        )
        # If the binary has a cubin file to directly launch, save it on the binary
        static_launcher = StaticTritonCompileResult.can_statically_launch(
            binary, self.inductor_meta, self.triton_meta, self.heuristic_type
        )

        if static_launcher is not None:
            result = StaticTritonCompileResult(
                static_launcher, cfg, compile_meta, self.inductor_meta
            )
            return result

        return TritonCompileResult(binary, cfg, compile_meta, self.inductor_meta)

    def bench(self, launcher, *args, with_profiler=False, **kwargs):
        """Measure the performance of a given launcher"""
        # we don't skip configs with spilled registers when auto-tuning custom
        # (user-written) Triton kernels, as (i) we don't have any knowledge or
        # control over the kernel code; (ii) there is empirical evidence that
        # for some (complicated) custom Triton kernels, a register-spilling
        # config may yield the best latency.
        if (
            not self.custom_kernel
            and launcher.n_spills is not None
            and launcher.n_spills
            > self.inductor_meta.get("spill_threshold", 32 if torch.version.hip else 16)
        ):
            log.debug(
                "Skip config %s because of register spilling: %d",
                launcher.config,
                launcher.n_spills,
            )
            return float("inf")

        device_interface = self.get_device_interface()
        stream = device_interface.get_raw_stream(device_interface.current_device())

        cpu_copies = self.copy_args_to_cpu_if_needed(*args, **kwargs)

        def kernel_call():
            cloned_args, cloned_kwargs = self.maybe_clone_args(
                cpu_copies, *args, **kwargs
            )
            # reset to zero before evaluating any config
            self.reset_to_zero_args(*args, **kwargs)
            kernel_name = self.inductor_meta.get("kernel_name", "triton kernel")
            if autograd_profiler._is_profiler_enabled:
                profiler_kwargs = self.get_profiler_kwargs(stream, launcher)
                with torch._C._profiler._RecordFunctionFast(
                    kernel_name,
                    cloned_args,
                    profiler_kwargs,
                ):
                    try:
                        launcher(
                            *cloned_args,
                            **cloned_kwargs,
                            stream=stream,
                        )
                    except Exception:
                        log.error("Failed during launch %s: ", kernel_name)
                        raise

            else:
                try:
                    launcher(
                        *cloned_args,
                        **cloned_kwargs,
                        stream=stream,
                    )
                except Exception:
                    log.error("Failed during launch %s: ", kernel_name)
                    raise
            self.restore_args_from_cpu(cpu_copies)

        # only use profiler when not already in a profiler instance
        if with_profiler and not autograd_profiler._is_profiler_enabled:
            from torch._inductor.utils import do_bench_using_profiling

            timing = do_bench_using_profiling(kernel_call, warmup=10, rep=40)
        else:
            benchmark_kwargs = (
                {}
                if self.device_props.type == "cpu"
                else {"rep": 40, "is_vetted_benchmarking": True}
            )
            timing = benchmarker.benchmark(
                fn=kernel_call,
                device=self.device_props.type,
                **benchmark_kwargs,  # type: ignore[arg-type]
            )
        
        # Store timing for heuristics validation if enabled
        from torch._inductor import config as inductor_config
        from torch._inductor.runtime.hints import HeuristicType
        if (inductor_config.heuristics_real_bench and 
            self.heuristic_type == HeuristicType.POINTWISE and 
            self.size_hints is not None):
            problem_key = _normalize_problem_key(self.size_hints)
            # Extract config dict from launcher
            config_dict = {
                'XBLOCK': launcher.config.kwargs.get('XBLOCK'),
                'YBLOCK': launcher.config.kwargs.get('YBLOCK'),
                'ZBLOCK': launcher.config.kwargs.get('ZBLOCK'),
                'num_warps': launcher.config.num_warps,
            }
            # Remove None values
            config_dict = {k: v for k, v in config_dict.items() if v is not None}
            _store_actual_timing(problem_key, config_dict, timing)
        
        return timing

    def copy_args_to_cpu_if_needed(self, *args, **kwargs):
        """
        To support benchmarking in the presence of mutated args, we need to avoid
        autotuning contanminating them. We try to pass cloned args to the kernel.
        If those clones would increase the peak memory usage, however, we instead
        copy to cpu and restore them after each iteration. Figure out the args
        to be copied and do the copying.
        """
        if not self.optimize_mem:
            return {}

        copies = {}
        try:
            budget = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
        except RuntimeError:
            # Possibly a custom CUDA allocator, see https://github.com/pytorch/pytorch/issues/163257
            return {}

        def maybe_copy(name, arg):
            if name in self.mutated_arg_names and arg.is_cuda:
                nonlocal budget
                assert isinstance(arg, torch.Tensor)
                required_storage_length = compute_required_storage_length(
                    arg.size(),
                    arg.stride(),
                    0,
                )
                size = required_storage_length * arg.element_size()
                if size > budget:
                    cpu_arg = torch.empty_strided(
                        (required_storage_length,),
                        (1,),
                        dtype=arg.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    cpu_arg.copy_(
                        arg.as_strided((required_storage_length,), (1,)),
                        non_blocking=True,
                    )
                    copies[name] = (arg, cpu_arg)
                else:
                    budget -= size

        for name, arg in zip(self.fn.arg_names, args):
            maybe_copy(name, arg)

        for name, arg in kwargs.items():
            maybe_copy(name, arg)

        return copies

    def restore_args_from_cpu(self, cpu_copies):
        for pair in cpu_copies.values():
            arg, cpu_arg = pair
            required_storage_length = compute_required_storage_length(
                arg.size(),
                arg.stride(),
                0,
            )
            arg.as_strided((required_storage_length,), (1,)).copy_(
                cpu_arg, non_blocking=True
            )

    def reset_to_zero_args(self, *args, **kwargs):
        if not self.reset_to_zero_arg_names:
            return
        for i, arg in enumerate(args):
            if self.fn.arg_names[i] in self.reset_to_zero_arg_names:
                assert isinstance(
                    arg,
                    torch.Tensor,
                ), (
                    "self.reset_to_zero_arg_names should only contain valid argument names"
                )
                arg.zero_()

        for name, arg in kwargs.items():
            if name in self.reset_to_zero_arg_names:
                assert isinstance(
                    arg,
                    torch.Tensor,
                ), (
                    "self.reset_to_zero_arg_names should only contain valid argument names"
                )
                arg.zero_()

    def maybe_clone_args(
        self, exclude: Container[str], *args, **kwargs
    ) -> tuple[list[Any], dict[str, Any]]:
        """
        Prepare new args and kwargs by cloning any in-place buffers
        (that are not in the provided exclusion list), to avoid autotune
        contaminating them. Avoid cloning the other buffers because it
        leads to increased memory usage.
        """
        from ..compile_fx import clone_preserve_strides

        def prepare_arg(name, arg):
            if name in self.mutated_arg_names and name not in exclude:
                assert isinstance(arg, torch.Tensor)
                return clone_preserve_strides(arg)
            else:
                return arg

        cloned_args = [
            prepare_arg(name, arg)
            for name, arg in itertools.zip_longest(self.fn.arg_names[: len(args)], args)
        ]
        cloned_kwargs = {name: prepare_arg(name, arg) for name, arg in kwargs.items()}
        return cloned_args, cloned_kwargs

    def clone_args(self, *args, **kwargs) -> tuple[list[Any], dict[str, Any]]:
        return self.maybe_clone_args(OrderedSet(), *args, **kwargs)

    def benchmark_all_configs(self, *args, **kwargs):
        # Scoring happens upfront in _apply_pointwise_heuristics() once the kernel
        # file is available, so no re-scoring is needed here.
        with (
            dynamo_timed(
                "CachingAutotuner.benchmark_all_configs",
                log_pt2_compile_event=True,
                metadata={"kernel_name": self.inductor_meta.get("kernel_name")},
                dynamo_compile_column_us="runtime_triton_autotune_time_us",
                compile_id=self.compile_id,
                is_backward=self.is_backward,
                log_waitcounter=True,
                waitcounter_name_override="triton_autotuner",
            ),
            # Temporarily disable due to spam
            # compilation_callback.callback_handler.install_callbacks(
            #     compilation_callback.CallbackTrigger.TRITON_AUTOTUNING,
            #     str(self.compile_id),
            # ),
        ):
            timings = {
                launcher: self.bench(launcher, *args, **kwargs)
                for launcher in self.launchers
            }

            for k, v in timings.items():
                self.coordesc_tuner.cache_benchmark_result(k.config, v)

            if log.isEnabledFor(logging.DEBUG):
                log.debug("Benchmark all input configs for %s, get:", self.fn.__name__)
                for k, v in timings.items():
                    log.debug(
                        "%s: %f, nreg %d, nspill %d, #shared-mem %s",
                        k.config,
                        v,
                        k.n_regs,
                        k.n_spills,
                        k.shared,
                    )

            if metrics.is_metric_table_enabled("kernel_autotune"):
                if self.fn.fn is None:
                    self.fn = self._reload_kernel().fn

                kernel_path = self.fn.fn.__code__.co_filename
                kernel_name = self.fn.__name__

                for k, v in timings.items():
                    metrics.log_kernel_autotune_result(
                        kernel_path, kernel_name, k.config, v
                    )

            self.reset_to_zero_args(*args, **kwargs)
            return timings

    def autotune_to_one_config(self, *args, **kwargs):
        """
        Select the best configuration after benchmarking.
        
        Standard behavior:
            Benchmark all configs → select fastest
        
        Heuristics REAL_BENCH mode (pointwise kernels):
            Benchmark ALL configs → select fastest from TOP 5 predicted
            
            This hybrid approach provides:
            1. Complete validation data (all configs benchmarked)
            2. Safety net (trust heuristics but allow flexibility)
            3. Protection from noise (don't select rank #10 that got lucky)
            
            Example:
                15 configs total
                Heuristics predict: #1, #2, #3, #4, #5 as top 5
                Benchmark all 15: actual fastest is #6
                BUT we select: #1 (best from top 5)
                Gap: Only 1.2% slower than #6 (acceptable!)
                
            Why top-5 restriction?
                - Heuristics 95%+ accurate → top 5 captures best configs
                - Protects against outliers (noise in benchmarking)
                - Still get excellent performance (within 5% of optimal)
                
        See: TOP_N_SELECTION_FEATURE.md for complete rationale
        """
        start_time = time.time_ns()
        timings = self.benchmark_all_configs(*args, **kwargs)
        benchmark_time_taken_ns = time.time_ns() - start_time
        
        # Top-5 Selection Logic (Heuristics REAL_BENCH Mode)
        # -------------------------------------------------------
        # Only applies to pointwise kernels with heuristics enabled.
        # Standard kernels and non-REAL_BENCH mode select from all configs.
        from torch._inductor import config as inductor_config
        from torch._inductor.runtime.hints import HeuristicType
        
        if (inductor_config.heuristics_real_bench and 
            self.heuristic_type == HeuristicType.POINTWISE and 
            self.size_hints is not None):
            problem_key = _normalize_problem_key(self.size_hints)
            top_n_configs = _TOP_N_CONFIGS_FOR_SELECTION.get(problem_key, [])
            
            if top_n_configs:
                # Filter timings to only include top N predicted configs
                # Match launcher configs to top_n_configs by extracting config dict
                filtered_timings = {}
                for launcher, timing in timings.items():
                    # Extract config dict from launcher
                    launcher_cfg = {
                        'XBLOCK': launcher.config.kwargs.get('XBLOCK'),
                        'YBLOCK': launcher.config.kwargs.get('YBLOCK'),
                        'ZBLOCK': launcher.config.kwargs.get('ZBLOCK'),
                        'num_warps': launcher.config.num_warps,
                    }
                    # Remove None values
                    launcher_cfg = {k: v for k, v in launcher_cfg.items() if v is not None}
                    
                    # Check if this launcher's config is in top N
                    if launcher_cfg in top_n_configs:
                        filtered_timings[launcher] = timing
                
                # Select best from top N only
                if filtered_timings:
                    self.launchers = [builtins.min(filtered_timings, key=filtered_timings.get)]
                    
                    # Log that we're restricting selection
                    msg = (
                        f"[HEURISTICS] Selected from top {len(top_n_configs)} predicted configs "
                        f"(benchmarked {len(timings)} total for validation)"
                    )
                    print(msg, flush=True)
                    log.info(msg)
                else:
                    # Fallback: use all timings if matching failed
                    self.launchers = [builtins.min(timings, key=timings.get)]
            else:
                # No top N configs stored, use all timings
                self.launchers = [builtins.min(timings, key=timings.get)]
        else:
            # Not using heuristics, select from all configs
            self.launchers = [builtins.min(timings, key=timings.get)]
        
        self.autotune_time_taken_ns = (
            self.precompile_time_taken_ns + benchmark_time_taken_ns
        )

        # log the best config
        launcher = self.launchers[0]
        log.debug(
            "Best config for %s: %s: %f, nreg %d, nspill %d, #shared-mem %s",
            self.fn.__name__,
            launcher.config,
            timings[launcher],
            launcher.n_regs,
            launcher.n_spills,
            launcher.shared,
        )

        if self.save_cache_hook:
            self.save_cache_hook(
                launcher.config,
                self.autotune_time_taken_ns,
                triton_cache_hash=launcher.cache_hash,
            )
        
        # Print heuristics validation summary if enabled
        from torch._inductor import config as inductor_config
        from torch._inductor.runtime.hints import HeuristicType
        if (inductor_config.heuristics_real_bench and 
            self.heuristic_type == HeuristicType.POINTWISE and 
            self.size_hints is not None):
            problem_key = _normalize_problem_key(self.size_hints)
            _print_heuristics_validation_summary(problem_key)

    def save_gpu_kernel(self, stream, launcher):
        key = self.inductor_meta.get("kernel_name", None)  # unique kernel name
        assert key is not None, "kernel_name can not be None"
        params = {
            "mangled_name": (
                launcher.bin.metadata.name
                if hasattr(launcher.bin.metadata, "name")
                else launcher.bin.metadata["name"]
            ),
            "num_warps": (
                launcher.bin.num_warps
                if hasattr(launcher.bin, "num_warps")
                else launcher.bin.metadata.num_warps
            ),
            "shared_mem": (
                launcher.bin.shared
                if hasattr(launcher.bin, "shared")
                else launcher.bin.metadata.shared
            ),
            "stream": stream,
            # User defined triton kernels will have arbitrary kwarg names
            "config": config_to_dict(launcher.config),
            "inductor_meta": self.inductor_meta,
            "triton_meta": self.triton_meta,
            "def_args": launcher.def_args,
            "call_args": launcher.call_args,
            "global_scratch": launcher.global_scratch,
            "profile_scratch": launcher.profile_scratch,
        }
        if self.device_props.type == "xpu":
            # On the XPU backend, threads_per_warp is not always 32.
            # For Intel GEMM Triton kernels, it can be 16.
            # This information must be preserved so that the Cpp wrapper
            # can launch the kernel with the correct configuration.
            params["threads_per_warp"] = getattr(
                launcher.bin.metadata, "threads_per_warp", 32
            )

        from torch._inductor import config
        from torch._inductor.codecache import CudaKernelParamCache

        bin_type = {"hip": "hsaco", "xpu": "spv"}.get(self.device_props.type, "cubin")
        binary = launcher.bin.asm[bin_type]

        # ROCm multi-arch: capture LLVM IR
        if torch.version.hip and config.aot_inductor.emit_multi_arch_kernel:
            # Multi-arch ROCm: Capture LLVM IR for cross-architecture compilation
            asm_type = "ll"

            # llir is the key to obtain LLVM IR from triton
            asm = launcher.bin.asm.get("llir", None)

            # CRITICAL: Multi-arch compilation cannot proceed without LLVM IR
            # Fail fast with clear error message pointing to the issue
            if not asm:
                available_keys = list(launcher.bin.asm.keys())
                raise RuntimeError(
                    f"ROCm multi-arch requires LLVM IR, but none found. "
                    f"Available keys: {available_keys}. "
                    f"Triton may need to be patched to emit LLVM IR."
                )

        # Everything else: capture architecture-specific assembly
        else:
            asm_type = {"hip": "amdgcn", "cuda": "ptx", "xpu": "spv"}.get(
                self.device_props.type, None
            )
            asm = launcher.bin.asm.get(asm_type, None)

        CudaKernelParamCache.set(key, params, binary, bin_type, asm, asm_type)
        self.cuda_kernel_saved = True

    def coordinate_descent_tuning(self, launcher, *args, **kwargs):
        """
        Coordinate descent tuning can be run with or without max-autotune.

        The only difference between these two is the starting config for coordinate_descent tuning.
        E.g., assuming regular autotune only get one config C1; while max-autotune get 4 configs C1, C2, C3, C4
        and max-autotune figure out C3 is the best.

        Then if coordinate desecnt tuning is run with max-autotune disabled, it will start from C1;
        while if coordinate descent tuning is run with max-autotune enabled, it will start from C3.
        """
        if self.heuristic_type in (
            HeuristicType.TEMPLATE,
            HeuristicType.USER_AUTOTUNE,
            HeuristicType.FIXED,
        ):
            # skip triton template
            return launcher

        if self.deterministic_mode and self.heuristic_type in (
            HeuristicType.REDUCTION,
            HeuristicType.PERSISTENT_REDUCTION,
            HeuristicType.SPLIT_SCAN,
        ):
            # Not only RBLOCK size matters for numericals of reduction.
            # num_warps also matters since that affect how much data
            # is handled by each thread, how many warp-reduction we do
            # in parallel and how much data is there for block
            # reduction.
            return launcher

        with dynamo_timed(
            "CachingAutotuner.coordinate_descent_tuning",
            # These generate too many pt2_compile_event logs:
            log_pt2_compile_event=False,
            metadata={"kernel_name": self.inductor_meta.get("kernel_name")},
            dynamo_compile_column_us="runtime_triton_autotune_time_us",
            compile_id=self.compile_id,
            is_backward=self.is_backward,
            log_waitcounter=True,
            waitcounter_name_override="triton_autotuner",
        ):
            return self._coordinate_descent_tuning(launcher, *args, **kwargs)

    def _coordinate_descent_tuning(self, launcher, *args, **kwargs):
        config2launcher = {launcher.config: launcher}

        # TODO: should we just load the kernels ahead of time if we know we're going to call this?
        if self.fn.fn is None:
            """
            We are in the parent process, while this program was compiled in a worker
            and the fn was dropped in prepare_for_pickle().  We haven't loaded the module
            containing the real fn yet.
            """
            assert hasattr(self, "_reload_kernel")
            assert callable(self._reload_kernel)
            self.fn = self._reload_kernel().fn

        def benchmark_one_config(config):
            with self.lock:
                launcher = self._precompile_config(config).make_launcher()
            config2launcher[config] = launcher

            out = self.bench(launcher, *args, **kwargs)
            counters["inductor"]["coordesc_tuning_bench"] += 1
            log.debug(
                "COORDESC: %s: %f, nreg %d, nspill %d, #shared-mem %d",
                launcher.config,
                out,
                launcher.n_regs,
                launcher.n_spills,
                launcher.shared,
            )
            return out

        assert not (
            self.heuristic_type == HeuristicType.PERSISTENT_REDUCTION
            and "R0_BLOCK" in launcher.config.kwargs
        ), (
            "Coordinate descent tuner relies on the assumption that persistent reduction's triton config does not have R0_BLOCK"
        )
        start_time = time.time_ns()
        best_config = self.coordesc_tuner.autotune(
            benchmark_one_config, launcher.config, None
        )
        coordesc_time_taken_ns = time.time_ns() - start_time
        best_config.found_by_coordesc = True

        if self.save_cache_hook:
            self.save_cache_hook(
                best_config,
                self.autotune_time_taken_ns + coordesc_time_taken_ns,
                found_by_coordesc=True,
            )

        if best_config not in config2launcher:
            # On a Coordesc cache hit, we might not have loaded the launcher
            # This can happen because PyCodeCache saves CachingAutotuners in memory,
            # even for separate compile IDs (which can have different inputs without changing output code)
            config2launcher[best_config] = self._precompile_config(
                best_config
            ).make_launcher()

        fn_hash = generate_lookup_hash_from_source_code(
            str(self.size_hints), self.fn.src
        )
        log.debug("Function hash %s has best config %s", fn_hash, best_config)
        return config2launcher[best_config]

    def get_profiler_kwargs(self, stream, launcher):
        kernel_kwargs_str = ",".join(
            f"{k}={v}" for (k, v) in launcher.config.kwargs.items()
        )

        ret = {
            "kernel_file": (self.filename or ""),
            "kernel_hash": self.kernel_hash,
            "kernel_backend": "triton",
            "stream": stream,
            "num_warps": launcher.config.num_warps,
            "num_stages": launcher.config.num_stages,
            "kernel_kwargs": kernel_kwargs_str,
        }
        if "kernel_name" in self.inductor_meta:
            ret["kernel_name"] = self.inductor_meta["kernel_name"]
        if "kernel_flop" in self.inductor_meta:
            ret["kernel_flop"] = self.inductor_meta["kernel_flop"]
        if "kernel_num_gb" in self.inductor_meta:
            ret["kernel_num_gb"] = self.inductor_meta["kernel_num_gb"]
        return ret

    def run(
        self,
        *args,
        stream,
        benchmark_run=False,
        **kwargs,
    ):  # type:ignore[override]
        """Launch triton kernel call and return result."""
        debug_mode = get_active_debug_mode()
        debug_call = None
        if debug_mode:
            arg_names = list(self.triton_meta.get("signature", {}).keys())
            kernel_kwargs = dict(zip(arg_names, args))
            kernel_kwargs.update(kwargs)
            debug_call = debug_mode.record_triton_kernel(
                kernel_name=self.fn.__name__, kwargs=kernel_kwargs
            )

        if hasattr(triton, "set_allocator"):

            def alloc_fn(size: int, align: int, stream: int | None):
                return torch.empty(
                    size, dtype=torch.int8, device=self.device_props.type
                )

            triton.set_allocator(alloc_fn)

        if self.triton_interpret:
            args, grid = self._interpret_args_grid(args, self.configs[0])
            return self.fn[grid](
                *args,
                **kwargs,
                **self.configs[0].kwargs,
            )

        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                start_time = time.time_ns()
                self.precompile()
                self.precompile_time_taken_ns = time.time_ns() - start_time
            if len(self.launchers) > 1:
                self.autotune_to_one_config(*args, **kwargs)

        if not getattr(
            self.launchers[0].config, "found_by_coordesc", False
        ) and self.inductor_meta.get("coordinate_descent_tuning", False):
            self.launchers = [
                self.coordinate_descent_tuning(self.launchers[0], *args, **kwargs)
            ]

        (launcher,) = self.launchers
        if launcher.store_cubin and (not benchmark_run or not self.cuda_kernel_saved):
            self.save_gpu_kernel(stream, launcher)

        # PyTorch execution trace replay calls CachingAutotuner::run() instead of calls launcher
        # so _RecordFunctionFast need to capture the args into CachingAutotuner::run()
        # make a copy here to avoid mutating the original args
        args_without_constexprs = tuple(args)

        if self.dump_launch_params:
            new_args, grid = self._interpret_args_grid(args, launcher.config)
            _dump_launch_params(new_args, kwargs, launcher, self.fn.__name__, grid)

        # it is faster than entering and exiting a context manager, even if the context
        # manager is a nullcontext.
        if autograd_profiler._is_profiler_enabled:
            profiler_kwargs = self.get_profiler_kwargs(stream, launcher)

            with torch._C._profiler._RecordFunctionFast(
                self.inductor_meta.get("kernel_name", "triton kernel"),
                args_without_constexprs,
                profiler_kwargs,
            ):
                result = launcher(
                    *args,
                    **kwargs,
                    stream=stream,
                )
        else:
            result = launcher(
                *args,
                **kwargs,
                stream=stream,
            )

        if debug_call:
            debug_call.finalize(self.get_device_interface())
        return result

    def _interpret_args_grid(
        self, args: tuple[Any, ...], cfg: Config
    ) -> tuple[tuple[Any, ...], tuple[int, int, int]]:
        if triton_version_uses_attrs_dict():

            def filtered_signature() -> list[str]:
                # constexprs are not passed in as args
                new_signature: list[str] = []
                from triton.runtime.interpreter import InterpretedFunction

                for i, x in enumerate(self.triton_meta["signature"].keys()):
                    if isinstance(self.fn, InterpretedFunction):
                        # These are torch compiled triton kernels that definitely
                        # have block size configs. Dynamo does not currently
                        # trace user defined triton kernels when TRITON_INTERPRET=1
                        if x not in cfg.kwargs:
                            new_signature.append(x)
                    elif i not in get_constexprs(self.fn):
                        # use constexprs rather than just configs since user
                        # defined triton kernels may not have any configs
                        new_signature.append(x)

                return new_signature

        else:

            def filtered_signature() -> list[str]:
                return list(self.triton_meta["signature"].keys())

        grid = GridExpr.from_meta(
            self.inductor_meta, cfg, mode=self.grid_mode
        ).eval_slow(
            dict(
                zip(
                    [
                        *filtered_signature(),
                        *self.inductor_meta.get("extra_launcher_args", ()),
                    ],
                    args,
                )
            )
        )
        if self.inductor_meta.get("extra_launcher_args"):
            args = args[: -len(self.inductor_meta["extra_launcher_args"])]
        return args, grid


class _ConstRepr:
    def __init__(self, value: str):
        self.value = value

    def __call__(self, _=None) -> str:
        return self.value


class CompileResult(Generic[_T]):
    """
    Base class representing compiled result.
    """

    def __init__(
        self,
        kernel: _T,
        config: Config,
        compile_meta: dict[str, Any],
        inductor_meta: dict[str, Any],
    ):
        self.kernel = kernel
        self.config = config
        self.compile_meta = compile_meta
        self.inductor_meta = inductor_meta

    def make_launcher(self) -> LauncherType: ...

    def _gen_launcher_code(self, scope, def_args, runner_args) -> LauncherType:
        grid = GridExpr.from_meta(self.inductor_meta, self.config)
        # grid.prefix is usually empty, grid.x_grid is something like `-(xnumel//-1024)`
        lines = [
            f"def launcher({', '.join(def_args)}, stream):",
            *[f"    {line}" for line in grid.prefix],
            f"    grid_0 = {grid.x_grid}",
            f"    grid_1 = {grid.y_grid}",
            f"    grid_2 = {grid.z_grid}",
            f"    runner({', '.join(runner_args)})",
        ]
        launcher_code = "\n".join(lines)
        exec(launcher_code, scope)
        return scope["launcher"]

    def _get_arg_lists(
        self, arg_names, constexprs
    ) -> tuple[list[str], list[str], OrderedSet[str]]:
        """
        Return a bunch of intermediate lists of args needed for generating
        launcher code.
        """
        compile_meta = self.compile_meta
        cfg = self.config
        known_constants = OrderedSet(
            arg for i, arg in enumerate(arg_names) if i in constexprs
        )

        """
        https://github.com/pytorch/pytorch/issues/115344

        self.fn.constexprs doesn't properly deal with None args, so when we filter out
        an arg in UserDefinedTritonKernel.codegen, we need to filter it here as well.
        We also don't want to modify self.fn.

        We know that we removed something from the signature if:
            1. It's in compile_meta["constants"]
            2. It isn't a constant we already know about
                Note: The value of interest has already been added to compile_meta['constants'],
                    so we use self.fn.constexprs instead.
            3. It isn't in the compile_meta signature
        """
        none_args = OrderedSet(
            k
            for k, v in compile_meta["constants"].items()
            if v is None and k not in known_constants
        )
        none_args = none_args.difference(OrderedSet(compile_meta["signature"].keys()))

        def _convert_constant(constant):
            if isinstance(constant, str):
                return "r'" + constant + "'"
            else:
                return repr(constant)

        if triton_version_uses_attrs_dict():
            call_args = arg_names
            def_args = arg_names
            implicit_constants = OrderedSet(
                (
                    "num_warps",
                    "num_stages",
                )
            ).union(OrderedSet(k for k in known_constants))
            if implicit_constants := implicit_constants & OrderedSet(
                compile_meta["constants"].keys()
            ):
                # num_warps/num_stages are special implicit args that are not in the signature
                # see test_triton_kernel_special_params
                def_args = [arg for arg in def_args if arg not in implicit_constants]
                repl = {
                    k: _convert_constant(compile_meta["constants"].get(k))
                    for k in implicit_constants
                }
                call_args = [repl.get(arg, arg) for arg in call_args]
        else:
            call_args = [
                arg
                for i, arg in enumerate(arg_names)
                if i not in constexprs and arg not in none_args
            ]
            cfg_dict = config_to_dict(cfg)
            def_args = [
                name
                for name in arg_names
                if name not in cfg_dict and name not in none_args
            ]

        if "extra_launcher_args" in self.inductor_meta:
            def_args = [*def_args, *self.inductor_meta["extra_launcher_args"]]

        return call_args, def_args, none_args


class CannotStaticallyLaunchKernel(Exception):
    pass


class StaticTritonCompileResult(CompileResult[StaticallyLaunchedCudaKernel]):
    """
    TritonCompileResult that uses StaticCudaLauncher,
    which vastly simplifies the setup and metadata needed to be kept.
    """

    @staticmethod
    def can_statically_launch(
        kernel: CompiledKernel,
        inductor_meta: dict[str, Any],
        triton_meta: dict[str, Any],
        heuristic_type: HeuristicType,
    ) -> StaticallyLaunchedCudaKernel | None:
        if not torch._inductor.config.use_static_cuda_launcher:
            return None

        def check_can_launch() -> StaticallyLaunchedCudaKernel:
            if triton_meta.get("device_type") != "cuda":
                # Only cuda kernels
                raise CannotStaticallyLaunchKernel("Non-cuda device")

            if torch._inductor.config.cpp_wrapper:
                # If we're running with cpp wrapper, it doesn't
                # make sense to statically compile since everything
                # is codegenned anyway
                raise CannotStaticallyLaunchKernel("Cpp wrapper enabled")

            if (
                heuristic_type == HeuristicType.USER_AUTOTUNE
                and not torch._inductor.config.static_launch_user_defined_triton_kernels
            ):
                # Don't support user defined triton kernels yet
                raise CannotStaticallyLaunchKernel("User defined triton kernel")

            if inductor_meta.get("store_cubin"):
                # Requires storing the entire binary
                raise CannotStaticallyLaunchKernel("store_cubin is enabled")

            if getattr(kernel.metadata, "launch_pdl", False) or getattr(
                kernel.metadata, "launch_cooperative_grid", False
            ):
                raise CannotStaticallyLaunchKernel(
                    "static launch does not support launch attributes"
                )

            cubin_location = os.path.join(
                triton_cache_dir(triton_meta.get("device", 0)),
                triton_hash_to_path_key(kernel.hash),
                f"{kernel.src.fn.__name__}.cubin",
            )

            if not os.path.exists(cubin_location):
                raise CannotStaticallyLaunchKernel(
                    f"Cubin path not found: {cubin_location}"
                )

            else:
                kernel._cubin_path = cubin_location

            try:
                static_kernel = StaticallyLaunchedCudaKernel(kernel)
            except NotImplementedError as e:
                raise CannotStaticallyLaunchKernel(f"NotImplemented: {str(e)}") from e

            return static_kernel

        try:
            result = check_can_launch()
            return result
        except CannotStaticallyLaunchKernel as e:
            log.info("Bypassing StaticallyLaunchedCudaKernel due to %s", str(e))  # noqa: G200
            if torch._inductor.config.strict_static_cuda_launcher:
                raise e
            return None

    def reload_cubin_path(self):
        """
        When loading from cache on disk, we want to reload cubin
        files from their appropriate location on disc.
        """
        cubin_location = os.path.join(
            triton_cache_dir(self.compile_meta.get("device", 0)),
            triton_hash_to_path_key(self.kernel.hash),
            f"{self.kernel.name}.cubin",
        )
        if not os.path.exists(cubin_location):
            if self.kernel.cubin_raw is not None:
                # We saved the raw cubin, so write it to he appropriate location
                self.kernel.reload_cubin_from_raw(cubin_location)
            else:
                raise RuntimeError(
                    "Cubin file saved by TritonBundler not found at %s", cubin_location
                )
        self.kernel.cubin_path = cubin_location

    def make_launcher(self) -> LauncherType:
        # If at least one static make_launcher call occurs,
        # we're sure static cuda launcher was used for this compile
        set_feature_use("static_cuda_launcher", True)
        # Load the binary on the parent
        if not self.kernel.cubin_path:
            self.reload_cubin_path()
        device = self.compile_meta.get("device", 0)
        if device is None:
            device = 0
        self.kernel.load_kernel(device)
        scope = {
            "runner": self.kernel.run,
        }

        # NOTE: Constexpr handling for triton and static cuda launcher

        # Triton kernels have two types of constexprs: *declared* ones, which are ones the user
        # has explicitly declared as tl.constexpr, and *implied* ones, which are expressions triton
        # deems constant while compiling/analyzing the code (i.e. unused parameters, for example)

        # Triton kernels handle constexprs slightly differently depending on which version of triton
        # we care about (we support 3.2.0 and 3.3.0).

        # In 3.2.0, triton kernels do not require passing any declared constexprs into the kernel
        # In 3.3.0, triton kernels require all declared constexprs be passed into the kernel, where
        # they are subsequently ignored.
        # When statically launching, since we're launching from the triton generated cubin, we actually want to
        # always get rid of all const exprs, declared or implied, since the underlying cubin file has all
        # of the constants stripped away anyway.

        # But CachingAutotuner.run will pass us a different number of arguments depending on
        # whether or not we're in triton 3.2.0 or later, so we grab def_args with the same logic
        # as the (non static) TritonCompileResult. We then generate call_args ourselves, since we
        # want only a subset of the arguments passed to triton.
        # Here, arg_names is exactly fn.src.arg_names and declared_constexprs is exactly fn.src.constexprs,
        # which matches behavior with regular TritonCompileResult
        _, def_args, none_args = self._get_arg_lists(
            self.kernel.arg_names, self.kernel.declared_constexprs
        )

        call_args = [
            arg
            for i, arg in enumerate(self.kernel.arg_names)
            if i not in self.kernel.full_constexprs and arg not in none_args
        ]

        # StaticallyLaunchedCudaKernel.run takes in order grid_0, grid_1, grid_2, stream, and call_args
        runner_args = ["grid_0", "grid_1", "grid_2", "stream", *call_args]
        launcher = self._gen_launcher_code(scope, def_args, runner_args)
        launcher.config = self.config  # type: ignore[attr-defined]
        launcher.n_regs = self.kernel.n_regs  # type: ignore[attr-defined]
        launcher.n_spills = self.kernel.n_spills  # type: ignore[attr-defined]
        launcher.shared = self.kernel.shared  # type: ignore[attr-defined]
        launcher.cache_hash = triton_hash_to_path_key(self.kernel.hash)  # type: ignore[attr-defined]
        launcher.store_cubin = False  # type: ignore[attr-defined]
        launcher._is_static = True  # type: ignore[attr-defined]
        return launcher


class TritonCompileResult(CompileResult[CompiledKernel]):
    """
    Upstream Triton CompileKernel can not be pickled.  This is a wrapper
    to support serialization and generate the launcher function.
    """

    @staticmethod
    @functools.lru_cache(32)
    def _kernel_metadata_cls(fields: tuple[str, ...]) -> Any:
        return namedtuple("KernelMetadata", sorted(fields))

    @staticmethod
    def _serialize_metadata(metadata):
        """
        Triton uses a nested class called KernelMetadata to store metadata information.
        Pickle does not work well with nested namedtuples, as the namedtuple doesn't appear
        in the toplevel namespace of the module. So these serialization/deser functions
        are used to convert the namedtuples to a dict and back.

        As for packed_metadata, depending on the triton backend, KernelMetadata can be
        a namedtuple, or a regular tuple! So the serialization function branches on whether
        the metadata to be serialized is a namedtuple or regular, serializable one.
        """

        def is_namedtuple(obj) -> bool:
            return (
                isinstance(obj, tuple)
                and hasattr(obj, "_asdict")
                and hasattr(obj, "_fields")
            )

        if is_namedtuple(metadata):
            return metadata._asdict()
        else:
            return metadata

    @staticmethod
    def _deserialize_metadata(metadata):
        if isinstance(metadata, dict):
            return TritonCompileResult._kernel_metadata_cls(tuple(metadata.keys()))(
                **metadata
            )
        else:
            return metadata

    def __getstate__(self) -> dict[str, Any]:
        kernel = self.kernel
        # replace the fields that don't pickle nicely
        kernel_state = {
            **kernel.__dict__,
            # See doc about serializing metadata above
            "metadata": self._serialize_metadata(kernel.metadata),
            "packed_metadata": self._serialize_metadata(
                getattr(kernel, "packed_metadata", None)
            ),
            "module": None,  # regenerated by kernel._init_handles()
            "function": None,  # regenerated by kernel._init_handles()
            "run": None,  # regenerated by kernel._init_handles()
        }
        return {**self.__dict__, "kernel": kernel_state}  # type: ignore[dict-item]

    def __setstate__(self, state: dict[str, Any]) -> None:
        # src = ASTSource.__new__(ASTSource)
        # src.__setstate__(state["kernel"]["src"])
        # TODO(jansel): need to fixup src.fn which is now None
        kernel = CompiledKernel.__new__(CompiledKernel)
        metadata = state["kernel"]["metadata"]
        packed_metadata = state["kernel"]["packed_metadata"]
        kernel.__dict__.update(
            {
                **state["kernel"],
                # "src": src,
                "metadata": self._deserialize_metadata(metadata),
                "packed_metadata": self._deserialize_metadata(packed_metadata),
            }
        )
        self.__dict__.update(state)
        self.kernel = kernel

    def make_launcher(self) -> LauncherType:
        """
        Launching triton kernels is performance sensitive, we compile
        a custom Python function get the grid() and reorder the args to
        the underlying wrapper.
        """
        cfg = self.config
        compile_meta = self.compile_meta
        binary = self.kernel
        fn = binary.src.fn
        binary._init_handles()
        (call_args, def_args, none_args) = self._get_arg_lists(
            fn.arg_names, get_constexprs(fn)
        )
        binary_shared = (
            binary.shared if hasattr(binary, "shared") else binary.metadata.shared
        )

        if knobs is None:
            launch_enter = binary.__class__.launch_enter_hook
            launch_exit = binary.__class__.launch_exit_hook
        else:
            launch_enter = knobs.runtime.launch_enter_hook
            launch_exit = knobs.runtime.launch_exit_hook

        import math as math_lib

        import triton as triton_lib

        import torch as torch_lib

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "launch_enter_hook": launch_enter,
            "launch_exit_hook": launch_exit,
            "metadata": (
                binary.packed_metadata
                if hasattr(binary, "packed_metadata")
                else binary.metadata
            ),
            "shared": binary_shared,
            "num_warps": (
                binary.num_warps
                if hasattr(binary, "num_warps")
                else binary.metadata.num_warps
            ),
            "cta_args": (
                (
                    binary.num_ctas,
                    *get_first_attr(binary, "cluster_dims", "clusterDims"),
                )
                if hasattr(binary, "num_ctas")
                else (
                    (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
                    if hasattr(binary, "metadata")
                    and hasattr(binary.metadata, "num_ctas")
                    and hasattr(binary.metadata, "cluster_dims")
                    else ()
                )
            ),
            "function": get_first_attr(binary, "function", "cu_function"),
            "runner": get_first_attr(binary, "run", "c_wrapper"),
            "math": math_lib,
            "torch": torch_lib,
            "triton": triton_lib,
        }

        if not hasattr(binary, "launch_metadata"):
            # launch args before CompiledKernel.launch_metadata is added.
            # TODO(jansel): delete this branch in mid-2025
            runner_args = [
                "grid_0",
                "grid_1",
                "grid_2",
                "num_warps",
                "*cta_args",
                "shared",
                "stream",
                "function",
                "launch_enter_hook",
                "launch_exit_hook",
                "metadata",
                *call_args,
            ]
        else:  # args after CompiledKernel.launch_metadata: https://github.com/triton-lang/triton/pull/3492
            # Getting the kernel launch args is extremely perf-sensitive.  Evaluating
            # `bin.launch_metadata` is relatively expensive, and returns None unless a
            # `launch_enter_hook` is installed.  So if we don't have that hook installed,
            # we want to burn None in to the launch args with zero overhead.
            # See https://github.com/pytorch/pytorch/issues/123597
            if launch_enter:
                launch_metadata = f"bin.launch_metadata((grid_0, grid_1, grid_2), stream, {', '.join(call_args)})"
            else:
                launch_metadata = "None"
            runner_args = [
                "grid_0",
                "grid_1",
                "grid_2",
                "stream",
                "function",
                "metadata",
                launch_metadata,
                "launch_enter_hook",
                "launch_exit_hook",
                *call_args,
            ]

        launcher = self._gen_launcher_code(scope, def_args, runner_args)

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = binary_shared
        launcher.cache_hash = triton_hash_to_path_key(binary.hash)
        launcher.store_cubin = self.inductor_meta.get("store_cubin", False)
        # store this global variable to avoid the high overhead of reading it when calling run
        if launcher.store_cubin:
            launcher.fn = fn
            launcher.bin = binary
            if triton_version_uses_attrs_dict():
                # arg filtering wasn't done above
                cfg_dict = config_to_dict(cfg)
                def_args = [x for x in def_args if x not in cfg_dict]
                call_args = [
                    x
                    for x in call_args
                    if compile_meta["signature"].get(x, "constexpr") != "constexpr"
                    and x not in none_args
                ]
            launcher.def_args = def_args
            launcher.call_args = call_args
            kernel_metadata = getattr(self.kernel, "metadata", None)

            # for the scratch arguments: None indicates that the kernel doesn't
            # take any scratch argument; otherwise a number indicates the number
            # of bytes of scratch that need to be provided.

            # in AMD's Triton backend, the global scratch size is never provided
            # (but for AMD it's safe to pass an extra null arg, so always include it)
            global_scratch: int | None = getattr(
                kernel_metadata,
                "global_scratch_size",
                (0 if torch.version.hip else None),
            )
            profile_scratch: int | None = getattr(
                kernel_metadata, "profile_scratch_size", None
            )
            launcher.global_scratch = global_scratch
            launcher.profile_scratch = profile_scratch
        return launcher


def _find_names(obj):
    import gc
    import inspect

    frame = inspect.currentframe()
    while frame is not None:
        frame.f_locals
        frame = frame.f_back
    obj_names = []
    for referrer in gc.get_referrers(obj):
        if isinstance(referrer, dict):
            for k, v in referrer.items():
                if v is obj:
                    obj_names.append(k)
    return obj_names


collected_calls: list[Any] = []


def start_graph():
    collected_calls.clear()


def end_graph(output_file):
    if len(collected_calls) == 0:
        return
    overall_time = sum(call[0] for call in collected_calls)
    overall_gb = sum(call[1] for call in collected_calls)
    cur_file = inspect.stack()[1].filename
    summary_str = (
        f"SUMMARY ({cur_file})\n"
        f"{overall_time:.2f}ms   \t {overall_gb:.2f} GB\t {overall_gb / (overall_time / 1e3):.2f}GB/s"
    )
    log.info(
        "%s",
        summary_str,
    )
    if output_file is not None:
        # sort perf numbers in descending order, i.e. placing the
        # most runtime-heavy kernels at the top of the list
        sorted_calls = sorted(collected_calls, key=lambda c: float(c[0]), reverse=True)
        try:
            with open(output_file, "a") as file:
                log.info(
                    "Save profile bandwidth results to %s",
                    output_file,
                )
                file.write("====================\n")
                file.write(f"TRITON KERNELS BANDWIDTH INFO ({cur_file})\n")
                for ms, num_gb, gb_per_s, kernel_name in sorted_calls:
                    # also display the runtime percentage for each kernel
                    percentage = f"{ms / overall_time * 100:.2f}%"
                    suffix = f" \t {percentage} \t {kernel_name}"
                    bw_info_str = create_bandwidth_info_str(
                        ms,
                        num_gb,
                        gb_per_s,
                        suffix=suffix,
                        color=False,
                    )
                    file.write(bw_info_str + "\n")
                file.write(f"{summary_str}\n\n")
        except Exception:
            log.warning(
                "failed to write profile bandwidth result into %s",
                output_file,
                exc_info=True,
            )


class DebugAutotuner(CachingAutotuner):
    def __init__(
        self,
        *args,
        regex_filter="",
        with_profiler=False,
        with_bandwidth_info=True,
        **kwargs,
    ):
        self.regex_filter = regex_filter
        self.with_profiler = with_profiler
        self.with_bandwidth_info = with_bandwidth_info
        super().__init__(*args, **kwargs)
        self.cached = None

    def run(self, *args, stream, **kwargs):
        if not self.with_bandwidth_info:
            super().run(*args, stream=stream, **kwargs, benchmark_run=True)
            return
        else:
            possible_names = _find_names(self)
            kernel_name = f"{max(possible_names, key=len)}"
            if not re.match(self.regex_filter, kernel_name):
                return
            if len(self.launchers) != 1:
                if len(self.launchers) == 0:
                    start_time = time.time_ns()
                    self.precompile()
                    self.precompile_time_taken_ns = time.time_ns() - start_time
                if len(self.launchers) > 1:
                    self.autotune_to_one_config(*args, **kwargs)
            (launcher,) = self.launchers

            if launcher.store_cubin:
                self.save_gpu_kernel(stream, launcher)

            if self.cached is None:
                ms = self.bench(launcher, *args, with_profiler=self.with_profiler)
                num_in_out_ptrs = len(
                    [
                        arg_name
                        for arg_name in self.fn.arg_names
                        if arg_name.startswith("in_out_ptr")
                    ]
                )
                num_gb = self.inductor_meta.get("kernel_num_gb", None)
                if num_gb is None:
                    num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9
                gb_per_s = num_gb / (ms / 1e3)
                self.cached = ms, num_gb, gb_per_s, kernel_name
                collected_calls.append((ms, num_gb, gb_per_s, kernel_name))
                log.info(
                    "%s",
                    create_bandwidth_info_str(
                        ms, num_gb, gb_per_s, suffix=f" \t {kernel_name}"
                    ),
                )
            else:
                # in AOTI, we will call the kernel and its timing info has been cached already
                collected_calls.append(self.cached)


def hash_configs(configs: list[Config]):
    """
    Hash used to check for changes in configurations
    """
    hasher = hashlib.sha256()
    for cfg in configs:
        hasher.update(
            f"{sorted(cfg.kwargs.items())} {cfg.num_warps} {cfg.num_stages}\n".encode()
        )
    return hasher.hexdigest()


def cached_autotune(
    size_hints: list[int] | None,
    configs: list[Config],
    triton_meta,
    heuristic_type,
    filename=None,
    inductor_meta=None,
    custom_kernel=False,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)
    assert len(configs) == 1 or filename
    inductor_meta = {} if inductor_meta is None else inductor_meta

    configs, autotune_cache, autotune_cache_info = check_autotune_cache(
        configs, filename, inductor_meta
    )
    mutated_arg_names = inductor_meta.pop("mutated_arg_names", ())
    optimize_mem = inductor_meta.pop("optimize_mem", True)

    if "restore_value" in triton_meta:
        mutated_arg_names += triton_meta.pop("restore_value")

    reset_to_zero_arg_names: list[str] = []
    if "reset_to_zero" in triton_meta:
        reset_to_zero_arg_names.extend(triton_meta.pop("reset_to_zero"))

    def decorator(fn):
        # Remove XBLOCK from config if it's not a function argument.
        # This way, coordinate descent tuning will not try to tune it.
        #
        # Context: When TritonKernel.no_x_dim is True, we hardcode XBLOCK to 1.
        import inspect

        if "XBLOCK" not in inspect.signature(fn.fn).parameters:
            for tconfig in configs:
                if "XBLOCK" in tconfig.kwargs:
                    assert tconfig.kwargs["XBLOCK"] == 1
                    tconfig.kwargs.pop("XBLOCK")

        if inductor_meta.get("profile_bandwidth"):
            return DebugAutotuner(
                fn,
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                regex_filter=inductor_meta["profile_bandwidth_regex"],
                with_profiler=inductor_meta[
                    "profile_bandwidth_with_do_bench_using_profiling"
                ],
                configs=configs,
                save_cache_hook=autotune_cache and autotune_cache.save,
                mutated_arg_names=mutated_arg_names,
                reset_to_zero_arg_names=reset_to_zero_arg_names,
                optimize_mem=optimize_mem,
                heuristic_type=heuristic_type,
                size_hints=size_hints,
                custom_kernel=custom_kernel,
                filename=filename,
                with_bandwidth_info=True,
            )
        return CachingAutotuner(
            fn,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            configs=configs,
            save_cache_hook=autotune_cache and autotune_cache.save,
            mutated_arg_names=mutated_arg_names,
            reset_to_zero_arg_names=reset_to_zero_arg_names,
            optimize_mem=optimize_mem,
            heuristic_type=heuristic_type,
            size_hints=size_hints,
            custom_kernel=custom_kernel,
            filename=filename,
            autotune_cache_info=autotune_cache_info,
        )

    return decorator


def unique_configs(configs: list[Config]):
    """Remove duplicate configurations"""
    seen: OrderedSet[Hashable] = OrderedSet()
    pruned_configs = []

    for cfg in configs:
        key = triton_config_to_hashable(cfg)
        if key not in seen:
            seen.add(key)
            pruned_configs.append(cfg)
    return pruned_configs


def check_config(cfg, *, xnumel=None, ynumel=None, znumel=None):
    for numel, label in zip((xnumel, ynumel, znumel), "XYZ"):
        if numel is None:
            continue
        block = cfg[f"{label}BLOCK"]
        if numel == 1:
            assert block == 1, (
                f"TritonKernel.indexing assumes numel == 1 => BLOCK == 1"
                f" but {label.lower()}numel=={numel} and {label}BLOCK={block} (cfg={cfg})."
            )
        max_block = TRITON_MAX_BLOCK[label]
        max_block_str = f'config.triton.max_block["{label}"]'
        assert max_block % block == 0, (
            f"TritonKernel.indexing assumes {label}BLOCK divides {max_block_str}"
            f" but {label}BLOCK={block} and {max_block_str}={max_block} (cfg={cfg})."
        )


def check_max_block(cfg: dict[str, int]):
    """
    Check that block sizes are within the maximum allowed.
    """
    for var, val in cfg.items():
        block_suffix = "BLOCK"
        if block_suffix in var:
            prefix = var.removesuffix(block_suffix)
            max_block = TRITON_MAX_BLOCK[prefix]
            assert val <= max_block, (
                f"'{var}' too large. Maximum: {max_block}. Actual: {val}."
            )


def _num_warps(num_warps, max_num_warps=8, min_num_warps=2, register_intensive=False):
    # On AMD GPU each warp has 64 lanes which is double the size on NV GPU,
    # therefore using half the number of warps here correspondingly.
    if torch.version.hip:
        max_num_warps = (max_num_warps + 1) // 2
        min_num_warps = (min_num_warps + 1) // 2
    # persistent reduction is register intensive
    if register_intensive:
        max_num_warps = max_num_warps // 2
    return next_power_of_2(min(max(num_warps, min_num_warps), max_num_warps))


def _check_max_grid_x(size_hints, x, num_warps):
    # Check if maxGridSize is exceeded - if so then must scale XBLOCK further
    max_grid_x = 2147483647
    warp_size = (
        64 if torch.version.hip else 32
    )  # TODO: query warp size once #129663 is merged
    num_blocks = (size_hints["x"] + x - 1) // x

    while (num_blocks * num_warps * warp_size) > max_grid_x and x < size_hints["x"]:
        x *= 2  # Scale up XBLOCK if grid exceeds limits
        num_blocks = num_blocks // 2
    if (num_blocks * num_warps * warp_size) > max_grid_x:
        raise AssertionError(
            "Reduction config exceeds cudaDeviceProp maxGridSize. Please raise a pytorch issue"
        )
    return x, num_blocks


def triton_config(
    size_hints,
    x,
    y=None,
    z=None,
    num_stages=1,
    num_elements_per_warp=256,
    min_elem_per_thread=0,
    num_warps=None,
    matrix_instr=None,
    waves_per_eu=None,
) -> Config:
    """
    Construct a pointwise triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.

    num_elements_per_warp is a suggestion for controlling how many warps
    the triton config should contain. e.g.: if x=16, y=8, z=4 then
    num_elements = 16*8*4 = 512. Then if we set num_elements_per_warp=128,
    we'll launch 512 (elem) / 128 (elem/warp) = 4 warps. Note that it's
    just a suggestion, and sometimes other adjustment heuristics will
    override the num_elements_per_warp.

    min_elem_per_thread controls the minimum number of elements
    processed by each thread. It's always enforced.
    """
    # Ideally we want to read this from some device config

    maxGridSize = [2147483647, 65535, 65535]

    target = conditional_product(x, y, z)
    if conditional_product(*size_hints.values()) < target:
        target //= 8

    # shrink sizes to size hints
    x = min(x, size_hints["x"])
    if y:
        y = min(y, size_hints["y"])
    if z:
        z = min(z, size_hints["z"])

    # if we are below original block size, scale up where we can;
    # or if the calculated grid size is larger than the limit, we bump up the corresponding dimension
    while x < min(size_hints["x"], TRITON_MAX_BLOCK["X"]) and (
        x * maxGridSize[0] < size_hints["x"] or conditional_product(x, y, z) < target
    ):
        x *= 2
    while (
        y
        and y < min(size_hints["y"], TRITON_MAX_BLOCK["Y"])
        and (
            y * maxGridSize[1] < size_hints["y"]
            or conditional_product(x, y, z) < target
        )
    ):
        y *= 2
    while (
        z
        and z < min(size_hints["z"], TRITON_MAX_BLOCK["Z"])
        and (
            z * maxGridSize[2] < size_hints["z"]
            or conditional_product(x, y, z) < target
        )
    ):
        z *= 2

    # Calculate num_warps if they are not hard passed to config
    if num_warps is None:
        num_warps = _num_warps(
            conditional_product(x, y, z) // num_elements_per_warp, min_num_warps=1
        )
    # we are going to arrive at 2 warps only if bs was too small due to
    # numel being too small. However to workaround some ptx bugs we still
    # want at least 4 warps if there's enough elements per thread
    # given that this is a rare situation, don't expect this to affect perf
    # in general
    # see https://github.com/pytorch/pytorch/pull/97950
    if conditional_product(x, y, z) >= 128 and not torch.version.hip:
        num_warps = max(num_warps, 4)
    xnumel = size_hints["x"]
    ynumel = size_hints.get("y")
    znumel = size_hints.get("z")

    # Increase x to satisfy min_elem_per_thread requirements.
    block_size = max(
        conditional_product(x, y, z),
        min_elem_per_thread * _NUM_THREADS_PER_WARP * num_warps,
    )
    x *= math.ceil(block_size / conditional_product(x, y, z))

    x, _num_blocks = _check_max_grid_x(size_hints, x, num_warps)
    x = min(x, size_hints["x"])

    cfg = {"XBLOCK": x}
    if y:
        cfg["YBLOCK"] = y
    if z:
        cfg["ZBLOCK"] = z
    check_max_block(cfg)
    check_config(cfg, xnumel=xnumel, ynumel=ynumel, znumel=znumel)
    config = Config(cfg, num_warps=num_warps, num_stages=num_stages)

    if torch.version.hip:
        if matrix_instr is not None:
            config.kwargs["matrix_instr_nonkdim"] = matrix_instr
        if waves_per_eu is not None:
            config.kwargs["waves_per_eu"] = waves_per_eu

    return config


def _get_nd_reduction_numels(r: int, size_hints: dict[str, int]) -> dict[str, int]:
    """
    Converts a linear reduction numel to ND, in row major order.
    This order is often desirable as it presents opportunities to coalesce memory
    accesses.
    For example, if r = 64 and size_hints = [32,32], this function returns [32, 2].
    This unraveling works because both r and size_hints are powers of 2.
    """
    # Shrink r to size_hints.
    r = min(r, get_total_reduction_numel(size_hints))
    num_reduction_dims = len(
        [prefix for prefix in size_hints if prefix_is_reduction(prefix)]
    )

    remaining = r
    rnumels = {}
    for idx in range(num_reduction_dims - 1, -1, -1):
        prefix = f"r{idx}_"
        max_size = min(size_hints[prefix], TRITON_MAX_BLOCK[prefix.upper()])
        dim = min(max_size, remaining)
        assert remaining % dim == 0, (
            f"Expected dimension '{dim}' to divide remaining size '{remaining}'"
        )
        rnumels[prefix] = dim
        remaining //= dim

    # Sanity check the results.
    final_numel = conditional_product(*rnumels.values())
    assert r == final_numel, (
        f"Expected ND reduction size ({rnumels}) to have {r} elements."
    )
    assert all(rnumels[prefix] <= size_hints[prefix] for prefix in rnumels), (
        f"rnumels exceed size_hints. {rnumels} > {size_hints}"
    )

    return rnumels


def triton_config_reduction(
    size_hints,
    x: int,
    r: int,
    num_stages=1,
    num_warps=None,
    register_intensive=False,
    waves_per_eu=None,
    dynamic_scale_rblock=True,
    reduction_hint=None,
    min_num_warps=None,
) -> Config:
    """
    Construct a reduction triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """
    # Convert the linear reduction numel into a multi-dimensional block.
    rnumels = _get_nd_reduction_numels(r, size_hints)

    # shrink sizes to size hints
    x = min(x, size_hints["x"])

    def total_numel() -> int:
        return conditional_product(x, *rnumels.values())

    target = total_numel()
    if conditional_product(*size_hints.values()) < target:
        target //= 8

    # if we are below original block size, scale up where we can
    while x < size_hints["x"] and total_numel() < target:
        x *= 2
    for prefix in sorted(rnumels):
        while rnumels[prefix] < size_hints[prefix] and total_numel() < target:
            rnumels[prefix] *= 2

    if num_warps is None:
        if reduction_hint == ReductionHint.INNER:
            # r is contiguous, ensure at least 8 elements per thread
            # xblock is usually 1-2, default to giving each thread more work
            num_warps = r // 128
        else:
            num_warps = total_numel() // 128

    max_num_warps = 16 if r <= 8192 else 32
    if min_num_warps is not None:
        _num_warps_func = functools.partial(_num_warps, min_num_warps=min_num_warps)
    else:
        _num_warps_func = _num_warps

    num_warps = _num_warps_func(
        num_warps, max_num_warps=max_num_warps, register_intensive=register_intensive
    )

    x, _num_blocks = _check_max_grid_x(size_hints, x, num_warps)

    for prefix in sorted(rnumels):
        while total_numel() > target:
            if rnumels[prefix] == 1:
                break
            rnumels[prefix] //= 2

    cfg = _get_config({"x": x, **rnumels})
    check_max_block(cfg)
    check_config(cfg, xnumel=size_hints["x"])
    config = InductorConfig(
        cfg,
        num_warps=num_warps,
        num_stages=num_stages,
        dynamic_scale_rblock=dynamic_scale_rblock,
    )

    if torch.version.hip:
        if waves_per_eu is not None:
            config.kwargs["waves_per_eu"] = waves_per_eu

    return config


def _get_config(numels: dict[str, int]) -> dict[str, int]:
    """
    Convert numels ("x", "r0_", etc.) to block sizes ("XBLOCK", "R0_BLOCK"), etc.
    """

    return {prefix.upper() + "BLOCK": numel for prefix, numel in numels.items()}


def triton_config_tiled_reduction(
    size_hints, x, y, r, num_stages=1, register_intensive=False, waves_per_eu=None
):
    """
    Construct a tile reduction triton config with some adjustment
    heuristics based on size_hints. Size_hints is a tuple of numels in
    each tile dimension and will be rounded up to the nearest power of 2.
    """
    # Convert the linear reduction numel into a multi-dimensional block.
    rnumels = _get_nd_reduction_numels(r, size_hints)

    # shrink sizes to size hints
    x = min(x, size_hints["x"])
    y = min(y, size_hints["y"])

    def total_numel() -> int:
        return conditional_product(x, y, *rnumels.values())

    target = total_numel()
    if conditional_product(*size_hints.values()) < target:
        target //= 8

    # if we are below original block size, scale up where we can
    while x < size_hints["x"] and total_numel() < target:
        x *= 2
    for prefix in sorted(rnumels):
        while rnumels[prefix] < size_hints[prefix] and total_numel() < target:
            rnumels[prefix] *= 2
    while y < size_hints["y"] and total_numel() < target:
        y *= 2

    cfg = _get_config({"x": x, "y": y, **rnumels})
    num_warps = _num_warps(total_numel() // 256, min_num_warps=1)
    num_warps = _num_warps(
        num_warps, max_num_warps=16, register_intensive=register_intensive
    )
    check_config(cfg, xnumel=size_hints["x"], ynumel=size_hints["y"])
    check_max_block(cfg)
    config = Config(cfg, num_warps=num_warps, num_stages=num_stages)
    if torch.version.hip:
        if waves_per_eu is not None:
            config.kwargs["waves_per_eu"] = waves_per_eu
    return config


def _maybe_filter_configs_for_tma_restrictions(inductor_meta, configs: list[Config]):
    tma_min_block_sizes: dict[str, int]
    if (tma_min_block_sizes := inductor_meta.get("tma_min_block_sizes")) and configs:
        # Rn blocks are not provided to the kernel for persistent reductions
        if inductor_meta.get("persistent_reduction"):
            tma_min_block_sizes = {
                block_type: block_size
                for block_type, block_size in tma_min_block_sizes.items()
                if not prefix_is_reduction(block_type.lower())
            }

        assert all(
            block_type in configs[0].kwargs for block_type in tma_min_block_sizes
        )

        # Add a config that is guaranteed to compile
        example_config = configs[0]
        config_block_sizes = {**example_config.kwargs}
        config_block_sizes.update(tma_min_block_sizes)
        new_configs = [
            Config(
                config_block_sizes,
                num_warps=example_config.num_warps,
                num_stages=example_config.num_stages,
                maxnreg=example_config.maxnreg,
                pre_hook=example_config.pre_hook,
            )
        ]
        # Remove configs that will not compile
        for c in configs:
            if all(
                c.kwargs.get(block_type) >= min_block_value
                for block_type, min_block_value in tma_min_block_sizes.items()
            ):
                new_configs.append(c)

        log.debug(
            "Filtering configs for TMA API restrictions. Input configs size: %d. Output configs size: %d",
            len(configs),
            len(new_configs),
        )
        return new_configs
    return configs


def _convert_to_pointwise_heuristics_metadata(size_hints, inductor_meta, triton_meta):
    """
    Convert Triton heuristics metadata to format expected by PointwiseHeuristics.
    
    Args:
        size_hints: Dict of dimension sizes
        inductor_meta: Inductor metadata dict
        triton_meta: Triton metadata dict
        
    Returns:
        problem_metadata dict for PointwiseHeuristics
    """
    # Extract dimensions from size_hints
    dimensions = tuple(size_hints.values())
    total_elements = functools.reduce(operator.mul, dimensions, 1)
    
    # Estimate number of inputs/outputs from metadata
    num_inputs = inductor_meta.get("num_inputs", 2)
    num_outputs = inductor_meta.get("num_outputs", 1)
    
    # Fusion depth - count fused operations
    fusion_depth = inductor_meta.get("fusion_depth", 1)
    
    # Element size - try to infer from dtype
    element_size = 4  # Default to fp32
    if "dtype" in triton_meta:
        dtype_str = str(triton_meta["dtype"])
        if "float16" in dtype_str or "half" in dtype_str or "bfloat16" in dtype_str:
            element_size = 2
        elif "float64" in dtype_str or "double" in dtype_str:
            element_size = 8
    
    # Vectorization - infer from problem, don't assume
    # Only use vector_width if explicitly provided by the caller
    vector_width = inductor_meta.get("vector_width", 1)  # Conservative default
    
    # Masking requirements
    has_mask = inductor_meta.get("has_mask", False)
    
    # Broadcasting detection
    has_broadcast = inductor_meta.get("has_broadcast", False)
    broadcast_tensor_bytes = inductor_meta.get("broadcast_tensor_bytes", 0)
    
    # Get device properties for hardware-specific parameters
    device_props = triton_meta.get("device")
    warp_size = device_props.warp_size if device_props and device_props.warp_size else 64  # Default to 64 for AMD
    max_threads_per_block = device_props.max_threads_per_block if device_props and device_props.max_threads_per_block else 1024
    
    return {
        'dimensions': dimensions,
        'total_elements': total_elements,
        'num_inputs': num_inputs,
        'num_outputs': num_outputs,
        'fusion_depth': fusion_depth,
        'element_size': element_size,
        'vector_width': vector_width,
        'has_mask': has_mask,
        'has_broadcast': has_broadcast,
        'broadcast_tensor_bytes': broadcast_tensor_bytes,
        'warp_size': warp_size,
        'max_threads_per_block': max_threads_per_block,
    }


def _apply_pointwise_heuristics(size_hints, inductor_meta, triton_meta, triton_config_with_settings, filename=None):
    """
    Generate and validate candidate configs for a pointwise kernel.

    This runs during ``pointwise()`` decorator application (post-codegen but
    before ``CachingAutotuner`` is constructed).  It intentionally does NOT
    score configs here – scoring requires ``self.fn.src`` and
    ``self.fn.arg_names`` from the Triton ``JITFunction``, which only become
    available once ``CachingAutotuner.__init__`` has run.

    Instead this function:
      1. Builds ``problem_metadata`` from the inductor / triton meta dicts.
      2. Generates the full set of candidate config dicts (block sizes × warps).
      3. Validates configs structurally (dimension matching, thread-count range).
      4. Stashes ``(all_valid_configs, problem_metadata)`` into
         ``inductor_meta['_heuristics_pending']`` so that
         ``CachingAutotuner.__init__`` can pick it up.

    Scoring, top-5 selection, and ``_TOP_N_CONFIGS_FOR_SELECTION`` population
    all happen in ``CachingAutotuner.__init__``.

    Returns:
        Tuple of (all_valid_configs, problem_metadata) or None if unavailable.
        all_valid_configs is a list of unsorted heuristic config dicts.
    """
    if not POINTWISE_HEURISTICS_AVAILABLE:
        return None

    try:
        # ── 1. Build problem metadata ──────────────────────────────────────
        problem_metadata = _convert_to_pointwise_heuristics_metadata(
            size_hints, inductor_meta, triton_meta
        )

        # ── 2. Generate all candidate configs ─────────────────────────────
        all_configs = PointwiseHeuristics.generate_all_candidate_configs(problem_metadata)
        if not all_configs:
            log.warning("[HEURISTICS] No candidate configs generated")
            return None

        # ── 3. Basic structural validation (no scoring yet) ───────────────
        # Full scoring happens in CachingAutotuner.__init__ once self.fn.src
        # and self.fn.arg_names are available.
        valid_configs = []
        problem_dims = PointwiseHeuristics.get_problem_dimensions(problem_metadata)
        for cfg in all_configs:
            try:
                block_dims = PointwiseHeuristics.get_block_dimensions(cfg)
                if len(block_dims) != len(problem_dims):
                    continue
                threads = PointwiseHeuristics.prod(block_dims)
                if threads < 16 or threads > 1024:
                    continue
                if cfg.get('num_warps', 0) <= 0:
                    continue
                valid_configs.append(cfg)
            except Exception:
                continue

        if not valid_configs:
            log.warning("[HEURISTICS] No structurally valid configs")
            return None

        log.info(
            "[HEURISTICS] Generated %d valid candidate configs "
            "(scoring deferred to CachingAutotuner.__init__)",
            len(valid_configs),
        )
        return (valid_configs, problem_metadata)

    except Exception as e:
        log.warning(
            "[HEURISTICS] Failed to generate configs: %s. "
            "Falling back to default configs.",
            str(e),
        )
        return None
def pointwise(
    size_hints,
    triton_meta,
    tile_hint=None,
    filename=None,
    min_elem_per_thread=0,
    inductor_meta=None,
):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    inductor_meta = {} if inductor_meta is None else inductor_meta
    assert not inductor_meta.get("no_x_dim")
    
    # Debug: Show we're being called
    if torch.version.hip:
        msg = f"[POINTWISE] Called for problem size: {tuple(size_hints.values())}"
        print(msg, flush=True)
        # Also write to file to ensure we see it
        try:
            with open("/tmp/pointwise_heuristics_calls.log", "a") as f:
                f.write(msg + "\n")
        except:
            pass

    numel = functools.reduce(operator.mul, size_hints.values())
    bs = max(256, min(numel // 128, 1024))

    hinted_configs = autotune_hints_to_configs(
        inductor_meta.get("autotune_hints", OrderedSet()),
        size_hints,
        bs,
        triton_meta["device"],
    )

    triton_config_with_settings = functools.partial(
        triton_config, min_elem_per_thread=min_elem_per_thread
    )

    # Try to use advanced pointwise heuristics for ROCm builds
    # This runs REGARDLESS of autotune_pointwise setting to provide optimal default configs
    # Can be disabled via TORCHINDUCTOR_POINTWISE_HEURISTICS=0 for testing/comparison
    configs = None
    use_heuristics = torch.version.hip and POINTWISE_HEURISTICS_AVAILABLE and _is_pointwise_heuristics_enabled()
    
    if not _is_pointwise_heuristics_enabled() and torch.version.hip and POINTWISE_HEURISTICS_AVAILABLE:
        msg = f"[POINTWISE] Advanced heuristics DISABLED (TORCHINDUCTOR_POINTWISE_HEURISTICS=0) - using original configs"
        log.info(msg)
        print(msg, flush=True)
    
    if use_heuristics:
        log.info(
            "[HEURISTICS] ROCm detected – generating candidate configs for problem: %s",
            tuple(size_hints.values()),
        )
        # _apply_pointwise_heuristics generates and structurally validates all
        # candidate configs.  Scoring is deferred to CachingAutotuner.__init__
        # where self.fn.src and self.fn.arg_names are available.
        heuristics_result = _apply_pointwise_heuristics(
            size_hints,
            inductor_meta,
            triton_meta,
            triton_config_with_settings,
            filename,
        )

        if heuristics_result:
            all_heuristic_configs, problem_metadata = heuristics_result

            # Stash for CachingAutotuner.__init__ to pick up for scoring.
            # Only problem_metadata and size_hints are needed – scoring is done
            # directly against self.configs (Triton Config objects) in __init__
            # so there is no need to pass the raw heuristic dicts through.
            inductor_meta['_heuristics_pending'] = {
                'problem_metadata': problem_metadata,
                'size_hints': size_hints,
            }

            # In REAL_BENCH mode we want ALL configs to reach CachingAutotuner
            # so every one is compiled and timed for the validation summary.
            # Signal check_autotune_cache to skip the on-disk cache READ
            # (the cache WRITE at the end of the run is still active).
            try:
                from torch._inductor import config as inductor_config
                if inductor_config.heuristics_real_bench:
                    inductor_meta['_heuristics_skip_cache_read'] = True
            except Exception:
                pass

            # Convert ALL heuristic config dicts → Triton Config objects.
            # IMPORTANT: we create Config objects DIRECTLY rather than going through
            # triton_config() / triton_config_with_settings().  triton_config() applies
            # aggressive scaling heuristics (it scales block sizes up toward a "target"
            # product) that collapse many distinct heuristic block-size combinations
            # into identical Triton Configs.  After unique_configs() deduplication in
            # cached_autotune() this can reduce 85+ candidate configs down to 1.
            #
            # Our heuristics already chose the exact block sizes they want (powers of
            # 2, ≤ size_hints dimension, 64-1024 threads).  We just need to clamp to
            # size_hints (handles edge cases where a hint dimension is tiny) and wrap
            # in a Config object – no scaling needed.
            triton_configs = []
            ndims = len(size_hints)
            for cfg in all_heuristic_configs:
                num_warps = cfg.get('num_warps', 4)
                try:
                    if ndims == 1:
                        xblock = min(cfg.get('XBLOCK', 256), size_hints['x'])
                        if size_hints['x'] == 1:
                            xblock = 1
                        kwargs = {'XBLOCK': xblock}
                    elif ndims == 2:
                        xblock = min(cfg.get('XBLOCK', 32), size_hints['x'])
                        yblock = min(cfg.get('YBLOCK', 32), size_hints['y'])
                        if size_hints['x'] == 1:
                            xblock = 1
                        if size_hints['y'] == 1:
                            yblock = 1
                        kwargs = {'XBLOCK': xblock, 'YBLOCK': yblock}
                    elif ndims == 3:
                        xblock = min(cfg.get('XBLOCK', 16), size_hints['x'])
                        yblock = min(cfg.get('YBLOCK', 16), size_hints['y'])
                        zblock = min(cfg.get('ZBLOCK', 16), size_hints['z'])
                        if size_hints['x'] == 1:
                            xblock = 1
                        if size_hints['y'] == 1:
                            yblock = 1
                        if size_hints['z'] == 1:
                            zblock = 1
                        kwargs = {'XBLOCK': xblock, 'YBLOCK': yblock, 'ZBLOCK': zblock}
                    else:
                        continue
                    triton_configs.append(Config(kwargs, num_warps=num_warps, num_stages=1))
                except Exception:
                    continue

            configs = triton_configs
            log.info(
                "[HEURISTICS] Passing all %d candidate configs to CachingAutotuner "
                "(will be pruned to top-5 in __init__ after scoring with fn.src)",
                len(configs),
            )
            print(
                f"[HEURISTICS] {len(configs)} candidate configs passed to autotuner "
                f"(scoring + pruning deferred to __init__)",
                flush=True,
            )
        else:
            log.info("[HEURISTICS] Heuristics failed – falling back to defaults")
            print("[HEURISTICS] Heuristics failed – falling back to defaults", flush=True)
    
    # Fall back to default config generation if heuristics not used or failed
    if configs is None and len(size_hints) == 1:
        if not inductor_meta.get("autotune_pointwise", False) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            configs = [triton_config_with_settings(size_hints, bs)]
        else:
            configs = [
                triton_config_with_settings(size_hints, bs, num_elements_per_warp=256),
                triton_config_with_settings(
                    size_hints, bs // 2, num_elements_per_warp=64
                ),
                *hinted_configs,
            ]
            # Additional configs appended for ROCm builds
            if torch.version.hip:
                configs.extend(
                    [
                        triton_config_with_settings(
                            size_hints, TRITON_MAX_BLOCK["X"], waves_per_eu=2
                        ),
                        triton_config_with_settings(
                            size_hints,
                            4096,  # wrt: better than the max_block for some kernel
                        ),
                        triton_config_with_settings(
                            size_hints,
                            2048,
                            num_warps=8,
                            num_stages=2,
                            waves_per_eu=1,  # 20% improvement
                        ),
                    ]
                )
                if inductor_meta.get("atomic_add_found"):
                    configs.extend(
                        [
                            triton_config_with_settings(
                                size_hints,
                                64,
                                num_warps=1,
                                num_stages=1,  # 250% improvement
                            )
                        ]
                    )
    if configs is None and len(size_hints) == 2:
        # Only avoiding tuning on TileHint.SQUARE if not on ROCm builds
        # ROCm has observed improvement by diverging here
        if (
            not inductor_meta.get("autotune_pointwise", False)
            or (torch.version.hip is None and tile_hint == TileHint.SQUARE)
        ) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            configs = [triton_config_with_settings(size_hints, 32, 32)]
        else:
            configs = [
                triton_config_with_settings(size_hints, 32, 32),
                triton_config_with_settings(size_hints, 64, 64),  # ~8% better for fp16
                triton_config_with_settings(size_hints, 256, 16),
                triton_config_with_settings(size_hints, 16, 256),
                triton_config_with_settings(size_hints, bs, 1),
                triton_config_with_settings(size_hints, 1, bs),
                *hinted_configs,
            ]
            # Additional configs appended for ROCm builds
            if torch.version.hip:
                configs.extend(
                    [
                        triton_config_with_settings(
                            size_hints, 64, 32
                        ),  # better for some kernels
                        triton_config_with_settings(
                            size_hints, 128, 16
                        ),  # +10% for some kernels
                        triton_config_with_settings(
                            size_hints, 128, 32
                        ),  # additional 10% more
                        triton_config_with_settings(
                            size_hints, 32, 512
                        ),  # +30% for some kernels
                    ]
                )
    if configs is None and len(size_hints) == 3:
        if not inductor_meta.get("autotune_pointwise", False):
            configs = [triton_config_with_settings(size_hints, 16, 16, 16)]
        else:
            configs = [
                triton_config_with_settings(size_hints, 16, 16, 16),
                triton_config_with_settings(size_hints, 64, 8, 8),
                triton_config_with_settings(size_hints, 8, 64, 8),
                triton_config_with_settings(size_hints, 8, 8, 64),
                triton_config_with_settings(size_hints, bs, 1, 1),
                triton_config_with_settings(size_hints, 1, bs, 1),
                triton_config_with_settings(size_hints, 1, 1, bs),
                *hinted_configs,
            ]

    if not configs:
        raise NotImplementedError(f"size_hints: {size_hints}")

    configs = _maybe_filter_configs_for_tma_restrictions(inductor_meta, configs)

    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.POINTWISE,
        filename=filename,
    )


def make_matmul_triton_config(sizes: dict[str, int], num_warps: int, num_stages: int):
    config = {
        "XBLOCK": sizes.get("x"),
        "YBLOCK": sizes.get("y"),
        "ZBLOCK": sizes.get("z"),
        "R0_BLOCK": sizes.get("r"),
    }
    # Remove keys with None values (i.e., missing in sizes)
    config = {k: v for k, v in config.items() if v is not None}
    return Config(config, num_warps=num_warps, num_stages=num_stages)


def _config_helper(bmm=False, persistent=False):
    # Each entry is: (sizes_dict, num_warps, num_stages)
    _base_mm_configs = [
        ({"x": 32, "y": 32, "r": 16}, 2, 1),
        ({"x": 32, "y": 32, "r": 128}, 4, 2),
        ({"x": 32, "y": 64, "r": 32}, 8, 5),
        ({"x": 64, "y": 32, "r": 32}, 8, 5),
        ({"x": 64, "y": 32, "r": 128}, 4, 5),
        ({"x": 64, "y": 64, "r": 16}, 4, 2),
        ({"x": 64, "y": 64, "r": 32}, 4, 2),
        ({"x": 64, "y": 64, "r": 64}, 8, 3),
        ({"x": 64, "y": 64, "r": 128}, 4, 5),
        ({"x": 64, "y": 128, "r": 32}, 4, 3),
        ({"x": 64, "y": 128, "r": 32}, 8, 4),
        ({"x": 64, "y": 128, "r": 64}, 4, 3),
        ({"x": 64, "y": 128, "r": 128}, 4, 4),
        ({"x": 128, "y": 64, "r": 32}, 4, 3),
        ({"x": 128, "y": 64, "r": 32}, 8, 4),
        ({"x": 128, "y": 128, "r": 32}, 8, 2),
        ({"x": 128, "y": 128, "r": 32}, 4, 3),
        ({"x": 128, "y": 128, "r": 64}, 4, 3),
        ({"x": 128, "y": 128, "r": 64}, 8, 5),
    ]
    out = []
    for sizes, w, s in _base_mm_configs:
        d = dict(sizes)
        if persistent:
            d.pop("r", None)
        if bmm:
            d["z"] = 1
        out.append((d, w, s))

    # Deduplicate by converting dicts to immutable frozensets
    deduped = {(frozenset(d.items()), w, s): (d, w, s) for d, w, s in out}

    return list(deduped.values())


triton_native_mm_configs = _config_helper(bmm=False, persistent=False)
triton_native_persistent_mm_configs = _config_helper(bmm=False, persistent=True)
triton_native_bmm_configs = _config_helper(bmm=True, persistent=False)
triton_native_persistent_bmm_configs = _config_helper(bmm=True, persistent=True)


def _reduction_configs(
    *,
    size_hints: dict[str, int],
    inductor_meta: dict[str, Any],
    triton_meta: dict[str, Any],
    num_dynamic=0,
) -> list[Config]:
    reduction_hint = inductor_meta.get("reduction_hint")

    # Convert reductions to 1D, to simplify heuristics.
    rnumel = get_total_reduction_numel(size_hints)

    # Is max autotune enabled
    max_autotune_enabled = inductor_meta.get("max_autotune") or inductor_meta.get(
        "max_autotune_pointwise"
    )

    register_intensive = False
    MAX_R0_BLOCK = 2048
    loads_and_red = inductor_meta.get("num_load", 0) + inductor_meta.get(
        "num_reduction", 0
    )
    if size_hints["x"] >= 1024 and loads_and_red >= 10:
        # A heuristics to reduce R0_BLOCK if a kernel potentially need many registers.
        # Consider load and reduction since load need move data into registers and
        # reduction needs an accumulator.
        #
        # The magic numbers are a bit arbitrary.
        #
        # We cannot rely on dynamically scaling down R0_BLOCK later, since sometimes
        # triton makes it to use less registers with worse perf. Check:
        # https://github.com/pytorch/pytorch/issues/126463
        #
        # The heuristic is a very simple one since registers can be reused. But
        # hopefully it can be a good enough indicator.
        MAX_R0_BLOCK = 1024
        register_intensive = True

    if triton_meta.get("native_matmul"):
        if len(size_hints) == 3:
            return [
                make_matmul_triton_config(sizes, num_warps, num_stages)
                for sizes, num_warps, num_stages in triton_native_mm_configs
            ]
        elif len(size_hints) == 4:
            return [
                make_matmul_triton_config(sizes, num_warps, num_stages)
                for sizes, num_warps, num_stages in triton_native_bmm_configs
            ]
        else:
            raise NotImplementedError("native matmul only supports mm/bmm pattern")

    def make_config(
        x,
        r,
        num_warps=None,
        num_stages=1,
        register_intensive=False,
        dynamic_scale_rblock=True,
        waves_per_eu=None,
    ):
        # For 3D case with tiling scores, create an adapted version
        if "y" in size_hints:
            assert "tiling_scores" in inductor_meta
            return adapt_config_for_tiling(
                size_hints,
                inductor_meta["tiling_scores"],
                x,
                r,
                num_warps=num_warps,
                num_stages=num_stages,
                register_intensive=register_intensive,
                waves_per_eu=waves_per_eu,
            )
        else:
            # For other cases, use the original function
            return triton_config_reduction(
                size_hints,
                x,
                r,
                num_warps=num_warps,
                num_stages=num_stages,
                register_intensive=register_intensive,
                waves_per_eu=waves_per_eu,
                dynamic_scale_rblock=dynamic_scale_rblock,
                reduction_hint=reduction_hint,
            )

    def outer_config_opt():
        # Default to 64 for vectorized loads
        max_x_block, x_block = 256, 64
        load_factor = inductor_meta.get("num_load", 0)
        x = size_hints["x"]
        num_warps = None

        # Try to use all SMs with small x
        if x <= 1024:
            x_block = max(min(x // 128, 8), 2)
            outer_r_block = min(rnumel, 64)
        # Lower bound x = 1024, 1024 // 16 = 128 around # of SMs
        elif x // 4096 <= 8:
            x_block = 16
            outer_r_block = 512 // x_block
        elif num_dynamic > 1:
            # Lots of compute with multiple dynamic shape per loop iteration
            # Larger RBLOCK minimizes loop iteration
            outer_r_block = max(min((rnumel // 64), 64), 8)
        elif num_dynamic == 1:
            # Dynamic shapes introduce a lot register pressure for indexing
            outer_r_block = (
                1
                if load_factor >= 3
                else min(next_power_of_2(max(rnumel, 128) // 128), 8)
            )
        else:
            x_block = max(min(max_x_block, next_power_of_2(x // 4096)), x_block)
            if load_factor < 4 or rnumel <= 128:
                outer_r_block = 512 // x_block
            else:
                # Heavier reductions contain a lot more overhead per loop iteration
                # We minimize the overhead by enlarging r block
                if rnumel >= 2048:
                    outer_r_block = 64
                else:
                    outer_r_block = 32
                x_block = min(x_block, 32)
                num_warps = 4

        # Set register intensive to true by default as we try to maximize tiles with heuristic
        return make_config(
            x_block,
            outer_r_block,
            num_warps=num_warps,
            register_intensive=register_intensive,
        )

    contiguous_config = make_config(
        2 if rnumel <= 2048 else 1,  # 1024 or less is persistent
        min(rnumel, MAX_R0_BLOCK),
        register_intensive=register_intensive,
    )
    tiny_config = make_config(
        2 * (256 // rnumel) if rnumel <= 256 else 1,
        min(rnumel, MAX_R0_BLOCK),
        register_intensive=register_intensive,
    )

    outer_config = make_config(64, 8, register_intensive=register_intensive)
    # TODO (paulzhan): Test heuristic on AMD and internal testing
    # for correctness
    if not torch.version.hip:
        outer_config = outer_config_opt()

    configs = []

    if inductor_meta.get("add_persistent_rblock") and loads_and_red <= 8:
        xnumel = max(4096 // rnumel, 1)
        c = make_config(
            xnumel,
            min(rnumel, 32768),
            register_intensive=register_intensive,
            dynamic_scale_rblock=False,
        )
        configs.append(c)

    result_configs = []

    # For 3d tiling, default to more autotuning only if max_autotune is enabled
    if "y" in size_hints and max_autotune_enabled:
        pass
    elif max_autotune_enabled:
        pass  # skip all these cases
    elif reduction_hint == ReductionHint.INNER:
        return configs + [contiguous_config]
    elif reduction_hint == ReductionHint.OUTER:
        return configs + [outer_config]
    elif reduction_hint == ReductionHint.OUTER_TINY:
        return configs + [tiny_config]
    else:
        # Default case: return a single config when autotuning is not enabled
        return configs + [contiguous_config]

    # We continue here under the following conditions:
    # - max_autotune_enabled is True
    result_configs = configs + [
        contiguous_config,
        outer_config,
        tiny_config,
        make_config(64, 64),
        make_config(8, 512),
        # halve the XBLOCK/Rn_BLOCK compared to outer_config
        # TODO: this may only be beneficial when each iteration of the reduction
        # is quite heavy. E.g. https://gist.github.com/shunting314/189a8ef69f90db9d614a823385147a72
        make_config(64, 4, num_warps=8),
    ]

    if torch.version.hip:
        result_configs.extend(
            [
                make_config(1024, 8, num_warps=4, num_stages=1, waves_per_eu=2),
                make_config(512, 8, num_warps=4, num_stages=1, waves_per_eu=1),
            ]
        )

    return result_configs


def match_target_block_product(
    size_hints, tiling_scores, target_block_product, min_block_size=1
):
    """
    Distribute block sizes across dimensions according to tiling scores,
    aiming to match a target product of block sizes.
    """
    total_score = sum(tiling_scores.values())
    if total_score == 0:
        # just assume even score with no minimum block size
        min_block_size = 1
        tiling_scores = dict.fromkeys(tiling_scores.keys(), target_block_product)

    # First, give each coalescing dimension at least min_block_size
    block_sizes = {}
    relative_scores = {}
    curr_block_product = 1

    for dim, score in tiling_scores.items():
        if score == 0:
            block_sizes[dim] = 1
            continue

        block_sizes[dim] = min_block_size
        curr_block_product *= min_block_size
        relative_scores[dim] = score / total_score

    # Scale up dimensions by their relative scores until we reach the target
    while curr_block_product < target_block_product and relative_scores:
        dim, score = max(relative_scores.items(), key=lambda item: item[1])

        # Check if we've hit the max for this dimension
        if (
            block_sizes[dim] >= TRITON_MAX_BLOCK[dim.capitalize()]
            or block_sizes[dim] >= size_hints[dim]
        ):
            del relative_scores[dim]
            continue

        block_sizes[dim] *= 2
        relative_scores[dim] /= 2
        curr_block_product *= 2

    return block_sizes


def adapt_config_for_tiling(
    size_hints,
    tiling_scores,
    original_x,
    original_r,
    num_warps=None,
    num_stages=1,
    register_intensive=False,
    persistent_reduction=False,
    waves_per_eu=None,
) -> Config:
    """
    Create an adapted configuration based on tiling scores,
    redistributing the same total block size (x * r) according to tiling scores.
    """
    assert all(s in tiling_scores for s in size_hints)
    target_block_product = original_x * original_r
    block_sizes = match_target_block_product(
        size_hints, tiling_scores, target_block_product
    )

    return triton_config_tiled_reduction(
        size_hints,
        block_sizes["x"],
        block_sizes["y"],
        block_sizes["r0_"],
        num_stages=num_stages,
        register_intensive=register_intensive,
        waves_per_eu=waves_per_eu,
    )


def filter_reduction_configs_for_determinism(
    inductor_meta: dict[str, Any], configs: list[Config]
) -> list[Config]:
    """
    Filter configs for reduction so the numerics can be deterministic.

    Heuristics:
    - skip reduction configs with too small RBLOCK
    - skip reduction configs with XBLOCK==1 if we are confident it will not perform well
    - if there is a tie, pick the config with second largest RBLOCK
    - if there is still a tie, pick the config with second largest num_warps
    - if there is still a tie, pick the config with second largest XBLOCK
    """
    configs = unique_configs(configs)
    assert len(configs) > 0

    def _do_filter_due_to_inductor_config():
        return (
            inductor_meta.get("deterministic", False)
            or inductor_meta.get("force_filter_reduction_configs", False)
        ) or inductor_meta.get("are_deterministic_algorithms_enabled")

    if not _do_filter_due_to_inductor_config() or len(configs) == 1:
        # no filtering happening if NOT in deterministic mode
        return configs

    if log.isEnabledFor(logging.DEBUG):
        log.debug("reduction configs before filtering:")
        for c in configs:
            log.debug("%s", c)
            log.debug("")

    def _has_too_small_rblock(config):
        rblock = config.kwargs.get("R0_BLOCK")
        # too small RBLOCK is likely to be bad
        return rblock is not None and rblock <= 4

    def _nonpromising_xblock_1(config):
        # kernel like https://gist.github.com/shunting314/0b3281c087e79bc915fe45985ff9d7d5
        # without a load/store having contiguous rdim is unlikely to perform well with XBLOCK==1
        return config.kwargs["XBLOCK"] == 1 and not inductor_meta.get(
            "has_loadstore_with_contiguous_rdim", True
        )

    newconfigs = [*filter(lambda x: not _has_too_small_rblock(x), configs)]
    # accept the filtering only if there are configs left
    if len(newconfigs) > 0:
        configs = newconfigs

    newconfigs = [*filter(lambda x: not _nonpromising_xblock_1(x), configs)]
    if len(newconfigs) > 0:
        configs = newconfigs

    assert len(configs) > 0

    def _r0_block(c):
        return c.kwargs.get("R0_BLOCK", -1)

    def _xblock(c):
        return c.kwargs.get("XBLOCK", -1)

    def _num_warps(c):
        return c.num_warps

    def _pick_second_largest(accessor):
        nonlocal configs
        configs = sorted(configs, key=lambda x: accessor(x))
        if accessor(configs[0]) != accessor(configs[-1]):
            max_val = accessor(configs[-1])
            configs = [*filter(lambda x: accessor(x) != max_val, configs)]
            second_max_val = accessor(configs[-1])
            configs = [*filter(lambda x: accessor(x) == second_max_val, configs)]
        return configs

    def _pick_config():
        nonlocal configs
        assert len(configs) > 0
        if len(configs) == 1:
            return configs[0]

        # break tie by R0_BLOCK
        configs = _pick_second_largest(_r0_block)
        if len(configs) == 1:
            return configs[0]

        # break tie by num_warps
        configs = _pick_second_largest(_num_warps)
        if len(configs) == 1:
            return configs[0]

        # break tie by XBLOCK
        configs = _pick_second_largest(_xblock)

        # there is still a tie, pick the first one
        return configs[0]

    configs = [_pick_config()]

    if log.isEnabledFor(logging.DEBUG):
        log.debug("reduction configs after filtering:")
        for c in configs:
            log.debug("%s", c)
            log.debug("")
    return configs


def reduction(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    """args to @triton.heuristics()"""
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    assert triton_meta is not None

    num_dynamic = 0
    for k in triton_meta["signature"]:
        if "ks" in k:
            num_dynamic += 1

    configs = _reduction_configs(
        size_hints=size_hints,
        inductor_meta=inductor_meta,
        triton_meta=triton_meta,
        num_dynamic=num_dynamic,
    )

    configs = _maybe_filter_configs_for_tma_restrictions(inductor_meta, configs)
    configs = filter_reduction_configs_for_determinism(inductor_meta, configs)

    return cached_autotune(
        size_hints,
        configs=configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.REDUCTION,
        filename=filename,
    )


def cooperative_reduction(
    size_hints,
    reduction_hint,
    triton_meta,
    filename,
    inductor_meta,
):
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    # Cooperative reductions currently only support a single reduction dimension.
    assert len(size_hints) == 2, (
        "Cooperative reductions don't support tiling reduction dims"
    )
    xnumel, rnumel = size_hints["x"], size_hints["r0_"]

    # TODO(jansel): we should base target on the SM count of the local GPU
    target = 64
    split = max(1, min(target // xnumel, TRITON_MAX_RSPLIT))
    assert rnumel >= split
    assert split <= TRITON_MAX_RSPLIT
    if inductor_meta["persistent_reduction"]:
        configs = _persistent_reduction_configs(
            {"x": xnumel, "r0_": rnumel // split},
            reduction_hint,
            inductor_meta,
            triton_meta,
        )
    else:
        configs = _reduction_configs(
            size_hints={"x": xnumel, "r0_": rnumel // split},
            inductor_meta=inductor_meta,
            triton_meta=triton_meta,
        )
    for config in configs:
        config.kwargs["RSPLIT"] = split
    # TODO(jansel): add more configs in max_autotune

    configs = _maybe_filter_configs_for_tma_restrictions(inductor_meta, configs)
    configs = filter_reduction_configs_for_determinism(inductor_meta, configs)
    return cached_autotune(
        size_hints,
        configs=configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.REDUCTION,
        filename=filename,
    )


def _persistent_reduction_configs(
    size_hints,
    reduction_hint=False,
    inductor_meta=None,
    triton_meta=None,
):
    xnumel = size_hints["x"]
    rnumel = get_total_reduction_numel(size_hints)

    MAX_PERSISTENT_BLOCK_NUMEL = 4096

    if triton_meta.get("native_matmul"):
        if len(size_hints) == 3:
            return [
                make_matmul_triton_config(sizes, num_warps, num_stages)
                for sizes, num_warps, num_stages in triton_native_persistent_mm_configs
            ]
        elif len(size_hints) == 4:
            return [
                make_matmul_triton_config(sizes, num_warps, num_stages)
                for sizes, num_warps, num_stages in triton_native_persistent_bmm_configs
            ]
        else:
            raise NotImplementedError("native matmul only supports mm/bmm pattern")

    max_autotune_enabled = inductor_meta.get("max_autotune") or inductor_meta.get(
        "max_autotune_pointwise"
    )

    if torch.version.hip:
        xblock_vals = [1, 4, 8, 16, 32, 64, 128, 256]
    else:
        xblock_vals = [1, 8, 32, 128]

    if "y" not in size_hints:
        configs = [
            triton_config_reduction(
                size_hints,
                xblock,
                rnumel,
                register_intensive=True,
                reduction_hint=reduction_hint,
            )
            for xblock in xblock_vals
            if xblock == 1
            or (rnumel * xblock <= MAX_PERSISTENT_BLOCK_NUMEL and xblock <= xnumel)
        ]
    else:
        configs = []
        assert "tiling_scores" in inductor_meta
        x_y_scores = {dim: inductor_meta["tiling_scores"][dim] for dim in ("x", "y")}
        for target_block_size in xblock_vals:
            if target_block_size * rnumel > MAX_PERSISTENT_BLOCK_NUMEL:
                continue

            block_sizes = match_target_block_product(
                size_hints, x_y_scores, target_block_size
            )
            configs.append(
                triton_config_tiled_reduction(
                    size_hints, block_sizes["x"], block_sizes["y"], rnumel
                )
            )

    tiny_configs = [
        triton_config_reduction(
            size_hints,
            2 * (256 // rnumel) if rnumel <= 256 else 1,
            rnumel,
        )
    ]

    # defer to more autotuning only when max_autotune is enabled
    if "y" in size_hints and max_autotune_enabled:
        pass
    # TODO(jansel): we should be able to improve these heuristics
    elif not max_autotune_enabled:  # Do not filter configs when tuning
        if reduction_hint == ReductionHint.INNER and rnumel >= 256:
            if rnumel > 1024 or xnumel // 8 < 128 or inductor_meta.get("RSPLIT_SIZE"):
                configs = configs[:1]
            else:
                if not torch.cuda.is_available():
                    # TODO(Intel): CUDA uses num_warps = 1 to disable shared memory.
                    # We apply different configurations from #168335.
                    # We currently let cost model in Triton to decide whether to use shared memory.
                    loads_and_stores = inductor_meta.get(
                        "num_load", 0
                    ) + inductor_meta.get("num_store", 0)
                    x_block = 8
                    if xnumel // x_block < 128 or loads_and_stores >= 5:
                        x_block = 1
                    num_warps, min_num_warps, reduction_hint = None, None, None
                else:
                    x_block = min(1024 // rnumel, 8)
                    num_warps, min_num_warps = 1, 1
                configs = [
                    triton_config_reduction(
                        size_hints,
                        x_block,
                        rnumel,
                        register_intensive=True,
                        num_warps=num_warps,
                        min_num_warps=min_num_warps,
                        reduction_hint=reduction_hint,
                    )
                ]

        elif reduction_hint == ReductionHint.OUTER:
            configs = configs[-1:]
        elif reduction_hint == ReductionHint.OUTER_TINY:
            configs = tiny_configs
        else:
            # Default case: return first config when autotuning is not enabled
            configs = configs[:1] if configs else tiny_configs
    else:
        if torch.version.hip:
            # If autotune is enabled append tiny configs
            for conf in tiny_configs:
                if conf not in configs:
                    configs.append(conf)

            # Additional custom configs in support of customer workloads
            configs.append(
                triton_config_reduction(
                    size_hints,
                    1,
                    rnumel,
                    num_stages=3,
                    num_warps=2,
                )  # 18% improvement
            )

    for c in configs:
        # we don't need Rn_BLOCK for persistent reduction
        for prefix in size_hints:
            if prefix_is_reduction(prefix):
                c.kwargs.pop(f"{prefix.upper()}BLOCK")

    return configs


def persistent_reduction(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    configs = _persistent_reduction_configs(
        size_hints, reduction_hint, inductor_meta, triton_meta
    )

    # This key is not added to the inductor meta as its clear from the heuristic
    # choice that it is persistent. Add it and remove it below so that persistent
    # configs can be filtered appropriately by _maybe_filter_configs_for_tma_restrictions
    persistent_reduction_key = "persistent_reduction"
    inductor_meta[persistent_reduction_key] = True
    configs = _maybe_filter_configs_for_tma_restrictions(inductor_meta, configs)
    inductor_meta.pop(persistent_reduction_key)

    if inductor_meta.get("RSPLIT_SIZE"):
        new_configs = []
        rsplit_size = inductor_meta.get("RSPLIT_SIZE")
        rnumel_hint = size_hints["r0_"]
        min_x_block = 1
        if rnumel_hint <= 512:
            min_x_block = 4
        x_block = min(max(rsplit_size // 32, min_x_block), 16)
        for c in configs:
            c.kwargs["RSPLIT_SIZE"] = rsplit_size
            # small XBLOCK to use less registers/smem
            c.kwargs["XBLOCK"] = x_block

            num_iters = rsplit_size // x_block
            c.kwargs["NUM_STAGES"] = min(max(num_iters // 4, 1), 3)

            if rnumel_hint <= 1024:
                c.num_warps //= 2
                c.num_warps = max(c.num_warps, 1)
                new_configs.append(c)

                # less warps so potentially each sm can run more thread blocks
                # Inside each thread block, we handle the split sequentially,
                # more thread blocks is beneficial here.
                newc = copy.deepcopy(c)
                newc.num_warps = 2
                new_configs.append(newc)
            else:
                # more warps for larger rows
                new_configs.append(c)

                if c.num_warps < 32:
                    newc = copy.deepcopy(c)
                    newc.num_warps *= 2
                    new_configs.append(newc)

        configs = unique_configs(new_configs)

    configs = filter_reduction_configs_for_determinism(inductor_meta, configs)
    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        filename=filename,
        heuristic_type=HeuristicType.PERSISTENT_REDUCTION,
    )


def split_scan(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    """Heuristic for TritonSplitScanKernel"""
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    assert triton_meta is not None
    if len(size_hints) != 2:
        raise NotImplementedError(f"size_hints: {size_hints}")

    configs = _reduction_configs(
        size_hints=size_hints, inductor_meta=inductor_meta, triton_meta=triton_meta
    )

    # Fixup configs to enforce the minimum Rn_BLOCK size
    min_rblock = inductor_meta.get("min_split_scan_rblock", 256)
    for cfg in configs:
        for var in list(cfg.kwargs.keys()):
            if var.startswith("R") and cfg.kwargs[var] < min_rblock:
                cfg.kwargs[var] = min_rblock

    configs = _maybe_filter_configs_for_tma_restrictions(inductor_meta, configs)
    configs = filter_reduction_configs_for_determinism(inductor_meta, configs)
    return cached_autotune(
        size_hints,
        configs=configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.SPLIT_SCAN,
        filename=filename,
    )


def template(
    num_stages,
    num_warps,
    triton_meta,
    num_consumer_groups=0,
    num_buffers_warp_spec=0,
    filename=None,
    inductor_meta=None,
):
    """
    Compile a triton template
    """
    # Prepare the base configuration
    config_args = {
        "num_stages": num_stages,
        "num_warps": num_warps,
    }

    # Conditionally add arguments based on HAS_WARP_SPEC
    if HAS_WARP_SPEC:
        config_args.update(
            {
                "num_consumer_groups": num_consumer_groups,
                "num_buffers_warp_spec": num_buffers_warp_spec,
            }
        )
    return cached_autotune(
        None,
        [triton.Config({}, **config_args)],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )


def _pop_config_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Extract triton.Config options that should become kwargs"""
    popped = {}
    for key in (
        "num_warps",
        "num_stages",
        "num_ctas",
        "maxnreg",
        "num_consumer_groups",
        "num_buffers_warp_spec",
    ):
        val = config.pop(key, None)
        if val is not None:
            popped[key] = val
    return popped


def config_to_dict(config: Config) -> dict[str, Any]:
    config_dict = {
        **config.kwargs,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }
    if HAS_WARP_SPEC:
        config_dict.update(
            {
                "num_consumer_groups": getattr(config, "num_consumer_groups", 0),
                "num_buffers_warp_spec": getattr(config, "num_buffers_warp_spec", 0),
            }
        )
    return config_dict


def config_from_dict(config: dict[str, Any]) -> Config:
    config = {**config}
    return Config(config, **_pop_config_kwargs(config))


def fixed_config(config, filename, triton_meta, inductor_meta):
    """
    Used when the configuration is already decided at compile time
    """
    config = {**config}
    return cached_autotune(
        None,
        [triton.Config(config, **_pop_config_kwargs(config))],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.FIXED,
        filename=filename,
    )


def user_autotune(
    configs, triton_meta, filename=None, inductor_meta=None, custom_kernel=False
):
    """
    Compile a user defined triton kernel
    """
    if len(configs) == 0:
        configs = [triton.Config({})]
    else:
        configs = [*map(config_from_dict, configs)]
    return cached_autotune(
        None,
        configs,
        triton_meta=triton_meta,
        heuristic_type=HeuristicType.USER_AUTOTUNE,
        filename=filename,
        inductor_meta=inductor_meta,
        custom_kernel=custom_kernel,
    )


def foreach(triton_meta, filename=None, inductor_meta=None):
    """
    Compile a triton foreach kernel
    """
    configs = []

    # Naive autotuning path for num_warps
    if not (
        inductor_meta.get("max_autotune") or inductor_meta.get("max_autotune_pointwise")
    ):
        configs.append(triton.Config({}, num_stages=1, num_warps=8))
    else:
        for warps in [1, 2, 4, 8]:
            configs.append(triton.Config({}, num_stages=1, num_warps=warps))

    return cached_autotune(
        None,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )


@dataclasses.dataclass
class GridExpr:
    """Generate code for grid size expressions in launcher"""

    inductor_meta: dict[str, Any]
    mode: Literal["python", "cpp"] = "python"
    prefix: list[str] = dataclasses.field(default_factory=list)
    x_grid: str | int = 1
    y_grid: str | int = 1
    z_grid: str | int = 1

    def __post_init__(self) -> None:
        assert self.mode in ("python", "cpp")

    def generate(self, meta: dict[str, int]) -> None:
        raise NotImplementedError

    def ceildiv(self, numel: str | int, block: None | int | str) -> str | int:
        if block is None or block == 1:
            return numel
        if isinstance(numel, int) and isinstance(block, int):
            return ceildiv(numel, block)  # constant fold
        # This trick only works in python, where
        # negative integer division is floored
        if self.mode == "python":
            return f"-(({numel}) // -({block}))"
        # For cpp code gen
        return f"(({numel} + ({block} - 1)) / ({block}))"

    def maximum(self, seq: list[int | str]) -> int | str:
        """Codegen for max function with constant folding, constants are represented as int"""
        items = self._constant_fold(max, seq)
        if len(items) <= 1:
            return items[0]
        if self.mode == "python":
            return f"max({', '.join(map(str, items))})"
        return functools.reduce(lambda x, y: f"std::max({x}, {y})", items)

    def summation(self, seq: list[int | str]) -> int | str:
        """Codegen for sum function with constant folding, constants are represented as int"""
        items = self._constant_fold(sum, seq)
        if len(items) <= 1:
            return items[0]
        return " + ".join(map(str, items))

    def _constant_fold(
        self, fn: Callable[[list[int]], int], seq: list[int | str]
    ) -> list[int | str]:
        """Constant fold through a commutative fn where ints are constants"""
        items: list[int | str] = [x for x in seq if not isinstance(x, int)]
        const_items = [x for x in seq if isinstance(x, int)]
        if const_items:
            items.append(fn(const_items))
        return items

    def assign_tmp(self, name: str, expr: str | int) -> str:
        # Grid functions are one per kernel, so name collisions are fine
        if self.mode == "python":
            return f"{name} = {expr}"
        if self.mode == "cpp":
            return f"uint32_t {name} = {expr};"
        raise AssertionError(f"invalid mode {self.mode}")

    @staticmethod
    def from_meta(
        inductor_meta: dict[str, Any],
        cfg: Config | dict[str, int],
        mode: Literal["python", "cpp"] = "python",
    ) -> GridExpr:
        grid_cls = globals()[inductor_meta["grid_type"]]
        assert issubclass(grid_cls, GridExpr)
        grid = grid_cls(inductor_meta=inductor_meta, mode=mode)
        if isinstance(cfg, Config):
            cfg = config_to_dict(cfg)
        grid.generate(cfg)
        return grid

    def eval_slow(self, meta: dict[str, int]) -> tuple[int, int, int]:
        scope = {**meta}
        for line in self.prefix:
            exec(line, scope)
        exec(f"grid_0 = {self.x_grid}", scope)
        exec(f"grid_1 = {self.y_grid}", scope)
        exec(f"grid_2 = {self.z_grid}", scope)
        return scope["grid_0"], scope["grid_1"], scope["grid_2"]


class Grid1D(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))


class Grid2D(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
        self.y_grid = self.ceildiv("ynumel", meta.get("YBLOCK"))


class Grid3D(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
        self.y_grid = self.ceildiv("ynumel", meta.get("YBLOCK"))
        self.z_grid = self.ceildiv("znumel", meta.get("ZBLOCK"))


class Grid2DWithYZOverflow(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
        self.prefix.extend(
            [
                self.assign_tmp(
                    "y_grid_raw_", self.ceildiv("ynumel", meta.get("YBLOCK"))
                ),
                self.assign_tmp(
                    "y_grid_div_", self.ceildiv("y_grid_raw_", get_max_y_grid())
                ),
            ]
        )
        self.y_grid = self.ceildiv("y_grid_raw_", "y_grid_div_")
        self.z_grid = "y_grid_div_"


class MixOrderReductionGrid(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        split_size = meta.get("RSPLIT_SIZE")
        xblock = meta.get("XBLOCK")
        assert split_size, "Missing RSPLIT_SIZE"
        assert xblock, "Missing XBLOCK"
        assert split_size % xblock == 0, f"{split_size=}, {xblock=}"
        self.x_grid = self.ceildiv("xnumel", split_size)


class CooperativeReductionGrid(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        self.x_grid = str(meta["RSPLIT"])
        self.y_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))


class SplitScanGrid(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        assert meta.get("XBLOCK", 1) == 1
        self.x_grid = self.ceildiv("r0_numel", meta.get("R0_BLOCK"))
        self.y_grid = "xnumel"


class FixedGrid(GridExpr):
    @staticmethod
    def setup_grid_as_args() -> dict[str, Any]:
        """Inductor meta so the launcher takes three extra grid arguments"""
        return {
            "grid_type": FixedGrid.__name__,
            "fixed_grid": ["_grid_0", "_grid_1", "_grid_2"],
            "extra_launcher_args": ["_grid_0", "_grid_1", "_grid_2"],
        }

    def generate(self, meta: dict[str, int]) -> None:
        self.x_grid, self.y_grid, self.z_grid = self.inductor_meta["fixed_grid"]


class PrecomputedGrid(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        for candidate in self.inductor_meta["precomputed_grids"]:
            if all(meta.get(k) == v for k, v in candidate["config"].items()):
                self.x_grid, self.y_grid, self.z_grid = candidate[self.mode]
                return
        raise AssertionError(
            f"Precomputed grid not found for {meta} in {self.inductor_meta['precomputed_grids']}"
        )


class ComboKernelGrid(GridExpr):
    def generate(self, meta: dict[str, int]):
        combo_meta = self.inductor_meta["combo_grid_meta"]
        if combo_meta["default_config"]:
            meta = {**combo_meta["default_config"], **meta}
        no_x_dims = []
        xnumels = []
        ynumels = []
        znumels = []
        for num in range(combo_meta["num_kernels"]):
            assert (
                combo_meta[f"xnumel_{num}"] is None or combo_meta[f"xnumel_{num}"] > 0
            )
            no_x_dims.append(combo_meta[f"no_x_dim_{num}"])
            xnumels.append(combo_meta[f"xnumel_{num}"] or f"xnumel_{num}")
            if f"ynumel_{num}" in combo_meta:
                ynumels.append(combo_meta[f"ynumel_{num}"] or f"ynumel_{num}")
            if f"znumel_{num}" in combo_meta:
                znumels.append(combo_meta[f"znumel_{num}"] or f"znumel_{num}")

        self.x_grid = self.combo_x_grid(xnumels, no_x_dims, meta)
        if combo_meta["min_blocks"]:
            self.x_grid = self.maximum([self.x_grid, combo_meta["min_blocks"]])
        if ynumels:
            self.y_grid = self.ceildiv(self.maximum(ynumels), meta.get("YBLOCK"))
        if znumels:
            self.z_grid = self.ceildiv(self.maximum(znumels), meta.get("ZBLOCK"))

    def combo_x_grid(
        self,
        xnumels: list[int | str],
        no_x_dims: list[bool],
        meta: dict[str, int],
    ) -> str | int:
        raise NotImplementedError


class SequentialComboKernelGrid(ComboKernelGrid):
    def combo_x_grid(
        self,
        xnumels: list[int | str],
        no_x_dims: list[bool],
        meta: dict[str, int],
    ) -> str | int:
        assert len(xnumels) == len(no_x_dims)
        return self.summation(
            [
                self.ceildiv(x, 1 if no_x_dim else meta.get("XBLOCK"))
                for x, no_x_dim in zip(xnumels, no_x_dims)
            ]
        )


class RoundRobinComboKernelGrid(ComboKernelGrid):
    def combo_x_grid(
        self,
        xnumels: list[int | str],
        no_x_dims: list[bool],
        meta: dict[str, int],
    ) -> str:
        assert len(xnumels) == len(no_x_dims)
        num_kernels = self.inductor_meta["combo_grid_meta"]["num_kernels"]
        exprs = [x for x, no_x_dim in zip(xnumels, no_x_dims) if no_x_dim]
        xnumels_x_dim = [x for x, no_x_dim in zip(xnumels, no_x_dims) if not no_x_dim]
        if xnumels_x_dim:
            exprs.append(self.ceildiv(self.maximum(xnumels_x_dim), meta.get("XBLOCK")))
        return f"({self.maximum(exprs)}) * {num_kernels}"

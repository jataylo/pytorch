#!/usr/bin/env python3
"""
Comprehensive Reduction Kernel Benchmark
Tests eager vs compile performance for reduction ops across shapes that
exercise INNER, OUTER, and PERSISTENT reduction paths in Triton Inductor.

Reduction hint classification (from Inductor's analysis):
  INNER      — reduction over the last (innermost) dimension; most common.
               e.g. x.sum(dim=-1), x.mean(dim=-1), x.var(dim=-1)
  OUTER      — reduction over a leading dimension, keeping the last.
               e.g. x.sum(dim=0)  on a 2-D tensor.
  PERSISTENT — small enough to fit the entire reduction in one block.
               Triton uses a static tile; no multi-block split.

Run with:
    python benchmark_reduction.py [--quick] [--warmup N] [--iters N]
                                  [--no-heuristics] [--clear-per-shape]

Compare against max-autotune baseline:
    TORCHINDUCTOR_MAX_AUTOTUNE=1 python benchmark_reduction.py --iters 100
"""

import math
import os
import time
from statistics import geometric_mean
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

# ── Default env-var setup (can be overridden by caller) ──────────────────────
os.environ.setdefault("TORCHINDUCTOR_REDUCTION_HEURISTICS", "1")
os.environ.setdefault("TORCHINDUCTOR_POINTWISE_HEURISTICS",  "1")
os.environ.setdefault("TORCHINDUCTOR_HEURISTICS_REAL_BENCH", "1")
os.environ.setdefault("TORCHINDUCTOR_DYNAMIC_SHAPES",        "0")


# ── Shape taxonomy ───────────────────────────────────────────────────────────
#
# Each shape is tagged with its expected reduction hint so we can group
# results by regime in the summary.
#
#  INNER shapes  — (batch, large_reduc_dim)
#    The reduction runs over axis=-1.  The heuristic should emit
#    INNER configs: many blocks (one per batch row), large R0_BLOCK.
#
#  OUTER shapes  — (large_output_dim, small_reduc_dim)
#    The reduction runs over axis=0.  Inductor classifies this as OUTER.
#    Grid = ceil(output_dim / XBLOCK); R0_BLOCK strides over the small dim.
#
#  SQUARE shapes — square 2-D tensors summed over axis=-1.
#    Falls into INNER; interesting because x == r.

INNER_SHAPES: List[Tuple[str, Tuple[int, ...]]] = [
    # (batch, reduction_dim)
    ("inner_tiny_r",         (1,     64)),
    ("inner_small_r",        (1,    256)),
    ("inner_medium_r",       (1,   2048)),
    ("inner_large_r",        (1,  16384)),
    ("inner_xlarge_r",       (1, 131072)),
    # batch × reduction
    ("inner_b8_r8192",       (8,   8192)),
    ("inner_b16_r4096",      (16,  4096)),
    ("inner_b32_r2048",      (32,  2048)),
    ("inner_b64_r1024",      (64,  1024)),
    ("inner_b128_r512",      (128,  512)),
    ("inner_b256_r256",      (256,  256)),
    ("inner_b512_r128",      (512,  128)),
    ("inner_b1024_r64",      (1024,  64)),
    # larger batches
    ("inner_b4096_r256",     (4096,  256)),
    ("inner_b16384_r64",     (16384,  64)),
    # 3-D: (B, T, C) summed over C (last dim)
    ("inner_3d_b4_t256_c64",    (4,  256,  64)),
    ("inner_3d_b8_t128_c128",   (8,  128, 128)),
    ("inner_3d_b16_t64_c256",   (16,  64, 256)),
    ("inner_3d_b32_t32_c512",   (32,  32, 512)),
    # odd sizes (non-power-of-2)
    ("inner_odd_b37_r999",    (37,   999)),
    ("inner_odd_b333_r777",   (333,  777)),
]

OUTER_SHAPES: List[Tuple[str, Tuple[int, ...]]] = [
    # (large_output, small_reduction) — sum over dim=0
    ("outer_tiny_r",         (64,      1)),
    ("outer_small_r",        (256,     4)),
    ("outer_medium_r",       (1024,   16)),
    ("outer_large_r",        (4096,   32)),
    ("outer_xlarge_r",       (16384,  64)),
    # odd sizes
    ("outer_odd_x999_r7",    (999,     7)),
    ("outer_odd_x3333_r33",  (3333,   33)),
]

SQUARE_SHAPES: List[Tuple[str, Tuple[int, ...]]] = [
    # (N, N) summed over dim=-1  (INNER, x == r)
    ("square_tiny",   (32,   32)),
    ("square_small",  (128, 128)),
    ("square_medium", (512, 512)),
    ("square_large",  (2048, 2048)),
]


# ── Benchmark harness ────────────────────────────────────────────────────────

class ReductionBenchmark:
    """
    Benchmark harness for reduction kernels.

    Structure mirrors benchmark_pointwise.py:
      - GPU-event timing (excludes Python / HIP_LAUNCH_BLOCKING overhead)
      - Optional cache-clear per shape to force fresh heuristic selection
      - Grouped summary with geometric-mean speedups
    """

    def __init__(
        self,
        device: str = "cuda",
        warmup_iters: int = 10,
        bench_iters:  int = 50,
        clear_cache_per_shape: bool = False,
    ):
        self.device               = device
        self.warmup_iters         = warmup_iters
        self.bench_iters          = bench_iters
        self.clear_cache_per_shape = clear_cache_per_shape

        self.results: List[Dict]            = []
        self._bench_timings: List[Tuple]    = []
        self._total_start: float            = 0.0

    # ── Timing ───────────────────────────────────────────────────────────────

    def _time_fn_gpu_events(self, fn: Callable, inputs: list, n_iters: int) -> float:
        """
        Measure pure GPU time via CUDA events (excludes Python dispatch overhead).

        Returns time in milliseconds (average over n_iters).
        """
        if self.device != "cuda":
            start = time.perf_counter()
            for _ in range(n_iters):
                fn(*inputs)
            return (time.perf_counter() - start) / n_iters * 1000

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_evt.record()
        for _ in range(n_iters):
            fn(*inputs)
        end_evt.record()
        torch.cuda.synchronize()
        return start_evt.elapsed_time(end_evt) / n_iters

    def benchmark_op(
        self,
        name:       str,
        eager_fn:   Callable,
        compile_fn: Callable,
        inputs:     list,
    ) -> Dict:
        """
        Benchmark one (eager, compiled) pair.

        Returns a dict with eager_ms, compile_ms, speedup, and wall_time_s.
        """
        _t0 = time.perf_counter()

        for _ in range(self.warmup_iters):
            eager_fn(*inputs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        eager_ms = self._time_fn_gpu_events(eager_fn, inputs, self.bench_iters)

        for _ in range(self.warmup_iters):
            compile_fn(*inputs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        compile_ms = self._time_fn_gpu_events(compile_fn, inputs, self.bench_iters)

        speedup = eager_ms / compile_ms if compile_ms > 0 else 0.0

        return {
            "name":        name,
            "eager_ms":    eager_ms,
            "compile_ms":  compile_ms,
            "speedup":     speedup,
            "wall_time_s": time.perf_counter() - _t0,
        }

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def clear_compilation_cache(self):
        """Clear all Triton / Inductor / Dynamo caches."""
        import gc
        import shutil

        try:
            torch._dynamo.reset()
            if hasattr(torch._inductor, "metrics"):
                torch._inductor.metrics.reset()
        except Exception as e:
            print(f"Warning: could not reset dynamo cache: {e}")

        for path in ["/tmp/torchinductor_root/",
                     "/tmp/torchinductor_autotune_cache/"]:
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                except Exception:
                    pass

        triton_cache = os.path.expanduser("~/.triton/cache/")
        if os.path.exists(triton_cache):
            try:
                shutil.rmtree(triton_cache)
            except Exception:
                pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Benchmark groups ──────────────────────────────────────────────────────
    # Each method follows the same pattern:
    #   1. Print a header.
    #   2. Define eager_fn.
    #   3. torch.compile the fn.
    #   4. Loop over shapes, call benchmark_op, store result.
    #   5. Optionally clear cache per shape.

    def _run_shapes(
        self,
        tag:        str,           # op tag stored in result['op']
        shapes:     List[Tuple],
        eager_fn:   Callable,
        compile_fn: Callable,
        make_inputs: Callable,     # shape → list[Tensor]
    ):
        for shape_name, shape in shapes:
            inputs = make_inputs(shape)
            result = self.benchmark_op(
                f"{tag}_{shape_name}", eager_fn, compile_fn, inputs
            )
            result["op"]    = tag
            result["shape"] = shape_name
            result["numel"] = math.prod(shape)
            self.results.append(result)

            numel_str = f"{result['numel']:>10,}"
            print(
                f"  {shape_name:<28s} numel={numel_str} | "
                f"Eager: {result['eager_ms']:8.4f}ms | "
                f"Compile: {result['compile_ms']:8.4f}ms | "
                f"Speedup: {result['speedup']:6.3f}x"
            )

            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    # ------------------------------------------------------------------
    # Phase 1 — INNER: sum over last dim
    # ------------------------------------------------------------------

    def bench_sum_inner(self):
        print("\n" + "=" * 80)
        print("  Benchmark 1: Sum (INNER) — x.sum(dim=-1)")
        print("  Hint: INNER  |  Regime: large R-dim per row")
        print("=" * 80)

        def eager_fn(x):   return x.sum(dim=-1)
        compile_fn = torch.compile(eager_fn)

        def make(shape):
            return [torch.randn(*shape, device=self.device)]

        self._run_shapes("sum_inner", INNER_SHAPES, eager_fn, compile_fn, make)

    def bench_mean_inner(self):
        print("\n" + "=" * 80)
        print("  Benchmark 2: Mean (INNER) — x.mean(dim=-1)")
        print("  Hint: INNER  |  Same layout as sum, different reduction body")
        print("=" * 80)

        def eager_fn(x):   return x.mean(dim=-1)
        compile_fn = torch.compile(eager_fn)

        def make(shape):
            return [torch.randn(*shape, device=self.device)]

        self._run_shapes("mean_inner", INNER_SHAPES, eager_fn, compile_fn, make)

    # ------------------------------------------------------------------
    # Phase 2 — INNER: variance / std (register-intensive Welford)
    # ------------------------------------------------------------------

    def bench_var_inner(self):
        print("\n" + "=" * 80)
        print("  Benchmark 3: Variance (INNER) — x.var(dim=-1)")
        print("  Hint: INNER  |  Welford accumulator: 3+ fp32 registers per element")
        print("  register_intensive=True → halved num_warps cap")
        print("=" * 80)

        def eager_fn(x):   return x.var(dim=-1, unbiased=False)
        compile_fn = torch.compile(eager_fn)

        def make(shape):
            return [torch.randn(*shape, device=self.device)]

        self._run_shapes("var_inner", INNER_SHAPES, eager_fn, compile_fn, make)

    def bench_std_inner(self):
        print("\n" + "=" * 80)
        print("  Benchmark 4: Std-dev (INNER) — x.std(dim=-1)")
        print("  Hint: INNER  |  sqrt(var) — same Welford + one sqrt per output")
        print("=" * 80)

        def eager_fn(x):   return x.std(dim=-1, unbiased=False)
        compile_fn = torch.compile(eager_fn)

        def make(shape):
            return [torch.randn(*shape, device=self.device)]

        self._run_shapes("std_inner", INNER_SHAPES, eager_fn, compile_fn, make)

    # ------------------------------------------------------------------
    # Phase 3 — INNER: softmax and log-softmax (persistent-reduction style)
    # ------------------------------------------------------------------

    def bench_softmax(self):
        print("\n" + "=" * 80)
        print("  Benchmark 5: Softmax (INNER) — F.softmax(x, dim=-1)")
        print("  Hint: INNER / PERSISTENT for small R")
        print("  Two-pass: (1) max reduction, (2) exp+sum, (3) divide")
        print("=" * 80)

        def eager_fn(x):   return F.softmax(x, dim=-1)
        compile_fn = torch.compile(eager_fn)

        def make(shape):
            return [torch.randn(*shape, device=self.device)]

        self._run_shapes("softmax", INNER_SHAPES, eager_fn, compile_fn, make)

    def bench_log_softmax(self):
        print("\n" + "=" * 80)
        print("  Benchmark 6: Log-Softmax (INNER) — F.log_softmax(x, dim=-1)")
        print("  Hint: INNER / PERSISTENT  |  numerically stable log(softmax(x))")
        print("=" * 80)

        def eager_fn(x):   return F.log_softmax(x, dim=-1)
        compile_fn = torch.compile(eager_fn)

        def make(shape):
            return [torch.randn(*shape, device=self.device)]

        self._run_shapes("log_softmax", INNER_SHAPES, eager_fn, compile_fn, make)

    # ------------------------------------------------------------------
    # Phase 4 — INNER: normalisation (layer norm, RMS norm)
    # ------------------------------------------------------------------

    def bench_layer_norm(self):
        print("\n" + "=" * 80)
        print("  Benchmark 7: Layer Norm (INNER) — F.layer_norm(x, [C])")
        print("  Hint: INNER  |  Heavy: mean + var + normalise in one fused kernel")
        print("=" * 80)

        def eager_fn(x, w, b):
            norm_shape = (x.shape[-1],)
            return F.layer_norm(x, norm_shape, weight=w, bias=b)

        compile_fn = torch.compile(eager_fn)

        def make(shape):
            c = shape[-1]
            return [
                torch.randn(*shape, device=self.device),
                torch.ones(c,       device=self.device),
                torch.zeros(c,      device=self.device),
            ]

        self._run_shapes("layer_norm", INNER_SHAPES, eager_fn, compile_fn, make)

    def bench_rms_norm(self):
        print("\n" + "=" * 80)
        print("  Benchmark 8: RMS Norm (INNER) — x / sqrt(mean(x^2) + eps)")
        print("  Hint: INNER  |  Simpler than LayerNorm: one-pass over r-dim")
        print("=" * 80)

        def eager_fn(x, w):
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            return x / rms * w

        compile_fn = torch.compile(eager_fn)

        def make(shape):
            c = shape[-1]
            return [
                torch.randn(*shape, device=self.device),
                torch.ones(c,       device=self.device),
            ]

        self._run_shapes("rms_norm", INNER_SHAPES, eager_fn, compile_fn, make)

    # ------------------------------------------------------------------
    # Phase 5 — INNER: amax / amin / norm
    # ------------------------------------------------------------------

    def bench_amax_inner(self):
        print("\n" + "=" * 80)
        print("  Benchmark 9: Amax (INNER) — x.amax(dim=-1)")
        print("  Hint: INNER  |  argmax-style reduction; no accumulator register pressure")
        print("=" * 80)

        def eager_fn(x):   return x.amax(dim=-1)
        compile_fn = torch.compile(eager_fn)

        def make(shape):
            return [torch.randn(*shape, device=self.device)]

        self._run_shapes("amax_inner", INNER_SHAPES, eager_fn, compile_fn, make)

    def bench_norm_inner(self):
        print("\n" + "=" * 80)
        print("  Benchmark 10: L2-Norm (INNER) — x.norm(dim=-1)")
        print("  Hint: INNER  |  square + sum + sqrt; one accumulator register")
        print("=" * 80)

        def eager_fn(x):   return x.norm(dim=-1)
        compile_fn = torch.compile(eager_fn)

        def make(shape):
            return [torch.randn(*shape, device=self.device)]

        self._run_shapes("norm_inner", INNER_SHAPES, eager_fn, compile_fn, make)

    # ------------------------------------------------------------------
    # Phase 6 — OUTER: sum over first dim
    # ------------------------------------------------------------------

    def bench_sum_outer(self):
        print("\n" + "=" * 80)
        print("  Benchmark 11: Sum (OUTER) — x.sum(dim=0)")
        print("  Hint: OUTER  |  Grid covers output dim; r-stride over leading dim")
        print("=" * 80)

        def eager_fn(x):   return x.sum(dim=0)
        compile_fn = torch.compile(eager_fn)

        def make(shape):
            # shape = (output_dim, reduction_dim); transpose for outer convention
            return [torch.randn(shape[1], shape[0], device=self.device)]

        self._run_shapes("sum_outer", OUTER_SHAPES, eager_fn, compile_fn, make)

    def bench_mean_outer(self):
        print("\n" + "=" * 80)
        print("  Benchmark 12: Mean (OUTER) — x.mean(dim=0)")
        print("  Hint: OUTER")
        print("=" * 80)

        def eager_fn(x):   return x.mean(dim=0)
        compile_fn = torch.compile(eager_fn)

        def make(shape):
            return [torch.randn(shape[1], shape[0], device=self.device)]

        self._run_shapes("mean_outer", OUTER_SHAPES, eager_fn, compile_fn, make)

    def bench_amax_outer(self):
        print("\n" + "=" * 80)
        print("  Benchmark 13: Amax (OUTER) — x.amax(dim=0)")
        print("  Hint: OUTER")
        print("=" * 80)

        def eager_fn(x):   return x.amax(dim=0)
        compile_fn = torch.compile(eager_fn)

        def make(shape):
            return [torch.randn(shape[1], shape[0], device=self.device)]

        self._run_shapes("amax_outer", OUTER_SHAPES, eager_fn, compile_fn, make)

    # ------------------------------------------------------------------
    # Phase 7 — Square: x and r are equal (interesting edge case)
    # ------------------------------------------------------------------

    def bench_sum_square(self):
        print("\n" + "=" * 80)
        print("  Benchmark 14: Sum Square — (N, N).sum(dim=-1)")
        print("  Hint: INNER  |  x == r; tests boundary between persistent and split")
        print("=" * 80)

        def eager_fn(x):   return x.sum(dim=-1)
        compile_fn = torch.compile(eager_fn)

        def make(shape):
            return [torch.randn(*shape, device=self.device)]

        self._run_shapes("sum_square", SQUARE_SHAPES, eager_fn, compile_fn, make)

    def bench_softmax_square(self):
        print("\n" + "=" * 80)
        print("  Benchmark 15: Softmax Square — (N, N).softmax(dim=-1)")
        print("  Hint: INNER / PERSISTENT  |  square tensor, tests persistent regime")
        print("=" * 80)

        def eager_fn(x):   return F.softmax(x, dim=-1)
        compile_fn = torch.compile(eager_fn)

        def make(shape):
            return [torch.randn(*shape, device=self.device)]

        self._run_shapes("softmax_square", SQUARE_SHAPES, eager_fn, compile_fn, make)

    # ------------------------------------------------------------------
    # Phase 8 — Heavy fused reductions (as in MLP / Attn patterns)
    # ------------------------------------------------------------------

    def bench_fused_norm_add(self):
        """Fused: y = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)"""
        print("\n" + "=" * 80)
        print("  Benchmark 16: Fused Norm+Add — manual layer-norm style")
        print("  mean + std + broadcast + divide — multiple fused reduction kernels")
        print("=" * 80)

        def eager_fn(x, w, b):
            mu  = x.mean(dim=-1, keepdim=True)
            sig = x.std (dim=-1, keepdim=True, unbiased=False)
            return (x - mu) / (sig + 1e-6) * w + b

        compile_fn = torch.compile(eager_fn)

        shapes = [
            ("b8_r2048",   (8,   2048)),
            ("b64_r512",   (64,   512)),
            ("b256_r256",  (256,  256)),
            ("b1024_r128", (1024, 128)),
            ("b4096_r64",  (4096,  64)),
        ]

        for shape_name, shape in shapes:
            c = shape[-1]
            inputs = [
                torch.randn(*shape, device=self.device),
                torch.ones (c,      device=self.device),
                torch.zeros(c,      device=self.device),
            ]
            result = self.benchmark_op(
                f"fused_norm_{shape_name}", eager_fn, compile_fn, inputs
            )
            result["op"]    = "fused_norm"
            result["shape"] = shape_name
            result["numel"] = math.prod(shape)
            self.results.append(result)

            print(
                f"  {shape_name:<28s} | "
                f"Eager: {result['eager_ms']:8.4f}ms | "
                f"Compile: {result['compile_ms']:8.4f}ms | "
                f"Speedup: {result['speedup']:6.3f}x"
            )
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_attention_softmax(self):
        """Attention-style: score = q*k / sqrt(d); w = softmax(score); out = w * v"""
        print("\n" + "=" * 80)
        print("  Benchmark 17: Attention Score + Softmax (INNER persistent)")
        print("  score = q*k/sqrt(d), softmax over seq_len, then weight*v")
        print("=" * 80)

        def eager_fn(q, k, v):
            d    = q.shape[-1]
            sc   = (q * k) / math.sqrt(d)
            w    = F.softmax(sc, dim=-1)
            return w * v

        compile_fn = torch.compile(eager_fn)

        # (batch × heads, seq_len, head_dim)
        shapes = [
            ("b1_s64_d64",    (1,    64,  64)),
            ("b1_s256_d64",   (1,   256,  64)),
            ("b1_s1024_d64",  (1,  1024,  64)),
            ("b8_s128_d64",   (8,   128,  64)),
            ("b8_s512_d64",   (8,   512,  64)),
            ("b32_s64_d128",  (32,   64, 128)),
            ("b32_s256_d128", (32,  256, 128)),
        ]

        for shape_name, (B, S, D) in shapes:
            inputs = [
                torch.randn(B, S, D, device=self.device) for _ in range(3)
            ]
            result = self.benchmark_op(
                f"attn_softmax_{shape_name}", eager_fn, compile_fn, inputs
            )
            result["op"]    = "attn_softmax"
            result["shape"] = shape_name
            result["numel"] = B * S * D
            self.results.append(result)

            print(
                f"  {shape_name:<28s} | "
                f"Eager: {result['eager_ms']:8.4f}ms | "
                f"Compile: {result['compile_ms']:8.4f}ms | "
                f"Speedup: {result['speedup']:6.3f}x"
            )
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    # ── Summary ───────────────────────────────────────────────────────────────

    def print_summary(self):
        print("\n" + "=" * 80)
        print("  SUMMARY")
        print("=" * 80)

        ops: Dict[str, List[float]] = {}
        for r in self.results:
            ops.setdefault(r["op"], []).append(r["speedup"])

        print("\nGeometric Mean Speedup by Operation:")
        print("-" * 55)
        all_speedups: List[float] = []
        for op, speedups in sorted(ops.items()):
            gmean = geometric_mean(speedups)
            all_speedups.extend(speedups)
            print(f"  {op:<32s}: {gmean:6.3f}x  (n={len(speedups)})")
        overall = geometric_mean(all_speedups)
        print("-" * 55)
        print(f"  {'OVERALL':<32s}: {overall:6.3f}x")
        print("=" * 80)

        best  = max(self.results, key=lambda r: r["speedup"])
        worst = min(self.results, key=lambda r: r["speedup"])
        print(f"\nBest speedup:  {best['name']:<48s} {best['speedup']:6.3f}x")
        print(f"Worst speedup: {worst['name']:<48s} {worst['speedup']:6.3f}x")

        # Breakdown by reduction-size bucket
        print("\nSpeedup by Problem Size (numel):")
        print("-" * 55)
        buckets = {
            "tiny   (<10K)":      [],
            "small  (10K–1M)":    [],
            "medium (1M–100M)":   [],
            "large  (>100M)":     [],
        }
        for r in self.results:
            n = r["numel"]
            if   n < 10_000:       buckets["tiny   (<10K)"].append(r["speedup"])
            elif n < 1_000_000:    buckets["small  (10K–1M)"].append(r["speedup"])
            elif n < 100_000_000:  buckets["medium (1M–100M)"].append(r["speedup"])
            else:                  buckets["large  (>100M)"].append(r["speedup"])

        for label, speedups in buckets.items():
            if speedups:
                print(f"  {label}: {geometric_mean(speedups):6.3f}x  ({len(speedups)} kernels)")

        print("=" * 80)

        # Wall-clock timing table
        if self._bench_timings:
            total_elapsed = time.perf_counter() - self._total_start
            print("\n" + "=" * 80)
            print("  WALL-CLOCK TIMING BREAKDOWN")
            print("=" * 80)
            hdr = f"  {'Benchmark':<34}  {'Time':>8}  {'Shapes':>6}  {'Avg/shape':>9}  {'%total':>7}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for label, elapsed, n_sub in self._bench_timings:
                avg = elapsed / n_sub if n_sub else 0.0
                pct = elapsed / total_elapsed * 100 if total_elapsed > 0 else 0.0
                print(f"  {label:<34}  {elapsed:7.1f}s  {n_sub:6d}  {avg:8.2f}s  {pct:6.1f}%")
            print(f"\n  Total wall time: {total_elapsed:.1f}s")
            print("=" * 80)

    def save_csv(self, filename: str = "reduction_benchmark_results.csv"):
        import csv
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["op", "shape", "numel", "eager_ms", "compile_ms", "speedup"],
            )
            writer.writeheader()
            for r in self.results:
                writer.writerow(
                    {k: r[k] for k in ["op", "shape", "numel", "eager_ms", "compile_ms", "speedup"]}
                )
        print(f"\n✅  Results saved to {filename}")

    # ── Orchestration ─────────────────────────────────────────────────────────

    def _time_bench(self, label: str, fn: Callable):
        """Run fn(), record wall time and result-count for the timing table."""
        n_before = len(self.results)
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        self._bench_timings.append((label, elapsed, len(self.results) - n_before))

    def run_all(self):
        print("\n" + "=" * 80)
        print("  REDUCTION KERNEL BENCHMARK SUITE")
        print("=" * 80)
        print(f"  Device          : {self.device}")
        print(f"  Warmup iters    : {self.warmup_iters}")
        print(f"  Bench iters     : {self.bench_iters}")
        print(f"  Reduction heur  : {os.environ.get('TORCHINDUCTOR_REDUCTION_HEURISTICS', '?')}")
        print(f"  Max autotune    : {os.environ.get('TORCHINDUCTOR_MAX_AUTOTUNE', '0')}")
        print(f"  Clear/shape     : {self.clear_cache_per_shape}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"  GPU             : {props.name}")
            if torch.version.hip:
                print(f"  ROCm            : {torch.version.hip}")
        print("=" * 80)

        self._total_start = time.perf_counter()

        print("\n" + "=" * 80)
        print("  PHASE 1 — INNER Reductions: Sum / Mean (simple accumulators)")
        print("=" * 80)
        self._time_bench("Sum   (INNER)",   self.bench_sum_inner)
        self._time_bench("Mean  (INNER)",   self.bench_mean_inner)

        print("\n" + "=" * 80)
        print("  PHASE 2 — INNER Reductions: Var / Std (Welford, register-intensive)")
        print("=" * 80)
        self._time_bench("Var   (INNER)",   self.bench_var_inner)
        self._time_bench("Std   (INNER)",   self.bench_std_inner)

        print("\n" + "=" * 80)
        print("  PHASE 3 — INNER / PERSISTENT: Softmax / Log-Softmax")
        print("=" * 80)
        self._time_bench("Softmax  (INNER)", self.bench_softmax)
        self._time_bench("LogSmax  (INNER)", self.bench_log_softmax)

        print("\n" + "=" * 80)
        print("  PHASE 4 — Normalisation: LayerNorm / RMSNorm")
        print("=" * 80)
        self._time_bench("LayerNorm",       self.bench_layer_norm)
        self._time_bench("RMSNorm",         self.bench_rms_norm)

        print("\n" + "=" * 80)
        print("  PHASE 5 — INNER: Amax / L2-Norm")
        print("=" * 80)
        self._time_bench("Amax  (INNER)",   self.bench_amax_inner)
        self._time_bench("Norm  (INNER)",   self.bench_norm_inner)

        print("\n" + "=" * 80)
        print("  PHASE 6 — OUTER Reductions: Sum / Mean / Amax over dim=0")
        print("=" * 80)
        self._time_bench("Sum  (OUTER)",    self.bench_sum_outer)
        self._time_bench("Mean (OUTER)",    self.bench_mean_outer)
        self._time_bench("Amax (OUTER)",    self.bench_amax_outer)

        print("\n" + "=" * 80)
        print("  PHASE 7 — Square tensors (x == r boundary)")
        print("=" * 80)
        self._time_bench("Sum  (square)",   self.bench_sum_square)
        self._time_bench("Softmax (square)", self.bench_softmax_square)

        print("\n" + "=" * 80)
        print("  PHASE 8 — Heavy fused reductions (multi-kernel patterns)")
        print("=" * 80)
        self._time_bench("Fused Norm+Add",   self.bench_fused_norm_add)
        self._time_bench("Attn Softmax",     self.bench_attention_softmax)

        self.print_summary()
        self.save_csv()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Triton reduction kernels")
    parser.add_argument("--device",         default="cuda",
                        help="Device (cuda / cpu)")
    parser.add_argument("--warmup",         type=int, default=10,
                        help="Warmup iterations per shape")
    parser.add_argument("--iters",          type=int, default=50,
                        help="Benchmark iterations per shape")
    parser.add_argument("--no-heuristics",  action="store_true",
                        help="Disable reduction heuristics (set env var to 0)")
    parser.add_argument("--clear-per-shape", action="store_true",
                        help="Clear Triton / Inductor cache after EACH shape")
    parser.add_argument("--quick",          action="store_true",
                        help="Use a small subset of shapes for fast iteration")
    parser.add_argument("--csv",            default="reduction_benchmark_results.csv",
                        help="Output CSV filename")
    args = parser.parse_args()

    # ── Configure env vars ───────────────────────────────────────────────────
    if args.no_heuristics:
        os.environ["TORCHINDUCTOR_REDUCTION_HEURISTICS"] = "0"
        print("\n⚠️  Reduction heuristics DISABLED")
    elif "TORCHINDUCTOR_REDUCTION_HEURISTICS" not in os.environ:
        os.environ["TORCHINDUCTOR_REDUCTION_HEURISTICS"] = "1"
        print("\n✅ Reduction heuristics ENABLED")
    else:
        val = os.environ["TORCHINDUCTOR_REDUCTION_HEURISTICS"]
        icon = "✅" if val == "1" else "⚠️ "
        print(f"\n{icon} Reduction heuristics {'ENABLED' if val == '1' else 'DISABLED'}"
              f"  (env var={val})")

    # ── Optionally shrink shape lists for quick mode ─────────────────────────
    if args.quick:
        print("\n⚡ QUICK MODE: reduced shape lists")
        # Overwrite module-level lists
        import benchmark_reduction as _self
        _self.INNER_SHAPES = [
            ("inner_b8_r8192",    (8,    8192)),
            ("inner_b64_r1024",   (64,   1024)),
            ("inner_b256_r256",   (256,   256)),
            ("inner_b1024_r64",   (1024,   64)),
            ("inner_3d_b8_t128_c128", (8, 128, 128)),
        ]
        _self.OUTER_SHAPES = [
            ("outer_medium_r",    (1024, 16)),
            ("outer_large_r",     (4096, 32)),
        ]
        _self.SQUARE_SHAPES = [
            ("square_small",  (128,  128)),
            ("square_medium", (512,  512)),
        ]

    # ── Run ──────────────────────────────────────────────────────────────────
    bench = ReductionBenchmark(
        device=args.device,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
        clear_cache_per_shape=args.clear_per_shape,
    )
    bench.run_all()
    bench.save_csv(args.csv)


if __name__ == "__main__":
    main()

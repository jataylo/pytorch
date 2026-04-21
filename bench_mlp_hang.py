#!/usr/bin/env python3
"""
Micro-benchmark for "Benchmark 13: Heavy Fusion – MLP Style" hang investigation.

Reproduces the exact computation from benchmark_pointwise.py::bench_heavy_fusion_mlp
for each shape, with:
  - TORCHINDUCTOR_HEURISTICS_BENCH_VERBOSE=1  → prints every config before/after bench
  - A wall-clock timeout per shape (default 120 s) to detect hangs automatically
  - Prints the exact kernel counts / config counts per shape

Usage:
  cd /root/pytorch
  rm -rf /tmp/torchinductor_root /tmp/torchinductor_autotune_cache ~/.triton/cache
  TORCHINDUCTOR_POINTWISE_HEURISTICS=1 \\
  TORCHINDUCTOR_HEURISTICS_REAL_BENCH=1 \\
  TORCHINDUCTOR_REDUCTION_HEURISTICS=1 \\
  TORCHINDUCTOR_POINTWISE_WAVES_PER_EU=1 \\
  TORCHINDUCTOR_HEURISTICS_TOP_N=5 \\
  TORCHINDUCTOR_HEURISTICS_DIVERSITY=1 \\
  TORCHINDUCTOR_HEURISTICS_BENCH_VERBOSE=1 \\
  HIP_LAUNCH_BLOCKING=0 \\
  HIP_VISIBLE_DEVICES=1 \\
  python bench_mlp_hang.py 2>&1 | tee bench_mlp_hang.log

To jump straight to the shape that hangs, set --start-shape index (0-based):
  python bench_mlp_hang.py --start-shape 1   # skip (1048576,), start at (16777216,)
"""

import argparse
import os
import sys
import time
import signal
import threading

# ── env defaults (can be overridden before import) ─────────────────────────────
os.environ.setdefault("TORCHINDUCTOR_POINTWISE_HEURISTICS", "1")
os.environ.setdefault("TORCHINDUCTOR_HEURISTICS_REAL_BENCH", "1")
os.environ.setdefault("TORCHINDUCTOR_REDUCTION_HEURISTICS", "1")
os.environ.setdefault("TORCHINDUCTOR_POINTWISE_WAVES_PER_EU", "1")
os.environ.setdefault("TORCHINDUCTOR_HEURISTICS_TOP_N", "5")
os.environ.setdefault("TORCHINDUCTOR_HEURISTICS_DIVERSITY", "1")
os.environ.setdefault("TORCHINDUCTOR_HEURISTICS_BENCH_VERBOSE", "1")
# Force synchronous Triton compilation so every compile stall is visible in the
# per-config BENCH_VERBOSE log and atexit workers never time out.  Async compile
# (32 workers) can make the main thread appear hung while workers are busy.
# Override with TORCHINDUCTOR_COMPILE_THREADS=<N> before running if you want
# async back (e.g. =4 for some parallelism without the 300s shutdown timeout).
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch  # noqa: E402

DEVICE = "cuda"

# ── shapes from bench_heavy_fusion_mlp ────────────────────────────────────────
ALL_SHAPES = [
    ("1D_large",        (1048576,)),
    ("1D_huge",         (16777216,)),
    ("2D_square_medium", (2048, 2048)),
    ("2D_square_large",  (4096, 4096)),
    ("2D_odd_mixed",    (1234, 5678)),
    ("3D_medium",       (64, 64, 64)),
    ("3D_batch_medium", (16, 256, 256)),
]


def mlp_fn(x, w1, b1, w2, b2):
    """Exact copy of eager_fn from bench_heavy_fusion_mlp."""
    h1 = x * w1 + b1
    h1_act = torch.nn.functional.gelu(h1)
    h2 = h1_act * w2 + b2
    residual = x + h2
    mean = torch.mean(residual)
    std = torch.std(residual)
    normalized = (residual - mean) / (std + 1e-6)
    return normalized


# ── timeout helper ─────────────────────────────────────────────────────────────
class _Watchdog:
    """Sends SIGALRM after `seconds` if not cancelled."""
    def __init__(self, seconds: int):
        self._seconds = seconds
        self._timer = None

    def _fire(self):
        print(
            f"\n[WATCHDOG] ⚠ Shape timed out after {self._seconds}s — "
            "likely a hang in benchmarking!  Sending SIGALRM.",
            flush=True,
        )
        os.kill(os.getpid(), signal.SIGALRM)

    def start(self):
        self._timer = threading.Timer(self._seconds, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self):
        if self._timer is not None:
            self._timer.cancel()


def _sigalrm_handler(signum, frame):
    raise TimeoutError("Shape timed out (SIGALRM from watchdog)")


signal.signal(signal.SIGALRM, _sigalrm_handler)


# ── benchmark one shape ────────────────────────────────────────────────────────
def run_shape(
    shape_name: str,
    shape: tuple,
    compile_fn,
    warmup: int = 5,
    iters: int = 20,
    timeout_s: int = 180,
):
    print(f"\n{'='*70}", flush=True)
    print(f"  Shape: {shape_name}  {shape}  ({torch.Size(shape).numel():,} elements)", flush=True)
    print(f"{'='*70}", flush=True)

    x  = torch.randn(shape, device=DEVICE, dtype=torch.float32)
    w1 = torch.randn(shape, device=DEVICE, dtype=torch.float32)
    b1 = torch.randn(shape, device=DEVICE, dtype=torch.float32)
    w2 = torch.randn(shape, device=DEVICE, dtype=torch.float32)
    b2 = torch.randn(shape, device=DEVICE, dtype=torch.float32)
    inputs = [x, w1, b1, w2, b2]

    dog = _Watchdog(timeout_s)
    dog.start()
    try:
        t0 = time.perf_counter()
        print(f"  [t={time.perf_counter()-t0:.1f}s] Starting first compiled call (triggers compilation)…", flush=True)
        out = compile_fn(*inputs)
        torch.cuda.synchronize()
        print(f"  [t={time.perf_counter()-t0:.1f}s] First call done (compiled).", flush=True)

        # Warmup
        print(f"  [t={time.perf_counter()-t0:.1f}s] Warmup ({warmup} iters)…", flush=True)
        for _ in range(warmup):
            compile_fn(*inputs)
        torch.cuda.synchronize()
        print(f"  [t={time.perf_counter()-t0:.1f}s] Warmup done.", flush=True)

        # Timed iters
        print(f"  [t={time.perf_counter()-t0:.1f}s] Timing ({iters} iters)…", flush=True)
        events_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        events_end   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            events_start[i].record()
            compile_fn(*inputs)
            events_end[i].record()
        torch.cuda.synchronize()
        times_ms = [s.elapsed_time(e) for s, e in zip(events_start, events_end)]
        median_ms = sorted(times_ms)[len(times_ms) // 2]
        print(f"  [t={time.perf_counter()-t0:.1f}s] median={median_ms:.4f} ms", flush=True)
        dog.cancel()
        return median_ms

    except TimeoutError as e:
        print(f"  [HANG DETECTED] {e}", flush=True)
        # Print stack of main thread (already in the handler, so just re-raise)
        raise
    except Exception as e:
        dog.cancel()
        print(f"  [ERROR] {e}", flush=True)
        raise


# ── clear caches between shapes ───────────────────────────────────────────────
def clear_caches():
    """Reset torch.compile and Triton caches so each shape is compiled fresh."""
    torch._dynamo.reset()
    # PyCodeCache exposes cache_clear() (lru_cache-style), not .cache.
    try:
        import torch._inductor.codecache as cc
        if hasattr(cc, "PyCodeCache"):
            pcc = cc.PyCodeCache
            if hasattr(pcc, "cache_clear"):
                pcc.cache_clear()
            elif hasattr(pcc, "clear"):
                pcc.clear()
    except Exception:
        pass  # non-fatal — dynamo.reset() is the important one
    print("  [cache cleared]", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MLP hang micro-benchmark")
    parser.add_argument("--start-shape", type=int, default=0,
                        help="0-based index into the shapes list to start from")
    parser.add_argument("--end-shape", type=int, default=None,
                        help="0-based index (exclusive) to stop at (default: all)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180,
                        help="Per-shape timeout in seconds")
    parser.add_argument("--no-clear", action="store_true",
                        help="Don't clear caches between shapes")
    args = parser.parse_args()

    shapes = ALL_SHAPES[args.start_shape : args.end_shape]
    if not shapes:
        print("No shapes selected.", flush=True)
        sys.exit(1)

    print("=" * 70, flush=True)
    print("  MLP Hang Micro-Benchmark", flush=True)
    print(f"  Shapes: {[s[0] for s in shapes]}", flush=True)
    print(f"  BENCH_VERBOSE: {os.environ.get('TORCHINDUCTOR_HEURISTICS_BENCH_VERBOSE', '0')}", flush=True)
    print(f"  REAL_BENCH:    {os.environ.get('TORCHINDUCTOR_HEURISTICS_REAL_BENCH', '0')}", flush=True)
    print(f"  TOP_N:         {os.environ.get('TORCHINDUCTOR_HEURISTICS_TOP_N', '?')}", flush=True)
    print(f"  TIMEOUT:       {args.timeout}s per shape", flush=True)
    print("=" * 70, flush=True)

    compile_fn = torch.compile(mlp_fn)

    for i, (shape_name, shape) in enumerate(shapes):
        if i > 0 and not args.no_clear:
            print(f"\n--- clearing caches before shape {shape_name} ---", flush=True)
            clear_caches()
            compile_fn = torch.compile(mlp_fn)

        try:
            run_shape(
                shape_name, shape, compile_fn,
                warmup=args.warmup, iters=args.iters, timeout_s=args.timeout,
            )
        except TimeoutError:
            print(f"\n[FATAL] Hang confirmed on shape {shape_name}  {shape}", flush=True)
            print("Aborting remaining shapes.", flush=True)
            sys.exit(2)
        except Exception as e:
            print(f"\n[ERROR on {shape_name}] {e}", flush=True)
            # Continue with next shape


def _graceful_shutdown():
    """Wait for async compile workers before process exits to avoid atexit errors."""
    try:
        from torch._inductor.async_compile import shutdown_compile_workers
        shutdown_compile_workers()
    except Exception:
        pass


if __name__ == "__main__":
    import atexit
    atexit.register(_graceful_shutdown)
    main()


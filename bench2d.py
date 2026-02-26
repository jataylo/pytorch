#!/usr/bin/env python3
"""bench2d.py — focused 2D/3D block kernel benchmark.

Forces true XBLOCK+YBLOCK (and XBLOCK+YBLOCK+ZBLOCK) Triton kernels by using
non-contiguous tensor access patterns that inductor cannot flatten to 1D.

Usage:
    rm -rf ~/.triton/cache/ && rm -rf /tmp/torchinductor_root/
    TORCHINDUCTOR_POINTWISE_HEURISTICS=1 TORCHINDUCTOR_HEURISTICS_REAL_BENCH=1 \
    TORCHINDUCTOR_HEURISTICS_VERBOSE=1 HIP_LAUNCH_BLOCKING=1 \
    HIP_VISIBLE_DEVICES=1 TORCHINDUCTOR_HEURISTICS_TOP_N=5 \
    python bench2d.py --warmup 5 --iters 20 2>&1 | tee bench2d_out.log
"""
import argparse
import time
import torch
import torch._dynamo


# ── helpers ───────────────────────────────────────────────────────────────────

def _reset_cache():
    """Clear all compile + autotune caches between shape runs.

    Cache layers that must all be cleared, in order:

    1. torch._dynamo — compilation graph trace cache.
    2. PyCodeCache (in-memory) — inductor keeps compiled module objects in a
       process-level dict keyed by source hash.  torch._dynamo.reset() does NOT
       clear this.  If it still holds a (symbolically compiled) module from a
       previous shape, inductor will reuse it without calling pointwise() again,
       so _apply_pointwise_heuristics never runs and _heuristics_pending is never
       set → scoring is silently skipped for every shape after the first.
    3. Disk module cache (/tmp/torchinductor_root) — serialised Python wrappers.
    4. Triton kernel cache (~/.triton/cache) — PTX/hsaco binaries + autotune
       results.  Without this the autotune disk cache returns a winner from a
       prior non-REAL_BENCH run → _autotune_cache_hit=True → scoring skipped.
    """
    import os
    import shutil

    # 1. dynamo trace cache
    torch._dynamo.reset()

    # 2. inductor in-memory module cache (PyCodeCache.modules_no_attr).
    #    PyCodeCache stores compiled module objects keyed by on-disk path.
    #    If this dict is not cleared, inductor returns the stale module object
    #    (with an already-autotuned CachingAutotuner whose _heuristics_pending
    #    was already popped) instead of re-loading and re-running pointwise().
    try:
        from torch._inductor.codecache import PyCodeCache
        PyCodeCache.cache_clear()           # clears modules + modules_no_attr
    except Exception:
        pass

    # 2a. Purge compiled kernel entries from sys.modules.
    #     _reload_python_module() registers each loaded kernel module under
    #     "torch._inductor.runtime.compile_tasks.<hash>" in sys.modules.
    #     PyCodeCache.cache_clear() only clears modules_no_attr (path→module),
    #     not sys.modules.  If the entry survives, Python's import machinery
    #     can return the cached module object without re-executing pointwise(),
    #     so _heuristics_pending is never injected and scoring is silently skipped.
    import sys
    _prefix = "torch._inductor.runtime.compile_tasks."
    _to_del = [k for k in list(sys.modules) if k.startswith(_prefix)]
    for _k in _to_del:
        del sys.modules[_k]

    # 3a. Clear the per-process heuristic state dicts.
    #     These are module-level globals in triton_heuristics.py that accumulate
    #     validation data and top-N selections keyed by problem_key.  When the
    #     same kernel module is reused across shapes (symbolic compilation), the
    #     stale keys from a previous shape can produce a false "scoring worked"
    #     or "scoring skipped" signal for the new shape.  Reset them so each
    #     shape starts with a clean slate.
    try:
        import torch._inductor.runtime.triton_heuristics as _th
        _th._TOP_N_CONFIGS_FOR_SELECTION.clear()
        _th._HEURISTICS_VALIDATION_DATA.clear()
        _th._HEURISTICS_FULL_RANKED_HDICTS.clear()
    except Exception:
        pass

    # 3. inductor disk module cache
    inductor_cache = "/tmp/torchinductor_root"
    if os.path.isdir(inductor_cache):
        shutil.rmtree(inductor_cache, ignore_errors=True)

    # 4. Triton PTX + autotune cache
    triton_cache = os.path.expanduser("~/.triton/cache")
    if os.path.isdir(triton_cache):
        shutil.rmtree(triton_cache, ignore_errors=True)


def _bench(name, eager_fn, compile_fn, inputs, warmup, iters):
    """Benchmark eager vs compiled, printing the result line."""
    device = inputs[0].device

    # eager warmup + time
    for _ in range(warmup):
        _ = eager_fn(*inputs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = eager_fn(*inputs)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / iters * 1000

    # compiled warmup (1st call triggers compilation + heuristics scoring +
    # REAL_BENCH autotuning, whose output appears here in the log)
    for _ in range(warmup):
        _ = compile_fn(*inputs)
        torch.cuda.synchronize()

    # compiled timed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = compile_fn(*inputs)
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) / iters * 1000

    speedup = eager_ms / compile_ms if compile_ms > 0 else 0.0
    tag = "✓" if speedup >= 1.0 else "✗"
    print(
        f"  {name:28s} | Eager: {eager_ms:8.4f}ms | Compile: {compile_ms:8.4f}ms"
        f" | Speedup: {speedup:6.3f}x {tag}",
        flush=True,
    )
    return speedup


# ── 2-D kernels ───────────────────────────────────────────────────────────────

def bench_2d_transpose(warmup, iters, device):
    """z = x.T + y*w  — transpose forces 2D indexing (XBLOCK + YBLOCK)."""
    print()
    print("=" * 80)
    print("  2D Transpose:  z = x.T + y*w   (forces XBLOCK + YBLOCK)")
    print("=" * 80)

    def eager_fn(x, y, w):
        return x.transpose(-2, -1) + y * w

    compile_fn = torch.compile(eager_fn)

    shapes = [
        ("2D tiny  (128×128)",   (128,  128)),
        ("2D small (512×512)",   (512,  512)),
        ("2D wide  (64×8192)",   (64,  8192)),
        ("2D tall  (8192×64)",   (8192,  64)),
        ("2D med   (1024×1024)", (1024, 1024)),
        ("2D large (2048×2048)", (2048, 2048)),
        ("2D huge  (4096×4096)", (4096, 4096)),
    ]

    for name, shape in shapes:
        _reset_cache()
        x = torch.randn(shape, device=device)
        y = torch.randn((shape[1], shape[0]), device=device)
        w = torch.randn((shape[1], shape[0]), device=device)
        # dynamic=False forces a shape-specialized kernel per shape, ensuring
        # pointwise() is re-executed and heuristics scoring runs fresh.
        compile_fn = torch.compile(eager_fn, dynamic=False)
        _bench(name, eager_fn, compile_fn, [x, y, w], warmup, iters)


def bench_2d_broadcast(warmup, iters, device):
    """z = x + bias  — broadcast forces XBLOCK + YBLOCK when shapes differ."""
    print()
    print("=" * 80)
    print("  2D Broadcast:  z = x + bias   (row-wise broadcast)")
    print("=" * 80)

    def eager_fn(x, bias):
        return x + bias

    compile_fn = torch.compile(eager_fn)

    shapes = [
        ("2D bias tiny  (128×128)",   (128,  128),  128),
        ("2D bias small (512×512)",   (512,  512),  512),
        ("2D bias med   (1024×1024)", (1024, 1024), 1024),
        ("2D bias large (2048×2048)", (2048, 2048), 2048),
    ]

    for name, shape, n_cols in shapes:
        _reset_cache()
        x    = torch.randn(shape, device=device)
        bias = torch.randn(n_cols, device=device)
        compile_fn = torch.compile(eager_fn, dynamic=False)
        _bench(name, eager_fn, compile_fn, [x, bias], warmup, iters)


def bench_2d_strided(warmup, iters, device):
    """z = x[:, ::2] + y  — strided column access forces 2D kernel."""
    print()
    print("=" * 80)
    print("  2D Strided:  z = x[:, ::2] + y   (non-unit column stride)")
    print("=" * 80)

    def eager_fn(x, y):
        return x[:, ::2] + y

    shapes = [
        ("2D strided (256×512)",   (256, 512)),
        ("2D strided (512×1024)",  (512, 1024)),
        ("2D strided (1024×2048)", (1024, 2048)),
    ]

    for name, shape in shapes:
        _reset_cache()
        x = torch.randn(shape, device=device)
        y = torch.randn((shape[0], shape[1] // 2), device=device)
        compile_fn = torch.compile(eager_fn, dynamic=False)
        _bench(name, eager_fn, compile_fn, [x, y], warmup, iters)


# ── 3-D kernels ───────────────────────────────────────────────────────────────

def bench_3d_permute(warmup, iters, device):
    """z = x.permute(2,1,0) + y*w  — forces XBLOCK+YBLOCK+ZBLOCK."""
    print()
    print("=" * 80)
    print("  3D Permute:  z = x.permute(2,1,0) + y*w  (XBLOCK+YBLOCK+ZBLOCK)")
    print("=" * 80)

    def eager_fn(x, y, w):
        return x.permute(2, 1, 0) + y * w

    shapes = [
        ("3D tiny   (16×16×16)",   (16,  16,  16)),
        ("3D small  (32×32×32)",   (32,  32,  32)),
        ("3D med    (64×64×64)",   (64,  64,  64)),
        ("3D batch  (4×512×512)",  (4,  512, 512)),
        ("3D batch  (8×256×256)",  (8,  256, 256)),
    ]

    for name, shape in shapes:
        _reset_cache()
        rshape = (shape[2], shape[1], shape[0])
        x = torch.randn(shape, device=device)
        y = torch.randn(rshape, device=device)
        w = torch.randn(rshape, device=device)
        compile_fn = torch.compile(eager_fn, dynamic=False)
        _bench(name, eager_fn, compile_fn, [x, y, w], warmup, iters)


# ── main ─────────────────────────────────────────────────────────────────────

def _print_env_diagnostics():
    """Print which heuristic env-vars are active so the user knows before kernels compile."""
    import os
    print()
    print("=" * 60)
    print("  bench2d — environment diagnostics")
    print("=" * 60)

    hip = getattr(torch.version, "hip", None)
    print(f"  torch.version.hip          : {hip!r}")
    if not hip:
        print(
            "  ⚠  ROCm not detected — pointwise heuristics WILL NOT run.\n"
            "     Scoring, validation tables, and top-N selection are\n"
            "     all disabled on CUDA/CPU builds."
        )

    def _flag(env, default="1"):
        val = os.environ.get(env, default)
        return f"{val!r}  (env {env}={val!r}, default={default!r})"

    print(f"  POINTWISE_HEURISTICS       : {_flag('TORCHINDUCTOR_POINTWISE_HEURISTICS')}")
    print(f"  HEURISTICS_REAL_BENCH      : {_flag('TORCHINDUCTOR_HEURISTICS_REAL_BENCH')}")
    print(f"  HEURISTICS_VERBOSE         : {_flag('TORCHINDUCTOR_HEURISTICS_VERBOSE', '0')}")
    print(f"  HEURISTICS_TOP_N           : {_flag('TORCHINDUCTOR_HEURISTICS_TOP_N', '5')}")
    print(f"  HIP_LAUNCH_BLOCKING        : {_flag('HIP_LAUNCH_BLOCKING', '0')}")
    print(f"  HIP_VISIBLE_DEVICES        : {os.environ.get('HIP_VISIBLE_DEVICES', '(not set)')!r}")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(description="2D/3D Triton kernel benchmark")
    parser.add_argument("--warmup", type=int, default=5,  help="warmup iterations")
    parser.add_argument("--iters",  type=int, default=20, help="timed iterations")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-2d",  action="store_true", help="skip 2D benchmarks")
    parser.add_argument("--no-3d",  action="store_true", help="skip 3D benchmarks")
    args = parser.parse_args()

    _print_env_diagnostics()

    device = torch.device(args.device)
    print(f"Device : {torch.cuda.get_device_name(device)}")
    print(f"Warmup : {args.warmup}   Iters: {args.iters}")

    if not args.no_2d:
        bench_2d_transpose(args.warmup, args.iters, device)
        bench_2d_broadcast(args.warmup, args.iters, device)
        bench_2d_strided(args.warmup, args.iters, device)

    if not args.no_3d:
        bench_3d_permute(args.warmup, args.iters, device)

    print("\nDone.")


if __name__ == "__main__":
    main()


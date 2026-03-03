#!/usr/bin/env python3
"""bench3d.py — XBLOCK × YBLOCK × ZBLOCK pointwise kernel benchmark.

Why Inductor doesn't emit ZBLOCK by default
────────────────────────────────────────────
Inductor's tiling analysis has two relevant config knobs:

  triton.prefer_nd_tiling  (default False) — when True, Inductor attempts to
      keep loop dimensions separate instead of collapsing them to 1-D.

  triton.max_tiles  (default None → 2) — hard cap on the number of tile
      dimensions.  3-D kernels (ZBLOCK) require max_tiles ≥ 3.

Both must be enabled; otherwise even complex permute/broadcast operations
collapse to 1-D or 2-D loops and no ZBLOCK config is ever produced.

Operation pattern that reliably produces ZBLOCK
───────────────────────────────────────────────
  z = x.permute(0, 2, 1) + y

where x is (D, H, W) and y is the permuted shape (D, W, H).

x.permute(0,2,1) has strides (H*W, 1, W) in the (D,W,H) output space.
Adjacent-dim contiguity check on those strides:
  • dims 0↔1:  stride[0]=H*W  vs  W*stride[1]=W*1=W  → H*W ≠ W  (non-contiguous)
  • dims 1↔2:  stride[1]=1    vs  H*stride[2]=H*W     → 1 ≠ H*W  (non-contiguous)
Inductor cannot merge any two dims → generates a 3-D tiling (z=D, y=H, x=W).

Usage (ROCm / HIP):
    rm -rf ~/.triton/cache/ && rm -rf /tmp/torchinductor_root/
    TORCHINDUCTOR_POINTWISE_HEURISTICS=1 TORCHINDUCTOR_HEURISTICS_REAL_BENCH=1 \\
    TORCHINDUCTOR_HEURISTICS_VERBOSE=1 HIP_LAUNCH_BLOCKING=1 \\
    HIP_VISIBLE_DEVICES=1 TORCHINDUCTOR_HEURISTICS_TOP_N=5 \\
    python bench3d.py --warmup 5 --iters 20 2>&1 | tee bench3d_out.log
"""
# ── Force 3-D tiling BEFORE any inductor import ──────────────────────────────
# These must be set before torch.compile() is ever called.
import torch._inductor.config as _inductor_cfg
_inductor_cfg.triton.prefer_nd_tiling = True   # keep dims separate
_inductor_cfg.triton.max_tiles = 3             # allow z+y+x tile dims

import argparse
import time
import torch
import torch._dynamo


# ── cache clearing ────────────────────────────────────────────────────────────

def _reset_cache():
    """Clear every layer of compile + autotune cache between shape runs."""
    import os
    import shutil
    import sys

    torch._dynamo.reset()

    try:
        from torch._inductor.codecache import PyCodeCache
        PyCodeCache.cache_clear()
    except Exception:
        pass

    _prefix = "torch._inductor.runtime.compile_tasks."
    for k in [k for k in list(sys.modules) if k.startswith(_prefix)]:
        del sys.modules[k]

    try:
        import torch._inductor.runtime.triton_heuristics as _th
        _th._TOP_N_CONFIGS_FOR_SELECTION.clear()
        _th._HEURISTICS_VALIDATION_DATA.clear()
        _th._HEURISTICS_FULL_RANKED_HDICTS.clear()
    except Exception:
        pass

    for path in ("/tmp/torchinductor_root", os.path.expanduser("~/.triton/cache")):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


# ── timing helper ─────────────────────────────────────────────────────────────

def _bench(name, fn, inputs, warmup, iters):
    """Compile and time one (name, fn, inputs) case, printing the result line."""
    eager_fn = fn  # fn IS the eager function; compile a fresh wrapper
    cfn = torch.compile(eager_fn, dynamic=False)

    for _ in range(warmup):
        eager_fn(*inputs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        eager_fn(*inputs)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / iters * 1000

    # First compiled call triggers codegen + heuristic scoring + REAL_BENCH bench
    for _ in range(warmup):
        cfn(*inputs)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        cfn(*inputs)
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) / iters * 1000

    speedup = eager_ms / compile_ms if compile_ms > 0 else 0.0
    tag = "✓" if speedup >= 1.0 else "✗"
    print(
        f"  {name:36s} | Eager: {eager_ms:8.4f}ms"
        f" | Compile: {compile_ms:8.4f}ms | Speedup: {speedup:6.3f}x {tag}",
        flush=True,
    )


def _section(title, note=""):
    print()
    print("=" * 80)
    print(f"  {title}")
    if note:
        print(f"  {note}")
    print("=" * 80)


# ── benchmark suites ──────────────────────────────────────────────────────────
#
# ALL suites use the pattern  z = f(x.permute(0,2,1)) + g(y)
# because this is the operation that reliably produces the 3-D tiling
# (z=D, y=H, x=W) with size_hints={'z':D,'y':H,'x':W}.
#
# Inductor's convert_tiling_to_3d() needs TWO competing 2-D tiling candidates
# where the y-dims have a divisibility relationship.  The (0,2,1) permute
# creates both candidates from a single op.

def bench_3d_basic(warmup, iters, device):
    """z = x.permute(0,2,1) + y  — simplest 3-D kernel."""
    _section(
        "3D basic permute  —  z = x.permute(0,2,1) + y",
        "XBLOCK=W, YBLOCK=H, ZBLOCK=D  (W is innermost, D is outermost)",
    )

    def _fn(x, y):
        return x.permute(0, 2, 1) + y

    shapes = [
        # (name,  D,  H,   W)   — x is (D,H,W), y is (D,W,H)
        ("3D  4×  32×  64",   4,  32,   64),
        ("3D  4×  64× 128",   4,  64,  128),
        ("3D  8×  32×  64",   8,  32,   64),
        ("3D  8×  64× 128",   8,  64,  128),
        ("3D  8× 128× 256",   8, 128,  256),
        ("3D 16×  32×  64",  16,  32,   64),
        ("3D 16×  64× 128",  16,  64,  128),
        ("3D 16× 128× 256",  16, 128,  256),
        ("3D 32×  64× 128",  32,  64,  128),
        ("3D 32× 128× 256",  32, 128,  256),
        ("3D 64×  64× 128",  64,  64,  128),
        ("3D 64× 128× 256",  64, 128,  256),
    ]

    for name, D, H, W in shapes:
        _reset_cache()
        x = torch.randn(D, H, W, device=device)
        y = torch.randn(D, W, H, device=device)
        _bench(name, _fn, [x, y], warmup, iters)


def bench_3d_fused_act(warmup, iters, device):
    """z = relu(x.permute(0,2,1)) + y * scale  — fused activation."""
    _section(
        "3D fused activation  —  relu(x.permute(0,2,1)) + y * scale",
        "Tests compute+BW balance in 3-D regime (relu = fast op)",
    )

    def _fn(x, y, scale):
        return torch.relu(x.permute(0, 2, 1)) + y * scale

    shapes = [
        ("3D relu  8×  64× 128",   8,  64,  128),
        ("3D relu 16×  64× 128",  16,  64,  128),
        ("3D relu 16× 128× 256",  16, 128,  256),
        ("3D relu 32×  64× 128",  32,  64,  128),
        ("3D relu 32× 128× 256",  32, 128,  256),
        ("3D relu 64×  64× 128",  64,  64,  128),
    ]

    for name, D, H, W in shapes:
        _reset_cache()
        x     = torch.randn(D, H, W, device=device)
        y     = torch.randn(D, W, H, device=device)
        scale = torch.randn(D, W, H, device=device)
        _bench(name, _fn, [x, y, scale], warmup, iters)


def bench_3d_gelu(warmup, iters, device):
    """z = gelu(x.permute(0,2,1)) + y  — medium compute path."""
    _section(
        "3D GELU  —  gelu(x.permute(0,2,1)) + y",
        "Tests medium-compute heuristics (gelu = medium op) in 3-D regime",
    )

    def _fn(x, y):
        return torch.nn.functional.gelu(x.permute(0, 2, 1)) + y

    shapes = [
        ("3D gelu  8×  64× 128",   8,  64,  128),
        ("3D gelu 16×  64× 128",  16,  64,  128),
        ("3D gelu 16× 128× 256",  16, 128,  256),
        ("3D gelu 32×  64× 256",  32,  64,  256),
        ("3D gelu 64×  64× 128",  64,  64,  128),
    ]

    for name, D, H, W in shapes:
        _reset_cache()
        x = torch.randn(D, H, W, device=device)
        y = torch.randn(D, W, H, device=device)
        _bench(name, _fn, [x, y], warmup, iters)


def bench_3d_sigmoid(warmup, iters, device):
    """z = sigmoid(x.permute(0,2,1)) * y + bias  — high compute path."""
    _section(
        "3D sigmoid  —  sigmoid(x.permute(0,2,1)) * y + bias",
        "Tests high-compute heuristics (sigmoid = slow op) in 3-D regime",
    )

    def _fn(x, y, bias):
        return torch.sigmoid(x.permute(0, 2, 1)) * y + bias

    shapes = [
        ("3D sig  8×  64× 128",   8,  64,  128),
        ("3D sig 16×  64× 128",  16,  64,  128),
        ("3D sig 32×  64× 256",  32,  64,  256),
        ("3D sig 64×  64× 128",  64,  64,  128),
    ]

    for name, D, H, W in shapes:
        _reset_cache()
        x    = torch.randn(D, H, W, device=device)
        y    = torch.randn(D, W, H, device=device)
        bias = torch.randn(D, W, H, device=device)
        _bench(name, _fn, [x, y, bias], warmup, iters)


def bench_3d_cubic(warmup, iters, device):
    """Cubic tensors  z = x.permute(0,2,1) + y  — equal dims."""
    _section(
        "3D cubic  —  z = x.permute(0,2,1) + y  (D=H=W)",
        "All dims equal; tests whether model handles symmetric ZBLOCK/YBLOCK split",
    )

    def _fn(x, y):
        return x.permute(0, 2, 1) + y

    for N in [16, 32, 48, 64, 96, 128]:
        _reset_cache()
        x = torch.randn(N, N, N, device=device)
        y = torch.randn(N, N, N, device=device)  # permute(0,2,1) of NxNxN = NxNxN
        _bench(f"3D cube  {N}×{N}×{N}", _fn, [x, y], warmup, iters)


# ── environment diagnostics + static scoring preview ─────────────────────────

def _print_env_diagnostics():
    import os

    print()
    print("=" * 60)
    print("  bench3d — environment diagnostics")
    print("=" * 60)

    hip = getattr(torch.version, "hip", None)
    print(f"  torch.version.hip          : {hip!r}")
    if not hip:
        print(
            "  ⚠  ROCm not detected — pointwise heuristics WILL NOT run.\n"
            "     ZBLOCK kernels will still be generated by prefer_nd_tiling,\n"
            "     but scoring/validation tables are disabled on CUDA builds."
        )

    import torch._inductor.config as c
    print(f"  triton.prefer_nd_tiling    : {c.triton.prefer_nd_tiling}  ← must be True for ZBLOCK")
    print(f"  triton.max_tiles           : {c.triton.max_tiles}  ← must be ≥3 for ZBLOCK")

    def _flag(env, default="1"):
        v = os.environ.get(env, default)
        return f"{v!r}  ({env}={v!r}, default={default!r})"

    print(f"  POINTWISE_HEURISTICS       : {_flag('TORCHINDUCTOR_POINTWISE_HEURISTICS')}")
    print(f"  HEURISTICS_REAL_BENCH      : {_flag('TORCHINDUCTOR_HEURISTICS_REAL_BENCH')}")
    print(f"  HEURISTICS_VERBOSE         : {_flag('TORCHINDUCTOR_HEURISTICS_VERBOSE', '0')}")
    print(f"  HEURISTICS_TOP_N           : {_flag('TORCHINDUCTOR_HEURISTICS_TOP_N', '5')}")
    print(f"  HIP_LAUNCH_BLOCKING        : {_flag('HIP_LAUNCH_BLOCKING', '0')}")
    print(f"  HIP_VISIBLE_DEVICES        : {os.environ.get('HIP_VISIBLE_DEVICES', '(not set)')!r}")
    print("=" * 60)
    print()

    _print_3d_scoring_preview()


def _print_3d_scoring_preview():
    """Static top-5 predictions for representative 3-D problems (no GPU needed)."""
    try:
        from torch._inductor.codegen.triton_heuristics_pointwise import PointwiseHeuristics
    except ImportError:
        return

    # Dimensions in (xnumel, ynumel, znumel) = (W, H, D) order — innermost first.
    # This matches what _convert_to_pointwise_heuristics_metadata now produces
    # after the dimension-ordering fix.
    problems = [
        ("x.perm(0,2,1)+y  D=4,H=32,W=64",    (64,  32,   4)),
        ("x.perm(0,2,1)+y  D=8,H=64,W=128",   (128,  64,   8)),
        ("x.perm(0,2,1)+y  D=16,H=128,W=256", (256, 128,  16)),
        ("x.perm(0,2,1)+y  D=32,H=64,W=128",  (128,  64,  32)),
        ("cubic             N=32",             ( 32,  32,  32)),
    ]

    print("  [Static 3-D scoring preview — dimensions = (xnumel=W, ynumel=H, znumel=D)]")
    for label, dims in problems:
        pm = {'dimensions': dims, 'total_elements': dims[0]*dims[1]*dims[2],
              'warp_size': 64, 'max_threads_per_block': 1024}
        cfgs = PointwiseHeuristics.generate_all_candidate_configs(pm)
        if not cfgs:
            print(f"\n  {label}: no configs generated")
            continue
        top5 = PointwiseHeuristics.prune_configs(cfgs, pm, top_n=5)
        print(f"\n  {label}  ({len(cfgs)} candidates → top-5):")
        print(f"  {'Rk':3s}  {'Score':6s}  {'XBLK':5s} {'YBLK':5s} {'ZBLK':5s}"
              f"  {'nw':3s}  {'BW':5s}  {'Occ':5s}  {'ept':3s}  {'blks':>6s}")
        for i, c in enumerate(top5, 1):
            s   = PointwiseHeuristics.score_config(c, pm)
            bw  = PointwiseHeuristics.estimate_memory_bandwidth(c, pm)
            occ = PointwiseHeuristics.estimate_occupancy_impact(c, pm)
            xbp = c['XBLOCK'] * c.get('YBLOCK', 1) * c.get('ZBLOCK', 1)
            ept = max(1, xbp // (c['num_warps'] * 64))
            blks = max(1, pm['total_elements'] // xbp)
            zb   = c.get('ZBLOCK', '—')
            print(f"  #{i:2d}   {s:.4f}  {c['XBLOCK']:5d} {c.get('YBLOCK',0):5d} {str(zb):>5}"
                  f"  {c['num_warps']:3d}  {bw:.3f}  {occ:.3f}  {ept:3d}  {blks:>6d}")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3-D Triton pointwise (XBLOCK×YBLOCK×ZBLOCK) kernel benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Suites (all use x.permute(0,2,1) pattern to guarantee ZBLOCK):
  --basic    z = x.permute(0,2,1) + y
  --relu     z = relu(x.permute(0,2,1)) + y*scale
  --gelu     z = gelu(x.permute(0,2,1)) + y        (medium compute)
  --sigmoid  z = sigmoid(x.permute(0,2,1))*y+bias  (high compute)
  --cubic    z = x.permute(0,2,1) + y  with D=H=W

No flag → run all suites.
""")
    parser.add_argument("--warmup",  type=int, default=5,    help="warmup iterations")
    parser.add_argument("--iters",   type=int, default=20,   help="timed iterations")
    parser.add_argument("--device",  type=str, default="cuda")
    parser.add_argument("--basic",   action="store_true")
    parser.add_argument("--relu",    action="store_true")
    parser.add_argument("--gelu",    action="store_true")
    parser.add_argument("--sigmoid", action="store_true")
    parser.add_argument("--cubic",   action="store_true")
    args = parser.parse_args()

    _print_env_diagnostics()

    device = torch.device(args.device)
    print(f"Device : {torch.cuda.get_device_name(device)}")
    print(f"Warmup : {args.warmup}   Iters: {args.iters}")
    print(f"Tiling : prefer_nd_tiling=True  max_tiles=3  (ZBLOCK enabled)")

    run_all = not any([args.basic, args.relu, args.gelu, args.sigmoid, args.cubic])

    if run_all or args.basic:
        bench_3d_basic(args.warmup, args.iters, device)

    if run_all or args.relu:
        bench_3d_fused_act(args.warmup, args.iters, device)

    if run_all or args.gelu:
        bench_3d_gelu(args.warmup, args.iters, device)

    if run_all or args.sigmoid:
        bench_3d_sigmoid(args.warmup, args.iters, device)

    if run_all or args.cubic:
        bench_3d_cubic(args.warmup, args.iters, device)

    print("\nDone.")


if __name__ == "__main__":
    main()

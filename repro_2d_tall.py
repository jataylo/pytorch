#!/usr/bin/env python3
"""repro_2d_tall.py — isolated repro for the 2D tall (8192×64) heuristics issue.

Run with:
    rm -rf ~/.triton/cache/ && rm -rf /tmp/torchinductor_root/
    TORCHINDUCTOR_POINTWISE_HEURISTICS=1 \\
    TORCHINDUCTOR_HEURISTICS_REAL_BENCH=1 \\
    TORCHINDUCTOR_HEURISTICS_VERBOSE=1 \\
    HIP_LAUNCH_BLOCKING=1 HIP_VISIBLE_DEVICES=1 \\
    TORCHINDUCTOR_HEURISTICS_TOP_N=5 \\
    python repro_2d_tall.py 2>&1 | tee repro_2d_tall.log

Expected (after fixes):
  • [POINTWISE] Called for problem size: (8192, 64)   — or reconstruction notice
  • [HEURISTICS] N candidate configs / scoring table
  • [HEURISTICS] ✓ Chosen: ... ★ (from top-N)
  • Validation table printed

The script tests a single shape (no loop, no caching ambiguity) and also
tests 128×128 → 8192×64 sequentially to reproduce the cached-module path.
"""
import os
import shutil
import torch
import torch._dynamo

# ── helpers ───────────────────────────────────────────────────────────────────

def _clear_all_caches(label=""):
    tag = f"[{label}] " if label else ""

    # dynamo trace cache
    torch._dynamo.reset()

    # inductor in-memory module cache
    try:
        from torch._inductor.codecache import PyCodeCache
        n = len(PyCodeCache.modules_no_attr)
        PyCodeCache.cache_clear()
        print(f"{tag}PyCodeCache cleared: {n} → {len(PyCodeCache.modules_no_attr)} entries",
              flush=True)
    except Exception as e:
        print(f"{tag}PyCodeCache.cache_clear() error: {e}", flush=True)

    # sys.modules: purge compiled kernel entries
    # _reload_python_module() stores each kernel under
    # "torch._inductor.runtime.compile_tasks.<hash>" in sys.modules.
    # PyCodeCache.cache_clear() does not touch sys.modules, so the module
    # object stays alive and gets reused without re-running pointwise().
    import sys
    _prefix = "torch._inductor.runtime.compile_tasks."
    _to_del = [k for k in list(sys.modules) if k.startswith(_prefix)]
    for _k in _to_del:
        del sys.modules[_k]
    print(f"{tag}sys.modules: removed {len(_to_del)} compiled kernel entries", flush=True)

    # heuristics global state dicts (keyed by problem_key)
    try:
        import torch._inductor.runtime.triton_heuristics as _th
        n_top = len(_th._TOP_N_CONFIGS_FOR_SELECTION)
        n_val = len(_th._HEURISTICS_VALIDATION_DATA)
        _th._TOP_N_CONFIGS_FOR_SELECTION.clear()
        _th._HEURISTICS_VALIDATION_DATA.clear()
        _th._HEURISTICS_FULL_RANKED_HDICTS.clear()
        print(f"{tag}heuristics state cleared: top_n={n_top} val={n_val}", flush=True)
    except Exception as e:
        print(f"{tag}heuristics state clear error: {e}", flush=True)

    # disk caches
    shutil.rmtree("/tmp/torchinductor_root", ignore_errors=True)
    shutil.rmtree(os.path.expanduser("~/.triton/cache"), ignore_errors=True)
    print(f"{tag}disk caches deleted", flush=True)


def _run_shape(label, shape, compile_fn, device, warmup=3):
    x = torch.randn(shape, device=device)
    y = torch.randn((shape[1], shape[0]), device=device)
    w = torch.randn((shape[1], shape[0]), device=device)
    print(f"\n{'='*60}", flush=True)
    print(f"  Running {label}: x={shape}  output=({shape[1]},{shape[0]})", flush=True)
    print(f"{'='*60}", flush=True)
    for i in range(warmup):
        _ = compile_fn(x, y, w)
        torch.cuda.synchronize()
        print(f"  warmup {i+1}/{warmup} done", flush=True)
    print(f"  {label} complete", flush=True)


def _make_eager_fn():
    """Return a fresh function object each call so dynamo can't reuse a
    prior compilation via code-object caching across shape runs."""
    def fn(x, y, w):
        return x.transpose(-2, -1) + y * w
    return fn


# ── Test 1: Single fresh compile of 8192×64 ──────────────────────────────────

def test_single_fresh():
    """Compile 8192×64 from scratch with empty caches.  Scoring MUST run."""
    print("\n" + "█"*70, flush=True)
    print("  TEST 1: single fresh compile of (8192, 64)", flush=True)
    print("█"*70, flush=True)

    _clear_all_caches("test1")
    device = torch.device("cuda")
    compile_fn = torch.compile(_make_eager_fn(), dynamic=False)
    _run_shape("2D tall (8192×64)", (8192, 64), compile_fn, device)


# ── Test 2: 128×128 then 8192×64 (the bench2d.py multi-shape path) ───────────

def test_sequential():
    """Compile 128×128 then clear caches and compile 8192×64.
    Each shape uses a FRESH function object (_make_eager_fn()) so dynamo
    cannot reuse a prior compilation via code-object caching.
    Both should produce '★ (from top-N)' in the ✓ Chosen line.
    """
    print("\n" + "█"*70, flush=True)
    print("  TEST 2: sequential 128×128 → 8192×64  (multi-shape path)", flush=True)
    print("█"*70, flush=True)

    device = torch.device("cuda")

    # ── Shape 1: 128×128 ──────────────────────────────────────────────────
    _clear_all_caches("test2-shape1")
    compile_fn = torch.compile(_make_eager_fn(), dynamic=False)
    _run_shape("2D tiny (128×128)", (128, 128), compile_fn, device)

    # ── Shape 2: 8192×64 ──────────────────────────────────────────────────
    _clear_all_caches("test2-shape2")
    compile_fn = torch.compile(_make_eager_fn(), dynamic=False)
    _run_shape("2D tall (8192×64)", (8192, 64), compile_fn, device)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"torch.version.hip = {torch.version.hip!r}", flush=True)
    if not torch.version.hip:
        print("⚠  Not a ROCm build — heuristics will not run.", flush=True)
        return

    print(f"HIP_VISIBLE_DEVICES = {os.environ.get('HIP_VISIBLE_DEVICES', '(not set)')!r}",
          flush=True)
    print(f"TORCHINDUCTOR_HEURISTICS_REAL_BENCH = "
          f"{os.environ.get('TORCHINDUCTOR_HEURISTICS_REAL_BENCH', '0')!r}", flush=True)
    print(f"TORCHINDUCTOR_HEURISTICS_VERBOSE = "
          f"{os.environ.get('TORCHINDUCTOR_HEURISTICS_VERBOSE', '0')!r}", flush=True)
    print(f"TORCHINDUCTOR_HEURISTICS_TOP_N = "
          f"{os.environ.get('TORCHINDUCTOR_HEURISTICS_TOP_N', '5')!r}", flush=True)

    test_single_fresh()
    test_sequential()

    print("\n\nDone.", flush=True)


if __name__ == "__main__":
    main()


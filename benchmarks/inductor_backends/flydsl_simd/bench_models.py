"""Whole models: Inductor with FlyDSL pointwise/reduction kernels vs Triton.

For each model and backend this runs a fresh child process twice -- once with empty
Inductor, Triton and FlyDSL caches (cold compile) and once reusing them (warm) -- and
reports time to the first result, then the steady-state step time and the FlyDSL kernel
and fallback counts. A third run compiles with ``mode="reduce-overhead"`` (CUDA graphs),
which removes per-launch CPU cost and leaves the kernels' GPU time; ``ratio`` is that
time over Triton's. Matmuls, convolutions and attention are library calls under either
backend, so the difference is entirely the generated kernels. ``kern`` is Inductor's
generated kernel count, which should match between backends when fusion is unchanged.
The models live in models.py.

    python benchmarks/inductor_backends/flydsl_simd/bench_models.py [--only NAME ...] [--quick] [--repeats N] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile


CHILD = r"""
import json, sys, time
import torch
import torch._inductor.config as ic
from torch._dynamo.utils import counters
from torch._inductor import metrics

name, backend, mode, here = sys.argv[1:5]
sys.path.insert(0, here)
from models import ALL_MODELS as MODELS

ic.flydsl.simd = backend == "flydsl"
mode = None if mode == "default" else mode
torch.manual_seed(0)
run = torch.compile(MODELS[name](), mode=mode)
compiled = run


def run():
    if mode is not None:
        torch.compiler.cudagraph_mark_step_begin()
    return compiled()


t0 = time.perf_counter()
run()
torch.cuda.synchronize()
first = time.perf_counter() - t0
for _ in range(5):
    run()
torch.cuda.synchronize()
best = float("inf")
for _ in range(5):
    t0 = time.perf_counter()
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    best = min(best, (time.perf_counter() - t0) / 10)
c = counters["flydsl_simd"]
fallbacks = {k[len("fallback: "):]: v for k, v in c.items() if k.startswith("fallback: ")}
print("RESULT " + json.dumps({"first_s": first, "step_ms": best * 1e3, "flydsl": c["flydsl_kernels"],
                             "triton": c["triton_kernels"], "fallbacks": fallbacks,
                             "kernels": metrics.generated_kernel_count}))
"""

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models import ALL_MODELS, MODELS


def child(name, backend, env, cwd, mode="default"):
    # Run outside the repository so a source checkout's torch/ does not shadow the build.
    out = subprocess.run(
        [sys.executable, "-c", CHILD, name, backend, mode, HERE],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise RuntimeError(f"{name}/{backend} failed:\n{out.stderr[-4000:]}")


def run_model(name, quick, gpu, repeats=1):
    """Both backends for one model, on one GPU; returns (rows, printed lines).

    With ``repeats`` > 1 the CUDA-graph step is timed that many times per backend, in
    fresh processes alternating between the backends, and reported as the median with
    the spread (max - min): a single run of a 10-30 ms step varies by up to ~0.8 ms.
    """
    import contextlib
    import statistics

    rows, lines = [], []
    with contextlib.ExitStack() as stack:
        envs, first, graphs, failed = {}, {}, {"triton": [], "flydsl": []}, {}
        for backend in ("triton", "flydsl"):
            d = stack.enter_context(tempfile.TemporaryDirectory())
            env = dict(os.environ)
            env.update(
                TORCHINDUCTOR_CACHE_DIR=os.path.join(d, "inductor"),
                TRITON_CACHE_DIR=os.path.join(d, "triton"),
                FLYDSL_RUNTIME_CACHE_DIR=os.path.join(d, "flydsl"),
            )
            if gpu is not None:
                env["HIP_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            envs[backend] = (env, d)
            try:
                cold = child(name, backend, env, d)
                warm = cold if quick else child(name, backend, env, d)
                first[backend] = (cold, warm)
            except RuntimeError as e:
                failed[backend] = str(e)
        for _ in range(repeats):
            for backend in ("triton", "flydsl"):
                if backend in failed:
                    continue
                env, d = envs[backend]
                try:
                    graphs[backend].append(
                        child(name, backend, env, d, "reduce-overhead")["step_ms"]
                    )
                except RuntimeError as e:
                    failed[backend] = str(e)
    base = None
    for backend in ("triton", "flydsl"):
        if backend in failed:
            lines.append(f"{name:18s} {backend:7s} FAILED: {failed[backend][-300:]}")
            rows.append(
                {"model": name, "backend": backend, "error": failed[backend][-4000:]}
            )
            continue
        cold, warm = first[backend]
        g = graphs[backend]
        row = {
            "model": name,
            "backend": backend,
            "cold_s": cold["first_s"],
            "warm_s": warm["first_s"],
            "step_ms": min(cold["step_ms"], warm["step_ms"]),
            "graph_step_ms": statistics.median(g),
            "graph_spread_ms": max(g) - min(g),
            "graph_runs_ms": g,
            "kernels": cold["kernels"],
            "flydsl": cold["flydsl"],
            "triton": cold["triton"],
            "fallbacks": cold["fallbacks"],
        }
        rows.append(row)
        if backend == "triton":
            base = row
        ratio = row["graph_step_ms"] / base["graph_step_ms"] if base else float("nan")
        kern = (
            ""
            if backend == "triton"
            else f"flydsl={row['flydsl']} triton={row['triton']} {row['fallbacks'] or ''}"
        )
        spread = f" ±{row['graph_spread_ms'] / 2:.2f}" if repeats > 1 else ""
        lines.append(
            f"{name:18s} {backend:7s} {row['cold_s']:7.1f} {row['warm_s']:7.1f} {row['step_ms']:8.2f}"
            f" {row['graph_step_ms']:9.2f} {ratio:6.3f} {row['kernels']:5d}  {kern}{spread}"
        )
    return rows, lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*")
    p.add_argument("--json")
    p.add_argument(
        "--quick",
        action="store_true",
        help="skip the warm-start run; time steps and CUDA graphs only",
    )
    p.add_argument(
        "--gpus",
        type=lambda s: [int(g) for g in s.split(",")],
        help="run models in parallel, one per listed GPU (both backends of a model share one)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="CUDA-graph timings per backend (fresh processes, alternating); median reported",
    )
    a = p.parse_args()
    names = [n for n in (ALL_MODELS if a.only else MODELS) if not a.only or n in a.only]
    print(
        f"{'model':18s} {'backend':7s} {'cold s':>7s} {'warm s':>7s} {'step ms':>8s} {'graph ms':>9s}"
        f" {'ratio':>6s} {'kern':>5s}  coverage",
        flush=True,
    )
    results = {}
    if a.gpus:
        import queue
        from concurrent.futures import ThreadPoolExecutor

        free = queue.Queue()
        for g in a.gpus:
            free.put(g)

        def job(name):
            g = free.get()
            try:
                out = run_model(name, a.quick, g, a.repeats)
            finally:
                free.put(g)
            print("\n".join(out[1]), flush=True)
            return out

        with ThreadPoolExecutor(len(a.gpus)) as ex:
            results.update(zip(names, ex.map(job, names)))
    else:
        for name in names:
            results[name] = run_model(name, a.quick, None, a.repeats)
            print("\n".join(results[name][1]), flush=True)
    rows = [r for n in names for r in results[n][0]]
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()

"""Accumulating-scatter benchmark suite: stock Inductor vs partitioned scatter (PS)
vs the scatter_preagg template, over many scatter ops, contention ratios
(tokens / destinations), key distributions, inner sizes and dtypes.

One worker process runs one (op, shard) job. For each shape it compiles every
mode, then times the modes interleaved on the same GPU, and appends one JSON
line per (shape, mode); finished shapes are skipped, so runs resume.

  python suite.py --out results/suite.jsonl [--quick] [--gpus 8] [--modes base ps preagg]
  python suite.py --worker --op index_add_dim0 --out ...   (one job)
"""

import argparse
import functools
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
import types
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch


aten = torch.ops.aten
DEV = "cuda"
MODES = ("base", "ps", "preagg", "both")
ELEMS = 1 << 28  # cap on tokens * inner elements per input


@dataclass(frozen=True)
class Shape:
    op: str
    n: int  # tokens (writes per inner element)
    v: int  # destinations along the scattered dim(s)
    d: int  # inner size
    dist: str
    dtype: str
    k: int = 0  # op-specific parameter (MoE top-k, upsample scale)

    @property
    def ratio(self) -> float:
        return self.n / self.v

    def key(self, mode: str) -> str:
        return json.dumps({**asdict(self), "mode": mode}, sort_keys=True)


# --------------------------------------------------------------------- inputs


def make_keys(n: int, v: int, dist: str, gen: torch.Generator) -> torch.Tensor:
    if dist == "uniform":
        return torch.randint(0, v, (n,), device=DEV, generator=gen)
    if dist == "hot50":
        tail = torch.randint(0, v, (n,), device=DEV, generator=gen)
        hot = torch.rand(n, device=DEV, generator=gen) < 0.5
        return torch.where(hot, 0, tail)
    if dist.startswith("zipf"):
        s = float(dist[4:])
        p = torch.arange(1, v + 1, device=DEV, dtype=torch.float64).pow(-s)
        cdf = torch.cumsum(p, 0)
        u = torch.rand(n, device=DEV, dtype=torch.float64, generator=gen) * cdf[-1]
        return torch.searchsorted(cdf, u).clamp_(max=v - 1)
    raise ValueError(dist)


def _randn(*size, dtype, gen):
    return torch.randn(*size, device=DEV, generator=gen).to(dtype)


def build(s: Shape, gen: torch.Generator):
    """(fn, args) for one shape; fn derives output dtype from its float inputs."""
    dt = getattr(torch, s.dtype)
    n, v, d = s.n, s.v, s.d
    op = s.op
    if op in ("upsample_nearest_bwd", "upsample_bilinear_bwd"):
        side = int(math.isqrt(v))
        out_side = side * s.k
        g = _randn(1, d, out_side, out_side, dtype=dt, gen=gen)
        if op == "upsample_nearest_bwd":
            return (
                lambda g: aten.upsample_nearest2d_backward(
                    g, [out_side, out_side], [1, d, side, side], float(s.k), float(s.k)
                ),
                [g],
            )
        return (
            lambda g: aten.upsample_bilinear2d_backward(
                g,
                [out_side, out_side],
                [1, d, side, side],
                False,
                float(s.k),
                float(s.k),
            ),
            [g],
        )
    if op == "moe_combine":
        tok = torch.arange(v, device=DEV).repeat_interleave(s.k)
        tok = tok[torch.randperm(tok.numel(), device=DEV, generator=gen)]
        y = _randn(tok.numel(), d, dtype=dt, gen=gen)
        gate = torch.rand(tok.numel(), device=DEV, generator=gen).to(dt)
        return (
            lambda tok, y, gate: y.new_zeros(v, d).index_add_(
                0, tok, y * gate[:, None]
            ),
            [tok, y, gate],
        )
    if op == "gnn_scatter_add":
        perm = torch.randperm(v, device=DEV, generator=gen)
        dst = perm[make_keys(n, v, s.dist, gen)]
        src = torch.randint(0, v, (n,), device=DEV, generator=gen)
        h = _randn(v, d, dtype=dt, gen=gen)
        ew = torch.rand(n, device=DEV, generator=gen).to(dt)
        return (
            lambda h, src, dst, ew: h.new_zeros(v, d).index_add_(
                0, dst, h[src] * ew[:, None]
            ),
            [h, src, dst, ew],
        )

    idx = make_keys(n, v, s.dist, gen)
    if op == "llm_emb_bwd":
        idx = torch.randperm(v, device=DEV, generator=gen)[idx]
        op = "emb_bwd"
    if op == "hist_int32":
        ones = torch.ones(n, device=DEV, dtype=torch.int32)
        return (
            lambda idx, x: x.new_zeros(v).index_put_((idx,), x, accumulate=True),
            [idx, ones],
        )
    if op in ("index_add_dim1", "scatter_add_dim1"):
        xt = _randn(d, n, dtype=dt, gen=gen)
        if op == "index_add_dim1":
            return lambda idx, xt: xt.new_zeros(d, v).index_add_(1, idx, xt), [idx, xt]
        return (
            lambda idx, xt: xt.new_zeros(d, v).scatter_add_(
                1, idx[None, :].expand(d, -1), xt
            ),
            [idx, xt],
        )

    x = _randn(n, d, dtype=dt, gen=gen)
    if op == "index_put":
        return (
            lambda idx, x: x.new_zeros(v, d).index_put_((idx,), x, accumulate=True),
            [idx, x],
        )
    if op == "index_put_2d":
        v1 = 16
        v0 = -(-v // v1)
        return (
            lambda i0, i1, x: x.new_zeros(v0, v1, d).index_put_(
                (i0, i1), x, accumulate=True
            ),
            [idx // v1, idx % v1, x],
        )
    if op == "index_add_dim0":
        return lambda idx, x: x.new_zeros(v, d).index_add_(0, idx, x), [idx, x]
    if op == "scatter_add_dim0":
        return (
            lambda idx, x: x.new_zeros(v, d).scatter_add_(
                0, idx[:, None].expand(-1, d), x
            ),
            [idx, x],
        )
    if op in ("scatter_reduce_self", "scatter_reduce_noself"):
        base = _randn(v, d, dtype=dt, gen=gen)
        include_self = op == "scatter_reduce_self"
        return (
            lambda base, idx, x: base.scatter_reduce(
                0, idx[:, None].expand(-1, d), x, "sum", include_self=include_self
            ),
            [base, idx, x],
        )
    if op == "scatter_reduce_add":
        return (
            lambda idx, x: x.new_zeros(v, d).scatter_(
                0, idx[:, None].expand(-1, d), x, reduce="add"
            ),
            [idx, x],
        )
    if op in ("emb_bwd", "emb_bwd_pad"):
        pad = 0 if op == "emb_bwd_pad" else -1
        return (
            lambda x, idx: aten.embedding_dense_backward(x, idx, v, pad, False),
            [x, idx],
        )
    if op == "masked_index_put":
        mask = (idx % 5 != 0)[:, None]
        return (
            lambda mask, idx, x: aten._unsafe_masked_index_put_accumulate(
                x.new_zeros(v, d), mask, [idx], x
            ),
            [mask, idx, x],
        )
    if op == "fused_producer":
        w = _randn(d, dtype=dt, gen=gen)
        return (
            lambda idx, x, w: x.new_zeros(v, d).index_add_(0, idx, x * w + 1.0),
            [idx, x, w],
        )
    raise ValueError(op)


# --------------------------------------------------------------------- shapes

GRID_OPS = (
    "index_put",
    "index_put_2d",
    "index_add_dim0",
    "index_add_dim1",
    "scatter_add_dim0",
    "scatter_add_dim1",
    "scatter_reduce_self",
    "scatter_reduce_noself",
    "scatter_reduce_add",
    "emb_bwd",
    "emb_bwd_pad",
    "masked_index_put",
    "fused_producer",
)
OPS = GRID_OPS + (
    "hist_int32",
    "upsample_nearest_bwd",
    "upsample_bilinear_bwd",
    "gnn_scatter_add",
    "moe_combine",
    "llm_emb_bwd",
)
LOWP = {
    "bfloat16": ("index_put", "index_add_dim0", "scatter_add_dim0", "emb_bwd"),
    "float16": ("index_add_dim0", "emb_bwd"),
}


def _grid(op, ratios, dists, ds, dtype, n_max=1 << 21):
    out = []
    for d in ds:
        n = min(n_max, ELEMS // d)
        for r in ratios:
            v = n // r
            if v < 1:
                continue
            out += [Shape(op, n, v, d, dist, dtype) for dist in dists]
    return out


def shapes_for(op: str, quick: bool) -> list[Shape]:
    if quick:
        ratios, dists, ds = (
            (1, 16, 256, 4096, 65536),
            ("uniform", "zipf1.0", "hot50"),
            (16, 256),
        )
        lowp_ds = (256,)
    else:
        ratios = tuple(4**i for i in range(9))
        dists = ("uniform", "zipf0.8", "zipf1.0", "zipf1.2", "hot50")
        ds = (1, 16, 64, 256, 1024)
        lowp_ds = (64, 256)
    if op in GRID_OPS:
        out = _grid(op, ratios, dists, ds, "float32")
        for dtype, ops in LOWP.items():
            if op in ops and (dtype == "bfloat16" or not quick):
                out += _grid(op, ratios, dists, lowp_ds, dtype)
        if not quick:
            out += _grid(op, ratios, dists, (64,), "float32", n_max=1 << 16)
        return out
    if op == "hist_int32":
        return _grid(op, ratios, dists, (1,), "int32")
    if op.startswith("upsample"):
        scales = (2, 4) if quick else (2, 4, 8)
        return [
            Shape(op, 65536 * k * k, 65536, d, "upsample", "float32", k)
            for k in scales
            for d in ((16,) if quick else (16, 64))
        ]
    if op == "gnn_scatter_add":
        cases = (
            [(1 << 21, 1 << 17), (1 << 22, 1 << 20)]
            if not quick
            else [(1 << 21, 1 << 17)]
        )
        return [
            Shape(op, e, v, d, dist, "float32")
            for e, v in cases
            for d in (64,) + (() if quick else (16, 256))
            for dist in ("zipf1.0", "uniform")
        ]
    if op == "moe_combine":
        return [
            Shape(op, t * k, t, d, "moe", dtype, k)
            for t in ((16384,) if quick else (8192, 32768))
            for k in ((2, 8) if quick else (2, 4, 8))
            for d in ((1024,) if quick else (1024, 4096))
            for dtype in ("bfloat16",) + (() if quick else ("float32",))
        ]
    if op == "llm_emb_bwd":
        return [
            Shape(op, n, 128256, d, "zipf1.0", dtype)
            for n in ((65536,) if quick else (16384, 65536))
            for d in ((4096,) if quick else (2048, 4096))
            for dtype in ("bfloat16",) + (() if quick else ("float32",))
        ]
    raise ValueError(op)


# --------------------------------------------------------------------- worker


def mode_patch(mode: str):
    """Inductor config for a benchmark mode, applied while that mode compiles."""
    import torch._inductor.config as icfg

    return icfg.patch(
        partitioned_scatter_enabled=mode in ("ps", "both"),
        scatter_preagg_enabled=mode in ("preagg", "both"),
    )


def _fresh(fn, tag: str):
    """Copy of fn with its own code object, so each variant gets its own dynamo cache."""
    code = fn.__code__.replace(co_name=f"{fn.__code__.co_name}_{tag}")
    return types.FunctionType(
        code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__
    )


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    scale = ref.abs().max().item()
    return (out.double() - ref.double()).abs().max().item() / (scale or 1.0)


def run_shape(
    s: Shape,
    variants: dict[str, Callable[[], Any]],
    rounds: int = 5,
    rep: int = 20,
) -> dict[str, dict]:
    """
    Compile ``s`` once per variant (each under its context manager), check it
    against an fp64 eager reference, then time all variants interleaved over
    ``rounds`` rounds on the same GPU. ``ms`` is the fastest round, which is
    robust to other work briefly sharing the GPU.
    """
    from torch._dynamo.utils import counters
    from triton.testing import do_bench

    torch._dynamo.reset()
    gen = torch.Generator(device=DEV).manual_seed(0)
    fn, args = build(s, gen)
    ref = fn(*[a.double() if a.is_floating_point() else a for a in args])
    results: dict[str, dict] = {}
    compiled = {}
    for name, ctx in variants.items():
        counters.clear()
        try:
            f = torch.compile(_fresh(fn, name), fullgraph=True, dynamic=False)
            t0 = time.time()
            with ctx():
                out = f(*args)
            torch.cuda.synchronize()
        except Exception as e:
            results[name] = {
                "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            }
            traceback.print_exc()
            continue
        ind = counters["inductor"]
        path = (
            "preagg"
            if ind["scatter_preagg_applied"]
            else "ps"
            if ind["partitioned_scatter_applied"]
            else "default"
        )
        results[name] = {
            "err": _rel_err(out, ref),
            "path": path,
            "compile_s": round(time.time() - t0, 2),
        }
        compiled[name] = f
        del out
    del ref
    times: dict[str, list[float]] = {n: [] for n in compiled}
    for _ in range(rounds):
        for n, f in compiled.items():
            times[n].append(do_bench(lambda: f(*args), warmup=3, rep=rep))
    for n, ts in times.items():
        results[n]["ms"] = min(ts)
        results[n]["ms_median"] = statistics.median(ts)
    return results


def load_done(out: str) -> set[str]:
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def _append(out: str, recs: list[dict]) -> None:
    with open(out, "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in recs))


def _record(s: Shape, mode: str, **fields) -> dict:
    return {**asdict(s), "mode": mode, "ratio": s.ratio, "key": s.key(mode), **fields}


def worker(args) -> None:
    done = load_done(args.out)
    pending_path = f"{args.out}.pending.{args.op}.{args.shard}"
    shapes = shapes_for(args.op, args.quick)[args.shard :: args.nshards]
    # A shape still marked pending from an earlier attempt crashed the worker.
    if os.path.exists(pending_path):
        with open(pending_path) as f:
            crashed = f.read().strip()
        for s in shapes:
            if json.dumps(asdict(s), sort_keys=True) == crashed:
                _append(
                    args.out,
                    [
                        _record(s, m, error="worker crashed")
                        for m in args.modes
                        if s.key(m) not in done
                    ],
                )
                done |= {s.key(m) for m in args.modes}
    for s in shapes:
        modes = [m for m in args.modes if s.key(m) not in done]
        if not modes:
            continue
        with open(pending_path, "w") as f:
            f.write(json.dumps(asdict(s), sort_keys=True))
        try:
            res = run_shape(s, {m: functools.partial(mode_patch, m) for m in modes})
        except Exception as e:
            traceback.print_exc()
            err = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            res = {m: {"error": err} for m in modes}
        _append(args.out, [_record(s, m, **res[m]) for m in modes])
        torch.cuda.empty_cache()
    if os.path.exists(pending_path):
        os.remove(pending_path)


def launch(args) -> None:
    """Run every (op, shard) job, one GPU per worker, retrying crashed jobs."""
    ops = args.ops or OPS
    jobs = [(op, shard) for op in ops for shard in range(args.nshards)]
    attempts = dict.fromkeys(jobs, 0)
    queue = list(jobs)
    running: dict[int, tuple] = {}
    env_var = "HIP_VISIBLE_DEVICES" if torch.version.hip else "CUDA_VISIBLE_DEVICES"
    gpus = args.gpu_ids or list(range(args.gpus))
    log_dir = f"{args.out}.logs"
    os.makedirs(log_dir, exist_ok=True)
    while queue or running:
        for gpu in gpus:
            if gpu not in running and queue:
                job = queue.pop(0)
                op, shard = job
                cmd = [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--out",
                    args.out,
                    "--op",
                    op,
                    "--shard",
                    str(shard),
                    "--nshards",
                    str(args.nshards),
                    "--modes",
                    *args.modes,
                ] + (["--quick"] if args.quick else [])
                with open(f"{log_dir}/{op}.{shard}.log", "a") as log:
                    proc = subprocess.Popen(
                        cmd,
                        env={**os.environ, env_var: str(gpu)},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                running[gpu] = (job, proc)
        time.sleep(1)
        for gpu, (job, proc) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[gpu]
            attempts[job] += 1
            if proc.returncode != 0 and attempts[job] < 4:
                queue.append(job)
            print(
                f"[{time.strftime('%H:%M:%S')}] {job} rc={proc.returncode} "
                f"queue={len(queue)} running={len(running)}",
                flush=True,
            )
    print(f"{len(load_done(args.out))} records in {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--op", choices=OPS)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=4)
    p.add_argument(
        "--modes", nargs="+", default=["base", "ps", "preagg"], choices=MODES
    )
    p.add_argument("--ops", nargs="+", choices=OPS)
    p.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    p.add_argument("--gpu-ids", type=int, nargs="+")
    p.add_argument("--list", action="store_true", help="print shape counts and exit")
    args = p.parse_args()
    if args.list:
        total = 0
        for op in args.ops or OPS:
            n = len(shapes_for(op, args.quick))
            total += n
            print(f"{op:24s} {n}")
        print(f"total shapes {total} x {len(args.modes)} modes")
        return
    worker(args) if args.worker else launch(args)


if __name__ == "__main__":
    main()

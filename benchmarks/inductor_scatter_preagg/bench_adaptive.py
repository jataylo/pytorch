"""Compare the per-block adaptive preagg prototype against regular Inductor,
partitioned scatter and gated preagg across index distributions.

  python bench_adaptive.py run --gpus 8 --out results/adaptive.jsonl
  python bench_adaptive.py summary results/adaptive.jsonl

Modes, all timed on identical inputs:
  base      preagg off, partitioned scatter off (plain Inductor atomics)
  ps        preagg off, partitioned scatter on
  preagg    gated preagg on top of partitioned scatter (the current stack)
  ungated   preagg forced on every shape, always sorting
  adaptive  gate removed, each block picks sort or plain atomics; swept over
            scatter_preagg_adaptive_min_saved

Contention is measured from the indices as the average number of tokens that
share a token's destination row (sum of squared row counts over tokens): about
1 for unique rows, 1 + n / rows for uniform ones, and far higher when a few
rows are hot.
"""

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from collections import defaultdict

import torch


ROWS = (64, 4096, 262144, 1048576)
DS = (32, 128, 512)
TOKENS = (262144, 1048576, 4194304)
MAX_VALUE_BYTES = 1 << 30
DISTS = (
    "uniform",
    "unique",
    "sorted",
    "hot50",
    "zipf",
    "fewhot",
    "mixed",
    "runs32",
)
MIN_SAVED = (1 / 16, 1 / 8)
MODES = {
    "base": dict(scatter_preagg_enabled=False, partitioned_scatter_enabled=False),
    "ps": dict(scatter_preagg_enabled=False, partitioned_scatter_enabled=True),
    "preagg": dict(
        scatter_preagg_enabled=True,
        partitioned_scatter_enabled=True,
        scatter_preagg_adaptive=False,
    ),
    "ungated": dict(
        scatter_preagg_enabled=True,
        scatter_preagg_force=True,
        scatter_preagg_adaptive=False,
    ),
    **{
        f"adaptive_{s:g}": dict(
            scatter_preagg_enabled=True,
            partitioned_scatter_enabled=True,
            scatter_preagg_adaptive=True,
            scatter_preagg_adaptive_min_saved=s,
        )
        for s in MIN_SAVED
    },
}


def make_index(dist, n, rows, gen):
    dev = "cuda"

    def uniform(k):
        return torch.randint(0, rows, (k,), device=dev, generator=gen)

    def hot_rows(k):
        return torch.randperm(rows, device=dev, generator=gen)[: min(k, rows)]

    if dist == "uniform":
        return uniform(n)
    if dist == "unique":
        # A permutation when n <= rows, otherwise every row equally often.
        base = torch.arange(n, device=dev) % rows
        return base[torch.randperm(n, device=dev, generator=gen)]
    if dist == "sorted":
        return uniform(n).sort().values
    if dist == "hot50":
        hot = torch.rand(n, device=dev, generator=gen) < 0.5
        return torch.where(hot, hot_rows(1)[0], uniform(n))
    if dist == "zipf":
        ranks = torch.arange(1, rows + 1, device=dev, dtype=torch.float64)
        probs = ranks.pow(-1.1)
        sample = torch.multinomial(probs, n, replacement=True, generator=gen)
        return torch.randperm(rows, device=dev, generator=gen)[sample]
    if dist == "fewhot":
        hot = hot_rows(10)
        pick = hot[torch.randint(0, len(hot), (n,), device=dev, generator=gen)]
        mask = torch.rand(n, device=dev, generator=gen) < 0.2
        return torch.where(mask, pick, uniform(n))
    if dist == "mixed":
        hot = hot_rows(8)
        first = hot[torch.randint(0, len(hot), (n // 2,), device=dev, generator=gen)]
        return torch.cat([first, uniform(n - n // 2)])
    if dist == "runs32":
        return uniform((n + 31) // 32).repeat_interleave(32)[:n]
    raise ValueError(dist)


def _index_put(x, i, v):
    return torch.index_put(x, (i,), v, accumulate=True)


def _scatter_add(x, i, v):
    return x.scatter_add(0, i[:, None].expand(-1, v.shape[1]), v)


def _embedding_bwd(x, i, v):
    return x + torch.ops.aten.embedding_dense_backward(v, i, x.shape[0], -1, False)


OPS = {
    "index_put": _index_put,
    "scatter_add": _scatter_add,
    "embedding": _embedding_bwd,
}


def cases():
    out = []
    for rows, d, n in itertools.product(ROWS, DS, TOKENS):
        if n * d * 4 > MAX_VALUE_BYTES:
            continue
        for dist in DISTS:
            out.append(("index_put", "float32", rows, d, n, dist))
    for dtype, rows, dist in itertools.product(
        ("bfloat16", "float16"), (64, 4096, 262144), DISTS
    ):
        out.append(("index_put", dtype, rows, 128, 1048576, dist))
    for op, rows, dist in itertools.product(
        ("scatter_add", "embedding"), (4096, 262144), ("uniform", "zipf", "hot50")
    ):
        out.append((op, "float32", rows, 128, 1048576, dist))
    return out


def run_case(op, dtype, rows, d, n, dist):
    import torch._dynamo
    from torch._dynamo.utils import counters
    from torch._inductor import config
    from triton.testing import do_bench

    gen = torch.Generator(device="cuda").manual_seed(hash((rows, d, n, dist)) % 2**31)
    dt = getattr(torch, dtype)
    x = torch.zeros(rows, d, device="cuda", dtype=dt)
    idx = make_index(dist, n, rows, gen)
    v = torch.randn(n, d, device="cuda", dtype=dt, generator=gen)
    fn = OPS[op]
    ref = fn(x.double(), idx, v.double())
    scale = ref.abs().amax().clamp_min(1.0).item()
    counts = torch.bincount(idx, minlength=rows).double()
    rec = dict(
        op=op,
        dtype=dtype,
        rows=rows,
        d=d,
        n=n,
        dist=dist,
        contention=(counts.square().sum() / n).item(),
        times={},
        err={},
        applied={},
    )
    for mode, patch in MODES.items():
        torch._dynamo.reset()
        counters.clear()
        try:
            with config.patch(**patch):
                compiled = torch.compile(fn)
                out = compiled(x, idx, v)
                rec["times"][mode] = do_bench(
                    lambda: compiled(x, idx, v), warmup=10, rep=50
                )
            rec["err"][mode] = (out.double() - ref).abs().amax().item() / scale
            rec["applied"][mode] = counters["inductor"]["scatter_preagg_applied"]
        except Exception as e:
            print(f"{rec['op']} {dtype} {rows} {d} {n} {dist} {mode}: {e!r:.200}")
    return rec


def worker(shard, nshards, out):
    for case in cases()[shard::nshards]:
        rec = run_case(*case)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        torch.cuda.empty_cache()


def run(args):
    env_var = "HIP_VISIBLE_DEVICES" if torch.version.hip else "CUDA_VISIBLE_DEVICES"
    gpus = args.gpu_ids.split(",") if args.gpu_ids else map(str, range(args.gpus))
    gpus = list(gpus)
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, "worker", str(i), str(len(gpus)), args.out],
            env={
                **os.environ,
                env_var: gpu,
                "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
            },
        )
        for i, gpu in enumerate(gpus)
    ]
    for p in procs:
        p.wait()


def _geomean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def _label(r):
    return (
        f"{r['op']} {r['dtype']} rows={r['rows']} d={r['d']} n={r['n']} "
        f"{r['dist']} (contention {r['contention']:.3g})"
    )


CONTENTION_BUCKETS = (
    (1.5, "~1"),
    (8, "1.5-8"),
    (64, "8-64"),
    (1024, "64-1K"),
    (16384, "1K-16K"),
)


def _bucket(contention):
    for hi, name in CONTENTION_BUCKETS:
        if contention < hi:
            return name
    return ">=16K"


def summary(args):
    with open(args.results) as f:
        rows = [json.loads(line) for line in f]
    modes = [m for m in MODES if m not in ("base", "ps")]
    groups = defaultdict(list)
    for r in rows:
        if all(m in r["times"] for m in MODES):
            groups[(r["op"], r["dtype"])].append(r)

    def best(r):
        return min(r["times"]["base"], r["times"]["ps"])

    for (op, dtype), sel in sorted(groups.items()):
        print(f"\n== {op} {dtype}: {len(sel)} cases ==")
        for m in modes:
            speed = [(best(r) / r["times"][m], r) for r in sel]
            slow = [(s, r) for s, r in speed if s < 1 / 1.03]
            err = max(r["err"][m] for r in sel)
            print(
                f"{m:>16}: {_geomean([s for s, _ in speed]):.3f}x of best(base, ps), "
                f"{len(slow)} >3% slower, max err {err:.1e}"
            )
            if m.startswith("adaptive"):
                for s, r in sorted(slow, key=lambda t: t[0])[: args.worst]:
                    print(f"{'':>18}{s:.2f}  {_label(r)}")

    print("\nGeomean speedup over regular Inductor (base) by measured contention.")
    print("'worst' is adaptive's worst ratio to best(base, ps) in the bucket.")
    adaptive = f"adaptive_{MIN_SAVED[0]:g}"
    cols = ["ps", "preagg", "ungated", *(m for m in modes if m.startswith("adaptive"))]
    for (op, dtype), sel in sorted(groups.items()):
        by_bucket = defaultdict(list)
        for r in sel:
            by_bucket[_bucket(r["contention"])].append(r)
        print(f"\n{op} {dtype}\n")
        print("| contention | cases | base | " + " | ".join(cols) + " | worst |")
        print("|---" * (len(cols) + 4) + "|")
        for _, name in (*CONTENTION_BUCKETS, (None, ">=16K")):
            b = by_bucket.get(name)
            if not b:
                continue
            cells = [
                _geomean([r["times"]["base"] / r["times"][m] for r in b]) for m in cols
            ]
            worst = min(best(r) / r["times"][adaptive] for r in b)
            print(
                f"| {name} | {len(b)} | 1.00 | "
                + " | ".join(f"{c:.2f}" for c in cells)
                + f" | {worst:.2f} |"
            )


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
        sys.exit()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    p.add_argument("--gpu-ids", default="")
    p.add_argument("--out", required=True)
    p = sub.add_parser("summary")
    p.add_argument("results")
    p.add_argument("--worst", type=int, default=5)
    args = parser.parse_args()
    run(args) if args.cmd == "run" else summary(args)

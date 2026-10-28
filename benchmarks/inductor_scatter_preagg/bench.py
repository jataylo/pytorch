"""Compare scatter pre-aggregation (config.scatter_preagg_enabled) against
regular Inductor across ops, index distributions, shapes and dtypes, timed on
identical inputs.

  python bench.py run --gpus 8 --out results/gfx942.jsonl
  python bench.py summary results/gfx942.jsonl --worst 10 > results/gfx942_summary.md

Cases are sharded over GPUs and appended as JSON lines; a rerun skips the
cases already in --out, and a shard that crashes restarts past the case it
crashed on.

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
import time
import zlib
from collections import defaultdict

import torch


aten = torch.ops.aten
MAX_VALUE_ELEMS = 1 << 28
MAX_OUT_BYTES = 1 << 32
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
SKEWED = ("uniform", "sorted", "hot50", "zipf", "fewhot", "runs32")
MODES = {"base": False, "preagg": True}
FIELDS = ("op", "dtype", "rows", "d", "n", "dist", "dynamic")


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
        sample = torch.multinomial(ranks.pow(-1.1), n, replacement=True, generator=gen)
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


def _index_put_2d(x, i0, i1, v):
    return torch.index_put(x, (i0, i1), v, accumulate=True)


def _masked_index_put(x, m, i, v):
    return aten._unsafe_masked_index_put_accumulate(x, m, [i], v)


def _embedding(x, i, v):
    return x + aten.embedding_dense_backward(v, i, x.shape[0], -1, False)


def _embedding_pad(x, i, v):
    return x + aten.embedding_dense_backward(v, i, x.shape[0], 0, False)


def _scatter_reduce(x, i, v):
    return x.scatter_reduce(0, i[:, None].expand_as(v), v, "sum")


def _scatter_reduce_noself(x, i, v):
    return x.scatter_reduce(0, i[:, None].expand_as(v), v, "sum", include_self=False)


def _fused_producer(x, i, v, w):
    return x.index_add(0, i, v * w + 1.0)


def _moe_combine(x, i, v, gate):
    return x.index_add(0, i, v * gate[:, None])


def _gnn(x, i, h, src, w):
    # Gathered messages h[src], weighted per edge, summed into their rows.
    return x.index_add(0, i, h[src] * w[:, None])


OPS = {
    "hist": _index_put,
    "index_put": _index_put,
    "index_add": lambda x, i, v: x.index_add(0, i, v),
    "scatter_add": lambda x, i, v: x.scatter_add(0, i[:, None].expand_as(v), v),
    "scatter_reduce": _scatter_reduce,
    "scatter_reduce_noself": _scatter_reduce_noself,
    "index_add_dim1": lambda x, i, v: x.index_add(1, i, v),
    "scatter_add_dim1": lambda x, i, v: x.scatter_add(1, i[None, :].expand_as(v), v),
    "index_put_2d": _index_put_2d,
    "masked_index_put": _masked_index_put,
    "embedding": _embedding,
    "embedding_pad": _embedding_pad,
    "fused_producer": _fused_producer,
    "moe_combine": _moe_combine,
    "gnn": _gnn,
}


def build(case, gen):
    """(fn, args, destination rows) for one case."""
    op, rows, d, n = case["op"], case["rows"], case["d"], case["n"]
    dt = getattr(torch, case["dtype"])
    idx = make_index(case["dist"], n, rows, gen)

    def rand(*size):
        return torch.rand(*size, device="cuda", generator=gen).to(dt)

    def randn(*size):
        return torch.randn(*size, device="cuda", generator=gen).to(dt)

    x = torch.zeros(rows, d, device="cuda", dtype=dt)
    if dt.is_floating_point:
        v = randn(n, d)
    else:
        v = torch.randint(-5, 5, (n, d), device="cuda", dtype=dt, generator=gen)
    if op == "hist":
        args = [x.view(-1), idx, torch.ones(n, device="cuda", dtype=dt)]
    elif op in ("scatter_reduce", "scatter_reduce_noself"):
        args = [randn(rows, d), idx, v]
    elif op in ("index_add_dim1", "scatter_add_dim1"):
        args = [x.t().contiguous(), idx, v.t().contiguous()]
    elif op == "index_put_2d":
        x2 = torch.zeros(-(-rows // 16), 16, d, device="cuda", dtype=dt)
        args = [x2, idx // 16, idx % 16, v]
    elif op == "masked_index_put":
        args = [x, (idx % 5 != 0)[:, None], idx, v]
    elif op == "fused_producer":
        args = [x, idx, v, randn(d)]
    elif op == "moe_combine":
        args = [x, idx, v, rand(n)]
    elif op == "gnn":
        src = torch.randint(0, rows, (n,), device="cuda", generator=gen)
        args = [x, idx, randn(rows, d), src, rand(n)]
    else:
        args = [x, idx, v]
    return OPS[op], args, idx


def _case(op, dtype, rows, d, n, dist, dynamic=False):
    return dict(op=op, dtype=dtype, rows=rows, d=d, n=n, dist=dist, dynamic=dynamic)


def _fits(dtype, rows, d, n):
    itemsize = getattr(torch, dtype).itemsize
    return n * d <= MAX_VALUE_ELEMS and rows * d * max(itemsize, 4) <= MAX_OUT_BYTES


def cases():
    out = []
    # index_put over the full shape grid.
    for rows, d, n in itertools.product(
        (1, 64, 4096, 65536, 262144, 1048576),
        (1, 8, 32, 48, 128, 512),
        (65536, 262144, 1048576, 4194304),
    ):
        if _fits("float32", rows, d, n) and (d != 512 or n <= 262144):
            for dist in ("uniform",) if rows == 1 else DISTS:
                out.append(_case("index_put", "float32", rows, d, n, dist))
    for d, dist in itertools.product((32, 128), DISTS):
        out.append(_case("index_put", "float32", 4194304, d, 1048576, dist))
    # Every op the template covers, 32-bit.
    ops = (
        "index_add",
        "index_add_dim1",
        "scatter_add",
        "scatter_add_dim1",
        "scatter_reduce",
        "scatter_reduce_noself",
        "index_put_2d",
        "masked_index_put",
        "embedding",
        "embedding_pad",
        "fused_producer",
        "moe_combine",
        "gnn",
    )
    for op, rows, d, dist in itertools.product(
        ops, (64, 4096, 262144, 1048576), (32, 128), SKEWED
    ):
        out.append(_case(op, "float32", rows, d, 1048576, dist))
    # 16-bit outputs.
    for dtype, op, rows, d, dist in itertools.product(
        ("float16", "bfloat16"),
        ("index_put", "index_add", "scatter_add", "embedding", "fused_producer"),
        (64, 4096, 262144),
        (16, 64, 128, 512),
        SKEWED,
    ):
        out.append(_case(op, dtype, rows, d, 262144 if d == 512 else 1048576, dist))
    # Integer outputs: histograms and row sums.
    for dtype, (op, d), rows, dist in itertools.product(
        ("int32", "int64"),
        (("hist", 1), ("index_put", 32)),
        (64, 4096, 262144),
        ("uniform", "sorted", "hot50", "zipf"),
    ):
        out.append(_case(op, dtype, rows, d, 1048576, dist))
    # Dynamic shapes.
    for dtype, rows, dist in itertools.product(
        ("float32", "float16"), (4096, 262144), ("uniform", "sorted", "hot50")
    ):
        out.append(_case("index_put", dtype, rows, 128, 1048576, dist, dynamic=True))
    return out


def _key(r):
    return tuple(r.get(f, False) for f in FIELDS)


def run_case(case):
    import torch._dynamo
    from torch._dynamo.utils import counters
    from torch._inductor import config
    from triton.testing import do_bench

    seed = zlib.crc32(json.dumps(case, sort_keys=True).encode())
    gen = torch.Generator(device="cuda").manual_seed(seed)
    fn, args, dest = build(case, gen)
    exact = not args[0].dtype.is_floating_point
    ref = fn(*[a.double() if a.dtype.is_floating_point else a for a in args])
    scale = ref.abs().amax().clamp_min(1.0).item()
    counts = torch.bincount(dest, minlength=case["rows"]).double()
    rec = dict(
        case,
        contention=(counts.square().sum() / dest.numel()).item(),
        times={},
        err={},
        applied={},
    )
    for mode, enabled in MODES.items():
        torch._dynamo.reset()
        counters.clear()
        try:
            with config.patch(scatter_preagg_enabled=enabled):
                compiled = torch.compile(fn, dynamic=case["dynamic"] or None)
                out = compiled(*args)
                rec["times"][mode] = do_bench(
                    lambda: compiled(*args), warmup=10, rep=50
                )
            diff = (out.double() - ref.double()).abs().amax().item()
            rec["err"][mode] = diff if exact else diff / scale
            rec["applied"][mode] = counters["inductor"]["scatter_preagg_applied"]
        except Exception as e:
            print(f"{_key(rec)} {mode}: {e!r:.200}", flush=True)
    return rec


def _done(out):
    if not os.path.exists(out):
        return set()
    with open(out) as f:
        return {_key(json.loads(line)) for line in f if line.strip()}


def worker(shard, nshards, out):
    done = _done(out)
    marker = f"{out}.shard{shard}"
    if os.path.exists(marker):
        with open(marker) as f:
            crashed = json.load(f)
        if _key(crashed) not in done:
            with open(out, "a") as f:
                f.write(json.dumps(dict(crashed, crashed=True, times={})) + "\n")
            done.add(_key(crashed))
    for case in cases()[shard::nshards]:
        if _key(case) in done:
            continue
        with open(marker, "w") as f:
            json.dump(case, f)
        rec = run_case(case)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        torch.cuda.empty_cache()
    os.remove(marker)


def run(args):
    env_var = "HIP_VISIBLE_DEVICES" if torch.version.hip else "CUDA_VISIBLE_DEVICES"
    gpus = args.gpu_ids.split(",") if args.gpu_ids else map(str, range(args.gpus))
    gpus = list(gpus)

    def launch(i):
        return subprocess.Popen(
            [sys.executable, __file__, "worker", str(i), str(len(gpus)), args.out],
            env={
                **os.environ,
                env_var: gpus[i],
                "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
            },
        )

    procs = {i: launch(i) for i in range(len(gpus))}
    restarts = defaultdict(int)
    while procs:
        for i, p in list(procs.items()):
            if p.poll() is None:
                continue
            del procs[i]
            if p.returncode != 0 and restarts[i] < 5:
                restarts[i] += 1
                procs[i] = launch(i)
        time.sleep(5)


def _geomean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def _label(r):
    dyn = " dynamic" if r.get("dynamic") else ""
    return (
        f"{r['op']} {r['dtype']} rows={r['rows']} d={r['d']} n={r['n']} "
        f"{r['dist']}{dyn} (contention {r['contention']:.3g})"
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
    crashed = [r for r in rows if r.get("crashed")]
    rows = [r for r in rows if all(m in r["times"] for m in MODES)]
    for r in rows:
        r["speedup"] = r["times"]["base"] / r["times"]["preagg"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["op"], r["dtype"])].append(r)

    print("Speedup of preagg over regular Inductor (base) by measured contention;")
    print("'slower' counts cases below 0.97, errors are the max, base / preagg.")
    for (op, dtype), sel in sorted(groups.items()):
        by_bucket = defaultdict(list)
        for r in sel:
            by_bucket[_bucket(r["contention"])].append(r)
        buckets = [(name, by_bucket[name]) for _, name in CONTENTION_BUCKETS]
        buckets += [(">=16K", by_bucket[">=16K"]), ("all", sel)]
        print(f"\n{op} {dtype}\n")
        print("| contention | cases | geomean | worst | slower | max err |")
        print("|---|---|---|---|---|---|")
        for name, b in buckets:
            if not b:
                continue
            speedups = [r["speedup"] for r in b]
            errs = " / ".join(f"{max(r['err'][m] for r in b):.0e}" for m in MODES)
            print(
                f"| {name} | {len(b)} | {_geomean(speedups):.2f} | {min(speedups):.2f}"
                f" | {sum(x < 0.97 for x in speedups)} | {errs} |"
            )
        slow = sorted((r["speedup"], r) for r in sel if r["speedup"] < 0.97)
        for ratio, r in slow[: args.worst]:
            print(f"  {ratio:.2f}  {_label(r)}")

    print("\nOverall:\n")
    print("| dtype | cases | geomean | worst | < 0.95 |")
    print("|---|---|---|---|---|")
    by_dtype = defaultdict(list)
    for r in rows:
        by_dtype[r["dtype"]].append(r["speedup"])
        by_dtype["all"].append(r["speedup"])
    for dtype, speedups in sorted(by_dtype.items()):
        print(
            f"| {dtype} | {len(speedups)} | {_geomean(speedups):.2f}"
            f" | {min(speedups):.2f} | {sum(x < 0.95 for x in speedups)} |"
        )
    if crashed:
        print(f"\n{len(crashed)} crashed cases:")
        for r in crashed:
            print("  " + " ".join(str(r[f]) for f in FIELDS))


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
    p.add_argument("--worst", type=int, default=0)
    args = parser.parse_args()
    run(args) if args.cmd == "run" else summary(args)

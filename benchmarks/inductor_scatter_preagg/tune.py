"""Sweep template configs over a grid of index_put shapes with real index data,
then score the default heuristic against the best config per shape.

  python tune.py sweep --gpus 8 --out results/tune_gfx942.jsonl
  python tune.py fit results/tune_gfx942.jsonl

Autotuning cannot stand in for this: it benchmarks integer inputs as synthetic
data, while the best config depends on how the indices are distributed. The
sweep times every (max BLOCK_D, BLOCK_T) pair on uniform and hot (half on one
destination) indices; `fit` reports, per row width, the fixed configs closest
to the per-shape best, the default's geomean against it, and its worst shapes.
"""

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from unittest import mock

import torch


DS = (32, 64, 96, 128, 256, 512)
ROWS = (64, 1024, 16384, 131072)
DTYPES = ("float32", "bfloat16")
KINDS = ("uniform", "hot")
NUM_TOKENS = 1_000_000
CONFIGS = [
    (bd, bt)
    for bd, bt in itertools.product((16, 32, 64, 128), (32, 64, 128, 256, 512, 1024))
    if 1024 <= bd * bt <= 16384
]


def _index_put(x, i, v):
    return torch.index_put(x, (i,), v, accumulate=True)


def _time(x, idx, v, cfg):
    import torch._dynamo
    from torch._inductor import config
    from torch._inductor.heuristics.template import triton as heur
    from triton.testing import do_bench

    real = heur.BaseConfigHeuristic.get_scatter_preagg_configs

    def configs(
        self, num_tokens, num_inner, dtype, inner_contiguous=True, num_keys=None
    ):
        if cfg is None:
            return real(self, num_tokens, num_inner, dtype, inner_contiguous, num_keys)
        bd, bt = cfg
        c = self._scatter_preagg_config(num_tokens, num_inner, bd, bd * bt, 4)
        return [heur.ScatterPreaggConfig(bt, c.block_d, c.d_iters, 1, 4)]

    torch._dynamo.reset()
    with (
        mock.patch.object(
            heur.BaseConfigHeuristic, "get_scatter_preagg_configs", configs
        ),
        config.patch(
            scatter_preagg_enabled=True, scatter_preagg_force=True, fx_graph_cache=False
        ),
    ):
        fn = torch.compile(_index_put)
        fn(x.clone(), idx, v)
        return do_bench(lambda: fn(x, idx, v), warmup=10, rep=60)


def sweep_worker(shard, nshards, out):
    torch.manual_seed(0)
    shapes = list(itertools.product(DS, ROWS, DTYPES, KINDS))[shard::nshards]
    for d, rows, dtype, kind in shapes:
        dt = getattr(torch, dtype)
        x = torch.zeros(rows, d, device="cuda", dtype=dt)
        idx = torch.randint(0, rows, (NUM_TOKENS,), device="cuda")
        if kind == "hot":
            idx = torch.where(torch.rand(NUM_TOKENS, device="cuda") < 0.5, 0, idx)
        v = torch.randn(NUM_TOKENS, d, device="cuda", dtype=dt)
        times = {}
        for cfg in [None, *CONFIGS]:
            if cfg is not None and cfg[0] > max(16, d):
                continue
            try:
                times[str(cfg)] = _time(x, idx, v, cfg)
            except Exception as e:
                print(f"{d} {rows} {dtype} {kind} {cfg}: {type(e).__name__}")
        rec = dict(d=d, rows=rows, dtype=dtype, kind=kind, times=times)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")


def sweep(args):
    env_var = "HIP_VISIBLE_DEVICES" if torch.version.hip else "CUDA_VISIBLE_DEVICES"
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, "worker", str(i), str(args.gpus), args.out],
            env={
                **os.environ,
                env_var: str(i),
                "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
            },
        )
        for i in range(args.gpus)
    ]
    for p in procs:
        p.wait()


def _geomean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


def fit(args):
    with open(args.results) as f:
        rows = [json.loads(line) for line in f]

    def best(r):
        return min(v for k, v in r["times"].items() if k != "None")

    by_d = defaultdict(list)
    for r in rows:
        by_d[r["d"]].append(r)
    for d, sel in sorted(by_d.items()):
        scores = []
        for key in {k for r in sel for k in r["times"] if k != "None"}:
            if all(key in r["times"] for r in sel):
                scores.append((_geomean([best(r) / r["times"][key] for r in sel]), key))
        top = ", ".join(f"{k} {s:.2f}" for s, k in sorted(scores, reverse=True)[:3])
        dflt = _geomean([best(r) / r["times"]["None"] for r in sel])
        print(f"d={d:4d}: default {dflt:.2f} of best; best fixed: {top}")
    ratios = [(best(r) / r["times"]["None"], r) for r in rows]
    print(f"overall default {_geomean([s for s, _ in ratios]):.3f} of best")
    for s, r in sorted(ratios, key=lambda t: t[0])[:8]:
        print(f"  {s:.2f}  d={r['d']} rows={r['rows']} {r['dtype']} {r['kind']}")


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        sweep_worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
        sys.exit()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sweep")
    p.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    p.add_argument("--out", required=True)
    p = sub.add_parser("fit")
    p.add_argument("results")
    args = parser.parse_args()
    sweep(args) if args.cmd == "sweep" else fit(args)

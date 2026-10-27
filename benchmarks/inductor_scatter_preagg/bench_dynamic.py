"""
Fully dynamic accumulating scatters (every dim marked dynamic): plain atomics,
partitioned scatter, scatter_preagg, and both enabled.

    python benchmarks/inductor_scatter_preagg/bench_dynamic.py
"""

import torch
from torch._inductor import config
from triton.testing import do_bench

dev = "cuda"
MODES = {"base": (False, False), "ps": (True, False), "preagg": (False, True), "both": (True, True)}


def hot(n, v, frac):
    return torch.where(torch.rand(n, device=dev) < frac, torch.randint(0, 4, (n,), device=dev),
                       torch.randint(0, v, (n,), device=dev))


def bench(fn, args, ps, pa):
    torch._dynamo.reset()
    with config.patch(partitioned_scatter_enabled=ps, scatter_preagg_enabled=pa, fx_graph_cache=False):
        a = [x.clone() for x in args]
        for t in a:
            for d in range(t.dim()):
                torch._dynamo.mark_dynamic(t, d)
        cfn = torch.compile(fn, fullgraph=True)
        cfn(*a)
        return do_bench(lambda: cfn(*a), warmup=10, rep=60)


def ip(x, i, v):
    return x.index_put_((i,), v, accumulate=True)


def ia(x, i, v):
    return x.index_add_(0, i, v)


print(f"{'case':40s}" + "".join(f"{m:>9s}" for m in MODES) + "  pa/best(base,ps)")
for name, fn in (("index_put", ip), ("index_add", ia)):
    for n, rows, d, frac, dt in [
        (1 << 18, 1024, 16, 0.5, torch.float32),
        (1 << 18, 1024, 48, 0.5, torch.float32),
        (1 << 18, 1024, 64, 0.5, torch.float32),
        (1 << 18, 1024, 100, 0.5, torch.float32),
        (1 << 18, 4096, 128, 0.1, torch.float32),
        (1 << 18, 512, 256, 0.5, torch.float32),
        (1 << 18, 1024, 64, 0.5, torch.float16),
        (1 << 20, 32768, 8, 0.0, torch.float32),
    ]:
        args = [torch.zeros(rows, d, device=dev, dtype=dt), hot(n, rows, frac), torch.randn(n, d, device=dev, dtype=dt)]
        ms = {m: bench(fn, args, *f) for m, f in MODES.items()}
        best = min(ms["base"], ms["ps"])
        print(f"{name} n={n} rows={rows} d={d} hot={frac} {str(dt)[6:]:8s}"[:40].ljust(40)
              + "".join(f"{ms[m]:9.3f}" for m in MODES) + f"  {best / ms['preagg']:.2f}x")

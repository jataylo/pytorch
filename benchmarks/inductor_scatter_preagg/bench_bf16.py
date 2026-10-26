"""bf16 index_add: python bench_bf16.py <mode> <V> <D> <dist>
mode: base (eager fallback on ROCm) | ps | preagg;  dist: hot50 | uniform"""

import sys

import torch
import torch._inductor.config as icfg
from torch._dynamo.utils import counters
from triton.testing import do_bench

mode, V, D, dist = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
icfg.scatter_preagg_enabled = mode == "preagg"
icfg.partitioned_scatter_enabled = mode == "ps"

N = 2_479_833
torch.manual_seed(0)
idx = torch.randint(0, V, (N,), device="cuda")
if dist == "hot50":
    idx = torch.where(torch.rand(N, device="cuda") < 0.5, 0, idx)
x = torch.randn(N, D, device="cuda").bfloat16()


def f(a):
    return torch.zeros(V, D, device="cuda", dtype=torch.bfloat16).index_add_(0, idx, a)


c = torch.compile(f)
out = c(x)
ref = torch.zeros(V, D, device="cuda", dtype=torch.float64).index_add_(
    0, idx, x.double()
)
err = ((out.double() - ref).abs().max() / ref.abs().max()).item()
t = do_bench(lambda: c(x), warmup=10, rep=100)
applied = counters["inductor"]["scatter_preagg_applied"]
ps = counters["inductor"]["partitioned_scatter_applied"]
print(
    f"V={V:5d} D={D:3d} {dist:7s} {mode:6s} {t:7.3f} ms  err={err:.1e}  preagg={applied} ps={ps}"
)

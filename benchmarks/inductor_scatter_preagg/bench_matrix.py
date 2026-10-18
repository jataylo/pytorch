"""One config per process: python bench_matrix.py <mode> <op> <V> <dist> <D>
mode: base | ps | preagg
dist: hot50 (half the tokens on row 0, rest uniform) | zipf (p_k ~ 1/k) | uniform"""
import sys

import torch
import torch._inductor.config as icfg
from torch._dynamo.utils import counters
from triton.testing import do_bench

mode, op, V, dist, D = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
icfg.partitioned_scatter_enabled = mode == "ps"
icfg.scatter_preagg_enabled = mode == "preagg"

N = 2_479_833
torch.manual_seed(0)
if dist == "hot50":
    tail = torch.randint(1, V, (N,), device="cuda")
    idx = torch.where(torch.rand(N, device="cuda") < 0.5, 0, tail)
elif dist == "zipf":
    p = 1.0 / torch.arange(1, V + 1, device="cuda", dtype=torch.float64)
    idx = torch.multinomial(p / p.sum(), N, replacement=True)
else:
    idx = torch.randint(0, V, (N,), device="cuda")

if op == "emb_bwd":
    x = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    i32 = idx.int()

    def f(a):
        return torch.ops.aten.embedding_dense_backward(a, i32, V, -1, False)
else:
    x = torch.randn(N, D, device="cuda")

    def f(a):
        return torch.zeros(V, D, device="cuda").index_add_(0, idx, a)

ref = torch.zeros(V, D, device="cuda", dtype=torch.float64).index_add_(0, idx, x.double())
c = torch.compile(f, fullgraph=True)
out = c(x)
err = (out.double() - ref).abs().max().item() / ref.abs().max().item()
t = do_bench(lambda: c(x), warmup=10, rep=100)
applied = counters["inductor"]["scatter_preagg_applied"]
ps = counters["inductor"]["partitioned_scatter_applied"]
print(f"RESULT {op},{V},{dist},{D},{mode},{t:.3f},{err:.1e},preagg={applied},ps={ps}", flush=True)

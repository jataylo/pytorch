"""Embedding backward on a sliced bf16 activation (hot row 0):
python bench_smoke.py [preagg 0|1] [V]"""

import sys

import torch
import torch._inductor.config as icfg
from torch._dynamo.utils import counters
from triton.testing import do_bench

icfg.scatter_preagg_enabled = sys.argv[1] == "1" if len(sys.argv) > 1 else True
V = int(sys.argv[2]) if len(sys.argv) > 2 else 24
N, D = 2_479_833, 32

tail = torch.randint(1, V, (N,), device="cuda", dtype=torch.int32)
idx = torch.where(torch.rand(N, device="cuda") < 0.5, 0, tail).int()
x = torch.randn(N, 224, device="cuda", dtype=torch.bfloat16)


def f(i, a):
    return torch.ops.aten.embedding_dense_backward(a[:, 184:216], i, V, -1, False)


c = torch.compile(f, fullgraph=True)
out = c(idx, x)
ref = torch.zeros(V, D, device="cuda", dtype=torch.float64).index_add_(
    0, idx.long(), x[:, 184:216].double()
)
err = (out.double() - ref).abs().max().item() / ref.abs().max().item()
print({k: v for k, v in counters["inductor"].items() if "preagg" in k or "partitioned" in k})
print(
    f"V={V} preagg={icfg.scatter_preagg_enabled} rel_err={err:.2e} "
    f"time={do_bench(lambda: c(idx, x), warmup=5, rep=50):.3f} ms"
)

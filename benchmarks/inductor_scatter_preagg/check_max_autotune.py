"""Correctness of index_put(accumulate=True) with the template under max_autotune."""

import torch
import torch._inductor.config as icfg

icfg.scatter_preagg_enabled = True
icfg.max_autotune = True
torch.manual_seed(0)
N = 50_000
x = torch.randn(24, 64, device="cuda")
idx = torch.where(
    torch.rand(N, device="cuda") < 0.5, 0, torch.randint(0, 24, (N,), device="cuda")
)
v = torch.randn(N, 64, device="cuda")


def fn(x, i, v):
    return x.index_put_((i,), v, accumulate=True)


exp = fn(x.clone(), idx, v)
c = torch.compile(fn)
for call in range(2):
    out = c(x.clone(), idx, v)
    print(f"call {call}: max abs diff {(out - exp).abs().max().item():.2e}")

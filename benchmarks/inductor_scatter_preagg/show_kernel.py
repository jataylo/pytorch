"""Print the generated pre-aggregation template kernel and its launches."""

import torch
import torch._inductor.config as icfg
from torch._inductor.utils import run_and_get_code

icfg.scatter_preagg_enabled = True
V, D, N = 24, 32, 200_000
idx = torch.randint(0, V, (N,), device="cuda")
x = torch.randn(N, D, device="cuda")
f = torch.compile(lambda a, i: torch.zeros(V, D, device="cuda").index_add_(0, i, a * 2))
_, (code,) = run_and_get_code(f, x, idx)
start = code.index("@triton.jit\ndef triton_tem")
print(code[start : code.index("'''", start)])
print("=== wrapper calls ===")
for line in code.splitlines():
    if ".run(" in line:
        print(line.strip())

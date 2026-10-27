"""
Deterministic-mode scatters: the sorted scatter_preagg lowering vs Inductor's
eager fallback, with the non-deterministic atomic path for reference.

    python benchmarks/inductor_scatter_preagg/bench_deterministic.py
"""

import torch
from torch._inductor import config
from triton.testing import do_bench


dev = "cuda"


def hot(n, v):
    return torch.where(
        torch.rand(n, device=dev) < 0.5,
        torch.randint(0, 4, (n,), device=dev),
        torch.randint(0, v, (n,), device=dev),
    )


CASES = [
    (
        "index_add fp32 1Mx64->4096 hot",
        lambda x, i, w: x.index_add_(0, i, w),
        lambda: (
            torch.zeros(4096, 64, device=dev),
            hot(1 << 20, 4096),
            torch.randn(1 << 20, 64, device=dev),
        ),
    ),
    (
        "index_add fp32 1Mx64->4096 uniform",
        lambda x, i, w: x.index_add_(0, i, w),
        lambda: (
            torch.zeros(4096, 64, device=dev),
            torch.randint(0, 4096, (1 << 20,), device=dev),
            torch.randn(1 << 20, 64, device=dev),
        ),
    ),
    (
        "index_put bf16 1Mx256->512 hot",
        lambda x, i, w: x.index_put_((i,), w, accumulate=True),
        lambda: (
            torch.zeros(512, 256, device=dev, dtype=torch.bfloat16),
            hot(1 << 20, 512),
            torch.randn(1 << 20, 256, device=dev, dtype=torch.bfloat16),
        ),
    ),
    (
        "emb_bwd bf16 1Mx256 v=32k hot",
        lambda i, g: torch.ops.aten.embedding_dense_backward(g, i, 32768, -1, False),
        lambda: (
            hot(1 << 20, 32768),
            torch.randn(1 << 20, 256, device=dev, dtype=torch.bfloat16),
        ),
    ),
    (
        "index_add fp32 2Mx1->1024 hot",
        lambda x, i, w: x.index_add_(0, i, w),
        lambda: (
            torch.zeros(1024, 1, device=dev),
            hot(1 << 21, 1024),
            torch.randn(1 << 21, 1, device=dev),
        ),
    ),
]

MODES = {
    "det fallback": (True, False),
    "det sorted": (True, True),
    "atomic preagg": (False, True),
}


def bench(fn, make, deterministic, preagg):
    torch._dynamo.reset()
    torch.use_deterministic_algorithms(deterministic)
    try:
        with config.patch(scatter_preagg_enabled=preagg, fx_graph_cache=False):
            args = make()
            cfn = torch.compile(fn, fullgraph=True)
            cfn(*args)
            return do_bench(lambda: cfn(*args), warmup=10, rep=100)
    finally:
        torch.use_deterministic_algorithms(False)


def main():
    print(f"{'case':36s}" + "".join(f"{m:>15s}" for m in MODES) + f"{'speedup':>10s}")
    for name, fn, make in CASES:
        ms = {m: bench(fn, make, *flags) for m, flags in MODES.items()}
        speedup = ms["det fallback"] / ms["det sorted"]
        print(
            f"{name:36s}"
            + "".join(f"{ms[m]:13.3f}ms" for m in MODES)
            + f"{speedup:9.1f}x"
        )


if __name__ == "__main__":
    main()

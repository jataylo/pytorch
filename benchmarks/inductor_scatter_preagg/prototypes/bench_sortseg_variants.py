"""Inductor-compatible variants of the sort + segmented-sum scatter.

k_sortseg_gather: values computed in natural token order (as Inductor's fused
producer would produce them) and permuted in registers with tl.gather.
fx_sortseg: same algorithm expressed purely as ATen/HOP ops for stock Inductor.
"""

import torch
import torch._dynamo
import triton
import triton.language as tl
from torch._higher_order_ops.associative_scan import associative_scan
from triton.testing import do_bench

from bench_local_reduce import _seg_combine, k_sortseg, make_idx, N, D, COL0, LD, dev


@triton.jit
def k_sortseg_gather(idx_ptr, x_ptr, out_ptr, n, V: tl.constexpr, CHUNK: tl.constexpr, D: tl.constexpr):
    LOG: tl.constexpr = CHUNK.bit_length() - 1
    pos = tl.arange(0, CHUNK)
    tok = tl.program_id(0) * CHUNK + pos
    m = tok < n
    d = tl.arange(0, D)
    # "fused producer": index math + value load in natural order
    i = tl.load(idx_ptr + tok, m, other=0)
    i = tl.where(i < 0, i + V, i)
    valid = m & (i >= 0) & (i < V)
    g = tl.load(x_ptr + (184 + d[None, :] + 224 * tok[:, None]), m[:, None], other=0.0).to(tl.float32)
    # pre-aggregation that a codegen'd atomic store would emit
    dest = tl.where(valid, i, V)
    key = tl.sort((dest << LOG) | pos)
    sdest = key >> LOG
    perm = tl.broadcast_to((key & (CHUNK - 1))[:, None], (CHUNK, D))
    gs = tl.gather(g, perm, 0)
    k2 = tl.broadcast_to(sdest[:, None], (CHUNK, D))
    _, acc = tl.associative_scan((k2, gs), 0, _seg_combine)
    nxt = tl.gather(sdest, tl.minimum(pos + 1, CHUNK - 1), 0)
    last = (sdest < V) & ((sdest != nxt) | (pos == CHUNK - 1))
    tl.atomic_add(out_ptr + sdest[:, None] * D + d[None, :], acc, last[:, None], sem="relaxed")


def _combine(a, b):
    ka, va = a
    kb, vb = b
    return kb.clone(), torch.where(ka == kb, va + vb, vb)


def fx_sortseg(idx, x, V, C=512):
    g = x[:, COL0 : COL0 + D].float()
    i = idx.long()
    i = torch.where(i < 0, i + V, i)
    pad = (-N) % C
    dest = torch.nn.functional.pad(torch.where((i >= 0) & (i < V), i, V), (0, pad), value=V)
    g = torch.nn.functional.pad(g, (0, 0, 0, pad)).view(-1, C, D)
    sdest, perm = torch.sort(dest.view(-1, C), dim=1, stable=True)
    gs = torch.gather(g, 1, perm[..., None].expand(-1, -1, D))
    k2 = sdest[..., None].expand(-1, -1, D)
    _, acc = associative_scan(_combine, (k2, gs), dim=1, combine_mode="pointwise")
    nxt = torch.cat([sdest[:, 1:], torch.full_like(sdest[:, :1], -1)], 1)
    last = (sdest < V) & (sdest != nxt)
    acc = torch.where(last[..., None], acc, 0.0).view(-1, D)
    rows = torch.where(last, sdest, 0).view(-1)
    out = torch.zeros(V, D, device=x.device, dtype=torch.float32)
    return out.index_put_((rows,), acc, accumulate=True)


def main():
    for dist in ["hot50", "uniform"]:
        for V in [24, 128, 6048]:
            idx = make_idx(V, dist)
            x = torch.randn(N, LD, device=dev, dtype=torch.bfloat16)
            ref = torch.zeros(V, D, device=dev, dtype=torch.float64).index_add_(
                0, idx.long(), x[:, COL0 : COL0 + D].double()
            )
            scale = ref.abs().max().item()
            grid = (triton.cdiv(N, 512),)

            def run(k):
                out = torch.zeros(V, D, device=dev)
                k[grid](idx, x, out, N, V=V, CHUNK=512, D=D, num_warps=2)
                return out

            torch._dynamo.reset()
            c = torch.compile(lambda a, b: fx_sortseg(a, b, V), fullgraph=True)
            row = []
            for name, fn in [
                ("sortseg_reload", lambda: run(k_sortseg)),
                ("sortseg_gather", lambda: run(k_sortseg_gather)),
                ("fx_sortseg", lambda: c(idx, x)),
            ]:
                out = fn()
                torch.cuda.synchronize()
                err = (out.double() - ref).abs().max().item() / scale
                row.append(f"{name} {do_bench(fn, warmup=5, rep=50):.3f}ms(e{err:.0e})")
            print(f"{dist:7s} V={V:5d} | " + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()

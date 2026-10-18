"""Compare ways of cutting atomic contention for the embedding_dense_backward scatter.

out[V, D] += x[n, 184:184+D] at row idx[n], N = 2.48M, bf16 -> fp32.
"""

import sys

import torch
import torch._dynamo
import torch._inductor.config as icfg
import triton
import triton.language as tl
from triton.testing import do_bench

N, D, COL0, LD = 2_479_833, 32, 184, 224
dev = "cuda"


@triton.jit
def k_orig(idx_ptr, x_ptr, out_ptr, xnumel, V: tl.constexpr, XBLOCK: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    x1 = xindex // 32
    x0 = xindex % 32
    i = tl.load(idx_ptr + x1, xmask).to(tl.int64)
    g = tl.load(x_ptr + (184 + x0 + 224 * x1), xmask).to(tl.float32)
    r = tl.where(i < 0, i + V, i)
    g = tl.where((i >= 0) & (i < V), g, 0.0)
    tl.atomic_add(out_ptr + (x0 + 32 * r), g, xmask, sem="relaxed")


@triton.jit
def k_unroll(idx_ptr, x_ptr, out_ptr, n, V: tl.constexpr, CHUNK: tl.constexpr, D: tl.constexpr):
    tok = tl.program_id(0) * CHUNK + tl.arange(0, CHUNK)
    m = tok < n
    i = tl.load(idx_ptr + tok, m, other=0).to(tl.int64)
    valid = m & (i >= 0) & (i < V)
    d = tl.arange(0, D)
    g = tl.load(x_ptr + (184 + d[None, :] + 224 * tok[:, None]), m[:, None]).to(tl.float32)
    g = tl.where(valid[:, None], g, 0.0)
    for v in tl.static_range(V):
        s = tl.sum(tl.where((i == v)[:, None], g, 0.0), axis=0)
        tl.atomic_add(out_ptr + v * D + d, s, sem="relaxed")


@triton.jit
def _seg_combine(ka, va, kb, vb):
    return kb, tl.where(ka == kb, va + vb, vb)


@triton.jit
def k_sortseg(idx_ptr, x_ptr, out_ptr, n, V: tl.constexpr, CHUNK: tl.constexpr, D: tl.constexpr):
    # Sort the chunk's rows by destination, segmented-sum equal destinations, and
    # issue one atomic per (distinct destination, d) instead of one per element.
    # Cost is O(CHUNK log CHUNK), independent of V.
    LOG: tl.constexpr = CHUNK.bit_length() - 1
    pos = tl.arange(0, CHUNK)
    tok = tl.program_id(0) * CHUNK + pos
    m = tok < n
    i = tl.load(idx_ptr + tok, m, other=0)
    i = tl.where(i < 0, i + V, i)
    valid = m & (i >= 0) & (i < V)
    dest = tl.where(valid, i, V)  # invalid rows sort to the end and are masked off
    key = tl.sort((dest << LOG) | pos)
    sdest = key >> LOG
    stok = tl.program_id(0) * CHUNK + (key & (CHUNK - 1))
    smask = sdest < V
    d = tl.arange(0, D)
    g = tl.load(x_ptr + (184 + d[None, :] + 224 * stok[:, None]), smask[:, None], other=0.0).to(tl.float32)
    k2 = tl.broadcast_to(sdest[:, None], (CHUNK, D))
    _, acc = tl.associative_scan((k2, g), 0, _seg_combine)
    nxt = tl.gather(sdest, tl.minimum(pos + 1, CHUNK - 1), 0)
    last = smask & ((sdest != nxt) | (pos == CHUNK - 1))
    tl.atomic_add(out_ptr + sdest[:, None] * D + d[None, :], acc, last[:, None], sem="relaxed")


def launch(kind, idx, x, V, chunk=512):
    out = torch.zeros(V, D, device=dev, dtype=torch.float32)
    if kind == "orig":
        xn = N * D
        k_orig[(triton.cdiv(xn, 64),)](idx, x, out, xn, V=V, XBLOCK=64, num_warps=1)
    elif kind == "unroll":
        k_unroll[(triton.cdiv(N, chunk),)](idx, x, out, N, V=V, CHUNK=chunk, D=D, num_warps=4)
    elif kind == "sortseg":
        k_sortseg[(triton.cdiv(N, chunk),)](idx, x, out, N, V=V, CHUNK=chunk, D=D, num_warps=4)
    return out


def eager_ref(idx, x, V):
    return torch.ops.aten.embedding_dense_backward(x[:, COL0 : COL0 + D], idx, V, -1, False)


def onehot_ref(idx, x, V):
    # Graph-level rewrite of index_put(accumulate) into a dense masked reduction.
    g = x[:, COL0 : COL0 + D].float()
    i = idx.long()
    i = torch.where(i < 0, i + V, i)
    hit = i[:, None] == torch.arange(V, device=x.device)[None, :]
    return torch.where(hit[:, :, None], g[:, None, :], 0.0).sum(0).to(x.dtype)


def compiled(fn, V, ps, idx, x):
    torch._dynamo.reset()
    icfg.partitioned_scatter_enabled = ps
    icfg.partitioned_scatter_force = ps
    c = torch.compile(lambda i, a: fn(i, a, V), fullgraph=True)
    c(idx, x)  # compile now, while the config above is in effect
    return c


def make_idx(V, dist):
    if dist == "hot50":
        tail = torch.randint(1, max(V, 2), (N,), device=dev, dtype=torch.int32) % V
        return torch.where(torch.rand(N, device=dev) < 0.5, 0, tail).int()
    return torch.randint(0, V, (N,), device=dev, dtype=torch.int32)


def main():
    Vs = [int(v) for v in sys.argv[1:]] or [24, 128, 501, 6048]
    for dist in ["hot50", "uniform"]:
        for V in Vs:
            idx = make_idx(V, dist)
            x = torch.randn(N, LD, device=dev, dtype=torch.bfloat16)
            ref = torch.zeros(V, D, device=dev, dtype=torch.float64).index_add_(
                0, idx.long(), x[:, COL0 : COL0 + D].double()
            )
            scale = ref.abs().max().item()
            cases = [("triton_orig", lambda: launch("orig", idx, x, V))]
            if V <= 128:
                cases.append(("triton_unroll", lambda: launch("unroll", idx, x, V)))
            cases.append(("triton_sortseg", lambda: launch("sortseg", idx, x, V)))
            for name, fn, ps in [("ind_base", eager_ref, False), ("ind_ps", eager_ref, True)] + (
                [("ind_onehot", onehot_ref, False)] if V <= 128 else []
            ):
                c = compiled(fn, V, ps, idx, x)
                cases.append((name, lambda c=c: c(idx, x)))
            row = []
            for name, fn in cases:
                out = fn()
                torch.cuda.synchronize()
                err = (out.double() - ref).abs().max().item() / scale
                ms = do_bench(fn, warmup=5, rep=50)
                row.append(f"{name} {ms:7.3f}ms(e{err:.0e})")
            print(f"{dist:7s} V={V:5d} | " + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()

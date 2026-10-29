"""Pointwise/reduction kernels: Inductor's FlyDSL codegen (config.flydsl.simd) vs Triton.

Each case is compiled twice in one process, once per backend, with default Inductor
settings (no max-autotune for either). Three numbers per backend:

* ``gpu_us``  -- one call replayed from a CUDA graph, so CPU launch cost is excluded;
* ``GB/s``    -- the minimum bytes the op must move (inputs read once, outputs written
                 once) over ``gpu_us``: what fraction of HBM bandwidth each kernel reaches;
* ``call_us`` -- the same call from a plain Python loop, i.e. including Inductor's wrapper
                 and each backend's launch path.

    python benchmarks/inductor_backends/flydsl_simd/bench_kernels.py [--only NAME ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch._inductor.config as ic
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters


ic.fx_graph_cache = False

bf16, f16, f32 = torch.bfloat16, torch.float16, torch.float32


def _r(*shape, dtype=f32):
    return lambda: torch.randn(*shape, device="cuda", dtype=dtype)


def _i(hi, *shape):
    return lambda: torch.randint(0, hi, shape, device="cuda")


def _rms(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * w


def _rms_bwd(x, w, g, rs):
    # RMSNorm's backward from its saved rstd: the input gradient (a row reduction) and
    # the weight gradient (a column reduction), one mix-order kernel.
    n, gw = x.float() * rs, g.float() * w.float()
    dx = rs * (gw - n * (gw * n).mean(-1, keepdim=True))
    return dx.to(x.dtype), (g.float() * n).sum(0).to(x.dtype)


def _qk_norm_rope(x, w, cos, sin):
    b, t, d = cos.shape
    q = _rms(x.view(b, t, -1, d), w).transpose(1, 2)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * c + torch.cat((-q[..., d // 2 :], q[..., : d // 2]), -1) * s


def _k_norm_rope_pos(x, w, inv_freq):
    # As Qwen3 prefill compiles it: cos/sin from the positions inside the kernel, the
    # 8 KV heads normalized and rotated.
    b, t, d = 16, 1024, inv_freq.numel() * 2
    pos = torch.arange(t, device=x.device, dtype=torch.float32)
    emb = torch.cat([pos[:, None] * inv_freq[None, :]] * 2, -1)
    cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
    k = _rms(x.view(b, t, -1, d), w).transpose(1, 2)
    return k * cos + torch.cat((-k[..., d // 2 :], k[..., : d // 2]), -1) * sin


def _ln_bwd(g, x, w):
    # Input gradient of layer_norm(x, w) with mean/rstd recomputed, as Inductor fuses it.
    xf, gf = x.float(), g.float() * w.float()
    var, mean = torch.var_mean(xf, -1, unbiased=False, keepdim=True)
    rstd = torch.rsqrt(var + 1e-5)
    xhat = (xf - mean) * rstd
    dx = rstd * (
        gf - gf.mean(-1, keepdim=True) - xhat * (gf * xhat).mean(-1, keepdim=True)
    )
    return dx.to(x.dtype)


def _qkv_cat(q, kt, v):
    # As the backward of qkv.split(768, -1) then per-head views: select_backward sums.
    grads = [q.transpose(1, 2), kt.permute(0, 3, 1, 2), v.transpose(1, 2)]
    out = 0
    for i, g in enumerate(grads):
        out = out + torch.ops.aten.select_backward(
            g.reshape(4096, 12, 64), [3, 4096, 12, 64], 0, i
        )
    return out.permute(1, 0, 2, 3).reshape(4096, 2304)


def _rms_bwd_embed(a, b, c, w, x, ix, e):
    # Llama's first RMSNorm backward with the embedding gradient's accumulate fused in.
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    g = (a + b + c) * w
    s = (g.float() * xf).sum(-1, keepdim=True)
    dx = g.float() * rstd + (-0.5 * s * rstd**3 / x.shape[-1]) * (2 * xf)
    h = (e + dx.to(e.dtype)).float()
    out = torch.zeros(32000, x.shape[-1], device=x.device)
    return out.index_put_((ix,), h, accumulate=True)


# Outputs drawn from the RNG: Inductor's Philox stream differs from eager's.
RANDOM = {"dropout_bf16", "randn_f32_16M"}


def _params(seed):
    shapes = [(768, 768), (768,), (3072, 768), (768, 3072), (2304, 768), (2304,)] * 8
    return lambda: [torch.randn(s, device="cuda") for s in shapes]


def _adam(ps, gs, ms, vs):
    ms = torch._foreach_add(torch._foreach_mul(ms, 0.9), torch._foreach_mul(gs, 0.1))
    vs = torch._foreach_addcmul(torch._foreach_mul(vs, 0.999), gs, gs, value=0.001)
    den = torch._foreach_add(torch._foreach_sqrt(vs), 1e-8)
    return torch._foreach_addcdiv(ps, ms, den, value=-1e-3), ms, vs


# name -> (fn, [arg factories])
CASES = {
    # pointwise
    "add_f32_16M": (lambda a, b: a + b, [_r(1 << 24), _r(1 << 24)]),
    "bcast_add_f32": (lambda a, b: a + b, [_r(8192, 4096), _r(4096)]),
    "silu_mul_bf16": (
        lambda a, b: F.silu(a) * b,
        [_r(8192, 8192, dtype=bf16), _r(8192, 8192, dtype=bf16)],
    ),
    "gelu_f16_64M": (lambda a: F.gelu(a, approximate="tanh"), [_r(1 << 26, dtype=f16)]),
    "transpose_f32": (lambda a: a.t().contiguous() * 2, [_r(4096, 4096)]),
    # GPT-2's embedding gradient back to the weight's dtype
    "cast_bf16_24M": (lambda a: a.to(bf16), [_r(32000, 768)]),
    "small_add_4K": (lambda a, b: a + b, [_r(4096), _r(4096)]),
    "embedding_bf16": (
        lambda w, ix: F.embedding(ix, w),
        [_r(32000, 4096, dtype=bf16), _i(32000, 8192)],
    ),
    # An embedding gradient on its own: zero fill, f32 scatter-add, cast to bf16
    "embedding_bwd_bf16": (
        lambda g, ix: torch.ops.aten.embedding_dense_backward(g, ix, 32000, -1, False),
        [_r(8192, 1024, dtype=bf16), _i(32000, 8192)],
    ),
    # reductions
    "sum_rows_f32": (lambda a: a.sum(-1), [_r(8192, 8192)]),
    "sum_cols_f32": (lambda a: a.sum(0), [_r(8192, 8192)]),
    "sum_short_rows": (lambda a: a.sum(-1), [_r(1 << 20, 32)]),
    "amax_rows_bf16": (lambda a: a.amax(-1), [_r(16384, 4096, dtype=bf16)]),
    "argmax_vocab": (lambda a: a.argmax(-1), [_r(4096, 32000)]),
    "var_mean_f32": (lambda a: torch.var_mean(a, -1), [_r(8192, 4096)]),
    "softmax_f32_4K": (lambda a: torch.softmax(a, -1), [_r(8192, 4096)]),
    "softmax_bf16_32K": (lambda a: torch.softmax(a, -1), [_r(4096, 32000, dtype=bf16)]),
    "softmax_short_rows": (lambda a: torch.softmax(a, -1), [_r(1 << 18, 64)]),
    "layernorm_bf16": (
        lambda x, w, b: F.layer_norm(x, (4096,), w, b),
        [_r(16384, 4096, dtype=bf16), _r(4096, dtype=bf16), _r(4096, dtype=bf16)],
    ),
    "rmsnorm_bf16": (_rms, [_r(16384, 4096, dtype=bf16), _r(4096, dtype=bf16)]),
    "rmsnorm_3072_bf16": (_rms, [_r(16384, 3072, dtype=bf16), _r(3072, dtype=bf16)]),
    "rmsnorm_bwd_2048_bf16": (
        _rms_bwd,
        [_r(16384, 2048, dtype=bf16), _r(2048, dtype=bf16)]
        + [_r(16384, 2048, dtype=bf16), _r(16384, 1)],
    ),
    "rmsnorm_bwd_3072_bf16": (
        _rms_bwd,
        [_r(16384, 3072, dtype=bf16), _r(3072, dtype=bf16)]
        + [_r(16384, 3072, dtype=bf16), _r(16384, 1)],
    ),
    "cross_entropy_32K": (
        lambda x, t: F.cross_entropy(x, t),
        [_r(4096, 32000), _i(32000, 4096)],
    ),
    # BERT's attention-probability gradient through dropout, 12 heads x 512 tokens
    "softmax_dropout_bwd_f16": (
        lambda g, y, m: torch.ops.aten._softmax_backward_data(g * m * 1.1111, y, -1, f16),
        [
            _r(98304, 512, dtype=f16),
            _r(98304, 512, dtype=f16),
            lambda: torch.rand(98304, 512, device="cuda") < 0.9,
        ],
    ),
    # BERT's loss tail: one block gathers a logit for each of 8192 tokens
    "nll_gather_f16": (
        lambda x, m, l, t: F.nll_loss(x.float() - m - l, t),
        [_r(8192, 30522, dtype=f16), _r(8192, 1), _r(8192, 1), _i(30522, 8192)],
    ),
    # GPT-small shapes (d=768, 12 heads x 512 tokens), where whole-model time is lost
    "add_layernorm_768": (
        lambda x, h, w, b: F.layer_norm(x + h, (768,), w, b),
        [
            _r(4096, 768, dtype=bf16),
            _r(4096, 768, dtype=bf16),
            _r(768, dtype=bf16),
            _r(768, dtype=bf16),
        ],
    ),
    "layernorm_bwd_768": (
        lambda g, x, w: _ln_bwd(g, x, w),
        [_r(4096, 768, dtype=bf16), _r(4096, 768, dtype=bf16), _r(768, dtype=bf16)],
    ),
    "masked_softmax_512": (
        lambda s, m: torch.softmax((s * 0.125).masked_fill(m, float("-inf")), -1),
        [
            _r(49152, 512, dtype=bf16),
            lambda: torch.ones(512, 512, device="cuda", dtype=torch.bool)
            .triu(1)
            .repeat(96, 1),
        ],
    ),
    # ResNet shapes: NCHW conv outputs, 64 channels of 64x64 over a batch of 64
    "conv_bias_nchw": (
        lambda x, b: x + b.view(1, 64, 1, 1),
        [_r(64, 64, 64, 64), _r(64)],
    ),
    "bn_stats_nchw": (lambda x: torch.var_mean(x, (0, 2, 3)), [_r(64, 64, 64, 64)]),
    "bn_bwd_reduce_nchw": (
        lambda g, x, m: (
            (g * (x - m.view(1, 64, 1, 1))).sum((0, 2, 3)),
            g.sum((0, 2, 3)),
        ),
        [_r(64, 64, 64, 64), _r(64, 64, 64, 64), _r(64)],
    ),
    "softmax_bwd_512": (
        lambda g, y: (g.float() - (g.float() * y.float()).sum(-1, keepdim=True))
        .mul(y.float())
        .to(bf16),
        [_r(49152, 512, dtype=bf16), _r(49152, 512, dtype=bf16)],
    ),
    # GPT-2's attention-probability gradient: causal mask and the 1/sqrt(d) scale fused
    "masked_softmax_bwd_512": (
        lambda g, y, m: (
            (g.float() * y.float() - y.float() * (g.float() * y.float()).sum(-1, True))
            .view(96, 512, 512)
            .masked_fill(m, 0.0)
            .mul(0.125)
            .to(bf16)
        ),
        [
            _r(49152, 512, dtype=bf16),
            _r(49152, 512, dtype=bf16),
            lambda: torch.ones(512, 512, device="cuda", dtype=torch.bool).triu(1),
        ],
    ),
    # LayerNorm weight/bias gradients at ViT-B shapes (197 tokens x 32): a split column
    # reduction of bf16 rows
    "ln_wgrad_vit": (
        lambda g, x, m, r: (
            (g.float() * ((x.float() - m) * r)).sum(0),
            g.float().sum(0),
        ),
        [
            _r(6304, 768, dtype=bf16),
            _r(6304, 768, dtype=bf16),
            _r(6304, 1),
            _r(6304, 1),
        ],
    ),
    # The same over (batch, tokens, dim): the split's row index wraps through a view
    "ln_wgrad_vit3d": (
        lambda g, x, m, r: (
            (g.float() * ((x.float() - m) * r)).sum((0, 1)),
            g.float().sum((0, 1)),
        ),
        [
            _r(32, 197, 768, dtype=bf16),
            _r(32, 197, 768, dtype=bf16),
            _r(32, 197, 1),
            _r(32, 197, 1),
        ],
    ),
    # Exact (erf) GELU forward and backward at ViT-B MLP shapes
    "gelu_erf_bf16": (lambda a: F.gelu(a), [_r(6304, 3072, dtype=bf16)]),
    "gelu_erf_bwd_bf16": (
        lambda g, a: torch.ops.aten.gelu_backward(g, a),
        [_r(6304, 3072, dtype=bf16), _r(6304, 3072, dtype=bf16)],
    ),
    # Random numbers (Philox per element)
    "dropout_bf16": (lambda a: F.dropout(a, 0.1, True), [_r(8192, 8192, dtype=bf16)]),
    "randn_f32_16M": (lambda a: a + torch.randn_like(a), [_r(1 << 24)]),
    # 2D tiles: transposed loads staged through LDS
    "permute_heads_bf16": (
        lambda x: x.view(32, 512, 12, 64).permute(0, 2, 1, 3).contiguous(),
        [_r(32 * 512, 768, dtype=bf16)],
    ),
    "nchw_to_nhwc_f32": (
        lambda x: x.contiguous(memory_format=torch.channels_last),
        [_r(64, 64, 64, 64)],
    ),
    # Llama pieces
    # RMSNorm weight gradient: a sum over tokens, contiguous along the channels. In
    # f32 and without Llama's bf16 rounding of the normalized input, so eager is a
    # fair reference for both backends.
    "rmsnorm_wgrad_bf16": (
        lambda a, b, h, rs: ((a.float() + b.float()) * (h.float() * rs)).sum((0, 1)),
        [_r(4, 1024, 1024, dtype=bf16)] * 3 + [_r(4, 1024, 1)],
    ),
    "rope_bf16": (
        lambda x, c, s: x * c + torch.cat((-x[..., 32:], x[..., :32]), -1) * s,
        [
            _r(8, 16, 1024, 64, dtype=bf16),
            _r(1024, 64, dtype=bf16),
            _r(1024, 64, dtype=bf16),
        ],
    ),
    # Qwen3-0.6B (16 heads of 128): q/k RMSNorm weight gradient, a sum over tokens and
    # heads of bf16 rows only 128 wide
    # GPT-2's fused QKV projection gradient: q, k (transposed) and v gradients to rows
    "qkv_grad_cat_bf16": (
        lambda q, kt, v: _qkv_cat(q, kt, v),
        [_r(8, 12, 512, 64, dtype=bf16), _r(8, 12, 64, 512, dtype=bf16)]
        + [_r(8, 12, 512, 64, dtype=bf16)],
    ),
    # ViT-B/16 MLP bias gradient (32 x 197 tokens), a split reduction's first stage
    "bias_grad_bf16": (lambda g: g.sum(0), [_r(6304, 3072, dtype=bf16)]),
    "rms_bwd_embed_bf16": (
        _rms_bwd_embed,
        [_r(4096, 1024, dtype=bf16)] * 3
        + [_r(1024, dtype=bf16), _r(4096, 1024, dtype=bf16), _i(32000, 4096)]
        + [_r(4096, 1024, dtype=bf16)],
    ),
    "head_wgrad_bf16": (
        lambda g: g.sum((0, 1, 2), keepdim=True),
        [_r(8, 1024, 16, 128, dtype=bf16)],
    ),
    # and the attention prologue: q RMSNorm over each head, to heads-major, RoPE
    "qk_norm_rope_bf16": (
        lambda x, w, c, s: _qk_norm_rope(x, w, c, s),
        [
            _r(16, 1024, 16 * 128, dtype=bf16),
            _r(128, dtype=bf16),
            _r(16, 1024, 128, dtype=bf16),
            _r(16, 1024, 128, dtype=bf16),
        ],
    ),
    "k_norm_rope_pos_bf16": (
        _k_norm_rope_pos,
        [
            _r(16, 1024, 8 * 128, dtype=bf16),
            _r(128, dtype=bf16),
            lambda: 1e-4 ** (torch.arange(64, device="cuda") / 64),
        ],
    ),
    # Scans
    "cumsum_rows_f32": (lambda a: a.cumsum(-1), [_r(8192, 4096)]),
    # Foreach (one combo launch over many parameters)
    "adam_foreach_f32": (
        lambda ps, gs, ms, vs: _adam(ps, gs, ms, vs),
        [_params(seed) for seed in range(4)],
    ),
}


def _nbytes(xs) -> int:
    if isinstance(xs, torch.Tensor):
        return xs.numel() * xs.element_size()
    if isinstance(xs, (tuple, list)):
        return sum(_nbytes(x) for x in xs)
    return 0


def _graph_time_us(fn, args, reps=50) -> float:
    """Median-free mean of a CUDA-graph replay: GPU time without CPU launch cost."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(*args)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(*args)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    # Each replay is too short to time alone; time a batch, a few times, keep the best.
    best = float("inf")
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(reps):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / reps)
    return best


def _call_time_us(fn, args, reps=200) -> float:
    for _ in range(10):
        fn(*args)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e6 / reps)
    return best


def _upcast_ref(fn, args, ref):
    def up(a):
        if isinstance(a, torch.Tensor) and a.dtype in (torch.float16, bf16):
            return a.float()
        if isinstance(a, (list, tuple)):
            return type(a)(up(x) for x in a)
        return a

    return pytree.tree_map(
        lambda o, r: o.to(r.dtype) if isinstance(r, torch.Tensor) else o,
        fn(*up(args)),
        ref,
    )


def run_case(name, fn, factories):
    torch.manual_seed(0)
    args = [f() for f in factories]
    ref = fn(*args)
    traffic = _nbytes(args) + _nbytes(ref)
    row = {"case": name, "bytes": traffic}
    for backend in ("triton", "flydsl"):
        ic.flydsl.simd = backend == "flydsl"
        torch._dynamo.reset()
        counters.clear()
        cfn = torch.compile(fn, dynamic=False)
        t0 = time.perf_counter()
        out = cfn(*args)
        torch.cuda.synchronize()
        row[f"{backend}_compile_s"] = time.perf_counter() - t0
        if name not in RANDOM:
            # Eager rounds bf16 intermediates the compiled kernels keep in f32, and
            # lands an element past the tolerance now and then: then hold both to an
            # f32 evaluation instead.
            try:
                torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2, equal_nan=True)
            except AssertionError:
                ref = _upcast_ref(fn, args, ref)
                torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2, equal_nan=True)
        if backend == "flydsl":
            c = counters["flydsl_simd"]
            row["flydsl_kernels"] = c["flydsl_kernels"]
            row["triton_fallbacks"] = c["triton_kernels"]
            row["fallback_reasons"] = {
                k[len("fallback: ") :]: v
                for k, v in c.items()
                if k.startswith("fallback: ")
            }
        row[f"{backend}_gpu_us"] = _graph_time_us(cfn, args)
        row[f"{backend}_call_us"] = _call_time_us(cfn, args)
    ic.flydsl.simd = False
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*")
    p.add_argument("--json")
    a = p.parse_args()
    rows = []
    hdr = (
        f"{'case':20s} {'tri us':>9s} {'fly us':>9s} {'fly/tri':>8s} {'tri GB/s':>9s} {'fly GB/s':>9s}"
        f" {'tri call':>9s} {'fly call':>9s}  kernels"
    )
    print(hdr, flush=True)
    for name, (fn, factories) in CASES.items():
        if a.only and name not in a.only:
            continue
        r = run_case(name, fn, factories)
        rows.append(r)
        gb = lambda us: r["bytes"] / (us * 1e-6) / 1e9  # noqa: E731
        print(
            f"{name:20s} {r['triton_gpu_us']:9.1f} {r['flydsl_gpu_us']:9.1f} "
            f"{r['flydsl_gpu_us'] / r['triton_gpu_us']:8.2f} {gb(r['triton_gpu_us']):9.0f} {gb(r['flydsl_gpu_us']):9.0f}"
            f" {r['triton_call_us']:9.1f} {r['flydsl_call_us']:9.1f}  fly={r['flydsl_kernels']} fallback={r['triton_fallbacks']}"
            f" {r['fallback_reasons'] or ''}",
            flush=True,
        )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(
                {"device": torch.cuda.get_device_name(), "rows": rows}, f, indent=1
            )


if __name__ == "__main__":
    main()

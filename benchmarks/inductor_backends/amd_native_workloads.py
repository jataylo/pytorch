"""Workloads for the native Triton pointwise, copy and reduction comparison."""

import torch


def _basic_workload(case, seed):
    torch.manual_seed(seed)
    family, shape, dtype = (
        case["family"],
        case["shape"],
        getattr(torch, case["dtype"]),
    )
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if family == "mean":
        return (lambda x: x.float().square().mean(), (x,))
    if family == "transpose":
        if len(shape) == 3:
            heads = case["transpose_heads"]
            return (
                lambda x: x.view(-1, heads, shape[-2], shape[-1])
                .permute(0, 2, 1, 3)
                .contiguous(),
                (x,),
            )
        return (lambda x: x.permute(0, 2, 1, 3).contiguous(), (x,))
    if family == "norm_parameters":
        h = torch.randn_like(x, dtype=torch.float32)
        keep = torch.rand(shape, device="cuda") > 0.1

        def norm_parameters(g, h, keep):
            value = g.float() * (keep.float() * 1.1111111111111112)
            return ((value * h).sum(1).t(), value.sum(1).t())

        return (norm_parameters, (x, h, keep))
    if family == "broadcast":
        scalar = torch.tensor(0.75, device="cuda")
        return (
            lambda x, s: (s * (1.0 / x.numel()) * (x.float() * 2.0)).to(x.dtype),
            (x, scalar),
        )
    if family == "cast_add":
        return (lambda x, y: x.float() + y.float(), (x, torch.randn_like(x)))
    if family == "gelu_backward_inplace":

        def gelu_backward_inplace(x, g):
            g.copy_(torch.ops.aten.gelu_backward(g, x, approximate="none"))
            return g

        return (gelu_backward_inplace, (x, torch.randn_like(x)))
    if family == "gelu_backward":
        return (
            lambda x, g: torch.ops.aten.gelu_backward(g, x, approximate="none"),
            (x, torch.randn_like(x)),
        )
    if family == "softmax_saved_stats":
        from torch._inductor import inductor_prims

        keep = torch.rand((shape[0], 1, shape[2], shape[3]), device="cuda") > 0.1
        keep[:, :, 0] = False

        def saved_softmax(x, keep):
            value = x + torch.where(keep, 0.0, -float("inf"))
            maximum, total = inductor_prims.prepare_softmax_online(value, -1)
            invalid = (value == -float("inf")).all(-1, keepdim=True)
            probability = torch.where(invalid, 0.0, torch.exp(value - maximum) / total)
            return (invalid, maximum, total, probability)

        return (saved_softmax, (x, keep))
    if family == "softmax":
        keep = torch.rand(shape, device="cuda") > 0.1
        return (
            lambda x, keep: torch.softmax(
                x.float().masked_fill(~keep, -float("inf")), -1
            ),
            (x, keep),
        )
    if family == "softmax_backward":
        probability = torch.softmax(x.float(), -1)
        grad = torch.randn_like(probability)
        keep = torch.rand(shape, device="cuda") > 0.1

        def softmax_backward(p, g, mask):
            product = g * (mask.float() * 1.1111111111111112) * p
            return torch.addcmul(product, -p, product.sum(-1, keepdim=True))

        return (softmax_backward, (probability, grad, keep))
    gamma = torch.randn(shape[-1], device="cuda")
    scale = torch.rand(shape[0], 1, device="cuda") / shape[-1]
    h = torch.randn(shape, device="cuda")
    keep = torch.rand(shape, device="cuda") > 0.1

    def norm_backward(g, weight, h, scale, keep):
        v = g.float() * weight
        value = scale * (
            v * g.shape[-1]
            - v.sum(-1, keepdim=True)
            - h * (v * h).sum(-1, keepdim=True)
        )
        return (value, (value * (keep.float() * 1.1111111111111112)).to(g.dtype))

    return (norm_backward, (x, gamma, h, scale, keep))


def _elementwise_workload(case, seed):
    torch.manual_seed(seed)
    family, shape, dtype = (
        case["family"],
        case["shape"],
        getattr(torch, case["dtype"]),
    )
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if family == "dropout_backward":
        return (
            lambda x, m: (x.float() * m.float() * 1.1111111111111112).to(x.dtype),
            (x, torch.rand(shape, device="cuda") > 0.1),
        )
    if family == "silu_gate":
        return (
            lambda x, y: (torch.nn.functional.silu(x.float()) * y.float()).to(x.dtype),
            (x, torch.randn_like(x)),
        )
    if family == "silu_backward":
        return (
            lambda x, g: torch.ops.aten.silu_backward(g, x),
            (x, torch.randn_like(x)),
        )
    if family == "gelu_forward":
        return (lambda x: torch.nn.functional.gelu(x, approximate="none"), (x,))
    if family == "layer_norm":
        weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
        bias = torch.randn_like(weight)
        return (
            lambda x, w, b: torch.native_layer_norm(x, (x.shape[-1],), w, b, 1e-05),
            (x, weight, bias),
        )
    if family == "rms_norm":
        weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32)

        def rms(x, w):
            r = torch.rsqrt(x.float().square().mean(-1, True) + 1e-06)
            return ((x.float() * r * w).to(x.dtype), r)

        return (rms, (x, weight))
    if family == "log_softmax":
        return (lambda x: torch.log_softmax(x.float(), -1), (x,))
    return _basic_workload(case, seed)


def workload(case, seed):
    torch.manual_seed(seed)
    family, shape, dtype = (
        case["family"],
        case["shape"],
        getattr(torch, case["dtype"]),
    )
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if family == "transpose" and case.get("transpose_mode") == "matrix":
        return lambda value: value.transpose(-2, -1).contiguous(), (x,)
    if family == "adamw":
        g = torch.randn_like(x)
        m = torch.randn_like(x, dtype=torch.float32)
        v = torch.rand_like(m)

        def adamw(p, g, m, v):
            mean = m * 0.9 + g.float() * 0.1
            variance = v * 0.999 + g.float().square() * 0.001
            return (
                (p.float() * 0.9999 - 0.001 * mean / (torch.sqrt(variance) + 1e-08)).to(
                    p.dtype
                ),
                mean,
                variance,
            )

        return (adamw, (x, g, m, v))
    if family == "cross_entropy":
        t = torch.randint(0, shape[-1], (shape[0],), device="cuda")
        return (
            lambda x, t: torch.nn.functional.cross_entropy(
                x.float(), t, reduction="none"
            ),
            (x, t),
        )
    if family == "cross_entropy_backward":
        t = torch.randint(0, shape[-1], (shape[0],), device="cuda")
        return (
            lambda x, t: torch.softmax(x.float(), -1)
            - torch.nn.functional.one_hot(t, x.shape[-1]).float(),
            (x, t),
        )
    if family == "cumsum":
        return (lambda x: torch.cumsum(x.float(), dim=-1), (x,))
    if family == "gather":
        i = torch.randint(
            0, shape[0], (case.get("index_count", shape[0] * 2),), device="cuda"
        )
        return (lambda x, i: torch.nn.functional.embedding(i, x), (x, i))
    if family == "scatter_add":
        i = torch.randint(0, shape[0] // 2, (shape[0],), device="cuda")

        def scatter_add(x, i):
            out = torch.zeros(
                (x.shape[0] // 2, x.shape[1]), device=x.device, dtype=torch.float32
            )
            return out.index_add(0, i, x.float())

        return (scatter_add, (x, i))
    if family == "batch_norm_backward":
        if case.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        _, mean, invstd = torch.ops.aten.native_batch_norm(
            x.float(), None, None, None, None, True, 0.1, 1e-05
        )
        g = torch.randn_like(x)
        w = torch.ones(shape[1], device="cuda")

        def batch_norm_backward(g, x, w, mean, rstd):
            return torch.ops.aten.native_batch_norm_backward(
                g.float(),
                x.float(),
                w,
                None,
                None,
                mean,
                rstd,
                True,
                1e-05,
                [True, True, True],
            )

        return (batch_norm_backward, (g, x, w, mean, invstd))
    if family == "group_norm_backward":
        if case.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        n, c, hh, ww = shape
        groups = case.get("groups", 32)
        _, mean, rstd = torch.ops.aten.native_group_norm(
            x.float(), None, None, n, c, hh * ww, groups, 1e-05
        )
        g = torch.randn_like(x)
        w = torch.ones(c, device="cuda")

        def group_norm_backward(g, x, w, mean, rstd):
            return torch.ops.aten.native_group_norm_backward(
                g.float(),
                x.float(),
                mean,
                rstd,
                w,
                n,
                c,
                hh * ww,
                groups,
                [True, True, True],
            )

        return (group_norm_backward, (g, x, w, mean, rstd))
    if family == "batch_norm":
        if case.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        return (
            lambda x: torch.ops.aten.native_batch_norm(
                x.float(), None, None, None, None, True, 0.1, 1e-05
            ),
            (x,),
        )
    if family == "group_norm":
        if case.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        return (
            lambda x: torch.nn.functional.group_norm(x.float(), case.get("groups", 32)),
            (x,),
        )
    return _elementwise_workload(case, seed)

"""Operator units for the wide-reduction-splits experiment."""

import torch


CONFIG = "triton.amd_wide_reduction_splits"


def cases():
    return [
        {
            "case": "softmax_2x32769_float32",
            "op": "softmax",
            "family": "wide-reduction-splits",
            "shape": (2, 32769),
            "dtype": "float32",
        },
        {
            "case": "softmax_2x32769_bfloat16",
            "op": "softmax",
            "family": "wide-reduction-splits",
            "shape": (2, 32769),
            "dtype": "bfloat16",
        },
        {
            "case": "softmax_8x65535_float32",
            "op": "softmax",
            "family": "wide-reduction-splits",
            "shape": (8, 65535),
            "dtype": "float32",
        },
        {
            "case": "softmax_8x65535_bfloat16",
            "op": "softmax",
            "family": "wide-reduction-splits",
            "shape": (8, 65535),
            "dtype": "bfloat16",
        },
        {
            "case": "norm_backward_8x131072_float32",
            "op": "norm_backward",
            "family": "wide-reduction-splits",
            "shape": (8, 131072),
            "dtype": "float32",
        },
        {
            "case": "norm_backward_8x131072_bfloat16",
            "op": "norm_backward",
            "family": "wide-reduction-splits",
            "shape": (8, 131072),
            "dtype": "bfloat16",
        },
        {
            "case": "norm_backward_63x65535_float32",
            "op": "norm_backward",
            "family": "wide-reduction-splits",
            "shape": (63, 65535),
            "dtype": "float32",
        },
        {
            "case": "norm_backward_63x65535_bfloat16",
            "op": "norm_backward",
            "family": "wide-reduction-splits",
            "shape": (63, 65535),
            "dtype": "bfloat16",
        },
    ]


def workload(case, seed):
    torch.manual_seed(seed)
    dtype = getattr(torch, case["dtype"])
    shape = tuple(case["shape"])
    op = case["op"]
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if op == "softmax":

        def fn(x):
            return torch.softmax(x.float(), dim=-1).to(x.dtype)

        return fn, (x,)

    def fn(x, grad, scale):
        v = x.float()
        g = grad.float() * scale.float()
        return (g - g.mean(-1, keepdim=True) - v * (g * v).mean(-1, keepdim=True)).to(
            x.dtype
        )

    return fn, (
        x,
        torch.randn_like(x),
        torch.randn(shape[0], 1, device="cuda", dtype=dtype),
    )

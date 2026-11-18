"""Operator units for the row-packing experiment."""

import torch


CONFIG = "triton.amd_reduction_row_packing"


def cases():
    return [
        {
            "case": "softmax_37x128_float32",
            "op": "softmax",
            "family": "row-packing",
            "shape": (37, 128),
            "dtype": "float32",
        },
        {
            "case": "softmax_37x128_bfloat16",
            "op": "softmax",
            "family": "row-packing",
            "shape": (37, 128),
            "dtype": "bfloat16",
        },
        {
            "case": "softmax_37x769_float32",
            "op": "softmax",
            "family": "row-packing",
            "shape": (37, 769),
            "dtype": "float32",
        },
        {
            "case": "softmax_37x769_bfloat16",
            "op": "softmax",
            "family": "row-packing",
            "shape": (37, 769),
            "dtype": "bfloat16",
        },
        {
            "case": "rms_37x256_float32",
            "op": "rms",
            "family": "row-packing",
            "shape": (37, 256),
            "dtype": "float32",
        },
        {
            "case": "rms_37x256_bfloat16",
            "op": "rms",
            "family": "row-packing",
            "shape": (37, 256),
            "dtype": "bfloat16",
        },
        {
            "case": "sum_37x1024_float32",
            "op": "sum",
            "family": "row-packing",
            "shape": (37, 1024),
            "dtype": "float32",
        },
        {
            "case": "sum_37x1024_bfloat16",
            "op": "sum",
            "family": "row-packing",
            "shape": (37, 1024),
            "dtype": "bfloat16",
        },
        {
            "case": "softmax_4096x128_float32",
            "op": "softmax",
            "family": "row-packing",
            "shape": (4096, 128),
            "dtype": "float32",
        },
        {
            "case": "softmax_4096x128_bfloat16",
            "op": "softmax",
            "family": "row-packing",
            "shape": (4096, 128),
            "dtype": "bfloat16",
        },
        {
            "case": "rms_4096x769_float32",
            "op": "rms",
            "family": "row-packing",
            "shape": (4096, 769),
            "dtype": "float32",
        },
        {
            "case": "rms_4096x769_bfloat16",
            "op": "rms",
            "family": "row-packing",
            "shape": (4096, 769),
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
    elif op == "rms":

        def fn(x):
            value = x.float()
            return (
                value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)
            ).to(x.dtype)
    else:

        def fn(x):
            return x.float().sum(-1)

    return fn, (x,)

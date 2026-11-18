"""Operator units for the persistent-selection experiment."""

import torch


CONFIG = "triton.amd_persistent_selection"


def cases():
    return [
        {
            "case": "softmax_8x4096_float32",
            "op": "softmax",
            "family": "persistent-selection",
            "shape": (8, 4096),
            "dtype": "float32",
        },
        {
            "case": "softmax_8x4096_bfloat16",
            "op": "softmax",
            "family": "persistent-selection",
            "shape": (8, 4096),
            "dtype": "bfloat16",
        },
        {
            "case": "softmax_64x8192_float32",
            "op": "softmax",
            "family": "persistent-selection",
            "shape": (64, 8192),
            "dtype": "float32",
        },
        {
            "case": "softmax_64x8192_bfloat16",
            "op": "softmax",
            "family": "persistent-selection",
            "shape": (64, 8192),
            "dtype": "bfloat16",
        },
        {
            "case": "rms_8x16384_float32",
            "op": "rms",
            "family": "persistent-selection",
            "shape": (8, 16384),
            "dtype": "float32",
        },
        {
            "case": "rms_8x16384_bfloat16",
            "op": "rms",
            "family": "persistent-selection",
            "shape": (8, 16384),
            "dtype": "bfloat16",
        },
        {
            "case": "rms_64x32768_float32",
            "op": "rms",
            "family": "persistent-selection",
            "shape": (64, 32768),
            "dtype": "float32",
        },
        {
            "case": "rms_64x32768_bfloat16",
            "op": "rms",
            "family": "persistent-selection",
            "shape": (64, 32768),
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
    else:

        def fn(x):
            v = x.float()
            return (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-5)).to(
                x.dtype
            )

    return fn, (x,)

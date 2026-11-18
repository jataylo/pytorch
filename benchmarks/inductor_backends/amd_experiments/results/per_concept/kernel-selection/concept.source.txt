"""Operator units for the kernel-selection experiment."""

import torch


CONFIG = "triton.amd_kernel_selection"


def cases():
    return [
        {
            "case": "sum_64x4099_float32",
            "op": "sum",
            "family": "kernel-selection",
            "shape": (64, 4099),
            "dtype": "float32",
        },
        {
            "case": "sum_64x4099_bfloat16",
            "op": "sum",
            "family": "kernel-selection",
            "shape": (64, 4099),
            "dtype": "bfloat16",
        },
        {
            "case": "sum_1024x128_float32",
            "op": "sum",
            "family": "kernel-selection",
            "shape": (1024, 128),
            "dtype": "float32",
        },
        {
            "case": "sum_1024x128_bfloat16",
            "op": "sum",
            "family": "kernel-selection",
            "shape": (1024, 128),
            "dtype": "bfloat16",
        },
        {
            "case": "softmax_37x769_float32",
            "op": "softmax",
            "family": "kernel-selection",
            "shape": (37, 769),
            "dtype": "float32",
        },
        {
            "case": "softmax_37x769_bfloat16",
            "op": "softmax",
            "family": "kernel-selection",
            "shape": (37, 769),
            "dtype": "bfloat16",
        },
        {
            "case": "swiglu_4608000_float32",
            "op": "swiglu",
            "family": "kernel-selection",
            "shape": (4608000,),
            "dtype": "float32",
        },
        {
            "case": "swiglu_4608000_bfloat16",
            "op": "swiglu",
            "family": "kernel-selection",
            "shape": (4608000,),
            "dtype": "bfloat16",
        },
    ]


def workload(case, seed):
    torch.manual_seed(seed)
    dtype = getattr(torch, case["dtype"])
    shape = tuple(case["shape"])
    op = case["op"]
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if op == "sum":

        def fn(x):
            return x.float().sum(-1)

        return fn, (x,)
    if op == "softmax":

        def fn(x):
            return torch.softmax(x.float(), dim=-1).to(x.dtype)

        return fn, (x,)

    def fn(x, y):
        return (torch.nn.functional.silu(x.float()) * y.float()).to(x.dtype)

    return fn, (x, torch.randn_like(x))

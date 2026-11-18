"""Operator units for the hierarchical-scan experiment."""

import torch


CONFIG = "triton.amd_hierarchical_scan"


def cases():
    return [
        {
            "case": "cumsum_2x4099_float32",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (2, 4099),
            "dtype": "float32",
        },
        {
            "case": "cumsum_2x4099_bfloat16",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (2, 4099),
            "dtype": "bfloat16",
        },
        {
            "case": "cumsum_8x8193_float32",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (8, 8193),
            "dtype": "float32",
        },
        {
            "case": "cumsum_8x8193_bfloat16",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (8, 8193),
            "dtype": "bfloat16",
        },
        {
            "case": "cumsum_2x32769_float32",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (2, 32769),
            "dtype": "float32",
        },
        {
            "case": "cumsum_2x32769_bfloat16",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (2, 32769),
            "dtype": "bfloat16",
        },
        {
            "case": "cumsum_8x32768_float32",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (8, 32768),
            "dtype": "float32",
        },
        {
            "case": "cumsum_8x32768_bfloat16",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (8, 32768),
            "dtype": "bfloat16",
        },
    ]


def workload(case, seed):
    torch.manual_seed(seed)
    dtype = getattr(torch, case["dtype"])
    shape = tuple(case["shape"])
    x = torch.randn(shape, device="cuda", dtype=dtype)
    x = x.abs()

    def fn(x):
        return torch.cumsum(x, dim=-1, dtype=torch.float32)

    return fn, (x,)

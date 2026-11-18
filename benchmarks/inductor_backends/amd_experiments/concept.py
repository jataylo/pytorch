"""Operator units for the two-pass-scan experiment."""

import torch


CONFIG = "triton.amd_two_pass_scan"


def cases():
    return [
        {
            "case": "cumsum_1x131072_float32",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (1, 131072),
            "dtype": "float32",
        },
        {
            "case": "cumsum_1x131072_bfloat16",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (1, 131072),
            "dtype": "bfloat16",
        },
        {
            "case": "cumsum_8x131073_float32",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (8, 131073),
            "dtype": "float32",
        },
        {
            "case": "cumsum_8x131073_bfloat16",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (8, 131073),
            "dtype": "bfloat16",
        },
        {
            "case": "cumsum_2x262144_float32",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (2, 262144),
            "dtype": "float32",
        },
        {
            "case": "cumsum_2x262144_bfloat16",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (2, 262144),
            "dtype": "bfloat16",
        },
        {
            "case": "cumsum_8x262145_float32",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (8, 262145),
            "dtype": "float32",
        },
        {
            "case": "cumsum_8x262145_bfloat16",
            "op": "cumsum",
            "family": "cumsum",
            "shape": (8, 262145),
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

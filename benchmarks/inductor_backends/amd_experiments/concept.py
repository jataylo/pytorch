"""Operator units for the transpose-tiling experiment."""

import torch


CONFIG = "triton.amd_transpose_tiling"


def cases():
    return [
        {
            "case": "transpose_4x16x65x129_float32",
            "op": "transpose",
            "family": "transpose",
            "shape": (4, 16, 65, 129),
            "dtype": "float32",
        },
        {
            "case": "transpose_4x16x65x129_bfloat16",
            "op": "transpose",
            "family": "transpose",
            "shape": (4, 16, 65, 129),
            "dtype": "bfloat16",
        },
        {
            "case": "transpose_1x8x64x128_float32",
            "op": "transpose",
            "family": "transpose",
            "shape": (1, 8, 64, 128),
            "dtype": "float32",
        },
        {
            "case": "transpose_1x8x64x128_bfloat16",
            "op": "transpose",
            "family": "transpose",
            "shape": (1, 8, 64, 128),
            "dtype": "bfloat16",
        },
        {
            "case": "transpose_4x4x127x65_float32",
            "op": "transpose",
            "family": "transpose",
            "shape": (4, 4, 127, 65),
            "dtype": "float32",
        },
        {
            "case": "transpose_4x4x127x65_bfloat16",
            "op": "transpose",
            "family": "transpose",
            "shape": (4, 4, 127, 65),
            "dtype": "bfloat16",
        },
        {
            "case": "transpose_8x32x128x64_float32",
            "op": "transpose",
            "family": "transpose",
            "shape": (8, 32, 128, 64),
            "dtype": "float32",
        },
        {
            "case": "transpose_8x32x128x64_bfloat16",
            "op": "transpose",
            "family": "transpose",
            "shape": (8, 32, 128, 64),
            "dtype": "bfloat16",
        },
        {
            "case": "transpose_4x1024x1024x1_float32",
            "op": "transpose",
            "family": "transpose",
            "shape": (4, 1024, 1024, 1),
            "dtype": "float32",
        },
        {
            "case": "transpose_4x1024x1024x1_bfloat16",
            "op": "transpose",
            "family": "transpose",
            "shape": (4, 1024, 1024, 1),
            "dtype": "bfloat16",
        },
        {
            "case": "transpose_1x2048x1024x1_float32",
            "op": "transpose",
            "family": "transpose",
            "shape": (1, 2048, 1024, 1),
            "dtype": "float32",
        },
        {
            "case": "transpose_1x2048x1024x1_bfloat16",
            "op": "transpose",
            "family": "transpose",
            "shape": (1, 2048, 1024, 1),
            "dtype": "bfloat16",
        },
        {
            "case": "transpose_4x128x256x4_float32",
            "op": "transpose",
            "family": "transpose",
            "shape": (4, 128, 256, 4),
            "dtype": "float32",
        },
        {
            "case": "transpose_4x128x256x4_bfloat16",
            "op": "transpose",
            "family": "transpose",
            "shape": (4, 128, 256, 4),
            "dtype": "bfloat16",
        },
        {
            "case": "transpose_1x513x257x1_float32",
            "op": "transpose",
            "family": "transpose",
            "shape": (1, 513, 257, 1),
            "dtype": "float32",
        },
        {
            "case": "transpose_1x513x257x1_bfloat16",
            "op": "transpose",
            "family": "transpose",
            "shape": (1, 513, 257, 1),
            "dtype": "bfloat16",
        },
    ]


def workload(case, seed):
    torch.manual_seed(seed)
    dtype = getattr(torch, case["dtype"])
    shape = tuple(case["shape"])
    x = torch.randn(shape, device="cuda", dtype=dtype)

    def fn(x):
        return x.permute(0, 2, 1, 3).contiguous()

    return fn, (x,)

"""Operator units for the reduction-indexing experiment."""

import torch


CONFIG = "triton.amd_reduction_indexing"
PATCH = {"triton.persistent_reductions": False}


def cases():
    return [
        {
            "case": "masked_gather_sum_4x4099_float32",
            "op": "masked_gather_sum",
            "family": "reduction-indexing",
            "shape": (4, 4099),
            "dtype": "float32",
        },
        {
            "case": "masked_gather_sum_4x4099_bfloat16",
            "op": "masked_gather_sum",
            "family": "reduction-indexing",
            "shape": (4, 4099),
            "dtype": "bfloat16",
        },
        {
            "case": "masked_gather_sum_8x4099_float32",
            "op": "masked_gather_sum",
            "family": "reduction-indexing",
            "shape": (8, 4099),
            "dtype": "float32",
        },
        {
            "case": "masked_gather_sum_8x4099_bfloat16",
            "op": "masked_gather_sum",
            "family": "reduction-indexing",
            "shape": (8, 4099),
            "dtype": "bfloat16",
        },
        {
            "case": "masked_gather_sum_4x32769_float32",
            "op": "masked_gather_sum",
            "family": "reduction-indexing",
            "shape": (4, 32769),
            "dtype": "float32",
        },
        {
            "case": "masked_gather_sum_4x32769_bfloat16",
            "op": "masked_gather_sum",
            "family": "reduction-indexing",
            "shape": (4, 32769),
            "dtype": "bfloat16",
        },
        {
            "case": "masked_gather_sum_8x32769_float32",
            "op": "masked_gather_sum",
            "family": "reduction-indexing",
            "shape": (8, 32769),
            "dtype": "float32",
        },
        {
            "case": "masked_gather_sum_8x32769_bfloat16",
            "op": "masked_gather_sum",
            "family": "reduction-indexing",
            "shape": (8, 32769),
            "dtype": "bfloat16",
        },
    ]


def workload(case, seed):
    torch.manual_seed(seed)
    dtype = getattr(torch, case["dtype"])
    shape = tuple(case["shape"])
    period = (shape[0] - 1) * shape[1] + 17
    x = torch.randn(period, device="cuda", dtype=dtype)

    def fn(x):
        index = torch.arange(shape[0] * shape[1], device=x.device).reshape(shape)
        return (
            torch.ops.aten._unsafe_masked_index(x, index < period, [index % period], 0)
            .float()
            .sum(-1)
        )

    return fn, (x,)

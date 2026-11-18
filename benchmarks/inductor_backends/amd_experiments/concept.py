"""Operator units for the pointwise-subtiling experiment."""

import torch


CONFIG = "triton.amd_pointwise_subtiling"


def cases():
    return [
        {
            "case": "scalar_4099_float32",
            "op": "scalar",
            "family": "pointwise-subtiling",
            "shape": (4099,),
            "dtype": "float32",
        },
        {
            "case": "scalar_4099_bfloat16",
            "op": "scalar",
            "family": "pointwise-subtiling",
            "shape": (4099,),
            "dtype": "bfloat16",
        },
        {
            "case": "scalar_4608000_float32",
            "op": "scalar",
            "family": "pointwise-subtiling",
            "shape": (4608000,),
            "dtype": "float32",
        },
        {
            "case": "scalar_4608000_bfloat16",
            "op": "scalar",
            "family": "pointwise-subtiling",
            "shape": (4608000,),
            "dtype": "bfloat16",
        },
        {
            "case": "cast_add_4608000_float32",
            "op": "cast_add",
            "family": "cast_add",
            "shape": (4608000,),
            "dtype": "float32",
        },
        {
            "case": "cast_add_4608000_bfloat16",
            "op": "cast_add",
            "family": "cast_add",
            "shape": (4608000,),
            "dtype": "bfloat16",
        },
        {
            "case": "swiglu_4608000_float32",
            "op": "swiglu",
            "family": "pointwise-subtiling",
            "shape": (4608000,),
            "dtype": "float32",
        },
        {
            "case": "swiglu_4608000_bfloat16",
            "op": "swiglu",
            "family": "pointwise-subtiling",
            "shape": (4608000,),
            "dtype": "bfloat16",
        },
        {
            "case": "scalar_16777216_float32",
            "op": "scalar",
            "family": "pointwise-subtiling",
            "shape": (16777216,),
            "dtype": "float32",
        },
        {
            "case": "scalar_16777216_bfloat16",
            "op": "scalar",
            "family": "pointwise-subtiling",
            "shape": (16777216,),
            "dtype": "bfloat16",
        },
        {
            "case": "swiglu_16777216_float32",
            "op": "swiglu",
            "family": "pointwise-subtiling",
            "shape": (16777216,),
            "dtype": "float32",
        },
        {
            "case": "swiglu_16777216_bfloat16",
            "op": "swiglu",
            "family": "pointwise-subtiling",
            "shape": (16777216,),
            "dtype": "bfloat16",
        },
    ]


def workload(case, seed):
    torch.manual_seed(seed)
    dtype = getattr(torch, case["dtype"])
    shape = tuple(case["shape"])
    op = case["op"]
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if op == "scalar":

        def fn(x, s):
            return (s * (1 / x.numel()) * (x.float() * 2)).to(x.dtype)

        return fn, (x, torch.randn((), device="cuda"))
    if op == "cast_add":

        def fn(x, y):
            return x.float() + y.float()
    else:

        def fn(x, y):
            return (torch.nn.functional.silu(x.float()) * y.float()).to(x.dtype)

    return fn, (x, torch.randn_like(x))

from __future__ import annotations

import sympy

import torch

from .. import config, ir
from ..codegen.common import BackendFeature
from ..runtime.hints import amd_scan_parameters, DeviceProperties
from ..select_algorithm import TritonTemplate
from ..virtualized import V


def _scan_grid(m, n, meta):
    return ((n + meta["B"] - 1) // meta["B"], m, 1)


def _totals_grid(m, chunks, meta):
    return (chunks, m, 1)


totals_template = TritonTemplate(
    name="amd_two_pass_scan_totals",
    grid=_totals_grid,
    source=r"""
{{def_kernel("X")}}
    row = tl.program_id(1).to(INDEX_DTYPE)
    chunk = tl.program_id(0).to(INDEX_DTYPE)
    n = {{size("X", 1)}}
    i = chunk * B + tl.arange(0, B)
    v = tl.load(X + row * {{stride("X", 0)}} + i * {{stride("X", 1)}}, i < n, other=0).to(tl.float64)
    total = tl.sum(v, 0)
    {{store_output(("row", "chunk"), "total", "True", val_shape=())}}
""",
)
scan_template = TritonTemplate(
    name="amd_two_pass_scan_output",
    grid=_scan_grid,
    source=r"""
{{def_kernel("X", "T")}}
    row = tl.program_id(1).to(INDEX_DTYPE)
    chunk = tl.program_id(0).to(INDEX_DTYPE)
    n = {{size("X", 1)}}
    k = {{size("T", 1)}}
    i = chunk * B + tl.arange(0, B)
    v = tl.load(X + row * {{stride("X", 0)}} + i * {{stride("X", 1)}}, i < n, other=0).to(tl.float64)
    local = tl.cumsum(v, 0)
    j = tl.arange(0, KB)
    carry = tl.sum(tl.load(T + row * k + j, j < chunk, other=0), 0)
    result = (local + carry).to(tl.float32)
    {{store_output(("row", "i"), "result", "i < n", val_shape=("B",))}}
""",
)


def two_pass_scan(
    x: ir.TensorBox, axis: int | None, dtype: torch.dtype | None
) -> ir.TensorBox | None:
    if (
        not config.triton.amd_two_pass_scan
        or torch.version.hip is None
        or x.get_device().type != "cuda"
        or not V.graph.has_feature(x.get_device(), BackendFeature.TRITON_TEMPLATES)
        or config.batch_invariant
        or config.deterministic
        or torch.are_deterministic_algorithms_enabled()
        or len(x.get_size()) != 2
        or axis not in (-1, 1)
        or (dtype or x.get_dtype()) != torch.float32
        or x.get_dtype() not in (torch.float16, torch.bfloat16, torch.float32)
        or not all(isinstance(n, sympy.Integer) for n in x.get_size())
    ):
        return None
    m, n = x.get_size()
    device = DeviceProperties.create(x.get_device())
    parameters = amd_scan_parameters(device, int(m), int(n))
    if parameters is None:
        return None
    block, warps = parameters
    chunks = (int(n) + block - 1) // block
    x.realize()
    totals = totals_template.generate(
        input_nodes=(x,),
        layout=ir.FixedLayout(
            x.get_device(), torch.float64, [m, sympy.Integer(chunks)]
        ),
        num_stages=1,
        num_warps=warps,
        B=block,
        ACC_TYPE="tl.float64",
    ).output_node()
    totals.realize()
    return scan_template.generate(
        input_nodes=(x, totals),
        layout=ir.FixedLayout(x.get_device(), torch.float32, [m, n]),
        num_stages=1,
        num_warps=warps,
        B=block,
        KB=1 << (chunks - 1).bit_length(),
    ).output_node()

# mypy: allow-untyped-defs
import functools
import logging
from typing import Any, Dict, Sequence, Tuple

import sympy

import torch
from torch._inductor.select_algorithm import realize_inputs
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .. import config as inductor_config
from ..codegen.wrapper import PythonWrapperCodegen
from ..ir import Layout
from ..runtime.runtime_utils import next_power_of_2
from ..utils import ceildiv as cdiv, get_num_sms, TMA_DESCRIPTOR_SIZE


log = logging.getLogger(__name__)


def triton_config(num_stages, num_warps, **kwargs):
    from triton import Config  # type: ignore[attr-defined]

    return Config(kwargs, num_stages=num_stages, num_warps=num_warps)


def filtered_configs(
    m: int,
    n: int,
    k: int,
    configs: Sequence[Tuple[int, int, int, int, int]],
    has_int8_tensor=False,
    scale=1,
    exclude=lambda m, n, k: False,
):
    """
    Heuristic to shrink configs when they are bigger than the input size

    :param scale: scale factor applied to the config values
    :param exclude: whether a given config should be excluded
    """
    from torch._inductor import config

    max_mm_configs = config.test_configs.max_mm_configs

    min_block_size = 16
    # block_k=16 seems to be causing issues
    # see: https://github.com/triton-lang/triton/issues/2156#issuecomment-1695897424
    min_block_size_k = 32 if has_int8_tensor else 16
    m = max(
        next_power_of_2(
            V.graph.sizevars.size_hint(
                m, fallback=torch._inductor.config.unbacked_symint_fallback  # type: ignore[arg-type]
            )
        ),
        min_block_size,
    )
    n = max(
        next_power_of_2(
            V.graph.sizevars.size_hint(
                n, fallback=torch._inductor.config.unbacked_symint_fallback  # type: ignore[arg-type]
            )
        ),
        min_block_size,
    )
    k = max(
        next_power_of_2(
            V.graph.sizevars.size_hint(
                k, fallback=torch._inductor.config.unbacked_symint_fallback  # type: ignore[arg-type]
            )
        ),
        min_block_size_k,
    )
    used = OrderedSet[tuple[int, int, int, int, int, int]]()
    for block_m, block_n, block_k, num_stages, num_warps in configs:
        # shrink configs for small sizes
        block_m = max(min(int(block_m * scale), m), min_block_size)
        block_n = max(min(int(block_n * scale), n), min_block_size)
        block_k = max(min(int(block_k * scale), k), min_block_size_k)

        if exclude(block_m, block_n, block_k):
            continue

        # each warp computes 16x16 tile = 256
        num_warps = min(num_warps, block_m * block_n // 256)
        if torch.version.hip:
            for matrix_instr_nonkdim in [0, 16]:
                if matrix_instr_nonkdim != 0 and (
                    block_m % matrix_instr_nonkdim != 0
                    or block_n % matrix_instr_nonkdim != 0
                ):
                    #  block_m and block_n must be a multiple of matrix_instr_nonkdim
                    continue
                if (
                    block_m,
                    block_n,
                    block_k,
                    num_stages,
                    num_warps,
                    matrix_instr_nonkdim,
                ) not in used and (
                    max_mm_configs is None or len(used) < max_mm_configs
                ):
                    used.add(
                        (
                            block_m,
                            block_n,
                            block_k,
                            num_stages,
                            num_warps,
                            matrix_instr_nonkdim,
                        )
                    )
                    yield triton_config(
                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                        BLOCK_K=block_k,
                        num_stages=num_stages,
                        num_warps=num_warps,
                        matrix_instr_nonkdim=matrix_instr_nonkdim,
                    )
        else:
            if (block_m, block_n, block_k, num_stages, num_warps, 0) not in used and (
                max_mm_configs is None or len(used) < max_mm_configs
            ):
                used.add((block_m, block_n, block_k, num_stages, num_warps, 0))
                yield triton_config(
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    num_stages=num_stages,
                    num_warps=num_warps,
                )


if inductor_config.max_autotune_custom_heuristic is None:
    mm_heuristics = V.choices.config_heuristics
else:
    mm_heuristics = inductor_config.max_autotune_custom_heuristic

if inductor_config.max_autotune_gemm_search_space != "EXHAUSTIVE":
    mm_kernel_configs = mm_heuristics.get_mm_configs()
else:
    mm_kernel_configs = mm_heuristics.get_exhaustive_mm_configs()

extra_mm_kernel_configs = mm_heuristics.get_extra_mm_configs()
int8_mm_kernel_configs = mm_heuristics.get_int8_mm_configs()
mixed_mm_kernel_configs_small_m = mm_heuristics.get_mixed_mm_configs()
persistent_mm_kernel_configs = mm_heuristics.get_persistent_mm_configs()
scaled_mm_kernel_configs = mm_heuristics.get_scaled_mm_configs()
scaled_persistent_mm_kernel_configs = mm_heuristics.get_scaled_persistent_mm_configs()

mixed_mm_kernel_configs = (
    mm_kernel_configs + mixed_mm_kernel_configs_small_m
    if inductor_config.max_autotune_gemm_search_space != "EXHAUSTIVE"
    else mm_kernel_configs
)

mm_configs = functools.partial(
    filtered_configs,
    configs=mm_kernel_configs,
)

extra_mm_configs = functools.partial(
    filtered_configs,
    configs=extra_mm_kernel_configs,
)

int8_mm_configs = functools.partial(
    filtered_configs,
    configs=int8_mm_kernel_configs,
)

mixed_mm_configs = functools.partial(
    filtered_configs,
    configs=mixed_mm_kernel_configs,
)

persistent_mm_configs = functools.partial(
    filtered_configs,
    configs=persistent_mm_kernel_configs,
)

scaled_mm_configs = functools.partial(
    filtered_configs,
    configs=scaled_mm_kernel_configs,
)

scaled_persistent_mm_configs = functools.partial(
    filtered_configs,
    configs=scaled_persistent_mm_kernel_configs,
)


def mm_grid(m, n, meta):
    """
    The CUDA grid size for matmul triton templates.
    """
    return (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), 1, 1)


def persistent_mm_grid(M: int, N: int, meta: Dict[str, Any]):
    """Defines the grid for persistent kernels."""
    return (
        min(meta["NUM_SMS"], cdiv(M, meta["BLOCK_M"]) * cdiv(N, meta["BLOCK_N"])),
        1,
        1,
    )


def acc_type(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return "tl.float32"
    return f"tl.{dtype}".replace("torch.", "")


def mm_options(config, sym_m, sym_n, sym_k, layout, b_prologue_cast_type=None):
    """
    Common options to matmul triton templates.
    """
    even_k_symbolic = (
        # it isn't worth guarding on this
        sympy.gcd(sym_k, config.kwargs["BLOCK_K"])
        == config.kwargs["BLOCK_K"]
    )
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32 and (
        not inductor_config.force_same_precision
        or ((sym_m % 16) == 0 and (sym_n % 16) == 0 and (sym_k % 8) == 0)
    )
    return dict(
        GROUP_M=8,
        EVEN_K=even_k_symbolic,
        ALLOW_TF32=allow_tf32,
        ACC_TYPE=acc_type(layout.dtype),
        B_PROLOGUE_CAST_TYPE=b_prologue_cast_type,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        **config.kwargs,
    )


def persistent_mm_options(mat1, mat2):
    return dict(
        A_ROW_MAJOR=not mat1.layout.is_transposed(),
        B_ROW_MAJOR=not mat2.layout.is_transposed(),
        NUM_SMS=get_num_sms(),
        TMA_SIZE=TMA_DESCRIPTOR_SIZE,
    )


def mm_args(
    mat1,
    mat2,
    *others,
    layout=None,
    out_dtype=None,
    use_4x2_dim=False,
    mat2_transposed=False,
):
    """
    Common arg processing for mm,bmm,addmm,etc
    """
    mat1, mat2 = realize_inputs(mat1, mat2)
    *b1, m, k1 = mat1.get_size()
    if mat2_transposed:
        *b2, n, k2 = mat2.get_size()
    else:
        *b2, k2, n = mat2.get_size()
    b = [V.graph.sizevars.guard_equals(a, b) for a, b in zip(b1, b2)]
    if use_4x2_dim:
        k2 = k2 * 2
    k = V.graph.sizevars.guard_equals(k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout

        if out_dtype is None:
            out_dtype = mat1.get_dtype()

        layout = FixedLayout(
            mat1.get_device(),
            out_dtype,
            [*b, m, n],
        )
    else:
        assert out_dtype is None, "out_dtype is ignored if layout is specified."
    from ..lowering import expand

    others = [realize_inputs(expand(x, layout.size)) for x in others]

    return [m, n, k, layout, mat1, mat2, *others]


def addmm_epilogue(dtype, alpha, beta):
    def epilogue(acc, bias):
        if alpha != 1:
            acc = V.ops.mul(acc, V.ops.constant(alpha, dtype))
        if beta != 1:
            bias = V.ops.mul(bias, V.ops.constant(beta, dtype))
        return V.ops.add(acc, bias)

    return epilogue


def _is_static_problem(layout: Layout) -> Tuple[bool, bool]:
    """
    Check if input tensors and output layout have static shapes and non-zero sizes.

    Args:
        layout: Output layout object with a 'size' attribute.

    Returns:
        Tuple[bool, bool]: (is_static, is_nonzero)
            is_static: True if all shapes are statically known
            is_nonzero: True if all dimensions are non-zero
    """
    static_shape = True
    static_size = PythonWrapperCodegen.statically_known_list_of_ints_or_none(
        layout.size
    )
    if static_size is None:
        nonzero = True
        for s in layout.size:
            sz = PythonWrapperCodegen.statically_known_int_or_none(s)
            if sz is not None and sz == 0:
                nonzero = False
                break
        return False, nonzero
    numel = 1
    for dim in static_size:
        numel *= dim
    nonzero = numel > 0
    return static_shape, nonzero

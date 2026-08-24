# mypy: allow-untyped-defs
"""Call into FlyDSL's flex flash-attention kernels for flex_attention (ROCm).

The FlyDSL analog of ``flex_flash_attention.py``. The kernels this targets are vendored
under ``torch._inductor.kernel.vendored_templates.flydsl.flex_kernels``: derived copies
of FlyDSL's flash kernels carrying ``score_mod`` / ``mask_mod`` hooks, a log-sum-exp
output, a BlockMask-driven KV-block-skip loop, and captured-tensor reads.

Forward only. There is no FlyDSL flash-attention backward kernel for any configuration,
so a graph that needs gradients must fall back to Triton.
"""

from __future__ import annotations

import functools
from typing import Any, Literal, TYPE_CHECKING

import sympy

import torch

from ...codegen.flydsl.flydsl_utils import (
    _flydsl_runtime_unavailable_reason,
    runtime_available,
)
from ...ir import FixedLayout
from ...lowering import empty_strided
from .common import infer_dense_strides, load_flex_template


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...ir import Subgraph


# Architectures the flex kernels are known to build for. gfx942 is the validated one;
# gfx950 carries the same hooks but has never been executed, so it is opt-in via
# FLYDSL_FLEX_ALLOW_UNVALIDATED_ARCH rather than silently trusted.
_VALIDATED_ARCHS = frozenset({"gfx942"})
_PORTED_ARCHS = frozenset({"gfx950"})

# Head dims the kernel is *correct* for, which is narrower than what it accepts.
#
# The vendored kernel documents "head_dim % 32 == 0, head_dim >= 64", but every
# configuration in its own test suite uses 128, and that claim does not hold: at
# head_dim=64 it builds happily and returns wrong numbers (~44% relative error on plain
# attention with no mods, reproduced against the untouched upstream kernel by
# phase1/probe_upstream_head_dim.py). Other multiples of 32 assert during build, which is
# at least a safe failure.
#
# So this is an allowlist rather than a rule. head_dim=64 is the common case for many
# models, so getting this wrong would route a lot of traffic to a kernel that returns
# plausible garbage instead of falling back to Triton.
_SUPPORTED_HEAD_DIMS = frozenset({128})

_SUPPORTED_DTYPES = frozenset({torch.bfloat16, torch.float16})


@functools.lru_cache(maxsize=1)
def flydsl_unavailable_reason() -> str | None:
    """Why the FlyDSL flex kernels cannot be used here, or None if they can.

    Cached; call ``flydsl_unavailable_reason.cache_clear()`` after installing something
    in the same interpreter to retry.
    """
    if torch.version.hip is None:
        return "FlyDSL flex kernels require a ROCm build of PyTorch"

    # The generic backend's check is the authority on whether FlyDSL itself can run: as
    # well as importability it verifies the _mlir extension, the JIT runtime .so, and
    # that the .so's own shared-library dependencies resolve. A FlyDSL install can be
    # importable and still fail at compile time on all three counts.
    if not runtime_available():
        return _flydsl_runtime_unavailable_reason() or "the FlyDSL runtime is unavailable"

    return None


def ensure_flydsl_available() -> bool:
    """Whether FlyDSL and the flex kernel package can serve a lowering here."""
    return flydsl_unavailable_reason() is None


def _current_arch() -> str | None:
    """gfx name of the current device, or None if this is not a ROCm GPU build."""
    if not torch.version.hip:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def _arch_supported(arch: str | None) -> tuple[bool, str]:
    if arch is None:
        return False, "FlyDSL flex kernels require a ROCm build with a visible device"
    if arch in _VALIDATED_ARCHS:
        return True, ""
    if arch in _PORTED_ARCHS:
        # The gfx950 kernel is a mechanical port that has never been run. Treat "written"
        # as "unsupported" until someone validates it, rather than shipping unexecuted
        # code to users by default.
        if torch._inductor.config.flydsl.allow_unvalidated_arch:
            return True, ""
        return (
            False,
            f"FlyDSL flex kernel for {arch} has not been validated on hardware; set "
            "torch._inductor.config.flydsl.allow_unvalidated_arch=True to try it anyway",
        )
    return False, f"FlyDSL flex kernels do not support {arch}"


def _head_dim_supported(query, value) -> tuple[bool, str]:
    from ...virtualized import V

    qk_head_dim = V.graph.sizevars.optimization_hint(query.get_size()[-1])
    v_head_dim = V.graph.sizevars.optimization_hint(value.get_size()[-1])
    supported = sorted(_SUPPORTED_HEAD_DIMS)
    for name, dim in (("qk", qk_head_dim), ("v", v_head_dim)):
        if dim not in _SUPPORTED_HEAD_DIMS:
            return (
                False,
                f"{name} head_dim {dim} is not supported by the FlyDSL flex kernels "
                f"(supported: {supported})",
            )
    if qk_head_dim != v_head_dim:
        return (
            False,
            f"FlyDSL flex kernels require matching qk/v head_dim, got "
            f"{qk_head_dim} and {v_head_dim}",
        )
    return True, ""


def _can_use_flydsl_flash_attention(
    query,
    key,
    value,
    subgraph: Subgraph,
    mask_graph: Subgraph,
    num_score_mod_placeholders: int,
) -> tuple[bool, str]:
    """Whether the FlyDSL flex kernels can serve this lowering.

    Returns ``(can_use, reason)``; ``reason`` explains the rejection when it cannot.
    """
    from .flex_flash_attention import input_buffers_require_grads

    unavailable = flydsl_unavailable_reason()
    if unavailable is not None:
        return False, f"FlyDSL flex kernels are unavailable: {unavailable}"

    arch_ok, arch_reason = _arch_supported(_current_arch())
    if not arch_ok:
        return False, arch_reason

    dtype = query.get_dtype()
    if dtype not in _SUPPORTED_DTYPES:
        return False, f"FlyDSL flex kernels support bf16/f16, got {dtype}"
    if key.get_dtype() != dtype or value.get_dtype() != dtype:
        return (
            False,
            "FlyDSL flex kernels require query/key/value to share a dtype, got "
            f"{dtype}, {key.get_dtype()}, {value.get_dtype()}",
        )

    head_dim_ok, head_dim_reason = _head_dim_supported(query, value)
    if not head_dim_ok:
        return False, head_dim_reason

    # Forward only: there is no FlyDSL flash backward kernel at all.
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (query, key, value) if hasattr(t, "requires_grad")
    ):
        return False, "FlyDSL flex kernels are forward-only (no backward kernel exists)"

    if input_buffers_require_grads(subgraph.graph_module, num_score_mod_placeholders):
        return False, "Captured buffers require gradients (forward-only backend)"

    return True, ""


def _use_flydsl_flash_attention(
    query,
    key,
    value,
    subgraph: Subgraph,
    mask_graph: Subgraph,
    kernel_options: dict[str, Any],
    num_score_mod_placeholders: int,
    backend: Literal["AUTO", "TRITON", "FLASH", "TRITON_DECODE", "FLYDSL"],
) -> bool:
    """Whether to route this lowering to FlyDSL.

    Experimental and never selected by ``AUTO``: it has to be asked for. When it is
    asked for and cannot be served, raise with the reason rather than silently falling
    through, matching how ``BACKEND='FLASH'`` behaves.
    """
    if backend != "FLYDSL":
        return False

    can_use, reason = _can_use_flydsl_flash_attention(
        query,
        key,
        value,
        subgraph,
        mask_graph,
        num_score_mod_placeholders,
    )
    if not can_use:
        raise RuntimeError(
            f"BACKEND='FLYDSL' but the FlyDSL flex kernels cannot be used: {reason}"
        )
    return True


def create_flex_flydsl_attention_kernel(
    query,
    key,
    value,
    block_mask,
    scale,
    kernel_options,
    score_mod_subgraph,
    mask_graph_subgraph,
    score_mod_other_buffers,
    mask_mod_other_buffers,
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    sparse_q_block_size,
    sparse_kv_block_size,
    *,
    mask_graph,
    subgraph,
):
    """Build the FlyDSL flex attention kernel choice.

    The FlyDSL analog of ``create_flex_flash_attention_kernel``, and deliberately shaped
    like it: assemble ``input_nodes``, hand the lowered subgraphs to a template, let it
    render, then select through ``autotune_select_algorithm``. Returns ``(out, lse)``.

    Where it diverges from the CuteDSL path, and why:

    - **No block-mask inputs**, which is why the block-mask parameters below are accepted
      and unused. This kernel runs one body over partial and fully-unmasked blocks alike,
      so it wants the union of the two block lists as a single array rather than the two
      FlexAttention supplies. Computing that union is host-side work on every call, so the
      KV-skip path waits for the step that plumbs BlockMask properly, which is what those
      parameters are kept for. Here the walk is dense with mask_mod applied to every
      element: correct, but it gives up the sparsity win, which on a long causal sequence
      is most of the point.
    - **One choice, not a config sweep.** The only tunable the kernel exposes is
      ``mod_vec_size``, and its useful range is {1, 2, 4}. Benchmarking is worthwhile but
      it is autotuning over three points, not a search.
    """
    from ...select_algorithm import autotune_select_algorithm
    from .flex_flash_attention import (
        is_trivial_mask_graph,
        is_trivial_score_graph,
        patch_fixed_layout_indexer_for_cutedsl,
        wrap_choice_render_with_cutedsl_indexer,
    )

    unavailable = flydsl_unavailable_reason()
    if unavailable is not None:
        raise RuntimeError(f"FlyDSL flex kernels are unavailable: {unavailable}")

    batch_size, num_heads, seq_len_q, qk_head_dim = query.get_size()
    num_kv_heads = key.get_size()[1]
    v_head_dim = value.get_size()[-1]
    device = query.get_device()
    dtype = query.get_dtype()
    if device is None:
        raise AssertionError("Device must be specified")

    has_score_mod = subgraph is not None and not is_trivial_score_graph(
        subgraph.graph_module
    )
    has_mask_mod = not is_trivial_mask_graph(mask_graph.graph_module)

    captures = list(score_mod_other_buffers) + list(mask_mod_other_buffers)
    _reject_unsupported_captures(captures)

    q_strides = query.get_stride()
    out_size = [batch_size, num_heads, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, q_strides)
    output = empty_strided(
        size=out_size, stride=out_strides, dtype=dtype, device=device
    )
    output_layout = FixedLayout(
        device=device,
        dtype=dtype,
        size=out_size,
        stride=[sympy.sympify(s) for s in output.get_stride()],
    )

    # LSE is a mutated input rather than a second output: the kernel writes it in place,
    # and natural log rather than log2, which is why FLYDSL is in
    # _NATURAL_LOG_LSE_BACKENDS on the wrapper side.
    lse = empty_strided(
        size=[batch_size, num_heads, seq_len_q],
        stride=None,
        dtype=torch.float32,
        device=device,
    )

    # Captures are deliberately absent: a mod's captures are discovered while it is
    # lowered, and the template registers them as kernel inputs then, so `generate()`
    # appends them here afterwards. Declaring them up front would also pass -- and
    # per call convert -- captures that no mod turns out to read.
    input_nodes = [query, key, value, lse]

    subgraphs = []
    if has_score_mod:
        subgraphs.append(score_mod_subgraph)
    if has_mask_mod:
        subgraphs.append(mask_graph_subgraph)

    choices: list[Any] = []
    error: NotImplementedError | None = None
    vec_sizes = _mod_vec_sizes(kernel_options, has_mod=has_score_mod or has_mask_mod)
    for mod_vec_size in vec_sizes:
        with patch_fixed_layout_indexer_for_cutedsl():
            error = _flydsl_flash_attention_template().maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=output_layout,
                mutated_inputs=[lse],
                subgraphs=subgraphs,
                SM_SCALE=scale,
                NUM_HEADS=num_heads,
                NUM_KV_HEADS=num_kv_heads,
                HEAD_DIM=qk_head_dim,
                DTYPE_STR=_FLYDSL_DTYPE_STR[dtype],
                HAS_SCORE_MOD=has_score_mod,
                HAS_MASK_MOD=has_mask_mod,
                MOD_VEC_SIZE=mod_vec_size,
            )
        if error is not None and len(vec_sizes) == 1:
            raise RuntimeError(f"FlyDSL template failed: {error}")

    if not choices:
        raise RuntimeError(f"FlyDSL template failed: {error}")

    # Captured-tensor reads are addressed per axis, not by a flattened offset, so the
    # deferred render at scheduling time needs the same indexer the choices were built
    # with. See Note [CuteDSL indexer patch].
    for choice in choices:
        wrap_choice_render_with_cutedsl_indexer(choice)

    # Autotune against what the choices actually take, not what was declared: rendering
    # appended the captures the mods read, and benchmarking builds one example tensor per
    # entry, so passing the declared list would call the kernel a tensor short. Every
    # choice renders the same mods, so they agree on the captures.
    autotune_input_nodes = choices[0].input_nodes
    if any(choice.input_nodes != autotune_input_nodes for choice in choices):
        raise AssertionError(
            "FlyDSL flex choices disagree on their inputs: "
            f"{[[n.get_name() for n in c.input_nodes] for c in choices]}"
        )

    template_output, _ = autotune_select_algorithm(
        "flex_flydsl_attention",
        choices,
        autotune_input_nodes,
        output_layout,
        return_multi_template=False,
    )

    return (template_output, lse)


_FLYDSL_DTYPE_STR = {
    torch.bfloat16: "bf16",
    torch.float16: "f16",
}


@functools.lru_cache(maxsize=1)
def _flydsl_flash_attention_template():
    """The registered FlyDSL flex template, built on first use.

    Deferred rather than built at import: reading the template file and compiling the
    jinja costs nothing here, but registration is global, and this module is imported on
    every ROCm Inductor run whether or not FLYDSL was asked for.
    """
    from ...codegen.flydsl.flydsl_flex_kernel import FlyDSLFlexTemplate

    return FlyDSLFlexTemplate(
        name="flydsl_flash_attention",
        source=load_flex_template("flydsl_flash_attention"),
    )


def _mod_vec_sizes(kernel_options: dict[str, Any], *, has_mod: bool) -> list[int]:
    """Vectorization widths to offer as autotune choices.

    The kernel evaluates the mods over 1, 2 or 4 contiguous KV columns at a time; 4 is
    the ceiling because the score layout only guarantees that many contiguous columns per
    lane. A wider mod amortises the per-call overhead but costs registers, and which wins
    depends on the mod, so it is measured rather than predicted.
    """
    requested = kernel_options.get("MOD_VEC_SIZE")
    if requested is not None:
        if requested not in (1, 2, 4):
            raise RuntimeError(f"MOD_VEC_SIZE must be 1, 2 or 4, got {requested}")
        return [int(requested)]
    if not has_mod or not torch._inductor.config.flydsl.autotune_mod_vec_size:
        # With no mod there is nothing to vectorize, so the three choices would compile
        # three identical kernels.
        return [1]
    return [1, 2, 4]


def _reject_unsupported_captures(captures: Sequence[Any]) -> None:
    """Reject captures the kernel cannot address before anything is rendered.

    Aux slots are a hard limit rather than a tuning choice: the kernel's signature carries
    exactly two, so a third capture has nowhere to go.
    """
    if len(captures) > _MAX_AUX_TENSORS:
        raise RuntimeError(
            f"BACKEND='FLYDSL' supports at most {_MAX_AUX_TENSORS} captured tensors "
            f"across score_mod and mask_mod, got {len(captures)}"
        )
    for buf in captures:
        rank = len(buf.get_size())
        if rank > 4:
            raise RuntimeError(
                f"captured tensor {buf.get_name()} has rank {rank}; the FlyDSL aux "
                "reader addresses at most (b, h, q, kv)"
            )


# The kernel signature carries two aux buffers.
_MAX_AUX_TENSORS = 2

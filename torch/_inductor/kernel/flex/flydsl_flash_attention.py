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
import itertools
from typing import Any, Literal, TYPE_CHECKING

import sympy

import torch

from ...codegen.flydsl.flydsl_utils import (
    _flydsl_runtime_unavailable_reason,
    runtime_available,
)
from ...ir import FixedLayout
from ...lowering import empty_strided, lowerings
from .common import (
    create_indices_fake,
    create_num_blocks_fake_generator,
    infer_dense_strides,
    load_flex_template,
    maybe_realize,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...ir import Subgraph


# Architectures the flex kernels are known to build for. gfx942 is the validated one;
# gfx950 carries the same hooks but has never been executed, so it is opt-in through
# config.flydsl.allow_unvalidated_arch rather than silently trusted.
_VALIDATED_ARCHS = frozenset({"gfx942"})
_PORTED_ARCHS = frozenset({"gfx950"})

# Head dims the kernel is *correct* for, which is narrower than what it accepts. The
# vendored kernel documents "head_dim % 32 == 0, head_dim >= 64" and that claim does not
# hold, so this stays an allowlist.
#
# What bounds it now is LDS: head_dim 288 needs 73728 B against gfx942's 65536 B, and the
# kernel rejects anything below 64 or not a multiple of 32. Everything in between works.
#
# 96, 160, 192 and 224 were excluded until the cooperative KV load learned to bound its
# LDS row, which covers both the partial lane group left when the workgroup does not divide
# into whole rows and the short final batch when the tile does not either. That alone still
# gave wrong answers, because K's XOR swizzle is only a permutation over a power-of-two
# extent: at head_dim 96 the row mask is 0b101 and `col ^ 80` leaves the row from col 32 up.
# The swizzle is off for those head_dims, so K takes bank conflicts and they run 25-55%
# below the power-of-two dims per FLOP. That is worth having -- they used to be refused
# outright -- but it is why they are not fast.
#
# 256 is supported as of the deferred-rescale fix. It was previously excluded for needing
# 66560 B of LDS against gfx942's 65536 B, and the V swizzle removed exactly the 1024 B of
# V padding that overshot, which then exposed a second bug underneath: the online-softmax
# rescale of the O accumulators is deferred into the GEMM2 loop and was keyed off the PV
# k-step, capping it at 4 rescales, so chunks 5-7 were never scaled and everything from
# d160 up was wrong by ~42%. It now fits at exactly 65536 B and matches head_dim 128's
# error to four digits.
_SUPPORTED_HEAD_DIMS = frozenset({64, 96, 128, 160, 192, 224, 256})

# The backward reaches the same head_dims, but it gets there differently and that is worth
# knowing before someone changes its tile. Its dq kernel needs K in both orientations --
# row-major as GEMM1's A operand, transposed as GEMM2's -- so it stages three KV tiles
# where the forward stages two, and at the forward's BLOCK_N of 64 that overran gfx942's
# 65536 B from head_dim 160 up. Its KV tile is 32 instead, which fits the whole ladder and
# was independently the faster choice (see `block_n` in flex_flash_bwd_generic).

# Per-workgroup LDS on gfx942/gfx950, which is what bounds a backward tile choice.
_LDS_LIMIT = 65536

_SUPPORTED_DTYPES = frozenset({torch.bfloat16, torch.float16})

_FLYDSL_DTYPE_STR = {
    torch.bfloat16: "bf16",
    torch.float16: "f16",
}

# Captured-tensor slots the kernel signature carries. Kept in step with
# `flex_flash_generic.MAX_AUX_TENSORS`, which is the definition; this copy exists so
# lowering can reject an over-capture without importing the kernel module, which must
# stay importable only where FlyDSL is installed.
_MAX_AUX_TENSORS = 4

# FlexAttention signals "no block mask" with this as both block sizes.
_NOOP_BLOCK_SIZE = 1 << 30


@functools.lru_cache(maxsize=1)
def flydsl_unavailable_reason() -> str | None:
    """Why the FlyDSL flex kernels cannot be used here, or None if they can.

    Cached; call ``flydsl_unavailable_reason.cache_clear()`` after installing something
    in the same interpreter to retry.
    """
    if torch.version.hip is None:
        return "FlyDSL flex kernels require a ROCm build of PyTorch"

    # The generic backend's check is the authority on whether FlyDSL itself can run: it
    # verifies the _mlir extension, the JIT runtime .so and that .so's own dependencies,
    # all of which an importable FlyDSL install can still fail at compile time.
    if not runtime_available():
        return (
            _flydsl_runtime_unavailable_reason() or "the FlyDSL runtime is unavailable"
        )

    return None


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


def _block_mask_usable(
    kv_num_blocks,
    kv_indices,
    sparse_q_block_size,
    sparse_kv_block_size,
    *,
    has_mask_mod,
    has_full_blocks,
) -> tuple[bool, str]:
    """Whether the kernel's KV block-skip loop can be driven for this lowering.

    Returns ``(usable, reason)``. Declining is always safe: the dense walk applies
    ``mask_mod`` to every element and gets the same answer, just without the sparsity.
    """
    from ...virtualized import V

    if kv_num_blocks is None or kv_indices is None:
        return False, "no block mask was supplied"

    # The kernel runs one body over every block it visits, with mask_mod applied, and has
    # no separate body for fully-unmasked ones -- so the builder rejects block_mask
    # without a mask_mod, and there would be nothing to skip anyway.
    if not has_mask_mod:
        return False, "block skipping needs a mask_mod"

    if not has_full_blocks and V.graph.sizevars.statically_known_true(
        sympy.And(
            sympy.Eq(sparse_q_block_size, _NOOP_BLOCK_SIZE),
            sympy.Eq(sparse_kv_block_size, _NOOP_BLOCK_SIZE),
        )
    ):
        return False, "the block mask is a no-op"

    # The host re-expresses the mask on the kernel's own (BLOCK_M, BLOCK_N_OUT) grid,
    # which needs one block size to divide the other on each axis. The kernel's tiles are
    # powers of two, so requiring the same of the mask is sufficient -- and checkable
    # here, before a launcher exists to ask.
    for name, size in (("q", sparse_q_block_size), ("kv", sparse_kv_block_size)):
        block = V.graph.sizevars.guard_int(size)
        if block <= 0 or block & (block - 1):
            return False, f"sparse_{name}_block_size {block} is not a power of two"
    return True, ""


def _kv_seq_lens_agree(key, value) -> tuple[bool, str]:
    """Whether key and value agree on a sequence length. Query may differ from both.

    Query differing is cross attention, and the kernels take the two extents separately:
    the forward tiles Q and walks KV, ``dq`` tiles Q and walks KV, ``dkdv`` tiles KV and
    walks Q, so each grid is sized by its own axis. Key and value must agree with *each
    other*, because one column index reads both.
    """
    from ...virtualized import V

    seq_len_kv = key.get_size()[-2]
    if not V.graph.sizevars.statically_known_true(
        sympy.Eq(seq_len_kv, value.get_size()[-2])
    ):
        return (
            False,
            "FlyDSL flex kernels require key and value to share a sequence length, got "
            f"{seq_len_kv} and {value.get_size()[-2]}",
        )
    return True, ""


def _can_use_flydsl_shapes_and_arch(query, key, value) -> tuple[bool, str]:
    """The checks both directions share: availability, architecture, dtype, head_dim."""
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

    seq_ok, seq_reason = _kv_seq_lens_agree(key, value)
    if not seq_ok:
        return False, seq_reason

    return _head_dim_supported(query, value)


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

    shared_ok, shared_reason = _can_use_flydsl_shapes_and_arch(query, key, value)
    if not shared_ok:
        return False, shared_reason

    # Note [FlyDSL forward and backward must be chosen together]
    # This forward writes LSE in natural log, where Triton's backward reads log2 -- it
    # does `exp2(scores - lse)`. So a FlyDSL forward paired with a Triton backward is not
    # a slower answer, it is a wrong one, and the pairing has to be prevented.
    #
    # It cannot be prevented from here. By lowering time query/key/value are Inductor IR
    # nodes, which carry no `requires_grad`, and the forward graph holds no signal that a
    # backward will follow; a `requires_grad` test here is simply dead code. So the
    # enforcement lives on the other side: `_use_flydsl_flash_attention_backward` raises
    # rather than returning False whenever FLYDSL was asked for and the backward cannot
    # serve the graph. The pair therefore cannot come apart -- an unsupported graph fails
    # loudly at the backward instead of quietly computing wrong gradients.
    if input_buffers_require_grads(subgraph.graph_module, num_score_mod_placeholders):
        return False, "FlyDSL has no gradients for captured buffers"

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


def _can_use_flydsl_backward_features(
    captured_grads: Sequence[Any],
    mutated_grads: Sequence[Any],
) -> tuple[bool, str]:
    """Whether the backward covers what this graph needs beyond shapes and arch.

    Separate from the shape and architecture checks so that what the *backward* has yet to
    implement reads as one list. As of P2c that list is one item: ``score_mod`` and
    ``mask_mod`` are supported (the joint graph is evaluated at the ``ds`` site), and only
    gradients *with respect to* a captured tensor are not.
    """
    # Gradients for captured buffers need the joint graph's `zeros_and_scatter` outputs
    # accumulated with atomics, which is the scatter-graph path `modification` still
    # refuses. FLASH's backward declines these the same way. Note this is narrower than it
    # sounds: a mod may *read* a captured tensor freely -- that is the aux-slot path, and
    # it works -- what is refused is a capture that itself requires grad.
    if captured_grads or mutated_grads:
        return False, "no gradients for captured buffers yet"

    return True, ""


def _can_use_flydsl_flash_attention_backward(
    query,
    key,
    value,
    joint_outputs,
) -> tuple[bool, str]:
    """Whether the FlyDSL backward can serve this lowering.

    The mods themselves are not inspected here: an op the shim cannot lower surfaces when
    the template renders, exactly as it does on the forward path, and duplicating that
    knowledge in the gate is how the two drift apart.
    """
    from ...virtualized import V

    shared_ok, shared_reason = _can_use_flydsl_shapes_and_arch(query, key, value)
    if not shared_ok:
        return False, shared_reason

    features_ok, features_reason = _can_use_flydsl_backward_features(
        captured_grads=joint_outputs.captured_grads_compute,
        mutated_grads=joint_outputs.mutated_grads,
    )
    if not features_ok:
        return False, features_reason

    # A broadcast KV batch would need dk/dv summed back down to Bkv after the kernel.
    # The forward has no such case to handle, so it is refused here rather than half-done.
    batch_q = query.get_size()[0]
    batch_kv = key.get_size()[0]
    if not V.graph.sizevars.statically_known_equals(batch_q, batch_kv):
        return (
            False,
            f"a broadcast KV batch is not supported (Bq={batch_q}, Bkv={batch_kv})",
        )

    return True, ""


def _use_flydsl_flash_attention_backward(
    query,
    key,
    value,
    joint_outputs,
    backend: str,
) -> bool:
    """Whether to route this backward lowering to FlyDSL.

    Raises rather than returning False when FLYDSL was asked for and cannot be served: a
    silent fall-through would hand Triton's backward the forward's natural-log LSE. See
    Note [FlyDSL forward and backward must be chosen together].
    """
    if backend != "FLYDSL":
        return False

    can_use, reason = _can_use_flydsl_flash_attention_backward(
        query, key, value, joint_outputs
    )
    if not can_use:
        raise RuntimeError(
            f"BACKEND='FLYDSL' but the FlyDSL flex backward cannot be used: {reason}"
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
    """Build the FlyDSL flex attention kernel choice, returning ``(out, lse)``.

    The FlyDSL analog of ``create_flex_flash_attention_kernel`` and shaped like it: build
    ``input_nodes``, hand the lowered subgraphs to a template, render, then select through
    ``autotune_select_algorithm``. Two differences from the CuteDSL path:

    - The block mask drives the kernel's KV block-skip loop where it can. The kernel runs
      one body over partial and fully-unmasked blocks alike, so it wants the union of the
      two block lists on its own tile grid; the template re-expresses FlexAttention's two
      lists that way per call. Where that does not apply the walk falls back to dense with
      mask_mod on every element, which is the same answer without the sparsity.
    - The autotune space is a few independent tunables (``mod_vec_size``, the Q tile, and
      the QK prefetch depth when asked for), not a config sweep.
    """
    from ...select_algorithm import autotune_select_algorithm
    from ...virtualized import V
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

    _reject_unsupported_captures(score_mod_other_buffers, mask_mod_other_buffers)

    q_strides = query.get_stride()
    layout_str = _kernel_layout(kernel_options, query, key, value)
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

    has_full_blocks = full_kv_num_blocks is not None and full_kv_indices is not None
    use_block_mask, _block_mask_reason = _block_mask_usable(
        kv_num_blocks,
        kv_indices,
        sparse_q_block_size,
        sparse_kv_block_size,
        has_mask_mod=has_mask_mod,
        has_full_blocks=has_full_blocks,
    )

    # Captures are deliberately absent: they are discovered while a mod is lowered and
    # registered as kernel inputs then, so `generate()` appends them afterwards. Declaring
    # them up front would also pass -- and per call convert -- captures no mod reads.
    input_nodes = [query, key, value, lse]
    if use_block_mask:
        # Order fixes the input_gen_fns indices below, and the captures appended during
        # rendering land after these.
        input_nodes.extend([kv_num_blocks, kv_indices])
        if has_full_blocks:
            input_nodes.extend([full_kv_num_blocks, full_kv_indices])

    subgraphs = []
    if has_score_mod:
        subgraphs.append(score_mod_subgraph)
    if has_mask_mod:
        subgraphs.append(mask_graph_subgraph)

    choices: list[Any] = []
    error: NotImplementedError | None = None
    vec_sizes = _mod_vec_sizes(kernel_options, has_mod=has_score_mod or has_mask_mod)
    prefetch_depths = _qk_prefetch_depths(kernel_options)
    block_ms = _forward_block_ms(kernel_options, num_heads=num_heads)
    kv_gpfetches = _forward_kv_gpfetch(kernel_options)
    for mod_vec_size, qk_prefetch_depth, block_m, kv_gpfetch in itertools.product(
        vec_sizes, prefetch_depths, block_ms, kv_gpfetches
    ):
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
                QK_PREFETCH_DEPTH=qk_prefetch_depth,
                LAYOUT=layout_str,
                BLOCK_M=block_m,
                ENABLE_KV_GPFETCH=kv_gpfetch,
                USE_BLOCK_MASK=use_block_mask,
                HAS_FULL_BLOCKS=use_block_mask and has_full_blocks,
                SPARSE_Q_BLOCK_SIZE=(
                    V.graph.sizevars.guard_int(sparse_q_block_size)
                    if use_block_mask
                    else 0
                ),
                SPARSE_KV_BLOCK_SIZE=(
                    V.graph.sizevars.guard_int(sparse_kv_block_size)
                    if use_block_mask
                    else 0
                ),
            )
        # With alternatives left to try, a failed build is one choice fewer rather than a
        # failed lowering; `if not choices` below catches the case where all of them fail.
        if (
            error is not None
            and len(vec_sizes) * len(prefetch_depths) * len(block_ms) * len(kv_gpfetches) == 1
        ):
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
    # entry, so the declared list would call the kernel a tensor short.
    autotune_input_nodes = choices[0].input_nodes
    if any(choice.input_nodes != autotune_input_nodes for choice in choices):
        raise AssertionError(
            "FlyDSL flex choices disagree on their inputs: "
            f"{[[n.get_name() for n in c.input_nodes] for c in choices]}"
        )

    # Benchmarking must see block lists that address real blocks: random i32 would send
    # the skip loop at arbitrary KV tiles, and the timing would not be the timing of the
    # mask actually being run.
    input_gen_fns: dict[int, Any] | None = None
    if use_block_mask:
        input_gen_fns = {
            4: create_num_blocks_fake_generator(kv_indices),
            5: create_indices_fake,
        }
        if has_full_blocks:
            input_gen_fns[6] = create_num_blocks_fake_generator(full_kv_indices)
            input_gen_fns[7] = create_indices_fake

    template_output, _ = autotune_select_algorithm(
        "flex_flydsl_attention",
        choices,
        autotune_input_nodes,
        output_layout,
        input_gen_fns=input_gen_fns,
        return_multi_template=False,
    )

    return (template_output, lse)


def create_flydsl_flash_attention_backward_kernel(
    query,
    key,
    value,
    out,
    logsumexp,
    grad_out,
    grad_logsumexp,
    scale,
    kernel_options,
    *,
    fw_subgraph_buffer,
    joint_subgraph_buffer,
    mask_graph_buffer,
    has_score_mod,
    has_mask_mod,
    kv_num_blocks=None,
    kv_indices=None,
    full_kv_num_blocks=None,
    full_kv_indices=None,
    sparse_q_block_size=None,
    sparse_kv_block_size=None,
):
    """Build the FlyDSL flex backward choice, returning ``(dq, dk, dv, ())``.

    Shaped like ``create_flex_flash_attention_backward_kernel`` rather than the Triton
    backward: all three gradients are preallocated, ``dq`` is the template's output layout
    and ``dk``/``dv`` are mutated inputs. That is how our launcher already works -- it
    fills outputs in place -- where the Triton path produces ``dk`` through an epilogue
    ``store_output`` and needs a fused kernel split by program id to do it.

    The trailing empty tuple is the captured-buffer gradients, which this backward does
    not produce; the gate refuses any graph that would need them. Note that a mod is still
    free to *read* a captured tensor -- that is the aux-slot path, and it works.

    The three subgraph buffers are the ones the caller already lowered. Their indices are
    fixed at 0/1/2 (forward score_mod, joint, mask) and all three are always passed even
    when trivial, so that the template's ``modification(subgraph_number=...)`` calls do not
    have to renumber themselves the way the forward's mask does.
    """
    from ...select_algorithm import autotune_select_algorithm
    from ...virtualized import V
    from .flex_flash_attention import (
        patch_fixed_layout_indexer_for_cutedsl,
        wrap_choice_render_with_cutedsl_indexer,
    )

    unavailable = flydsl_unavailable_reason()
    if unavailable is not None:
        raise RuntimeError(f"FlyDSL flex kernels are unavailable: {unavailable}")

    # Imported here, not at module scope: this module is loaded on every ROCm Inductor run
    # whether or not FLYDSL was asked for, and the kernel module pulls in flydsl itself.
    from ..vendored_templates.flydsl.flex_kernels.flex_flash_bwd_generic import (
        DEFAULT_DQ_BLOCK_M,
    )

    batch_size, num_heads, seq_len_q, qk_head_dim = query.get_size()
    num_kv_heads = key.get_size()[1]
    seq_len_kv = key.get_size()[2]
    v_head_dim = value.get_size()[-1]
    device = query.get_device()
    dtype = query.get_dtype()
    if device is None:
        raise AssertionError("Device must be specified")

    if grad_logsumexp is not None:
        raise RuntimeError(
            "BACKEND='FLYDSL' backward does not support a logsumexp gradient"
        )

    # `do` joins q/k/v in the vote: it is read like them, and the gradients below take
    # their permutation from their references, so agreeing here means nothing is copied.
    layout_str = _kernel_layout(kernel_options, query, key, value, grad_out)

    def _grad_like(reference, size):
        return empty_strided(
            size=size,
            stride=[
                sympy.sympify(s)
                for s in infer_dense_strides(size, reference.get_stride())
            ],
            dtype=dtype,
            device=device,
        )

    grad_query = _grad_like(query, [batch_size, num_heads, seq_len_q, qk_head_dim])
    grad_key = _grad_like(key, [batch_size, num_kv_heads, seq_len_kv, qk_head_dim])
    grad_value = _grad_like(value, [batch_size, num_kv_heads, seq_len_kv, v_head_dim])

    # delta[b, h, m] = sum_d do[b, h, m, d] * o[b, h, m, d], the row term the softmax
    # derivative subtracts. Built here out of existing lowerings rather than in the
    # kernel, as the Triton backward does: it is an elementwise product and a row sum,
    # which Inductor can fuse into its neighbours, and keeping it out leaves the kernel
    # one pass over one fewer tensor.
    delta = lowerings[torch.ops.aten.mul](out, grad_out)
    delta = lowerings[torch.ops.aten.sum](delta, axis=-1)
    delta = lowerings[torch.ops.prims.convert_element_type](delta, torch.float32)
    (delta,) = maybe_realize([delta])

    # dq is the choice's output; dk and dv are filled in place.
    output_layout = FixedLayout(
        device=device,
        dtype=dtype,
        size=[batch_size, num_heads, seq_len_q, qk_head_dim],
        stride=[sympy.sympify(s) for s in grad_query.get_stride()],
    )

    # Captures are absent here for the same reason as on the forward path: they are
    # discovered while a mod is lowered and registered as kernel inputs then, so
    # `generate()` appends them afterwards.
    input_nodes = [
        query,
        key,
        value,
        grad_out,
        logsumexp,
        delta,
        grad_key,
        grad_value,
    ]

    choices: list[Any] = []
    error: NotImplementedError | None = None
    # Hinted to an int: `_backward_tiles` compares an LDS footprint against a limit, and a
    # sympy expression there would make that comparison a symbolic Boolean.
    tiles = _backward_tiles(
        kernel_options, V.graph.sizevars.optimization_hint(qk_head_dim)
    )
    mod_vec_size = _backward_mod_vec_size(kernel_options)

    # Block skipping is a policy decision rather than an autotune knob, because enabling
    # it changes the kernel's *inputs* (the two list tensors), and choices in one autotune
    # group have to agree on those. The forward decides the same way.
    #
    # One decision for both kernels, from the `dq` grid, whose tile is fixed. Deliberately
    # tile-independent even though `dkdv`'s real grid shrinks as its tile grows: making
    # the answer depend on the tile being swept lets the two interact badly. Autotuning
    # benchmarks every choice against *dense* fake lists (see `input_gen_fns` below), so it
    # cannot see the sparsity -- it would pick the widest tile on dense merits and, if that
    # were also the tile whose grid fell under the threshold, silently take the choice with
    # skipping disabled. Measured doing exactly that: 1x8x4096 head_dim 64 causal picked
    # KV tile 256 with skipping off and ran the backward at its dense speed.
    #
    # So the grid here is a proxy for "is this problem big enough to fill the machine",
    # which is what the threshold is really asking, and every choice gets the same answer.
    hint = V.graph.sizevars.optimization_hint
    has_full_blocks = full_kv_num_blocks is not None and full_kv_indices is not None

    block_skip, _block_skip_reason = _backward_block_skip(
        kv_num_blocks,
        kv_indices,
        sparse_q_block_size,
        sparse_kv_block_size,
        has_mask_mod=has_mask_mod,
        has_full_blocks=has_full_blocks,
        workgroups=(
            hint(batch_size)
            * -(-hint(seq_len_q) // DEFAULT_DQ_BLOCK_M)
            * hint(num_heads)
        ),
        device=device,
    )

    if block_skip:
        input_nodes.extend([kv_num_blocks, kv_indices])
        if has_full_blocks:
            input_nodes.extend([full_kv_num_blocks, full_kv_indices])

    for dkdv_kv_tile, dkdv_q_tile in tiles:
        with patch_fixed_layout_indexer_for_cutedsl():
            error = _flydsl_flash_attention_backward_template().maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=output_layout,
                mutated_inputs=[grad_key, grad_value],
                subgraphs=[
                    fw_subgraph_buffer,
                    joint_subgraph_buffer,
                    mask_graph_buffer,
                ],
                SM_SCALE=scale,
                NUM_HEADS=num_heads,
                NUM_KV_HEADS=num_kv_heads,
                HEAD_DIM=qk_head_dim,
                DTYPE_STR=_FLYDSL_DTYPE_STR[dtype],
                HAS_SCORE_MOD=has_score_mod,
                HAS_MASK_MOD=has_mask_mod,
                MOD_VEC_SIZE=mod_vec_size,
                LAYOUT=layout_str,
                DKDV_KV_TILE=dkdv_kv_tile,
                DKDV_Q_TILE=dkdv_q_tile,
                DQ_BLOCK_SKIP=block_skip,
                DKDV_BLOCK_SKIP=block_skip,
                HAS_FULL_BLOCKS=has_full_blocks,
                SPARSE_Q_BLOCK_SIZE=sparse_q_block_size or 0,
                SPARSE_KV_BLOCK_SIZE=sparse_kv_block_size or 0,
            )
        # A tile can be refused for its LDS footprint at large head_dim, which is fine as
        # long as one survives; a lone requested tile failing is not.
        if error is not None and len(tiles) == 1:
            raise RuntimeError(f"FlyDSL backward template failed: {error}")
    if not choices:
        raise RuntimeError(f"FlyDSL backward template failed: {error}")

    # Captured-tensor reads are addressed per axis, not by a flattened offset, so the
    # deferred render at scheduling time needs the same indexer the choices were built
    # with. See Note [CuteDSL indexer patch].
    for choice in choices:
        wrap_choice_render_with_cutedsl_indexer(choice)

    # Autotune against what the choices actually take: rendering appended the captures the
    # mods read, so the declared list would call the kernel a tensor short.
    autotune_input_nodes = choices[0].input_nodes
    if any(choice.input_nodes != autotune_input_nodes for choice in choices):
        raise AssertionError(
            "FlyDSL flex backward choices disagree on their inputs: "
            f"{[[n.get_name() for n in c.input_nodes] for c in choices]}"
        )
    # Benchmarking must see block lists that address real blocks. The default for an i32
    # input is zeros, which would make every count zero, every skip loop empty, and the
    # measured backward a prologue with no reduction in it -- fast, and not a time anything
    # runs at. The generators fill in plausible counts and in-range indices instead.
    input_gen_fns: dict[int, Any] | None = None
    if block_skip:
        input_gen_fns = {
            8: create_num_blocks_fake_generator(kv_indices),
            9: create_indices_fake,
        }
        if has_full_blocks:
            input_gen_fns[10] = create_num_blocks_fake_generator(full_kv_indices)
            input_gen_fns[11] = create_indices_fake

    template_output, _ = autotune_select_algorithm(
        "flydsl_flash_attention_backward",
        choices,
        autotune_input_nodes,
        output_layout,
        input_gen_fns=input_gen_fns,
        return_multi_template=False,
    )

    return (template_output, grad_key, grad_value, ())


@functools.lru_cache(maxsize=1)
def _flydsl_flash_attention_backward_template():
    """The registered FlyDSL flex backward template, built on first use."""
    from ...codegen.flydsl.flydsl_flex_kernel import FlyDSLFlexTemplate

    return FlyDSLFlexTemplate(
        name="flydsl_flash_attention_backward",
        source=load_flex_template("flydsl_flash_attention_backward"),
    )


@functools.lru_cache(maxsize=1)
def _flydsl_flash_attention_template():
    """The registered FlyDSL flex template, built on first use.

    Deferred rather than built at import because registration is global and this module is
    imported on every ROCm Inductor run, whether or not FLYDSL was asked for.
    """
    from ...codegen.flydsl.flydsl_flex_kernel import FlyDSLFlexTemplate

    return FlyDSLFlexTemplate(
        name="flydsl_flash_attention",
        source=load_flex_template("flydsl_flash_attention"),
    )


def _mod_vec_sizes(kernel_options: dict[str, Any], *, has_mod: bool) -> list[int]:
    """Vectorization widths to offer as autotune choices.

    The kernel evaluates the mods over 1, 2 or 4 contiguous KV columns at a time; 4 is the
    ceiling because the score layout guarantees only that many contiguous columns per lane.
    A wider mod amortises per-call overhead but costs registers, and which wins depends on
    the mod, so it is measured rather than predicted.
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


# LDS a dk/dv build needs: four tiles (Q and DO, each row-major and transposed) of
# q_tile * head_dim elements at 2 B each. `flex_flash_bwd_generic` is the definition and
# raises on a build that overruns; this copy exists so an autotune choice that cannot fit
# is never offered, without importing the kernel module (which is only importable where
# FlyDSL is installed). Note that the *kv* tile costs no LDS at all -- it lives in
# registers as a preloaded B operand -- so only the q tile appears here.
#
# The two row-major tiles carry the same 8-element row padding the kernel adds wherever
# the XOR swizzle cannot be expressed, so this has to know the same rule: undercounting
# would offer a tile the builder then rejects.
def _dkdv_lds_bytes(q_tile: int, head_dim: int) -> int:
    granules = head_dim // 16
    pad = 0 if granules & (granules - 1) == 0 else 8
    return 2 * q_tile * (head_dim + pad) * 2 + 2 * q_tile * head_dim * 2


def _backward_tiles(kernel_options: dict[str, Any], head_dim: int) -> list[tuple[int, int]]:
    """``(kv_tile, q_tile)`` shapes to offer the dk/dv kernel as autotune choices.

    The KV tile is the output tile (32 rows per wave, so 128 is 4 waves) and the Q tile is
    the reduction tile staged in LDS. Measured across six shapes, ``(128, 32)`` won four
    and was 9% and 21% behind on the other two -- a short 8-head shape preferring a bigger
    Q tile, a 32-head one a bigger KV tile -- so all three are worth offering.

    The dq kernel's tile is deliberately not swept; see ``autotune_backward_tile``.
    """
    requested = kernel_options.get("DKDV_TILE")
    if requested is not None:
        try:
            kv_tile, q_tile = (int(v) for v in requested)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"DKDV_TILE must be a (kv_tile, q_tile) pair of ints, got {requested!r}"
            ) from exc
        return [(kv_tile, q_tile)]

    tiles = [(128, 32)]
    if torch._inductor.config.flydsl.autotune_backward_tile:
        tiles += [(256, 32), (128, 64)]
    # A q tile of 64 does not fit past head_dim 128, so past there the sweep is over the
    # kv tile alone. The default (128, 32) fits the whole ladder: at head_dim 256 it lands
    # on exactly 65536 B.
    return [t for t in tiles if _dkdv_lds_bytes(t[1], head_dim) <= _LDS_LIMIT]


# Workgroups per compute unit below which backward block skipping stops paying, and can
# cost a few percent. Measured on gfx942 (80 CUs), causal, bf16, head_dim 64 and 128: the
# gain tracks *this ratio*, not the sequence length -- 0.89-1.18x below 2, and 1.54-2.39x
# from 3.2 up, settling near 1.7x for dq and 2.0x for dk/dv.
#
# The reason is load imbalance rather than anything about the skipping itself. Causal work
# per tile is triangular, so one workgroup walks every tile of its axis and another walks
# one. Below a couple of workgroups per CU they are all resident at once and the kernel
# finishes when the *heaviest* one does, which skipping does not move; the light ones were
# already free. Above it the average is what counts and the triangle halves the work.
_BLOCK_SKIP_MIN_WGS_PER_CU = 2


def _enough_workgroups_to_skip(workgroups, device) -> tuple[bool, str]:
    """Whether this grid is big enough for block skipping to pay. See the constant above."""
    num_cus = torch.cuda.get_device_properties(device).multi_processor_count
    if workgroups < _BLOCK_SKIP_MIN_WGS_PER_CU * num_cus:
        return (
            False,
            f"{workgroups} workgroups over {num_cus} CUs is under "
            f"{_BLOCK_SKIP_MIN_WGS_PER_CU}/CU, where the triangular load imbalance eats "
            "the saving",
        )
    return True, ""


def _backward_block_skip(
    kv_num_blocks,
    kv_indices,
    sparse_q_block_size,
    sparse_kv_block_size,
    *,
    has_mask_mod,
    has_full_blocks,
    workgroups,
    device,
) -> tuple[bool, str]:
    """Whether the backward kernels should walk block lists instead of their axes densely.

    Same ``(usable, reason)`` contract as the forward's ``_block_mask_usable``, whose
    checks this defers to before adding the occupancy one. Declining is equally safe: the
    dense walk applies ``mask_mod`` to every element and gets the same gradients, just
    without the sparsity.
    """
    usable, reason = _block_mask_usable(
        kv_num_blocks,
        kv_indices,
        sparse_q_block_size,
        sparse_kv_block_size,
        has_mask_mod=has_mask_mod,
        has_full_blocks=has_full_blocks,
    )
    if not usable:
        return False, reason
    return _enough_workgroups_to_skip(workgroups, device)


def _backward_mod_vec_size(kernel_options: dict[str, Any]) -> int:
    """Mod vectorization width for the backward. One, unless asked for.

    Not swept, and not for the forward's reason. The ``dq`` kernel can take 2 or 4 -- its
    score fragment has the same 4-contiguous KV structure the forward's does -- but the
    ``dk``/``dv`` kernel cannot: there ``kv_idx`` is the per-lane constant and it is
    ``q_idx`` that runs contiguously, so the vectorized ABI does not apply and the mods
    are always called one element at a time. So the reachable win is at most half the
    forward's 0.6-2.1%, against multiplying the tile sweep by three. Available as
    ``kernel_options={"MOD_VEC_SIZE": n}`` for anyone who wants to measure it.
    """
    requested = kernel_options.get("MOD_VEC_SIZE")
    if requested is None:
        return 1
    if requested not in (1, 2, 4):
        raise RuntimeError(f"MOD_VEC_SIZE must be 1, 2 or 4, got {requested}")
    return int(requested)


def _qk_prefetch_depths(kernel_options: dict[str, Any]) -> list[int]:
    """QK prefetch depths to offer as autotune choices.

    How many K packs come out of LDS before the MFMA chain starts. Deeper hides more LDS
    latency and costs registers. Depths above 2 also emit a ``sched_group_barrier`` so the
    scheduler keeps the extra reads together, without which a deeper prefetch can be far
    *slower* -- unhinted depth 5 measured 6x worse on one shape.

    Off unless asked for: the win is 0.6-2.1% where it exists and three of six measured
    shapes prefer the default, which does not pay for four times the builds.
    """
    requested = kernel_options.get("QK_PREFETCH_DEPTH")
    if requested is not None:
        if not (isinstance(requested, int) and requested >= 1):
            raise RuntimeError(f"QK_PREFETCH_DEPTH must be a positive int, got {requested}")
        return [int(requested)]
    if not torch._inductor.config.flydsl.autotune_qk_prefetch_depth:
        return [2]
    return [2, 3, 4, 5]


def _forward_block_ms(kernel_options: dict[str, Any], *, num_heads: int) -> list[int]:
    """Q tile heights to offer the forward as autotune choices.

    The kernel's own default is 256 rows once there are 32 heads and 128 below that, on the
    reasoning that a taller tile amortises the KV walk when there are enough heads to keep
    the machine busy. Measured at 2x32x4096 the taller tile wins at exactly one head_dim
    (160, by 14%) and loses at the other six -- by 19% at 64 and by 2.5x at 256, where 256
    rows of accumulator drop occupancy to one wave per CU and the kernel to 26.9 TFLOP/s
    against 66.3. So the height is measured on the shapes where the default reaches for
    256, which is the only place the two differ.

    Below 32 heads there is nothing to sweep: both the default and every measurement agree
    on 128, and offering a second point there would double the builds of the common case to
    re-answer a settled question.
    """
    requested = kernel_options.get("BLOCK_M")
    if requested is not None:
        if not (isinstance(requested, int) and requested > 0 and requested % 64 == 0):
            raise RuntimeError(
                f"BLOCK_M must be a positive multiple of 64, got {requested!r}"
            )
        return [int(requested)]
    if num_heads < 32 or not torch._inductor.config.flydsl.autotune_forward_block_m:
        # Matches flex_flash_generic's own default, so an autotune-off build is unchanged.
        return [256 if num_heads >= 32 else 128]
    return [128, 256]


def _forward_kv_gpfetch(kernel_options: dict[str, Any]) -> list[bool]:
    """Whether to stage K through registers, offered as autotune choices.

    Off, a KV tile's global read goes straight to LDS and lands right in front of the
    barrier GEMM1 waits on. On, the read is split so the loop can issue tile i+1's global
    load before computing tile i, and V's read moves inside GEMM1 -- upstream's main gfx942
    lever, since the DMA-to-LDS prefetch that would otherwise do this needs a 16-byte
    ``buffer_load_lds`` only gfx950 has.

    It is swept rather than defaulted because the measurement does not point one way. At
    2x32x4096 it wins 6-15% at head_dim 96/192/224 and at every causal dim up to 224, and
    loses 7-9% at 64 and 128 dense and 256 causal -- the extra live registers for the
    in-flight tile cost more than the overlap buys wherever the K walk was already cheap
    enough to hide. Numerics are bit-identical either way, so this only trades speed.
    """
    requested = kernel_options.get("ENABLE_KV_GPFETCH")
    if requested is not None:
        return [bool(requested)]
    if not torch._inductor.config.flydsl.autotune_kv_gpfetch:
        # Matches flex_flash_generic's own default, so an autotune-off build is unchanged.
        return [True]
    return [False, True]


def _kernel_layout(kernel_options: dict[str, Any], *tensors: Any) -> str:
    """Which layout to build the kernel for, read off the strides it will be handed.

    FlexAttention always hands us ``[B, H, S, D]``, but a model producing q/k/v from a
    projection reshaped to ``[B, S, H, D]`` and transposed hands over a *view*: the sizes
    are BHSD and the memory is BSHD. Building a BHSD kernel for that makes the interface
    materialise three contiguous copies of q/k/v, and the output layout follows q's
    permutation so it takes a fourth on the way back.

    Measured at 4x16x4096x128 bf16 that is 574 us per call against a 461 us kernel, which
    is where the sparse-mask forward deficit came from: a short KV walk cannot amortise it,
    so `sliding_window` read as 0.71x while the kernel itself was beating Triton's by 1.19x.

    The kernel indexes either layout, so the fix is to build the one the caller already
    has and copy nothing. Anything that is neither (a slice, an expanded head) keeps BHSD
    and its copies, which is correct and no slower than before.

    Not free in the kernel, though: a BHSD ``(batch, head)`` slice is contiguous and a KV
    tile is one run, where BSHD strides each token row by ``num_heads * head_dim``, and
    that measured 8-32% of the kernel. The copies are the larger cost at the shapes tried
    -- they are four passes over q/k/v-sized memory whatever the walk does -- but they grow
    with ``S`` where the kernel grows with ``S**2``, so a long enough dense shape would
    rather copy. ``kernel_options={"LAYOUT": ...}`` pins it for anyone who has measured
    their own shape.
    """
    from ...virtualized import V

    requested = kernel_options.get("LAYOUT")
    if requested is not None:
        if requested not in ("bhsd", "bshd"):
            raise RuntimeError(f"LAYOUT must be 'bhsd' or 'bshd', got {requested!r}")
        return str(requested)

    def bshd_dense(tensor: Any) -> bool:
        """Whether this ``[B, H, S, D]`` tensor is a view of dense ``[B, S, H, D]``."""
        _, num_heads, seq_len, head_dim = tensor.get_size()
        # The head axis steps one row, and the token axis steps a whole row of heads.
        want = (
            seq_len * num_heads * head_dim,
            head_dim,
            num_heads * head_dim,
            1,
        )
        return all(
            V.graph.sizevars.statically_known_equals(actual, expected)
            for actual, expected in zip(tensor.get_stride(), want)
        )

    # Every tensor has to agree, since one launcher addresses all of them. Where a size
    # is 1 the two patterns coincide and either answer is free of copies.
    if all(bshd_dense(t) for t in tensors):
        return "bshd"
    return "bhsd"


def _reject_unsupported_captures(
    score_mod_other_buffers: Sequence[Any],
    mask_mod_other_buffers: Sequence[Any],
) -> None:
    """Reject captures the kernel cannot address, before anything is rendered.

    CPU 0-d captures are rejected earlier, in ``flex_attention()``, because by this point
    they have been realized and no longer look like scalars.
    """
    captures = list(score_mod_other_buffers) + list(mask_mod_other_buffers)
    # Aux slots are a hard limit rather than a tuning choice: the kernel's signature
    # carries a fixed number, so a capture past the last one has nowhere to go.
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

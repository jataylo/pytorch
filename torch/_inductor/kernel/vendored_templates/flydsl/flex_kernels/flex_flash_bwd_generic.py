# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlexAttention flash-attention **backward** kernel builders (generic / portable path).

No upstream donor: FlyDSL ships no gfx942 backward, and its gfx950 one is welded to a
parity framework we do not carry. So this is written against our own forward
(`flex_flash_generic.py`), which turns out to already hold both GEMM shapes the backward
needs. See Note [Backward GEMMs are the forward's two GEMMs] below.

Two kernels: ``dq``, and ``dk``/``dv`` together. ``dq`` reduces over the KV axis and
``dk``/``dv`` over the Q axis, so they cannot share a loop nest; ``dk`` and ``dv`` do share
one, since they share the whole ``s``/``p``/``dp``/``ds`` prologue and differ only in which
operand the final GEMM takes.

``score_mod`` and ``mask_mod`` are supported, at two call sites rather than the forward's
one: the score has to be *recomputed* through the mod (``lse`` was written against the
modified score), and the gradient has to be carried back *through* it by a joint graph at
the ``ds`` site. See ``joint_mod`` on either builder.

Not handled here, by construction:

- **Gradients with respect to a captured tensor.** A mod may freely *read* one -- that is
  the aux-slot path, shared with the forward -- but a capture that itself requires grad
  needs the joint graph's ``zeros_and_scatter`` outputs accumulated with atomics. The gate
  in ``flydsl_flash_attention.py`` refuses those.
- ``block_mask``: both walks are dense. ``mask_mod`` is evaluated on every tile rather
  than whole tiles being skipped, so a causal backward does the work of a full one. That
  is a performance gap, not a correctness one, and it is the largest one left here.
- ``causal`` as a build flag: the caller expresses causality as a ``mask_mod``, which is
  the supported path.

Note [Backward GEMMs are the forward's two GEMMs]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The usual reason a flash backward is harder than its forward is that it wants products in
both orientations, and an MFMA accumulator cannot be fed back as an operand without a
transpose through LDS. Our forward avoids that here because it already computes its scores
transposed -- ``GEMM1`` is ``k @ qᵀ``, giving ``sᵀ[n, m]`` -- and consumes them with
``GEMM2 = vᵀ @ p``. Every backward product is one of those two shapes with different
operands, in whichever orientation the kernel that wants it needs:

    dq kernel (reduces over n)      dkdv kernel (reduces over m)
    sᵀ  = k  @ qᵀ   GEMM1           s   = q  @ kᵀ   GEMM1, operands swapped
    dpᵀ = v  @ doᵀ  GEMM1           dp  = do @ vᵀ   GEMM1, operands swapped
    dqᵀ = kᵀ @ dsᵀ  GEMM2           dvᵀ = doᵀ @ p   GEMM2
                                    dkᵀ = qᵀ  @ ds  GEMM2

and the MFMA fragment layouts line up without a single register shuffle. Two properties do
the work, and both hold in either orientation:

- The accumulator's ``j`` index is ``lane % 32``, constant per lane. In ``dq`` that is
  ``m``, so ``lse[m]`` and ``delta[m]`` are *scalars* in registers -- the cheapest the
  elementwise middle can be, and no cross-lane reduction anywhere.
- ``C``'s 16 slots walk ``i = (lane//32)*4 + (s//4)*8 + s%4``, so slots ``[4p:4p+4]`` hold
  4 *consecutive* ``i``. That is exactly the ``k``-contiguity an A- or B-operand pack
  wants, so a score accumulator feeds the next GEMM directly, the way the forward's ``p``
  already does. An MFMA's A and B operand layouts have the same shape, which is why the
  same fragment serves as either.

What differs between the two kernels is which axis is the reduction, and that decides two
things. In ``dkdv`` the accumulator is ``C[m, n]`` with ``n = lane % 32``, so ``lse[m]``
and ``delta[m]`` spread across the 16 slots instead of being scalars -- still cheap (4
groups of 4 consecutive ``m``, so 4 vector loads) but not free. More importantly ``m`` is
summed over there, so an out-of-range Q row *must* be masked to ``p = 0``; in ``dq`` an
out-of-range row is harmless because nothing reduces across rows and its store is dropped.

The cost neither kernel escapes: GEMM2 wants its A operand transposed relative to GEMM1's,
so the reduction-axis tensors are staged into LDS *twice*, in both orientations. ``dq``
stages K twice plus V (three tiles); ``dkdv`` stages Q and DO twice each (four). Which is
why both use a smaller reduction tile than the forward -- see ``block_n`` and ``block_m``
on the two builders.
"""

import math as host_math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl

# Not `from flydsl.expr import buffer_ops`: that module is gone in FlyDSL 0.3.1. See
# the sibling `buffer_ops.py`.
from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels import buffer_ops
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_generic import (
    _extract_aligned_pointer,
    _pointer_load,
)
from torch._inductor.kernel.vendored_templates.flydsl.kernels.kernels_common import (
    dtype_to_elem_type,
)

_LOG2E = host_math.log2(host_math.e)

# The one tile shape this file builds. BLOCK_M is per-workgroup Q rows, 32 of them per
# wave (one MFMA32 tile), and BLOCK_N is the KV tile the loop walks.
_ROWS_PER_WAVE = 32
_WARP_SIZE = 64

# Per-workgroup LDS on gfx942/gfx950. What the three KV tiles (K row-major, Kᵀ, V) have
# to fit inside, and what makes `block_n` the knob that matters here.
_LDS_LIMIT = 65536

# Aux (captured-tensor) slots, matching the forward's `MAX_AUX_TENSORS`. A kernel
# signature is fixed-arity, so all four are declared unconditionally and unused ones are
# handed a 1-element dummy by the interface.
MAX_AUX_TENSORS = 4

# Default tiles, exported so a caller can size the grid without building a launcher --
# which the Inductor lowering needs in order to decide whether block skipping will pay.
DEFAULT_DQ_BLOCK_M = 128
DEFAULT_DQ_BLOCK_N = 32


_VMCNT_LO_MASK = 0xF
_VMCNT_HI_MASK = 0x3
_VMCNT_HI_SHIFT = 14
_LGKMCNT_EXPCNT_BASE = 0x0F70


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    rocdl.s_waitcnt(val)


def _make_aux_reader(rsrc, strides):
    """A captured-tensor reader callable as ``(b, h, q_idx, kv_idx) -> f32``.

    Same contract and same stride convention as the forward's -- a 4-tuple of element
    strides where 0 means "broadcast over this axis", so an ``[H]`` ALiBi slope table is
    ``(0, 1, 0, 0)``. Kept identical deliberately: a mod body is lowered once by Inductor
    and has to run unchanged at all three call sites (forward score, backward score,
    backward ds).
    """
    _skv = strides[3]

    def _offset(coords):
        off = None
        for stride, coord in zip(strides, coords):
            if const_expr(stride == 0):
                continue
            term = fx.Int32(coord) if const_expr(stride == 1) else fx.Int32(coord) * fx.Int32(stride)
            off = term if off is None else off + term
        return fx.Int32(0) if off is None else off

    def _read(b, h, q_idx, kv_idx):
        return fx.Float32(
            buffer_ops.buffer_load(rsrc, _raw(_offset((b, h, q_idx, kv_idx))), 1, dtype=None)
        )

    def _read_vec(b, h, q_idx, kv_base, n):
        if const_expr(_skv != 1 or n not in (2, 4)):
            return [_read(b, h, q_idx, kv_base + fx.Int32(i)) for i in range(n)]
        packed = buffer_ops.buffer_load(
            rsrc, _raw(_offset((b, h, q_idx, kv_base))), vec_width=n, dtype=None
        )
        return [fx.Float32(Vec(packed)[i]) for i in range(n)]

    _read.vec = _read_vec
    return _read


def _mod_callers(score_mod, mask_mod, joint_mod, mod_vec, mod_kw):
    """Lift the three mod callables to the list ABI the call sites use.

    Returns ``(call_score, call_mask, call_joint)``, each taking and returning lists of
    ``mod_vec`` values. ``joint_mod`` is the one the forward has no equivalent of: it is
    the chain rule through ``score_mod``, called as
    ``joint_mod(pre_mod_score, b, h, q_idx, kv_idx, grad_post_mod) -> grad_pre_mod``.
    """

    def call_score(scores, b, h, q_idx, kvs):
        if const_expr(mod_vec == 1):
            return [score_mod(scores[0], b, h, q_idx, kvs[0], **mod_kw)]
        return list(score_mod(scores, b, h, q_idx, kvs, **mod_kw))

    def call_mask(b, h, q_idx, kvs):
        if const_expr(mod_vec == 1):
            return [mask_mod(b, h, q_idx, kvs[0], **mod_kw)]
        return list(mask_mod(b, h, q_idx, kvs, **mod_kw))

    def call_joint(scores, b, h, q_idx, kvs, grads):
        if const_expr(mod_vec == 1):
            return [joint_mod(scores[0], b, h, q_idx, kvs[0], grads[0], **mod_kw)]
        return list(joint_mod(scores, b, h, q_idx, kvs, grads, **mod_kw))

    return call_score, call_mask, call_joint


def _validate_mod_args(
    score_mod, joint_mod, mod_vec_size, num_aux_tensors, aux_specs, aux_numels
):
    """Shared argument checks for the mod hooks on both backward builders."""
    if joint_mod is not None and score_mod is None:
        raise ValueError("joint_mod needs a score_mod: it is the chain rule through one")
    if score_mod is not None and joint_mod is None:
        # Silently dropping the chain rule would produce plausible-looking gradients that
        # are wrong by exactly the mod's derivative -- the worst failure mode available.
        raise ValueError(
            "score_mod without joint_mod would compute gradients through an unmodified "
            "score; pass joint_mod=identity explicitly if the mod really is the identity"
        )
    if mod_vec_size not in (1, 2, 4):
        raise ValueError(f"mod_vec_size must be 1, 2 or 4, got {mod_vec_size}")
    if not (0 <= num_aux_tensors <= MAX_AUX_TENSORS):
        raise ValueError(f"num_aux_tensors must be 0 to {MAX_AUX_TENSORS}, got {num_aux_tensors}")
    if num_aux_tensors and (aux_numels is None or len(aux_numels) != num_aux_tensors):
        # See Note [aux reads run past the logical extent] in flex_flash_generic.py
        raise ValueError(
            "aux_numels must supply one element count per aux tensor "
            f"(num_aux_tensors={num_aux_tensors}); it bounds the buffer descriptor, and "
            "without it the padding lanes read past the tensor"
        )
    if num_aux_tensors and (aux_specs is None or len(aux_specs) != num_aux_tensors):
        raise ValueError(
            "aux_specs must supply one (sb, sh, sq, skv) stride tuple per aux tensor "
            f"(num_aux_tensors={num_aux_tensors})"
        )


def build_flex_flash_bwd_dq_module(
    num_heads,
    head_dim,
    dtype_str="bf16",
    sm_scale=None,
    num_kv_heads=None,
    layout="bhsd",
    block_m=DEFAULT_DQ_BLOCK_M,
    block_n=DEFAULT_DQ_BLOCK_N,
    waves_per_eu=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    score_mod=None,
    mask_mod=None,
    joint_mod=None,
    mod_key=None,
    mod_vec_size=1,
    num_aux_tensors=0,
    aux_specs=None,
    aux_numels=None,
    block_mask=False,
    enable_kv_gpfetch=None,
):
    """Build the ``dq`` launcher: ``(Q, K, V, DO, LSE, DELTA, DQ, AUX0..3, KVNB, KVI, B, S)``.

    ``LSE`` is ``[B, H_q, S]`` f32 in **natural** log -- what our forward writes, and not
    what the Triton backward reads (log2). ``DELTA`` is ``rowsum(do * o)``, ``[B, H_q, S]``
    f32, computed in the lowering so Inductor can fuse it. ``DQ`` is written, not
    accumulated into, so the caller need not zero it.

    For GQA/MQA pass ``num_kv_heads < num_heads``: every ``num_heads // num_kv_heads``
    consecutive Q heads share one KV head, exactly as in the forward. ``dq`` needs no
    reduction across the group -- each Q head owns its own rows -- which is why this
    kernel is GQA-clean while ``dk``/``dv`` will not be.

    ``block_n`` defaults to 32 rather than the forward's 64, and it is the one tile
    parameter here worth arguing about. Three KV tiles at ``block_n`` 64 and head_dim 128
    are 48 KB, past half of gfx942's 64 KB, so only one workgroup is resident; halving it
    to 24 KB buys the second and measured +51% to +73% across shapes at head_dim 128,
    +31% at 96 and +14% at 64. It never lost -- the one shape that preferred 64 was small
    enough to be running at under 2 TFLOP/s either way. It is also what makes head_dim
    256 representable at all. ``block_m`` is *not* similarly interesting: 128 measured
    best on 6 of 7 shapes, and the losses to 64 and 256 were large and one-sided.

    FlexAttention hooks, all evaluated at build time with FlyDSL device values:

    ``score_mod(score, b, h, q_idx, kv_idx) -> score``
        The same callable the forward takes, at the same point in the same score domain
        (``q·k * sm_scale``). The backward has to *recompute* the modified score, because
        ``lse`` was written against it.

    ``joint_mod(score, b, h, q_idx, kv_idx, grad_score) -> grad_score``
        The chain rule through ``score_mod``: given the pre-mod score and the gradient
        with respect to the post-mod score, return the gradient with respect to the
        pre-mod score. This is what the forward has no equivalent of, and it is required
        whenever ``score_mod`` is passed -- see ``_validate_mod_args``.

    ``mask_mod(b, h, q_idx, kv_idx) -> i1``
        True keeps the element. Applied after ``score_mod``, as in the forward. A masked
        element gets ``p = 0`` and so contributes nothing to any gradient, which is what
        makes masking correct here without touching the ``ds`` site: the joint graph is
        linear in its cotangent, so a zero in gives a zero out.

    ``block_mask`` turns the dense KV walk into a walk of this Q tile's block list, so
    tiles the mask proved empty are never loaded. Needs a ``mask_mod``, for the same
    reason the forward's does: there is one body for every block visited and no separate
    fast path for fully-unmasked ones, so the list has to be the *union* of
    FlexAttention's partial and full lists and every visited block gets masked. The list
    is supplied on this kernel's own ``(BLOCK_M, BLOCK_N)`` grid -- see
    ``regrid_block_mask(..., walk="kv")``.
    """
    gpu_arch = get_hip_arch()

    if layout not in ("bshd", "bhsd"):
        raise ValueError(f"layout must be 'bshd' or 'bhsd', got {layout!r}")
    if dtype_str not in ("f16", "bf16"):
        raise ValueError(f"flex_flash_bwd_generic supports f16 and bf16, got {dtype_str!r}")
    # Floor matches the forward, which cannot produce an LSE below 64 in the first place.
    # There is no ceiling here: what bounds head_dim is the LDS budget, checked below
    # once BLOCK_N is known, because a narrower KV tile buys head_dim back.
    if head_dim % 32 != 0 or head_dim < 64:
        raise ValueError(f"head_dim must be a multiple of 32 and at least 64, got {head_dim}")
    if num_kv_heads is None:
        num_kv_heads = num_heads
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
    _validate_mod_args(
        score_mod, joint_mod, mod_vec_size, num_aux_tensors, aux_specs, aux_numels
    )
    if block_mask and mask_mod is None:
        raise ValueError("block_mask needs a mask_mod: every visited block is masked")

    BLOCK_M = int(block_m)
    BLOCK_N = int(block_n)
    K_SUB_N = 32
    N_STRIPS = BLOCK_N // K_SUB_N
    if BLOCK_M % _ROWS_PER_WAVE != 0:
        raise ValueError(f"block_m ({BLOCK_M}) must be a multiple of {_ROWS_PER_WAVE}")
    # A power of two so the Kᵀ tile's XOR swizzle stays a permutation within a row (the
    # same constraint that turns K's swizzle off at head_dim 96), and at least K_SUB_N
    # because the strip loop steps by it.
    if BLOCK_N % K_SUB_N != 0 or BLOCK_N & (BLOCK_N - 1) != 0:
        raise ValueError(f"block_n ({BLOCK_N}) must be a power of two and a multiple of {K_SUB_N}")
    NUM_WAVES = BLOCK_M // _ROWS_PER_WAVE
    BLOCK_SIZE = NUM_WAVES * _WARP_SIZE

    # MFMA 32x32x8 throughout, i.e. the gfx942 shape. gfx950's 32x32x16 and its
    # hardware-transposing LDS read (`ds_read_tr16_b64`, which would retire the Kᵀ tile)
    # are both left for later: this is a correctness-first build, and one MFMA shape
    # keeps the fragment maps in the module docstring true on every arch it runs on.
    MFMA_LANE_K = 4
    K_STEP = 8
    K_STEPS = head_dim // K_STEP
    D_CHUNK = 32
    D_CHUNKS = head_dim // D_CHUNK
    DS_K_STEP = 8
    DS_K_STEPS = K_SUB_N // DS_K_STEP

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS_Q = num_heads
    NUM_HEADS_KV = num_kv_heads
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV
    HEAD_DIM = head_dim
    LAYOUT = layout
    BHSD = LAYOUT == "bhsd"
    STRIDE_TOKEN_Q = HEAD_DIM if BHSD else NUM_HEADS_Q * HEAD_DIM
    STRIDE_TOKEN_KV = HEAD_DIM if BHSD else NUM_HEADS_KV * HEAD_DIM

    # ---- LDS geometry ----
    # K and V row-major share the forward's XOR swizzle (identical access pattern: one
    # vector per lane at a fixed row). Kᵀ shares the forward's *transposed* V swizzle,
    # for the same reason -- the tile has the same [HEAD_DIM][BLOCK_N] shape and the same
    # two access patterns, a strided scalar store and a 4-contiguous vector read.
    KT_STRIDE = BLOCK_N
    K_GRANULES = HEAD_DIM // 16
    K_SWZ_POW2 = K_GRANULES & (K_GRANULES - 1) == 0
    K_SWZ_ROWMASK = (K_GRANULES - 1) if K_SWZ_POW2 else 0
    # And where the swizzle cannot be expressed -- head_dim 96 / 160 / 192 / 224, whose
    # granule count is not a power of two -- the row is padded instead. See the forward's
    # K_PAD for the bank arithmetic; there is no DMA path here, so the padding is
    # unconditional on the head_dims that need it.
    K_PAD = 0 if K_SWZ_POW2 else 8
    K_STRIDE = HEAD_DIM + K_PAD

    VEC_WIDTH = 8
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD
    LOAD_HAS_IDLE_LANES = BLOCK_SIZE % THREADS_PER_ROW_LOAD != 0
    NUM_BATCHES_KV = max(1, -(-BLOCK_N // ROWS_PER_BATCH_LOAD))
    KV_NEEDS_GUARD = LOAD_HAS_IDLE_LANES or (NUM_BATCHES_KV * ROWS_PER_BATCH_LOAD != BLOCK_N)

    # Hold the next KV tile in registers so its global read overlaps this tile's GEMMs.
    # This is the gfx942 form of the pipelining the gfx950 backward gets from a DMA into a
    # second LDS buffer -- gfx942 has neither that instruction nor the LDS to spare, but it
    # can carry the tile in registers instead.
    #
    # On by default, which the forward's version is not: there it wins at some head_dims
    # and loses at others, while here it won at all of 64/96/128/192/256 (2x32x4096, by
    # 3.8% to 24.4%, best at 192) with bit-identical results. The reason to have doubted it
    # was register pressure, and that measured clear: 329 -> 363 VGPRs of 512 at head_dim
    # 192 with no spilling. dkdv is the kernel where this does not fit -- see its own note.
    _pipe_kv = (
        os.getenv("FLYDSL_FLASH_ATTN_BWD_KV_GPFETCH", "1") == "1"
        if enable_kv_gpfetch is None
        else bool(enable_kv_gpfetch)
    )

    KT_SWZ_MASK = BLOCK_N // 4 - 1
    KT_SWZ_DSHIFT = VEC_WIDTH.bit_length() - 1

    LDS_K_TILE = BLOCK_N * K_STRIDE
    LDS_KT_TILE = HEAD_DIM * KT_STRIDE
    LDS_V_TILE = BLOCK_N * K_STRIDE
    LDS_KT_BASE = LDS_K_TILE
    LDS_V_BASE = LDS_K_TILE + LDS_KT_TILE
    LDS_TOTAL = LDS_K_TILE + LDS_KT_TILE + LDS_V_TILE
    LDS_BYTES = LDS_TOTAL * 2
    if LDS_BYTES > _LDS_LIMIT:
        raise ValueError(
            f"head_dim {HEAD_DIM} at block_n {BLOCK_N} needs {LDS_BYTES} B of LDS against "
            f"the {_LDS_LIMIT} B limit (three KV tiles: K row-major, Kᵀ, V). Halve block_n."
        )

    if waves_per_eu is None:
        # Asking for 2 when the LDS footprint already forbids a second resident workgroup
        # only earns a "failed to meet occupancy target" warning per compile, so the ask
        # tracks the footprint. This is also the knob block_n exists to move: three KV
        # tiles pass half of gfx942's 64 KB from head_dim 96 up at block_n 64, and
        # halving block_n halves the footprint.
        waves_per_eu = 2 if LDS_BYTES * 2 <= _LDS_LIMIT else 1

    SCORE_MOD = score_mod
    MASK_MOD = mask_mod
    JOINT_MOD = joint_mod
    HAS_SCORE_MOD = score_mod is not None
    HAS_MASK_MOD = mask_mod is not None
    HAS_ANY_MOD = HAS_SCORE_MOD or HAS_MASK_MOD
    MOD_VEC = int(mod_vec_size)
    NUM_AUX = int(num_aux_tensors)
    AUX_SPECS = tuple(tuple(int(s) for s in spec) for spec in (aux_specs or ()))
    AUX_NUMELS = tuple(int(n) for n in (aux_numels or ()))

    # The mod bodies are traced into the kernel, so two builds with different mods must
    # not collide on the LDS symbol name. `mod_key` is the caller's identity for its mods
    # (Inductor passes the subgraph hash); without one, fall back to the callables'
    # names, which is enough to keep hand-written mods apart.
    if mod_key is None and HAS_ANY_MOD:
        mod_key = "_".join(
            getattr(m, "__name__", "mod") for m in (score_mod, mask_mod, joint_mod) if m is not None
        )
    MOD_TAG = f"_{mod_key}_v{MOD_VEC}_a{NUM_AUX}" if HAS_ANY_MOD else ""
    USE_BLOCK_MASK = bool(block_mask)

    PATH_TAG = f"m{BLOCK_M}_n{BLOCK_N}_d{HEAD_DIM}_{dtype_str}{MOD_TAG}{'_bm' if USE_BLOCK_MASK else ''}"
    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"flex_flash_bwd_dq_smem_{PATH_TAG}_{LAYOUT}",
    )
    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + LDS_TOTAL * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flex_flash_bwd_dq_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        AUX0: fx.Tensor,
        AUX1: fx.Tensor,
        AUX2: fx.Tensor,
        AUX3: fx.Tensor,
        KV_NUM_BLOCKS: fx.Tensor,
        KV_INDICES: fx.Tensor,
        seq_len_q: fx.Int32,
        seq_len_kv: fx.Int32,
    ):
        elem_dtype = dtype_to_elem_type(dtype_str)
        elem_type = elem_dtype.ir_type
        k_ptr = _extract_aligned_pointer(K)
        v_ptr = _extract_aligned_pointer(V)

        fm_fast = fx.arith.FastMathFlags.fast
        v4f16_type = Vec.make_type(4, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        mfma_pack_type = v4f16_type

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def mfma_acc(a, b, c):
            if const_expr(dtype_str == "bf16"):
                a = Vec(a).bitcast(fx.Int16)
                b = Vec(b).bitcast(fx.Int16)
                return rocdl.mfma_f32_32x32x8bf16_1k(v16f32_type, [a, b, c])
            return rocdl.mfma_f32_32x32x8f16(v16f32_type, [a, b, c])

        seq_len_q_v = fx.Index(seq_len_q)
        seq_len_kv_v = fx.Index(seq_len_kv)

        # ---- LDS view ----
        lds = SmemPtr(allocator.get_base(), lds_offset, elem_type, shape=(LDS_TOTAL,)).get()

        # ---- Thread / wave decomposition ----
        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // _WARP_SIZE
        lane = tid % _WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32
        wave_q_offset = wave_id * _ROWS_PER_WAVE

        # ---- Block -> (batch, q head, q tile) ----
        q_head_idx = block_id % NUM_HEADS_Q
        batch_q_tile_id = block_id // NUM_HEADS_Q
        num_q_tiles = (seq_len_q_v + BLOCK_M - 1) // BLOCK_M
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * BLOCK_M
        # Python ternary so FlyDSL's `if`-rewriter does not turn this into a dynamic
        # dispatch and lose the binding (same reason as the forward).
        kv_head_idx = q_head_idx if GQA_GROUP_SIZE == 1 else q_head_idx // GQA_GROUP_SIZE

        # ---- Cooperative load decomposition ----
        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        def _kv_row_valid(lds_row):
            row_valid = lds_row < fx.Index(BLOCK_N)
            if const_expr(LOAD_HAS_IDLE_LANES):
                row_valid = row_valid & (load_row_in_batch < fx.Index(ROWS_PER_BATCH_LOAD))
            return row_valid

        # ---- Global flat indices ----
        if const_expr(BHSD):
            slice_q = (batch_idx * NUM_HEADS_Q + q_head_idx) * seq_len_q_v
            slice_kv = (batch_idx * NUM_HEADS_KV + kv_head_idx) * seq_len_kv_v

            def global_idx_q(token_idx, col):
                return (slice_q + token_idx) * STRIDE_TOKEN_Q + col

            def global_idx_kv(token_idx, col):
                return (slice_kv + token_idx) * STRIDE_TOKEN_KV + col

        else:

            def global_idx_q(token_idx, col):
                token = batch_idx * seq_len_q_v + token_idx
                return token * STRIDE_TOKEN_Q + q_head_idx * HEAD_DIM + col

            def global_idx_kv(token_idx, col):
                token = batch_idx * seq_len_kv_v + token_idx
                return token * STRIDE_TOKEN_KV + kv_head_idx * HEAD_DIM + col

        def _kv_row_clamp(row_idx):
            # KV comes in through raw pointers (no hardware bounds), so a partial tile's
            # lanes read a duplicated in-bounds row; the KV padding mask at the score site
            # then zeroes their contribution.
            last = seq_len_kv_v - fx.Index(1)
            return fx.Index(ArithValue(row_idx < seq_len_kv_v).select(row_idx, last))

        def load_global_vec(ptr, base_idx, vec_elems):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            return _pointer_load(Vec.make_type(vec_elems, elem_dtype), gep)

        def _bitcast_i32(value):
            return fx.Int32(ArithValue(value).bitcast(fx.Int32.ir_type))

        def _pack_bf16_pair(lo, hi):
            lo_i32 = _bitcast_i32(lo)
            hi_i32 = _bitcast_i32(hi)
            return (hi_i32 & fx.Int32(0xFFFF0000)) | lo_i32.shrui(fx.Int32(16))

        def pack_v4(f32_vals):
            """Four f32 -> one MFMA operand pack, by the forward's route for this dtype."""
            if const_expr(dtype_str == "bf16"):
                packed = [
                    _pack_bf16_pair(f32_vals[0], f32_vals[1]),
                    _pack_bf16_pair(f32_vals[2], f32_vals[3]),
                ]
                return Vec.from_elements(packed, fx.Int32).bitcast(elem_dtype).ir_value()
            halves = [fx.Float32(v).to(elem_dtype) for v in f32_vals]
            return Vec.from_elements(halves, elem_dtype).ir_value()

        def _k_swizzle(row_idx, col_idx):
            mask = (row_idx & fx.Index(K_SWZ_ROWMASK)) << fx.Index(4)
            return col_idx ^ mask

        def _kt_swizzle(d_idx, n_idx):
            m = (d_idx ^ (d_idx >> fx.Index(KT_SWZ_DSHIFT))) & fx.Index(KT_SWZ_MASK)
            return (((n_idx >> fx.Index(2)) ^ m) << fx.Index(2)) | (n_idx & fx.Index(3))

        # ---- Cooperative KV load: one global read of K feeds both K orientations ----
        # ---- KV staged in two halves, so the global read can run ahead ----
        # The gfx950 backward pipelines this with a DMA into a second LDS buffer; gfx942
        # has neither the 16-byte `buffer_load_lds` that needs nor the LDS to spare for a
        # second buffer. What it can do is hold the next tile in registers, which is the
        # same overlap without the extra LDS -- see `_pipe_kv` at the loop.
        def coop_load_kv_global(tile_start):
            k_vecs, v_vecs = [], []
            for batch in range_constexpr(NUM_BATCHES_KV):
                lds_row = load_row_in_batch + fx.Index(batch * ROWS_PER_BATCH_LOAD)
                row_idx = _kv_row_clamp(tile_start + lds_row)
                g_idx = global_idx_kv(row_idx, load_col_base)
                k_vecs.append(load_global_vec(k_ptr, g_idx, VEC_WIDTH))
                v_vecs.append(load_global_vec(v_ptr, g_idx, VEC_WIDTH))
            return k_vecs, v_vecs

        def coop_store_kv_lds(k_vecs, v_vecs):
            for batch in range_constexpr(NUM_BATCHES_KV):
                lds_row = load_row_in_batch + fx.Index(batch * ROWS_PER_BATCH_LOAD)
                k_vec = k_vecs[batch]
                v_vec = v_vecs[batch]

                def _store(lds_row=lds_row, k_vec=k_vec, v_vec=v_vec):
                    swz = _k_swizzle(lds_row, load_col_base)
                    Vec(k_vec).store(lds, [fx.Index(0) + lds_row * K_STRIDE + swz])
                    Vec(v_vec).store(lds, [fx.Index(LDS_V_BASE) + lds_row * K_STRIDE + swz])
                    for _e in range_constexpr(VEC_WIDTH):
                        kt_d = load_col_base + fx.Index(_e)
                        kt_idx = fx.Index(LDS_KT_BASE) + kt_d * KT_STRIDE + _kt_swizzle(kt_d, lds_row)
                        Vec.from_elements([Vec(k_vec)[_e]], elem_dtype).store(lds, [kt_idx])

                if const_expr(KV_NEEDS_GUARD):
                    if _kv_row_valid(lds_row):
                        _store()
                else:
                    _store()

        def coop_load_kv(tile_start):
            for batch in range_constexpr(NUM_BATCHES_KV):
                lds_row = load_row_in_batch + fx.Index(batch * ROWS_PER_BATCH_LOAD)
                row_idx = _kv_row_clamp(tile_start + lds_row)
                g_idx = global_idx_kv(row_idx, load_col_base)

                def _store(lds_row=lds_row, g_idx=g_idx):
                    k_vec = load_global_vec(k_ptr, g_idx, VEC_WIDTH)
                    v_vec = load_global_vec(v_ptr, g_idx, VEC_WIDTH)
                    Vec(k_vec).store(lds, [fx.Index(0) + lds_row * K_STRIDE + _k_swizzle(lds_row, load_col_base)])
                    Vec(v_vec).store(
                        lds, [fx.Index(LDS_V_BASE) + lds_row * K_STRIDE + _k_swizzle(lds_row, load_col_base)]
                    )
                    # Kᵀ: scalar stores, VEC_WIDTH of them, once per tile. The alternative
                    # is a strided scalar *read* per GEMM2 step, which happens
                    # D_CHUNKS * DS_K_STEPS times as often.
                    for _e in range_constexpr(VEC_WIDTH):
                        kt_d = load_col_base + fx.Index(_e)
                        kt_idx = fx.Index(LDS_KT_BASE) + kt_d * KT_STRIDE + _kt_swizzle(kt_d, lds_row)
                        Vec.from_elements([Vec(k_vec)[_e]], elem_dtype).store(lds, [kt_idx])

                if const_expr(KV_NEEDS_GUARD):
                    if _kv_row_valid(lds_row):
                        _store()
                else:
                    _store()

        # num_records bound: Q/DO/DQ rows past seq_len_q read 0 and drop their store, so a
        # partial Q tile needs no explicit row predicate. The bound is the end of the
        # innermost region contiguous in the layout whose tail this block could run into.
        if const_expr(BHSD):
            _q_nrec_bytes = _raw((slice_q + seq_len_q_v) * fx.Index(STRIDE_TOKEN_Q * 2))
            _row_nrec = _raw((slice_q + seq_len_q_v) * fx.Index(4))
        else:
            _q_nrec_bytes = _raw((batch_idx + fx.Index(1)) * seq_len_q_v * fx.Index(STRIDE_TOKEN_Q * 2))
            _row_nrec = _raw(
                (batch_idx * fx.Index(NUM_HEADS_Q) + q_head_idx + fx.Index(1)) * seq_len_q_v * fx.Index(4)
            )
        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=False, num_records_bytes=_q_nrec_bytes)
        do_rsrc = buffer_ops.create_buffer_resource(DO, max_size=False, num_records_bytes=_q_nrec_bytes)
        dq_rsrc = buffer_ops.create_buffer_resource(DQ, max_size=False, num_records_bytes=_q_nrec_bytes)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=False, num_records_bytes=_row_nrec)
        delta_rsrc = buffer_ops.create_buffer_resource(DELTA, max_size=False, num_records_bytes=_row_nrec)

        # ---- Per-lane row quantities ----
        # m = lane % 32 is constant per lane in every accumulator below, so `lse` and
        # `delta` are scalars here rather than anything needing a cross-lane reduction.
        # An out-of-range row reads 0 (num_records) and its store is dropped, so no
        # predicate: nothing in this kernel reduces across rows.
        q_row = q_start + wave_q_offset + lane_mod_32
        q_row_i32 = fx.Int32(q_row)
        row_off = (
            (slice_q + q_row) if const_expr(BHSD) else ((batch_idx * fx.Index(NUM_HEADS_Q) + q_head_idx) * seq_len_q_v + q_row)
        )
        lse_val = fx.Float32(buffer_ops.buffer_load(lse_rsrc, row_off, vec_width=1, dtype=T.f32))
        delta_val = fx.Float32(buffer_ops.buffer_load(delta_rsrc, row_off, vec_width=1, dtype=T.f32))

        # ---- Preload the Qᵀ and DOᵀ B-operand packs (register-resident) ----
        # B operand: j = lane % 32 (the row m), k-subblock = (lane//32)*MFMA_LANE_K. Both
        # tensors are num_records-bounded, so OOB rows read 0.
        q_b_packs = []
        do_b_packs = []
        for ks in range_constexpr(K_STEPS):
            col = fx.Index(ks * K_STEP) + lane_div_32 * MFMA_LANE_K
            g_idx = global_idx_q(q_row, col)
            q_b_packs.append(buffer_ops.buffer_load(q_rsrc, g_idx, vec_width=MFMA_LANE_K, dtype=elem_dtype))
            do_b_packs.append(buffer_ops.buffer_load(do_rsrc, g_idx, vec_width=MFMA_LANE_K, dtype=elem_dtype))

        # ---- Captured aux tensors, and the mod ABI ----
        if const_expr(NUM_AUX > 0):
            _aux_readers = [
                _make_aux_reader(
                    # See Note [aux reads run past the logical extent]
                    buffer_ops.create_buffer_resource(_buf, num_records_bytes=4 * AUX_NUMELS[_i]),
                    AUX_SPECS[_i],
                )
                for _i, _buf in enumerate([AUX0, AUX1, AUX2, AUX3][:NUM_AUX])
            ]
            _mod_kw = {"aux": _aux_readers}
        else:
            _mod_kw = {}
        _call_score_mod, _call_mask_mod, _call_joint_mod = _mod_callers(
            SCORE_MOD, MASK_MOD, JOINT_MOD, MOD_VEC, _mod_kw
        )

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        # p = exp(s_raw*σ - lse) = exp2(s_raw*(σ*log2e) - lse*log2e): one fma per element.
        # With a score_mod the mod has to see the FlexAttention score domain (q·k * σ), so
        # the σ is spent at the mod site and only log2(e) is folded into the fma -- the
        # same split the forward makes, for the same reason.
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        c_softmax_log2e = fx.Float32(_LOG2E) if HAS_SCORE_MOD else c_sm_scale_log2e
        c_sm_scale = fx.Float32(sm_scale)
        neg_lse_log2e = fx.Float32(_fmul(lse_val, fx.Float32(-_LOG2E)))

        k_swz_mask = (lane_mod_32 & fx.Index(K_SWZ_ROWMASK)) << fx.Index(4)

        def _row_major_idx(region_base, ks, strip):
            """A-operand address in a row-major KV tile: i = n, k = d."""
            col = fx.Index(ks * K_STEP) + lane_div_32 * MFMA_LANE_K
            row = lane_mod_32 + fx.Index(strip * K_SUB_N)
            return fx.Index(region_base) + row * K_STRIDE + (col ^ k_swz_mask)

        def _kt_pack_idx(dc, ks, strip):
            """A-operand address in the Kᵀ tile: i = d, k = n (4 contiguous)."""
            d_pos = fx.Index(dc * D_CHUNK) + lane_mod_32
            n_base = fx.Index(ks * DS_K_STEP + strip * K_SUB_N) + lane_div_32 * fx.Index(4)
            return fx.Index(LDS_KT_BASE) + d_pos * KT_STRIDE + _kt_swizzle(d_pos, n_base)

        # ---- KV loop: dq accumulates over n, so the KV axis is the reduction ----
        # Dense, or this Q tile's block list. The list is already on this kernel's
        # (BLOCK_M, BLOCK_N) grid and is the union of FlexAttention's partial and full
        # lists, so a visited tile always needs its mask_mod and an unvisited one is
        # genuinely empty -- see regrid_block_mask(..., walk="kv").
        if const_expr(USE_BLOCK_MASK):
            # KV_NUM_BLOCKS [B, H_q, num_q_tiles] i32
            # KV_INDICES    [B, H_q, num_q_tiles, num_kv_tiles] i32
            _nb_rsrc = buffer_ops.create_buffer_resource(KV_NUM_BLOCKS, max_size=True)
            _kvi_rsrc = buffer_ops.create_buffer_resource(KV_INDICES, max_size=True)
            _kv_tiles_total = (seq_len_kv_v + fx.Index(BLOCK_N - 1)) // fx.Index(BLOCK_N)
            _row_lin = (batch_idx * fx.Index(NUM_HEADS_Q) + q_head_idx) * num_q_tiles + q_tile_idx
            _n_visit = fx.Index(
                fx.Int32(
                    buffer_ops.buffer_load(
                        _nb_rsrc, _raw(fx.Int32(_row_lin)), vec_width=1, dtype=T.i32
                    )
                )
            )
            _kvi_base = _row_lin * _kv_tiles_total
            _kv_lo, _kv_hi, _kv_step = fx.Index(0), _n_visit, 1
        else:
            _kv_lo, _kv_hi, _kv_step = 0, seq_len_kv_v, BLOCK_N

        def _kv_tile_start(iv):
            """Only for a live iteration; a prefetch must use the clamped form below."""
            if const_expr(USE_BLOCK_MASK):
                return (
                    fx.Index(
                        fx.Int32(
                            buffer_ops.buffer_load(
                                _kvi_rsrc,
                                _raw(fx.Int32(_kvi_base + iv)),
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                    )
                    * fx.Index(BLOCK_N)
                )
            return iv

        def _kv_prefetch_tile_start(iv):
            """Tile start for a prefetch, which may reach one tile past the walk.

            `_kv_row_clamp` bounds only the top, so a stale negative int32 read from past
            the end of the tile list would become a negative row and address off the front
            of the tensor. Clamp the index to the last live entry and the tile to a
            non-negative row: a prefetch then re-reads a covered tile at worst. The dense
            walk is already safe, since an index past the end is a large positive row.
            """
            if const_expr(not USE_BLOCK_MASK):
                return iv
            _last_iv = _kv_hi - fx.Index(1)
            _iv_hi = fx.Index(ArithValue(iv < _kv_hi).select(iv, _last_iv))
            _iv_safe = fx.Index(ArithValue(_iv_hi < fx.Index(0)).select(fx.Index(0), _iv_hi))
            _tile = _kv_tile_start(_iv_safe)
            return fx.Index(ArithValue(_tile < fx.Index(0)).select(fx.Index(0), _tile))

        init_args = [c_zero_v16f32 for _ in range_constexpr(D_CHUNKS)]
        if const_expr(_pipe_kv):
            _kv0_k, _kv0_v = coop_load_kv_global(_kv_prefetch_tile_start(fx.Index(_kv_lo)))
            for _b in range_constexpr(NUM_BATCHES_KV):
                init_args.append(_kv0_k[_b])
            for _b in range_constexpr(NUM_BATCHES_KV):
                init_args.append(_kv0_v[_b])
        loop_results = init_args
        for kv_iv, inner_iter_args in range(_kv_lo, _kv_hi, _kv_step, init=init_args):
            kv_start = _kv_tile_start(kv_iv)
            dq_accs = [inner_iter_args[i] for i in range_constexpr(D_CHUNKS)]

            # Two barriers per tile: the first retires the previous iteration's LDS reads
            # before they are overwritten, the second publishes this tile's stores.
            if const_expr(_pipe_kv):
                # This tile's rows have been in flight since the previous iteration, so
                # only the LDS write is left to do here; the next tile's global read is
                # issued straight after and overlaps the GEMMs below.
                _cur_k = [inner_iter_args[D_CHUNKS + _b] for _b in range_constexpr(NUM_BATCHES_KV)]
                _cur_v = [
                    inner_iter_args[D_CHUNKS + NUM_BATCHES_KV + _b]
                    for _b in range_constexpr(NUM_BATCHES_KV)
                ]
                gpu.barrier()
                _waitcnt_vm_n(0)
                coop_store_kv_lds(_cur_k, _cur_v)
                rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                _next_k, _next_v = coop_load_kv_global(
                    _kv_prefetch_tile_start(kv_iv + fx.Index(_kv_step))
                )
                gpu.barrier()
            else:
                gpu.barrier()
                coop_load_kv(kv_start)
                gpu.barrier()

            for strip in range_constexpr(N_STRIPS):
                # ==== GEMM1: sᵀ[n, m] = k @ qᵀ, and dpᵀ[n, m] = v @ doᵀ ====
                s_acc = c_zero_v16f32
                dp_acc = c_zero_v16f32
                for ks in range_constexpr(K_STEPS):
                    k_pack = Vec.load(mfma_pack_type, lds, [_row_major_idx(0, ks, strip)])
                    v_pack = Vec.load(mfma_pack_type, lds, [_row_major_idx(LDS_V_BASE, ks, strip)])
                    s_acc = mfma_acc(k_pack, q_b_packs[ks], s_acc)
                    dp_acc = mfma_acc(v_pack, do_b_packs[ks], dp_acc)

                # ==== Elementwise: dsᵀ = joint(p * (dpᵀ - delta)) * σ ====
                # No online softmax and no rescaling: the forward already wrote the final
                # `lse`, so p is recovered in one fma + exp2 per element.
                #
                # Element -> coordinate map, the transpose of the forward's and with the
                # same 4-contiguous structure (which is what lets MOD_VEC reach 4):
                #   q_idx  = q_row                                     (one row per lane)
                #   kv_idx = kv_start + strip*K_SUB_N + (lane//32)*4 + (r//4)*8 + r%4
                kv_col_i32 = (
                    fx.Int32(kv_start)
                    + fx.Int32(lane_div_32) * fx.Int32(4)
                    + fx.Int32(strip * K_SUB_N)
                )
                seq_len_kv_i32 = fx.Int32(seq_len_kv_v)
                kv_cols = [
                    kv_col_i32 + fx.Int32((r // 4) * 8 + (r % 4)) for r in range_constexpr(16)
                ]

                # Pre-mod score in the FlexAttention domain (q·k * σ). Kept per element
                # because the joint graph is evaluated at it, not at the post-mod score.
                if const_expr(HAS_SCORE_MOD):
                    pre_mod = [_fmul(Vec(s_acc)[r], c_sm_scale) for r in range_constexpr(16)]
                else:
                    pre_mod = [Vec(s_acc)[r] for r in range_constexpr(16)]

                post_mod = list(pre_mod)
                if const_expr(HAS_ANY_MOD):
                    _mod_b = fx.Int32(batch_idx)
                    _mod_h = fx.Int32(q_head_idx)
                    for _g in range_constexpr(4):
                        for _sub in range_constexpr(4 // MOD_VEC):
                            _rs = [_g * 4 + _sub * MOD_VEC + _t for _t in range(MOD_VEC)]
                            _kvs = [kv_cols[_r] for _r in _rs]
                            _sc = [fx.Float32(post_mod[_r]) for _r in _rs]
                            if const_expr(HAS_SCORE_MOD):
                                _sc = _call_score_mod(_sc, _mod_b, _mod_h, q_row_i32, _kvs)
                            if const_expr(HAS_MASK_MOD):
                                _keep = _call_mask_mod(_mod_b, _mod_h, q_row_i32, _kvs)
                                _sc = [
                                    ArithValue(_kp).select(_sv, c_neg_inf)
                                    for _kp, _sv in zip(_keep, _sc)
                                ]
                            for _i, _r in enumerate(_rs):
                                post_mod[_r] = _sc[_i]

                ds_vals = []
                for r in range_constexpr(16):
                    # KV padding mask: a column past seq_len_kv is a clamped duplicate row,
                    # so force p = 0 rather than letting it contribute to dq. After the
                    # mods, so a score_mod cannot resurrect it -- the forward orders these
                    # the same way and for the same reason.
                    s_masked = ArithValue(kv_cols[r] >= seq_len_kv_i32).select(
                        c_neg_inf, fx.Float32(post_mod[r])
                    )
                    exponent = fmath.fma(
                        fx.Float32(s_masked), c_softmax_log2e, neg_lse_log2e, fastmath=fm_fast
                    )
                    p = ArithValue(exponent).exp2(fastmath=fm_fast)
                    # Gradient with respect to the *post*-mod score.
                    dp_minus_delta = _fsub(Vec(dp_acc)[r], delta_val)
                    ds_vals.append(_fmul(fx.Float32(p), fx.Float32(dp_minus_delta)))

                # ==== Joint-graph site: post-mod gradient -> pre-mod gradient ====
                # The chain rule through score_mod, evaluated at the pre-mod score. A
                # masked element arrives here with ds = 0 and leaves with 0, the joint
                # graph being linear in its cotangent -- which is why masking needs no
                # separate handling at this site.
                if const_expr(HAS_SCORE_MOD):
                    _mod_b = fx.Int32(batch_idx)
                    _mod_h = fx.Int32(q_head_idx)
                    for _g in range_constexpr(4):
                        for _sub in range_constexpr(4 // MOD_VEC):
                            _rs = [_g * 4 + _sub * MOD_VEC + _t for _t in range(MOD_VEC)]
                            _out = _call_joint_mod(
                                [fx.Float32(pre_mod[_r]) for _r in _rs],
                                _mod_b,
                                _mod_h,
                                q_row_i32,
                                [kv_cols[_r] for _r in _rs],
                                [fx.Float32(ds_vals[_r]) for _r in _rs],
                            )
                            for _i, _r in enumerate(_rs):
                                ds_vals[_r] = _out[_i]

                # d/d(q·k) = d/d(score) * σ, the score domain being q·k * σ. Folded into
                # the multiply above when there is no mod to spend σ at.
                ds_vals = [_fmul(fx.Float32(v), c_sm_scale) for v in ds_vals]

                # Slots [4p:4p+4] are 4 consecutive n, which is what a k-contiguous
                # operand pack wants -- so dsᵀ feeds GEMM2 with no shuffle.
                ds_packs = [pack_v4(ds_vals[p * 4 : p * 4 + 4]) for p in range_constexpr(DS_K_STEPS)]

                # ==== GEMM2: dqᵀ[d, m] += kᵀ @ dsᵀ ====
                for dc in range_constexpr(D_CHUNKS):
                    for ks in range_constexpr(DS_K_STEPS):
                        kt_pack = Vec.load(mfma_pack_type, lds, [_kt_pack_idx(dc, ks, strip)])
                        dq_accs[dc] = mfma_acc(kt_pack, ds_packs[ks], dq_accs[dc])

            _yield_args = list(dq_accs)
            if const_expr(_pipe_kv):
                for _b in range_constexpr(NUM_BATCHES_KV):
                    _yield_args.append(_next_k[_b])
                for _b in range_constexpr(NUM_BATCHES_KV):
                    _yield_args.append(_next_v[_b])
            loop_results = yield _yield_args

        # ---- Store dq ----
        # Accumulator is dqᵀ[d, m]: j = m = lane % 32, and the 16 slots walk d as
        # `(lane//32)*4 + (s//4)*8 + s%4`, so each group of 4 slots is 4 contiguous d and
        # goes out as one dwordx2. Same map as the forward's O store; OOB rows drop on
        # the num_records bound.
        dq_finals = [loop_results[dc] for dc in range_constexpr(D_CHUNKS)]
        for dc in range_constexpr(D_CHUNKS):
            for grp in range_constexpr(4):
                r0 = grp * 4
                vals = [fx.Float32(Vec(dq_finals[dc])[r0 + i]).to(elem_dtype) for i in range_constexpr(4)]
                pack = Vec.from_elements(vals, elem_dtype).bitcast(fx.Int32)
                out2 = Vec.from_elements([_raw(pack[0]), _raw(pack[1])], fx.Int32)
                d_col = fx.Index(dc * D_CHUNK) + lane_div_32 * fx.Index(4) + fx.Index(grp * 8)
                g_idx = global_idx_q(q_row, d_col)
                buffer_ops.buffer_store(out2, dq_rsrc, g_idx * fx.Index(2), offset_is_bytes=True)

    @flyc.jit
    def launch_flex_flash_bwd_dq(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        AUX0: fx.Tensor,
        AUX1: fx.Tensor,
        AUX2: fx.Tensor,
        AUX3: fx.Tensor,
        KV_NUM_BLOCKS: fx.Tensor,
        KV_INDICES: fx.Tensor,
        batch_size: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_kv: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len_q)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS_Q

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else None
        )
        flex_flash_bwd_dq_kernel(
            Q,
            K,
            V,
            DO,
            LSE,
            DELTA,
            DQ,
            AUX0,
            AUX1,
            AUX2,
            AUX3,
            KV_NUM_BLOCKS,
            KV_INDICES,
            seq_len_q,
            seq_len_kv,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": f"{BLOCK_SIZE},{BLOCK_SIZE}",
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_compile_hints):
            return launch_flex_flash_bwd_dq(*args, **kwargs)

    _launch.q_block_size = BLOCK_M
    _launch.kv_block_size = BLOCK_N
    _launch.layout = LAYOUT
    _launch.head_dim = HEAD_DIM
    _launch.smem_bytes = allocator.ptr
    _launch.num_aux_tensors = NUM_AUX
    _launch.aux_specs = AUX_SPECS
    _launch.max_aux_tensors = MAX_AUX_TENSORS
    _launch.use_block_mask = USE_BLOCK_MASK
    return _launch


def build_flex_flash_bwd_dkdv_module(
    num_heads,
    head_dim,
    dtype_str="bf16",
    sm_scale=None,
    num_kv_heads=None,
    layout="bhsd",
    block_n=128,
    block_m=32,
    waves_per_eu=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    score_mod=None,
    mask_mod=None,
    joint_mod=None,
    mod_key=None,
    num_aux_tensors=0,
    aux_specs=None,
    aux_numels=None,
    block_mask=False,
    enable_q_gpfetch=None,
):
    """Build the ``dk``/``dv`` launcher: ``(Q, K, V, DO, LSE, DELTA, DK, DV, AUX0..3, QNB, QI, B, S)``.

    Mirror image of the ``dq`` kernel: a workgroup owns a KV tile and reduces over the Q
    axis, so ``block_n`` is the *output* tile (32 rows per wave, as always) and ``block_m``
    is the reduction tile staged in LDS. Both gradients come out of one kernel because they
    share the entire ``s``/``p``/``dp``/``ds`` prologue and differ only in the last GEMM's
    A operand -- ``doᵀ`` for ``dv``, ``qᵀ`` for ``dk``.

    ``block_m`` defaults to 32 for the same reason ``dq``'s ``block_n`` is 32: four LDS
    tiles here (Q and DO, each in both orientations) are 32 KB at head_dim 128, which is
    the most that leaves two workgroups resident.

    GQA/MQA is handled *inside* the kernel, not after it. ``dk``/``dv`` for a KV head are
    the sum over every Q head sharing it, so the group is a compile-time-unrolled loop
    around the Q walk, accumulating into the same registers. No atomics and no post-pass:
    one workgroup owns each ``(batch, kv_head, kv_tile)`` outright.

    Takes the same ``score_mod``/``joint_mod``/``mask_mod`` hooks as the ``dq`` builder,
    with the same signatures, and evaluates them at the same two sites. It takes no
    ``mod_vec_size``: the vectorized mod ABI is defined over *contiguous kv_idx*, and here
    ``kv_idx`` is the per-lane constant while ``q_idx`` is what runs contiguously across
    the accumulator slots. So the mods are always called one element at a time. That costs
    only the aux-read vectorization, which was the sole real win of a wider vector anyway.

    ``block_mask`` turns the dense Q walk into a walk of this KV tile's block list -- the
    *transpose* of what ``dq`` and the forward walk, since this kernel reduces over the
    other axis. The list is per Q head, not per KV head, and that costs nothing here
    because the GQA group is already the outer loop: each Q head walks its own list into
    the same accumulators. See ``regrid_block_mask(..., walk="q")``.
    """
    gpu_arch = get_hip_arch()

    if layout not in ("bshd", "bhsd"):
        raise ValueError(f"layout must be 'bshd' or 'bhsd', got {layout!r}")
    if dtype_str not in ("f16", "bf16"):
        raise ValueError(f"flex_flash_bwd_generic supports f16 and bf16, got {dtype_str!r}")
    if head_dim % 32 != 0 or head_dim < 64:
        raise ValueError(f"head_dim must be a multiple of 32 and at least 64, got {head_dim}")
    if num_kv_heads is None:
        num_kv_heads = num_heads
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
    _validate_mod_args(
        score_mod, joint_mod, 1, num_aux_tensors, aux_specs, aux_numels
    )
    if block_mask and mask_mod is None:
        raise ValueError("block_mask needs a mask_mod: every visited block is masked")

    BLOCK_N = int(block_n)
    BLOCK_M = int(block_m)
    if BLOCK_N % _ROWS_PER_WAVE != 0:
        raise ValueError(f"block_n ({BLOCK_N}) must be a multiple of {_ROWS_PER_WAVE}")
    # Power of two and at least 32: the transposed Q/DO tiles' XOR swizzle is only a
    # permutation over a power-of-two extent, and the subtile loop steps by 32.
    if BLOCK_M % 32 != 0 or BLOCK_M & (BLOCK_M - 1) != 0:
        raise ValueError(f"block_m ({BLOCK_M}) must be a power of two and a multiple of 32")

    NUM_WAVES = BLOCK_N // _ROWS_PER_WAVE
    BLOCK_SIZE = NUM_WAVES * _WARP_SIZE
    M_SUBTILES = BLOCK_M // 32

    MFMA_LANE_K = 4
    K_STEP = 8
    K_STEPS = head_dim // K_STEP
    D_CHUNK = 32
    D_CHUNKS = head_dim // D_CHUNK
    MS_K_STEP = 8
    MS_K_STEPS = 32 // MS_K_STEP

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS_Q = num_heads
    NUM_HEADS_KV = num_kv_heads
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV
    HEAD_DIM = head_dim
    LAYOUT = layout
    BHSD = LAYOUT == "bhsd"
    STRIDE_TOKEN_Q = HEAD_DIM if BHSD else NUM_HEADS_Q * HEAD_DIM
    STRIDE_TOKEN_KV = HEAD_DIM if BHSD else NUM_HEADS_KV * HEAD_DIM

    # ---- LDS geometry: Q and DO, each row-major and transposed ----
    QT_STRIDE = BLOCK_M
    Q_GRANULES = HEAD_DIM // 16
    Q_SWZ_POW2 = Q_GRANULES & (Q_GRANULES - 1) == 0
    Q_SWZ_ROWMASK = (Q_GRANULES - 1) if Q_SWZ_POW2 else 0
    # Padded on the head_dims the swizzle cannot express, as in the forward and the dq
    # kernel. Q and DO share this stride, so both tiles get it.
    Q_PAD = 0 if Q_SWZ_POW2 else 8
    Q_STRIDE = HEAD_DIM + Q_PAD

    VEC_WIDTH = 8
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD
    LOAD_HAS_IDLE_LANES = BLOCK_SIZE % THREADS_PER_ROW_LOAD != 0
    NUM_BATCHES_Q = max(1, -(-BLOCK_M // ROWS_PER_BATCH_LOAD))
    Q_NEEDS_GUARD = LOAD_HAS_IDLE_LANES or (NUM_BATCHES_Q * ROWS_PER_BATCH_LOAD != BLOCK_M)

    # Stage Q/DO through registers so tile i+1's global read overlaps tile i's compute, as
    # `dq` does with KV. Unlike `dq`, where it wins everywhere, here it splits by head_dim,
    # and sharply: measured over 1x8x4096 and 2x16x2048, dense and causal, it is worth
    # 1.01-1.10x at 160/192/224 and costs 0.84-0.94x at 64/96/128/256, with all four
    # measurements per head_dim agreeing in sign. This kernel is the tightest of the three
    # for registers -- four LDS orientations against `dq`'s three, 2 * NUM_BATCHES_Q
    # carried vectors, and two sets of accumulators -- so below 160 the carried tile costs
    # more than the overlap buys, and at 256 the kernel already holds the whole LDS at one
    # workgroup per CU and has nothing left to spend. So it is a default keyed on the band
    # rather than a tuning axis, which is also what keeps it off the autotune bill.
    _DKDV_Q_GPFETCH_HEAD_DIMS = (160, 192, 224)
    _q_gpfetch_env = os.getenv("FLYDSL_FLASH_ATTN_BWD_DKDV_Q_GPFETCH", "")
    if enable_q_gpfetch is not None:
        _pipe_q = bool(enable_q_gpfetch)
    elif _q_gpfetch_env in ("0", "1"):
        _pipe_q = _q_gpfetch_env == "1"
    else:
        _pipe_q = HEAD_DIM in _DKDV_Q_GPFETCH_HEAD_DIMS

    QT_SWZ_MASK = BLOCK_M // 4 - 1
    QT_SWZ_DSHIFT = VEC_WIDTH.bit_length() - 1

    LDS_Q_TILE = BLOCK_M * Q_STRIDE
    LDS_QT_TILE = HEAD_DIM * QT_STRIDE
    LDS_Q_BASE = 0
    LDS_DO_BASE = LDS_Q_TILE
    LDS_QT_BASE = 2 * LDS_Q_TILE
    LDS_DOT_BASE = 2 * LDS_Q_TILE + LDS_QT_TILE
    LDS_TOTAL = 2 * LDS_Q_TILE + 2 * LDS_QT_TILE
    LDS_BYTES = LDS_TOTAL * 2
    if LDS_BYTES > _LDS_LIMIT:
        raise ValueError(
            f"head_dim {HEAD_DIM} at block_m {BLOCK_M} needs {LDS_BYTES} B of LDS against "
            f"the {_LDS_LIMIT} B limit (four Q/DO tiles, each orientation). Halve block_m."
        )

    if waves_per_eu is None:
        waves_per_eu = 2 if LDS_BYTES * 2 <= _LDS_LIMIT else 1

    SCORE_MOD = score_mod
    MASK_MOD = mask_mod
    JOINT_MOD = joint_mod
    HAS_SCORE_MOD = score_mod is not None
    HAS_MASK_MOD = mask_mod is not None
    HAS_ANY_MOD = HAS_SCORE_MOD or HAS_MASK_MOD
    NUM_AUX = int(num_aux_tensors)
    AUX_SPECS = tuple(tuple(int(s) for s in spec) for spec in (aux_specs or ()))
    AUX_NUMELS = tuple(int(n) for n in (aux_numels or ()))
    if mod_key is None and HAS_ANY_MOD:
        mod_key = "_".join(
            getattr(m, "__name__", "mod") for m in (score_mod, mask_mod, joint_mod) if m is not None
        )
    MOD_TAG = f"_{mod_key}_a{NUM_AUX}" if HAS_ANY_MOD else ""
    USE_BLOCK_MASK = bool(block_mask)

    PATH_TAG = f"n{BLOCK_N}_m{BLOCK_M}_d{HEAD_DIM}_{dtype_str}{MOD_TAG}{'_bm' if USE_BLOCK_MASK else ''}"
    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"flex_flash_bwd_dkdv_smem_{PATH_TAG}_{LAYOUT}",
    )
    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + LDS_TOTAL * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flex_flash_bwd_dkdv_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        AUX0: fx.Tensor,
        AUX1: fx.Tensor,
        AUX2: fx.Tensor,
        AUX3: fx.Tensor,
        Q_NUM_BLOCKS: fx.Tensor,
        Q_INDICES: fx.Tensor,
        seq_len_q: fx.Int32,
        seq_len_kv: fx.Int32,
    ):
        elem_dtype = dtype_to_elem_type(dtype_str)
        elem_type = elem_dtype.ir_type
        q_ptr = _extract_aligned_pointer(Q)
        do_ptr = _extract_aligned_pointer(DO)

        fm_fast = fx.arith.FastMathFlags.fast
        v4f16_type = Vec.make_type(4, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        mfma_pack_type = v4f16_type

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def mfma_acc(a, b, c):
            if const_expr(dtype_str == "bf16"):
                a = Vec(a).bitcast(fx.Int16)
                b = Vec(b).bitcast(fx.Int16)
                return rocdl.mfma_f32_32x32x8bf16_1k(v16f32_type, [a, b, c])
            return rocdl.mfma_f32_32x32x8f16(v16f32_type, [a, b, c])

        seq_len_q_v = fx.Index(seq_len_q)
        seq_len_kv_v = fx.Index(seq_len_kv)

        lds = SmemPtr(allocator.get_base(), lds_offset, elem_type, shape=(LDS_TOTAL,)).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // _WARP_SIZE
        lane = tid % _WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32
        wave_n_offset = wave_id * _ROWS_PER_WAVE

        # ---- Block -> (batch, kv head, kv tile) ----
        kv_head_idx = block_id % NUM_HEADS_KV
        batch_kv_tile_id = block_id // NUM_HEADS_KV
        num_kv_tiles = (seq_len_kv_v + BLOCK_N - 1) // BLOCK_N
        kv_tile_idx = batch_kv_tile_id % num_kv_tiles
        batch_idx = batch_kv_tile_id // num_kv_tiles
        kv_start = kv_tile_idx * BLOCK_N
        q_head_base = kv_head_idx * fx.Index(GQA_GROUP_SIZE)

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        def _q_row_valid(lds_row):
            row_valid = lds_row < fx.Index(BLOCK_M)
            if const_expr(LOAD_HAS_IDLE_LANES):
                row_valid = row_valid & (load_row_in_batch < fx.Index(ROWS_PER_BATCH_LOAD))
            return row_valid

        # ---- Global flat indices. Q/DO are indexed per Q head, which varies over the
        # GQA group, so their slice is a function rather than a constant. ----
        if const_expr(BHSD):
            slice_kv = (batch_idx * NUM_HEADS_KV + kv_head_idx) * seq_len_kv_v

            def q_plane(q_head):
                return (batch_idx * NUM_HEADS_Q + q_head) * seq_len_q_v

            def global_idx_q(q_head, token_idx, col):
                return (q_plane(q_head) + token_idx) * STRIDE_TOKEN_Q + col

            def global_idx_kv(token_idx, col):
                return (slice_kv + token_idx) * STRIDE_TOKEN_KV + col

        else:

            def q_plane(q_head):
                return batch_idx * seq_len_q_v * fx.Index(NUM_HEADS_Q) + q_head * seq_len_q_v

            def global_idx_q(q_head, token_idx, col):
                token = batch_idx * seq_len_q_v + token_idx
                return token * STRIDE_TOKEN_Q + q_head * HEAD_DIM + col

            def global_idx_kv(token_idx, col):
                token = batch_idx * seq_len_kv_v + token_idx
                return token * STRIDE_TOKEN_KV + kv_head_idx * HEAD_DIM + col

        def _row_clamp(row_idx):
            last = seq_len_q_v - fx.Index(1)
            return fx.Index(ArithValue(row_idx < seq_len_q_v).select(row_idx, last))

        def load_global_vec(ptr, base_idx, vec_elems):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            return _pointer_load(Vec.make_type(vec_elems, elem_dtype), gep)

        def _bitcast_i32(value):
            return fx.Int32(ArithValue(value).bitcast(fx.Int32.ir_type))

        def _pack_bf16_pair(lo, hi):
            lo_i32 = _bitcast_i32(lo)
            hi_i32 = _bitcast_i32(hi)
            return (hi_i32 & fx.Int32(0xFFFF0000)) | lo_i32.shrui(fx.Int32(16))

        def pack_v4(f32_vals):
            if const_expr(dtype_str == "bf16"):
                packed = [
                    _pack_bf16_pair(f32_vals[0], f32_vals[1]),
                    _pack_bf16_pair(f32_vals[2], f32_vals[3]),
                ]
                return Vec.from_elements(packed, fx.Int32).bitcast(elem_dtype).ir_value()
            halves = [fx.Float32(v).to(elem_dtype) for v in f32_vals]
            return Vec.from_elements(halves, elem_dtype).ir_value()

        def _q_swizzle(row_idx, col_idx):
            mask = (row_idx & fx.Index(Q_SWZ_ROWMASK)) << fx.Index(4)
            return col_idx ^ mask

        def _qt_swizzle(d_idx, m_idx):
            m = (d_idx ^ (d_idx >> fx.Index(QT_SWZ_DSHIFT))) & fx.Index(QT_SWZ_MASK)
            return (((m_idx >> fx.Index(2)) ^ m) << fx.Index(2)) | (m_idx & fx.Index(3))

        # ---- Cooperative Q/DO load: one global read each, two LDS orientations each ----
        # Split into the global read and the LDS write, so `_pipe_q` can put a tile's read
        # an iteration ahead of its use. The read is unconditional and row-clamped; the
        # guard belongs to the write, which is the half that must not touch a padding row.
        def coop_load_q_do_global(q_head, tile_start):
            q_vecs, do_vecs = [], []
            for batch in range_constexpr(NUM_BATCHES_Q):
                lds_row = load_row_in_batch + fx.Index(batch * ROWS_PER_BATCH_LOAD)
                row_idx = _row_clamp(tile_start + lds_row)
                g_idx = global_idx_q(q_head, row_idx, load_col_base)
                q_vecs.append(load_global_vec(q_ptr, g_idx, VEC_WIDTH))
                do_vecs.append(load_global_vec(do_ptr, g_idx, VEC_WIDTH))
            return q_vecs, do_vecs

        def coop_store_q_do_lds(q_vecs, do_vecs):
            for batch in range_constexpr(NUM_BATCHES_Q):
                lds_row = load_row_in_batch + fx.Index(batch * ROWS_PER_BATCH_LOAD)
                q_vec = q_vecs[batch]
                do_vec = do_vecs[batch]

                def _store(lds_row=lds_row, q_vec=q_vec, do_vec=do_vec):
                    swz = _q_swizzle(lds_row, load_col_base)
                    Vec(q_vec).store(lds, [fx.Index(LDS_Q_BASE) + lds_row * Q_STRIDE + swz])
                    Vec(do_vec).store(lds, [fx.Index(LDS_DO_BASE) + lds_row * Q_STRIDE + swz])
                    for _e in range_constexpr(VEC_WIDTH):
                        t_d = load_col_base + fx.Index(_e)
                        t_off = t_d * QT_STRIDE + _qt_swizzle(t_d, lds_row)
                        Vec.from_elements([Vec(q_vec)[_e]], elem_dtype).store(
                            lds, [fx.Index(LDS_QT_BASE) + t_off]
                        )
                        Vec.from_elements([Vec(do_vec)[_e]], elem_dtype).store(
                            lds, [fx.Index(LDS_DOT_BASE) + t_off]
                        )

                if const_expr(Q_NEEDS_GUARD):
                    if _q_row_valid(lds_row):
                        _store()
                else:
                    _store()

        def coop_load_q_do(q_head, tile_start):
            for batch in range_constexpr(NUM_BATCHES_Q):
                lds_row = load_row_in_batch + fx.Index(batch * ROWS_PER_BATCH_LOAD)
                row_idx = _row_clamp(tile_start + lds_row)
                g_idx = global_idx_q(q_head, row_idx, load_col_base)

                def _store(lds_row=lds_row, g_idx=g_idx):
                    q_vec = load_global_vec(q_ptr, g_idx, VEC_WIDTH)
                    do_vec = load_global_vec(do_ptr, g_idx, VEC_WIDTH)
                    swz = _q_swizzle(lds_row, load_col_base)
                    Vec(q_vec).store(lds, [fx.Index(LDS_Q_BASE) + lds_row * Q_STRIDE + swz])
                    Vec(do_vec).store(lds, [fx.Index(LDS_DO_BASE) + lds_row * Q_STRIDE + swz])
                    # Transposed copies: VEC_WIDTH scalar stores each, once per tile,
                    # against a strided scalar read per GEMM2 step if we skipped them.
                    for _e in range_constexpr(VEC_WIDTH):
                        t_d = load_col_base + fx.Index(_e)
                        t_off = t_d * QT_STRIDE + _qt_swizzle(t_d, lds_row)
                        Vec.from_elements([Vec(q_vec)[_e]], elem_dtype).store(
                            lds, [fx.Index(LDS_QT_BASE) + t_off]
                        )
                        Vec.from_elements([Vec(do_vec)[_e]], elem_dtype).store(
                            lds, [fx.Index(LDS_DOT_BASE) + t_off]
                        )

                if const_expr(Q_NEEDS_GUARD):
                    if _q_row_valid(lds_row):
                        _store()
                else:
                    _store()

        # K/V and DK/DV are all indexed by the kv head this block owns, so one
        # num_records bound serves them: the end of the innermost region contiguous in the
        # layout whose tail this block could run into.
        if const_expr(BHSD):
            _kv_nrec_bytes = _raw((slice_kv + seq_len_kv_v) * fx.Index(STRIDE_TOKEN_KV * 2))
        else:
            _kv_nrec_bytes = _raw((batch_idx + fx.Index(1)) * seq_len_kv_v * fx.Index(STRIDE_TOKEN_KV * 2))
        k_rsrc = buffer_ops.create_buffer_resource(K, max_size=False, num_records_bytes=_kv_nrec_bytes)
        v_rsrc = buffer_ops.create_buffer_resource(V, max_size=False, num_records_bytes=_kv_nrec_bytes)
        dk_rsrc = buffer_ops.create_buffer_resource(DK, max_size=False, num_records_bytes=_kv_nrec_bytes)
        dv_rsrc = buffer_ops.create_buffer_resource(DV, max_size=False, num_records_bytes=_kv_nrec_bytes)
        # LSE/DELTA are bounded by the whole tensor rather than per Q head: the head
        # varies over the GQA group, and a resource cannot. An out-of-range row can then
        # read the next head's value, which is why `p` is masked to 0 there rather than
        # relying on the read to return 0 -- and it has to be masked anyway, since `m` is
        # this kernel's reduction axis. See the mask at the score site.
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        delta_rsrc = buffer_ops.create_buffer_resource(DELTA, max_size=True)

        # ---- Preload K and V as B operands (register-resident for the whole Q walk) ----
        # B operand: j = n = lane % 32 (this wave's KV row), k-subblock = (lane//32)*4.
        kv_row = kv_start + wave_n_offset + lane_mod_32
        k_b_packs = []
        v_b_packs = []
        for ks in range_constexpr(K_STEPS):
            col = fx.Index(ks * K_STEP) + lane_div_32 * MFMA_LANE_K
            g_idx = global_idx_kv(kv_row, col)
            k_b_packs.append(buffer_ops.buffer_load(k_rsrc, g_idx, vec_width=MFMA_LANE_K, dtype=elem_dtype))
            v_b_packs.append(buffer_ops.buffer_load(v_rsrc, g_idx, vec_width=MFMA_LANE_K, dtype=elem_dtype))

        # ---- Captured aux tensors, and the mod ABI (mod_vec is always 1 here) ----
        if const_expr(NUM_AUX > 0):
            _aux_readers = [
                _make_aux_reader(
                    # See Note [aux reads run past the logical extent]
                    buffer_ops.create_buffer_resource(_buf, num_records_bytes=4 * AUX_NUMELS[_i]),
                    AUX_SPECS[_i],
                )
                for _i, _buf in enumerate([AUX0, AUX1, AUX2, AUX3][:NUM_AUX])
            ]
            _mod_kw = {"aux": _aux_readers}
        else:
            _mod_kw = {}
        _call_score_mod, _call_mask_mod, _call_joint_mod = _mod_callers(
            SCORE_MOD, MASK_MOD, JOINT_MOD, 1, _mod_kw
        )

        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        c_zero_f = fx.Float32(0.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        # With a score_mod the σ is spent at the mod site so the mod sees the
        # FlexAttention score domain; see the same split in the dq builder.
        c_softmax_log2e = fx.Float32(_LOG2E) if HAS_SCORE_MOD else c_sm_scale_log2e
        c_sm_scale = fx.Float32(sm_scale)
        c_neg_log2e = fx.Float32(-_LOG2E)

        q_swz_mask = (lane_mod_32 & fx.Index(Q_SWZ_ROWMASK)) << fx.Index(4)

        def _row_major_idx(region_base, ks, m_sub):
            """GEMM1 A operand in a row-major Q/DO tile: i = m, k = d."""
            col = fx.Index(ks * K_STEP) + lane_div_32 * MFMA_LANE_K
            row = lane_mod_32 + fx.Index(m_sub * 32)
            return fx.Index(region_base) + row * Q_STRIDE + (col ^ q_swz_mask)

        def _transposed_idx(region_base, dc, ks, m_sub):
            """GEMM2 A operand in a transposed Q/DO tile: i = d, k = m (4 contiguous)."""
            d_pos = fx.Index(dc * D_CHUNK) + lane_mod_32
            m_base = fx.Index(ks * MS_K_STEP + m_sub * 32) + lane_div_32 * fx.Index(4)
            return fx.Index(region_base) + d_pos * QT_STRIDE + _qt_swizzle(d_pos, m_base)

        # ---- Q loop, wrapped in the GQA group ----
        # dk/dv for one KV head is the sum over every Q head sharing it, so the group is
        # unrolled around the Q walk and accumulates into the same registers.
        init_args = [c_zero_v16f32 for _ in range_constexpr(2 * D_CHUNKS)]
        carried = init_args
        if const_expr(USE_BLOCK_MASK):
            # Q_NUM_BLOCKS [B, H_q, num_kv_tiles] i32
            # Q_INDICES    [B, H_q, num_kv_tiles, num_q_tiles] i32
            _qnb_rsrc = buffer_ops.create_buffer_resource(Q_NUM_BLOCKS, max_size=True)
            _qi_rsrc = buffer_ops.create_buffer_resource(Q_INDICES, max_size=True)
            _q_tiles_total = (seq_len_q_v + fx.Index(BLOCK_M - 1)) // fx.Index(BLOCK_M)
        for gq in range_constexpr(GQA_GROUP_SIZE):
            q_head = q_head_base + fx.Index(gq)
            lse_plane = q_plane(q_head) if const_expr(BHSD) else (
                (batch_idx * fx.Index(NUM_HEADS_Q) + q_head) * seq_len_q_v
            )
            # Per Q head, which is free here: the group is the outer loop, so each head
            # walks its own list into the shared dk/dv accumulators.
            if const_expr(USE_BLOCK_MASK):
                _row_lin = (
                    batch_idx * fx.Index(NUM_HEADS_Q) + q_head
                ) * num_kv_tiles + kv_tile_idx
                _n_visit = fx.Index(
                    fx.Int32(
                        buffer_ops.buffer_load(
                            _qnb_rsrc, _raw(fx.Int32(_row_lin)), vec_width=1, dtype=T.i32
                        )
                    )
                )
                _qi_base = _row_lin * _q_tiles_total
                _m_lo, _m_hi, _m_step = fx.Index(0), _n_visit, 1
            else:
                _m_lo, _m_hi, _m_step = 0, seq_len_q_v, BLOCK_M

            def _q_tile_start(iv):
                """First row of the Q tile a live iteration visits."""
                if const_expr(not USE_BLOCK_MASK):
                    return iv
                return fx.Index(
                    fx.Int32(
                        buffer_ops.buffer_load(
                            _qi_rsrc,
                            _raw(fx.Int32(_qi_base + iv)),
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                ) * fx.Index(BLOCK_M)

            def _q_prefetch_tile_start(iv):
                """As above, but safe for the lookahead that runs one past the walk.

                Same two-sided clamp as the forward and `dq`: the index is pinned to the
                last live entry so the read cannot leave the tile list, and the tile it
                yields is pinned non-negative so a stale value cannot address off the
                front of Q. Costs a re-read of a tile already covered, nothing worse.
                """
                if const_expr(not USE_BLOCK_MASK):
                    return iv
                _last_iv = _m_hi - fx.Index(1)
                _iv_hi = fx.Index(ArithValue(iv < _m_hi).select(iv, _last_iv))
                _iv_safe = fx.Index(ArithValue(_iv_hi < fx.Index(0)).select(fx.Index(0), _iv_hi))
                _tile = _q_tile_start(_iv_safe)
                return fx.Index(ArithValue(_tile < fx.Index(0)).select(fx.Index(0), _tile))

            # Primed per Q head: the group is the outer loop, so each head's walk starts
            # its own pipeline rather than inheriting the previous head's in-flight tile.
            _loop_init = list(carried)
            if const_expr(_pipe_q):
                _q0, _do0 = coop_load_q_do_global(
                    q_head, _q_prefetch_tile_start(fx.Index(_m_lo))
                )
                for _b in range_constexpr(NUM_BATCHES_Q):
                    _loop_init.append(_q0[_b])
                for _b in range_constexpr(NUM_BATCHES_Q):
                    _loop_init.append(_do0[_b])
            for m_iv, inner_iter_args in range(_m_lo, _m_hi, _m_step, init=_loop_init):
                m_tile = _q_tile_start(m_iv)
                dk_accs = [inner_iter_args[i] for i in range_constexpr(D_CHUNKS)]
                dv_accs = [inner_iter_args[D_CHUNKS + i] for i in range_constexpr(D_CHUNKS)]

                if const_expr(_pipe_q):
                    # This tile's rows have been in flight since the previous iteration,
                    # so only the LDS write is left; the next tile's read is issued
                    # straight after and overlaps the GEMMs below.
                    _cur_q = [
                        inner_iter_args[2 * D_CHUNKS + _b]
                        for _b in range_constexpr(NUM_BATCHES_Q)
                    ]
                    _cur_do = [
                        inner_iter_args[2 * D_CHUNKS + NUM_BATCHES_Q + _b]
                        for _b in range_constexpr(NUM_BATCHES_Q)
                    ]
                    gpu.barrier()
                    _waitcnt_vm_n(0)
                    coop_store_q_do_lds(_cur_q, _cur_do)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    _next_q, _next_do = coop_load_q_do_global(
                        q_head, _q_prefetch_tile_start(m_iv + fx.Index(_m_step))
                    )
                    gpu.barrier()
                else:
                    gpu.barrier()
                    coop_load_q_do(q_head, m_tile)
                    gpu.barrier()

                for m_sub in range_constexpr(M_SUBTILES):
                    # ==== GEMM1: s[m, n] = q @ kᵀ, dp[m, n] = do @ vᵀ ====
                    # Operands swapped relative to the dq kernel: here the reduction is
                    # over m, so the accumulator must have m on its slot axis.
                    s_acc = c_zero_v16f32
                    dp_acc = c_zero_v16f32
                    for ks in range_constexpr(K_STEPS):
                        q_pack = Vec.load(mfma_pack_type, lds, [_row_major_idx(LDS_Q_BASE, ks, m_sub)])
                        do_pack = Vec.load(mfma_pack_type, lds, [_row_major_idx(LDS_DO_BASE, ks, m_sub)])
                        s_acc = mfma_acc(q_pack, k_b_packs[ks], s_acc)
                        dp_acc = mfma_acc(do_pack, v_b_packs[ks], dp_acc)

                    # ==== Per-row lse/delta: 16 slots, 4 groups of 4 consecutive m ====
                    m_base = m_tile + fx.Index(m_sub * 32) + lane_div_32 * fx.Index(4)
                    lse_vals = []
                    delta_vals = []
                    for grp in range_constexpr(4):
                        off = lse_plane + m_base + fx.Index(grp * 8)
                        lse_vals.append(buffer_ops.buffer_load(lse_rsrc, off, vec_width=4, dtype=T.f32))
                        delta_vals.append(buffer_ops.buffer_load(delta_rsrc, off, vec_width=4, dtype=T.f32))

                    # ==== Elementwise: ds[m, n] = joint(p * (dp - delta[m])) * σ ====
                    # Element -> coordinate map, the mirror of the dq kernel's: here it is
                    # kv_idx that is the per-lane constant and q_idx that runs across the
                    # slots, which is why the mods are called one element at a time.
                    #   kv_idx = kv_start + wave_n_offset + lane%32   (one column per lane)
                    #   q_idx  = m_tile + m_sub*32 + (lane//32)*4 + (r//4)*8 + r%4
                    seq_len_q_i32 = fx.Int32(seq_len_q_v)
                    m_base_i32 = fx.Int32(m_base)
                    kv_row_i32 = fx.Int32(kv_row)
                    m_rows = [
                        m_base_i32 + fx.Int32((r // 4) * 8 + (r % 4)) for r in range_constexpr(16)
                    ]

                    if const_expr(HAS_SCORE_MOD):
                        pre_mod = [_fmul(Vec(s_acc)[r], c_sm_scale) for r in range_constexpr(16)]
                    else:
                        pre_mod = [Vec(s_acc)[r] for r in range_constexpr(16)]

                    post_mod = list(pre_mod)
                    if const_expr(HAS_ANY_MOD):
                        _mod_b = fx.Int32(batch_idx)
                        _mod_h = fx.Int32(q_head)
                        for r in range_constexpr(16):
                            _sv = fx.Float32(post_mod[r])
                            if const_expr(HAS_SCORE_MOD):
                                _sv = _call_score_mod(
                                    [_sv], _mod_b, _mod_h, m_rows[r], [kv_row_i32]
                                )[0]
                            if const_expr(HAS_MASK_MOD):
                                _keep = _call_mask_mod(
                                    _mod_b, _mod_h, m_rows[r], [kv_row_i32]
                                )[0]
                                _sv = ArithValue(_keep).select(fx.Float32(_sv), c_neg_inf)
                            post_mod[r] = _sv

                    p_vals = []
                    ds_vals = []
                    for r in range_constexpr(16):
                        grp, sub = r // 4, r % 4
                        lse_r = fx.Float32(Vec(lse_vals[grp])[sub])
                        delta_r = fx.Float32(Vec(delta_vals[grp])[sub])
                        exponent = fmath.fma(
                            fx.Float32(post_mod[r]),
                            c_softmax_log2e,
                            fx.Float32(_fmul(lse_r, c_neg_log2e)),
                            fastmath=fm_fast,
                        )
                        p = fx.Float32(ArithValue(exponent).exp2(fastmath=fm_fast))
                        # m is the reduction axis here, so a row past seq_len_q would be
                        # *summed into* dk/dv rather than dropped at the store the way an
                        # out-of-range row is in the dq kernel. Select p rather than
                        # multiplying by a mask: the lse read above is bounded by the whole
                        # tensor, so it can be another head's value and the exponent can
                        # come out inf, which a multiply would turn into NaN.
                        p = ArithValue(m_rows[r] >= seq_len_q_i32).select(c_zero_f, p)
                        p_vals.append(p)
                        dp_minus_delta = _fsub(Vec(dp_acc)[r], delta_r)
                        ds_vals.append(_fmul(p, fx.Float32(dp_minus_delta)))

                    # ==== Joint-graph site: post-mod gradient -> pre-mod gradient ====
                    if const_expr(HAS_SCORE_MOD):
                        _mod_b = fx.Int32(batch_idx)
                        _mod_h = fx.Int32(q_head)
                        for r in range_constexpr(16):
                            ds_vals[r] = _call_joint_mod(
                                [fx.Float32(pre_mod[r])],
                                _mod_b,
                                _mod_h,
                                m_rows[r],
                                [kv_row_i32],
                                [fx.Float32(ds_vals[r])],
                            )[0]

                    # d/d(q·k) = d/d(score) * σ.
                    ds_vals = [_fmul(fx.Float32(v), c_sm_scale) for v in ds_vals]

                    # Slots [4p:4p+4] are 4 consecutive m, which is the k-contiguity the
                    # next GEMM's B operand wants -- so both fragments feed it directly.
                    p_packs = [pack_v4(p_vals[p * 4 : p * 4 + 4]) for p in range_constexpr(MS_K_STEPS)]
                    ds_packs = [pack_v4(ds_vals[p * 4 : p * 4 + 4]) for p in range_constexpr(MS_K_STEPS)]

                    # ==== GEMM2: dvᵀ[d, n] += doᵀ @ p, dkᵀ[d, n] += qᵀ @ ds ====
                    for dc in range_constexpr(D_CHUNKS):
                        for ks in range_constexpr(MS_K_STEPS):
                            dot_pack = Vec.load(
                                mfma_pack_type, lds, [_transposed_idx(LDS_DOT_BASE, dc, ks, m_sub)]
                            )
                            qt_pack = Vec.load(
                                mfma_pack_type, lds, [_transposed_idx(LDS_QT_BASE, dc, ks, m_sub)]
                            )
                            dv_accs[dc] = mfma_acc(dot_pack, p_packs[ks], dv_accs[dc])
                            dk_accs[dc] = mfma_acc(qt_pack, ds_packs[ks], dk_accs[dc])

                _yielded = dk_accs + dv_accs
                if const_expr(_pipe_q):
                    for _b in range_constexpr(NUM_BATCHES_Q):
                        _yielded = _yielded + [_next_q[_b]]
                    for _b in range_constexpr(NUM_BATCHES_Q):
                        _yielded = _yielded + [_next_do[_b]]
                carried = yield _yielded
            if const_expr(_pipe_q):
                # Drop the in-flight tile: the next Q head primes its own, and the store
                # below wants only the accumulators.
                carried = [carried[i] for i in range_constexpr(2 * D_CHUNKS)]

        # ---- Store dk and dv ----
        # Accumulators are dkᵀ/dvᵀ[d, n]: j = n = lane % 32, 16 slots walking d as
        # `(lane//32)*4 + (s//4)*8 + s%4`, so each group of 4 slots is 4 contiguous d and
        # goes out as one dwordx2. Rows past seq_len_q drop on the num_records bound.
        dk_finals = [carried[i] for i in range_constexpr(D_CHUNKS)]
        dv_finals = [carried[D_CHUNKS + i] for i in range_constexpr(D_CHUNKS)]
        for acc_list, rsrc in ((dk_finals, dk_rsrc), (dv_finals, dv_rsrc)):
            for dc in range_constexpr(D_CHUNKS):
                for grp in range_constexpr(4):
                    r0 = grp * 4
                    vals = [
                        fx.Float32(Vec(acc_list[dc])[r0 + i]).to(elem_dtype)
                        for i in range_constexpr(4)
                    ]
                    pack = Vec.from_elements(vals, elem_dtype).bitcast(fx.Int32)
                    out2 = Vec.from_elements([_raw(pack[0]), _raw(pack[1])], fx.Int32)
                    d_col = fx.Index(dc * D_CHUNK) + lane_div_32 * fx.Index(4) + fx.Index(grp * 8)
                    g_idx = global_idx_kv(kv_row, d_col)
                    buffer_ops.buffer_store(out2, rsrc, g_idx * fx.Index(2), offset_is_bytes=True)

    @flyc.jit
    def launch_flex_flash_bwd_dkdv(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        AUX0: fx.Tensor,
        AUX1: fx.Tensor,
        AUX2: fx.Tensor,
        AUX3: fx.Tensor,
        Q_NUM_BLOCKS: fx.Tensor,
        Q_INDICES: fx.Tensor,
        batch_size: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_kv: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len_kv)
        num_kv_tiles = (sl_idx + BLOCK_N - 1) // BLOCK_N
        grid_x = bs_idx * num_kv_tiles * NUM_HEADS_KV

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else None
        )
        flex_flash_bwd_dkdv_kernel(
            Q,
            K,
            V,
            DO,
            LSE,
            DELTA,
            DK,
            DV,
            AUX0,
            AUX1,
            AUX2,
            AUX3,
            Q_NUM_BLOCKS,
            Q_INDICES,
            seq_len_q,
            seq_len_kv,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": f"{BLOCK_SIZE},{BLOCK_SIZE}",
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_compile_hints):
            return launch_flex_flash_bwd_dkdv(*args, **kwargs)

    _launch.q_block_size = BLOCK_M
    _launch.kv_block_size = BLOCK_N
    _launch.layout = LAYOUT
    _launch.head_dim = HEAD_DIM
    _launch.smem_bytes = allocator.ptr
    _launch.num_aux_tensors = NUM_AUX
    _launch.aux_specs = AUX_SPECS
    _launch.max_aux_tensors = MAX_AUX_TENSORS
    _launch.use_block_mask = USE_BLOCK_MASK
    return _launch

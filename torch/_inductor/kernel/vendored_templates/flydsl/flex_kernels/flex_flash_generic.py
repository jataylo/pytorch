# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlexAttention-capable flash-attention forward kernel builder (generic / portable path).

Derived from `kernels/flash_attn_generic.py`. **This is a separate experimental copy** —
the vendored kernels under `flydsl-rfc/kernels/` are left untouched so the flex work can
iterate without destabilising them. Re-sync by diffing against the upstream file.

FlexAttention additions over the upstream kernel (forward only):

- ``score_mod(score, b, h, q_idx, kv_idx) -> score`` evaluated at the score site,
  before the masks, in the FlexAttention score domain (``q·k * sm_scale``).
- ``mask_mod(b, h, q_idx, kv_idx) -> bool`` (True = keep) applied after ``score_mod``.
- ``return_lse``: emit the per-row log-sum-exp that FlexAttention's template signature
  and backward pass require (upstream computes and discards it).
- ``block_mask``: skip whole KV blocks using a ``BlockMask``-style ``kv_indices`` list
  instead of walking the KV axis densely, with ``mask_mod`` applied only on partial
  blocks.
- ``mod_vec_size``: evaluate the mods over 1/2/4 contiguous KV elements at a time.
- ``aux_tensors``: captured tensors read inside a mod at ``(b, h, q_idx, kv_idx)``.

Removed relative to upstream, to keep the experiment small: the gfx950 DUALWAVE_SWP
dispatch (see `flex_flash_950.py`), the M128/M256 runtime auto-dispatch (one tile shape
per build), varlen/cross-seqlen, and the backward pass (none exists upstream either).

- True MFMA32 remap: `mfma_f32_32x32x16bf16` / `mfma_f32_32x32x16f16` for both GEMM stages.
- Tile shape: BLOCK_M=128 or 256 (auto-selected), BLOCK_N=64.
- BLOCK_M=128: 4 waves (256 threads), BLOCK_M=256: 8 waves (512 threads).
- Per-wave Q rows: 32.
- GEMM1 uses `K @ Q^T` so S/P live in MFMA32 register layout.
- Online softmax over KV dimension is done in registers.
- P is kept in registers and fed directly to GEMM2 (`V^T @ P`) without LDS roundtrip.
- K and V use separate LDS regions with DMA-to-LDS prefetch and XOR swizzle.
- For H>=32, both M=128 and M=256 variants are built and dispatched at runtime.

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,) where num_q_tiles = seq_len / BLOCK_M.
Block:  (256,) or (512,) depending on BLOCK_M.

Requires: head_dim == 128. The inherited claim of "head_dim % 32 == 0, head_dim >= 64" is
wrong: 64 builds and returns wrong numbers (~44% relative error), and the other multiples
of 32 assert during build. See phase1/probe_upstream_head_dim.py.

seq_len needs no alignment. The inherited claim of "seq_len % 128 == 0" is stale --
ragged lengths are handled by the bounds checks in the tile loop, and match eager to the
same tolerance as aligned ones.
"""

import math as host_math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from torch._inductor.kernel.vendored_templates.flydsl.kernels.kernels_common import (
    dtype_to_elem_type,
)

_LOG2E = host_math.log2(host_math.e)  # 1.4426950408889634
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3


def _llvm_value(value):
    """Unwrap FlyDSL scalar/vector wrappers for LLVM pointer load ops."""
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _extract_aligned_pointer(tensor, address_space=None) -> ir.Value:
    """Extract the aligned LLVM pointer from a FlyDSL tensor/memref."""
    from flydsl._mlir.dialects import fly as _fly

    ptr_type = ir.Type.parse("!llvm.ptr" if address_space is None else f"!llvm.ptr<{address_space}>")
    return _fly.extract_aligned_pointer_as_index(ptr_type, _llvm_value(tensor))


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    rocdl.s_waitcnt(val)


def build_flex_flash_generic_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="f16",
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    path_tag="auto",
    num_kv_heads=None,
    score_mod=None,
    mask_mod=None,
    mod_key=None,
    return_lse=False,
    mod_vec_size=1,
    num_aux_tensors=0,
    aux_specs=None,
    block_mask=False,
    sparse_kv_block_size=None,
):
    """Build the FlexAttention-capable flash-attention forward launcher.

    For GQA/MQA pass ``num_kv_heads < num_heads``. ``num_heads`` is the Q head
    count, ``num_kv_heads`` is the KV head count, and we require
    ``num_heads % num_kv_heads == 0``. Default ``num_kv_heads = num_heads`` (MHA).
    Q/O still have ``num_heads`` heads; K/V have ``num_kv_heads`` heads, with
    every ``num_heads // num_kv_heads`` consecutive Q heads sharing one KV head.

    FlexAttention hooks (all evaluated at build time with FlyDSL device values):

    ``score_mod(score, b, h, q_idx, kv_idx) -> score``
        Per-element score transform. ``score`` is f32 in the FlexAttention score
        domain (``q·k * sm_scale``); the four indices are i32. Runs *before* the
        masks, matching ``torch.nn.attention.flex_attention`` semantics: a
        non-additive mod (soft-cap) applied after a mask would turn ``-inf`` back
        into a finite value.

    ``mask_mod(b, h, q_idx, kv_idx) -> i1``
        True keeps the element, False forces it to ``-inf``. Applied after
        ``score_mod``, and skipped entirely on blocks the ``BlockMask`` marks as
        fully unmasked.

    With ``mod_vec_size > 1`` both mods are handed lists of ``mod_vec_size``
    contiguous-in-``kv_idx`` values instead of scalars (``score`` and ``kv_idx``
    become lists; ``b``/``h``/``q_idx`` stay scalar, being uniform across the
    group). ``mod_vec_size`` must divide 4 — the score layout only guarantees 4
    contiguous KV columns per lane (see the coordinate map at the score site).

    ``mod_key`` is an opaque token mixed into the generated symbol names. Pass a
    stable hash of the lowered subgraph whenever a mod is supplied: FlyDSL's JIT
    cache key hashes traced source plus *scalar closure values*, so two mods
    differing only in a constant read from a module global collide and the stale
    binary is silently reused.

    ``return_lse`` appends an LSE output (``[B, H, S]`` f32, natural log) to the
    launcher.

    ``num_aux_tensors`` (0-2) enables captured-tensor reads inside the mods. Each
    needs an ``aux_specs`` entry ``(stride_b, stride_h, stride_q, stride_kv)`` of
    element strides, with 0 meaning "broadcast over this axis" — so an ``[H]``
    ALiBi slope table is ``(0, 1, 0, 0)`` and a full ``[B,H,S,S]`` bias is
    ``(H*S*S, S*S, S, 1)``. Mods then receive an extra ``aux`` keyword: a list of
    readers callable as ``aux[i](b, h, q_idx, kv_idx) -> f32``.

    ``block_mask`` switches the KV loop from a dense walk to a ``kv_indices``-driven
    one; ``sparse_kv_block_size`` must then equal the kernel's KV tile
    (``BLOCK_N_OUT``, queryable via ``launcher.kv_block_size``).
    """
    gpu_arch = get_hip_arch()

    if mod_vec_size not in (1, 2, 4):
        raise ValueError(f"mod_vec_size must be 1, 2 or 4 (4 contiguous KV cols per lane), got {mod_vec_size}")
    if num_aux_tensors not in (0, 1, 2):
        raise ValueError(f"num_aux_tensors must be 0, 1 or 2, got {num_aux_tensors}")
    if (score_mod is not None or mask_mod is not None) and mod_key is None:
        raise ValueError(
            "pass an explicit mod_key when supplying score_mod/mask_mod: the JIT cache key "
            "cannot see constants a mod reads from module globals, so distinct mods would "
            "silently share a compiled binary"
        )
    if block_mask and mask_mod is None:
        raise ValueError("block_mask=True requires a mask_mod (partial blocks need per-element masking)")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

    BLOCK_N = 64
    K_SUB_N = 32
    WARP_SIZE = 64

    # Upstream dispatches gfx950 D=128 bf16/f16 to the DUALWAVE_SWP kernel here, and
    # builds both M=128 and M=256 variants for H>=32 to route on B*S at runtime. Both
    # are dropped in the flex build: one tile shape per build keeps the mod-injection
    # cost attributable to a single kernel, and the gfx950 fast path lives in
    # `flex_flash_950.py` with its own hooks (its score domain differs, so it cannot
    # share this one's mod call sites).

    # Positional index of seq_len in the launcher signature
    # (Q, K, V, O, LSE, KV_NUM_BLOCKS, KV_INDICES, AUX0, AUX1, batch_size, seq_len).
    _SEQ_LEN_ARG = 10

    def _extract_seq_len(args, kwargs):
        """Return the launch-time seq_len as int, or None if not statically known."""
        S = args[_SEQ_LEN_ARG] if len(args) > _SEQ_LEN_ARG else kwargs.get("seq_len", None)
        try:
            return int(S)
        except (TypeError, ValueError):
            return None

    def _guard_seqlen(_dispatched):
        """Enforce the only correctness floor (seq_len >= 1). A symbolic/non-int
        seq_len is let through; dense routing is a perf policy, not a bound."""

        def _guarded(*args, **kwargs):
            S_int = _extract_seq_len(args, kwargs)
            if S_int is not None and S_int < 1:
                raise ValueError(f"flex_flash_generic: seq_len must be >= 1, got {S_int}.")
            return _dispatched(*args, **kwargs)

        if hasattr(_dispatched, "compile"):
            _guarded.compile = _dispatched.compile
        return _guarded

    if block_m is not None:
        BLOCK_M = block_m
    else:
        BLOCK_M = 256 if num_heads >= 32 else 128

    if flat_work_group_size is None:
        if BLOCK_M <= 128:
            flat_work_group_size = 256
        else:
            flat_work_group_size = 512
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    if path_tag.upper() in ("N32", "N128"):
        PATH_TAG = path_tag.upper()
    elif dtype_str in ("f16", "bf16") and causal and head_dim == 128:
        PATH_TAG = "N128"
    else:
        PATH_TAG = "N32"
    BLOCK_N_OUT = 128 if PATH_TAG == "N128" else BLOCK_N
    N_SUBTILES = BLOCK_N_OUT // BLOCK_N
    ENABLE_PREFETCH_3BUF = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_PREFETCH3", "0") == "1"
    # buffer_load_dwordx4_lds (16B DMA-to-LDS) requires gfx950+; gfx94x only has dword (4B).
    _has_lds_load_b128 = not gpu_arch.startswith("gfx942")
    ENABLE_DMA = _has_lds_load_b128 and (
        PATH_TAG == "N128" or (os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_DMA", "0") == "1")
    )
    ENABLE_LDS_VEC16 = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    REDUCE_MODE = os.getenv("FLYDSL_FLASH_ATTN_FUNC_REDUCE_MODE", "xor").strip().lower()
    if REDUCE_MODE not in ("xor", "ds_bpermute"):
        REDUCE_MODE = "xor"
    NUM_PREFETCH_K = 3 if ENABLE_PREFETCH_3BUF else (2 if ENABLE_DMA else 1)
    NUM_PREFETCH_V = 3 if ENABLE_PREFETCH_3BUF else 1
    CK_LDS_SEQ = (1, 2, 0, 1, 0, 1, 2, 0) if ENABLE_PREFETCH_3BUF else (0,)

    # gfx950+ has ds_read_tr16_b64 (HW transpose LDS read); gfx942 needs V^T stored in LDS.
    USE_HW_TR = gpu_arch.startswith("gfx950")

    # MFMA32 K-dimension: 16 on gfx950+ (CDNA4) for both GEMMs.
    USE_K16 = gpu_arch.startswith("gfx950")

    # 128-bit permlane-fused O-store needs gfx950 (permlane32_swap + cvt_pk_bf16_f32,
    # both CDNA4-only); gfx942 falls back to a per-lane dwordx2 store via .to(elem_dtype).
    USE_PERMLANE_OSTORE = gpu_arch.startswith("gfx950")
    K_STEP_QK = 16 if USE_K16 else 8
    K_STEPS_QK = head_dim // K_STEP_QK
    D_CHUNK = 32
    D_CHUNKS = head_dim // D_CHUNK
    PV_K_STEP = 16 if USE_K16 else 8
    PV_K_STEPS = K_SUB_N // PV_K_STEP  # 2 steps per sub-tile (K=16) or 4 (K=8)

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 32 == 0, f"head_dim ({head_dim}) must be divisible by 32"
    assert head_dim >= 64, f"head_dim ({head_dim}) must be >= 64"
    assert flat_work_group_size in (
        128,
        256,
        512,
    ), f"flat_work_group_size must be 128, 256, or 512, got {flat_work_group_size}"
    assert dtype_str in ("f16", "bf16"), "flex_flash_generic only supports f16 and bf16"
    assert BLOCK_N % 32 == 0
    assert BLOCK_N_OUT % BLOCK_N == 0

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    # The block-skip loop maps one BlockMask KV block to one kernel KV tile, so the
    # two block sizes must agree. Supporting a coarser BlockMask would mean an inner
    # loop over sub-tiles; a finer one cannot be expressed at all.
    if block_mask:
        if sparse_kv_block_size is None:
            sparse_kv_block_size = BLOCK_N_OUT
        elif sparse_kv_block_size != BLOCK_N_OUT:
            raise ValueError(
                f"sparse_kv_block_size ({sparse_kv_block_size}) must equal the kernel KV tile "
                f"BLOCK_N_OUT ({BLOCK_N_OUT}); build the BlockMask with "
                f"BLOCK_SIZE=({BLOCK_M}, {BLOCK_N_OUT})"
            )
        if ENABLE_DMA:
            # The DMA path prefetches tile 0 before the loop and tile n+1 from inside
            # it, both assuming the KV walk is contiguous. Block-skip breaks that
            # assumption, and re-deriving the prefetch from the index list means
            # reading one block ahead of the loop bound.
            raise NotImplementedError(
                "block_mask is not implemented on the DMA-to-LDS prefetch path "
                "(gfx950+, or FLYDSL_FLASH_ATTN_FUNC_ENABLE_DMA=1): its prefetch "
                "addresses assume a contiguous KV walk"
            )

    if num_aux_tensors:
        if aux_specs is None or len(aux_specs) != num_aux_tensors:
            raise ValueError(
                f"aux_specs must supply one (sb, sh, sq, skv) stride tuple per aux tensor "
                f"(num_aux_tensors={num_aux_tensors})"
            )
        aux_specs = [tuple(int(s) for s in spec) for spec in aux_specs]
        if any(len(spec) != 4 for spec in aux_specs):
            raise ValueError("each aux_specs entry must be a 4-tuple of element strides (sb, sh, sq, skv)")

    NUM_HEADS_Q = num_heads
    NUM_HEADS_KV = num_kv_heads
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV
    HEAD_DIM = head_dim
    CAUSAL = causal
    SCORE_MOD = score_mod
    MASK_MOD = mask_mod
    HAS_SCORE_MOD = score_mod is not None
    HAS_MASK_MOD = mask_mod is not None
    HAS_ANY_MOD = HAS_SCORE_MOD or HAS_MASK_MOD
    MOD_VEC = mod_vec_size
    NUM_AUX = num_aux_tensors
    AUX_SPECS = aux_specs or []
    RETURN_LSE = bool(return_lse)
    USE_BLOCK_MASK = bool(block_mask)
    STRIDE_TOKEN_Q = NUM_HEADS_Q * HEAD_DIM
    STRIDE_TOKEN_KV = NUM_HEADS_KV * HEAD_DIM

    # Mixed into generated symbol names so two builds differing only in a mod cannot
    # collide in the JIT cache (see the mod_key contract in the docstring).
    MOD_TAG = "nomod" if mod_key is None else str(mod_key).replace("-", "_")[:48]

    # Bank-conflict-free LDS strides.
    # K uses XOR swizzle (col ^ ((row & 7) << 4)) at 16-element granularity
    # instead of padding. This enables ds_read_b128 (stride is 256B-aligned).
    K_STRIDE = HEAD_DIM
    if USE_HW_TR:
        V_STRIDE = HEAD_DIM if ENABLE_DMA else HEAD_DIM + 4
    else:
        VT_STRIDE = BLOCK_N + 2
        V_STRIDE = VT_STRIDE

    # Vectorized cooperative load constants.
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
    assert HEAD_DIM % VEC_WIDTH == 0
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    assert BLOCK_SIZE % THREADS_PER_ROW_LOAD == 0
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        assert BLOCK_N % ROWS_PER_BATCH_LOAD == 0
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    # K/V circular buffers; defaults to 1/1, optional 3/3 with CK-like LDS sequence.
    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    if USE_HW_TR:
        LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    else:
        LDS_V_TILE_SIZE = HEAD_DIM * VT_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    # Distinct from the vendored kernel's symbol, and per-variant: both modules can be
    # imported into one process, and two flex builds differing only in a mod would
    # otherwise share an LDS global.
    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"flex_flash_generic_smem_{PATH_TAG}_{MOD_TAG}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * 2

    # All optional buffers are always in the signature and the host passes 1-element
    # placeholders for the disabled ones. Building a variadic signature would have to
    # fight flyc's annotation introspection, and the unused pointers cost a couple of
    # SGPRs: every read/write below is behind `const_expr`, so no code is emitted.
    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flex_flash_generic_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        KV_NUM_BLOCKS: fx.Tensor,
        KV_INDICES: fx.Tensor,
        AUX0: fx.Tensor,
        AUX1: fx.Tensor,
        seq_len: fx.Int32,
    ):
        elem_dtype = dtype_to_elem_type(dtype_str)
        elem_type = elem_dtype.ir_type
        compute_type = fx.Float32.ir_type
        k_ptr = _extract_aligned_pointer(K)
        v_ptr = _extract_aligned_pointer(V)

        # All FP operations use aggressive fast-math (no NaN/Inf checks, reassociation).
        # The unsafe_fp_math/fast_fp_math builder params control LLVM-level attributes only.
        fm_fast = fx.arith.FastMathFlags.fast
        v4f16_type = Vec.make_type(4, elem_dtype)
        v8f16_type = Vec.make_type(8, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        mfma_pack_type = v8f16_type if USE_K16 else v4f16_type
        MFMA_LANE_K = 8 if USE_K16 else 4

        def _mfma(mfma_fn, a, b, c):
            return mfma_fn(v16f32_type, [a, b, c])

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        def mfma_acc(a, b, c):
            if const_expr(dtype_str == "bf16"):
                if const_expr(USE_K16):
                    return _mfma(rocdl.mfma_f32_32x32x16_bf16, a, b, c)
                a = Vec(a).bitcast(fx.Int16)
                b = Vec(b).bitcast(fx.Int16)
                return _mfma(rocdl.mfma_f32_32x32x8bf16_1k, a, b, c)
            if const_expr(USE_K16):
                return _mfma(rocdl.mfma_f32_32x32x16_f16, a, b, c)
            return _mfma(rocdl.mfma_f32_32x32x8f16, a, b, c)

        seq_len_v = fx.Index(seq_len)

        # ---- LDS view ----
        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()

        # ---- Thread / block indices ----
        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)

        # ---- Wave decomposition ----
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32  # 0/1

        # ---- ds_read_b64_tr_b16 lane decomposition ----
        # Hardware does 4×4 transpose within blocks of 16 lanes.
        # tr_k_group selects which of 4 K-rows within the block,
        # tr_col_sub selects which 4-column sub-group within 16 columns.
        tr_k_group = (lane % 16) // 4  # 0..3: K-row offset within 4-row group
        tr_col_sub = lane % 4  # 0..3: 4-column sub-group
        tr_col_half = (lane % 32) // 16  # 0 or 1: first/second 16-column half

        # ---- ds_read_b64_tr_b16 helper ----

        def ds_read_tr_v4f16(lds_elem_idx):
            """Read v4f16 from LDS with hardware transpose.

            Within each block of 16 lanes, the hardware performs a 4×4
            transpose across 4 groups of 4 lanes.  After the transpose,
            result[lane, elem_e] = Input[source_lane, lane%4] where
            source_lane = e*4 + (lane%16)//4.  This naturally produces
            the MFMA A-operand layout when per-lane addresses point to
            the correct K-row and D-column sub-group.
            """
            byte_offset = lds_elem_idx * 2 + lds_kv_offset
            byte_i64 = fx.Int64(byte_offset)
            ptr = buffer_ops.create_llvm_ptr(byte_i64, address_space=3)
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        # ---- Wave offsets ----
        wave_q_offset = wave_id * ROWS_PER_WAVE

        # ---- Decompose block_id ----
        # Each block computes one Q head (per batch, per Q-tile).
        q_head_idx = block_id % NUM_HEADS_Q
        batch_q_tile_id = block_id // NUM_HEADS_Q
        num_q_tiles = (seq_len_v + BLOCK_M - 1) // BLOCK_M
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * BLOCK_M
        # GQA/MQA: every GQA_GROUP_SIZE consecutive Q heads share one KV head.
        # Use Python ternary (ast.IfExp) so FlyDSL's `if`-rewriter doesn't
        # turn this into a dynamic dispatch and lose `kv_head_idx`.
        kv_head_idx = q_head_idx if GQA_GROUP_SIZE == 1 else q_head_idx // GQA_GROUP_SIZE

        # ---- Cooperative load decomposition ----
        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        # ---- Helper: global flat indices ----
        # Q/O are laid out with NUM_HEADS_Q heads; K/V with NUM_HEADS_KV.
        def global_idx_q(token_idx, col):
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN_Q + q_head_idx * HEAD_DIM + col

        def global_idx_kv(token_idx, col):
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN_KV + kv_head_idx * HEAD_DIM + col

        def _kv_row_clamp(row_idx):
            # Non-DMA KV loads use raw pointers (no hardware bounds), so clamp the
            # global KV row to the last valid token; partial-tile lanes then read a
            # duplicated in-bounds row whose contribution the score-side causal /
            # padding mask discards. (The DMA path is bounded by num_records.)
            last = seq_len_v - fx.Index(1)
            return fx.Index(ArithValue(row_idx < seq_len_v).select(row_idx, last))

        def _load_global_half_vec(ptr, base_idx, vec_elems: int):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            return _pointer_load(Vec.make_type(vec_elems, elem_dtype), gep)

        def _store_global_half(ptr, base_idx, val):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            _pointer_store(val, gep)

        def load_global_f16x4(rsrc, base_idx):
            return _load_global_half_vec(rsrc, base_idx, 4)

        def load_global_mfma_pack(rsrc, base_idx):
            return _load_global_half_vec(rsrc, base_idx, MFMA_LANE_K)

        def load_global_f16xN(rsrc, base_idx):
            return _load_global_half_vec(rsrc, base_idx, VEC_WIDTH)

        def _bitcast_i32(value):
            return fx.Int32(ArithValue(value).bitcast(fx.Int32.ir_type))

        def _pack_bf16_pair(lo, hi, shift, mask):
            lo_i32 = _bitcast_i32(lo)
            hi_i32 = _bitcast_i32(hi)
            return (hi_i32 & mask) | lo_i32.shrui(shift)

        def bf16_trunc_pack_v4(f32_vals):
            """Pack f32 values into bf16 by keeping the upper 16 bits."""
            _c16 = fx.Int32(16)
            _cmask = fx.Int32(0xFFFF0000)
            packed = [
                _pack_bf16_pair(f32_vals[0], f32_vals[1], _c16, _cmask),
                _pack_bf16_pair(f32_vals[2], f32_vals[3], _c16, _cmask),
            ]
            return Vec.from_elements(packed, fx.Int32).bitcast(elem_dtype).ir_value()

        def bf16_trunc_pack_v8(f32_vals):
            """Pack 8 f32 values into v8bf16 via bitwise truncation (upper 16 bits)."""
            _c16 = fx.Int32(16)
            _cmask = fx.Int32(0xFFFF0000)
            pairs = []
            for j in range_constexpr(4):
                pairs.append(_pack_bf16_pair(f32_vals[j * 2], f32_vals[j * 2 + 1], _c16, _cmask))
            return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()

        def k_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(buf_id * LDS_K_TILE_SIZE)
            return buf_id * fx.Index(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id):
            return fx.Index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)

        # ---- K XOR swizzle: col ^ ((row & 7) << 4) at 16-element granularity ----
        def _k_swizzle(row_idx, col_idx):
            mask = (row_idx & fx.Index(0x7)) << fx.Index(4)
            return col_idx ^ mask

        # ---- Cooperative K load (row-major, XOR-swizzled) ----
        def coop_load_k(tile_start, buf_id=0):
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = _kv_row_clamp(tile_start + load_row_in_batch + row_offset)
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        swz_col = _k_swizzle(lds_row, load_col_base)
                        lds_idx = k_base + lds_row * K_STRIDE + swz_col
                        vec = load_global_f16xN(k_ptr, g_idx)
                        Vec(vec).store(lds_kv, [lds_idx])
                else:
                    g_idx = global_idx_kv(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = k_base + lds_row * K_STRIDE + swz_col
                    vec = load_global_f16xN(k_ptr, g_idx)
                    Vec(vec).store(lds_kv, [lds_idx])

        # ---- Cooperative V load ----
        def _v_store_row_major(v_base, lds_row, vec):
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base
            Vec(vec).store(lds_kv, [lds_idx])

        def _v_store_transposed(v_base, lds_row, vec):
            for _e in range_constexpr(VEC_WIDTH):
                elem = Vec(vec)[_e]
                vt_d = load_col_base + _e
                vt_idx = v_base + vt_d * VT_STRIDE + lds_row
                v1 = Vec.from_elements([elem], elem_dtype)
                v1.store(lds_kv, [vt_idx])

        _v_store_to_lds = _v_store_row_major if USE_HW_TR else _v_store_transposed

        def coop_load_v(tile_start, buf_id=0):
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = _kv_row_clamp(tile_start + load_row_in_batch + row_offset)
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        vec = load_global_f16xN(v_ptr, g_idx)
                        _v_store_to_lds(v_base, lds_row, vec)
                else:
                    g_idx = global_idx_kv(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    vec = load_global_f16xN(v_ptr, g_idx)
                    _v_store_to_lds(v_base, lds_row, vec)

        def coop_load_v_global(tile_start):
            """Issue global loads for V, return vectors (non-blocking)."""
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = _kv_row_clamp(tile_start + load_row_in_batch + row_offset)
                g_idx = global_idx_kv(row_idx, load_col_base)
                vecs.append(load_global_f16xN(v_ptr, g_idx))
            return vecs

        def coop_store_v_lds(vecs, buf_id=0):
            """Write previously-loaded V vectors to LDS."""
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        lds_row = load_row_in_batch + row_offset
                        _v_store_to_lds(v_base, lds_row, vecs[batch])
                else:
                    lds_row = load_row_in_batch + row_offset
                    _v_store_to_lds(v_base, lds_row, vecs[batch])

        # Per-batch num_records bound: rows >= seq_len read/write past this batch's
        # region, so OOB loads return 0 and OOB stores drop (arbitrary-seqlen safe;
        # aligned hot path unchanged). Same asm trick, used for K/V/Q loads + O-store.
        _kv_nrec_bytes = _raw((batch_idx + fx.Index(1)) * seq_len_v * fx.Index(STRIDE_TOKEN_KV * 2))
        _q_nrec_bytes = _raw((batch_idx + fx.Index(1)) * seq_len_v * fx.Index(STRIDE_TOKEN_Q * 2))
        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=False, num_records_bytes=_q_nrec_bytes)
        o_rsrc = buffer_ops.create_buffer_resource(O, max_size=False, num_records_bytes=_q_nrec_bytes)

        # ---- DMA loading for K (buffer_load_dwordx4 ... lds) ----
        if const_expr(ENABLE_DMA):
            k_rsrc = buffer_ops.create_buffer_resource(K, max_size=False, num_records_bytes=_kv_nrec_bytes)
            DMA_BYTES = 16  # buffer_load_dwordx4 = 16 bytes per lane
            DMA_BATCH_BYTES = BLOCK_SIZE * DMA_BYTES
            K_TILE_BYTES = BLOCK_N * K_STRIDE * 2
            NUM_DMA_K = K_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_K_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (HEAD_DIM * 2)
            lds_kv_base_idx = buffer_ops.extract_base_index(lds_kv, address_space=3)
            _dma_size = fx.Int32(DMA_BYTES)
            _dma_soff = fx.Int32(0)
            _dma_off = fx.Int32(0)
            _dma_aux = fx.Int32(1)

            def coop_dma_k(tile_start, buf_id=0):
                """Load K tile via DMA with XOR-swizzled global fetch."""
                if const_expr(isinstance(buf_id, int)):
                    k_lds_byte_base = lds_kv_base_idx + fx.Index(buf_id * LDS_K_TILE_SIZE * 2)
                else:
                    k_lds_byte_base = lds_kv_base_idx + buf_id * fx.Index(LDS_K_TILE_SIZE * 2)
                for d in range_constexpr(NUM_DMA_K):
                    lds_addr = (
                        k_lds_byte_base + wave_id * fx.Index(WARP_SIZE * DMA_BYTES) + fx.Index(d * DMA_BATCH_BYTES)
                    )
                    lds_i64 = fx.Int64(lds_addr)
                    lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, lds_i64)
                    lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)

                    row_in_tile = tid // LANES_PER_K_ROW + fx.Index(d * ROWS_PER_DMA_BATCH)
                    swiz_col_f16 = (tid % LANES_PER_K_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & fx.Index(0x7)) << fx.Index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = batch_idx * seq_len_v + tile_start + row_in_tile
                    global_byte = (
                        global_row * fx.Index(STRIDE_TOKEN_KV * 2) + kv_head_idx * fx.Index(HEAD_DIM * 2) + col_byte
                    )
                    voffset = fx.Int32(global_byte)

                    rocdl.raw_ptr_buffer_load_lds(
                        k_rsrc,
                        lds_ptr,
                        _dma_size,
                        voffset,
                        _dma_soff,
                        _dma_off,
                        _dma_aux,
                    )

        # ---- V XOR swizzle: col ^ ((row & 3) << 4) at 16-element granularity ----
        def _v_swizzle(row_idx, col_idx):
            mask = (row_idx & fx.Index(0x3)) << fx.Index(4)
            return col_idx ^ mask

        # ---- DMA loading for V (buffer_load_dwordx4 ... lds) ----
        if const_expr(ENABLE_DMA):
            v_rsrc = buffer_ops.create_buffer_resource(V, max_size=False, num_records_bytes=_kv_nrec_bytes)
            V_TILE_BYTES = BLOCK_N * V_STRIDE * 2
            NUM_DMA_V = V_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_V_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH_V = DMA_BATCH_BYTES // (HEAD_DIM * 2)

            def coop_dma_v(tile_start, buf_id=0):
                """Load V tile via DMA with XOR-swizzled global fetch."""
                v_lds_byte_base = lds_kv_base_idx + fx.Index((LDS_V_BASE + buf_id * LDS_V_TILE_SIZE) * 2)
                for d in range_constexpr(NUM_DMA_V):
                    lds_addr = (
                        v_lds_byte_base + wave_id * fx.Index(WARP_SIZE * DMA_BYTES) + fx.Index(d * DMA_BATCH_BYTES)
                    )
                    lds_i64 = fx.Int64(lds_addr)
                    lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, lds_i64)
                    lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)

                    row_in_tile = tid // LANES_PER_V_ROW + fx.Index(d * ROWS_PER_DMA_BATCH_V)
                    swiz_col_f16 = (tid % LANES_PER_V_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & fx.Index(0x3)) << fx.Index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = batch_idx * seq_len_v + tile_start + row_in_tile
                    global_byte = (
                        global_row * fx.Index(STRIDE_TOKEN_KV * 2) + kv_head_idx * fx.Index(HEAD_DIM * 2) + col_byte
                    )
                    voffset = fx.Int32(global_byte)

                    rocdl.raw_ptr_buffer_load_lds(
                        v_rsrc,
                        lds_ptr,
                        _dma_size,
                        voffset,
                        _dma_soff,
                        _dma_off,
                        _dma_aux,
                    )

        # ---- Preload Q^T B-operand packs once (register-resident) ----
        # B operand: j = lane_mod_32, k-subblock = lane_div_32*MFMA_LANE_K. Q is
        # num_records-bounded (q_rsrc) so OOB rows read 0 -- no q_in_bounds select.
        q_row = q_start + wave_q_offset + lane_mod_32
        q_row_i32 = fx.Int32(q_row)
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            g_idx = global_idx_q(q_row, q_col)
            q_b_packs.append(buffer_ops.buffer_load(q_rsrc, g_idx, vec_width=MFMA_LANE_K, dtype=elem_dtype))

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        # Scores reach the softmax raw (q·k), so the exp2 path folds sm_scale*log2(e)
        # into its fma. A score_mod must instead see the FlexAttention score domain
        # (q·k * sm_scale), so when one is active the scores are scaled by sm_scale at
        # the mod site and the softmax path only applies log2(e). This keeps the same
        # one-fma-per-element cost downstream instead of a scale/unscale round trip.
        # A mask_mod alone does not read the score, so it does not shift the domain.
        c_score_scale = fx.Float32(sm_scale)
        c_softmax_log2e = fx.Float32(_LOG2E) if HAS_SCORE_MOD else c_sm_scale_log2e
        # m_running is the max of whatever domain the scores are in, so recovering a
        # natural-log LSE needs the matching factor: 1.0 once score_mod pre-scaled
        # them, sm_scale otherwise.
        c_lse_m_scale = fx.Float32(1.0 if HAS_SCORE_MOD else sm_scale)
        c_ln2 = fx.Float32(1.0 / _LOG2E)
        # Finite floor for the running max whenever a mod is active (see init_args).
        # Small enough that log2(e)*floor stays well inside f32 -- a sentinel near
        # -FLT_MAX would overflow that product to -inf and reintroduce the NaN.
        c_neg_floor = fx.Float32(-1.0e30)
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)

        # ---- Captured aux tensors (read inside a mod at (b, h, q_idx, kv_idx)) ----
        # Each spec is a 4-tuple of element strides; 0 means "broadcast over this
        # axis", which is how a [H] ALiBi slope table or an [S] relative-position
        # vector is expressed without materialising a full [B,H,S,S] bias.
        def _make_aux_reader(_rsrc, _strides):
            _sb, _sh, _sq, _skv = _strides

            def _offset(coords):
                off = None
                for _stride, _coord in zip(_strides, coords):
                    if const_expr(_stride == 0):
                        continue
                    term = fx.Int32(_coord) if const_expr(_stride == 1) else fx.Int32(_coord) * fx.Int32(_stride)
                    off = term if off is None else off + term
                return fx.Int32(0) if off is None else off

            def _read(b, h, q_idx, kv_idx):
                return fx.Float32(buffer_ops.buffer_load(_rsrc, _raw(_offset((b, h, q_idx, kv_idx))), 1, dtype=None))

            def _read_vec(b, h, q_idx, kv_base, n):
                """n contiguous-in-kv values in one load.

                This is the reason mod_vec_size exists: a per-element reader emits n
                dword loads for what a bias tensor can serve as a single dwordx4. Only
                valid for a kv-contiguous tensor; anything else falls back to n scalar
                reads.
                """
                if const_expr(_skv != 1 or n not in (2, 4)):
                    return [_read(b, h, q_idx, kv_base + fx.Int32(i)) for i in range(n)]
                packed = buffer_ops.buffer_load(_rsrc, _raw(_offset((b, h, q_idx, kv_base))), vec_width=n, dtype=None)
                return [fx.Float32(Vec(packed)[i]) for i in range(n)]

            _read.vec = _read_vec
            return _read

        if const_expr(NUM_AUX > 0):
            _aux_bufs = [AUX0, AUX1][:NUM_AUX]
            _aux_readers = [
                _make_aux_reader(buffer_ops.create_buffer_resource(_buf, max_size=True), AUX_SPECS[_i])
                for _i, _buf in enumerate(_aux_bufs)
            ]
            _mod_kw = {"aux": _aux_readers}
        else:
            _mod_kw = {}

        # ---- Mod ABI adapters ----
        # MOD_VEC == 1 keeps the plain scalar signature; above that the mod is handed
        # lists of contiguous-in-kv values so it can do the arithmetic once per group.
        def _call_score_mod(scores, b, h, q_idx, kvs):
            if const_expr(MOD_VEC == 1):
                return [SCORE_MOD(scores[0], b, h, q_idx, kvs[0], **_mod_kw)]
            return list(SCORE_MOD(scores, b, h, q_idx, kvs, **_mod_kw))

        def _call_mask_mod(b, h, q_idx, kvs):
            if const_expr(MOD_VEC == 1):
                return [MASK_MOD(b, h, q_idx, kvs[0], **_mod_kw)]
            return list(MASK_MOD(b, h, q_idx, kvs, **_mod_kw))
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_32_i32 = fx.Int32(32)
        c4_i32 = fx.Int32(4)
        lane_i32 = fx.Int32(lane)
        lane_xor_32_i32 = lane_i32 ^ shuf_32_i32
        lane_xor_32_byte = lane_xor_32_i32 * c4_i32

        def reduction_peer(v_f32):
            if const_expr(REDUCE_MODE == "ds_bpermute"):
                v_i32 = fx.Int32(ArithValue(v_f32).bitcast(fx.Int32.ir_type))
                peer_i32 = rocdl.ds_bpermute(fx.Int32.ir_type, lane_xor_32_byte, v_i32)
                return fx.Float32(ArithValue(peer_i32).bitcast(compute_type))
            return fx.Float32(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        # ---- KV loop bounds ----
        # Dense: walk [0, kv_upper) in BLOCK_N_OUT steps, where causal stops at the
        # diagonal. BlockMask: walk this Q block's kv_indices list instead, so blocks
        # the mask proved empty are never loaded at all. That is where the sparsity win
        # comes from -- masking alone still pays for the tile.
        _q_end = q_start + BLOCK_M
        if const_expr(CAUSAL):
            kv_upper = fx.Index(ArithValue(_q_end < seq_len_v).select(_q_end, seq_len_v))
        else:
            kv_upper = seq_len_v

        if const_expr(USE_BLOCK_MASK):
            # kv_num_blocks [B, H_q, num_q_blocks] i32
            # kv_indices    [B, H_q, num_q_blocks, num_kv_blocks] i32
            # A BlockMask built with H=1 must be expanded by the host; broadcasting it
            # here would need a second stride set for no real gain.
            _nb_rsrc = buffer_ops.create_buffer_resource(KV_NUM_BLOCKS, max_size=True)
            _kvi_rsrc = buffer_ops.create_buffer_resource(KV_INDICES, max_size=True)
            _kv_blocks_total = (seq_len_v + fx.Index(BLOCK_N_OUT - 1)) // fx.Index(BLOCK_N_OUT)
            _qblk_lin = (batch_idx * fx.Index(NUM_HEADS_Q) + q_head_idx) * num_q_tiles + q_tile_idx
            _n_visit = fx.Index(
                fx.Int32(buffer_ops.buffer_load(_nb_rsrc, _raw(fx.Int32(_qblk_lin)), vec_width=1, dtype=T.i32))
            )
            _kvi_base = _qblk_lin * _kv_blocks_total
            _kv_lo, _kv_hi, _kv_step = fx.Index(0), _n_visit, 1
        else:
            _kv_lo, _kv_hi, _kv_step = 0, kv_upper, BLOCK_N_OUT

        # Loop-carried: [m_old, l_old, o_acc_chunks..., (buf_id if DMA dbuf)]
        _use_dma_dbuf = ENABLE_DMA and not ENABLE_PREFETCH_3BUF
        # A mod can leave a whole KV tile -inf for a given row -- a mask_mod that
        # rejects it, or (just as legally) a score_mod written as
        # `where(cond, score, -inf)`. Then m_running and the tile max are both -inf,
        # so m_running - m_new is (-inf) - (-inf) = NaN, which poisons l and o for
        # the rest of the loop. The dense causal path never sees this (kv_upper
        # stops at the diagonal) and neither does block-skip on its own: a visited
        # block need only hold a valid element in *some* row, not this one.
        #
        # Seeding the running max at a finite sentinel is enough, because it is an
        # invariant from then on: max(finite, -inf) is still finite. An empty tile
        # gives corr = exp2(0) = 1 against l = 0, masked elements give
        # p = exp2(-inf + BIG*scale) = 0, and once a real score appears
        # m_new >= row_max so the exponent stays <= 0. Costs nothing in the loop,
        # and the mod-free build keeps the upstream -inf seed exactly.
        init_args = [c_neg_floor if HAS_ANY_MOD else c_neg_inf, c_zero_f]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)
        if const_expr(_use_dma_dbuf):
            init_args.append(fx.Index(0))
            coop_dma_k(fx.Index(0), buf_id=0)

        loop_results = init_args
        for _kv_iv, inner_iter_args in range(_kv_lo, _kv_hi, _kv_step, init=init_args):
            if const_expr(USE_BLOCK_MASK):
                # Gather this iteration's KV tile from the index list. Everything
                # downstream (prefetch addresses, masks, the mod site) already derives
                # from kv_block_start, so it does not care that the value stopped being
                # the affine loop induction variable.
                kv_block_start = (
                    fx.Index(
                        fx.Int32(
                            buffer_ops.buffer_load(
                                _kvi_rsrc, _raw(fx.Int32(_kvi_base + _kv_iv)), vec_width=1, dtype=T.i32
                            )
                        )
                    )
                    * fx.Index(BLOCK_N_OUT)
                )
            else:
                kv_block_start = _kv_iv
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            _cur_buf_id = inner_iter_args[2 + D_CHUNKS] if _use_dma_dbuf else None
            preload_k_count = NUM_PREFETCH_K if NUM_PREFETCH_K < N_SUBTILES else N_SUBTILES

            if const_expr(ENABLE_PREFETCH_3BUF):
                for pre_k in range_constexpr(preload_k_count):
                    pre_k_slot = CK_LDS_SEQ[pre_k % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                    pre_k_start = kv_block_start + pre_k * BLOCK_N
                    if const_expr(ENABLE_DMA):
                        coop_dma_k(pre_k_start, pre_k_slot)
                    else:
                        coop_load_k(pre_k_start, pre_k_slot)
                if const_expr(ENABLE_DMA):
                    rocdl.s_waitcnt(0)
                else:
                    rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, 0)
                gpu.barrier()

            for kv_sub in range_constexpr(N_SUBTILES):
                kv_start = kv_block_start + kv_sub * BLOCK_N

                if const_expr(ENABLE_PREFETCH_3BUF):
                    k_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                elif const_expr(_use_dma_dbuf):
                    if const_expr(kv_sub % 2 == 0):
                        _k_buf_id = _cur_buf_id
                    else:
                        _k_buf_id = fx.Index(1) - _cur_buf_id
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    _next_k_buf_id = fx.Index(1) - _k_buf_id
                    if const_expr(kv_sub + 1 < N_SUBTILES):
                        coop_dma_k(
                            kv_block_start + (kv_sub + 1) * BLOCK_N,
                            _next_k_buf_id,
                        )
                    else:
                        _next_kv = kv_block_start + fx.Index(BLOCK_N_OUT)
                        _has_next = _next_kv < kv_upper
                        if _has_next:
                            coop_dma_k(_next_kv, _next_k_buf_id)
                    rocdl.sched_barrier(0)
                    k_base = k_buf_base(_k_buf_id)
                else:
                    k_slot = 0
                    coop_load_k(kv_start, k_slot)
                    gpu.barrier()
                if const_expr(not _use_dma_dbuf):
                    k_base = k_buf_base(k_slot)

                if const_expr(not USE_HW_TR or (not ENABLE_DMA and not ENABLE_PREFETCH_3BUF)):
                    _v_vecs_prefetch = coop_load_v_global(kv_start)

                # ==== GEMM1: bulk-read all K packs, then pipeline MFMAs ====
                k_hi_offset = K_SUB_N * K_STRIDE
                # XOR swizzle: col ^ ((row & 0x7) << 4) avoids LDS bank conflicts
                k_swz_mask = (lane_mod_32 & fx.Index(0x7)) << fx.Index(4)

                def _k_idx_lo(ks):
                    col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    return k_base + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)

                def _k_idx_hi(ks):
                    col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    return k_base + k_hi_offset + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)

                _QK_PREFETCH_DEPTH = 2
                k_packs_lo = [None] * K_STEPS_QK
                k_packs_hi = [None] * K_STEPS_QK
                for p in range_constexpr(_QK_PREFETCH_DEPTH):
                    k_packs_lo[p] = Vec.load(mfma_pack_type, lds_kv, [_k_idx_lo(p)])
                    k_packs_hi[p] = Vec.load(mfma_pack_type, lds_kv, [_k_idx_hi(p)])

                if const_expr(ENABLE_DMA and not ENABLE_PREFETCH_3BUF):
                    coop_dma_v(kv_start, 0)
                    rocdl.sched_barrier(0)

                s_acc_lo = c_zero_v16f32
                s_acc_hi = c_zero_v16f32
                for ks in range_constexpr(K_STEPS_QK):
                    s_acc_lo = mfma_acc(k_packs_lo[ks], q_b_packs[ks], s_acc_lo)
                    s_acc_hi = mfma_acc(k_packs_hi[ks], q_b_packs[ks], s_acc_hi)
                    if const_expr(ks + _QK_PREFETCH_DEPTH < K_STEPS_QK):
                        k_packs_lo[ks + _QK_PREFETCH_DEPTH] = Vec.load(
                            mfma_pack_type, lds_kv, [_k_idx_lo(ks + _QK_PREFETCH_DEPTH)]
                        )
                        k_packs_hi[ks + _QK_PREFETCH_DEPTH] = Vec.load(
                            mfma_pack_type, lds_kv, [_k_idx_hi(ks + _QK_PREFETCH_DEPTH)]
                        )

                # ==== Online softmax over 64 KV positions ====
                s_raw_lo = []
                s_raw_hi = []
                for r in range_constexpr(16):
                    s_raw_lo.append(Vec(s_acc_lo)[r])
                    s_raw_hi.append(Vec(s_acc_hi)[r])

                # ==== FlexAttention mod site (score_mod then mask_mod) ====
                # Ahead of the kernel's causal/padding masks, because FlexAttention
                # evaluates score_mod on unmasked scores and only then applies
                # mask_mod; a non-additive mod (e.g. soft-cap tanh) would otherwise
                # turn a masked -inf back into a finite value (cap*tanh(-inf/cap) =
                # -cap, which resurrects the position with weight ~exp(-cap)).
                #
                # Element -> coordinate map (identical in flex_flash_950.py):
                #   q_idx  = q_start + wave_id*ROWS_PER_WAVE + lane%32   (one row per lane)
                #   kv_idx = kv_start + (lane//32)*4 + ((r//4)*8 + r%4) + K_SUB_N*n_strip
                # with n_strip = 0 for s_raw_lo and 1 for s_raw_hi. Writing r = 4g + j
                # gives offset 8g + j, so each group of 4 consecutive r is contiguous in
                # kv_idx and groups are 8 apart -- which is why MOD_VEC caps at 4.
                #
                # mask_mod is applied on every visited block, including ones the
                # BlockMask marks fully-unmasked. Skipping it there needs the full and
                # partial block lists walked by *separate* loops with separately
                # emitted bodies (as FlexAttention does), because the predicate is
                # runtime data and FlyDSL's dynamic `if` propagates rebound locals, not
                # writes into these score lists. Deferred: the block-skip below is the
                # larger win, and this only costs instructions on blocks we visit.
                if const_expr(HAS_ANY_MOD):
                    _mod_b = fx.Int32(batch_idx)
                    _mod_h = fx.Int32(q_head_idx)
                    _mod_kv0 = fx.Int32(kv_start) + fx.Int32(lane_div_32) * fx.Int32(4)
                    for _strip, _s_list in ((0, s_raw_lo), (1, s_raw_hi)):
                        for _g in range_constexpr(4):
                            _g_kv0 = _mod_kv0 + fx.Int32(_strip * K_SUB_N + _g * 8)
                            for _sub in range_constexpr(4 // MOD_VEC):
                                _rs = [_g * 4 + _sub * MOD_VEC + _t for _t in range(MOD_VEC)]
                                _kvs = [_g_kv0 + fx.Int32(_sub * MOD_VEC + _t) for _t in range(MOD_VEC)]
                                if const_expr(HAS_SCORE_MOD):
                                    _sc = _call_score_mod(
                                        [_fmul(_s_list[_r], c_score_scale) for _r in _rs],
                                        _mod_b,
                                        _mod_h,
                                        q_row_i32,
                                        _kvs,
                                    )
                                else:
                                    _sc = [_s_list[_r] for _r in _rs]
                                if const_expr(HAS_MASK_MOD):
                                    _keep = _call_mask_mod(_mod_b, _mod_h, q_row_i32, _kvs)
                                    _sc = [ArithValue(_kp).select(_sv, c_neg_inf) for _kp, _sv in zip(_keep, _sc)]
                                for _i, _r in enumerate(_rs):
                                    _s_list[_r] = _sc[_i]

                if const_expr(CAUSAL):
                    kv_start_i32 = fx.Int32(kv_start)
                    lane_div_32_i32 = fx.Int32(lane_div_32)
                    q_start_i32 = fx.Int32(q_start)
                    max_kv_col_i32 = kv_start_i32 + fx.Int32(BLOCK_N - 1)
                    tile_needs_mask = max_kv_col_i32 > q_start_i32
                    s_raw_lo_0 = s_raw_lo[0]
                    s_raw_lo_1 = s_raw_lo[1]
                    s_raw_lo_2 = s_raw_lo[2]
                    s_raw_lo_3 = s_raw_lo[3]
                    s_raw_lo_4 = s_raw_lo[4]
                    s_raw_lo_5 = s_raw_lo[5]
                    s_raw_lo_6 = s_raw_lo[6]
                    s_raw_lo_7 = s_raw_lo[7]
                    s_raw_lo_8 = s_raw_lo[8]
                    s_raw_lo_9 = s_raw_lo[9]
                    s_raw_lo_10 = s_raw_lo[10]
                    s_raw_lo_11 = s_raw_lo[11]
                    s_raw_lo_12 = s_raw_lo[12]
                    s_raw_lo_13 = s_raw_lo[13]
                    s_raw_lo_14 = s_raw_lo[14]
                    s_raw_lo_15 = s_raw_lo[15]
                    s_raw_hi_0 = s_raw_hi[0]
                    s_raw_hi_1 = s_raw_hi[1]
                    s_raw_hi_2 = s_raw_hi[2]
                    s_raw_hi_3 = s_raw_hi[3]
                    s_raw_hi_4 = s_raw_hi[4]
                    s_raw_hi_5 = s_raw_hi[5]
                    s_raw_hi_6 = s_raw_hi[6]
                    s_raw_hi_7 = s_raw_hi[7]
                    s_raw_hi_8 = s_raw_hi[8]
                    s_raw_hi_9 = s_raw_hi[9]
                    s_raw_hi_10 = s_raw_hi[10]
                    s_raw_hi_11 = s_raw_hi[11]
                    s_raw_hi_12 = s_raw_hi[12]
                    s_raw_hi_13 = s_raw_hi[13]
                    s_raw_hi_14 = s_raw_hi[14]
                    s_raw_hi_15 = s_raw_hi[15]

                    if tile_needs_mask:
                        lane_off_i32 = lane_div_32_i32 * fx.Int32(4)
                        kv_col_lo_0 = kv_start_i32 + lane_off_i32 + fx.Int32(0)
                        s_raw_lo_0 = ArithValue(kv_col_lo_0 > q_row_i32).select(c_neg_inf, s_raw_lo_0)
                        s_raw_hi_0 = ArithValue(kv_col_lo_0 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_0
                        )
                        kv_col_lo_1 = kv_start_i32 + lane_off_i32 + fx.Int32(1)
                        s_raw_lo_1 = ArithValue(kv_col_lo_1 > q_row_i32).select(c_neg_inf, s_raw_lo_1)
                        s_raw_hi_1 = ArithValue(kv_col_lo_1 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_1
                        )
                        kv_col_lo_2 = kv_start_i32 + lane_off_i32 + fx.Int32(2)
                        s_raw_lo_2 = ArithValue(kv_col_lo_2 > q_row_i32).select(c_neg_inf, s_raw_lo_2)
                        s_raw_hi_2 = ArithValue(kv_col_lo_2 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_2
                        )
                        kv_col_lo_3 = kv_start_i32 + lane_off_i32 + fx.Int32(3)
                        s_raw_lo_3 = ArithValue(kv_col_lo_3 > q_row_i32).select(c_neg_inf, s_raw_lo_3)
                        s_raw_hi_3 = ArithValue(kv_col_lo_3 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_3
                        )
                        kv_col_lo_4 = kv_start_i32 + lane_off_i32 + fx.Int32(8)
                        s_raw_lo_4 = ArithValue(kv_col_lo_4 > q_row_i32).select(c_neg_inf, s_raw_lo_4)
                        s_raw_hi_4 = ArithValue(kv_col_lo_4 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_4
                        )
                        kv_col_lo_5 = kv_start_i32 + lane_off_i32 + fx.Int32(9)
                        s_raw_lo_5 = ArithValue(kv_col_lo_5 > q_row_i32).select(c_neg_inf, s_raw_lo_5)
                        s_raw_hi_5 = ArithValue(kv_col_lo_5 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_5
                        )
                        kv_col_lo_6 = kv_start_i32 + lane_off_i32 + fx.Int32(10)
                        s_raw_lo_6 = ArithValue(kv_col_lo_6 > q_row_i32).select(c_neg_inf, s_raw_lo_6)
                        s_raw_hi_6 = ArithValue(kv_col_lo_6 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_6
                        )
                        kv_col_lo_7 = kv_start_i32 + lane_off_i32 + fx.Int32(11)
                        s_raw_lo_7 = ArithValue(kv_col_lo_7 > q_row_i32).select(c_neg_inf, s_raw_lo_7)
                        s_raw_hi_7 = ArithValue(kv_col_lo_7 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_7
                        )
                        kv_col_lo_8 = kv_start_i32 + lane_off_i32 + fx.Int32(16)
                        s_raw_lo_8 = ArithValue(kv_col_lo_8 > q_row_i32).select(c_neg_inf, s_raw_lo_8)
                        s_raw_hi_8 = ArithValue(kv_col_lo_8 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_8
                        )
                        kv_col_lo_9 = kv_start_i32 + lane_off_i32 + fx.Int32(17)
                        s_raw_lo_9 = ArithValue(kv_col_lo_9 > q_row_i32).select(c_neg_inf, s_raw_lo_9)
                        s_raw_hi_9 = ArithValue(kv_col_lo_9 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_9
                        )
                        kv_col_lo_10 = kv_start_i32 + lane_off_i32 + fx.Int32(18)
                        s_raw_lo_10 = ArithValue(kv_col_lo_10 > q_row_i32).select(c_neg_inf, s_raw_lo_10)
                        s_raw_hi_10 = ArithValue(kv_col_lo_10 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_10
                        )
                        kv_col_lo_11 = kv_start_i32 + lane_off_i32 + fx.Int32(19)
                        s_raw_lo_11 = ArithValue(kv_col_lo_11 > q_row_i32).select(c_neg_inf, s_raw_lo_11)
                        s_raw_hi_11 = ArithValue(kv_col_lo_11 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_11
                        )
                        kv_col_lo_12 = kv_start_i32 + lane_off_i32 + fx.Int32(24)
                        s_raw_lo_12 = ArithValue(kv_col_lo_12 > q_row_i32).select(c_neg_inf, s_raw_lo_12)
                        s_raw_hi_12 = ArithValue(kv_col_lo_12 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_12
                        )
                        kv_col_lo_13 = kv_start_i32 + lane_off_i32 + fx.Int32(25)
                        s_raw_lo_13 = ArithValue(kv_col_lo_13 > q_row_i32).select(c_neg_inf, s_raw_lo_13)
                        s_raw_hi_13 = ArithValue(kv_col_lo_13 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_13
                        )
                        kv_col_lo_14 = kv_start_i32 + lane_off_i32 + fx.Int32(26)
                        s_raw_lo_14 = ArithValue(kv_col_lo_14 > q_row_i32).select(c_neg_inf, s_raw_lo_14)
                        s_raw_hi_14 = ArithValue(kv_col_lo_14 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_14
                        )
                        kv_col_lo_15 = kv_start_i32 + lane_off_i32 + fx.Int32(27)
                        s_raw_lo_15 = ArithValue(kv_col_lo_15 > q_row_i32).select(c_neg_inf, s_raw_lo_15)
                        s_raw_hi_15 = ArithValue(kv_col_lo_15 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_15
                        )

                    s_raw_lo = [
                        s_raw_lo_0,
                        s_raw_lo_1,
                        s_raw_lo_2,
                        s_raw_lo_3,
                        s_raw_lo_4,
                        s_raw_lo_5,
                        s_raw_lo_6,
                        s_raw_lo_7,
                        s_raw_lo_8,
                        s_raw_lo_9,
                        s_raw_lo_10,
                        s_raw_lo_11,
                        s_raw_lo_12,
                        s_raw_lo_13,
                        s_raw_lo_14,
                        s_raw_lo_15,
                    ]
                    s_raw_hi = [
                        s_raw_hi_0,
                        s_raw_hi_1,
                        s_raw_hi_2,
                        s_raw_hi_3,
                        s_raw_hi_4,
                        s_raw_hi_5,
                        s_raw_hi_6,
                        s_raw_hi_7,
                        s_raw_hi_8,
                        s_raw_hi_9,
                        s_raw_hi_10,
                        s_raw_hi_11,
                        s_raw_hi_12,
                        s_raw_hi_13,
                        s_raw_hi_14,
                        s_raw_hi_15,
                    ]
                else:
                    # Non-causal KV padding mask: keys with absolute column >= seq_len
                    # -> -inf, so OOB KV (0 or duplicated row) doesn't leak into softmax.
                    # Col layout (mirrors causal): lo = kv_start + lane_div_32*4 +
                    # ((r//4)*8 + r%4); hi = +K_SUB_N.
                    kv_start_i32 = fx.Int32(kv_start)
                    lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(4)
                    seq_len_i32 = fx.Int32(seq_len_v)
                    for r in range_constexpr(16):
                        _off = (r // 4) * 8 + (r % 4)
                        kv_col = kv_start_i32 + lane_off_i32 + fx.Int32(_off)
                        s_raw_lo[r] = ArithValue(kv_col >= seq_len_i32).select(c_neg_inf, s_raw_lo[r])
                        s_raw_hi[r] = ArithValue(kv_col + fx.Int32(K_SUB_N) >= seq_len_i32).select(
                            c_neg_inf, s_raw_hi[r]
                        )

                local_max = s_raw_lo[0]
                for r in range_constexpr(15):
                    local_max = _fmax(local_max, s_raw_lo[r + 1])
                for r in range_constexpr(16):
                    local_max = _fmax(local_max, s_raw_hi[r])
                peer_max = reduction_peer(local_max)
                row_max = _fmax(local_max, peer_max)
                # m_running is seeded finite under a mod (see init_args), so this max
                # cannot produce -inf and needs no floor here.
                m_new_raw = _fmax(m_running, row_max)

                diff_m_raw = _fsub(m_running, m_new_raw)
                diff_m_scaled = _fmul(diff_m_raw, c_softmax_log2e)
                corr = ArithValue(diff_m_scaled).exp2(fastmath=fm_fast)

                scaled_max = _fmul(c_softmax_log2e, m_new_raw)
                neg_scaled_max = _fsub(c_zero_f, scaled_max)

                p_vals_lo = []
                p_vals_hi = []
                local_sum = c_zero_f
                for r in range_constexpr(16):
                    diff_lo = fmath.fma(s_raw_lo[r], c_softmax_log2e, neg_scaled_max, fastmath=fm_fast)
                    p_lo = ArithValue(diff_lo).exp2(fastmath=fm_fast)
                    p_vals_lo.append(p_lo)
                    local_sum = _fadd(local_sum, p_lo)
                for r in range_constexpr(16):
                    diff_hi = fmath.fma(s_raw_hi[r], c_softmax_log2e, neg_scaled_max, fastmath=fm_fast)
                    p_hi = ArithValue(diff_hi).exp2(fastmath=fm_fast)
                    p_vals_hi.append(p_hi)
                    local_sum = _fadd(local_sum, p_hi)

                peer_sum = reduction_peer(local_sum)
                tile_sum = _fadd(local_sum, peer_sum)
                l_corr = _fmul(corr, l_running)
                l_new = _fadd(l_corr, tile_sum)

                # ==== Rescale O accumulators ====
                corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(16)
                if const_expr(not USE_HW_TR):
                    o_accs[0] = _fmul(Vec(o_accs[0]), corr_vec)
                else:
                    for dc in range_constexpr(D_CHUNKS):
                        o_accs[dc] = _fmul(Vec(o_accs[dc]), corr_vec)

                if const_expr(ENABLE_PREFETCH_3BUF and (kv_sub + preload_k_count) < N_SUBTILES):
                    next_k_sub = kv_sub + preload_k_count
                    next_k_start = kv_block_start + next_k_sub * BLOCK_N
                    next_k_slot = CK_LDS_SEQ[next_k_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                    if const_expr(ENABLE_DMA):
                        coop_dma_k(next_k_start, next_k_slot)
                    else:
                        coop_load_k(next_k_start, next_k_slot)

                if const_expr(ENABLE_PREFETCH_3BUF):
                    v_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_V
                    v_base = v_buf_base(v_slot)
                    coop_load_v(kv_start, v_slot)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()
                elif const_expr(ENABLE_DMA):
                    v_base = v_buf_base(0)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                else:
                    v_slot = 0
                    v_base = v_buf_base(v_slot)
                    _waitcnt_vm_n(0)
                    coop_store_v_lds(_v_vecs_prefetch, v_slot)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()

                # ==== Build P packs for lo and hi halves ====
                if const_expr(dtype_str == "bf16" and not USE_K16):
                    p_packs_lo = []
                    p_packs_hi = []
                    for pks in range_constexpr(PV_K_STEPS):
                        p_base = pks * 4
                        p_packs_lo.append(bf16_trunc_pack_v4(p_vals_lo[p_base : p_base + 4]))
                        p_packs_hi.append(bf16_trunc_pack_v4(p_vals_hi[p_base : p_base + 4]))
                elif const_expr(dtype_str == "bf16" and USE_K16):
                    p_packs_lo = []
                    p_packs_hi = []
                    for pks in range_constexpr(PV_K_STEPS):
                        p_base = pks * 8
                        p_packs_lo.append(bf16_trunc_pack_v8(p_vals_lo[p_base : p_base + 8]))
                        p_packs_hi.append(bf16_trunc_pack_v8(p_vals_hi[p_base : p_base + 8]))
                else:
                    p_f16_lo = []
                    p_f16_hi = []
                    for r in range_constexpr(16):
                        p_f16_lo.append(fx.Float32(p_vals_lo[r]).to(elem_dtype))
                        p_f16_hi.append(fx.Float32(p_vals_hi[r]).to(elem_dtype))

                    if const_expr(USE_K16):
                        p_packs_lo = []
                        p_packs_hi = []
                        for pks in range_constexpr(PV_K_STEPS):
                            p_base = pks * 8
                            p_packs_lo.append(
                                Vec.from_elements(
                                    [
                                        p_f16_lo[p_base + 0],
                                        p_f16_lo[p_base + 1],
                                        p_f16_lo[p_base + 2],
                                        p_f16_lo[p_base + 3],
                                        p_f16_lo[p_base + 4],
                                        p_f16_lo[p_base + 5],
                                        p_f16_lo[p_base + 6],
                                        p_f16_lo[p_base + 7],
                                    ],
                                    elem_dtype,
                                ).ir_value()
                            )
                            p_packs_hi.append(
                                Vec.from_elements(
                                    [
                                        p_f16_hi[p_base + 0],
                                        p_f16_hi[p_base + 1],
                                        p_f16_hi[p_base + 2],
                                        p_f16_hi[p_base + 3],
                                        p_f16_hi[p_base + 4],
                                        p_f16_hi[p_base + 5],
                                        p_f16_hi[p_base + 6],
                                        p_f16_hi[p_base + 7],
                                    ],
                                    elem_dtype,
                                ).ir_value()
                            )
                    else:
                        p_packs_lo = []
                        p_packs_hi = []
                        for pks in range_constexpr(PV_K_STEPS):
                            p_base = pks * 4
                            p_packs_lo.append(
                                Vec.from_elements(
                                    [
                                        p_f16_lo[p_base],
                                        p_f16_lo[p_base + 1],
                                        p_f16_lo[p_base + 2],
                                        p_f16_lo[p_base + 3],
                                    ],
                                    elem_dtype,
                                ).ir_value()
                            )
                            p_packs_hi.append(
                                Vec.from_elements(
                                    [
                                        p_f16_hi[p_base],
                                        p_f16_hi[p_base + 1],
                                        p_f16_hi[p_base + 2],
                                        p_f16_hi[p_base + 3],
                                    ],
                                    elem_dtype,
                                ).ir_value()
                            )

                # Build flat (dc, pks) schedule for interleaved GEMM2.
                _steps = [(dc, pks) for dc in range(D_CHUNKS) for pks in range(PV_K_STEPS)]
                TOTAL_PV = len(_steps)

                def _read_v_pack(step_idx):
                    dc, pks = _steps[step_idx]
                    if const_expr(USE_HW_TR):
                        d_col = fx.Index(dc * D_CHUNK) + tr_col_half * 16 + tr_col_sub * 4
                        k_row = fx.Index(pks * PV_K_STEP) + lane_div_32 * 4 + tr_k_group
                        _d_col_eff = _v_swizzle(k_row, d_col) if ENABLE_DMA else d_col
                        lds_lo = v_base + k_row * V_STRIDE + _d_col_eff
                        lds_hi = lds_lo + fx.Index(K_SUB_N * V_STRIDE)
                        if const_expr(USE_K16):
                            vl_a = ds_read_tr_v4f16(lds_lo)
                            vl_b = ds_read_tr_v4f16(lds_lo + fx.Index(8 * V_STRIDE))
                            vl = Vec(vl_a).shuffle(Vec(vl_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                            vh_a = ds_read_tr_v4f16(lds_hi)
                            vh_b = ds_read_tr_v4f16(lds_hi + fx.Index(8 * V_STRIDE))
                            vh = Vec(vh_a).shuffle(Vec(vh_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                        else:
                            vl = ds_read_tr_v4f16(lds_lo)
                            vh = ds_read_tr_v4f16(lds_hi)
                    else:
                        d_pos = fx.Index(dc * D_CHUNK) + lane_mod_32
                        k_base = fx.Index(pks * PV_K_STEP) + lane_div_32 * 4
                        v_lo_idx = v_base + d_pos * VT_STRIDE + k_base
                        v_hi_idx = v_lo_idx + fx.Index(K_SUB_N)
                        vl = Vec.load(v4f16_type, lds_kv, [v_lo_idx])
                        vh = Vec.load(v4f16_type, lds_kv, [v_hi_idx])
                    return vl, vh

                # Pre-read V for the first step.
                v_lo_cur, v_hi_cur = _read_v_pack(0)

                # ==== GEMM2: O += V^T_lo @ P_lo + V^T_hi @ P_hi ====
                for si in range_constexpr(TOTAL_PV):
                    dc, pks = _steps[si]
                    if const_expr(si + 1 < TOTAL_PV):
                        v_lo_nxt, v_hi_nxt = _read_v_pack(si + 1)
                    o_accs[dc] = mfma_acc(v_lo_cur, p_packs_lo[pks], o_accs[dc])
                    o_accs[dc] = mfma_acc(v_hi_cur, p_packs_hi[pks], o_accs[dc])
                    if const_expr(not USE_HW_TR and dc == 0 and pks < D_CHUNKS - 1):
                        o_accs[pks + 1] = Vec(o_accs[pks + 1]) * corr_vec
                    if const_expr(si + 1 < TOTAL_PV):
                        v_lo_cur = v_lo_nxt
                        v_hi_cur = v_hi_nxt

                m_running = m_new_raw
                l_running = l_new

            _yield_args = [m_running, l_running] + o_accs
            if const_expr(_use_dma_dbuf):
                if const_expr(N_SUBTILES % 2 == 1):
                    _yield_args.append(fx.Index(1) - _cur_buf_id)
                else:
                    _yield_args.append(_cur_buf_id)
            loop_results = yield _yield_args

        # ---- Normalize and store O (128-bit buffer_store_dwordx4) ----
        # gfx950: pack 4 f32 -> 2 bf16 dwords (cvt_pk_bf16_f32), permlane32_swap fuses
        # each lane's 4 cols with its half-wave partner's -> 8 cols/store. O is
        # num_records-bounded (o_rsrc) -> partial-q-tile OOB rows drop.
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        # ---- LSE (log-sum-exp), the quantity upstream computes and throws away ----
        # FlexAttention's template signature takes LOGSUMEXP as an output, return_lse
        # exposes it, and the backward pass needs it to rebuild P without storing it.
        # Derivation: the kernel forms p = 2^(s*σ*L - m*σ*L) = e^((s-m)*σ) with
        # σ = sm_scale and L = log2(e), so l = Σ e^((s-m)σ) and
        #   ln Σ e^(s*σ) = m*σ + ln(l),
        # where σ folds into c_lse_m_scale (1.0 if score_mod already pre-scaled the
        # scores) and ln(l) = log2(l) * ln2. A fully-masked row leaves m = -inf, l = 0
        # and yields -inf, which is what flex_attention returns.
        if const_expr(RETURN_LSE):
            m_final = loop_results[0]
            lse_val = _fadd(
                _fmul(m_final, c_lse_m_scale),
                _fmul(fx.Float32(fmath.log2(l_final, fastmath=fm_fast)), c_ln2),
            )
            lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
            # Layout [B, H_q, S]. Lanes 0-31 and 32-63 hold the same 32 rows (the row
            # reduction is a half-wave xor shuffle), so only the low half stores.
            lse_idx = (batch_idx * fx.Index(NUM_HEADS_Q) + q_head_idx) * seq_len_v + q_row
            if ArithValue(lane_div_32 == fx.Index(0)):
                if ArithValue(q_row < seq_len_v):
                    buffer_ops.buffer_store(_raw(fx.Float32(lse_val)), lse_rsrc, _raw(fx.Int32(lse_idx)))

        inv_l = rocdl.rcp(T.f32, l_final)
        if const_expr(HAS_ANY_MOD):
            # A row left entirely masked -- by a mask_mod, or by a score_mod that
            # returns -inf -- has l = 0, and o * rcp(0) = 0 * inf = NaN.
            # flex_attention returns 0 for such a row.
            inv_l = ArithValue(l_final == c_zero_f).select(c_zero_f, fx.Float32(inv_l))
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(16)
        v_o = [Vec(o_finals[dc]) * inv_l_vec for dc in range_constexpr(D_CHUNKS)]

        if const_expr(USE_PERMLANE_OSTORE):
            # gfx950: 128-bit permlane-fused store (cvt_pk_bf16_f32 + permlane32_swap).
            pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
            is_hi_half = ArithValue(lane_div_32 != fx.Index(0))

            def _o_pack_2dw(dc, store_group):
                # 4 f32 outputs -> 2 packed-16bit dwords (lo = cols 0,1; hi = cols 2,3).
                r_base = store_group * 4
                if const_expr(dtype_str == "bf16"):
                    lo = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base], Vec(v_o[dc])[r_base + 1])
                    hi = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base + 2], Vec(v_o[dc])[r_base + 3])
                    return lo, hi
                o_f16 = [fx.Float32(Vec(v_o[dc])[r_base + i]).to(elem_dtype) for i in range_constexpr(4)]
                pack = Vec.from_elements(o_f16, elem_dtype).bitcast(fx.Int32)
                return _raw(pack[0]), _raw(pack[1])

            def _swap_halves(dw):
                # permlane32_swap(a,b) -> (a.lo|b.lo, a.hi|b.hi); with a=b=dw the
                # partner dword dw[lane^32] is result[1] on low lanes, [0] on high.
                swapped = rocdl.permlane32_swap(pair_i32_ty, _raw(dw), _raw(dw), False, False)
                lo_res = llvm.extractvalue(T.i32, swapped, [0])
                hi_res = llvm.extractvalue(T.i32, swapped, [1])
                return is_hi_half.select(lo_res, hi_res)

            for dc in range_constexpr(D_CHUNKS):
                for g in range_constexpr(2):
                    d0_a, d1_a = _o_pack_2dw(dc, 2 * g)
                    d0_b, d1_b = _o_pack_2dw(dc, 2 * g + 1)
                    # low lanes: own group-2g cols 0-3 ++ partner's cols 4-7;
                    # high lanes: partner's group-(2g+1) cols 0-3 ++ own cols 4-7.
                    y0_a, y1_a = _swap_halves(d0_a), _swap_halves(d1_a)
                    y0_b, y1_b = _swap_halves(d0_b), _swap_halves(d1_b)
                    w0 = is_hi_half.select(y0_b, _raw(d0_a))
                    w1 = is_hi_half.select(y1_b, _raw(d1_a))
                    w2 = is_hi_half.select(_raw(d0_b), y0_a)
                    w3 = is_hi_half.select(_raw(d1_b), y1_a)
                    o_pack = Vec.from_elements([fx.Int32(w0), fx.Int32(w1), fx.Int32(w2), fx.Int32(w3)], fx.Int32)
                    d_col = fx.Index(dc * D_CHUNK) + (fx.Index(2 * g) + lane_div_32) * fx.Index(8)
                    o_global = global_idx_q(q_row, d_col)
                    buffer_ops.buffer_store(o_pack, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True)
        else:
            # gfx942 fallback (no permlane32_swap / cvt_pk_bf16_f32): each lane stores
            # its 16 cols as 4 dwordx2 groups via .to(elem_dtype); col map d_col =
            # dc*D_CHUNK + lane_div_32*4 + 8*grp + r. num_records bound drops OOB rows.
            for dc in range_constexpr(D_CHUNKS):
                for grp in range_constexpr(4):
                    r0 = grp * 4
                    o_f16 = [fx.Float32(Vec(v_o[dc])[r0 + i]).to(elem_dtype) for i in range_constexpr(4)]
                    pack = Vec.from_elements(o_f16, elem_dtype).bitcast(fx.Int32)
                    o2 = Vec.from_elements([_raw(pack[0]), _raw(pack[1])], fx.Int32)
                    d_col = fx.Index(dc * D_CHUNK) + lane_div_32 * fx.Index(4) + fx.Index(grp * 8)
                    o_global = global_idx_q(q_row, d_col)
                    buffer_ops.buffer_store(o2, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True)

    @flyc.jit
    def launch_flex_flash_generic(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        KV_NUM_BLOCKS: fx.Tensor,
        KV_INDICES: fx.Tensor,
        AUX0: fx.Tensor,
        AUX1: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
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
        flex_flash_generic_kernel(
            Q,
            K,
            V,
            O,
            LSE,
            KV_NUM_BLOCKS,
            KV_INDICES,
            AUX0,
            AUX1,
            seq_len,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": (
                    f"{int(flat_work_group_size)},{int(flat_work_group_size)}"
                    if const_expr(flat_work_group_size is not None)
                    else None
                ),
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    # Best MI355X FMHA numbers so far were measured with ROCm/llvm-project
    # `felix/tune_fmha` at c8cf6da4367c010c7cbbb7789a9c4349e7407619.
    # Other LLVM revisions can compile/run this kernel, but usually leave a
    # few percent of peak throughput on the table.
    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return launch_flex_flash_generic(*args, **kwargs)

    def _compile(Q, K, V, O, LSE, KVNB, KVI, AUX0, AUX1, batch_size, seq_len, stream=None):  # noqa: E741
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return flyc.compile(
                launch_flex_flash_generic,
                Q,
                K,
                V,
                O,
                LSE,
                KVNB,
                KVI,
                AUX0,
                AUX1,
                batch_size,
                seq_len,
                fx.Stream(stream),
            )

    _launch.compile = _compile

    # Set on the outermost wrapper: the host side reads these to size the LSE buffer
    # and to build a BlockMask whose block size matches the kernel's KV tile.
    _guarded = _guard_seqlen(_launch)
    _guarded.kv_block_size = BLOCK_N_OUT
    _guarded.q_block_size = BLOCK_M
    _guarded.returns_lse = RETURN_LSE
    _guarded.num_aux_tensors = NUM_AUX
    _guarded.aux_specs = list(AUX_SPECS)
    _guarded.uses_block_mask = USE_BLOCK_MASK
    return _guarded

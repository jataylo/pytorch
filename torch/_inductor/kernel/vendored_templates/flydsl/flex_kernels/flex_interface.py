# SPDX-License-Identifier: Apache-2.0
"""Host-side entry point for the flex flash kernels.

The kernels carry every optional buffer in their signature and rely on the host to pass
1-element placeholders for the disabled ones (see the note at the kernel definition), so
callers should go through `flex_flash_attn` rather than invoking a launcher directly.

Also converts a `torch.nn.attention.flex_attention.BlockMask` into the two flat i32
arrays the block-skip loop walks.
"""

from __future__ import annotations

import math

import torch

_DUMMY_CACHE: dict[tuple, torch.Tensor] = {}


def _dummy(device, dtype=torch.float32):
    """1-element placeholder for a disabled optional buffer."""
    key = (str(device), dtype)
    if key not in _DUMMY_CACHE:
        _DUMMY_CACHE[key] = torch.zeros(1, device=device, dtype=dtype)
    return _DUMMY_CACHE[key]


def block_mask_tensors(mask_fn, batch, num_heads, seq_len, q_block, kv_block, device):
    """Build the (kv_num_blocks, kv_indices) pair the kernel's block-skip loop reads.

    Uses `create_block_mask` and then takes the **union** of the partial and
    fully-unmasked block lists, because this kernel visits both with `mask_mod` applied
    (it does not yet emit a separate body for full blocks). Returns i32 tensors shaped
    `[B, H, n_q_blocks]` and `[B, H, n_q_blocks, n_kv_blocks]`, expanded over heads: a
    head-broadcast BlockMask (H=1) is materialised here rather than adding a second
    stride set to the kernel.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    n_q = math.ceil(seq_len / q_block)
    n_kv = math.ceil(seq_len / kv_block)

    bm = create_block_mask(
        mask_fn, batch, num_heads, seq_len, seq_len, device=str(device), BLOCK_SIZE=(q_block, kv_block)
    )

    occ = torch.zeros(batch, num_heads, n_q, n_kv, dtype=torch.bool, device=device)

    def _mark(nums, idxs):
        if nums is None or idxs is None:
            return
        nums = nums.expand(batch, num_heads, n_q)
        idxs = idxs.expand(batch, num_heads, n_q, idxs.shape[-1])
        ar = torch.arange(idxs.shape[-1], device=device).view(1, 1, 1, -1)
        valid = ar < nums.unsqueeze(-1)
        # Out-of-place clamp: `idxs` is a view of the caller's BlockMask, and the
        # expand above can make it stride-0, where an in-place write is both a
        # mutation of their tensor and an aliased-write error. The clamp only tames
        # the padding entries past kv_num_blocks, which `valid` discards anyway.
        safe_idxs = idxs.long().clamp(0, n_kv - 1)
        occ.logical_or_(torch.zeros_like(occ).scatter_(-1, safe_idxs, valid))

    _mark(bm.kv_num_blocks, bm.kv_indices)
    _mark(getattr(bm, "full_kv_num_blocks", None), getattr(bm, "full_kv_indices", None))

    counts = occ.sum(-1).to(torch.int32)
    # Stable argsort on the negated occupancy lists visited blocks first, in increasing
    # block order, which is the layout the loop expects.
    order = torch.argsort((~occ).to(torch.int8), dim=-1, stable=True).to(torch.int32)
    return counts.contiguous(), order.contiguous()


_I32_MAX = 2**31 - 1


def _check_aux(aux, specs, B, H, S):
    """Reject aux_specs that would address outside the tensor.

    The kernel reads aux through a `max_size` buffer descriptor with an i32 element
    offset, so a stride tuple that does not match the tensor's real layout reads
    whatever is next in memory -- plausible-looking numbers, no fault. There is enough
    information here to name the mistake instead, so do it on the host.
    """
    for i, (t, spec) in enumerate(zip(aux, specs)):
        if t.dtype != torch.float32:
            raise ValueError(f"aux[{i}] must be float32 (the kernel reads it as f32), got {t.dtype}")
        if not t.is_contiguous():
            raise ValueError(f"aux[{i}] must be contiguous; its aux_specs strides describe the flat layout")
        sb, sh, sq, skv = spec
        max_off = sb * (B - 1) + sh * (H - 1) + sq * (S - 1) + skv * (S - 1)
        if max_off >= t.numel():
            raise ValueError(
                f"aux[{i}] aux_specs {spec} addresses element {max_off} at "
                f"(b,h,q,kv)=({B - 1},{H - 1},{S - 1},{S - 1}) but the tensor holds only "
                f"{t.numel()}; a [B,H,S,S] bias wants (H*S*S, S*S, S, 1) and 0 means broadcast"
            )
        if max_off > _I32_MAX:
            raise ValueError(
                f"aux[{i}] needs element offsets up to {max_off}, past the kernel's i32 "
                f"addressing limit of {_I32_MAX}; slice the tensor or use broadcast strides"
            )


def prepare(launcher, q, k, v, kv_num_blocks=None, kv_indices=None, aux=None, out=None, lse=None):
    """Bind buffers once and return `(run, out, lse)` where `run()` only launches.

    Benchmarks must use this: flattening, `contiguous()` and the output/LSE allocations
    are tens of microseconds, which is the same order as the kernel itself at small
    shapes and would otherwise be attributed to the mod.

    `out` and `lse` may be caller-allocated, which is what the Inductor path needs: the
    compiler has already decided where the results live. Both are written through
    `reshape(-1)`, so both must be contiguous.
    """
    B, S, H, _ = q.shape
    device = q.device
    if out is None:
        out = torch.empty_like(q)
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous; the kernel writes it as a flat buffer")

    if not launcher.returns_lse:
        if lse is not None:
            raise ValueError("this launcher was built with return_lse=False; it will not write lse")
        lse = _dummy(device)
    elif lse is None:
        lse = torch.empty((B, H, S), device=device, dtype=torch.float32)
    elif lse.shape != (B, H, S) or lse.dtype != torch.float32 or not lse.is_contiguous():
        raise ValueError(
            f"lse must be a contiguous float32 tensor of shape {(B, H, S)}, got "
            f"{tuple(lse.shape)} {lse.dtype} contiguous={lse.is_contiguous()}"
        )

    if launcher.uses_block_mask:
        if kv_num_blocks is None or kv_indices is None:
            raise ValueError("this launcher was built with block_mask=True; pass kv_num_blocks and kv_indices")
        nb, kvi = kv_num_blocks.reshape(-1), kv_indices.reshape(-1)
    else:
        nb = kvi = _dummy(device, torch.int32)

    aux = list(aux or [])
    if len(aux) != launcher.num_aux_tensors:
        raise ValueError(f"launcher expects {launcher.num_aux_tensors} aux tensors, got {len(aux)}")
    _check_aux(aux, getattr(launcher, "aux_specs", []), B, H, S)
    aux_args = [t.reshape(-1) for t in aux] + [_dummy(device)] * (2 - len(aux))

    args = (
        q.contiguous().reshape(-1),
        k.contiguous().reshape(-1),
        v.contiguous().reshape(-1),
        out.reshape(-1),
        lse.reshape(-1),
        nb,
        kvi,
        aux_args[0],
        aux_args[1],
        B,
        S,
    )
    stream = torch.cuda.current_stream(device)

    def run():
        launcher(*args, stream=stream)

    return run, out, lse


def flex_flash_attn(launcher, q, k, v, kv_num_blocks=None, kv_indices=None, aux=None, out=None):
    """Run a flex flash kernel once.

    q/k/v are BSHD. Returns `out` or `(out, lse)` when the launcher was built with
    `return_lse=True`; LSE is `[B, H, S]` f32 in natural log.
    """
    run, out, lse = prepare(launcher, q, k, v, kv_num_blocks, kv_indices, aux, out)
    run()
    return (out, lse) if launcher.returns_lse else out


def flex_flash_attn_bhsd(
    launcher, q, k, v, *, out, lse=None, kv_num_blocks=None, kv_indices=None, aux=None
):
    """Run a flex flash kernel on **BHSD** tensors, writing `out` and `lse` in place.

    This is the entry point the Inductor template calls, and it is the layout adapter
    between two fixed conventions: `torch.nn.attention.flex_attention` hands Inductor
    `[B, H, S, D]`, while these kernels index `[B, S, H, D]`. Everything else here
    (`prepare`, `flex_flash_attn`) is BSHD, so the name says which one this is.

    **The transposes cost three copies of q/k/v and usually one of the output.** The
    kernel's coordinate map is BSHD throughout, so this cannot be fixed by passing
    strides -- it needs the kernel's index math changed, which is perf work rather than
    integration work. At the shapes where FlexAttention is interesting the copies are
    small next to the attention itself, but they are not free and they are the first
    thing to remove when this path is benchmarked.

    `out` is written in place. If its `transpose(1, 2)` is already contiguous -- an
    output whose memory order is BSHD -- the kernel writes straight into it; otherwise a
    BSHD scratch buffer is used and copied back.
    """
    if q.ndim != 4:
        raise ValueError(f"expected BHSD tensors, got shape {tuple(q.shape)}")

    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()

    out_bshd_view = out.transpose(1, 2)
    writes_out_directly = out_bshd_view.is_contiguous()
    # Sized from the output rather than from q: v_head_dim need not equal qk_head_dim.
    out_buf = (
        out_bshd_view
        if writes_out_directly
        else torch.empty(
            out_bshd_view.shape, dtype=out.dtype, device=out.device
        )
    )

    writes_lse_directly = lse is None or lse.is_contiguous()

    run, out_buf, lse_buf = prepare(
        launcher,
        q_bshd,
        k_bshd,
        v_bshd,
        kv_num_blocks,
        kv_indices,
        aux,
        out=out_buf,
        lse=lse if writes_lse_directly else None,
    )
    run()

    if not writes_out_directly:
        out.copy_(out_buf.transpose(1, 2))
    if lse is not None and not writes_lse_directly:
        lse.copy_(lse_buf)
    return out, lse

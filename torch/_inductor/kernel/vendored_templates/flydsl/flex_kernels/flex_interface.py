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
from torch.utils.weak import WeakTensorKeyDictionary

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

    return _occupancy_to_lists(occ)


def _occupancy_to_lists(occ):
    """Turn a `[B, H, n_q, n_kv]` bool occupancy into the `(counts, indices)` i32 pair.

    A stable argsort on the negated occupancy lists visited blocks first, in increasing
    block order, which is the layout the loop expects. A prefix-sum compaction computes
    the same thing in more arithmetic but more kernels, and measured slower here: the
    occupancy is a few thousand bytes, so launch count is what costs, not work.
    """
    counts = occ.sum(-1).to(torch.int32)
    order = torch.argsort((~occ).to(torch.int8), dim=-1, stable=True).to(torch.int32)
    return counts.contiguous(), order.contiguous()


def _rescale_blocks(occ, dim, src_block, dst_block, n_dst):
    """Re-express `occ` along `dim` from `src_block`-sized blocks to `dst_block`-sized ones.

    Splitting (`dst < src`) repeats each block over the finer tiles it covers; merging
    (`dst > src`) takes `any` over the coarser tile's constituents. Both are exact given
    the divisibility the caller checked: a tile is visited iff it overlaps a visited
    block, and `mask_mod` still runs per element on everything visited.
    """
    if dst_block < src_block:
        occ = occ.repeat_interleave(src_block // dst_block, dim=dim)
    elif dst_block > src_block:
        merge = dst_block // src_block
        pad = (-occ.shape[dim]) % merge
        if pad:
            shape = list(occ.shape)
            shape[dim] = pad
            occ = torch.cat([occ, occ.new_zeros(shape)], dim=dim)
        occ = occ.unflatten(dim, (-1, merge)).any(dim + 1)

    n = occ.shape[dim]
    if n > n_dst:
        occ = occ.narrow(dim, 0, n_dst)
    elif n < n_dst:
        shape = list(occ.shape)
        shape[dim] = n_dst - n
        occ = torch.cat([occ, occ.new_zeros(shape)], dim=dim)
    return occ


def _regrid_uncached(
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    *,
    batch,
    num_heads,
    sparse_q_block_size,
    sparse_kv_block_size,
    q_block_size,
    kv_block_size,
    num_q_tiles,
    num_kv_tiles,
    walk="kv",
):
    """Re-express a FlexAttention BlockMask on this kernel's own tile grid.

    ``walk`` picks which axis the resulting list runs along, because the three kernels
    that want one do not all walk the same axis. ``"kv"`` gives ``[B, H, n_q]`` counts
    and ``[B, H, n_q, n_kv]`` indices -- a list of KV tiles per Q tile, which is what the
    forward and the ``dq`` backward walk. ``"q"`` gives the transpose, a list of Q tiles
    per KV tile, which is what the ``dk``/``dv`` backward walks.

    Both come from the same occupancy, transposed, rather than from FlexAttention's own
    ``q_indices``. That is deliberate: its q-side lists are on its own sparse grid and
    would need this same conversion anyway, and deriving both orientations from one
    occupancy is what guarantees the two backward kernels agree about which blocks exist.
    Two kernels disagreeing there would drop gradient contributions on one side only.

    FlexAttention supplies two disjoint block lists -- partial blocks, which still need
    `mask_mod` per element, and fully-unmasked ones -- on its own
    `(sparse_q_block_size, sparse_kv_block_size)` grid. This kernel walks a *single*
    list and applies `mask_mod` to every block it visits, on its own
    `(BLOCK_M, BLOCK_N_OUT)` grid. So the two lists are unioned and the grid converted:
    a split along KV, where the kernel's 64-wide tile is finer than FlexAttention's
    128 default, and usually a merge along Q, where `BLOCK_M` is 256 for the
    `num_heads >= 32` builds.

    Visiting the union rather than the partial list alone is what keeps this correct:
    the kernel has no separate body for fully-unmasked blocks, so a full block that
    went unvisited would simply be dropped from the softmax.

    Every op here is small, so the whole conversion is launch-bound at a roughly fixed
    cost per call regardless of sequence length; `regrid_block_mask` memoises it.
    """
    device = kv_indices.device
    n_q_mask = kv_indices.shape[-2]
    n_kv_mask = kv_indices.shape[-1]

    # Both lists are marked by one scatter over their concatenation, rather than a
    # scatter per list: at this size the cost is per-launch, not per-element.
    def _broadcast(nums, idxs):
        # A head- or batch-broadcast BlockMask is materialised here; broadcasting inside
        # the kernel would need a second stride set for no real gain.
        nums = nums.expand(batch, num_heads, n_q_mask)
        idxs = idxs.expand(batch, num_heads, n_q_mask, idxs.shape[-1])
        ar = torch.arange(idxs.shape[-1], device=device).view(1, 1, 1, -1)
        # Out-of-place clamp: `idxs` is a view of the caller's BlockMask and the expand
        # above can make it stride-0, where an in-place write is both a mutation of
        # their tensor and an aliased-write error. The clamp only tames the padding
        # entries past `nums`, which `valid` discards anyway.
        return idxs.long().clamp(0, n_kv_mask - 1), ar < nums.unsqueeze(-1)

    safe, valid = _broadcast(kv_num_blocks, kv_indices)
    if full_kv_num_blocks is not None and full_kv_indices is not None:
        full_safe, full_valid = _broadcast(full_kv_num_blocks, full_kv_indices)
        safe = torch.cat([safe, full_safe], dim=-1)
        valid = torch.cat([valid, full_valid], dim=-1)

    # `amax` rather than a plain scatter_ because the destinations are not unique: the
    # entries past `nums` are padding that still holds real block ids, so a padding
    # entry of one list can land on a block the other list visits. A plain scatter_
    # would write whichever of True/False got there last -- unspecified -- and could
    # drop a visited block. amax makes any order give the union. (uint8: reduce
    # scatters are not implemented for bool.)
    marked = torch.zeros(batch, num_heads, n_q_mask, n_kv_mask, dtype=torch.uint8, device=device)
    marked.scatter_reduce_(-1, safe, valid.to(torch.uint8), reduce="amax")
    occ = marked.bool()

    occ = _rescale_blocks(occ, 3, sparse_kv_block_size, kv_block_size, num_kv_tiles)
    occ = _rescale_blocks(occ, 2, sparse_q_block_size, q_block_size, num_q_tiles)
    if walk == "q":
        occ = occ.transpose(-1, -2)
    elif walk != "kv":
        raise ValueError(f"walk must be 'kv' or 'q', got {walk!r}")
    return _occupancy_to_lists(occ)


# kv_indices -> {grid -> (tensors, versions, lists)}
_REGRID_CACHE = WeakTensorKeyDictionary()


def regrid_block_mask(kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, **grid):
    """`_regrid_uncached`, memoised on the identity and version of the input tensors.

    The conversion is launch-bound at a fixed cost per call, which is a large fraction of
    a short-sequence kernel, while a BlockMask is typically built once and reused for
    every step of a run. So the result is cached against the exact tensors it came from.

    The cache is keyed on `kv_indices` in a weak dictionary, so an entry dies with the
    BlockMask that produced it. Reuse requires that all four tensors be the *same objects*
    (compared by identity, not contents) at the same `_version`, so an in-place edit or a
    freshly built mask misses and recomputes. Distinct masks that happen to hold equal
    contents also miss; that costs time, never correctness.

    One entry *per grid*, because a single BlockMask is regridded to more than one: the
    forward's tiles, and the two backward kernels' opposite-axis walks. Folding the grid
    into a single-slot stamp instead makes the three evict each other on every step and
    turns the memo into an unconditional recompute -- which is invisible in the answers and
    only shows up as a lost win. The entry count is bounded by the number of kernels, and
    a stale version replaces its own grid's entry rather than adding one.
    """
    tensors = (kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices)
    grid_key = tuple(sorted(grid.items()))
    versions = tuple(None if t is None else t._version for t in tensors)

    by_grid = _REGRID_CACHE.get(kv_indices)
    if by_grid is None:
        by_grid = {}
        _REGRID_CACHE[kv_indices] = by_grid

    hit = by_grid.get(grid_key)
    if hit is not None:
        cached_tensors, cached_versions, result = hit
        if cached_versions == versions and all(
            a is b for a, b in zip(cached_tensors, tensors)
        ):
            return result

    result = _regrid_uncached(*tensors, **grid)
    # Holding the inputs strongly is what lets the check above use identity: a recycled
    # id() cannot masquerade as a live tensor. They are siblings of the weak key, so this
    # extends no lifetime that matters, and they are a few KB.
    by_grid[grid_key] = (tensors, versions, result)
    return result


_I32_MAX = 2**31 - 1


def _check_aux(aux, specs, B, H, S_q, S_kv):
    """Reject aux_specs that would address outside the tensor.

    The kernel reads aux at an i32 element offset built from these strides, so a tuple
    that does not match the tensor's real layout reads the wrong element of it --
    plausible-looking numbers, no fault. There is enough information here to name the
    mistake instead, so do it on the host.

    This bounds the *logical* extent. The descriptor is separately bounded by the tensor's
    size, which is what keeps the score tile's padding lanes from reading off the end; see
    Note [aux reads run past the logical extent].
    """
    for i, (t, spec) in enumerate(zip(aux, specs)):
        if t.dtype != torch.float32:
            raise ValueError(f"aux[{i}] must be float32 (the kernel reads it as f32), got {t.dtype}")
        if not t.is_contiguous():
            raise ValueError(f"aux[{i}] must be contiguous; its aux_specs strides describe the flat layout")
        sb, sh, sq, skv = spec
        max_off = sb * (B - 1) + sh * (H - 1) + sq * (S_q - 1) + skv * (S_kv - 1)
        if max_off >= t.numel():
            raise ValueError(
                f"aux[{i}] aux_specs {spec} addresses element {max_off} at "
                f"(b,h,q,kv)=({B - 1},{H - 1},{S_q - 1},{S_kv - 1}) but the tensor holds only "
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
    bhsd = getattr(launcher, "layout", "bshd") == "bhsd"
    S, S_KV = _seq_lens(q, k, v, axis=2 if bhsd else 1)
    if bhsd:
        B, H, _, _ = q.shape
    else:
        B, _, H, _ = q.shape
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
    _check_aux(aux, getattr(launcher, "aux_specs", []), B, H, S, S_KV)
    # The kernel signature carries every slot whether or not the mods use it, so pad the
    # tail with a 1-element dummy rather than shortening the argument list.
    max_aux = getattr(launcher, "max_aux_tensors", 2)
    aux_args = [t.reshape(-1) for t in aux] + [_dummy(device)] * (max_aux - len(aux))

    args = (
        q.contiguous().reshape(-1),
        k.contiguous().reshape(-1),
        v.contiguous().reshape(-1),
        out.reshape(-1),
        lse.reshape(-1),
        nb,
        kvi,
        *aux_args,
        B,
        S,
        S_KV,
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


def _seq_lens(q, k, v, axis):
    """``(seq_len_q, seq_len_kv)``, requiring only that K and V agree with each other.

    Q and KV may differ -- that is cross attention, and every kernel here takes the two
    extents separately. K and V must match, because they are indexed by the same column.
    """
    seq_q, seq_kv = q.shape[axis], k.shape[axis]
    if v.shape[axis] != seq_kv:
        raise ValueError(
            f"key and value must share a sequence length, got {seq_kv} and {v.shape[axis]}"
        )
    return seq_q, seq_kv


def _layout_view(launcher, tensor):
    """A BHSD tensor as the launcher's layout wants to see it.

    Both layouts are the same four axes in a different order, so this is a view either
    way. It exists so that the `contiguous()` the kernel argument needs is a no-op on
    whichever layout the caller's memory is already in: a BSHD launcher fed q/k/v that
    came out of a projection reshaped to ``[B, S, H, D]`` copies nothing, where a BHSD
    one would copy every one of them.
    """
    return tensor if getattr(launcher, "layout", "bshd") == "bhsd" else tensor.transpose(1, 2)


def _bwd_block_lists(
    launcher, block_mask, *, batch, num_heads, seq_len_q, seq_len_kv, walk, device
):
    """The (counts, indices) pair a backward kernel walks, or 1-element dummies.

    ``block_mask`` is FlexAttention's raw tuple
    ``(kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices,
    sparse_q_block_size, sparse_kv_block_size)``. It is regridded onto *this* launcher's
    tiles, which differ between the two backward kernels and from the forward's, so each
    gets its own conversion (memoised, so a reused BlockMask pays once).
    """
    if not getattr(launcher, "use_block_mask", False):
        return [_dummy(device, dtype=torch.int32)] * 2
    if block_mask is None:
        raise ValueError("this launcher was built with block_mask=True but none was passed")

    kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, sparse_q, sparse_kv = (
        block_mask
    )
    q_block = launcher.q_block_size
    kv_block = launcher.kv_block_size
    counts, indices = regrid_block_mask(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        batch=batch,
        num_heads=num_heads,
        sparse_q_block_size=sparse_q,
        sparse_kv_block_size=sparse_kv,
        q_block_size=q_block,
        kv_block_size=kv_block,
        num_q_tiles=-(-seq_len_q // q_block),
        num_kv_tiles=-(-seq_len_kv // kv_block),
        walk=walk,
    )
    return [counts.reshape(-1), indices.reshape(-1)]


def flex_flash_bwd_dq_bhsd(
    launcher, q, k, v, do, lse, delta, *, dq, aux=None, block_mask=None
):
    """Run the ``dq`` backward kernel on **BHSD** tensors, writing ``dq`` in place.

    ``lse`` must be the natural-log LSE our forward writes, and ``delta`` the
    ``rowsum(do * o)`` the lowering computes; both are ``[B, H_q, S]`` f32 in either
    layout, since the kernel indexes them ``(b, h, s)`` regardless.

    Everything else is passed flat, so it must be dense in the launcher's layout. Feeding
    a launcher built for the layout the caller's memory is already in is what keeps that
    free -- otherwise this copies q, k, v and do, and dq on the way back.
    """
    if q.ndim != 4:
        raise ValueError(f"expected BHSD tensors, got shape {tuple(q.shape)}")

    S, S_KV = _seq_lens(q, k, v, axis=2)
    B, H, _, _ = q.shape
    for name, t in (("lse", lse), ("delta", delta)):
        if t.shape != (B, H, S) or t.dtype != torch.float32:
            raise ValueError(
                f"{name} must be a float32 tensor of shape {(B, H, S)}, got {tuple(t.shape)} {t.dtype}"
            )

    dq_view = _layout_view(launcher, dq)
    direct = dq_view.is_contiguous()
    buf = dq_view if direct else torch.empty(dq_view.shape, dtype=dq.dtype, device=dq.device)
    args = (
        _layout_view(launcher, q).contiguous().reshape(-1),
        _layout_view(launcher, k).contiguous().reshape(-1),
        _layout_view(launcher, v).contiguous().reshape(-1),
        _layout_view(launcher, do).contiguous().reshape(-1),
        lse.contiguous().reshape(-1),
        delta.contiguous().reshape(-1),
        buf.reshape(-1),
        *_bwd_aux_args(launcher, aux, B, H, S, S_KV, q.device),
        # dq owns its Q rows and walks KV, so it wants a KV list per Q tile.
        *_bwd_block_lists(
            launcher,
            block_mask,
            batch=B,
            num_heads=H,
            seq_len_q=S,
            seq_len_kv=S_KV,
            walk="kv",
            device=q.device,
        ),
        B,
        S,
        S_KV,
    )
    launcher(*args, stream=torch.cuda.current_stream(q.device))
    if not direct:
        dq_view.copy_(buf)
    return dq


def _bwd_aux_args(launcher, aux, B, H, S_q, S_kv, device):
    """Validate the captured tensors a backward kernel's mods read, and pad the slots.

    A kernel signature is fixed-arity, so the unused slots take a 1-element dummy -- the
    same arrangement the forward uses, and the reason `MAX_AUX_TENSORS` is a constant
    rather than per-build.
    """
    aux = list(aux or [])
    expected = getattr(launcher, "num_aux_tensors", 0)
    if len(aux) != expected:
        raise ValueError(f"launcher expects {expected} aux tensors, got {len(aux)}")
    _check_aux(aux, getattr(launcher, "aux_specs", []), B, H, S_q, S_kv)
    max_aux = getattr(launcher, "max_aux_tensors", 4)
    return [t.contiguous().reshape(-1) for t in aux] + [_dummy(device)] * (max_aux - len(aux))


def flex_flash_bwd_dkdv_bhsd(
    launcher, q, k, v, do, lse, delta, *, dk, dv, aux=None, block_mask=None
):
    """Run the ``dk``/``dv`` backward kernel on **BHSD** tensors, writing both in place.

    Same contract as ``flex_flash_bwd_dq_bhsd``: natural-log ``lse``, ``delta`` from the
    lowering, everything else flat and therefore dense in the launcher's layout.
    ``dk``/``dv`` are shaped by the *KV* head count; the kernel sums the GQA group
    internally, so no post-pass is needed.
    """
    if q.ndim != 4:
        raise ValueError(f"expected BHSD tensors, got shape {tuple(q.shape)}")

    S, S_KV = _seq_lens(q, k, v, axis=2)
    B, H, _, _ = q.shape
    for name, t in (("lse", lse), ("delta", delta)):
        if t.shape != (B, H, S) or t.dtype != torch.float32:
            raise ValueError(
                f"{name} must be a float32 tensor of shape {(B, H, S)}, got {tuple(t.shape)} {t.dtype}"
            )

    dk_view, dv_view = _layout_view(launcher, dk), _layout_view(launcher, dv)
    dk_direct, dv_direct = dk_view.is_contiguous(), dv_view.is_contiguous()
    dk_buf = dk_view if dk_direct else torch.empty(dk_view.shape, dtype=dk.dtype, device=dk.device)
    dv_buf = dv_view if dv_direct else torch.empty(dv_view.shape, dtype=dv.dtype, device=dv.device)
    args = (
        _layout_view(launcher, q).contiguous().reshape(-1),
        _layout_view(launcher, k).contiguous().reshape(-1),
        _layout_view(launcher, v).contiguous().reshape(-1),
        _layout_view(launcher, do).contiguous().reshape(-1),
        lse.contiguous().reshape(-1),
        delta.contiguous().reshape(-1),
        dk_buf.reshape(-1),
        dv_buf.reshape(-1),
        *_bwd_aux_args(launcher, aux, B, H, S, S_KV, q.device),
        # This kernel owns its KV rows and walks Q, so it wants the transposed list: a
        # Q list per KV tile, indexed by *Q* head.
        *_bwd_block_lists(
            launcher,
            block_mask,
            batch=B,
            num_heads=H,
            seq_len_q=S,
            seq_len_kv=S_KV,
            walk="q",
            device=q.device,
        ),
        B,
        S,
        S_KV,
    )
    launcher(*args, stream=torch.cuda.current_stream(q.device))
    if not dk_direct:
        dk_view.copy_(dk_buf)
    if not dv_direct:
        dv_view.copy_(dv_buf)
    return dk, dv


def flex_flash_attn_bhsd(
    launcher, q, k, v, *, out, lse=None, kv_num_blocks=None, kv_indices=None, aux=None
):
    """Run a flex flash kernel on **BHSD** tensors, writing `out` and `lse` in place.

    This is the entry point the Inductor template calls: `torch.nn.attention.flex_attention`
    hands Inductor `[B, H, S, D]`. Everything else here (`prepare`, `flex_flash_attn`)
    takes whatever layout its launcher was built for, so the name says which one this is.

    A launcher built with `layout="bhsd"` is fed directly. A `"bshd"` one is fed the
    transposed views, which is *also* free when the caller's BHSD tensors are BSHD-dense --
    the common case for q/k/v that came out of a projection reshaped to `[B, S, H, D]`,
    where `transpose(1, 2)` hands back exactly the contiguous memory underneath. So the
    caller should build the launcher for the layout its tensors are actually in; get it
    wrong and this costs three copies of q/k/v and usually one of the output, measured at
    574 us against a 461 us kernel at 4x16x4096x128. Inductor picks with `_kernel_layout`.

    `out` is written in place: straight into it when it can be, otherwise through a scratch
    buffer in the launcher's layout that is copied back.
    """
    if q.ndim != 4:
        raise ValueError(f"expected BHSD tensors, got shape {tuple(q.shape)}")

    # `prepare` calls `contiguous()` itself, so passing the launcher's view of each input
    # is all it takes: a no-op when the memory is already dense that way round, and the
    # copy it used to always make when it is not.
    out_view = _layout_view(launcher, out)
    direct = out_view.is_contiguous()
    # Sized from the output rather than from q: v_head_dim need not equal qk_head_dim.
    out_buf = (
        out_view
        if direct
        else torch.empty(out_view.shape, dtype=out.dtype, device=out.device)
    )
    # LSE is [B, H, S] in both layouts -- the kernel indexes it (b, h, s) either way.
    lse_direct = lse is None or lse.is_contiguous()

    run, out_buf, lse_buf = prepare(
        launcher,
        _layout_view(launcher, q),
        _layout_view(launcher, k),
        _layout_view(launcher, v),
        kv_num_blocks,
        kv_indices,
        aux,
        out=out_buf,
        lse=lse if lse_direct else None,
    )
    run()

    if not direct:
        out_view.copy_(out_buf)
    if lse is not None and not lse_direct:
        lse.copy_(lse_buf)
    return out, lse

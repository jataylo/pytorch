# SPDX-License-Identifier: Apache-2.0
# local: true
"""What each GPU architecture gives the FlyDSL flex kernels.

One table, read from both sides of the backend: Inductor lowering asks whether it can
serve a graph on this device at all, and the kernel builders ask which instructions to
select. Before this, both sides pattern-matched the arch name at the point of use, which
went wrong in two ways worth naming because they are the reason this file exists.

The first is that a name match cannot express "I have not been told about you". The
forward's DMA-to-LDS prefetch was selected by ``not gpu_arch.startswith("gfx942")``, so
every arch that was not gfx942 -- including RDNA parts with no ``buffer_load_dwordx4_lds``
at all -- claimed the instruction by default, and the failure would land in the assembler
rather than in a gate. Capabilities are declared positively here: an arch gets a feature
only by naming it, and an arch nobody has entered gets nothing.

The second is that lowering's supported-arch list was a separate allowlist of names, which
is a claim about *validation* pretending to be a claim about *hardware*. Those come apart:
gfx950 has every instruction the CDNA4 paths want and none of them have run here. So this
table says what the silicon has, and ``matrix_core`` is what lowering gates on -- the
vendored flex bodies emit MFMA, so an arch whose matrix core is WMMA is refused for a
reason that names the missing body instead of the arch.
"""

from __future__ import annotations

import dataclasses


# Matrix-core families. The flex kernel bodies are written against one or the other, not
# both: the fragment layouts differ, so the softmax and accumulator layouts built around
# them differ too, which is why this selects a body rather than an instruction.
MFMA = "mfma"  # CDNA, v_mfma_*
WMMA = "wmma"  # RDNA, v_wmma_*


@dataclasses.dataclass(frozen=True)
class ArchCaps:
    """Hardware the flex kernels can ask for, per architecture.

    Every field is a property of the silicon rather than of our confidence in it. Whether
    a path has been *validated* is not recorded here -- see the head-dim allowlist in
    ``flydsl_flash_attention`` for that -- because mixing the two is what made the old
    allowlist unable to say "this arch has the instructions and has never been run".
    """

    # Which matrix-core family this arch has, and therefore which kernel body can serve
    # it. Not a feature flag: there is no arch with both.
    matrix_core: str

    # Per-workgroup LDS, which is what bounds a tile choice. The addressable cap rather
    # than a policy number: the compiler enforces it, and asking for more fails the build
    # with "local memory (N) exceeds limit (M)". Both the backward's tile filter in
    # lowering and the builders' own check read it from here, so the two cannot disagree
    # about which tiles exist.
    lds_budget_bytes: int

    # 16 B DMA-to-LDS (``buffer_load_dwordx4_lds``), which lets a KV tile land in LDS
    # without passing through registers. CDNA3 only has the 4 B form, which is not worth
    # the loop restructuring, so it uses the register-staged prefetch instead.
    dma_to_lds_b128: bool

    # ``ds_read_tr16_b64``, a transposing LDS read. Without it the second GEMM's V operand
    # has to be written to LDS already transposed, which costs the store-side swizzle.
    lds_transpose_read: bool

    # MFMA32 with K=16 rather than K=8, halving the number of MFMA issues per tile.
    mfma_k16: bool

    # ``permlane32_swap`` + ``cvt_pk_bf16_f32``, which together turn the O store into one
    # 128-bit store per lane pair instead of a per-lane dwordx2.
    permlane_o_store: bool


_BY_ARCH: dict[str, ArchCaps] = {
    # CDNA3. The architecture this backend was developed and benchmarked on, and the only
    # one whose numbers here come from hardware.
    "gfx942": ArchCaps(
        matrix_core=MFMA,
        lds_budget_bytes=65536,
        dma_to_lds_b128=False,
        lds_transpose_read=False,
        mfma_k16=False,
        permlane_o_store=False,
    ),
    # CDNA4. Same MFMA family as gfx942 and served by the same kernel bodies, with four
    # additions the bodies already select on. None of it has run on silicon here; what
    # backs it is that every admitted head dim lowers for a gfx950 target and comes out
    # with less register pressure than on gfx942 (see
    # `test_gfx950_builds_from_a_gfx942_host`).
    #
    # The 2.5x LDS is not a spare-headroom note: the forward at head_dim 256 allocates
    # 66048 B here against 65536 on gfx942, because the V swizzle's padding lands
    # differently, and it only fits at all because of this number. It is also what makes
    # the wider backward Q tile reachable at head_dim 256 (131072 B), which gfx942 cannot
    # hold. The doubled bank count is the matching caveat: the K swizzle was derived
    # against 32 banks, so it is a permutation either way -- answers do not change -- but
    # its conflict-freedom was never re-derived for 64.
    "gfx950": ArchCaps(
        matrix_core=MFMA,
        lds_budget_bytes=163840,
        dma_to_lds_b128=True,
        lds_transpose_read=True,
        mfma_k16=True,
        permlane_o_store=True,
    ),
    # RDNA4. Entered so the refusal can say what is missing, which is a kernel body: the
    # flex bodies emit MFMA and this part has WMMA, and the two are not a matter of
    # swapping an instruction. RDNA4's WMMA is 16x16x16 on wave32 against CDNA's 32x32x16
    # on wave64, so a lane holds 8 accumulator elements rather than 16 columns of one row.
    # Every score-site index -- the softmax column mapping, the causal and window bounds,
    # the bias and dropout offsets -- is derived from that layout, and the donor's gfx1201
    # forward even computes S = K Qᵀ rather than Q Kᵀ because the result of one WMMA has to
    # land as the operand of the next. None of the MFMA flex glue survives that.
    #
    # The capabilities below are still worth recording accurately, because they are what a
    # WMMA body would be built against: no 16 B DMA-to-LDS (RDNA4 has no global-to-LDS path
    # at all), no transposing LDS read, and 64 KB of LDS -- CDNA4's 2.5x is a CDNA change,
    # and WGP mode does not raise it.
    "gfx1201": ArchCaps(
        matrix_core=WMMA,
        lds_budget_bytes=65536,
        dma_to_lds_b128=False,
        lds_transpose_read=False,
        mfma_k16=False,
        permlane_o_store=False,
    ),
}


def caps_for(arch: str | None) -> ArchCaps | None:
    """Capabilities for a ``gfxNNN`` name, or None if this table has never heard of it.

    Matched by prefix so the target-feature suffixes a device name carries
    (``gfx942:sramecc+:xnack-``) do not each need an entry.
    """
    if not arch:
        return None
    for name, caps in _BY_ARCH.items():
        if arch.startswith(name):
            return caps
    return None


def known_archs() -> tuple[str, ...]:
    """Every arch this table describes, for error messages that list the alternatives."""
    return tuple(sorted(_BY_ARCH))


def require_caps(arch: str | None, matrix_core: str = MFMA) -> ArchCaps:
    """Capabilities for ``arch``, or raise saying why this body cannot serve it.

    For kernel builders, which have no way to decline. Inductor's gate refuses both of
    these cases before a build is attempted, so reaching either means something bypassed
    lowering -- a direct builder call from a test or benchmark, most likely with
    ``FLYDSL_GPU_ARCH`` set to an arch nobody has entered.
    """
    caps = caps_for(arch)
    if caps is None:
        raise ValueError(
            f"no capability entry for {arch!r}; add one to "
            f"vendored_templates/flydsl/arch_caps.py (known: {', '.join(known_archs())})"
        )
    if caps.matrix_core != matrix_core:
        raise ValueError(
            f"this kernel emits {matrix_core} and {arch} has {caps.matrix_core}"
        )
    return caps

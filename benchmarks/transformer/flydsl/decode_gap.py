"""What a decode shape costs without a decode kernel, bare and with a mod in the graph.

`use_decode` in `flex_attention.py` is reachable only from `BACKEND='TRITON_DECODE'` or
from `AUTO`, so `BACKEND='FLYDSL'` never routes to `flex_decoding`: a decode shape is
served by the ordinary prefill kernel. That is correct -- mods and all -- and the point of
this script is to price it rather than to leave "no decode support" as an unquantified
hole. The deficit is against Triton's *decode* kernel; against Triton's prefill kernel, on
the same shape, FlyDSL is still well ahead, and quoting only the first number would be as
misleading as quoting only the second.

This script found the answer, in three parts, and none of them was a decode kernel. Two were
padding. The FlyDSL column used to be *flat* from `Sq` 1 to 8 while Triton decode rose,
because the autotuner picked `BLOCK_M=128` and a single query row was padded into a 128-row
tile; the MFMA issue count follows the tile and not the rows in it, so the padding was the
cost. Selecting a 64-row tile for `Sq <= 64` moved the column 1.6x. That left the MHA row
at 1.20x against the GQA rows at ~3x, and since MHA is the one shape with no group to pack,
the difference *was* the packing: a group of 4 was streaming the same KV tile from 4
workgroups and padding 63 of 64 rows in each. Giving the whole group one tile moved the GQA
rows another 2.4x, to 1.27x at `(1, 8192)` and *0.96x* -- ahead of Triton decode -- by
`Sq` 8.

The third was parallelism, which is what Triton decode's own `SPLIT_KV` is for, and the
residual's shape said so: `Skv` 4096 measured *worse* (1.47x) than 8192 (1.27x), because the
packed grid is `B * Hkv` workgroups regardless of how long the KV walk is, so at 64
workgroups on 80 CUs there was idle machine and the shorter the walk the more the fixed costs
showed. Splitting the walk across workgroups gives it something to do, and it is the largest
of the three on the shapes where the grid was smallest.

Sparse decode -- the same three wins under a block mask, which used to refuse two of them --
is `sparse_gaps.py`. This script adds the *mod* axis, which at decode is in the worst
position it can be: the tile is 64 rows of which one is real, so the mod site's vector-ALU
work is paid on 64x the elements the answer needs, against a KV walk with little MFMA to hide it
behind. Triton decode pays the same tax in the same place, which is what makes the column
comparable.

The `--mods` variants:

  none     no mod at all
  alibi    a score_mod reading `h`, `q_idx` and `kv_idx`: three coordinate materialisations
  softcap  `tanh`, the most expensive mod in the matrix
  mask     an all-true BlockMask, so the mask_mod site rather than the score one, on a walk
           exactly as long as the bare row's. `sparse_gaps.py --mask-cost` asks this at
           prefill, where it costs nothing; at decode the tile is 1/64th real.

Note [the t_dec column is not comparable across mods]
The baseline moves: Triton's decode kernel is *faster* with an all-true BlockMask than with
no mask at all -- 1.5x at `Hkv` 8 and 9x at `Hkv` 1 (563 us of templated kernel against 52
us on the same shape, profiled). The gap tracks the workgroup count, and is absent at `Hkv`
32 where `B * Hkv` already fills the machine, so its no-mask path is not dividing the KV
walk the way its masked path does: `block_mask=None` gets one sparse block of `1 << 30`
from `_create_empty_block_mask`, and a one-block walk has nothing to split.

So `mod=none` flatters this backend and `mod=mask` is the fairer decode baseline even for
the question "what does a bare decode cost". FlyDSL's own mask cost is the honest 6-8% the
two FlyDSL columns show.

`--check` verifies the prefill kernel against eager at these shapes with a `score_mod`
before timing anything, since a fast wrong answer is not a baseline.

    python benchmarks/transformer/flydsl/decode_gap.py
    python benchmarks/transformer/flydsl/decode_gap.py --check
    python benchmarks/transformer/flydsl/decode_gap.py --mods none,alibi
    python benchmarks/transformer/flydsl/decode_gap.py --wall
"""

import argparse

import torch

from _common import backend_options, best_us, device_us, geomean

from torch.nn.attention.flex_attention import create_block_mask, flex_attention


# Long KV against a handful of query rows, which is what a generation step looks like.
# The MHA row (Hq == Hkv) is in deliberately: it is the one shape where a GQA group cannot
# be packed into the M tile, so it isolates how much of the deficit the packing owns -- and
# it is now the slowest row, which is the packing having taken the rest.
SHAPES = [
    # B, Hq, Hkv, Sq, Skv, D
    (8, 32, 8, 1, 4096, 128),
    (8, 32, 8, 1, 8192, 128),
    (8, 32, 32, 1, 8192, 128),
    (8, 32, 8, 4, 8192, 128),
    (8, 32, 8, 8, 8192, 128),
    (32, 32, 8, 1, 8192, 128),
    # Group 32 (MQA) and group 2, the ends of the packing's range: one Q tile holds the
    # whole group at 32, and at 2 there are 32 padded rows left in it either way.
    (8, 32, 1, 1, 8192, 128),
    (8, 32, 16, 1, 8192, 128),
    # head_dim 64, where the tile is half the LDS and the walk twice the blocks per byte.
    (8, 32, 8, 1, 8192, 64),
    # A cache long enough that the split is the only thing keeping the grid busy.
    (8, 32, 8, 1, 32768, 128),
]

# `TRITON` is named rather than left to default, because the default is `AUTO` and `AUTO`
# routes a decode shape straight to `flex_decoding` -- an unnamed column would silently be
# a second copy of the decode one, which is how this was first written.
BACKENDS = ("TRITON", "TRITON_DECODE", "FLYDSL")


def _alibi(score, b, h, q_idx, kv_idx):
    return score + 0.125 * (h + 1) * (kv_idx - q_idx)


def _softcap(score, b, h, q_idx, kv_idx):
    return 20.0 * torch.tanh(score / 20.0)


def _alltrue(b, h, q_idx, kv_idx):
    """Admits everything, but is a real graph, so it is lowered and evaluated."""
    return kv_idx >= 0


# name -> (score_mod, mask_mod). A mask_mod here is built into a BlockMask, so it reaches
# the kernel as one -- which is the point of the variant; see the module docstring.
MODS = {
    "none": (None, None),
    "alibi": (_alibi, None),
    "softcap": (_softcap, None),
    "mask": (None, _alltrue),
}


def _inputs(b, hq, hkv, sq, skv, d):
    return (
        torch.randn(b, hq, sq, d, device="cuda", dtype=torch.bfloat16),
        torch.randn(b, hkv, skv, d, device="cuda", dtype=torch.bfloat16),
        torch.randn(b, hkv, skv, d, device="cuda", dtype=torch.bfloat16),
    )


def _compiled(backend, gqa, score_mod=None, block_mask=None):
    options = backend_options("flydsl") if backend == "FLYDSL" else {"BACKEND": backend}

    def call(q, k, v):
        return flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=block_mask,
            enable_gqa=gqa,
            kernel_options=options,
        )

    # A reset per backend: the shapes here blow through the recompile limit otherwise, and
    # a fallback to eager reads as "every backend is identical" rather than as a failure.
    torch._dynamo.reset()
    return torch.compile(call, dynamic=False)


def check():
    """The prefill kernel against eager at decode shapes, with a mod in the graph."""

    def score_mod(score, b, h, q_idx, kv_idx):
        return score * 0.5

    for b, hq, hkv, sq, skv, d in SHAPES[:3]:
        q, k, v = _inputs(min(b, 2), hq, hkv, sq, skv, d)
        gqa = hq != hkv
        with torch.no_grad():
            got = _compiled("FLYDSL", gqa, score_mod)(q, k, v)
            want = flex_attention(q, k, v, score_mod=score_mod, enable_gqa=gqa)
        rel = (got.float() - want.float()).abs().max() / want.float().abs().max()
        status = "ok" if rel < 2e-2 else "MISMATCH"
        print(f"  B{min(b, 2)} Hq{hq} Hkv{hkv} Sq{sq} Skv{skv} D{d}: rel {rel:.3g} {status}")
        if rel >= 2e-2:
            raise SystemExit("decode-shape correctness check failed")


def _us(t):
    return "n/a" if t is None else f"{t:.0f}"


def table(mod_name, wall):
    """One shape ladder for one mod, against Triton's prefill and decode kernels.

    Returns `{shape: {backend: us}}` as well as printing, so `main` can ask the question
    no single row can: which spelling of the same computation each backend is fastest at.
    """
    score_mod, mask_mod = MODS[mod_name]
    measure = best_us if wall else device_us
    print(f"mod={mod_name}")
    print(
        f"  {'B, Hq, Hkv, Sq, Skv, D':<26}{'triton':>9}{'t_dec':>8}{'flydsl':>9}"
        f"{'fly/tri':>9}{'fly/dec':>9}"
    )
    ratios = []
    cells = {}
    for shape in SHAPES:
        b, hq, hkv, sq, skv, d = shape
        q, k, v = _inputs(*shape)
        gqa = hq != hkv
        # Head-broadcast (`H=None`), because Triton's decode kernel refuses a block mask
        # with a real head axis under GQA and the column would go `n/a` for no reason
        # this variant is about. `sparse_gaps.py` is where that distinction is priced.
        block_mask = (
            create_block_mask(mask_mod, b, None, sq, skv, device="cuda")
            if mask_mod is not None
            else None
        )

        times = {}
        for backend in BACKENDS:
            fn = _compiled(backend, gqa, score_mod, block_mask)
            times[backend] = measure(fn, q, k, v)

        cells[shape] = times
        label = f"{b}, {hq}, {hkv}, {sq}, {skv}, {d}"
        tri, dec, fly = times["TRITON"], times["TRITON_DECODE"], times["FLYDSL"]
        # Reported as triton/flydsl, so above 1.00x is FlyDSL ahead -- the direction the
        # rest of this directory uses. The original version of this script printed
        # flydsl/t_decode, where lower was better, and the two were easy to confuse.
        vs_tri = None if None in (tri, fly) else tri / fly
        vs_dec = None if None in (dec, fly) else dec / fly
        if vs_dec is not None:
            ratios.append(vs_dec)
        print(
            f"  {label:<26}{_us(tri):>9}{_us(dec):>8}{_us(fly):>9}"
            f"{'n/a' if vs_tri is None else f'{vs_tri:.2f}x':>9}"
            f"{'n/a' if vs_dec is None else f'{vs_dec:.2f}x':>9}",
            flush=True,
        )
    if ratios:
        print(
            f"  geomean vs t_decode over {len(ratios)} shapes: {geomean(ratios):.2f}x"
            "  (>1 = FlyDSL faster)"
        )
    return ratios, cells


def dense_summary(none_cells, mask_cells):
    """Dense decode with each backend spelling it whichever way suits it best.

    `none` and `mask` compute the same thing -- an all-true mask masks nothing -- so a
    backend's cost for dense decode is the cheaper of its two columns, and the comparison
    that owes nobody an excuse is best against best. It matters because the two backends
    do not agree about which spelling is cheaper: this one is faster with no mask (the
    mask costs it 6-8%), and Triton's decode kernel is faster *with* one, by up to 9x, for
    the reason in Note [the t_dec column is not comparable across mods]. Quoting only
    `mod=none` would hand this backend that 9x, which is not its to take.
    """
    print("dense decode, each backend at its cheaper spelling of the same math:")
    print(
        f"  {'B, Hq, Hkv, Sq, Skv, D':<26}{'t_dec':>8}{'spell':>7}{'flydsl':>9}"
        f"{'spell':>7}{'fly/dec':>9}"
    )
    ratios = []
    for shape in none_cells:
        best = {}
        for backend in ("TRITON_DECODE", "FLYDSL"):
            options = {
                "none": none_cells[shape][backend],
                "mask": mask_cells[shape][backend],
            }
            options = {k: v for k, v in options.items() if v is not None}
            best[backend] = (
                min(options.items(), key=lambda kv: kv[1]) if options else (None, None)
            )
        (dec_spell, dec), (fly_spell, fly) = best["TRITON_DECODE"], best["FLYDSL"]
        ratio = None if None in (dec, fly) else dec / fly
        if ratio is not None:
            ratios.append(ratio)
        label = ", ".join(str(x) for x in shape)
        print(
            f"  {label:<26}{_us(dec):>8}{dec_spell or 'n/a':>7}{_us(fly):>9}"
            f"{fly_spell or 'n/a':>7}"
            f"{'n/a' if ratio is None else f'{ratio:.2f}x':>9}"
        )
    if ratios:
        print(
            f"  geomean over {len(ratios)} shapes: {geomean(ratios):.2f}x"
            "  (>1 = FlyDSL faster)"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify against eager first")
    parser.add_argument(
        "--mods", default=",".join(MODS), help=f"comma-separated subset of {list(MODS)}"
    )
    parser.add_argument(
        "--wall",
        action="store_true",
        help="time end to end instead of on the device timeline; includes ~350us of "
        "Inductor host work per call, which at decode sizes is most of it",
    )
    args = parser.parse_args()

    if args.check:
        print("correctness at decode shapes:")
        check()
        print()

    print(
        "microseconds of "
        + ("wall clock per call" if args.wall else "device time per call")
        + ", best of 3\n"
    )
    per_mod, cells = {}, {}
    for mod_name in args.mods.split(","):
        per_mod[mod_name], cells[mod_name] = table(mod_name, args.wall)
        print()

    if len(per_mod) > 1:
        print("cost of the mod, FlyDSL against Triton decode (geomean vs t_decode):")
        for mod_name, ratios in per_mod.items():
            if ratios:
                print(f"  {mod_name:<10}{geomean(ratios):>6.2f}x")
        print()

    if "none" in cells and "mask" in cells:
        dense_summary(cells["none"], cells["mask"])


if __name__ == "__main__":
    main()

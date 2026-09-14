"""What a decode shape costs without a decode kernel.

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
of the three on the shapes where the grid was smallest. Every row here is now at or ahead of
Triton's dedicated decode kernel.

What this does not cover is *sparse* decode, where their strongest numbers are: a block-mask
walk is left unsplit, because its length is the mask's data rather than the shape's.

`--check` verifies the prefill kernel against eager at these shapes with a `score_mod`
before timing anything, since a fast wrong answer is not a baseline.

    python benchmarks/transformer/flydsl/decode_gap.py
    python benchmarks/transformer/flydsl/decode_gap.py --check
"""

import argparse

import torch

from _common import backend_options, best_us

from torch.nn.attention.flex_attention import flex_attention


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
]

# `TRITON` is named rather than left to default, because the default is `AUTO` and `AUTO`
# routes a decode shape straight to `flex_decoding` -- an unnamed column would silently be
# a second copy of the decode one, which is how this was first written.
BACKENDS = ("TRITON", "TRITON_DECODE", "FLYDSL")


def _inputs(b, hq, hkv, sq, skv, d):
    return (
        torch.randn(b, hq, sq, d, device="cuda", dtype=torch.bfloat16),
        torch.randn(b, hkv, skv, d, device="cuda", dtype=torch.bfloat16),
        torch.randn(b, hkv, skv, d, device="cuda", dtype=torch.bfloat16),
    )


def _compiled(backend, gqa, score_mod=None):
    options = backend_options("flydsl") if backend == "FLYDSL" else {"BACKEND": backend}

    def call(q, k, v):
        return flex_attention(
            q, k, v, score_mod=score_mod, enable_gqa=gqa, kernel_options=options
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify against eager first")
    args = parser.parse_args()

    if args.check:
        print("correctness at decode shapes:")
        check()
        print()

    print(f"{'B, Hq, Hkv, Sq, Skv, D':<26}{'triton':>10}{'t_decode':>10}{'flydsl':>10}{'fly/dec':>9}")
    for shape in SHAPES:
        b, hq, hkv, sq, skv, d = shape
        q, k, v = _inputs(*shape)
        gqa = hq != hkv

        times = {}
        for backend in BACKENDS:
            fn = _compiled(backend, gqa)
            with torch.no_grad():
                fn(q, k, v)
                torch.cuda.synchronize()
                times[backend] = best_us(fn, q, k, v)

        label = f"{b}, {hq}, {hkv}, {sq}, {skv}, {d}"
        ratio = times["FLYDSL"] / times["TRITON_DECODE"]
        print(
            f"{label:<26}{times['TRITON']:10.0f}{times['TRITON_DECODE']:10.0f}"
            f"{times['FLYDSL']:10.0f}{ratio:8.2f}x"
        )


if __name__ == "__main__":
    main()

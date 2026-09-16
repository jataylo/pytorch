"""The two sparse gaps, priced. One was real; the other turned out not to exist.

**Sparse decode was real: 2.5-8.4x.** Decode had two wins turned off whenever a block mask
was present. `PACK_GQA` was refused because a packed Q tile spans the GQA group while a
BlockMask is indexed per head; the host now hands over the group's *union*, which is exact
for any mask that does not read `h` and conservative otherwise, so it is allowed. And the
KV split was pinned to 1 because a masked walk's length is the mask's data rather than the
shape's; it now takes the same rule as the dense walk, on the grounds that a surplus slice
at decode runs zero iterations concurrently with the ones that have work. See Note [sizing
a split against a mask you cannot see] in `flydsl_flash_attention.py`.

    B, Hq, Hkv, Sq, Skv, D        mask      t_dec  before   after   gain  /t_dec
    8, 32,  8, 1,  4096, 128      cache90     142     658     190  3.47x   1.33x
    8, 32,  8, 1,  4096, 128      window1024   69     204      74  2.74x   1.08x
    8, 32,  8, 1,  8192, 128      cache90     267    1292     353  3.66x   1.32x
    8, 32,  8, 1,  8192, 128      window1024   69     202      73  2.78x   1.06x
    8, 32, 32, 1,  8192, 128      cache90     995    1252    1089  1.15x   1.09x
    8, 32, 32, 1,  8192, 128      window1024  142     202     210  0.96x   1.48x
    8, 32,  8, 8,  8192, 128      cache90     394    1317     339  3.88x   0.86x
    8, 32,  8, 8,  8192, 128      window1024   90     204      79  2.59x   0.87x
   32, 32,  8, 1,  8192, 128      cache90     988    4382    1101  3.98x   1.11x
   32, 32,  8, 1,  8192, 128      window1024  146     663     213  3.12x   1.45x
    8, 64,  8, 1, 16384, 128      cache90     575    4977     596  8.35x   1.04x
    8, 64,  8, 1, 16384, 128      window1024   69     374      83  4.50x   1.21x

The one row that does not move is MHA (`Hq == Hkv`), which has no GQA group to pack, so
all it gets is the split -- 1.15x on the dense-ish mask and 0.96x on the sparse one. That
0.96x is the over-split the note above predicts and prices: a 1024-wide window over an
8192 cache is 12.5% dense, so the dense rule cuts a 16-block walk eight ways and the
combine shows. Four percent on the one shape that gains nothing either way.

Against Triton this goes from 3-8x behind its decode kernel to 0.86-1.48x of it. And
`t_dec` is `n/a` for most of the *per-head* table, because Triton's decode kernel handles
a KV head's whole group in one block and so refuses any block mask with a real head axis
under `enable_gqa` -- the case the union was written for. This kernel serves both layouts
at the same speed.

**Fully-unmasked blocks were not.** A BlockMask sorts blocks into partial ones, which need
`mask_mod` per element, and fully-unmasked ones, which do not. The walk visits one unioned
list and runs `mask_mod` on all of it, which looks like pure overhead on the full blocks --
for a causal mask at Sq 4096 that is ~97% of the walk evaluating a comparison whose answer
was settled at mask-build time.

It measures as nothing. `--mask-cost` runs an all-true mask over a dense walk, so every
block is visited, every block is full, and the mask is overhead by construction; against a
build with no mask at all it lands within 2%, sign varying between runs (-1.1% to +1.7%
on the last one, -1.5% to +1.2% on the one before). The mod site is vector-ALU work and
the loop is MFMA-bound, so a compare and a select per element co-issue with the matrix ops and never
reach the critical path.

And skipping it is *worse*, because the skip cannot be free. Which blocks are full is
runtime data, so the predicate is a runtime branch, and FlyDSL's dynamic `if` yields the
locals its body rebinds -- all 32 scores cross the merge point as branch results. That
costs register copies and costs the scheduler its interleaving. Measured 0.84-0.89x on the
all-true mask, which is the best case a skip can have (100% of blocks elided), and
0.77-0.80x on real causal, window and document masks at prefill:

    B, Hq, Hkv, Sq, Skv, D        mask   no skip    skip    gain
    2, 32, 8, 1024, 1024, 128   causal       354     458   0.77x
    2, 32, 8, 4096, 4096, 128   causal      4108    5210   0.79x
    2, 32, 8, 4096, 4096, 128   window      2026    2571   0.79x
    2, 32, 8, 4096, 4096, 128     doc8      1149    1448   0.79x
    2, 32, 8, 8192, 8192, 128   causal     15442   19402   0.80x
    4, 16, 4, 2048, 2048, 128   causal      1113    1451   0.77x

That last table is the one thing here this script cannot re-measure: the skip was
reverted, so there is no build left to fill its column. What `--mask-cost` measures is the
claim the revert rests on -- that the mod site costs nothing to begin with, so there is
nothing for a skip to win back.

The only structure that avoids the branch is FlexAttention's own -- partial and full lists
walked by separate loops with separately emitted bodies -- and that would buy a 1% ceiling
with double the emitted loop body and double the compile time. So the walk applies
`mask_mod` everywhere deliberately, and `--mask-cost` is the regression test for the claim.

    python benchmarks/transformer/flydsl/sparse_gaps.py --mask-cost
    python benchmarks/transformer/flydsl/sparse_gaps.py --decode
    python benchmarks/transformer/flydsl/sparse_gaps.py --check
"""

import argparse

import torch

from _common import backend_options, best_us, device_us

from torch.nn.attention.flex_attention import create_block_mask, flex_attention


# One query row against a long cache, which is what a generation step looks like.
DECODE_SHAPES = [
    # B, Hq, Hkv, Sq, Skv, D
    (8, 32, 8, 1, 4096, 128),
    (8, 32, 8, 1, 8192, 128),
    (8, 32, 32, 1, 8192, 128),
    (8, 32, 8, 8, 8192, 128),
    (32, 32, 8, 1, 8192, 128),
    (8, 64, 8, 1, 16384, 128),
]

# Dense walks, for pricing the mod site itself.
MASK_COST_SHAPES = [
    (2, 32, 8, 1024, 1024, 128),
    (2, 32, 8, 4096, 4096, 128),
    (2, 32, 8, 4096, 4096, 64),
    (2, 32, 8, 8192, 8192, 128),
]


def causal(b, h, q, kv):
    return q >= kv


def window(w):
    def f(b, h, q, kv):
        return (q >= kv) & (q - kv < w)

    f.__name__ = f"window{w}"
    return f


# Note [a decode mask has to put the query at the end of the cache]
# `q >= kv` at `Sq` 1 means `0 >= kv`, which admits exactly one KV block no matter how long
# the cache is. Timing that measures an almost-empty walk and says nothing about sparse
# decode -- it was the first version of this table, and it made the split look like a 3.7x
# regression because there was nothing to split. A generation step has the query at row
# `cache_len`, so the masks below are written against the *end* of the KV axis.
def cache_causal(skv):
    """Attend to a cache filled to 90%: what a batch mid-generation looks like."""
    limit = int(skv * 0.9)

    def f(b, h, q, kv):
        return kv < limit

    f.__name__ = "cache90"
    return f


def decode_window(w):
    """A sliding window anchored at the decode position, so `w / skv` dense."""

    def make(skv):
        lo = max(skv - w, 0)

        def f(b, h, q, kv):
            return kv >= lo

        f.__name__ = f"window{w}"
        return f

    make.label = f"window{w}"
    return make


DECODE_MASKS = (cache_causal, decode_window(1024))


def alltrue(b, h, q, kv):
    """Admits everything, but is a real graph, so it is lowered and evaluated."""
    return kv >= 0


def _inputs(b, hq, hkv, sq, skv, d):
    return (
        torch.randn(b, hq, sq, d, device="cuda", dtype=torch.bfloat16),
        torch.randn(b, hkv, skv, d, device="cuda", dtype=torch.bfloat16),
        torch.randn(b, hkv, skv, d, device="cuda", dtype=torch.bfloat16),
    )


def _compiled(backend, gqa, block_mask, **pins):
    options = (
        backend_options("flydsl", **pins) if backend == "FLYDSL" else {"BACKEND": backend}
    )

    def call(q, k, v):
        return flex_attention(
            q, k, v, block_mask=block_mask, enable_gqa=gqa, kernel_options=options
        )

    # A reset per column: these shapes blow through the recompile limit otherwise, and a
    # fallback to eager reads as "every column is identical" rather than as a failure.
    torch._dynamo.reset()
    return torch.compile(call, dynamic=False)


def _mask_for(mask_fn, b, hq, sq, skv, per_head=True):
    """A BlockMask over these shapes, either per-head or head-broadcast.

    The distinction decides whether Triton's decode kernel will take the shape at all: it
    handles a KV head's whole GQA group in one block, so it cannot give the group's heads
    different walks and refuses any block mask with a real head axis under `enable_gqa`
    (`_use_flex_decoding` wants `kv_indices.size(1) == 1`). Head-broadcast, it is eligible.
    This kernel takes both, because packing unions the group's lists rather than requiring
    them to be equal -- see Note [packing the GQA group into the Q tile].
    """
    return create_block_mask(
        mask_fn, b, hq if per_head else None, sq, skv, device="cuda", BLOCK_SIZE=(128, 128)
    )


def _time(fn, q, k, v):
    """Microseconds, or None if the backend refuses the shape."""
    try:
        with torch.no_grad():
            fn(q, k, v)
            torch.cuda.synchronize()
            return best_us(fn, q, k, v)
    except Exception:
        return None


def _us(t):
    return "n/a" if t is None else f"{t:.0f}"


def _ratio(a, b):
    return "n/a" if a is None or b is None else f"{a / b:.2f}x"


def check():
    """The new sparse decode paths against eager before anything is timed."""
    for b, hq, hkv, sq, skv, d in DECODE_SHAPES[:3]:
        b = min(b, 2)
        q, k, v = _inputs(b, hq, hkv, sq, skv, d)
        gqa = hq != hkv
        bm = _mask_for(cache_causal(skv), b, hq, sq, skv)
        with torch.no_grad():
            got = _compiled("FLYDSL", gqa, bm)(q, k, v)
            want = flex_attention(q, k, v, block_mask=bm, enable_gqa=gqa)
        rel = (got.float() - want.float()).abs().max() / want.float().abs().max()
        status = "ok" if rel < 2e-2 else "MISMATCH"
        print(f"  B{b} Hq{hq} Hkv{hkv} Sq{sq} Skv{skv} D{d}: rel {rel:.3g} {status}")
        if rel >= 2e-2:
            raise SystemExit("sparse decode correctness check failed")


def mask_cost_table():
    """What lowering a mask_mod costs when it cannot possibly change the answer.

    Three columns walking every KV tile and producing the same answer, so the only thing
    that varies is the mod site: no mask at all, against an explicitly built all-true
    BlockMask whose every block is visited and full.
    """
    print("mask_mod cost on a dense walk (all-true mask, so pure overhead):")
    print(f"{'B, Hq, Hkv, Sq, Skv, D':<28}{'no mask':>10}{'masked':>10}{'cost':>9}")
    for shape in MASK_COST_SHAPES:
        b, hq, hkv, sq, skv, d = shape
        q, k, v = _inputs(*shape)
        gqa = hq != hkv
        bm = _mask_for(alltrue, b, hq, sq, skv)
        # Pinned, or the two columns autotune to different tile heights and the comparison
        # measures the tile rather than the mod.
        pins = dict(BLOCK_M=128, MOD_VEC_SIZE=1, QK_PREFETCH_DEPTH=2)
        t_none = _time(_compiled("FLYDSL", gqa, None, **pins), q, k, v)
        t_mask = _time(_compiled("FLYDSL", gqa, bm, **pins), q, k, v)
        label = f"{b}, {hq}, {hkv}, {sq}, {skv}, {d}"
        print(
            f"{label:<28}{t_none:10.0f}{t_mask:10.0f}"
            f"{(t_mask / t_none - 1) * 100:8.1f}%"
        )


def decode_table(per_head):
    """Sparse decode: what the two decode wins are worth once a mask is allowed them.

    `before` is what a masked decode used to get, when a block mask refused both packing
    and the split; `after` lets the rules choose, which is what the lowering now does.
    `t_dec` is Triton's dedicated decode kernel and `triton` its general one -- the decode
    column is `n/a` wherever it declines the shape, which under GQA is every per-head
    block mask.
    """
    layout = "per-head" if per_head else "head-broadcast"
    print(f"sparse decode ({layout} block mask):")
    print(
        f"{'B, Hq, Hkv, Sq, Skv, D':<28}{'mask':>11}{'triton':>8}{'t_dec':>7}"
        f"{'before':>8}{'after':>8}{'gain':>7}{'/t_dec':>8}"
    )
    for shape in DECODE_SHAPES:
        b, hq, hkv, sq, skv, d = shape
        q, k, v = _inputs(*shape)
        gqa = hq != hkv
        for make_mask in DECODE_MASKS:
            mask_fn = make_mask(skv)
            bm = _mask_for(mask_fn, b, hq, sq, skv, per_head=per_head)
            before = device_us(
                _compiled("FLYDSL", gqa, bm, PACK_GQA=False, NUM_KV_SPLITS=1), q, k, v
            )
            after = device_us(_compiled("FLYDSL", gqa, bm), q, k, v)
            triton = device_us(_compiled("TRITON", gqa, bm), q, k, v)
            dec = device_us(_compiled("TRITON_DECODE", gqa, bm), q, k, v)
            label = f"{b}, {hq}, {hkv}, {sq}, {skv}, {d}"
            print(
                f"{label:<28}{mask_fn.__name__:>11}{_us(triton):>8}{_us(dec):>7}"
                f"{_us(before):>8}{_us(after):>8}"
                f"{_ratio(before, after):>7}{_ratio(after, dec):>8}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify against eager first")
    parser.add_argument("--decode", action="store_true", help="the sparse decode table")
    parser.add_argument("--mask-cost", action="store_true", help="the mod-site cost table")
    args = parser.parse_args()
    both = not (args.decode or args.mask_cost)

    if args.check:
        print("correctness at sparse decode shapes:")
        check()
        print()

    if args.mask_cost or both:
        mask_cost_table()
        print()
    if args.decode or both:
        decode_table(per_head=False)
        print()
        decode_table(per_head=True)


if __name__ == "__main__":
    main()

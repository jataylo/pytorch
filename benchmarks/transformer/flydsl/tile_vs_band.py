"""Does a taller Q tile pay for itself on a narrow mask, or walk KV it does not need?

`walk_cost.py` fits a 560 us per-call fixed cost that `profile_call.py` then finds entirely
*inside* the attention kernel -- one kernel on the device timeline, no regrid or copy
kernels beside it. So it is not host work and not the mod site. This script asks the next
question, which is whether it is a fixed cost at all.

A BlockMask is a *set of Q tiles*, each with its own KV list. Merging Q rows into a taller
tile therefore merges their lists, and the merge is a union: a 256-row tile walks every KV
block that any of its 256 rows needs. On a dense or causal mask that costs nothing, because
neighbouring rows want nearly the same KV. On a narrow band it costs the width of the band
-- the union over 256 rows of a band that is 128 wide is most of twice the band.

So the two columns here are the walk the mask asks for and the walk the tile actually takes,
measured on the host from the regridded lists, next to the time. It is mostly the tile:

    band  triton us  t rows   M64 us  rows  M128 us  rows  M256 us  rows
       1      557.2     128    649.5   128    609.3   128    779.7   256
       2      748.4     252    913.0   252    841.6   252    972.6   376
       4     1136.8     488   1458.3   488   1284.1   488   1353.2   608
       8     1892.6     912   2466.0   912   2070.0   912   2049.4  1024
      16     4463.5    1568   4037.4  1568   3420.4  1568   3149.0  1664

The autotuner picks 256 at every band width. It does benchmark the mask -- the lowering
hands it `create_num_blocks_fake_generator`, flex's shared stand-in -- but that stand-in is
*fully occupied*: `kv_num_blocks` is the whole capacity of `kv_indices` and the indices are
an arange. So every band here is ranked as a dense walk, where 256 is the fastest tile at
head_dim 128 since the padded K layout made it so. On the narrowest band that tile walks
256 KV rows per Q row where the mask asks for 128, and the over-walk decays as the band
widens (2.00x, 1.49x, 1.25x, 1.12x, 1.06x) -- the same shape as the deficit.

Pinning the tile to the mask's own 128 is worth most of it on the narrow end: band 1 goes
from 1.40x behind Triton to 1.09x. But the crossover is four or five blocks up, and every
mask anyone actually writes sits past it, because the union only costs what the band fails
to fill. Measured at head_dim 128 with `mod_matrix.py --flydsl-options BLOCK_M=...`:

    mask                 M128 us   M256 us
    causal                4310.2    3709.4
    document_mask         6830.6    5994.5
    prefix_lm             4493.7    3808.6
    sliding_window (512)  1461.8    1450.9

So the tall tile is right by 14-18% on three and free on the fourth, with
`sliding_window`'s 512-token window sitting on the tie. Pinning to the mask's grid would
trade 18% on the common masks for 22% on a band narrower than one tile of rows, so the
default stays 256 and `BLOCK_M=128` stays available for a genuinely narrow band.

The short-walk deficit is therefore two things: a tile height ranked on a denser walk than
the real one, which only bites below the crossover, and a per-row cost ~1.1x Triton's until
the walk is long enough for the pipelining to pay. Only the second is in the kernel, and on
real masks it is all of it.

    python benchmarks/transformer/flydsl/tile_vs_band.py
"""

import argparse

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from _common import BLOCK, backend_options, band_mask, best_us, score_mod as sm

from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_interface import (
    regrid_block_mask,
)


BATCH, HEADS, SEQ_LEN, HEAD_DIM = 4, 16, 4096, 128
DTYPE = torch.bfloat16
# The forward's tile heights. 256 is what the autotuner prefers at head_dim 128 since the
# padded K layout made it the fastest dense choice.
BLOCK_MS = (64, 128, 256)
# Its KV step, which the regrid splits the mask's 128-wide blocks down to.
KV_TILE = 64


def _mask(blocks_per_tile):
    return create_block_mask(
        band_mask(blocks_per_tile), BATCH, 1, SEQ_LEN, SEQ_LEN, device="cuda"
    )


def _rows_walked(block_mask, q_block_size, kv_block_size):
    """KV rows the kernel visits per Q row, once the lists are on its own tile grid.

    Per Q row rather than in total, so the three tile heights are comparable: a taller tile
    has fewer of them. This is the quantity the walk's cost is proportional to.
    """
    num_blocks, _ = regrid_block_mask(
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        block_mask.full_kv_num_blocks,
        block_mask.full_kv_indices,
        batch=BATCH,
        num_heads=HEADS,
        sparse_q_block_size=block_mask.BLOCK_SIZE[0],
        sparse_kv_block_size=block_mask.BLOCK_SIZE[1],
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        num_q_tiles=-(-SEQ_LEN // q_block_size),
        num_kv_tiles=-(-SEQ_LEN // kv_block_size),
    )
    blocks = int(num_blocks.sum())
    # Every tile in the grid contributes its list, so dividing by the tile count gives the
    # average tile's walk, and multiplying by the block width gives rows.
    tiles = num_blocks.numel()
    return blocks * kv_block_size / tiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = parser.parse_args()

    q, k, v = sm.generate_inputs(
        BATCH, HEADS, SEQ_LEN, HEADS, SEQ_LEN, HEAD_DIM, DTYPE, torch.device("cuda"),
        requires_grad=False,
    )
    print(
        f"B={BATCH} H={HEADS} S={SEQ_LEN} D={HEAD_DIM}, forward only. "
        f"'rows' is KV rows walked per Q row."
    )
    print(
        f"{'band':>5}{'triton us':>11}{'t rows':>8}"
        + "".join(f"{f'M{m} us':>9}{'rows':>7}" for m in BLOCK_MS)
    )

    for blocks_per_tile in args.blocks:
        block_mask = _mask(blocks_per_tile)
        call = dict(block_mask=block_mask)

        torch._dynamo.reset()
        compiled = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")
        with torch.no_grad():
            triton_call = dict(call, kernel_options=backend_options("triton"))
            compiled(q, k, v, **triton_call)
            triton_us = best_us(compiled, q, k, v, **triton_call)
        # Triton walks the mask's own grid, so its rows are the mask's own.
        triton_rows = _rows_walked(block_mask, BLOCK, BLOCK)

        cells = []
        for block_m in BLOCK_MS:
            torch._dynamo.reset()
            c = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")
            fly_call = dict(call, kernel_options=backend_options("flydsl", BLOCK_M=block_m))
            with torch.no_grad():
                c(q, k, v, **fly_call)
                us = best_us(c, q, k, v, **fly_call)
            cells.append((us, _rows_walked(block_mask, block_m, KV_TILE)))

        print(
            f"{blocks_per_tile:>5}{triton_us:>11.1f}{triton_rows:>8.0f}"
            + "".join(f"{us:>9.1f}{rows:>7.0f}" for us, rows in cells),
            flush=True,
        )

    print(
        "\nrows are the mask's, unioned over the rows of one tile: a taller tile walks the\n"
        "union of everything its Q rows need, which on a narrow band is most of twice the band.\n"
        "The union only costs what the band fails to fill, so the tall tile turns back into the\n"
        "right choice a few blocks up -- see the real-mask table in this script's docstring."
    )


if __name__ == "__main__":
    main()

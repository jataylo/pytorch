"""What does each backend pay per Q tile before it does any KV work?

Walk exactly N blocks of KV per Q tile by masking on the block index, so the length of the
walk is the only thing that varies, and fit time against the number of blocks walked. The
intercept is the cost paid once per call whatever the mask does; the slope is what a block
of KV actually costs.

This is the measurement that found the layout copies: the intercept came out at 1075 us
against Triton's 41 us while the slope was 1.6x *better*, which is not a shape a kernel
problem takes. See `profile_call.py` for what the intercept turned out to be. It reads
around 350 us now, so an intercept back up in the thousands means the copies are back.

    python benchmarks/transformer/flydsl/walk_cost.py --d 128
"""

import argparse

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from _common import BLOCK, backend_options, band_mask, best_us, score_mod as sm


BATCH, HEADS, SEQ_LEN = 4, 16, 4096
DTYPE = torch.bfloat16


def run(blocks_per_tile, backend, head_dim):
    q, k, v = sm.generate_inputs(
        BATCH,
        HEADS,
        SEQ_LEN,
        HEADS,
        SEQ_LEN,
        head_dim,
        DTYPE,
        torch.device("cuda"),
        requires_grad=False,
    )
    block_mask = create_block_mask(
        band_mask(blocks_per_tile), BATCH, 1, SEQ_LEN, SEQ_LEN, device="cuda"
    )
    walked = int(block_mask.kv_num_blocks.sum() + block_mask.full_kv_num_blocks.sum())
    torch._dynamo.reset()
    compiled = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")
    call = dict(block_mask=block_mask, kernel_options=backend_options(backend))
    with torch.no_grad():
        compiled(q, k, v, **call)
        return best_us(compiled, q, k, v, **call), walked


def fit(xs, ys):
    """Least-squares `(intercept, slope)` of y against x."""
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum(
        (x - mean_x) ** 2 for x in xs
    )
    return mean_y - slope * mean_x, slope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=128, help="head dim")
    args = parser.parse_args()

    workgroups = BATCH * HEADS * SEQ_LEN // BLOCK
    print(
        f"B={BATCH} H={HEADS} S={SEQ_LEN} D={args.d}, forward only, "
        f"{workgroups} workgroups"
    )
    print(
        f"{'blk/tile':>9}{'walked':>8}{'triton us':>11}{'flydsl us':>11}{'speedup':>9}"
    )

    rows = []
    for blocks_per_tile in (1, 2, 4, 8, 16):
        triton_us, walked = run(blocks_per_tile, "triton", args.d)
        flydsl_us, _ = run(blocks_per_tile, "flydsl", args.d)
        rows.append((walked, triton_us, flydsl_us))
        print(
            f"{blocks_per_tile:>9}{walked:>8}{triton_us:>11.1f}{flydsl_us:>11.1f}"
            f"{triton_us / flydsl_us:>8.2f}x",
            flush=True,
        )

    walked = [r[0] for r in rows]
    for name, column in (("triton", 1), ("flydsl", 2)):
        fixed, per_block = fit(walked, [r[column] for r in rows])
        print(f"{name}: {fixed:8.1f} us fixed + {per_block * 1e3:6.2f} ns/block")


if __name__ == "__main__":
    main()

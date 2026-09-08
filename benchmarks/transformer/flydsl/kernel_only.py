"""Time the forward per head_dim with no Inductor in the picture.

Builds the launcher directly, so nothing here depends on the lowering, the block-mask
regrid or the autotuner picking the same tiles twice. That makes it the right place to
price a change to the kernel itself -- a swizzle, a tile shape, a FlyDSL version bump --
where the end-to-end matrix would bury it under everything else a call does.

    python benchmarks/transformer/flydsl/kernel_only.py
"""

import argparse

import torch

from _common import time_launcher_us

from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_generic import (
    build_flex_flash_generic_module,
)
from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_interface import (
    prepare,
)


BATCH, HEADS, SEQ_LEN = 2, 8, 4096
HEAD_DIMS = [64, 96, 128, 160, 192, 224, 256]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-dims", type=int, nargs="+", default=HEAD_DIMS)
    parser.add_argument("--layout", default="bhsd", choices=("bhsd", "bshd"))
    parser.add_argument("--dense", action="store_true", help="drop the causal mask")
    args = parser.parse_args()

    causal = not args.dense
    print(
        f"B={BATCH} H={HEADS} S={SEQ_LEN} {args.layout}, "
        f"{'causal' if causal else 'dense'}, forward only"
    )
    for head_dim in args.head_dims:
        shape = (
            (BATCH, HEADS, SEQ_LEN, head_dim)
            if args.layout == "bhsd"
            else (BATCH, SEQ_LEN, HEADS, head_dim)
        )
        q, k, v = (
            torch.randn(*shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        )
        launcher = build_flex_flash_generic_module(
            num_heads=HEADS,
            head_dim=head_dim,
            causal=causal,
            dtype_str="bf16",
            layout=args.layout,
        )
        run, _, _ = prepare(launcher, q, k, v, out=torch.empty_like(q))
        us = time_launcher_us(run)
        # Halved for causal, so the number is comparable across the two.
        flops = 2 * 2 * BATCH * HEADS * SEQ_LEN * SEQ_LEN * head_dim
        flops *= 0.5 if causal else 1.0
        print(
            f"D={head_dim:3d}: {us:8.1f} us  {flops / us / 1e6:6.1f} TFLOP/s",
            flush=True,
        )


if __name__ == "__main__":
    main()

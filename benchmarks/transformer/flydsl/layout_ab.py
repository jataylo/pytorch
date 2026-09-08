"""For BSHD memory, is the kernel better off indexing it or copying it to BHSD?

Inductor picks the layout off the strides, so on the inputs a model produces it builds a
BSHD kernel and copies nothing. That is not free in the kernel: a BHSD `(batch, head)`
slice is contiguous where BSHD strides every token row by `num_heads * head_dim`. This
pins `LAYOUT` both ways on the same inputs and prices the trade.

The copies are four passes over q/k/v-sized memory, so they grow with `S`, where the
kernel's penalty grows with the work -- `S**2` for a dense mask. So a long enough dense
shape should prefer to copy. It has not happened yet at any shape tried here, which is why
the stride check is the default rather than a heuristic on sequence length.

    python benchmarks/transformer/flydsl/layout_ab.py
    python benchmarks/transformer/flydsl/layout_ab.py --bwd
"""

import argparse

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from _common import bshd_inputs, best_us


# (batch, heads, seq_len, head_dim, causal). The forward set walks S out to where the
# copies should start to look cheap; the backward set stays at shapes it is used at.
FORWARD_SHAPES = [
    (4, 16, 4096, 128, True),
    (4, 16, 4096, 128, False),
    (2, 16, 8192, 128, False),
    (1, 16, 16384, 128, False),
    (1, 16, 32768, 128, False),
]
BACKWARD_SHAPES = [
    (2, 8, 1024, 64, True),
    (4, 16, 4096, 128, True),
    (4, 16, 4096, 128, False),
    (4, 16, 4096, 64, True),
]


def causal(b, h, q, kv):
    return q >= kv


def run(batch, heads, seq_len, head_dim, masked, layout, backward):
    torch.manual_seed(0)
    q, k, v = bshd_inputs(batch, heads, seq_len, head_dim, grad=backward)
    block_mask = (
        create_block_mask(causal, batch, 1, seq_len, seq_len, device="cuda")
        if masked
        else None
    )
    torch._dynamo.reset()
    options = {"BACKEND": "FLYDSL", "LAYOUT": layout}

    if backward:
        compiled = torch.compile(flex_attention, dynamic=False)
        out = compiled(q, k, v, block_mask=block_mask, kernel_options=options)
        grad = torch.randn_like(out)
        # Retaining the graph keeps the measurement to the backward kernels, without a
        # forward being replayed inside the timed region.
        torch.autograd.grad(out, (q, k, v), grad, retain_graph=True)
        return best_us(
            lambda: torch.autograd.grad(out, (q, k, v), grad, retain_graph=True)
        )

    compiled = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")
    call = dict(block_mask=block_mask, kernel_options=options)
    with torch.no_grad():
        compiled(q, k, v, **call)
        return best_us(compiled, q, k, v, **call)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bwd", action="store_true", help="measure the backward instead"
    )
    args = parser.parse_args()
    shapes = BACKWARD_SHAPES if args.bwd else FORWARD_SHAPES

    print(f"BSHD memory as a BHSD view, {'backward' if args.bwd else 'forward'} only")
    print(f"{'shape':>18}{'mask':>8}{'bshd us':>10}{'bhsd us':>10}{'bshd/bhsd':>11}")
    for batch, heads, seq_len, head_dim, masked in shapes:
        bshd_us = run(batch, heads, seq_len, head_dim, masked, "bshd", args.bwd)
        bhsd_us = run(batch, heads, seq_len, head_dim, masked, "bhsd", args.bwd)
        shape = f"{batch}x{heads}x{seq_len}x{head_dim}"
        print(
            f"{shape:>18}{'causal' if masked else 'dense':>8}{bshd_us:>10.1f}"
            f"{bhsd_us:>10.1f}{bhsd_us / bshd_us:>10.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()

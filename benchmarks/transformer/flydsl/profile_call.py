"""Device kernels for one flex-attention call, by self time.

`walk_cost.py` says how much is being paid outside the walk; this says what is spending it.
A fixed cost inside the kernel shows up as the attention kernel itself taking longer than
its walk can explain, and a fixed cost around it shows up as company: elementwise kernels
for the mask regrid, or `copy_` for a layout conversion.

Run with a short walk (`--blocks 1`), where anything fixed is most of the call, and again
with a long one (`--blocks 16`) to confirm it is fixed rather than proportional.

A healthy forward is one kernel and nothing else. It used to be five: four `aten::copy_`
worth 574 us, converting BSHD inputs into the BHSD layout the launcher was hardcoded to.

    python benchmarks/transformer/flydsl/profile_call.py
    python benchmarks/transformer/flydsl/profile_call.py --shapes    # what got copied
"""

import argparse
from collections import defaultdict

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.profiler import ProfilerActivity, profile

from _common import backend_options, band_mask, score_mod as sm


BATCH, HEADS, SEQ_LEN, HEAD_DIM = 4, 16, 4096, 128
DTYPE = torch.bfloat16
CALLS = 10


def profile_call(blocks_per_tile, backend, record_shapes):
    q, k, v = sm.generate_inputs(
        BATCH,
        HEADS,
        SEQ_LEN,
        HEADS,
        SEQ_LEN,
        HEAD_DIM,
        DTYPE,
        torch.device("cuda"),
        requires_grad=False,
    )
    block_mask = create_block_mask(
        band_mask(blocks_per_tile), BATCH, 1, SEQ_LEN, SEQ_LEN, device="cuda"
    )
    torch._dynamo.reset()
    compiled = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")
    call = dict(block_mask=block_mask, kernel_options=backend_options(backend))

    with torch.no_grad():
        for _ in range(5):
            compiled(q, k, v, **call)
        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CUDA], record_shapes=record_shapes
        ) as prof:
            for _ in range(CALLS):
                compiled(q, k, v, **call)
            torch.cuda.synchronize()

    per_kernel, counts = defaultdict(float), defaultdict(int)
    for event in prof.key_averages(group_by_input_shape=record_shapes):
        if event.self_device_time_total <= 0:
            continue
        # Shapes name the tensor a copy is copying, which is what identifies it as q/k/v.
        key = (event.key, str(event.input_shapes)[:60] if record_shapes else "")
        per_kernel[key] += event.self_device_time_total / CALLS
        counts[key] += event.count // CALLS
    return per_kernel, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=[1, 16],
        help="KV blocks walked per Q tile",
    )
    parser.add_argument("--backends", nargs="+", default=["triton", "flydsl"])
    parser.add_argument(
        "--shapes",
        action="store_true",
        help="split by input shape, to name what a copy is copying",
    )
    args = parser.parse_args()

    for blocks_per_tile in args.blocks:
        for backend in args.backends:
            per_kernel, counts = profile_call(blocks_per_tile, backend, args.shapes)
            total = sum(per_kernel.values())
            print(
                f"\n=== {backend}, {blocks_per_tile} block(s)/tile: "
                f"{total:.1f} us of device time ==="
            )
            for key, us in sorted(per_kernel.items(), key=lambda kv: -kv[1])[:8]:
                name, shapes = key
                print(f"  {us:9.1f} us  x{counts[key]:<4} {name[:76]}")
                if shapes:
                    print(f"{'':>25}{shapes}")


if __name__ == "__main__":
    main()

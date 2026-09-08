"""FlyDSL against Triton across the score_mod/mask_mod matrix, one row per cell.

Both backends see the same tensors, the same mods and the same BlockMask, compiled and
measured back to back in one process, so the ratio is not carrying an input or mask
difference between them. This is the number the README quotes as the geomean.

    python benchmarks/transformer/flydsl/mod_matrix.py            # forward
    python benchmarks/transformer/flydsl/mod_matrix.py --bwd      # backward
    python benchmarks/transformer/flydsl/mod_matrix.py --mods causal,alibi
"""

import argparse
import itertools
import traceback

import torch
from torch.nn.attention.flex_attention import flex_attention

from _common import backend_options, best_us, geomean, score_mod as sm


MODS = [
    "noop",
    "causal",
    "rel",
    "head_bias",
    "alibi",
    "sliding_window",
    "document_mask",
    "prefix_lm",
    "softcap",
]
HEAD_DIMS = [64, 128]
BATCH, HEADS_Q, HEADS_KV, SEQ_LEN = 4, 16, 16, 4096
DTYPE = torch.bfloat16


def measure(attn_type, head_dim, backward):
    shape = (BATCH, HEADS_Q, SEQ_LEN, HEADS_KV, SEQ_LEN, head_dim)
    q, k, v = sm.generate_inputs(
        *shape,
        DTYPE,
        torch.device("cuda"),
        requires_grad=backward,
        # document_mask is defined over packed sequences, so it needs the ragged input.
        nested_tensors=(attn_type == "document_mask"),
    )
    mod = sm.generate_score_mod(attn_type, shape)
    block_mask, _ = sm.generate_block_mask(attn_type, shape)

    timings = {}
    for backend in ("triton", "flydsl"):
        # Each backend gets its own compile; reset so Dynamo does not reject the second
        # for exceeding the recompile limit on the same code object.
        torch._dynamo.reset()
        compiled = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")
        call = dict(
            score_mod=mod,
            block_mask=block_mask,
            enable_gqa=True,
            kernel_options=backend_options(backend),
        )
        if backward:
            out = compiled(q, k, v, **call)
            grad = torch.randn_like(out)
            timings[backend] = (
                best_us(
                    lambda o=out, g=grad: torch.autograd.grad(
                        o, [q, k, v], g, retain_graph=True
                    )
                ),
                out.detach(),
            )
        else:
            with torch.no_grad():
                out = compiled(q, k, v, **call)
                timings[backend] = (best_us(compiled, q, k, v, **call), out)

    (triton_us, triton_out), (flydsl_us, flydsl_out) = (
        timings["triton"],
        timings["flydsl"],
    )
    # Relative to the largest element rather than elementwise: the small ones are dominated
    # by the order the two backends happen to accumulate in, which is not a disagreement.
    scale = triton_out.float().abs().max().clamp(min=1e-6)
    rel_err = ((flydsl_out.float() - triton_out.float()).abs().max() / scale).item()
    return triton_us, flydsl_us, rel_err


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bwd", action="store_true", help="measure the backward instead"
    )
    parser.add_argument("--mods", default=",".join(MODS), help="comma-separated subset")
    parser.add_argument(
        "--head-dims", default=",".join(str(d) for d in HEAD_DIMS), type=str
    )
    args = parser.parse_args()
    mods = args.mods.split(",")
    head_dims = [int(d) for d in args.head_dims.split(",")]

    header = ("mod", "D", "triton us", "flydsl us", "fly/tri", "relerr")
    print(
        f"{header[0]:<16}{header[1]:>5}{header[2]:>12}{header[3]:>12}"
        f"{header[4]:>10}{header[5]:>10}"
    )
    rows = []
    for attn_type, head_dim in itertools.product(mods, head_dims):
        try:
            triton_us, flydsl_us, rel_err = measure(attn_type, head_dim, args.bwd)
        except Exception as e:  # a mod failing is a result, not a reason to stop
            print(
                f"{attn_type:<16}{head_dim:>5}   FAILED {type(e).__name__}: "
                f"{str(e)[:90]}",
                flush=True,
            )
            traceback.print_exc()
            continue
        rows.append((attn_type, head_dim, triton_us / flydsl_us))
        print(
            f"{attn_type:<16}{head_dim:>5}{triton_us:>12.1f}{flydsl_us:>12.1f}"
            f"{triton_us / flydsl_us:>9.2f}x{rel_err:>10.1e}",
            flush=True,
        )

    if not rows:
        return
    print(
        f"\ngeomean speedup over {len(rows)} cells: "
        f"{geomean([r[2] for r in rows]):.2f}x  (>1 = FlyDSL faster)   "
        f"direction={'bwd' if args.bwd else 'fwd'}"
    )
    for name, cells in (
        ("wins", [r for r in rows if r[2] > 1]),
        ("losses", [r for r in rows if r[2] <= 1]),
    ):
        print(
            f"  {name}: {len(cells)} -> "
            + ", ".join(
                f"{r[0]}/D{r[1]} {r[2]:.2f}x"
                for r in sorted(cells, key=lambda r: -r[2])
            )
        )


if __name__ == "__main__":
    main()

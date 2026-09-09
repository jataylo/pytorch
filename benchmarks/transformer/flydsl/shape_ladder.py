"""The shape and head_dim ladder: FlyDSL against Triton and aten, both autotuned.

Where `mod_matrix.py` holds the shape fixed and sweeps the mods, this sweeps shapes and
head_dims and reports TFLOP/s with aten SDPA as a third reference. It is the script the
tables in `torch/_inductor/codegen/flydsl/MULTI_ARCH_ROADMAP.md` were taken from, so its
timing deliberately stays as it was — median of 15 rather than the best of 3 the rest of
this directory uses — because changing it would stop the numbers being comparable.

The default-config Triton comparison flatters FlyDSL by ~6x because Inductor's out-of-box
flex config is poor on ROCm. This script enables max_autotune so both backends are measured
at their best, which is the only comparison that should inform a keep/drop decision.

Every row is checked against eager and against the device's own peak before it is
believed: a run sharing the machine with other GPU work produced above-peak figures that
looked like a large win, so a timing that cannot be right is reported as such rather than
tabulated.

    python benchmarks/transformer/flydsl/shape_ladder.py
    python benchmarks/transformer/flydsl/shape_ladder.py --backward
"""

import argparse
import itertools

import torch
import torch.nn.functional as F
from torch._inductor import config
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


config.max_autotune = True

DTYPE = torch.bfloat16
# The whole supported ladder. This used to read "64 and 128; 96 fails a load-geometry
# assertion and 256 wants more LDS than gfx942 has" -- both were fixed in P1 (a bounded
# cooperative-load row plus the K swizzle turned off for non-power-of-two head_dims, and
# the V swizzle reclaiming the 1024 B that 256 overshot by), and the backward reaches the
# same ladder. 96/160/224 have no K swizzle, so they run below the power-of-two dims per
# FLOP; they are on the ladder because they are correct, not because they are fast.
HEAD_DIMS = (64, 96, 128, 160, 192, 224, 256)
PEAK = None


def alibi(score, b, h, q_idx, kv_idx):
    return score + 0.125 * (h + 1) * (kv_idx - q_idx)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def bench(fn, warmup=5, iters=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main(backward=False):
    global PEAK
    p = torch.cuda.get_device_properties(0)
    PEAK = p.multi_processor_count * 1024 * (p.clock_rate * 1e3) / 1e12
    print(
        f"{p.name} ({p.gcnArchName}), peak bf16 {PEAK:.1f} TFLOP/s, max_autotune=True"
    )

    shapes = [(1, 8, 4096), (2, 16, 2048), (4, 8, 4096), (2, 32, 4096)]
    print(f"bf16, forward{' + backward' if backward else ' only'}\n")
    # Both directions run the same three variants. The backward used to run `plain` alone
    # because its gate refused score_mod and mask_mod; it serves both since P2c, and the
    # causal row is the one worth watching because only that one exercises block skipping.
    variants = ("plain", "score_mod", "causal")

    hdr = (
        f"{'shape':>14} {'D':>4} {'variant':>10} {'FlyDSL':>16} {'Triton':>16} "
        f"{'aten':>16} {'fly/tri':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    for D, (B, H, S), variant in itertools.product(HEAD_DIMS, shapes, variants):
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = [
            torch.randn(
                (B, H, S, D),
                device="cuda",
                dtype=DTYPE,
                generator=g,
                requires_grad=backward,
            )
            for _ in range(3)
        ]
        grad_out = (
            torch.randn((B, H, S, D), device="cuda", dtype=DTYPE, generator=g)
            if backward
            else None
        )
        # Forward is 2 GEMMs, backward 5, so forward+backward is 7/2 the forward's FLOPs.
        gemms = 14.0 if backward else 4.0
        flops = gemms * B * H * S * S * D * (0.5 if variant == "causal" else 1.0)

        kwargs = {}
        if variant == "score_mod":
            kwargs["score_mod"] = alibi
        elif variant == "causal":
            kwargs["block_mask"] = create_block_mask(
                causal_mask, None, None, S, S, device="cuda"
            )

        ref = None
        if variant != "score_mod":
            ref = F.scaled_dot_product_attention(q, k, v, is_causal=variant == "causal")
            if backward:
                # Grads have to be checked too, and detached: `ref` is otherwise a live
                # graph that every later backward would accumulate into.
                ref_grads = torch.autograd.grad(ref, (q, k, v), grad_out)
                ref = ref.detach()

        cell = {}
        for backend in ("FLYDSL", "TRITON"):
            try:
                torch._dynamo.reset()
                c = torch.compile(flex_attention, fullgraph=True, dynamic=False)

                def fn(c=c, backend=backend):
                    out = c(q, k, v, kernel_options={"BACKEND": backend}, **kwargs)
                    if not backward:
                        return out
                    # `grad` rather than `.backward()`: accumulating into q/k/v.grad
                    # across iterations grows without bound and times the accumulate.
                    return torch.autograd.grad(out, (q, k, v), grad_out)

                got = fn()
                out = got[0] if backward else got
                if ref is not None and not backward:
                    rel = (
                        (out.float() - ref.float()).norm() / ref.float().norm()
                    ).item()
                    if rel > 2e-2:
                        raise RuntimeError(f"wrong numbers (l2_rel {rel:.3g})")
                elif ref is not None:
                    for name, a, b in zip(("dq", "dk", "dv"), got, ref_grads):
                        rel = ((a.float() - b.float()).norm() / b.float().norm()).item()
                        if rel > 2e-2:
                            raise RuntimeError(f"wrong {name} (l2_rel {rel:.3g})")
                us = bench(fn)
                tf = flops / (us * 1e-6) / 1e12
                if tf > PEAK:
                    raise RuntimeError(f"{tf:.0f} TF exceeds peak {PEAK:.0f} TF")
                cell[backend] = (us, tf)
            except Exception as e:
                cell[backend] = None
                print(f"  {backend} unusable: {str(e).splitlines()[0][:70]}")

        if variant in ("plain", "causal"):
            ic = variant == "causal"

            def aten_fn(ic=ic):
                o = F.scaled_dot_product_attention(q, k, v, is_causal=ic)
                return torch.autograd.grad(o, (q, k, v), grad_out) if backward else o

            us = bench(aten_fn)
            aten = f"{us:7.0f}us {flops / (us * 1e-6) / 1e12:5.1f}TF"
        else:
            aten = f"{'-':>16}"

        def fmt(x):
            return f"{x[0]:7.0f}us {x[1]:5.1f}TF" if x else f"{'FAIL':>16}"

        ratio = (
            f"{cell['FLYDSL'][1] / cell['TRITON'][1]:.2f}x"
            if cell.get("FLYDSL") and cell.get("TRITON")
            else "-"
        )
        print(
            f"{f'{B}x{H}x{S}':>14} {D:>4} {variant:>10} {fmt(cell.get('FLYDSL')):>16} "
            f"{fmt(cell.get('TRITON')):>16} {aten:>16} {ratio:>8}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backward", action="store_true", help="measure forward + backward"
    )
    main(backward=parser.parse_args().backward)

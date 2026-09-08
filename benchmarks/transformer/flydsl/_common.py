"""Shared plumbing for the FlyDSL flex-attention benchmarks.

These scripts all measure the same kernel against the same reference, so they agree here
about how inputs are built, how a call is timed, and what a block of the mask is.
"""

import math
import os
import sys

import torch


# score_mod.py, one directory up, is where the mods and the shapes come from: measuring
# against the generators the Triton backend is tuned on is what keeps the comparison fair.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import score_mod


# create_block_mask's granularity at these shapes. The kernel's own tiles are chosen by
# autotuning and are not this, which is the whole reason the block lists get regridded.
BLOCK = 128


def backend_options(backend, **extra):
    """Kernel options selecting a backend, plus whatever the caller pins on top.

    Triton gets an empty dict rather than the benchmark's own `get_kernel_options`: the
    only mod that pins anything there is `document_mask`, and its `BLOCK_N=32` costs
    Triton 3.7x at these shapes. Both backends autotune, which is the honest comparison.
    """
    options = {"BACKEND": "FLYDSL"} if backend == "flydsl" else {}
    options.update(extra)
    return options


def best_us(fn, *args, repeats=3, **kwargs):
    """Microseconds for the best of `repeats` timing runs.

    The box is usually shared. Interference only ever slows a run down, so the minimum is
    the closest thing to an uncontended measurement that a busy machine will give up.
    """
    return min(
        score_mod.benchmark_torch_function_in_microseconds(fn, *args, **kwargs)
        for _ in range(repeats)
    )


def time_launcher_us(run, warmup=20, iters=30, repeats=5):
    """Microseconds per call for a FlyDSL launcher invoked directly.

    Bypasses Inductor entirely, so what is left is the kernel plus the argument marshalling
    in `prepare`. Used where the question is about the generated ISA rather than about the
    lowering around it.
    """
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    best = float("inf")
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            run()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters * 1000)
    return best


def bshd_inputs(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, grad=False):
    """q/k/v the way a model hands them over: BSHD memory viewed as `[B, H, S, D]`.

    A projection reshaped to `[B, S, H, D]` and transposed produces exactly this, and
    FlexAttention passes the *view* down, so the sizes say BHSD while the memory says
    BSHD. Which layout the kernel is built for follows the strides, not the sizes; see
    `_kernel_layout` in torch/_inductor/kernel/flex/flydsl_flash_attention.py.
    """
    return tuple(
        torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=dtype)
        .transpose(1, 2)
        .detach()
        .requires_grad_(grad)
        for _ in range(3)
    )


def band_mask(blocks_per_tile):
    """A causal mask narrowed to `blocks_per_tile` blocks of KV per Q tile.

    Varying the band width varies the length of the walk and nothing else, which is what
    separates the cost of walking a block from the cost paid once per workgroup.
    """

    def mask_mod(b, h, q, kv):
        return (q >= kv) & (q // BLOCK - kv // BLOCK < blocks_per_tile)

    return mask_mod


def geomean(values):
    """Geometric mean, which is the right average for a column of ratios."""
    return math.exp(sum(math.log(v) for v in values) / len(values))

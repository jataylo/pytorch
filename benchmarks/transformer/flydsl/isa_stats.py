"""Registers, spills and time for a few forward builds, read out of the generated ISA.

Compiles each config with ASM dumping on and greps the metadata the assembler emits, so
the register pressure story is measured rather than inferred. Used to price anything that
changes what the compiler has to hold live: a FlyDSL version bump, a tile shape, or the
occupancy request itself.

`--waves` adds the 1-wave/EU build alongside the default 2. Asking for 2 caps the register
budget and is what makes head_dim 128 spill; relaxing it does clear the spills, and at the
shapes here it costs more throughput than the spills do, which is why the default stands.

    python benchmarks/transformer/flydsl/isa_stats.py
    python benchmarks/transformer/flydsl/isa_stats.py --waves
"""

import argparse
import contextlib
import glob
import io
import os
import re
import shutil
import tempfile

# These are read when flydsl's env manager is imported, so they have to be set before
# anything pulls it in -- including the vendored kernels below.
DUMP_DIR = os.environ.setdefault(
    "FLYDSL_DUMP_DIR", os.path.join(tempfile.gettempdir(), "flydsl_isa_stats")
)
os.environ.setdefault("FLYDSL_DEBUG_DUMP_ASM", "1")
# The ISA is written by the same pass that writes the per-stage MLIR, so asking for the
# assembly without asking for the IR silently gets neither.
os.environ.setdefault("FLYDSL_DUMP_IR", "1")
# A cache hit hands back a binary without going near the assembler, and so writes nothing
# to read. Every build here has to be a real compile.
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl

from _common import time_launcher_us

from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_generic import (
    build_flex_flash_generic_module,
)
from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_interface import (
    prepare,
)


BATCH, HEADS, SEQ_LEN = 2, 8, 4096

# The corners worth watching: the two head dims in general use, and the two builds known
# to spill (a 256-wide Q tile with KV staging on, which holds the most live at once).
CONFIGS = [
    dict(head_dim=64, causal=True),
    dict(head_dim=128, causal=True),
    dict(head_dim=128, causal=False),
    dict(head_dim=192, causal=True),
    dict(head_dim=128, causal=True, block_m=256, enable_kv_gpfetch=True),
    dict(head_dim=192, causal=True, block_m=256, enable_kv_gpfetch=True),
]


def isa_stats():
    """Register counts from the most recently dumped ISA, or -1 where absent."""
    dumps = sorted(
        glob.glob(os.path.join(DUMP_DIR, "**", "*.s"), recursive=True),
        key=os.path.getmtime,
    )
    if not dumps:
        return (-1, -1, -1, -1)
    with open(dumps[-1]) as fh:
        asm = fh.read()

    def grab(pattern):
        match = re.search(pattern, asm)
        return int(match.group(1)) if match else -1

    return (
        grab(r"\.vgpr_count:\s*(\d+)"),
        grab(r"\.agpr_count:\s*(\d+)"),
        grab(r"\.vgpr_spill_count:\s*(\d+)"),
        len(re.findall(r"\bds_read", asm)),
    )


def measure(config):
    # A stale dump from the previous config would be read as this one's.
    shutil.rmtree(DUMP_DIR, ignore_errors=True)
    os.makedirs(DUMP_DIR, exist_ok=True)

    q, k, v = (
        torch.randn(
            BATCH,
            HEADS,
            SEQ_LEN,
            config["head_dim"],
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    )
    # Dumping is per-stage and chatty, and it prints a line per stage to stdout, which
    # would land in the middle of the table. The warnings it writes to stderr are kept.
    with contextlib.redirect_stdout(io.StringIO()):
        launcher = build_flex_flash_generic_module(
            num_heads=HEADS, dtype_str="bf16", layout="bhsd", **config
        )
        run, _, _ = prepare(launcher, q, k, v, out=torch.empty_like(q))
        us = time_launcher_us(run)
    return us, isa_stats()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--waves", action="store_true", help="also build each config at 1 wave/EU"
    )
    args = parser.parse_args()

    print(f"flydsl {flydsl.__version__}, B={BATCH} H={HEADS} S={SEQ_LEN}")
    print(
        f"{'D':>4}{'causal':>8}{'BM':>6}{'stage':>7}{'wpe':>5}{'vgpr':>6}{'agpr':>6}"
        f"{'spill':>7}{'ds_read':>9}{'us':>9}{'TF/s':>8}"
    )
    for config in CONFIGS:
        for waves_per_eu in (2, 1) if args.waves else (None,):
            build = dict(config)
            if waves_per_eu is not None:
                build["waves_per_eu"] = waves_per_eu
            us, (vgpr, agpr, spill, ds_read) = measure(build)
            flops = 2 * 2 * BATCH * HEADS * SEQ_LEN * SEQ_LEN * config["head_dim"]
            flops *= 0.5 if config["causal"] else 1.0
            print(
                f"{config['head_dim']:>4}{str(config['causal']):>8}"
                f"{str(config.get('block_m')):>6}"
                f"{str(config.get('enable_kv_gpfetch')):>7}"
                f"{str(waves_per_eu):>5}{vgpr:>6}{agpr:>6}{spill:>7}{ds_read:>9}"
                f"{us:>9.1f}{flops / us / 1e6:>8.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

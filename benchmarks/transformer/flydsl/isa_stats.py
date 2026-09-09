"""Registers, spills and time for a few builds, read out of the generated ISA.

Compiles each config with ASM dumping on and greps the metadata the assembler emits, so
the register pressure story is measured rather than inferred. Used to price anything that
changes what the compiler has to hold live: a FlyDSL version bump, a tile shape, or the
occupancy request itself.

`--waves` adds the 1-wave/EU build alongside the default 2. Asking for 2 caps the register
budget and is what makes head_dim 128 spill; relaxing it does clear the spills, and at the
shapes here it costs more throughput than the spills do, which is why the default stands.

`--backward` reads the two backward kernels instead. That is the direction with a standing
`waves-per-eu` warning on `dkdv`, and 256 VGPRs is the cliff to watch: two waves per SIMD
need a wave to fit in half of the 512 the part has, and LLVM has no reason to stop at 256
when the occupancy request is already 1.

`--arch` compiles for another GPU instead of this one, which is how a machine with only
gfx942 in it says anything at all about gfx950. The arch reaches the kernels through
`FLYDSL_GPU_ARCH` and decides real instruction selection -- MFMA K=16, the transposing LDS
read, the permlane O store, DMA-to-LDS -- so the ISA read back is the ISA that part would
run. Register pressure and spills are what this can honestly report cross-arch; there is
no timing column, because the binary cannot execute here.

That mode runs one config per subprocess, and the reason is worth stating because it looks
like over-engineering. FlyDSL has no compile-without-dispatch entry point: `flyc.compile`
launches the kernel once on the way to returning a fast callable. For a foreign target that
dispatch fails -- by which point the assembler has already written the ISA, so the numbers
are there -- but it fails by invalidating the HIP context, and every later GPU call in the
process then dies with `invalid resource handle`. A child per config confines that to the
child.

    python benchmarks/transformer/flydsl/isa_stats.py
    python benchmarks/transformer/flydsl/isa_stats.py --waves
    python benchmarks/transformer/flydsl/isa_stats.py --backward
    python benchmarks/transformer/flydsl/isa_stats.py --arch gfx950
    python benchmarks/transformer/flydsl/isa_stats.py --arch gfx950 --backward
"""

import argparse
import contextlib
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
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

from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_bwd_generic import (
    build_flex_flash_bwd_dkdv_module,
    build_flex_flash_bwd_dq_module,
)
from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_generic import (
    build_flex_flash_generic_module,
)
from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_interface import (
    flex_flash_bwd_dkdv_bhsd,
    flex_flash_bwd_dq_bhsd,
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

# The backward's own corners: the two head dims in general use, the widest dkdv tile the
# sweep can pick (dkdv only -- `dq`'s block_n is its KV walk and 256 does not fit), and the
# LSE-gradient build, which holds a third row plane live alongside lse and delta.
BWD_CONFIGS = [
    dict(head_dim=64),
    dict(head_dim=128),
    dict(head_dim=128, dkdv_block_n=256),
    dict(head_dim=128, has_dlse=True),
    dict(head_dim=256),
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


def _compile_for_isa(run):
    """Drive one build far enough to write ISA, for a target we cannot execute.

    There is no compile-only entry point: the launch is how compilation is reached. On a
    foreign target it fails, and it does so *after* the assembler has run, so the dump is
    complete by the time the error arrives. The HIP context does not survive it, which is
    why this only ever runs in a child process.
    """
    try:
        run()
    except Exception:  # noqa: BLE001 - a foreign-arch dispatch is expected to fail
        pass
    return isa_stats()


def measure(config, launch=True):
    # A stale dump from the previous config would be read as this one's.
    shutil.rmtree(DUMP_DIR, ignore_errors=True)
    os.makedirs(DUMP_DIR, exist_ok=True)

    # Dumping is per-stage and chatty, and it prints a line per stage to stdout, which
    # would land in the middle of the table. The warnings it writes to stderr are kept.
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
    with contextlib.redirect_stdout(io.StringIO()):
        launcher = build_flex_flash_generic_module(
            num_heads=HEADS, dtype_str="bf16", layout="bhsd", **config
        )
        run, _, _ = prepare(launcher, q, k, v, out=torch.empty_like(q))
        if not launch:
            return None, _compile_for_isa(run)
        us = time_launcher_us(run)
    return us, isa_stats()


def _label(config, which):
    """What distinguishes this build from the plain one at the same head_dim."""
    if config.get("has_dlse"):
        return "dlse"
    if which == "dkdv" and "dkdv_block_n" in config:
        return f"n{config['dkdv_block_n']}"
    return "-"


def measure_backward(which, config, launch=True):
    """One backward kernel, built and timed on its own like the forward ones."""
    shutil.rmtree(DUMP_DIR, ignore_errors=True)
    os.makedirs(DUMP_DIR, exist_ok=True)

    head_dim = config["head_dim"]
    build = {
        k_: v_ for k_, v_ in config.items() if k_ not in ("head_dim", "dkdv_block_n")
    }
    if which == "dkdv" and "dkdv_block_n" in config:
        build["block_n"] = config["dkdv_block_n"]

    q, k, v, do = (
        torch.randn(
            BATCH, HEADS, SEQ_LEN, head_dim, device="cuda", dtype=torch.bfloat16
        )
        for _ in range(4)
    )
    dq, dk, dv = (torch.empty_like(q) for _ in range(3))
    # Real values rather than zeros: `lse` reaches an exp2 and a zero row would make every
    # probability 1, which is not a live set the kernel ever holds.
    lse, delta = (
        torch.randn(BATCH, HEADS, SEQ_LEN, device="cuda", dtype=torch.float32)
        for _ in range(2)
    )

    dlse = (
        torch.randn(BATCH, HEADS, SEQ_LEN, device="cuda", dtype=torch.float32)
        if config.get("has_dlse")
        else None
    )
    with contextlib.redirect_stdout(io.StringIO()):
        if which == "dq":
            launcher = build_flex_flash_bwd_dq_module(
                num_heads=HEADS,
                head_dim=head_dim,
                dtype_str="bf16",
                layout="bhsd",
                **build,
            )

            def run():
                flex_flash_bwd_dq_bhsd(
                    launcher, q, k, v, do, lse, delta, dq=dq, dlse=dlse
                )
        else:
            launcher = build_flex_flash_bwd_dkdv_module(
                num_heads=HEADS,
                head_dim=head_dim,
                dtype_str="bf16",
                layout="bhsd",
                **build,
            )

            def run():
                flex_flash_bwd_dkdv_bhsd(
                    launcher, q, k, v, do, lse, delta, dk=dk, dv=dv, dlse=dlse
                )

        if not launch:
            return None, _compile_for_isa(run)
        us = time_launcher_us(run)
    return us, isa_stats()


def _jobs(args):
    """Every build this invocation will make, flattened so ``--one`` can index it."""
    if args.backward:
        return [
            (which, config) for config in BWD_CONFIGS for which in ("dq", "dkdv")
        ]
    jobs = []
    for config in CONFIGS:
        for waves_per_eu in (2, 1) if args.waves else (None,):
            build = dict(config)
            if waves_per_eu is not None:
                build["waves_per_eu"] = waves_per_eu
            jobs.append((build, waves_per_eu))
    return jobs


def _run_job(args, job, launch):
    if args.backward:
        which, config = job
        return measure_backward(which, config, launch=launch)
    build, _ = job
    return measure(build, launch=launch)


def _stats_from_child(args, index):
    """One job's ISA numbers, measured in a child process.

    The child is what makes a foreign target survivable; see the module docstring. Its
    stdout carries one JSON line, and its exit status is ignored -- a failed dispatch is
    the expected end of a cross-arch build, so what matters is whether the numbers came
    back, not how the process finished.
    """
    cmd = [sys.executable, __file__, "--arch", args.arch, "--one", str(index)]
    if args.backward:
        cmd.append("--backward")
    if args.waves:
        cmd.append("--waves")
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    for line in reversed(done.stdout.splitlines()):
        if line.startswith(_CHILD_PREFIX):
            return tuple(json.loads(line[len(_CHILD_PREFIX) :]))
    sys.stderr.write(
        f"child for job {index} reported nothing:\n{done.stdout}\n{done.stderr}\n"
    )
    return (-1, -1, -1, -1)


_CHILD_PREFIX = "ISA_JSON "


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--waves", action="store_true", help="also build each config at 1 wave/EU"
    )
    parser.add_argument(
        "--backward", action="store_true", help="read the two backward kernels instead"
    )
    parser.add_argument(
        "--arch",
        help="compile for this gfx target instead of the installed GPU (no timing)",
    )
    parser.add_argument(
        "--one",
        type=int,
        help=argparse.SUPPRESS,  # one job, reported as JSON; see `_stats_from_child`
    )
    args = parser.parse_args()

    host = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    target = args.arch or host
    launch = target == host
    if not launch:
        # Read by `get_rocm_arch`, which the kernel builders call, and passed on as the
        # compilation target. Set here rather than at import: nothing reads it until a
        # build starts.
        os.environ["FLYDSL_GPU_ARCH"] = target

    jobs = _jobs(args)

    if args.one is not None:
        _, stats = _run_job(args, jobs[args.one], launch)
        print(f"{_CHILD_PREFIX}{json.dumps(list(stats))}", flush=True)
        return

    print(f"flydsl {flydsl.__version__}, B={BATCH} H={HEADS} S={SEQ_LEN}, arch {target}")
    if not launch:
        print(f"built for {target} on {host}; not launched, so no timing")
    us_col = f"{'us':>9}" if launch else f"{'':>9}"

    def us_cell(us):
        return f"{us:>9.1f}" if us is not None else f"{'-':>9}"

    def stats_for(index, job):
        if launch:
            return _run_job(args, job, launch)
        return None, _stats_from_child(args, index)

    if args.backward:
        print(
            f"{'kernel':>7}{'D':>5}{'variant':>9}{'vgpr':>6}{'agpr':>6}{'spill':>7}"
            f"{'ds_read':>9}{us_col}"
        )
        for index, job in enumerate(jobs):
            which, config = job
            us, (vgpr, agpr, spill, ds_read) = stats_for(index, job)
            print(
                f"{which:>7}{config['head_dim']:>5}"
                f"{_label(config, which):>9}{vgpr:>6}{agpr:>6}{spill:>7}"
                f"{ds_read:>9}{us_cell(us)}",
                flush=True,
            )
        return

    print(
        f"{'D':>4}{'causal':>8}{'BM':>6}{'stage':>7}{'wpe':>5}{'vgpr':>6}{'agpr':>6}"
        f"{'spill':>7}{'ds_read':>9}{us_col}{'TF/s' if launch else '':>8}"
    )
    for index, job in enumerate(jobs):
        build, waves_per_eu = job
        us, (vgpr, agpr, spill, ds_read) = stats_for(index, job)
        flops = 2 * 2 * BATCH * HEADS * SEQ_LEN * SEQ_LEN * build["head_dim"]
        flops *= 0.5 if build["causal"] else 1.0
        tf = f"{flops / us / 1e6:>8.1f}" if us is not None else f"{'-':>8}"
        print(
            f"{build['head_dim']:>4}{str(build['causal']):>8}"
            f"{str(build.get('block_m')):>6}"
            f"{str(build.get('enable_kv_gpfetch')):>7}"
            f"{str(waves_per_eu):>5}{vgpr:>6}{agpr:>6}{spill:>7}{ds_read:>9}"
            f"{us_cell(us)}{tf}",
            flush=True,
        )


if __name__ == "__main__":
    main()

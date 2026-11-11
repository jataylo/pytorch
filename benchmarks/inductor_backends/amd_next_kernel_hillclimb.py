import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path


os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import torch


SOURCE_ROOT = Path(
    os.environ.get(
        "AMD_CODEGEN_SOURCE_ROOT", Path(torch.__file__).resolve().parent.parent
    )
).resolve()
sys.path.insert(0, str(SOURCE_ROOT / "benchmarks/inductor_backends"))
import amd_benchmark_protocol as gpu_protocol
import amd_extended_kernel_hillclimb as previous


base = previous.base
base.measure = gpu_protocol.measure
FAMILIES = (
    "adamw",
    "cross_entropy",
    "cross_entropy_backward",
    "cumsum",
    "gather",
    "scatter_add",
    "batch_norm",
    "group_norm",
    "batch_norm_backward",
    "group_norm_backward",
)
VARIANTS = ("triton", "v15_triton", "optimized_triton", "flydsl")
previous_workload = base.workload
previous_validate = base.validate
CURRENT_FAMILY = None


def validate(actual, expected, exact=False):
    if CURRENT_FAMILY != "cumsum":
        return previous_validate(actual, expected, exact)
    from torch.testing._internal.common_utils import TestCase

    TestCase().assertEqual(actual, expected, atol=0.0002, rtol=0.0001, equal_nan=True)
    return (actual - expected).abs().nan_to_num().max().item()


base.validate = validate


def workload(case, seed):
    global CURRENT_FAMILY
    CURRENT_FAMILY = case["family"]
    torch.manual_seed(seed)
    family, shape, dtype = (
        case["family"],
        case["shape"],
        getattr(torch, case["dtype"]),
    )
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if family == "adamw":
        g = torch.randn_like(x)
        m = torch.randn_like(x, dtype=torch.float32)
        v = torch.rand_like(m)

        def adamw(p, g, m, v):
            mean = m * 0.9 + g.float() * 0.1
            variance = v * 0.999 + g.float().square() * 0.001
            return (
                (p.float() * 0.9999 - 0.001 * mean / (torch.sqrt(variance) + 1e-08)).to(
                    p.dtype
                ),
                mean,
                variance,
            )

        return (adamw, (x, g, m, v))
    if family == "cross_entropy":
        t = torch.randint(0, shape[-1], (shape[0],), device="cuda")
        return (
            lambda x, t: torch.nn.functional.cross_entropy(
                x.float(), t, reduction="none"
            ),
            (x, t),
        )
    if family == "cross_entropy_backward":
        t = torch.randint(0, shape[-1], (shape[0],), device="cuda")
        return (
            lambda x, t: torch.softmax(x.float(), -1)
            - torch.nn.functional.one_hot(t, x.shape[-1]).float(),
            (x, t),
        )
    if family == "cumsum":
        return (lambda x: torch.cumsum(x.float(), dim=-1), (x,))
    if family == "gather":
        i = torch.randint(
            0, shape[0], (case.get("index_count", shape[0] * 2),), device="cuda"
        )
        return (lambda x, i: torch.nn.functional.embedding(i, x), (x, i))
    if family == "scatter_add":
        i = torch.randint(0, shape[0] // 2, (shape[0],), device="cuda")

        def scatter_add(x, i):
            out = torch.zeros(
                (x.shape[0] // 2, x.shape[1]), device=x.device, dtype=torch.float32
            )
            return out.index_add(0, i, x.float())

        return scatter_add, (x, i)
    if family == "batch_norm_backward":
        if case.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        _, mean, invstd = torch.ops.aten.native_batch_norm(
            x.float(), None, None, None, None, True, 0.1, 1e-05
        )
        g = torch.randn_like(x)
        w = torch.ones(shape[1], device="cuda")

        def batch_norm_backward(g, x, w, mean, rstd):
            return torch.ops.aten.native_batch_norm_backward(
                g.float(),
                x.float(),
                w,
                None,
                None,
                mean,
                rstd,
                True,
                1e-5,
                [True, True, True],
            )

        return batch_norm_backward, (g, x, w, mean, invstd)
    if family == "group_norm_backward":
        if case.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        n, c, hh, ww = shape
        groups = case.get("groups", 32)
        _, mean, rstd = torch.ops.aten.native_group_norm(
            x.float(), None, None, n, c, hh * ww, groups, 1e-05
        )
        g = torch.randn_like(x)
        w = torch.ones(c, device="cuda")

        def group_norm_backward(g, x, w, mean, rstd):
            return torch.ops.aten.native_group_norm_backward(
                g.float(),
                x.float(),
                mean,
                rstd,
                w,
                n,
                c,
                hh * ww,
                groups,
                [True, True, True],
            )

        return group_norm_backward, (g, x, w, mean, rstd)
    if family == "batch_norm":
        if case.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        return (
            lambda x: torch.ops.aten.native_batch_norm(
                x.float(), None, None, None, None, True, 0.1, 1e-05
            ),
            (x,),
        )
    if family == "group_norm":
        if case.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        return (
            lambda x: torch.nn.functional.group_norm(x.float(), case.get("groups", 32)),
            (x,),
        )
    return previous_workload(case, seed)


def patch(variant, path):
    from torch._inductor import config

    optimized = variant in ("v15_triton", "optimized_triton")
    return config.patch(
        {
            "triton.amd_kernel_optimizations": optimized,
            "triton.amd_extended_kernel_optimizations": optimized,
            "triton.amd_further_kernel_optimizations": variant == "optimized_triton",
            "triton.amd_approximate_math": False,
            "triton.amd_autotune_diagnostics_path": str(
                path.with_name(path.stem + ".tuning.jsonl")
            ),
            "autotune_local_cache": False,
            "flydsl.simd": variant == "flydsl",
            "flydsl.simd_autotune_backend": False,
            "flydsl.simd_diagnostics_path": str(path),
            "triton.cudagraphs": False,
            "max_autotune": False,
            "max_autotune_pointwise": True,
            "triton.autotune_pointwise": True,
            "max_autotune_gemm_backends": "ATEN",
            "max_autotune_conv_backends": "ATEN",
            "fx_graph_cache": False,
        }
    )


base.workload = workload
base.patch = patch
base.VARIANTS = VARIANTS
old_fingerprint = base.source_fingerprint


def fingerprint():
    sources = [
        Path(__file__).resolve(),
        Path(gpu_protocol.__file__).resolve(),
        *sorted((SOURCE_ROOT / "torch/_inductor").rglob("*.py")),
    ]
    return old_fingerprint() | {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


base.source_fingerprint = fingerprint


def cases(quick=False):
    shapes_by_family = {
        "adamw": [(4099,), (65537,), (4608000,), (41943040,)],
        "cross_entropy": [(37, 127), (256, 1024), (32, 32768), (8, 128112)],
        "cross_entropy_backward": [(37, 127), (256, 1024), (32, 32768), (8, 128112)],
        "cumsum": [(37, 127), (256, 1024), (1024, 8192), (64, 32768)],
        "gather": [(1024, 64), (4096, 128), (32768, 128), (16384, 1024)],
        "scatter_add": [(1024, 64), (4096, 128), (32768, 128), (16384, 1024)],
    }
    norm_shapes = [(2, 32, 7, 7), (8, 128, 32, 32), (16, 256, 16, 16), (8, 128, 32, 32)]
    result = []
    for family in FAMILIES:
        shapes = shapes_by_family.get(family, norm_shapes)
        for dtype in ("bfloat16", "float32"):
            for index, shape in enumerate(shapes[:1] if quick else shapes):
                case = dict(
                    family=family,
                    dtype=dtype,
                    shape=shape,
                    case=f"{family}_{dtype}_{index}",
                )
                if family not in shapes_by_family and index == 3:
                    case["channels_last"] = True
                result.append(case)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--case-manifest", type=Path)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error("rounds and repeats must be positive")
    families = args.families.split(",")
    if not args.case_manifest and set(families) - set(FAMILIES):
        parser.error(
            "Unknown families: " + ", ".join(sorted(set(families) - set(FAMILIES)))
        )
    selected = (
        json.loads(args.case_manifest.read_text())
        if args.case_manifest
        else [case for case in cases(args.quick) if case["family"] in families]
    )
    selected = json.loads(json.dumps(selected))
    names = [case["case"] for case in selected]
    if len(names) != len(set(names)):
        parser.error("Case names must be unique")
    if any(Path(name).name != name or name in (".", "..") for name in names):
        parser.error("Case names must be single directory names")
    torch.set_num_threads(8)
    protocol = dict(
        rounds=args.rounds,
        repeats=args.repeats,
        variants=list(VARIANTS),
        measurement=gpu_protocol.MEASUREMENT_PROTOCOL,
    )
    sources = fingerprint()
    manifest = dict(
        cases=selected,
        **protocol,
        source_fingerprint=sources,
        cumsum_validation_atol=2e-4,
        cumsum_validation_rtol=1e-4,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (
        open("/dev/shm/pytorch-simd-v2-entry.lock", "a") as entry,
        open("/dev/shm/pytorch-simd-v2-benchmark.lock", "a") as timing,
    ):
        fcntl.flock(entry, fcntl.LOCK_EX)
        fcntl.flock(timing, fcntl.LOCK_EX)
        manifest_path = args.output / "manifest.json"
        existing = list(args.output.glob("*/result.json"))
        if existing and not manifest_path.exists():
            raise RuntimeError(
                "Existing results have no protocol manifest: use a fresh output directory"
            )
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError(
                "Source or benchmark protocol changed: use a fresh output directory"
            )
        if any(path.parent.name not in names for path in existing):
            raise RuntimeError(
                "Output contains unselected cases: use a fresh output directory"
            )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        for case in selected:
            result_path = args.output / case["case"] / "result.json"
            if result_path.exists():
                old = json.loads(result_path.read_text())
                if old.get("source_fingerprint") != sources or old.get("case") != case:
                    raise RuntimeError(
                        "Per-case source or inputs changed: use a fresh output directory"
                    )
                if old.get("benchmark_protocol") != protocol:
                    raise RuntimeError(
                        "Per-case protocol changed: use a fresh output directory"
                    )
            else:
                print("RUN", case, flush=True)
                result = base.run_case(case, args.output, args.rounds, args.repeats)
                if fingerprint() != sources:
                    raise RuntimeError(
                        "Source changed during benchmark: use a fresh output directory"
                    )
                result["benchmark_protocol"] = protocol
                for row in result["rows"]:
                    row["measured_invocation"] = gpu_protocol.INVOCATION_PROTOCOL[
                        row["invocation"]
                    ]
                    row["host_overhead_included"] = gpu_protocol.HOST_OVERHEAD_INCLUDED[
                        row["invocation"]
                    ]
                result_path.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    "DONE",
                    case["case"],
                    "failures",
                    len(result["failures"]),
                    flush=True,
                )
            base.export(args.output)
        if fingerprint() != sources:
            raise RuntimeError(
                "Source changed during benchmark: use a fresh output directory"
            )


if __name__ == "__main__":
    main()

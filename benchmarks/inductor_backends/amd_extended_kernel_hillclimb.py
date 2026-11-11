import argparse
import fcntl
import hashlib
import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_kernel_hillclimb as base


FAMILIES = (
    "dropout_backward",
    "silu_gate",
    "silu_backward",
    "gelu_forward",
    "layer_norm",
    "rms_norm",
    "log_softmax",
)
VARIANTS = ("triton", "v14_triton", "optimized_triton", "flydsl")
original_workload = base.workload


def workload(case, seed):
    torch.manual_seed(seed)
    family, shape, dtype = case["family"], case["shape"], getattr(torch, case["dtype"])
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if family == "dropout_backward":
        return lambda x, m: (x.float() * m.float() * 1.1111111111111112).to(x.dtype), (
            x,
            torch.rand(shape, device="cuda") > 0.1,
        )
    if family == "silu_gate":
        return lambda x, y: (torch.nn.functional.silu(x.float()) * y.float()).to(
            x.dtype
        ), (x, torch.randn_like(x))
    if family == "silu_backward":
        return lambda x, g: torch.ops.aten.silu_backward(g, x), (x, torch.randn_like(x))
    if family == "gelu_forward":
        return lambda x: torch.nn.functional.gelu(x, approximate="none"), (x,)
    if family == "layer_norm":
        weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
        bias = torch.randn_like(weight)
        return lambda x, w, b: torch.native_layer_norm(x, (x.shape[-1],), w, b, 1e-5), (
            x,
            weight,
            bias,
        )
    if family == "rms_norm":
        weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32)

        def rms(x, w):
            r = torch.rsqrt(x.float().square().mean(-1, True) + 1e-6)
            return ((x.float() * r) * w).to(x.dtype), r

        return rms, (x, weight)
    if family == "log_softmax":
        return lambda x: torch.log_softmax(x.float(), -1), (x,)
    return original_workload(case, seed)


original_fingerprint = base.source_fingerprint


def fingerprint():
    return original_fingerprint() | {
        str(Path(__file__).resolve()): hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
    }


base.source_fingerprint = fingerprint


def patch(variant, path):
    from torch._inductor import config

    return config.patch(
        {
            "triton.amd_kernel_optimizations": variant
            in ("v14_triton", "optimized_triton"),
            "triton.amd_extended_kernel_optimizations": variant == "optimized_triton",
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shapes_point = [(4099,), (65537,), (4608000,), (41943040,)]
    shapes_reduce = [(37, 127), (256, 1024), (8192, 256), (1024, 8192), (256, 16384)]
    torch.set_num_threads(8)
    with open("/dev/shm/pytorch-simd-v2-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (args.output / "manifest.json").write_text(
            json.dumps(
                dict(
                    variants=VARIANTS,
                    rounds=args.rounds,
                    repeats=args.repeats,
                    device=torch.cuda.get_device_name(),
                    source_fingerprint=fingerprint(),
                    approximate_math=False,
                ),
                indent=2,
            )
        )
        selected = []
        if args.case_manifest:
            selected = json.loads(args.case_manifest.read_text())
        else:
            for family in args.families.split(","):
                if family not in FAMILIES:
                    parser.error("Unknown family: " + family)
                shapes = (
                    shapes_reduce
                    if family in ("layer_norm", "rms_norm", "log_softmax")
                    else shapes_point
                )
                for dtype in ("bfloat16", "float32"):
                    for index, shape in enumerate(shapes[:1] if args.quick else shapes):
                        selected.append(
                            dict(
                                family=family,
                                dtype=dtype,
                                shape=shape,
                                case=f"{family}_{dtype}_{index}",
                            )
                        )
        for case in selected:
            path = args.output / case["case"] / "result.json"
            if path.exists():
                previous = json.loads(path.read_text())
                if previous["source_fingerprint"] != fingerprint():
                    raise RuntimeError("Source changed: use a fresh output directory")
            else:
                print("RUN", case, flush=True)
                result = base.run_case(case, args.output, args.rounds, args.repeats)
                print(
                    "DONE",
                    case["case"],
                    "failures",
                    len(result["failures"]),
                    flush=True,
                )
            base.export(args.output)

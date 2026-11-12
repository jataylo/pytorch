"""Compare compiled AMD kernel families with optional wave64 optimizations."""

import argparse
import csv
import fcntl
import hashlib
import inspect
import json
import math
import statistics
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import TestCase


FAMILIES = (
    "mean",
    "softmax_backward",
    "transpose",
    "norm_backward",
    "broadcast",
    "softmax",
    "cast_add",
    "gelu_backward",
)
VARIANTS = ("triton", "optimized_triton", "flydsl", "optimized_triton_approx")


def cases(quick=False):
    reductions = [
        (37, 64),
        (256, 128),
        (8192, 256),
        (1024, 1536),
        (4096, 1024),
        (1024, 8192),
    ]
    transposes = [
        (2, 3, 37, 64),
        (1, 8, 128, 64),
        (16, 32, 128, 80),
        (32, 32, 128, 80),
        (12, 12, 577, 64),
        (8, 16, 1024, 64),
    ]
    sizes = [4099, 65537, 4608000, 41943040, 102410240, 268435456]
    means = [
        (37, 127),
        (256, 128),
        (4096, 1024),
        (4096, 32768),
        (4096, 256008),
        (16384, 50265),
    ]
    result = []
    for family in FAMILIES:
        shapes = (
            transposes
            if family == "transpose"
            else means
            if family == "mean"
            else [(n,) for n in sizes]
            if family in ("broadcast", "cast_add", "gelu_backward")
            else reductions
        )
        for dtype in (torch.bfloat16, torch.float32):
            for index, shape in enumerate(shapes[:1] if quick else shapes):
                result.append(
                    dict(
                        family=family,
                        dtype=str(dtype).split(".")[-1],
                        shape=shape,
                        case=f"{family}_{str(dtype).split('.')[-1]}_{index}",
                    )
                )
    return result


def workload(case, seed):
    torch.manual_seed(seed)
    family, shape, dtype = case["family"], case["shape"], getattr(torch, case["dtype"])
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if family == "mean":
        return lambda x: x.float().square().mean(), (x,)
    if family == "transpose":
        if len(shape) == 3:
            heads = case["transpose_heads"]
            return lambda x: x.view(-1, heads, shape[-2], shape[-1]).permute(
                0, 2, 1, 3
            ).contiguous(), (x,)
        return lambda x: x.permute(0, 2, 1, 3).contiguous(), (x,)
    if family == "norm_parameters":
        h = torch.randn_like(x, dtype=torch.float32)
        keep = torch.rand(shape, device="cuda") > 0.1

        def norm_parameters(g, h, keep):
            value = g.float() * (keep.float() * 1.1111111111111112)
            return (value * h).sum(1).t(), value.sum(1).t()

        return norm_parameters, (x, h, keep)
    if family == "broadcast":
        scalar = torch.tensor(0.75, device="cuda")
        return lambda x, s: ((s * (1.0 / x.numel())) * (x.float() * 2.0)).to(x.dtype), (
            x,
            scalar,
        )
    if family == "cast_add":
        return lambda x, y: x.float() + y.float(), (x, torch.randn_like(x))
    if family == "gelu_backward_inplace":

        def gelu_backward_inplace(x, g):
            g.copy_(torch.ops.aten.gelu_backward(g, x, approximate="none"))
            return g

        return gelu_backward_inplace, (x, torch.randn_like(x))
    if family == "gelu_backward":
        return lambda x, g: torch.ops.aten.gelu_backward(g, x, approximate="none"), (
            x,
            torch.randn_like(x),
        )
    if family == "softmax_saved_stats":
        from torch._inductor import inductor_prims

        keep = torch.rand((shape[0], 1, shape[2], shape[3]), device="cuda") > 0.1
        keep[:, :, 0] = False

        def saved_softmax(x, keep):
            value = x + torch.where(keep, 0.0, -float("inf"))
            maximum, total = inductor_prims.prepare_softmax_online(value, -1)
            invalid = (value == -float("inf")).all(-1, keepdim=True)
            probability = torch.where(invalid, 0.0, torch.exp(value - maximum) / total)
            return invalid, maximum, total, probability

        return saved_softmax, (x, keep)
    if family == "softmax":
        keep = torch.rand(shape, device="cuda") > 0.1
        return lambda x, keep: torch.softmax(
            x.float().masked_fill(~keep, -float("inf")), -1
        ), (x, keep)
    if family == "softmax_backward":
        probability = torch.softmax(x.float(), -1)
        grad = torch.randn_like(probability)
        keep = torch.rand(shape, device="cuda") > 0.1

        def softmax_backward(p, g, mask):
            product = (g * (mask.float() * 1.1111111111111112)) * p
            return torch.addcmul(product, -p, product.sum(-1, keepdim=True))

        return softmax_backward, (probability, grad, keep)
    gamma = torch.randn(shape[-1], device="cuda")
    scale = torch.rand(shape[0], 1, device="cuda") / shape[-1]
    h = torch.randn(shape, device="cuda")
    keep = torch.rand(shape, device="cuda") > 0.1

    def norm_backward(g, weight, h, scale, keep):
        v = g.float() * weight
        value = scale * (
            (v * g.shape[-1] - v.sum(-1, keepdim=True))
            - h * (v * h).sum(-1, keepdim=True)
        )
        return value, (value * (keep.float() * 1.1111111111111112)).to(g.dtype)

    return norm_backward, (x, gamma, h, scale, keep)


def patch(variant, diagnostics):
    return config.patch(
        {
            "triton.amd_kernel_optimizations": variant
            in ("optimized_triton", "optimized_triton_approx"),
            "triton.amd_approximate_math": variant == "optimized_triton_approx",
            "triton.amd_autotune_diagnostics_path": str(
                diagnostics.with_name(diagnostics.stem + ".tuning.jsonl")
            ),
            "autotune_local_cache": False,
            "flydsl.simd": variant == "flydsl",
            "flydsl.simd_autotune_backend": False,
            "flydsl.simd_diagnostics_path": str(diagnostics),
            "triton.cudagraphs": False,
            "max_autotune": False,
            "max_autotune_pointwise": True,
            "triton.autotune_pointwise": True,
            "max_autotune_gemm_backends": "ATEN",
            "max_autotune_conv_backends": "ATEN",
            "fx_graph_cache": False,
        }
    )


def validate(actual, expected, exact=False):
    left = actual if isinstance(actual, tuple) else (actual,)
    right = expected if isinstance(expected, tuple) else (expected,)
    errors = []
    for a, b in zip(left, right):
        TestCase().assertEqual(
            a,
            b,
            atol=0
            if exact
            else 0.02
            if a.dtype in (torch.bfloat16, torch.float16)
            else 1e-4,
            rtol=0
            if exact
            else 0.02
            if a.dtype in (torch.bfloat16, torch.float16)
            else 1e-4,
            equal_nan=True,
        )
        errors.append((a.float() - b.float()).abs().nan_to_num().max().item())
    return max(errors)


def measure(entries, inputs, rounds, repeats, cache, invocation, originals):
    flush = torch.empty(256 << 20, device="cuda", dtype=torch.uint8)
    warm = [
        torch.empty(0, device="cuda", dtype=torch.uint8).set_(x.untyped_storage())
        for x in inputs
    ]
    events = [
        [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(repeats)
        ]
        for _ in entries
    ]
    samples = [[] for _ in entries]
    for index in range(rounds):
        torch.cuda._sleep(40_000_000)
        for j in range(len(entries)):
            i = (index + j) % len(entries)
            for start, end in events[i]:
                flush.zero_()
                for input_index, original in originals.items():
                    inputs[input_index].copy_(original)
                if cache == "warm":
                    for value in warm:
                        value.add_(0)
                start.record()
                if invocation == "graph_replay":
                    entries[i]["graph"].replay()
                else:
                    entries[i]["fn"](*inputs)
                end.record()
        torch.cuda.synchronize()
        for i, pairs in enumerate(events):
            samples[i].append(
                statistics.mean(start.elapsed_time(end) * 1000 for start, end in pairs)
            )
    return samples


def reference_output(fn, inputs, family):
    """Build eager gold outside compilation and timing; scans accumulate exact inputs in FP64."""
    value = fn(*inputs)
    if family == "cumsum":
        value = torch.cumsum(inputs[0].to(torch.float64), dim=-1).to(value.dtype)
    return value


def run_case(case, output, rounds, repeats):
    folder = output / case["case"]
    folder.mkdir(exist_ok=True)
    fn, inputs = workload(case, 2026)
    originals = {
        index: inputs[index].clone() for index in case.get("mutated_input_indices", [])
    }

    def restore():
        for index, original in originals.items():
            inputs[index].copy_(original)

    def reference():
        restore()
        value = reference_output(fn, inputs, case["family"])
        return (
            tuple(item.clone() for item in value)
            if isinstance(value, (tuple, list))
            else value.clone()
        )

    expected = reference()
    entries, failures = [], []
    for variant in VARIANTS:
        torch._dynamo.reset()
        start = time.monotonic()
        try:
            with patch(variant, folder / f"{variant}.jsonl"):
                restore()
                records = []
                parameter_names = list(inspect.signature(fn).parameters)

                def backend(gm, example_inputs):
                    from torch._inductor.compile_fx import compile_fx

                    order = []
                    for node in gm.graph.nodes:
                        if node.op != "placeholder":
                            continue
                        grapharg = node.meta.get("grapharg")
                        local = getattr(
                            getattr(grapharg, "source", None), "local_name", None
                        )
                        if local is None:
                            name = str(node.target)
                            local = (
                                name[2:-1]
                                if name.startswith("L_") and name.endswith("_")
                                else name
                            )
                        if local not in parameter_names:
                            raise RuntimeError(
                                f"Cannot freeze compiled input {node.target}: {local}"
                            )
                        order.append(parameter_names.index(local))
                    lowered = compile_fx(gm, example_inputs)
                    records.append((lowered, tuple(order)))
                    return lowered

                compiled = torch.compile(
                    fn, backend=backend, fullgraph=True, dynamic=False
                )
                actual, code = run_and_get_code(compiled, *inputs)
                error = validate(
                    actual,
                    expected,
                    case["family"] in ("transpose", "broadcast", "cast_add"),
                )
                source = "\n".join(code)
                (folder / f"{variant}.py").write_text(source)
                if len(records) != 1:
                    raise RuntimeError(
                        f"Expected one compiled graph, got {len(records)}"
                    )
                lowered, order = records[0]
                returns_tuple = isinstance(expected, tuple)

                def frozen(
                    *args, _lowered=lowered, _order=order, _returns_tuple=returns_tuple
                ):
                    value = _lowered(*[args[index] for index in _order])
                    return (
                        tuple(value)
                        if _returns_tuple and isinstance(value, (tuple, list))
                        else value
                        if _returns_tuple
                        else value[0]
                        if isinstance(value, (tuple, list))
                        else value
                    )

                restore()
                validate(
                    frozen(*inputs),
                    expected,
                    case["family"] in ("transpose", "broadcast", "cast_add"),
                )
                for _ in range(3):
                    restore()
                    actual = frozen(*inputs)
                graph = torch.cuda.CUDAGraph()
                restore()
                with torch.cuda.graph(graph):
                    captured = frozen(*inputs)
                entries.append(
                    dict(
                        variant=variant,
                        fn=frozen,
                        graph=graph,
                        captured=captured,
                        max_abs_error=error,
                        compile_records=records,
                        compile_seconds=time.monotonic() - start,
                        explicit_layout_kernels=source.count("'use_gluon': True"),
                        approximate_math_kernels=source.count(
                            "'amd_approximate_math': True"
                        ),
                        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    )
                )
        except Exception:
            failures.append(dict(variant=variant, error=traceback.format_exc()))
            (folder / f"{variant}.error.txt").write_text(failures[-1]["error"])
    for seed in (1701, 2027):
        _, fresh = workload(case, seed)
        for dst, src in zip(inputs, fresh):
            dst.copy_(src)
        for index in originals:
            originals[index].copy_(fresh[index])
        expected = reference()
        for entry in entries:
            restore()
            entry["graph"].replay()
            torch.cuda.synchronize()
            validate(
                entry["captured"],
                expected,
                case["family"] in ("transpose", "broadcast", "cast_add"),
            )
    rows = []
    samples = {}
    for invocation in ("direct", "graph_replay"):
        for cache in ("cold", "warm"):
            for entry, values in zip(
                entries,
                measure(entries, inputs, rounds, repeats, cache, invocation, originals),
            ):
                samples[entry["variant"], cache, invocation] = values
                rows.append(
                    dict(
                        **case,
                        variant=entry["variant"],
                        cache=cache,
                        invocation=invocation,
                        runtime_us=statistics.median(values),
                        p10_us=sorted(values)[int(0.1 * (len(values) - 1))],
                        p90_us=sorted(values)[int(0.9 * (len(values) - 1))],
                        compile_seconds=entry["compile_seconds"],
                        max_abs_error=entry["max_abs_error"],
                        explicit_layout_kernels=entry["explicit_layout_kernels"],
                        approximate_math_kernels=entry["approximate_math_kernels"],
                        source_sha256=entry["source_sha256"],
                    )
                )
    if any(len(entry["compile_records"]) != 1 for entry in entries):
        raise RuntimeError(
            "A timed variant recompiled instead of using its frozen backend"
        )
    report = dict(
        case=case,
        rows=rows,
        frozen_backend_callbacks=[len(entry["compile_records"]) for entry in entries],
        source_fingerprint=source_fingerprint(),
        samples_us={
            variant + "_" + cache + "_" + invocation: values
            for (variant, cache, invocation), values in samples.items()
        },
        failures=failures,
    )
    (folder / "result.json").write_text(json.dumps(report, indent=2))
    del entries
    torch.cuda.empty_cache()
    return report


def export(output):
    results = [json.loads(p.read_text()) for p in sorted(output.glob("*/result.json"))]
    rows = [row for result in results for row in result["rows"]]
    if not rows:
        return
    by_case = {(r["case"], r["cache"], r["invocation"], r["variant"]): r for r in rows}
    for row in rows:
        baseline = by_case.get((row["case"], row["cache"], row["invocation"], "triton"))
        row["speedup_vs_triton_factor"] = (
            baseline["runtime_us"] / row["runtime_us"] if baseline else None
        )
        row["time_saved_vs_triton_percent"] = (
            100 * (1 - row["runtime_us"] / baseline["runtime_us"]) if baseline else None
        )
    with (output / "per_case.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(key for row in rows for key in row))
        )
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        if row["time_saved_vs_triton_percent"] is not None:
            groups[
                row["family"], row["variant"], row["cache"], row["invocation"]
            ].append(row)
    summary = [
        dict(
            family=family,
            variant=variant,
            cache=cache,
            invocation=invocation,
            cases=len(values),
            geomean_time_saved_percent=100
            * (
                1
                - math.exp(
                    statistics.mean(
                        math.log(1 - r["time_saved_vs_triton_percent"] / 100)
                        for r in values
                    )
                )
            ),
            minimum_time_saved_percent=min(
                r["time_saved_vs_triton_percent"] for r in values
            ),
            maximum_time_saved_percent=max(
                r["time_saved_vs_triton_percent"] for r in values
            ),
        )
        for (family, variant, cache, invocation), values in sorted(groups.items())
    ]
    with (output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    failures = [
        dict(case=result["case"], **failure)
        for result in results
        for failure in result["failures"]
    ]
    (output / "failures.json").write_text(json.dumps(failures, indent=2))
    for invocation in ("direct", "graph_replay"):
        folder = output / invocation
        folder.mkdir(exist_ok=True)
        with (folder / "per_case.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(dict.fromkeys(key for row in rows for key in row))
            )
            writer.writeheader()
            writer.writerows([row for row in rows if row["invocation"] == invocation])
        (folder / "failures.json").write_text(json.dumps(failures, indent=2))


def source_fingerprint():
    import torch._inductor.codegen.amd_kernel_optimizations as lowering
    import torch._inductor.codegen.triton as codegen
    import torch._inductor.runtime.amd_autotune as tuning
    import torch._inductor.runtime.triton_heuristics as heuristics

    return {
        str(Path(module.__file__).resolve()): hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
        for module in (lowering, codegen, tuning, heuristics, config)
    } | {
        str(Path(__file__).resolve()): hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    selected = (
        json.loads(args.case_manifest.read_text())
        if args.case_manifest
        else [
            case
            for case in cases(args.quick)
            if case["family"] in args.families.split(",")
        ]
    )
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                cases=selected,
                variants=VARIANTS,
                rounds=args.rounds,
                repeats=args.repeats,
                device=torch.cuda.get_device_name(),
                torch_version=torch.__version__,
                source_fingerprint=source_fingerprint(),
                protocol="Frozen compile_fx callable plus CUDA graph replay,256MiBflush,cold/warm inputs; correctness on3seeds;ATENGEMM/conv;approximate math separately labelled",
            ),
            indent=2,
        )
    )
    with (
        open("/dev/shm/pytorch-simd-v2-entry.lock", "a") as entry,
        open("/dev/shm/pytorch-simd-v2-benchmark.lock", "a") as timing,
    ):
        fcntl.flock(entry, fcntl.LOCK_EX)
        fcntl.flock(timing, fcntl.LOCK_EX)
        for case in selected:
            if (args.output / case["case"] / "result.json").exists():
                continue
            print("START", case["case"], flush=True)
            result = run_case(case, args.output, args.rounds, args.repeats)
            export(args.output)
            print("DONE", case["case"], "failures", len(result["failures"]), flush=True)
    export(args.output)


if __name__ == "__main__":
    main()

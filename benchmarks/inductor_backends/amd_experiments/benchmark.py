"""Benchmark one independent lowering against its disabled, stock-equivalent control."""

import argparse
import contextlib
import csv
import fcntl
import gc
import hashlib
import inspect
import json
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import amd_benchmark_protocol as gpu_protocol
import concept

import torch
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import TestCase


workload = concept.workload
measure = gpu_protocol.measure
VARIANTS = ("triton", "feature")


def patch(variant, diagnostics):
    return config.patch(
        {
            concept.CONFIG: variant == "feature",
            "autotune_local_cache": False,
            "autotune_remote_cache": False,
            "fx_graph_cache": False,
            "triton.cudagraphs": False,
            "max_autotune": False,
            "max_autotune_pointwise": True,
            "max_autotune_gemm_backends": "ATEN",
            "max_autotune_conv_backends": "ATEN",
            **getattr(concept, "PATCH", {}),
        }
    )


def source_fingerprint():
    root = Path(__file__).resolve().parents[3]
    files = [*root.glob("torch/_inductor/**/*.py")]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {
        "inductor_sha256": digest.hexdigest(),
        "root": str(root),
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "dirty_diff_sha256": hashlib.sha256(
            subprocess.check_output(
                ["git", "diff", "HEAD", "--", "torch/_inductor"], cwd=root
            )
        ).hexdigest(),
    }


def validate(actual, expected, exact=False, *, family=None):
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
            else 0.0002
            if family == "cumsum"
            else 0.0001,
            rtol=0
            if exact
            else 0.02
            if a.dtype in (torch.bfloat16, torch.float16)
            else 0.0001,
            equal_nan=True,
        )
        errors.append((a.float() - b.float()).abs().nan_to_num().max().item())
    return max(errors)


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
    entries, failures = ([], [])
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
                    family=case["family"],
                )
                source = "\n".join(code)
                (folder / f"{variant}.source.txt").write_text(source)
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
                    family=case["family"],
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
                family=case["family"],
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    with contextlib.ExitStack() as locks:
        entry_lock = locks.enter_context(
            open("/dev/shm/pytorch-simd-v2-entry.lock", "a+")
        )
        benchmark_lock = locks.enter_context(
            open("/dev/shm/pytorch-simd-v2-benchmark.lock", "a+")
        )
        fcntl.flock(entry_lock, fcntl.LOCK_EX)
        fcntl.flock(benchmark_lock, fcntl.LOCK_EX)
        torch.cuda.init()
        torch.cuda.synchronize()
        gc.collect()
        gc.disable()
        try:
            for case in concept.cases():
                print("CASE", case["case"], flush=True)
                try:
                    reports.append(
                        run_case(case, args.output, args.rounds, args.repeats)
                    )
                except Exception:
                    failure = {
                        "case": case,
                        "rows": [],
                        "failures": [
                            {"variant": "validation", "error": traceback.format_exc()}
                        ],
                    }
                    (args.output / case["case"] / "result.json").write_text(
                        json.dumps(failure, indent=2)
                    )
                    reports.append(failure)
        finally:
            torch.cuda.synchronize()
            gc.enable()
        rows = [row for report in reports for row in report["rows"]]
        if rows:
            with (args.output / "per_case.csv").open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
                )
                writer.writeheader()
                writer.writerows(rows)
        (args.output / "manifest.json").write_text(
            json.dumps(
                {
                    "concept": concept.CONFIG,
                    "device": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "triton": __import__("triton").__version__,
                    "protocol": gpu_protocol.MEASUREMENT_PROTOCOL,
                    "source": source_fingerprint(),
                    "rounds": args.rounds,
                    "repeats": args.repeats,
                    "physical_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "failed_cases": sum(bool(r["failures"]) for r in reports),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

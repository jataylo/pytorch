"""Compare standard Inductor Triton with the native AMD optimization policy.

Run from a built checkout with --output pointing to a fresh result directory.
Direct timing captures GPU kernel sequences; inline graph replay timing includes
possible host launch gaps. Both cold and warm input protocols are exported.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import traceback
from pathlib import Path

import amd_native_benchmark as benchmark

import torch


VARIANTS = ("triton", "native_triton")


def main():
    """Run the comparison and retain successful measurements and failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        help="JSON workload list; defaults to the focused 16 cases",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error("rounds and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("output must be an empty directory")
    cases = [
        dict(family=family, shape=shape, dtype=dtype, case=f"{family}_{dtype}")
        for family, shape in (
            ("cast_add", (4608000,)),
            ("broadcast", (4608000,)),
            ("transpose", (16, 32, 128, 80)),
            ("softmax", (4096, 1024)),
            ("softmax_backward", (1024, 8192)),
            ("norm_backward", (4096, 1024)),
            ("cross_entropy_backward", (256, 1024)),
            ("cumsum", (64, 32768)),
        )
        for dtype in ("bfloat16", "float32")
    ]
    if args.cases is not None:
        cases = json.loads(args.cases.read_text())
    if not isinstance(cases, list) or not cases:
        parser.error("cases must be a nonempty JSON list")
    names = [case["case"] for case in cases]
    if len(names) != len(set(names)) or any(
        Path(name).name != name or name in (".", "..") for name in names
    ):
        parser.error("case names must be unique single path components")
    base = benchmark
    torch.set_num_threads(8)
    manifest = dict(
        cases=cases,
        variants=VARIANTS,
        rounds=args.rounds,
        repeats=args.repeats,
        measurement_protocol=benchmark.gpu_protocol.MEASUREMENT_PROTOCOL,
        source_fingerprint=benchmark.fingerprint(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    rows, failures = [], []
    with (
        open("/dev/shm/pytorch-simd-v2-entry.lock", "a") as entry,
        open("/dev/shm/pytorch-simd-v2-benchmark.lock", "a") as timing,
    ):
        fcntl.flock(entry, fcntl.LOCK_EX)
        fcntl.flock(timing, fcntl.LOCK_EX)
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        for case in cases:
            try:
                result = base.run_case(case, args.output, args.rounds, args.repeats)
                for row in result["rows"]:
                    invocation = row["invocation"]
                    row["measured_invocation"] = (
                        benchmark.gpu_protocol.INVOCATION_PROTOCOL[invocation]
                    )
                    row["host_overhead_included"] = (
                        benchmark.gpu_protocol.HOST_OVERHEAD_INCLUDED[invocation]
                    )
                result["benchmark_protocol"] = manifest["measurement_protocol"]
                result_path = args.output / case["case"] / "result.json"
                result_path.write_text(json.dumps(result, indent=2) + "\n")
                rows.extend(result["rows"])
                failures.extend(
                    dict(case=case["case"], **failure) for failure in result["failures"]
                )
                print(case["case"], "failures", len(result["failures"]), flush=True)
            except Exception:
                failures.append(
                    dict(
                        case=case["case"],
                        variant="case_execution",
                        error=traceback.format_exc(),
                    )
                )
                print(case["case"], "failed", flush=True)
            (args.output / "failures.json").write_text(
                json.dumps(failures, indent=2) + "\n"
            )
            if rows:
                with (args.output / "raw.csv").open("w", newline="") as target:
                    writer = csv.DictWriter(target, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    base.export(args.output)
    print("DONE", len(rows), "rows", len(failures), "failures", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

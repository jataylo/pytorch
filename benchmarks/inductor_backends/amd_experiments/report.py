"""Summarize independent sibling experiment worktrees without combining feature gains."""

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


DESCRIPTIONS = {
    "pointwise-subtiling": "Process several smaller pointwise tiles per program, reusing scalar loads and arithmetic; retain flat candidates.",
    "row-packing": "Prove row-tail masks before admitting larger row groups and device-bounded warp configurations to persistent reductions.",
    "transpose-tiling": "Compare two-dimensional transpose tiles and read orders with the original flat copy, proving strides and masks.",
    "reduction-indexing": "Remove an integer modulo only when the load mask proves that its nonnegative index is already below the modulus.",
    "persistent-selection": "Use live-value and hardware resource estimates to admit persistent kernels, then compare with looped MultiKernel alternatives.",
    "wide-reduction-splits": "Create or refine parallel partial reductions for underfilled wide rows with coupled broadcast epilogues; retain the existing reduction arithmetic.",
    "hierarchical-scan": "Compare additive FP32 chunk scans plus preceding-chunk offsets with the original associative scan, keeping launch geometry fixed.",
    "two-pass-scan": "Compute chunk totals and prefixes in FP64 across two kernels, then cast to FP32. This changes accumulation precision and is experimental.",
    "kernel-selection": "Expand generic reduction configurations and use bounded, mutation-safe paired tuning and confirmation; leave emitted kernel math unchanged.",
}


def kernels(source):
    result = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if "@triton.jit" not in node.value:
            continue
        tree = ast.parse(node.value)
        for fn in tree.body:
            if isinstance(fn, ast.FunctionDef):
                fn.name = "kernel"
                fn.decorator_list = []
        result.append(ast.dump(tree, include_attributes=False))
    return result


def exercised(slug, stock, feature):
    markers = {
        "pointwise-subtiling": "AMD_UNROLL",
        "row-packing": "'amd_native_family': 'row_reduction'",
        "transpose-tiling": "'amd_native_family': 'tiled_transpose'",
        "persistent-selection": "multi_kernel",
        "hierarchical-scan": "AMD_SCAN_TILE",
        "two-pass-scan": "j < chunk",
        "kernel-selection": "'amd_kernel_autotune': True",
    }
    if slug in markers:
        return markers[slug] in feature
    if slug == "reduction-indexing":
        return "Mod()" in "\n".join(kernels(stock)) and "\n".join(kernels(stock)).count(
            "Mod()"
        ) > "\n".join(kernels(feature)).count("Mod()")
    return kernels(stock) != kernels(feature)


def percentile(values, fraction):
    values = sorted(values)
    return values[round(fraction * (len(values) - 1))]


def bootstrap(stock, feature):
    rng = random.Random(2026)
    values = []
    for _ in range(2000):
        indices = [rng.randrange(len(stock)) for _ in stock]
        ratio = statistics.median(stock[i] for i in indices) / statistics.median(
            feature[i] for i in indices
        )
        values.append(100 * (ratio - 1))
    return percentile(values, 0.025), percentile(values, 0.975)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            lineterminator="\n",
            fieldnames=list(dict.fromkeys(key for row in rows for key in row)),
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(path):
    slug = path.name
    output = path / "benchmarks/inductor_backends/amd_experiments/results"
    comparisons, failures = [], []
    for result in sorted(output.glob("*/result.json")):
        report = json.loads(result.read_text())
        case = report["case"]
        failures += [
            dict(case=case["case"], **failure) for failure in report["failures"]
        ]
        # Never compare a partially compiled or numerically failing pair.
        if report["failures"]:
            continue
        sources = {
            variant: (result.parent / f"{variant}.source.txt").read_text()
            for variant in ("triton", "feature")
        }
        for variant, source in sources.items():
            expected = next(
                r["source_sha256"] for r in report["rows"] if r["variant"] == variant
            )
            if hashlib.sha256(source.encode()).hexdigest() != expected:
                raise RuntimeError(
                    f"Measured source snapshot changed: {result.parent}/{variant}"
                )
        applied = exercised(slug, sources["triton"], sources["feature"])
        for invocation in ("direct", "graph_replay"):
            for cache in ("warm", "cold"):
                pair = {
                    r["variant"]: r
                    for r in report["rows"]
                    if r["invocation"] == invocation and r["cache"] == cache
                }
                if len(pair) != 2:
                    continue
                a, b = pair["triton"], pair["feature"]
                low, high = bootstrap(
                    report["samples_us"][f"triton_{cache}_{invocation}"],
                    report["samples_us"][f"feature_{cache}_{invocation}"],
                )
                speedup = 100 * (a["runtime_us"] / b["runtime_us"] - 1)
                comparisons.append(
                    dict(
                        concept=slug,
                        **case,
                        cache=cache,
                        invocation=invocation,
                        feature_exercised=applied,
                        stock_us=a["runtime_us"],
                        feature_us=b["runtime_us"],
                        speedup_percent=speedup,
                        time_saved_percent=100
                        * (1 - b["runtime_us"] / a["runtime_us"]),
                        paired_ci95_low_percent=low,
                        paired_ci95_high_percent=high,
                        outcome="win"
                        if speedup > 2
                        else "regression"
                        if speedup < -2
                        else "within_2_percent",
                        stock_compile_seconds=a["compile_seconds"],
                        feature_compile_seconds=b["compile_seconds"],
                        stock_kernel_count=len(kernels(sources["triton"])),
                        feature_kernel_count=len(kernels(sources["feature"])),
                        stock_source_sha256=a["source_sha256"],
                        feature_source_sha256=b["source_sha256"],
                    )
                )
    write_csv(output / "comparison.csv", comparisons)
    (output / "failures.json").write_text(json.dumps(failures, indent=2))
    summary = []
    for invocation in ("direct", "graph_replay"):
        for cache in ("warm", "cold"):
            selected = [
                r
                for r in comparisons
                if r["cache"] == cache and r["invocation"] == invocation
            ]
            eligible = [r for r in selected if r["feature_exercised"]]
            if not selected:
                continue

            def geomean(rows):
                return (
                    100
                    * (
                        math.exp(
                            statistics.mean(
                                math.log(1 + r["speedup_percent"] / 100) for r in rows
                            )
                        )
                        - 1
                    )
                    if rows
                    else None
                )

            summary.append(
                dict(
                    concept=slug,
                    cache=cache,
                    invocation=invocation,
                    valid_cases=len(selected),
                    exercised_cases=len(eligible),
                    failed_cases=len({r["case"] for r in failures}),
                    geomean_speedup_percent=geomean(selected),
                    exercised_geomean_speedup_percent=geomean(eligible),
                    best_speedup_percent=max(r["speedup_percent"] for r in selected),
                    worst_speedup_percent=min(r["speedup_percent"] for r in selected),
                    wins=sum(r["speedup_percent"] > 2 for r in selected),
                    regressions=sum(r["speedup_percent"] < -2 for r in selected),
                    within_2_percent=sum(
                        abs(r["speedup_percent"]) <= 2 for r in selected
                    ),
                    stock_compile_seconds=sum(
                        r["stock_compile_seconds"] for r in selected
                    ),
                    feature_compile_seconds=sum(
                        r["feature_compile_seconds"] for r in selected
                    ),
                )
            )
    write_csv(output / "summary.csv", summary)
    return comparisons, summary, failures


def number(value):
    return "N/A" if value is None else f"{value:+.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing the nine feature worktrees",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows, all_summary = [], []
    for slug, description in DESCRIPTIONS.items():
        rows, summary, failures = summarize(args.root / slug)
        all_rows += rows
        all_summary += summary
        primary = next(
            (
                r
                for r in summary
                if r["cache"] == "warm" and r["invocation"] == "direct"
            ),
            None,
        )
        md = f"# {slug}\n\n{description}\n\nIndependent branch: `experiment/amd-lowering-{slug}`. Parent: `experiment/amd-lowering-bench-base`, based on upstream `dddd90223f0`. No other optimization branch is required. The disabled arm uses stock lowering; see `concept.py` for the single enabled configuration and workload shapes.\n\n"
        if primary:
            md += f"Measured on one MI355X, physical GPU 3: {primary['valid_cases']} valid pairs, {primary['exercised_cases']} exercising this feature, {primary['failed_cases']} failed cases. Warm captured GPU sequence geomean speedup: **{number(primary['geomean_speedup_percent'])}**, best {number(primary['best_speedup_percent'])}, worst {number(primary['worst_speedup_percent'])}. Wins/regressions outside a conservative 2% band: {primary['wins']}/{primary['regressions']}. These are operator units, not model speedups.\n\n"
        md += "| Case | Stock us | Feature us | Speedup | Paired 95% interval | Feature exercised |\n|---|---:|---:|---:|---|---|\n"
        for row in rows:
            if row["cache"] == "warm" and row["invocation"] == "direct":
                md += f"| {row['case']} | {row['stock_us']:.3f} | {row['feature_us']:.3f} | {number(row['speedup_percent'])} | {number(row['paired_ci95_low_percent'])} to {number(row['paired_ci95_high_percent'])} | {row['feature_exercised']} |\n"
        md += "\n`results/comparison.csv` includes all four invocation/cache protocols, compile costs, source hashes and kernel counts. `results/summary.csv` contains aggregate statistics. Per-case `result.json` preserves every timing sample; `*.source.txt` files preserve the measured generated code. Intervals resample paired timing rounds and do not include variation between separate compilations. Small gains and candidate-selection instability need repeat trials. Failed or ineligible cases are not credited as lowering wins.\n\nRun the commands in `results/run.sh`; validation/lint commands and outputs are in `results/validation/`. Benchmarks check three input seeds; scans use a FP64-prefix reference.\n\n"
        if slug == "two-pass-scan":
            md += "**Precision contract:** FP64 partials intentionally differ from the stock FP32 scan tree, especially for overflow and cancellation. Nonnegative performance inputs are used in both arms; signed, nonfinite, overflow, stride and exclusion behavior are covered by the unit tests. This branch requires a separate numerical-semantics decision before upstreaming.\n\n"
        if not primary or not primary["exercised_cases"]:
            md += "**No independently exercised workload:** do not claim a performance benefit from this run; add a triggering workload first.\n\n"
        md += "## Agent details\n\nPrepared with an AI assistant for human review before upstream submission.\n"
        (
            args.root / slug / "benchmarks/inductor_backends/amd_experiments/RESULTS.md"
        ).write_text(md)
    write_csv(args.output / "all_comparisons.csv", all_rows)
    write_csv(args.output / "all_summaries.csv", all_summary)


if __name__ == "__main__":
    main()

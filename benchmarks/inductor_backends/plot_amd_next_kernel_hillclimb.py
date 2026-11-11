"""Plot regular Triton, previous optimized Triton, extended Triton and FlyDSL."""

import argparse
import csv
import html
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import plot_amd_kernel_hillclimb as base


COLORS = {
    "triton": "#3976bd",
    "v15_triton": "#8793a0",
    "optimized_triton": "#15966a",
    "flydsl": "#e58b28",
}
LABELS = {
    "triton": "Regular Triton",
    "v15_triton": "Previous optimized Triton",
    "optimized_triton": "New optimized Triton",
    "flydsl": "FlyDSL",
}


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def compare(directory):
    rows = list(csv.DictReader((directory / "per_case.csv").open()))
    groups = defaultdict(dict)
    for row in rows:
        groups[row["case"], row["cache"], row["invocation"]][row["variant"]] = row
    comparisons = []
    for (case, cache, invocation), variants in groups.items():
        if set(variants) != set(COLORS):
            continue
        candidate = variants["optimized_triton"]
        row = {
            key: candidate.get(key, "") for key in ("family", "dtype", "shape", "model")
        }
        row.update(case=case, cache=cache, invocation=invocation)
        for variant, value in variants.items():
            row[variant + "_us"] = float(value["runtime_us"])
        for name, variant in [
            ("regular", "triton"),
            ("previous", "v15_triton"),
            ("flydsl", "flydsl"),
        ]:
            ratio = float(candidate["runtime_us"]) / float(
                variants[variant]["runtime_us"]
            )
            row["time_saved_vs_" + name + "_percent"] = 100 * (1 - ratio)
            row["speedup_vs_" + name + "_factor"] = 1 / ratio
        comparisons.append(row)
    write_csv(directory / "comparison.csv", comparisons)
    summaries = []
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[row["family"], row["cache"], row["invocation"]].append(row)
    for (family, cache, invocation), values in sorted(grouped.items()):
        row = dict(family=family, cache=cache, invocation=invocation, cases=len(values))
        for baseline in ("regular", "previous", "flydsl"):
            field = "time_saved_vs_" + baseline + "_percent"
            row["geomean_" + field] = 100 * (
                1
                - math.exp(
                    statistics.mean(
                        math.log(1 - value[field] / 100) for value in values
                    )
                )
            )
            row["best_" + field] = max(value[field] for value in values)
            row["worst_" + field] = min(value[field] for value in values)
            row["wins_over_3_percent_vs_" + baseline] = sum(
                value[field] > 3 for value in values
            )
            row["regressions_over_3_percent_vs_" + baseline] = sum(
                value[field] < -3 for value in values
            )
        summaries.append(row)
    write_csv(directory / "comparison_summary.csv", summaries)
    base.COLORS = COLORS
    base.LABELS = LABELS
    original = base.summary_chart
    base.summary_chart = lambda rows, cache: original(rows, cache).replace(
        "Common successful three-backend cases only",
        "Common successful four-implementation cases",
    )
    for invocation in ("direct", "graph_replay"):
        sys.argv = [sys.argv[0], str(directory / invocation)]
        base.main()
    table = []
    for row in summaries:
        if row["cache"] == "warm" and row["invocation"] == "direct":
            table.append(
                "<tr><td>"
                + html.escape(row["family"])
                + "</td><td>"
                + str(row["cases"])
                + "</td>"
                + "".join(
                    f"<td>{row['geomean_time_saved_vs_' + baseline + '_percent']:.2f}%</td>"
                    for baseline in ("regular", "previous", "flydsl")
                )
                + "</tr>"
            )
    failures = json.loads((directory / "failures.json").read_text())
    (directory / "report.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Extended AMD Inductor hill climb</title>
<style>body{max-width:1400px;margin:32px auto;font:16px Arial;color:#253445}td,th{padding:9px;border-bottom:1px solid #ddd}img{width:100%;height:auto}</style>
<h1>Extended Inductor lowering: paired comparisons</h1>
<p>All math functions retain the existing precision policy. Lowering/layout changes and balanced cache tuning are opt-in. Four implementations are measured in rotating rounds with the same inputs: regular Triton, previous optimized Triton, new optimized Triton, and FlyDSL. Negative percentages are regressions.</p>
<p>These are compiled operation graphs on one MI355X, not full-model training throughput. Accuracy checks use three input seeds. Captured GPU kernel sequences and inline graph replay are reported separately; inline replay can include CPU launch gaps. A 256 MiB flush precedes each sample; warm measurements then touch input storage. Mutation restores occur before timing.</p>
"""
        + f"<p>{len(groups) // 4} cases; {len(failures)} variant failures. "
        + '<a href="comparison.csv">Per-case comparisons</a> | <a href="comparison_summary.csv">Per-operation summary</a> | <a href="per_case.csv">Raw timing CSV</a> | <a href="failures.json">Failures</a></p>'
        + "<h2>Warm inputs, direct execution: geometric mean runtime reduction</h2><table><tr><th>Operation</th><th>Cases</th><th>vs regular Triton</th><th>vs previous optimized Triton</th><th>vs FlyDSL</th></tr>"
        + "".join(table)
        + "</table>"
        + '<img src="direct/graphs/summary_warm.svg"/><p><a href="direct/report.html">Direct execution: all per-operation graphs</a> | <a href="graph_replay/report.html">CUDA graph replay: all graphs</a></p>'
    )
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    compare(args.directory)

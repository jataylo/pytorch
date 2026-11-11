"""Plot exact and opt-in approximate codegen benchmarks in separate figures."""

import argparse
import sys
from pathlib import Path

import plot_amd_kernel_hillclimb as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    base.COLORS["optimized_triton_approx"] = "#9463c7"
    base.LABELS["optimized_triton_approx"] = "Optimized + approximate math"
    original_runtime = base.runtime_chart

    def runtime(rows, family, cache, logarithmic):
        rows = [
            dict(row, shape=(row.get("model", "") + " " + row["shape"]).strip())
            for row in rows
        ]
        return original_runtime(rows, family, cache, logarithmic)

    base.runtime_chart = runtime
    original_summary = base.summary_chart
    base.summary_chart = lambda rows, cache: original_summary(rows, cache).replace(
        "three-backend", "four-variant"
    )
    for mode in ["direct", "graph_replay"]:
        folder = args.directory / mode
        sys.argv = [sys.argv[0], str(folder)]
        base.main()
        p = folder / "report.html"
        s = (
            p.read_text()
            .replace(
                "No approximate math functions are enabled.",
                "The purple variant enables approximate GELU/softmax math separately from the green variant.",
            )
            .replace(
                "compiled operation graphs, not full-model training throughput.",
                "compiled operation graphs, not full-model training throughput. Invocation mode: "
                + mode
                + ".",
            )
        )
        p.write_text(s)
    (args.directory / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Inductor lowering hill climb</title><h1>Inductor lowering hill climb</h1><p>Original Triton, optimized Triton, FlyDSL, and opt-in approximate math. All regressions and failures are retained.</p><ul><li><a href="direct/report.html">Direct compiled GPU execution</a></li><li><a href="graph_replay/report.html">CUDA graph replay</a></li><li><a href="per_case.csv">All measurements CSV</a></li><li><a href="failures.json">Failures</a></li></ul>'
    )


if __name__ == "__main__":
    main()

"""Export standalone SVG performance figures from amd_kernel_hillclimb results."""

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree


COLORS = {"triton": "#3976bd", "optimized_triton": "#15966a", "flydsl": "#e58b28"}
LABELS = {
    "triton": "Original Triton",
    "optimized_triton": "Optimized Triton",
    "flydsl": "FlyDSL",
}


def esc(value):
    return html.escape(str(value), quote=True)


def text(x, y, value, size=13, anchor="start", fill="#253445"):
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}">{esc(value)}</text>'


def runtime_chart(rows, family, cache, logarithmic):
    rows = [r for r in rows if r["family"] == family and r["cache"] == cache]
    cases = sorted(
        {r["case"] for r in rows},
        key=lambda case: (case.split("_")[-2], int(case.split("_")[-1])),
    )
    width, left, right = 1400, 405, 1180
    height = 155 + 83 * len(cases)
    maximum = max((float(r["p90_us"]) for r in rows), default=1) * 1.2
    maximum = max(maximum, 1)

    def xpos(value):
        if logarithmic:
            return left + (right - left) * (math.log10(max(value, 0.1)) + 1) / (
                math.log10(maximum) + 1
            )
        return left + (right - left) * value / maximum

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif">',
        text(24, 30, f"{family}: compiled graph runtime ({cache} inputs)", 22),
        text(
            24,
            53,
            "MI355X | median of rotating rounds | whiskers: p10-p90 | shorter is faster",
            13,
        ),
    ]
    for i, (variant, color) in enumerate(COLORS.items()):
        x = 30 + i * 230
        parts += [
            f'<rect x="{x}" y="70" width="15" height="12" fill="{color}"/>',
            text(x + 23, 81, LABELS[variant]),
        ]
    ticks = [maximum * i / 5 for i in range(6)]
    if logarithmic:
        ticks = [
            10**exponent * factor
            for exponent in range(-1, math.ceil(math.log10(maximum)) + 1)
            for factor in (1, 3)
            if 10**exponent * factor <= maximum
        ]
    for tick in ticks:
        x = xpos(tick)
        parts += [
            f'<path d="M{x:.2f},105 V{height - 42}" stroke="#e0e5eb"/>',
            text(x, height - 24, f"{tick:.3g}", 11, "middle"),
        ]
    lookup = {(r["case"], r["variant"]): r for r in rows}
    for i, case in enumerate(cases):
        y = 118 + 83 * i
        representative = next(r for r in rows if r["case"] == case)
        parts += [
            text(
                left - 16,
                y + 20,
                representative["dtype"] + " " + representative["shape"],
                12,
                "end",
            )
        ]
        for j, (variant, color) in enumerate(COLORS.items()):
            row = lookup.get((case, variant))
            bar_y = y + 19 * j
            if row is None:
                parts.append(
                    text(left + 8, bar_y + 12, "failed / unavailable", 11, fill=color)
                )
                continue
            value = float(row["runtime_us"])
            x = xpos(value)
            parts.append(
                f'<rect x="{left}" y="{bar_y}" width="{max(0.4, x - left):.2f}" height="14" fill="{color}" rx="2"/>'
            )
            p10, p90 = xpos(float(row["p10_us"])), xpos(float(row["p90_us"]))
            parts.append(
                f'<path d="M{p10:.2f},{bar_y + 7} H{p90:.2f}" stroke="#253445" stroke-width="1.3"/>'
            )
            parts.append(text(x + 6, bar_y + 12, f"{value:.3g} us", 11))
    label = (
        "Runtime (microseconds, log scale; bars start at 0.1 us)"
        if logarithmic
        else "Runtime (microseconds, linear scale)"
    )
    parts += [text((left + right) / 2, height - 5, label, 12, "middle"), "</g></svg>"]
    return "\n".join(parts)


def common_summary(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["case"], row["cache"]][row["variant"]] = row
    comparable = defaultdict(list)
    for (case, cache), variants in groups.items():
        if set(variants) == set(COLORS):
            for variant, row in variants.items():
                comparable[row["family"], cache, variant].append(
                    float(row["runtime_us"]) / float(variants["triton"]["runtime_us"])
                )
    return [
        dict(
            family=family,
            cache=cache,
            variant=variant,
            common_cases=len(values),
            normalized_runtime_percent=100
            * math.exp(statistics.mean(math.log(v) for v in values)),
            time_saved_percent=100
            * (1 - math.exp(statistics.mean(math.log(v) for v in values))),
        )
        for (family, cache, variant), values in sorted(comparable.items())
    ]


def summary_chart(rows, cache):
    rows = [r for r in rows if r["cache"] == cache]
    families = sorted({r["family"] for r in rows})
    width, height, left, right = 1200, 155 + 85 * len(families), 255, 1000
    maximum = max([125, *[r["normalized_runtime_percent"] * 1.15 for r in rows]])

    def xpos(value):
        return left + (right - left) * value / maximum

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif">',
        text(24, 30, f"Category geometric means ({cache} inputs)", 22),
        text(
            24,
            54,
            "Common successful three-backend cases only; Original Triton = 100%; lower is faster",
            13,
        ),
    ]
    for i, (variant, color) in enumerate(COLORS.items()):
        x = 30 + i * 230
        parts += [
            f'<rect x="{x}" y="70" width="15" height="12" fill="{color}"/>',
            text(x + 23, 81, LABELS[variant]),
        ]
    for value in range(0, math.ceil(maximum), 25):
        x = xpos(value)
        parts += [
            f'<path d="M{x:.2f},105 V{height - 40}" stroke="#e0e5eb"/>',
            text(x, height - 22, value, 11, "middle"),
        ]
    for i, family in enumerate(families):
        y = 118 + i * 85
        parts.append(text(left - 16, y + 22, family, 13, "end"))
        for j, (variant, color) in enumerate(COLORS.items()):
            row = next(
                r for r in rows if r["family"] == family and r["variant"] == variant
            )
            value = row["normalized_runtime_percent"]
            x = xpos(value)
            parts += [
                f'<rect x="{left}" y="{y + j * 19}" width="{x - left:.2f}" height="14" fill="{color}"/>',
                text(
                    x + 6,
                    y + j * 19 + 12,
                    f"{value:.1f}% (n={row['common_cases']})",
                    11,
                ),
            ]
    parts += [
        text(
            (left + right) / 2, height - 4, "Normalized graph runtime (%)", 12, "middle"
        ),
        "</g></svg>",
    ]
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    with (directory / "per_case.csv").open() as f:
        rows = list(csv.DictReader(f))
    graphs = directory / "graphs"
    graphs.mkdir(exist_ok=True)
    summary = common_summary(rows)
    with (directory / "summary_common.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    pages = []
    for cache in ("cold", "warm"):
        name = f"summary_{cache}.svg"
        path = graphs / name
        path.write_text(summary_chart(summary, cache))
        ElementTree.parse(path)
        pages.append(
            f'<h2>{cache.title()} inputs</h2><img src="graphs/{name}" style="width:100%"/>'
        )
        for family in sorted({r["family"] for r in rows}):
            for logarithmic in (True, False):
                name = f"{family}_{cache}_{'log' if logarithmic else 'linear'}.svg"
                path = graphs / name
                path.write_text(runtime_chart(rows, family, cache, logarithmic))
                ElementTree.parse(path)
                if logarithmic:
                    pages.append(
                        f'<details><summary>{esc(family)} runtime</summary><img src="graphs/{name}" style="width:100%"/><p><a href="graphs/{family}_{cache}_linear.svg">Linear scale figure</a></p></details>'
                    )
    failures = json.loads((directory / "failures.json").read_text())
    intro = f'<h1>AMD kernel codegen experiments</h1><p>{len({r["case"] for r in rows})} cases with measured results; {len(failures)} variant failures. These are compiled operation graphs, not full-model training throughput. No approximate math functions are enabled. Both successful and failed cases are retained in the CSV/JSON exports.</p><p><a href="per_case.csv">Per-case CSV</a> | <a href="summary_common.csv">Common-case summary</a> | <a href="failures.json">Failures</a></p>'
    (directory / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>AMD kernel codegen benchmark</title><style>body{max-width:1400px;margin:32px auto;font:16px Arial;color:#253445}summary{cursor:pointer;padding:12px}details{border:1px solid #ddd;margin:10px 0}img{height:auto}</style>'
        + intro
        + "\n".join(pages)
    )
    print(graphs.resolve(), flush=True)


if __name__ == "__main__":
    main()

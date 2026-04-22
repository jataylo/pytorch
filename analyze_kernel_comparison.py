#!/usr/bin/env python3
"""analyze_kernel_comparison.py — Pivot phase2_timings.jsonl into a wide CSV.

Reads the flat JSONL produced by run_phase2_benchmarks.py and generates:

  kernel_comparison.csv
      One row per kernel.  Columns:
        kernel_name, kernel_type, kernel_file,
        autotune_ms, autotune_gb_per_s,
        no_tune_ms, no_tune_gb_per_s, no_tune_vs_autotune_pct,
        <mode>_ms, <mode>_gb_per_s, <mode>_vs_autotune_pct,  ...
        best_heuristic_label, best_heuristic_ms, best_heuristic_vs_autotune_pct

  kernel_comparison_summary.txt
      Per-mode statistics: median speedup, #faster, #slower, #match,
      worst regressions, best wins.

Speedup sign convention:
  positive  → heuristic is FASTER than autotune (good)
  negative  → heuristic is SLOWER than autotune (bad)

Usage:
    python analyze_kernel_comparison.py  \\
        --jsonl   phase2_timings.jsonl   \\
        --out-csv kernel_comparison.csv  \\
        --out-txt kernel_comparison_summary.txt
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  WARN: bad JSON on line {lineno}: {exc}", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def speedup_pct(autotune_ms: float, heuristic_ms: float) -> float:
    """Positive = heuristic faster, negative = heuristic slower."""
    if autotune_ms <= 0 or not math.isfinite(autotune_ms):
        return float("nan")
    return (autotune_ms - heuristic_ms) / autotune_ms * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl",   default="phase2_timings.jsonl")
    ap.add_argument("--out-csv", default="kernel_comparison.csv")
    ap.add_argument("--out-txt", default="kernel_comparison_summary.txt")
    args = ap.parse_args()

    records = load_jsonl(args.jsonl)
    print(f"Loaded {len(records)} records from {args.jsonl}")

    # ── group by kernel (name + file) ─────────────────────────────────────
    # Use (kernel_name, basename) as the unique key so renamed compilations
    # with the same hash are still distinguished.
    KernelKey = tuple  # (kernel_name, file_basename)

    by_kernel: dict[KernelKey, dict[str, dict]] = defaultdict(dict)
    for r in records:
        key: KernelKey = (r["kernel_name"], os.path.basename(r["kernel_file"]))
        by_kernel[key][r["mode_label"]] = r

    print(f"Unique kernels: {len(by_kernel)}")

    # ── determine mode column order ────────────────────────────────────────
    all_modes: list[str] = []
    seen_modes: set[str] = set()
    for mode_map in by_kernel.values():
        for m in mode_map:
            if m not in seen_modes:
                seen_modes.add(m)
                all_modes.append(m)

    # Sort: autotune first, no_tune second, then alphabetical heuristic modes
    def mode_sort_key(m: str) -> tuple:
        if m == "autotune":   return (0, m)
        if m == "no_tune":    return (1, m)
        return (2, m)

    all_modes.sort(key=mode_sort_key)
    heuristic_modes = [m for m in all_modes if m not in ("autotune", "no_tune")]
    print(f"Modes: {all_modes}")

    # ── build wide table ───────────────────────────────────────────────────
    header_base = ["kernel_name", "kernel_type", "kernel_file"]
    header_at   = ["autotune_ms", "autotune_gb_per_s"]
    header_nt   = ["no_tune_ms", "no_tune_gb_per_s", "no_tune_vs_autotune_pct"]
    header_h    = []
    for m in heuristic_modes:
        header_h += [f"{m}_ms", f"{m}_gb_per_s", f"{m}_vs_autotune_pct"]
    header_best = ["best_heuristic_label", "best_heuristic_ms",
                   "best_heuristic_vs_autotune_pct"]
    header = header_base + header_at + header_nt + header_h + header_best

    rows: list[dict] = []
    for (kname, kbasename), mode_map in sorted(by_kernel.items()):
        at_rec  = mode_map.get("autotune", {})
        nt_rec  = mode_map.get("no_tune",  {})
        at_ms   = at_rec.get("timing_ms")
        at_gbs  = at_rec.get("gb_per_s")

        row: dict = {
            "kernel_name":    kname,
            "kernel_type":    at_rec.get("kernel_type", nt_rec.get("kernel_type", "other")),
            "kernel_file":    at_rec.get("kernel_file", nt_rec.get("kernel_file", kbasename)),
            "autotune_ms":    f"{at_ms:.4f}"  if at_ms  is not None else "",
            "autotune_gb_per_s": f"{at_gbs:.2f}" if at_gbs is not None else "",
        }

        # no_tune
        nt_ms  = nt_rec.get("timing_ms")
        nt_gbs = nt_rec.get("gb_per_s")
        nt_vs  = speedup_pct(at_ms, nt_ms) if at_ms and nt_ms else float("nan")
        row["no_tune_ms"]              = f"{nt_ms:.4f}"  if nt_ms  is not None else ""
        row["no_tune_gb_per_s"]        = f"{nt_gbs:.2f}" if nt_gbs is not None else ""
        row["no_tune_vs_autotune_pct"] = f"{nt_vs:.2f}"  if math.isfinite(nt_vs) else ""

        # heuristic modes
        best_h_label: str         = ""
        best_h_ms:    float|None  = None
        best_h_vs:    float       = float("-inf")

        for m in heuristic_modes:
            rec   = mode_map.get(m, {})
            h_ms  = rec.get("timing_ms")
            h_gbs = rec.get("gb_per_s")
            h_vs  = speedup_pct(at_ms, h_ms) if at_ms and h_ms else float("nan")

            row[f"{m}_ms"]                = f"{h_ms:.4f}"  if h_ms  is not None else ""
            row[f"{m}_gb_per_s"]          = f"{h_gbs:.2f}" if h_gbs is not None else ""
            row[f"{m}_vs_autotune_pct"]   = f"{h_vs:.2f}"  if math.isfinite(h_vs) else ""

            if h_ms is not None and math.isfinite(h_vs) and h_vs > best_h_vs:
                best_h_vs    = h_vs
                best_h_ms    = h_ms
                best_h_label = m

        row["best_heuristic_label"]           = best_h_label
        row["best_heuristic_ms"]              = f"{best_h_ms:.4f}" if best_h_ms is not None else ""
        row["best_heuristic_vs_autotune_pct"] = (f"{best_h_vs:.2f}"
                                                  if math.isfinite(best_h_vs) else "")
        rows.append(row)

    # ── write CSV ──────────────────────────────────────────────────────────
    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {args.out_csv}")

    # ── per-mode statistics ────────────────────────────────────────────────
    def mode_stats(mode_label: str, is_heuristic: bool = True) -> dict:
        deltas = []
        regressions = []  # (pct, kernel_name)
        for row in rows:
            vs_key = f"{mode_label}_vs_autotune_pct" if is_heuristic else "no_tune_vs_autotune_pct"
            if mode_label == "no_tune":
                vs_str = row.get("no_tune_vs_autotune_pct", "")
                ms_str = row.get("no_tune_ms", "")
                kname  = row["kernel_name"]
            else:
                vs_str = row.get(f"{mode_label}_vs_autotune_pct", "")
                ms_str = row.get(f"{mode_label}_ms", "")
                kname  = row["kernel_name"]
            if not vs_str:
                continue
            pct = float(vs_str)
            if not math.isfinite(pct):
                continue
            deltas.append(pct)
            if pct < -5.0:
                regressions.append((pct, kname))

        if not deltas:
            return {"n": 0}

        deltas.sort()
        n = len(deltas)
        n_faster   = sum(1 for d in deltas if d > 1.0)
        n_match    = sum(1 for d in deltas if -1.0 <= d <= 1.0)
        n_slower   = sum(1 for d in deltas if d < -1.0)
        median     = deltas[n // 2]
        mean       = sum(deltas) / n
        regressions.sort()
        return {
            "n":          n,
            "n_faster":   n_faster,
            "n_match":    n_match,
            "n_slower":   n_slower,
            "median_pct": median,
            "mean_pct":   mean,
            "worst_regressions": regressions[:10],
        }

    lines = [
        f"Kernel comparison summary",
        f"{'='*80}",
        f"Kernels: {len(rows)}",
        f"Autotune (baseline) vs heuristic modes",
        f"  + positive % = heuristic FASTER than autotune",
        f"  - negative % = heuristic SLOWER than autotune",
        "",
    ]

    summary_modes = [("no_tune", False)] + [(m, True) for m in heuristic_modes]
    for ml, is_h in summary_modes:
        st = mode_stats(ml, is_h)
        if st.get("n", 0) == 0:
            lines.append(f"{ml}: no data")
            continue
        lines.append(f"{ml}:")
        lines.append(f"  n={st['n']}  faster={st['n_faster']}  "
                     f"match={st['n_match']}  slower={st['n_slower']}")
        lines.append(f"  median={st['median_pct']:+.2f}%  mean={st['mean_pct']:+.2f}%")
        if st["worst_regressions"]:
            lines.append("  Worst regressions (>5% slower):")
            for pct, kn in st["worst_regressions"]:
                lines.append(f"    {pct:+6.2f}%  {kn}")
        lines.append("")

    # Global: best heuristic vs autotune
    bh_wins  = sum(1 for r in rows if r["best_heuristic_vs_autotune_pct"] and
                   float(r["best_heuristic_vs_autotune_pct"]) > 1.0)
    bh_slow  = sum(1 for r in rows if r["best_heuristic_vs_autotune_pct"] and
                   float(r["best_heuristic_vs_autotune_pct"]) < -1.0)
    bh_vals  = [float(r["best_heuristic_vs_autotune_pct"])
                for r in rows if r["best_heuristic_vs_autotune_pct"]]
    lines.append("─"*80)
    lines.append("BEST heuristic vs autotune (best mode per kernel):")
    lines.append(f"  faster: {bh_wins}  within 1%: {len(bh_vals)-bh_wins-bh_slow}  slower: {bh_slow}")
    if bh_vals:
        bh_vals.sort()
        lines.append(f"  median: {bh_vals[len(bh_vals)//2]:+.2f}%  "
                     f"mean: {sum(bh_vals)/len(bh_vals):+.2f}%")

    # Top regressions with best heuristic
    worst_best = sorted(
        [(float(r["best_heuristic_vs_autotune_pct"]), r["kernel_name"])
         for r in rows if r["best_heuristic_vs_autotune_pct"]
         and float(r["best_heuristic_vs_autotune_pct"]) < -5.0]
    )[:15]
    if worst_best:
        lines.append("\n  Kernels where BEST heuristic is still >5% slower than autotune:")
        for pct, kn in worst_best:
            lines.append(f"    {pct:+6.2f}%  {kn}")

    summary_txt = "\n".join(lines)
    print("\n" + summary_txt)

    with open(args.out_txt, "w") as f:
        f.write(summary_txt + "\n")
    print(f"\nSummary → {args.out_txt}")


if __name__ == "__main__":
    main()

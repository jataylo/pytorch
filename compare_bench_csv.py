#!/usr/bin/env python3
"""
compare_bench_csv.py  –  Compare two BENCH_VERBOSE CSVs (A vs B).

Usage:
    python compare_bench_csv.py <a.csv> <b.csv> [output.csv]

Shows, for every kernel+shape that appears in both files:
    • Speedup of B over A  (A_us / B_us)
    • Whether the winning config changed

Output CSV columns:
    kernel_name, size_hints,
    a_us, b_us, speedup_BoverA,
    a_config, b_config, config_changed

Positive speedup_BoverA > 1.0 means B is faster.
"""

import csv
import sys
from pathlib import Path


def _get_us(row: dict) -> float:
    """Return the timing value regardless of whether the column is fastest_us or chosen_us."""
    return float(row.get("fastest_us") or row.get("chosen_us") or 0)


def load_csv(path: Path) -> dict[tuple, dict]:
    """Return {(kernel_name, size_hints): row} from a parsed CSV."""
    data: dict[tuple, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["kernel_name"], row["size_hints"])
            existing = data.get(key)
            if existing is None or _get_us(row) < _get_us(existing):
                data[key] = row
    return data


def compare(a: dict, b: dict) -> list[dict]:
    """Return comparison rows for keys present in both dicts."""
    rows = []
    common_keys = sorted(set(a) & set(b))
    for key in common_keys:
        ra, rb = a[key], b[key]
        a_us = _get_us(ra)
        b_us = _get_us(rb)
        speedup = a_us / b_us if b_us > 0 else float("inf")
        cfg_changed = ra["config"].strip() != rb["config"].strip()
        rows.append(
            dict(
                kernel_name=key[0],
                size_hints=key[1],
                a_us=a_us,
                b_us=b_us,
                speedup_BoverA=round(speedup, 4),
                a_config=ra["config"],
                b_config=rb["config"],
                config_changed="yes" if cfg_changed else "no",
            )
        )
    return rows


def print_summary(rows: list[dict], a_name: str, b_name: str) -> None:
    if not rows:
        print("No common kernel+shape pairs found.")
        return

    faster_b = [r for r in rows if r["speedup_BoverA"] > 1.005]
    faster_a = [r for r in rows if r["speedup_BoverA"] < 0.995]
    same     = [r for r in rows if 0.995 <= r["speedup_BoverA"] <= 1.005]
    changed  = [r for r in rows if r["config_changed"] == "yes"]

    speedups = [r["speedup_BoverA"] for r in rows]
    a_times  = [r["a_us"] for r in rows]
    b_times  = [r["b_us"] for r in rows]

    mean_speedup   = sum(speedups) / len(speedups)
    sorted_speedups = sorted(speedups)
    n = len(sorted_speedups)
    if n % 2 == 1:
        median_speedup = sorted_speedups[n // 2]
    else:
        median_speedup = (sorted_speedups[n // 2 - 1] + sorted_speedups[n // 2]) / 2
    total_a = sum(a_times)
    total_b = sum(b_times)
    total_speedup = total_a / total_b if total_b > 0 else float("inf")

    print(f"\n{'='*72}")
    print(f"  A = {a_name}")
    print(f"  B = {b_name}")
    print(f"{'='*72}")
    print(f"  Common pairs : {len(rows)}")
    print(f"  B faster     : {len(faster_b)}  ({100*len(faster_b)/len(rows):.1f}%)")
    print(f"  A faster     : {len(faster_a)}  ({100*len(faster_a)/len(rows):.1f}%)")
    print(f"  ≈ same       : {len(same)}  ({100*len(same)/len(rows):.1f}%)")
    print(f"  Config changed: {len(changed)}")
    print(f"")
    print(f"  Speedup B/A  (mean)  : {mean_speedup:.4f}x")
    print(f"  Speedup B/A  (median): {median_speedup:.4f}x")
    print(f"  Speedup B/A  (total) : {total_speedup:.4f}x  "
          f"(ΣA={total_a:.1f}µs  ΣB={total_b:.1f}µs)")

    # ── top gainers (B faster) ────────────────────────────────────────────────
    if faster_b:
        print(f"\n  Top 15 speedups  (B faster, sorted by speedup desc):")
        print(f"  {'Speedup':>8}  {'A µs':>10}  {'B µs':>10}  {'Cfg?':>5}  Kernel / size_hints")
        print(f"  {'-'*70}")
        for r in sorted(faster_b, key=lambda x: -x["speedup_BoverA"])[:15]:
            chg = "diff" if r["config_changed"] == "yes" else "same"
            print(
                f"  {r['speedup_BoverA']:>7.3f}x"
                f"  {r['a_us']:>10.3f}"
                f"  {r['b_us']:>10.3f}"
                f"  {chg:>5}"
                f"  {r['kernel_name']}  {r['size_hints']}"
            )

    # ── top regressions (A faster) ────────────────────────────────────────────
    if faster_a:
        print(f"\n  Top 15 regressions  (A faster, sorted by regression desc):")
        print(f"  {'Speedup':>8}  {'A µs':>10}  {'B µs':>10}  {'Cfg?':>5}  Kernel / size_hints")
        print(f"  {'-'*70}")
        for r in sorted(faster_a, key=lambda x: x["speedup_BoverA"])[:15]:
            chg = "diff" if r["config_changed"] == "yes" else "same"
            print(
                f"  {r['speedup_BoverA']:>7.3f}x"
                f"  {r['a_us']:>10.3f}"
                f"  {r['b_us']:>10.3f}"
                f"  {chg:>5}"
                f"  {r['kernel_name']}  {r['size_hints']}"
            )

    # ── config changes ─────────────────────────────────────────────────────────
    if changed:
        print(f"\n  Config changes ({len(changed)} kernels):")
        print(f"  {'Speedup':>8}  Kernel / size_hints")
        print(f"  {'':8}    A config")
        print(f"  {'':8}    B config")
        print(f"  {'-'*70}")
        for r in sorted(changed, key=lambda x: -abs(x["speedup_BoverA"] - 1))[:20]:
            print(f"  {r['speedup_BoverA']:>7.3f}x  {r['kernel_name']}  {r['size_hints']}")
            print(f"  {'':8}    A: {r['a_config']}")
            print(f"  {'':8}    B: {r['b_config']}")

    print()


def write_csv(rows: list[dict], out_path: Path) -> None:
    fields = [
        "kernel_name", "size_hints",
        "a_us", "b_us", "speedup_BoverA",
        "a_config", "b_config", "config_changed",
    ]
    sorted_rows = sorted(rows, key=lambda r: -r["speedup_BoverA"])
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted_rows)


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    a_path = Path(sys.argv[1])
    b_path = Path(sys.argv[2])
    for p in (a_path, b_path):
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(1)

    out_path = (
        Path(sys.argv[3])
        if len(sys.argv) >= 4
        else Path(f"compare_{a_path.stem}_vs_{b_path.stem}.csv")
    )

    a_data = load_csv(a_path)
    b_data = load_csv(b_path)
    print(f"Loaded A: {len(a_data)} rows from {a_path.name}")
    print(f"Loaded B: {len(b_data)} rows from {b_path.name}")

    rows = compare(a_data, b_data)
    print_summary(rows, a_path.name, b_path.name)

    write_csv(rows, out_path)
    print(f"Wrote comparison CSV → {out_path}")


if __name__ == "__main__":
    main()

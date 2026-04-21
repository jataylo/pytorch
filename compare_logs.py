#!/usr/bin/env python3
"""
compare_logs.py  –  Extract and compare benchmark timings from heuristic log files.

Outputs a CSV with raw eager_ms / compile_ms columns so you can see:
  • Did the baseline (eager) change between runs?
  • Did the compile time change independently?
  • What config was chosen (verbose logs only)?

Usage:
    # Single log → CSV
    python compare_logs.py tuning44_verboses_1.log

    # Compare two logs → CSV (A vs B)
    python compare_logs.py tuning35_verboses_1.log tuning44_verboses_1.log

    # Custom output path
    python compare_logs.py -o results.csv tuning35.log tuning44.log

    # Include chosen config column (parsed from verbose logs)
    python compare_logs.py --configs tuning35.log tuning44.log

    # Print summary table to stdout (no CSV)
    python compare_logs.py --summary --no-csv tuning35.log tuning44.log
"""

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ── Regex patterns ─────────────────────────────────────────────────────────────

# "  PHASE 2: Additional Elementwise Operations (5 new ops)"
RE_PHASE = re.compile(r"^\s+PHASE\s+(\d+):\s+(.+)$")

# "  Benchmark 5: Tanh Activation (z = tanh(x))"
RE_BENCHMARK = re.compile(r"^\s+Benchmark\s+(\d+):\s+(.+)$")

# "  1D_tiny | Eager: 0.0145ms | Compile: 0.0244ms | Speedup: 0.592x"
RE_RESULT = re.compile(
    r"^\s+([\w]+)\s+\|\s+Eager:\s+([\d.]+)ms\s+\|\s+Compile:\s+([\d.]+)ms\s+\|\s+Speedup:\s+([\d.]+)x"
)

# "[HEURISTICS] ✓ Chosen: {'XBLOCK': 128}  num_warps=2  n_spills=0  time=0.018521ms ★"
RE_CHOSEN = re.compile(
    r"\[HEURISTICS\].*Chosen:\s+(\{[^}]+\})\s+num_warps=(\d+)"
)

# ── Op name mapping ────────────────────────────────────────────────────────────

_OP_MAP = {
    "elementwise add": "add",
    "elementwise multiply": "mul",
    "relu + sigmoid": "relu_sigmoid",
    "gelu": "gelu",
    "tanh": "tanh",
    "silu": "silu",
    "squared relu": "squared_relu",
    "bias add": "bias_relu",
    "leaky relu": "leaky_relu",
    "2d pointwise": "2d_specific",
    "3d pointwise": "3d_specific",
    "fused pointwise": "fused",
    "heavy fusion - mlp": "heavy_mlp",
    "heavy fusion - attention": "heavy_attn",
    "heavy fusion - conv": "heavy_conv",
    "heavy fusion - branching": "heavy_branching",
}

def _op_name(raw: str) -> str:
    lower = raw.lower()
    for key, name in _OP_MAP.items():
        if lower.startswith(key):
            return name
    words = re.sub(r"\(.*\)", "", raw).strip().split()
    return "_".join(w.lower() for w in words[:2])


def geomean(values: List[float]) -> float:
    vals = [v for v in values if v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# ── Record ─────────────────────────────────────────────────────────────────────

class Record:
    __slots__ = ("phase", "op", "benchmark", "eager_ms", "compile_ms",
                 "speedup", "chosen_cfg")

    def __init__(self, phase, op, benchmark, eager, compile_, speedup, cfg=""):
        self.phase      = phase
        self.op         = op
        self.benchmark  = benchmark
        self.eager_ms   = eager
        self.compile_ms = compile_
        self.speedup    = speedup
        self.chosen_cfg = cfg  # e.g. "XBLOCK=128 num_warps=2" (verbose only)


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_log(path: str) -> List[Record]:
    """
    Parse a benchmark log.  Works for both compact (N=1) and verbose logs.
    In verbose logs the chosen config is extracted from the
    '✓ Chosen: ...' line that appears just before each result line.
    """
    records: List[Record] = []
    current_phase   = "unknown"
    current_op      = "unknown"
    pending_cfg     = ""   # last ✓ Chosen line seen

    with open(path) as fh:
        for line in fh:
            m = RE_PHASE.match(line)
            if m:
                current_phase = f"phase{m.group(1)}"
                continue

            m = RE_BENCHMARK.match(line)
            if m:
                current_op = _op_name(m.group(2))
                continue

            m = RE_CHOSEN.search(line)
            if m:
                # Compact form: "XBLOCK=128 num_warps=2 [waves_per_eu=2]"
                xblock_dict = m.group(1)   # e.g. "{'XBLOCK': 128, 'waves_per_eu': 2}"
                num_warps   = m.group(2)
                # Pull out XBLOCK value
                xm = re.search(r"'XBLOCK':\s*(\d+)", xblock_dict)
                xb = xm.group(1) if xm else "?"
                # Also look for YBLOCK / ZBLOCK / waves_per_eu
                ym   = re.search(r"'YBLOCK':\s*(\d+)", xblock_dict)
                zm   = re.search(r"'ZBLOCK':\s*(\d+)", xblock_dict)
                wpem = re.search(r"'waves_per_eu':\s*(\d+)", xblock_dict)
                parts = [f"XBLOCK={xb}"]
                if ym:
                    parts.append(f"YBLOCK={ym.group(1)}")
                if zm:
                    parts.append(f"ZBLOCK={zm.group(1)}")
                parts.append(f"num_warps={num_warps}")
                if wpem and wpem.group(1) != "0":
                    parts.append(f"wpe={wpem.group(1)}")
                pending_cfg = " ".join(parts)
                continue

            m = RE_RESULT.match(line)
            if m:
                records.append(Record(
                    phase      = current_phase,
                    op         = current_op,
                    benchmark  = m.group(1),
                    eager      = float(m.group(2)),
                    compile_   = float(m.group(3)),
                    speedup    = float(m.group(4)),
                    cfg        = pending_cfg,
                ))
                pending_cfg = ""  # consumed
                continue

    return records


# ── Aggregation ────────────────────────────────────────────────────────────────

Key = Tuple[str, str, str]   # (phase, op, benchmark)

def aggregate(records: List[Record]) -> Dict[Key, Record]:
    """
    Group by (phase, op, benchmark) and compute means/geomeans.
    For repeated runs (verbose logs with many kernel iterations) the
    means across all repetitions are used.
    The chosen_cfg is taken from the first non-empty occurrence.
    """
    buckets: Dict[Key, List[Record]] = defaultdict(list)
    for r in records:
        buckets[(r.phase, r.op, r.benchmark)].append(r)

    agg: Dict[Key, Record] = {}
    for key, recs in buckets.items():
        cfg = next((r.chosen_cfg for r in recs if r.chosen_cfg), "")
        agg[key] = Record(
            phase      = key[0],
            op         = key[1],
            benchmark  = key[2],
            eager      = sum(r.eager_ms   for r in recs) / len(recs),
            compile_   = sum(r.compile_ms for r in recs) / len(recs),
            speedup    = geomean([r.speedup for r in recs]),
            cfg        = cfg,
        )
    return agg


# ── CSV writers ────────────────────────────────────────────────────────────────

SINGLE_BASE = ["phase", "op", "benchmark",
               "eager_ms", "compile_ms", "speedup_x"]
SINGLE_CFG  = SINGLE_BASE + ["chosen_cfg"]

COMPARE_BASE = [
    "phase", "op", "benchmark",
    # Eager times – tells you if the baseline changed
    "eager_A_ms", "eager_B_ms", "eager_delta_pct",
    # Compile times – tells you if heuristic quality changed
    "compile_A_ms", "compile_B_ms", "compile_delta_pct",
    # Within-run speedup (each compile vs its own eager)
    "speedup_A_x", "speedup_B_x", "speedup_delta_x",
    # Cross-baseline speedups – pin one baseline, swap the compile
    # to isolate whether a "regression" is real or just baseline drift:
    #   spd_AvsA = eager_A / compile_A  (same as speedup_A)
    #   spd_BvsA = eager_A / compile_B  ← B's compile vs OLD baseline (fair comparison)
    #   spd_AvB  = eager_B / compile_A  ← A's compile vs NEW baseline
    #   spd_BvsB = eager_B / compile_B  (same as speedup_B)
    # If spd_BvsA > spd_AvsA the apparent regression is baseline-only, not a heuristic loss.
    "spd_AvsA", "spd_BvsA", "spd_AvB", "spd_BvsB",
]
COMPARE_CFG = COMPARE_BASE + ["chosen_cfg_A", "chosen_cfg_B"]


def _pct(a: float, b: float) -> str:
    if a <= 0:
        return "N/A"
    return f"{(b - a) / a * 100:+.1f}%"


def write_single_csv(agg: Dict[Key, Record], path: str, configs: bool) -> None:
    headers = SINGLE_CFG if configs else SINGLE_BASE
    rows = sorted(agg.values(), key=lambda r: (r.phase, r.op, r.benchmark))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            row = {
                "phase":       r.phase,
                "op":          r.op,
                "benchmark":   r.benchmark,
                "eager_ms":    f"{r.eager_ms:.4f}",
                "compile_ms":  f"{r.compile_ms:.4f}",
                "speedup_x":   f"{r.speedup:.4f}",
            }
            if configs:
                row["chosen_cfg"] = r.chosen_cfg
            w.writerow(row)
    print(f"[OK] Wrote {len(rows)} rows → {path}")


def write_compare_csv(agg_a: Dict[Key, Record], agg_b: Dict[Key, Record],
                      path: str, label_a: str, label_b: str,
                      configs: bool) -> None:
    headers = COMPARE_CFG if configs else COMPARE_BASE
    all_keys = sorted(set(agg_a) | set(agg_b))
    rows = []
    for key in all_keys:
        ra = agg_a.get(key)
        rb = agg_b.get(key)
        phase, op, bench = key

        row: Dict = {"phase": phase, "op": op, "benchmark": bench}

        if ra and rb:
            # Cross-baseline speedups: eager_X / compile_Y
            def _spd(eager, compile_):
                return f"{eager / compile_:.4f}" if compile_ > 0 else "N/A"

            row.update({
                "eager_A_ms":        f"{ra.eager_ms:.4f}",
                "eager_B_ms":        f"{rb.eager_ms:.4f}",
                "eager_delta_pct":   _pct(ra.eager_ms, rb.eager_ms),
                "compile_A_ms":      f"{ra.compile_ms:.4f}",
                "compile_B_ms":      f"{rb.compile_ms:.4f}",
                "compile_delta_pct": _pct(ra.compile_ms, rb.compile_ms),
                "speedup_A_x":       f"{ra.speedup:.4f}",
                "speedup_B_x":       f"{rb.speedup:.4f}",
                "speedup_delta_x":   f"{rb.speedup / ra.speedup:+.4f}",
                # Cross-baseline: pin baseline, swap compile — reveals real heuristic delta
                "spd_AvsA": _spd(ra.eager_ms, ra.compile_ms),  # = speedup_A
                "spd_BvsA": _spd(ra.eager_ms, rb.compile_ms),  # B's compile vs OLD baseline
                "spd_AvB":  _spd(rb.eager_ms, ra.compile_ms),  # A's compile vs NEW baseline
                "spd_BvsB": _spd(rb.eager_ms, rb.compile_ms),  # = speedup_B
            })
        elif ra:
            row.update({
                "eager_A_ms": f"{ra.eager_ms:.4f}", "eager_B_ms": "N/A",
                "eager_delta_pct": "N/A",
                "compile_A_ms": f"{ra.compile_ms:.4f}", "compile_B_ms": "N/A",
                "compile_delta_pct": "N/A",
                "speedup_A_x": f"{ra.speedup:.4f}", "speedup_B_x": "N/A",
                "speedup_delta_x": "N/A",
                "spd_AvsA": f"{ra.speedup:.4f}", "spd_BvsA": "N/A",
                "spd_AvB": "N/A", "spd_BvsB": "N/A",
            })
        else:
            row.update({
                "eager_A_ms": "N/A", "eager_B_ms": f"{rb.eager_ms:.4f}",
                "eager_delta_pct": "N/A",
                "compile_A_ms": "N/A", "compile_B_ms": f"{rb.compile_ms:.4f}",
                "compile_delta_pct": "N/A",
                "speedup_A_x": "N/A", "speedup_B_x": f"{rb.speedup:.4f}",
                "speedup_delta_x": "N/A",
                "spd_AvsA": "N/A", "spd_BvsA": "N/A",
                "spd_AvB": "N/A", "spd_BvsB": f"{rb.speedup:.4f}",
            })

        if configs:
            row["chosen_cfg_A"] = ra.chosen_cfg if ra else ""
            row["chosen_cfg_B"] = rb.chosen_cfg if rb else ""

        rows.append(row)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows → {path}  ({label_a} vs {label_b})")


# ── Summary printer ────────────────────────────────────────────────────────────

def print_summary(agg_a: Dict[Key, Record],
                  agg_b: Optional[Dict[Key, Record]],
                  label_a: str, label_b: str) -> None:

    def by_op(agg: Dict) -> Dict[str, List]:
        d: Dict[str, List] = defaultdict(list)
        for key, r in agg.items():
            d[key[1]].append(r)
        return d

    ops_a = by_op(agg_a)
    ops_b = by_op(agg_b) if agg_b else {}
    all_ops = sorted(set(ops_a) | set(ops_b))

    def _fmt(v):
        return f"{v:.4f}" if not math.isnan(v) else "   N/A"
    def _fdelta(a, b):
        return f"{(b-a)/a*100:+.1f}%" if (not math.isnan(a) and a > 0) else "  N/A"

    if agg_b:
        # Table 1: raw timings + within-run speedup
        print(f"\n{'Op':<20}  {'eager_A':>8}  {'eager_B':>8}  {'Δeager':>8}  "
              f"{'cmp_A':>8}  {'cmp_B':>8}  {'Δcmp':>8}  {'spd_A':>7}  {'spd_B':>7}  {'Δspd':>7}")
        print("-" * 108)
        all_eager_A, all_eager_B, all_cmp_A, all_cmp_B = [], [], [], []
        all_spd_A, all_spd_B = [], []

        op_data = {}
        for op in all_ops:
            ea = geomean([r.eager_ms   for r in ops_a.get(op, [])])
            eb = geomean([r.eager_ms   for r in ops_b.get(op, [])])
            ca = geomean([r.compile_ms for r in ops_a.get(op, [])])
            cb = geomean([r.compile_ms for r in ops_b.get(op, [])])
            sa = geomean([r.speedup    for r in ops_a.get(op, [])])
            sb = geomean([r.speedup    for r in ops_b.get(op, [])])
            op_data[op] = (ea, eb, ca, cb, sa, sb)

            print(f"  {op:<18}  {_fmt(ea):>8}  {_fmt(eb):>8}  {_fdelta(ea,eb):>8}  "
                  f"{_fmt(ca):>8}  {_fmt(cb):>8}  {_fdelta(ca,cb):>8}  "
                  f"{_fmt(sa):>7}  {_fmt(sb):>7}  {_fdelta(sa,sb):>7}")

            for lst, v in [(all_eager_A, ea), (all_eager_B, eb), (all_cmp_A, ca),
                           (all_cmp_B, cb), (all_spd_A, sa), (all_spd_B, sb)]:
                if not math.isnan(v):
                    lst.append(v)

        print("-" * 108)
        ea_o = geomean(all_eager_A); eb_o = geomean(all_eager_B)
        ca_o = geomean(all_cmp_A);   cb_o = geomean(all_cmp_B)
        sa_o = geomean(all_spd_A);   sb_o = geomean(all_spd_B)
        print(f"  {'OVERALL':<18}  {ea_o:.4f}  {eb_o:.4f}  {(eb_o-ea_o)/ea_o*100:+.1f}%  "
              f"{ca_o:.4f}  {cb_o:.4f}  {(cb_o-ca_o)/ca_o*100:+.1f}%  "
              f"{sa_o:.3f}x  {sb_o:.3f}x  {(sb_o-sa_o)/sa_o*100:+.1f}%")

        # Compute per-op eager drift to decide which verdict method to use
        # BvsA_impr = cmp_A/cmp_B  — valid only when eager drift is small (<40%)
        # When drift is large the GPU thermal state dominates absolute compile times;
        # use Δspd (within-run speedup ratio delta) as the primary verdict instead.
        HIGH_DRIFT_THRESHOLD = 0.40  # 40% eager drift

        # Detect overall drift level for the header note
        overall_eager_drift = abs(eb_o - ea_o) / ea_o if ea_o > 0 else 0.0
        high_drift_mode = overall_eager_drift > HIGH_DRIFT_THRESHOLD

        # Table 2: cross-baseline speedups
        # spd_AvsA = eA/cA  spd_BvsA = eA/cB  spd_AvB = eB/cA  spd_BvsB = eB/cB
        print(f"\n  Cross-baseline speedups  (pin one baseline, swap compile to isolate real delta)")
        print(f"  spd_AvsA = eager_A/cmp_A   spd_BvsA = eager_A/cmp_B  (← B vs OLD baseline)")
        print(f"  spd_AvB  = eager_B/cmp_A   spd_BvsB = eager_B/cmp_B")
        if high_drift_mode:
            print(f"  ⚠  Overall eager drift {overall_eager_drift*100:+.0f}% — BvsA_impr is unreliable.")
            print(f"     Verdict uses Δspd (within-run speedup ratio B÷A): positive = B improved.\n")
        else:
            print(f"  'BvsA improvement' > 1.0 → B's compile is better, regression is baseline-only\n")
        print(f"{'Op':<20}  {'spd_AvsA':>9}  {'spd_BvsA':>9}  {'spd_AvB':>9}  {'spd_BvsB':>9}  "
              f"{'BvsA_impr':>11}  {'Δspd':>7}  {'verdict':>24}")
        print("-" * 110)
        for op in all_ops:
            ea, eb, ca, cb, sa, sb = op_data[op]
            sAA = ea / ca if ca > 0 else float("nan")
            sBA = ea / cb if cb > 0 else float("nan")
            sAB = eb / ca if ca > 0 else float("nan")
            sBB = eb / cb if cb > 0 else float("nan")
            impr = sBA / sAA if (sAA > 0 and not math.isnan(sAA)) else float("nan")
            delta_spd = (sb - sa) / sa if (sa > 0 and not math.isnan(sa)) else float("nan")
            # Per-op eager drift
            op_eager_drift = abs(eb - ea) / ea if (ea > 0 and not math.isnan(ea)) else 0.0

            # Choose verdict method: use Δspd when eager drift is high for this op
            use_delta_spd = op_eager_drift > HIGH_DRIFT_THRESHOLD
            if math.isnan(impr) or math.isnan(delta_spd):
                verdict = "N/A"
            elif use_delta_spd:
                # Verdict based on within-run speedup ratio delta
                if delta_spd >= -0.02:
                    verdict = "✓ baseline drift only"
                elif delta_spd >= -0.08:
                    verdict = "~ minor regression"
                else:
                    verdict = "✗ TRUE REGRESSION"
            else:
                # Original BvsA_impr verdict (valid when drift is small)
                if impr >= 0.99:
                    verdict = "✓ baseline drift only"
                elif impr >= 0.95:
                    verdict = "~ minor regression"
                else:
                    verdict = "✗ TRUE REGRESSION"

            drift_flag = "⚠" if use_delta_spd else " "
            delta_spd_str = f"{delta_spd*100:+.1f}%" if not math.isnan(delta_spd) else "  N/A"
            print(f"  {op:<18}{drift_flag} {_fmt(sAA):>9}  {_fmt(sBA):>9}  {_fmt(sAB):>9}  {_fmt(sBB):>9}  "
                  f"{impr:>+11.3f}  {delta_spd_str:>7}  {verdict:>24}")

        # Overall cross-baseline
        sAA_o = ea_o / ca_o if ca_o > 0 else float("nan")
        sBA_o = ea_o / cb_o if cb_o > 0 else float("nan")
        sAB_o = eb_o / ca_o if ca_o > 0 else float("nan")
        sBB_o = eb_o / cb_o if cb_o > 0 else float("nan")
        impr_o = sBA_o / sAA_o if sAA_o > 0 else float("nan")
        delta_spd_o = (sb_o - sa_o) / sa_o if sa_o > 0 else float("nan")
        delta_spd_o_str = f"{delta_spd_o*100:+.1f}%" if not math.isnan(delta_spd_o) else "  N/A"
        print("-" * 110)
        print(f"  {'OVERALL':<19} {_fmt(sAA_o):>9}  {_fmt(sBA_o):>9}  {_fmt(sAB_o):>9}  "
              f"{_fmt(sBB_o):>9}  {impr_o:>+11.3f}  {delta_spd_o_str:>7}")

        if high_drift_mode:
            print(f"\n  ⚠  High thermal drift detected ({overall_eager_drift*100:.0f}% eager shift).")
            print(f"     Δspd column is the reliable metric: it compares (compile/eager) ratio within each run.")
            print(f"     Positive Δspd = B's heuristics found a proportionally faster config.")

        # Top regressions / improvements by within-run speedup delta (drift-corrected)
        paired = [(key, agg_a[key], agg_b[key])
                  for key in set(agg_a) & set(agg_b)]

        # Compute per-shape speedup delta and absolute compile delta
        speedup_diffs = []
        compile_diffs = []
        for key, ra, rb in paired:
            spd_a = ra.eager_ms / ra.compile_ms if ra.compile_ms > 0 else float("nan")
            spd_b = rb.eager_ms / rb.compile_ms if rb.compile_ms > 0 else float("nan")
            if not (math.isnan(spd_a) or math.isnan(spd_b)):
                d_spd = (spd_b - spd_a) / spd_a  # positive = B improved
                speedup_diffs.append((d_spd, key[1], key[2],
                                      ra.eager_ms, rb.eager_ms,
                                      ra.compile_ms, rb.compile_ms,
                                      spd_a, spd_b))
            compile_diffs.append((rb.compile_ms - ra.compile_ms,
                                  key[1], key[2], ra.compile_ms, rb.compile_ms,
                                  ra.eager_ms, rb.eager_ms))

        # Sort speedup diffs: most-regressed first (most negative)
        speedup_diffs.sort(key=lambda x: x[0])

        print(f"\nTop 10 speedup regressions  (B has worse within-run speedup ratio):")
        print(f"  {'Op':20} {'Benchmark':22}  {'spd_A':>7}  {'spd_B':>7}  {'Δspd':>8}  {'cmp_A':>9}  {'cmp_B':>9}  Δeager")
        for d_spd, op, bench, ea, eb, ca, cb, sa, sb in speedup_diffs[:10]:
            if d_spd >= 0:
                break  # no regressions
            eager_drift_pct = (eb - ea) / ea * 100 if ea > 0 else 0
            print(f"  {op:20} {bench:22}  {sa:7.4f}  {sb:7.4f}  {d_spd*100:+7.1f}%  "
                  f"{ca:.4f}ms  {cb:.4f}ms  ({eager_drift_pct:+.0f}% eager)")

        print(f"\nTop 10 speedup improvements (B has better within-run speedup ratio):")
        print(f"  {'Op':20} {'Benchmark':22}  {'spd_A':>7}  {'spd_B':>7}  {'Δspd':>8}  {'cmp_A':>9}  {'cmp_B':>9}  Δeager")
        for d_spd, op, bench, ea, eb, ca, cb, sa, sb in reversed(speedup_diffs[-10:]):
            if d_spd <= 0:
                break
            eager_drift_pct = (eb - ea) / ea * 100 if ea > 0 else 0
            print(f"  {op:20} {bench:22}  {sa:7.4f}  {sb:7.4f}  {d_spd*100:+7.1f}%  "
                  f"{ca:.4f}ms  {cb:.4f}ms  ({eager_drift_pct:+.0f}% eager)")

        # Also show absolute compile-time table when drift is low (classic view)
        if not high_drift_mode:
            compile_diffs_sorted = sorted(compile_diffs, key=lambda x: x[0], reverse=True)
            print(f"\nTop 10 compile-time regressions  (B slower than A, absolute):")
            for diff, op, bench, ca, cb, ea, eb in compile_diffs_sorted[:10]:
                pct = diff / ca * 100 if ca > 0 else 0
                print(f"  {op:20} {bench:22}  A={ca:.4f}ms  B={cb:.4f}ms  ({pct:+.1f}%)")

            print(f"\nTop 10 compile-time improvements (B faster than A, absolute):")
            for diff, op, bench, ca, cb, ea, eb in compile_diffs_sorted[-10:]:
                pct = diff / ca * 100 if ca > 0 else 0
                print(f"  {op:20} {bench:22}  A={ca:.4f}ms  B={cb:.4f}ms  ({pct:+.1f}%)")

    else:
        # Single log: just per-op eager and compile geomeans
        print(f"\n{'Op':<20}  {'eager_ms':>10}  {'compile_ms':>12}  {'speedup':>9}")
        print("-" * 58)
        for op in all_ops:
            ea = geomean([r.eager_ms   for r in ops_a[op]])
            ca = geomean([r.compile_ms for r in ops_a[op]])
            sa = geomean([r.speedup    for r in ops_a[op]])
            print(f"  {op:<20}  {ea:>10.4f}  {ca:>12.4f}  {sa:>8.3f}x")
        all_recs = [r for rs in ops_a.values() for r in rs]
        print("-" * 58)
        print(f"  {'OVERALL':<20}  {geomean([r.eager_ms for r in all_recs]):>10.4f}"
              f"  {geomean([r.compile_ms for r in all_recs]):>12.4f}"
              f"  {geomean([r.speedup for r in all_recs]):>8.3f}x")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", metavar="LOG",
                    help="One or two log files (two = A vs B comparison).")
    ap.add_argument("-o", "--output", metavar="CSV",
                    help="Output CSV path (auto-named if omitted).")
    ap.add_argument("--configs", action="store_true",
                    help="Extract chosen config from verbose logs into CSV.")
    ap.add_argument("--summary", action="store_true",
                    help="Print per-operation summary table to stdout.")
    ap.add_argument("--no-csv", action="store_true",
                    help="Skip writing CSV (useful with --summary).")
    args = ap.parse_args()

    if len(args.logs) > 2:
        ap.error("Provide at most two log files.")

    def load(path):
        print(f"Parsing {os.path.basename(path)} …", end=" ", flush=True)
        recs = parse_log(path)
        agg  = aggregate(recs)
        has_cfg = any(r.chosen_cfg for r in agg.values())
        print(f"{len(recs)} raw records, {len(agg)} unique benchmarks"
              f"{', configs found ✓' if has_cfg else ''}")
        return agg

    if len(args.logs) == 1:
        agg_a = load(args.logs[0])
        if args.summary:
            print_summary(agg_a, None, os.path.basename(args.logs[0]), "")
        if not args.no_csv:
            stem = os.path.splitext(args.logs[0])[0]
            out  = args.output or f"{stem}.csv"
            write_single_csv(agg_a, out, args.configs)
    else:
        agg_a = load(args.logs[0])
        agg_b = load(args.logs[1])
        label_a = os.path.basename(args.logs[0])
        label_b = os.path.basename(args.logs[1])
        if args.summary:
            print_summary(agg_a, agg_b, label_a, label_b)
        if not args.no_csv:
            stem_a = os.path.splitext(label_a)[0]
            stem_b = os.path.splitext(label_b)[0]
            out = args.output or f"compare_{stem_a}_vs_{stem_b}.csv"
            write_compare_csv(agg_a, agg_b, out, label_a, label_b, args.configs)


if __name__ == "__main__":
    main()

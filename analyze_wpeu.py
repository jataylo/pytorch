#!/usr/bin/env python3
"""
analyze_wpeu.py  --  parse tuning47_verboses_5_nodiverse_wpeu.log and
answer: for which problem types does waves_per_eu actually win?

Extracts per-kernel entries:
  op, shape, elements, ops_per_element, regime, num_blocks,
  chosen_cfg, chosen_ms, best_cfg, best_ms, slowdown,
  best_has_wpe, chosen_has_wpe, wpe_helps
"""
import re, sys, math
from collections import defaultdict

LOG = sys.argv[1] if len(sys.argv) > 1 else "tuning47_verboses_5_nodiverse_wpeu.log"

# ── regexes ────────────────────────────────────────────────────────────────
RE_BENCHMARK = re.compile(r"Benchmark\s+\d+:\s+(.+)")
RE_SHAPE_RESULT = re.compile(
    r"^\s*([\w]+)\s+\|\s+Eager:\s+([\d.]+)ms\s+\|\s+Compile:\s+([\d.]+)ms"
)
RE_OPS = re.compile(r"Ops / element\s*:\s*([\d.]+)\s+\(weighted\)")
RE_ELEMENTS = re.compile(r"([\d,]+)\s+elements\s+·\s+\d+\s+B/scalar")
RE_REGIME = re.compile(r"Regime\s*:\s*(\S+(?:-\S+)*)")
RE_NUM_BLOCKS = re.compile(r"(\d+)\s+blocks\s+\|")
RE_CHOSEN = re.compile(
    r"✓ Chosen:\s+(\{[^}]+\})\s+num_warps=(\d+)\s+.*?time=([\d.]+)ms"
)
RE_BEST = re.compile(
    r"#\s+1\s+([\d.]+)\s+.*?(\{[^}]+\})\s+←\s+best"
)
RE_WPE_IN_CFG = re.compile(r"'waves_per_eu':\s*([1-9]\d*)")
RE_XBLOCK = re.compile(r"'XBLOCK':\s*(\d+)")
RE_NWARPS_CFG = re.compile(r"'num_warps':\s*(\d+)")

def has_wpe(cfg_str):
    m = RE_WPE_IN_CFG.search(cfg_str)
    return int(m.group(1)) if m else 0

def parse_cfg(cfg_str, num_warps_str):
    xb = int(RE_XBLOCK.search(cfg_str).group(1)) if RE_XBLOCK.search(cfg_str) else 0
    nw = int(num_warps_str) if num_warps_str else (
        int(RE_NWARPS_CFG.search(cfg_str).group(1)) if RE_NWARPS_CFG.search(cfg_str) else 0
    )
    wpe = has_wpe(cfg_str)
    return {"XBLOCK": xb, "num_warps": nw, "waves_per_eu": wpe}

# ── parser state machine ───────────────────────────────────────────────────
entries = []   # list of dicts

cur_op       = None
cur_elements = None
cur_ops_pe   = None
cur_regime   = None
cur_shape    = None
pending_chosen = None   # (cfg_str, nw_str, time_ms)

with open(LOG) as fh:
    for line in fh:

        m = RE_BENCHMARK.search(line)
        if m and "Benchmark" in line and "PHASE" not in line:
            cur_op = m.group(1).strip()
            continue

        m = RE_OPS.search(line)
        if m:
            cur_ops_pe = float(m.group(1))
            continue

        m = RE_ELEMENTS.search(line)
        if m:
            cur_elements = int(m.group(1).replace(",", ""))
            continue

        m = RE_REGIME.search(line)
        if m:
            cur_regime = m.group(1)
            continue

        m = RE_CHOSEN.search(line)
        if m:
            pending_chosen = (m.group(1), m.group(2), float(m.group(3)))
            continue

        m = RE_SHAPE_RESULT.match(line)
        if m:
            cur_shape = m.group(1)
            # shape result comes AFTER chosen but BEFORE best — store to use later
            continue

        m = RE_BEST.search(line)
        if m and pending_chosen:
            best_ms   = float(m.group(1))
            best_cfg  = m.group(2)

            # how many blocks did the best config imply?
            # We can infer from elements / XBLOCK
            best_xb = int(RE_XBLOCK.search(best_cfg).group(1)) if RE_XBLOCK.search(best_cfg) else 1
            num_blocks_est = max(1, math.ceil((cur_elements or 1) / best_xb))

            chosen_cfg_str, chosen_nw, chosen_ms = pending_chosen
            slowdown = chosen_ms / best_ms if best_ms > 0 else 1.0

            entries.append({
                "op":        cur_op or "unknown",
                "shape":     cur_shape or "?",
                "elements":  cur_elements or 0,
                "ops_pe":    cur_ops_pe or 0,
                "regime":    cur_regime or "?",
                "num_blocks": num_blocks_est,
                "chosen_cfg": chosen_cfg_str,
                "chosen_nw":  int(chosen_nw),
                "chosen_ms":  chosen_ms,
                "chosen_wpe": has_wpe(chosen_cfg_str),
                "best_cfg":   best_cfg,
                "best_wpe":   has_wpe(best_cfg),
                "best_ms":    best_ms,
                "slowdown":   slowdown,          # chosen / best  (1.0 = perfect)
            })
            pending_chosen = None
            continue

# ── analysis ───────────────────────────────────────────────────────────────
print(f"\nParsed {len(entries)} kernel invocations from {LOG}\n")

# For each entry: did wpe actually help?
# "wpe wins"   = best_wpe > 0  (the fastest config in the whole pool had wpe)
# "wpe chosen" = chosen_wpe > 0 (our heuristic picked a wpe config)
# "wpe hurts"  = best_wpe == 0 AND chosen_wpe > 0 (we chose wpe but no-wpe was faster overall)

wpe_wins_entries   = [e for e in entries if e["best_wpe"] > 0]
wpe_chosen_entries = [e for e in entries if e["chosen_wpe"] > 0]
no_wpe_best        = [e for e in entries if e["best_wpe"] == 0]

print("=" * 80)
print("OVERALL STATS")
print("=" * 80)
print(f"  Total kernel calls     : {len(entries)}")
print(f"  Best config has wpe    : {len(wpe_wins_entries)} ({100*len(wpe_wins_entries)/max(1,len(entries)):.0f}%)")
print(f"  Chosen config has wpe  : {len(wpe_chosen_entries)} ({100*len(wpe_chosen_entries)/max(1,len(entries)):.0f}%)")
print(f"  Best config has no-wpe : {len(no_wpe_best)} ({100*len(no_wpe_best)/max(1,len(entries)):.0f}%)")

# ── breakdown by ops_per_element ──────────────────────────────────────────
print("\n" + "=" * 80)
print("WPE WIN RATE BY ops_per_element (higher ops = more compute-heavy)")
print("=" * 80)
print(f"  {'ops_pe':>8}  {'total':>6}  {'wpe_wins':>9}  {'win_rate':>9}  {'avg_slowdown_when_wpe_chosen':>30}")
print("  " + "-" * 72)

ops_buckets = defaultdict(list)
for e in entries:
    ops_buckets[round(e["ops_pe"])].append(e)

for ops in sorted(ops_buckets):
    bucket = ops_buckets[ops]
    wins   = [e for e in bucket if e["best_wpe"] > 0]
    chosen_wpe_here = [e for e in bucket if e["chosen_wpe"] > 0]
    # slowdown when we chose a wpe config
    sd = [e["slowdown"] for e in chosen_wpe_here]
    avg_sd = sum(sd) / len(sd) if sd else float("nan")
    print(f"  {ops:>8}  {len(bucket):>6}  {len(wins):>9}  {100*len(wins)/max(1,len(bucket)):>8.0f}%  {avg_sd:>30.3f}x")

# ── breakdown by problem size ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("WPE WIN RATE BY PROBLEM SIZE")
print("=" * 80)
print(f"  {'size_bucket':>20}  {'total':>6}  {'wpe_wins':>9}  {'win_rate':>9}")
print("  " + "-" * 52)

def size_label(n):
    if n < 4096:         return "tiny   (<4K)"
    elif n < 65536:      return "small  (4K-64K)"
    elif n < 1048576:    return "medium (64K-1M)"
    elif n < 16777216:   return "large  (1M-16M)"
    else:                return "huge   (>16M)"

size_buckets = defaultdict(list)
for e in entries:
    size_buckets[size_label(e["elements"])].append(e)

for lbl in ["tiny   (<4K)", "small  (4K-64K)", "medium (64K-1M)", "large  (1M-16M)", "huge   (>16M)"]:
    bucket = size_buckets.get(lbl, [])
    if not bucket:
        continue
    wins = [e for e in bucket if e["best_wpe"] > 0]
    print(f"  {lbl:>20}  {len(bucket):>6}  {len(wins):>9}  {100*len(wins)/max(1,len(bucket)):>8.0f}%")

# ── breakdown by regime ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("WPE WIN RATE BY ROOFLINE REGIME")
print("=" * 80)
print(f"  {'regime':>20}  {'total':>6}  {'wpe_wins':>9}  {'win_rate':>9}")
print("  " + "-" * 52)

regime_buckets = defaultdict(list)
for e in entries:
    lbl = e["regime"].split("(")[0].strip()
    regime_buckets[lbl].append(e)

for r, bucket in sorted(regime_buckets.items()):
    wins = [e for e in bucket if e["best_wpe"] > 0]
    print(f"  {r:>20}  {len(bucket):>6}  {len(wins):>9}  {100*len(wins)/max(1,len(bucket)):>8.0f}%")

# ── breakdown by op type ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print("WPE WIN RATE BY OP TYPE")
print("=" * 80)
print(f"  {'op':>30}  {'total':>6}  {'wpe_wins':>9}  {'win%':>6}  {'avg_ops_pe':>10}  {'chosen_wpe_slowdown':>20}")
print("  " + "-" * 90)

op_buckets = defaultdict(list)
for e in entries:
    op_buckets[e["op"]].append(e)

for op in sorted(op_buckets):
    bucket = op_buckets[op]
    wins   = [e for e in bucket if e["best_wpe"] > 0]
    avg_ops = sum(e["ops_pe"] for e in bucket) / len(bucket)
    chosen_wpe = [e for e in bucket if e["chosen_wpe"] > 0]
    sd_chosen = sum(e["slowdown"] for e in chosen_wpe) / len(chosen_wpe) if chosen_wpe else float("nan")
    print(f"  {op[:30]:>30}  {len(bucket):>6}  {len(wins):>9}  {100*len(wins)/max(1,len(bucket)):>5.0f}%  {avg_ops:>10.1f}  {sd_chosen:>20.3f}x")

# ── where chosen_wpe > best (wpe choice was BAD) ─────────────────────────
bad = [e for e in entries if e["chosen_wpe"] > 0 and e["slowdown"] > 1.05]
bad.sort(key=lambda e: -e["slowdown"])
print("\n" + "=" * 80)
print(f"CASES WHERE HEURISTIC CHOSE WPE BUT IT HURT (slowdown > 1.05x)  [{len(bad)} cases]")
print("=" * 80)
print(f"  {'op':>18}  {'shape':>18}  {'elems':>8}  {'ops_pe':>7}  {'slowdown':>9}  chosen_wpe  best_wpe")
print("  " + "-" * 90)
for e in bad[:25]:
    print(f"  {(e['op'] or '')[:18]:>18}  {e['shape']:>18}  {e['elements']:>8,}  "
          f"{e['ops_pe']:>7.0f}  {e['slowdown']:>9.3f}x  "
          f"wpe={e['chosen_wpe']}   best_wpe={e['best_wpe']}")

# ── where wpe won but heuristic did NOT choose it (missed opportunity) ────
missed = [e for e in entries if e["best_wpe"] > 0 and e["chosen_wpe"] == 0 and e["slowdown"] > 1.03]
missed.sort(key=lambda e: -e["slowdown"])
print("\n" + "=" * 80)
print(f"MISSED WPE WINS (best had wpe, heuristic chose no-wpe, slowdown > 1.03x)  [{len(missed)} cases]")
print("=" * 80)
print(f"  {'op':>18}  {'shape':>18}  {'elems':>8}  {'ops_pe':>7}  {'slowdown':>9}  best_wpe   chosen_cfg")
print("  " + "-" * 90)
for e in missed[:25]:
    print(f"  {(e['op'] or '')[:18]:>18}  {e['shape']:>18}  {e['elements']:>8,}  "
          f"{e['ops_pe']:>7.0f}  {e['slowdown']:>9.3f}x  "
          f"wpe={e['best_wpe']}   {e['chosen_cfg'][:40]}")

# ── size-vs-opspe heatmap of wpe win rate ─────────────────────────────────
print("\n" + "=" * 80)
print("HEATMAP: ops_pe (rows) × size_bucket (cols) → wpe win rate")
print("=" * 80)
size_order = ["tiny   (<4K)", "small  (4K-64K)", "medium (64K-1M)", "large  (1M-16M)", "huge   (>16M)"]
size_short  = ["<4K", "4K-64K", "64K-1M", "1M-16M", ">16M"]
heatmap = defaultdict(lambda: defaultdict(list))
for e in entries:
    heatmap[round(e["ops_pe"])][size_label(e["elements"])].append(e)

all_ops = sorted(heatmap.keys())
print(f"  {'ops_pe':>7}  " + "  ".join(f"{s:>10}" for s in size_short))
print("  " + "-" * 70)
for ops in all_ops:
    row = []
    for lbl in size_order:
        bucket = heatmap[ops].get(lbl, [])
        if not bucket:
            row.append("   —")
        else:
            wins = sum(1 for e in bucket if e["best_wpe"] > 0)
            row.append(f"{100*wins//max(1,len(bucket)):>3}%")
    print(f"  {ops:>7}  " + "    ".join(row))

print("\nDone.\n")


"""Tabulate run_matrix.sh output: python analyze_matrix.py results/*.txt

Speedups are best(base, ps) / mode and preagg / mode (geomean, worst, best).
Modes other than base / ps / preagg in results/ are runtime-check variants that
were benchmarked and dropped (preagg always sorts):
  adjN   sort if >= N adjacent tokens in a block share a destination
  shareN sort if the expected number of other tokens sharing each token's
         destination (estimated from all pairs in 16-token groups) is >= N
  pairN  sort if the 16-token groups contain >= N equal-destination pairs
"""

import collections
import math
import sys


def geomean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


rows = collections.defaultdict(dict)
for path in sys.argv[1:]:
    for line in open(path):
        if not line.startswith("RESULT"):
            continue
        op, V, dist, D, mode, ms, *_ = line.split()[1].split(",")
        rows[(op, int(V), dist, int(D))][mode] = float(ms)

modes = sorted({m for r in rows.values() for m in r} - {"base", "ps"})
modes.sort(key=lambda m: m != "preagg")

print(f"{'shape':36s} {'base':>8s} {'ps':>8s} " + " ".join(f"{m:>8s}" for m in modes))
for k in sorted(rows):
    r = rows[k]
    cells = " ".join(f"{r[m]:8.3f}" if m in r else f"{'-':>8s}" for m in modes)
    print(f"{str(k):36s} {r['base']:8.3f} {r['ps']:8.3f} {cells}")

print()
for m in modes:
    shapes = [r for r in rows.values() if m in r]
    vs_best = [min(r["base"], r["ps"]) / r[m] for r in shapes]
    line = (
        f"{m:8s} vs best(base,ps): geomean={geomean(vs_best):.2f}x worst={min(vs_best):.2f}x "
        f"slower={sum(x < 0.97 for x in vs_best)}/{len(shapes)}"
    )
    if m != "preagg" and all("preagg" in r for r in shapes):
        vs_sort = [r["preagg"] / r[m] for r in shapes]
        line += (
            f" | vs preagg: geomean={geomean(vs_sort):.2f}x "
            f"worst={min(vs_sort):.2f}x best={max(vs_sort):.2f}x"
        )
    print(line)

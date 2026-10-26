"""Summarize suite.py results.

  python report.py results/suite.jsonl [--before old.jsonl] [--export-sets prefix] [--no-plots]

Speedups are best(base, ps) / preagg unless noted; > 1 means the template is faster.
--before compares the preagg rows of an earlier run (e.g. before a template
change) against this one on the shapes both contain.
--export-sets writes <prefix>_loss.json (template fired and lost) and
<prefix>_guard.json (largest wins) for hillclimb.py.
"""

import argparse
import collections
import json
import math
import os

SLOWER = 0.97


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def shape_id(r):
    return (r["op"], r["n"], r["v"], r["d"], r["dist"], r["dtype"], r.get("k", 0))


def load(path):
    rows = collections.defaultdict(dict)
    errors = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        r = json.loads(line)
        if "ms" in r:
            rows[shape_id(r)][r["mode"]] = r
        else:
            errors.append(r)
    return rows, errors


def ratio_bucket(r):
    return 4 ** round(math.log(max(r, 1), 4))


def summarize(name, speedups):
    if not speedups:
        return f"{name:28s} {'-':>5s}"
    return (
        f"{name:28s} n={len(speedups):4d} geomean={geomean(speedups):5.2f}x "
        f"worst={min(speedups):5.2f}x best={max(speedups):7.2f}x "
        f"slower={sum(s < SLOWER for s in speedups):3d}"
    )


def tolerance(dtype):
    return {"float32": 1e-4, "float64": 1e-9, "int32": 0.0}.get(dtype, 0.1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("--before")
    p.add_argument("--export-sets")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--worst", type=int, default=25)
    args = p.parse_args()

    rows, errors = load(args.results)
    full = {k: m for k, m in rows.items() if {"base", "ps", "preagg"} <= m.keys()}
    print(
        f"{len(rows)} shapes, {len(full)} with base/ps/preagg, {len(errors)} errored runs"
    )
    for e in errors[:10]:
        print("  ERROR", e["mode"], shape_id(e), e.get("error"))

    def speedup(m):
        return min(m["base"]["ms"], m["ps"]["ms"]) / m["preagg"]["ms"]

    fired = {k: m for k, m in full.items() if m["preagg"]["path"] == "preagg"}
    print()
    print(summarize("all shapes", [speedup(m) for m in full.values()]))
    print(summarize("template fired", [speedup(m) for m in fired.values()]))
    print(
        summarize(
            "template declined", [speedup(m) for k, m in full.items() if k not in fired]
        )
    )
    print(
        summarize(
            "vs base only", [m["base"]["ms"] / m["preagg"]["ms"] for m in full.values()]
        )
    )
    print(
        summarize(
            "ps vs base", [m["base"]["ms"] / m["ps"]["ms"] for m in full.values()]
        )
    )

    def table(title, keyfn, subset=full):
        groups = collections.defaultdict(list)
        for k, m in subset.items():
            groups[keyfn(k, m)].append(speedup(m))
        print(f"\n== {title}")
        for g in sorted(groups, key=str):
            print(summarize(str(g), groups[g]))

    table("by op", lambda k, m: k[0])
    table("by op (fired only)", lambda k, m: k[0], fired)
    table("by contention ratio N/V", lambda k, m: ratio_bucket(k[1] / k[2]))
    table("by distribution", lambda k, m: k[4])
    table("by inner size D", lambda k, m: k[3])
    table("by dtype", lambda k, m: k[5])
    table("by ratio x dist", lambda k, m: (ratio_bucket(k[1] / k[2]), k[4]))

    print(f"\n== worst {args.worst} shapes (op, N, V, D, dist, dtype, k)")
    for k, m in sorted(full.items(), key=lambda km: speedup(km[1]))[: args.worst]:
        print(
            f"{speedup(m):5.2f}x {str(k):70s} base={m['base']['ms']:8.3f} "
            f"ps={m['ps']['ms']:8.3f} preagg={m['preagg']['ms']:8.3f} "
            f"[{m['ps']['path']}/{m['preagg']['path']}]"
        )

    print("\n== accuracy outliers (preagg err > tolerance and > 2x reference modes)")
    n_out = 0
    for k, m in sorted(full.items()):
        e = m["preagg"]["err"]
        ref = max(m["base"]["err"], m["ps"]["err"])
        if e > tolerance(k[5]) and e > 2 * ref:
            n_out += 1
            print(
                f"  {str(k):70s} preagg={e:.1e} base={m['base']['err']:.1e} ps={m['ps']['err']:.1e}"
            )
    print(f"  {n_out} outliers")

    if args.before:
        before, _ = load(args.before)
        common = [
            (before[k]["preagg"]["ms"] / m["preagg"]["ms"], k)
            for k, m in rows.items()
            if "preagg" in m and "preagg" in before.get(k, {})
        ]
        print(f"\n== preagg before / after on {len(common)} common shapes")
        print(summarize("all", [s for s, _ in common]))
        per_op = collections.defaultdict(list)
        for s, k in common:
            per_op[k[0]].append(s)
        for op in sorted(per_op):
            print(summarize(op, per_op[op]))

    if args.export_sets:
        losses = [k for k, m in fired.items() if speedup(m) < SLOWER]
        wins = sorted(fired, key=lambda k: -speedup(fired[k]))[:40]
        for tag, keys in (("loss", losses), ("guard", wins)):
            path = f"{args.export_sets}_{tag}.json"
            with open(path, "w") as f:
                json.dump(
                    [
                        {
                            "shape": list(k),
                            "best_ref_ms": min(
                                full[k]["base"]["ms"], full[k]["ps"]["ms"]
                            ),
                            "preagg_ms": full[k]["preagg"]["ms"],
                        }
                        for k in keys
                    ],
                    f,
                    indent=1,
                )
            print(f"wrote {len(keys)} shapes to {path}")

    if not args.no_plots:
        plot(full, speedup, os.path.splitext(args.results)[0] + "_heatmaps")


def plot(full, speedup, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    by_op = collections.defaultdict(dict)
    for k, m in full.items():
        # fp32 main grid only (skips low-precision and the small-N series)
        if k[5] not in ("float32", "int32") or k[1] < min(1 << 21, (1 << 28) // k[3]):
            continue
        by_op[k[0]].setdefault(k[4], {})[(ratio_bucket(k[1] / k[2]), k[3])] = speedup(m)
    for op, dists in by_op.items():
        dists = {d: g for d, g in dists.items() if len(g) > 1}
        if not dists:
            continue
        fig, axes = plt.subplots(
            1, len(dists), figsize=(3.2 * len(dists), 3.2), squeeze=False
        )
        for ax, (dist, grid) in zip(axes[0], sorted(dists.items())):
            ratios = sorted({r for r, _ in grid})
            ds = sorted({d for _, d in grid})
            z = [
                [math.log2(grid.get((r, d), float("nan"))) for d in ds] for r in ratios
            ]
            im = ax.imshow(
                z, cmap="RdBu", vmin=-3, vmax=3, aspect="auto", origin="lower"
            )
            ax.set_xticks(range(len(ds)), [str(d) for d in ds])
            ax.set_yticks(range(len(ratios)), [str(r) for r in ratios])
            ax.set_xlabel("D")
            ax.set_ylabel("N/V")
            ax.set_title(dist)
            for i, r in enumerate(ratios):
                for j, d in enumerate(ds):
                    if (r, d) in grid:
                        ax.text(
                            j,
                            i,
                            f"{grid[(r, d)]:.2g}",
                            ha="center",
                            va="center",
                            fontsize=7,
                        )
        fig.suptitle(f"{op}: best(base, ps) / preagg  (fp32)")
        fig.colorbar(im, ax=axes[0].tolist(), label="log2 speedup")
        fig.savefig(f"{out_dir}/{op}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"heatmaps in {out_dir}/")


if __name__ == "__main__":
    main()

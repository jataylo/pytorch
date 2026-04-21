#!/usr/bin/env python3
"""
Analyze HEURISTICS VALIDATION blocks from verbose logs to find regressions.

Usage:
  python analyze_regressions.py tuning70_verboses_5_diverse_wpeu_noblock.log [--top N] [--pool]
"""
import re, sys, argparse
from collections import defaultdict

def parse_log(log_path):
    blocks = []
    with open(log_path, 'r', errors='replace') as f:
        lines = f.readlines()

    # First pass: find kernel names from [BENCH_VERBOSE] lines
    # Then associate with next [HEURISTICS VALIDATION] block
    kernel_name = ''
    size_hints  = ''
    i = 0
    while i < len(lines):
        line = lines[i]

        # Track kernel name from BENCH_VERBOSE header lines (n_configs, not individual bench lines)
        bvm = re.search(r'\[BENCH_VERBOSE\] kernel=(\S+)\s+size_hints=({[^}]+})', line)
        if bvm:
            kernel_name = bvm.group(1)
            size_hints  = bvm.group(2)
            i += 1
            continue

        # Start of validation block
        if '[HEURISTICS VALIDATION]' in line:
            # Collect block text until next separator or next BENCH_VERBOSE
            block_lines = []
            j = i
            while j < len(lines):
                if j > i and '[HEURISTICS VALIDATION]' in lines[j]:
                    break
                if j > i and re.search(r'\[BENCH_VERBOSE\] kernel=\S+\s+size_hints=', lines[j]):
                    break
                block_lines.append(lines[j])
                j += 1

            block = ''.join(block_lines)

            # Parse CHOSEN section
            chosen_cfg_m   = re.search(r'📊 HEURISTIC CHOSEN.*?Config\s*:\s*({[^}]+})', block, re.DOTALL)
            chosen_score_m = re.search(r'📊 HEURISTIC CHOSEN.*?Score\s*:\s*([\d.]+)\s*\(rank #(\d+)', block, re.DOTALL)
            chosen_time_m  = re.search(r'📊 HEURISTIC CHOSEN.*?Measured\s*:\s*([\d.]+)\s*ms', block, re.DOTALL)
            chosen_fact_m  = re.search(r'📊 HEURISTIC CHOSEN.*?Factors\s*:\s*([^\n]+)', block, re.DOTALL)

            # Parse ACTUAL section
            actual_cfg_m   = re.search(r'🏆 ACTUAL Best.*?Config\s*:\s*({[^}]+})', block, re.DOTALL)
            actual_score_m = re.search(r'🏆 ACTUAL Best.*?Score\s*:\s*([\d.]+)\s*\(rank #(\d+)', block, re.DOTALL)
            actual_time_m  = re.search(r'🏆 ACTUAL Best.*?Measured\s*:\s*([\d.]+)\s*ms', block, re.DOTALL)

            # Pool size
            pool_m = re.search(r'Selection\s*:\s*winner chosen from heuristic top-(\d+)', block)
            total_m = re.search(r'Configs\s*:\s*(\d+) scored.*?(\d+) benchmarked', block)

            # Benchmark Results table
            bench_rows = re.findall(
                r'#\s*(\d+)\s+([\d.]+)\s+#\s*(\d+)\s+([\d.]+)\s+({[^}]+})\s*(.*)',
                block
            )

            if not (chosen_cfg_m and actual_cfg_m and chosen_time_m and actual_time_m):
                i = j
                continue

            chosen_time = float(chosen_time_m.group(1)) * 1000  # ms→µs
            actual_time = float(actual_time_m.group(1)) * 1000

            speedup = chosen_time / actual_time if actual_time > 0 else float('nan')

            # Extract XBLOCK, num_warps from cfg strings
            def xb(s): m = re.search(r"'XBLOCK':\s*(\d+)", s); return int(m.group(1)) if m else 0
            def nw(s): m = re.search(r"'num_warps':\s*(\d+)", s); return int(m.group(1)) if m else 0
            def wpe(s): m = re.search(r"'waves_per_eu':\s*(\d+)", s); return int(m.group(1)) if m else 0
            def fmt(s): return f"x={xb(s)} nw={nw(s)}" + (f" wpe={wpe(s)}" if wpe(s) else "")

            chosen_cfg = chosen_cfg_m.group(1)
            actual_cfg = actual_cfg_m.group(1)

            # Check if actual best was in pool (marked with ★ in bench results table)
            # In the table, pool configs have ★ at end
            actual_in_pool = False
            bench_data = []
            for row in bench_rows:
                rank_bench = int(row[0])
                time_bench = float(row[1]) * 1000
                pred_rank  = int(row[2])
                score_bench = float(row[3])
                cfg_bench  = row[4]
                flags      = row[5]
                in_pool    = '★' in flags
                is_best    = '← best' in flags
                bench_data.append({
                    'rank': rank_bench, 'time': time_bench, 'pred_rank': pred_rank,
                    'score': score_bench, 'cfg': cfg_bench, 'in_pool': in_pool
                })
                if is_best and in_pool:
                    actual_in_pool = True

            # Determine if actual best was in pool (pool is actual best in_pool)
            # Also check: is actual best xb/nw the same as any pool entry?
            actual_xb = xb(actual_cfg)
            actual_nw_v = nw(actual_cfg)
            for bd in bench_data:
                if bd['in_pool'] and xb(bd['cfg']) == actual_xb and nw(bd['cfg']) == actual_nw_v:
                    actual_in_pool = True

            blocks.append({
                'kernel':        kernel_name,
                'sizes':         size_hints,
                'chosen_cfg':    chosen_cfg,
                'chosen_xb':     xb(chosen_cfg),
                'chosen_nw':     nw(chosen_cfg),
                'chosen_score':  float(chosen_score_m.group(1)) if chosen_score_m else 0,
                'chosen_rank':   int(chosen_score_m.group(2)) if chosen_score_m else 0,
                'chosen_time':   chosen_time,
                'chosen_factors': chosen_fact_m.group(1).strip() if chosen_fact_m else '',
                'actual_cfg':    actual_cfg,
                'actual_xb':     xb(actual_cfg),
                'actual_nw':     nw(actual_cfg),
                'actual_score':  float(actual_score_m.group(1)) if actual_score_m else 0,
                'actual_rank':   int(actual_score_m.group(2)) if actual_score_m else 0,
                'actual_time':   actual_time,
                'speedup':       speedup,   # > 1 means chosen is SLOWER (regression)
                'actual_in_pool': actual_in_pool,
                'pool_n':        int(pool_m.group(1)) if pool_m else 0,
                'total_scored':  int(total_m.group(1)) if total_m else 0,
                'bench':         bench_data,
                'fmt_chosen':    fmt(chosen_cfg),
                'fmt_actual':    fmt(actual_cfg),
            })
            i = j
            continue

        i += 1
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logfile')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--pool', action='store_true', help='Show benchmark table for each regression')
    ap.add_argument('--threshold', type=float, default=1.02, help='Regression threshold')
    args = ap.parse_args()

    blocks = parse_log(args.logfile)
    print(f"Parsed {len(blocks)} validation blocks\n")

    regressions = sorted([b for b in blocks if b['speedup'] > args.threshold],
                         key=lambda b: b['speedup'], reverse=True)
    improvements = [b for b in blocks if b['speedup'] < 1/args.threshold]
    neutral = [b for b in blocks if 1/args.threshold <= b['speedup'] <= args.threshold]

    print(f"Kernels: {len(blocks)}  Regressions(>{args.threshold:.0%}): {len(regressions)}  "
          f"Improvements: {len(improvements)}  Same: {len(neutral)}")
    print()

    # ── Top regressions ──────────────────────────────────────────────────────
    print(f"{'='*80}")
    print(f"  TOP-{args.top} REGRESSIONS  (chosen_time / actual_best_time)")
    print(f"{'='*80}")
    for i, b in enumerate(regressions[:args.top], 1):
        pool_tag = '' if b['actual_in_pool'] else '  [ACTUAL NOT IN POOL]'
        xb_dir = ''
        if b['actual_xb'] > b['chosen_xb']: xb_dir = f' ↑XBLOCK({b["chosen_xb"]}→{b["actual_xb"]})'
        elif b['actual_xb'] < b['chosen_xb']: xb_dir = f' ↓XBLOCK({b["chosen_xb"]}→{b["actual_xb"]})'
        if b['actual_nw'] != b['chosen_nw'] and not xb_dir:
            xb_dir = f' nw({b["chosen_nw"]}→{b["actual_nw"]})'

        print(f"#{i:02d}  {b['speedup']:.3f}x  {b['kernel']}  {b['sizes']}{pool_tag}")
        print(f"      Chosen  rank#{b['chosen_rank']:2d}: {b['fmt_chosen']:20s}  "
              f"score={b['chosen_score']:.4f}  {b['chosen_time']:.3f}µs")
        print(f"      Actual  rank#{b['actual_rank']:2d}: {b['fmt_actual']:20s}  "
              f"score={b['actual_score']:.4f}  {b['actual_time']:.3f}µs{xb_dir}")
        print(f"      Factors: {b['chosen_factors']}")

        if args.pool and b['bench']:
            def _xb(s): m = re.search(r"'XBLOCK':\s*(\d+)", s); return int(m.group(1)) if m else 0
            def _nw(s): m = re.search(r"'num_warps':\s*(\d+)", s); return int(m.group(1)) if m else 0
            pool_configs = [bd for bd in b['bench'] if bd['in_pool']]
            non_pool = sorted([bd for bd in b['bench'] if not bd['in_pool']], key=lambda x: x['time'])
            print(f"      Pool configs (benchmarked in top-N):")
            for bd in sorted(pool_configs, key=lambda x: x['time']):
                mark = '★' if (_xb(bd['cfg']) == b['actual_xb'] and _nw(bd['cfg']) == b['actual_nw']) else ' '
                print(f"        {mark} {bd['time']:.3f}µs  pr#{bd['pred_rank']:2d}  "
                      f"sc={bd['score']:.4f}  {bd['cfg']}")
            if non_pool:
                print(f"      Best NOT in pool:")
                for bd in non_pool[:3]:
                    mark = '★' if (_xb(bd['cfg']) == b['actual_xb'] and _nw(bd['cfg']) == b['actual_nw']) else ' '
                    print(f"        {mark} {bd['time']:.3f}µs  pr#{bd['pred_rank']:2d}  "
                          f"sc={bd['score']:.4f}  {bd['cfg']}")
        print()

    # ── Pattern analysis ─────────────────────────────────────────────────────
    print(f"{'='*80}")
    print("  PATTERN ANALYSIS")
    print(f"{'='*80}")

    not_in_pool = sum(1 for b in regressions if not b['actual_in_pool'])
    in_pool_wrong = len(regressions) - not_in_pool
    print(f"\nActual best NOT in heuristic pool: {not_in_pool}/{len(regressions)} regressions")
    print(f"Actual best in pool but chose wrong from pool: {in_pool_wrong}/{len(regressions)} regressions")

    # XBLOCK direction
    up   = sum(1 for b in regressions if b['actual_xb'] > b['chosen_xb'])
    down = sum(1 for b in regressions if b['actual_xb'] < b['chosen_xb'])
    same = len(regressions) - up - down
    print(f"\nXBLOCK change direction:")
    print(f"  Actual best needs LARGER XBLOCK:  {up}")
    print(f"  Actual best needs SMALLER XBLOCK: {down}")
    print(f"  Same XBLOCK, diff num_warps:      {same}")

    # Score tier of actual best
    high = sum(1 for b in regressions if b['actual_score'] >= 0.97)
    mid  = sum(1 for b in regressions if 0.93 <= b['actual_score'] < 0.97)
    low  = sum(1 for b in regressions if b['actual_score'] < 0.93)
    print(f"\nActual-best predicted score:")
    print(f"  ≥ 0.97 (scoring bug — top tier but not rank #1): {high}")
    print(f"  0.93–0.97 (pool boundary issue):                 {mid}")
    print(f"  < 0.93 (needs guarantee to enter pool):          {low}")

    # Rank of actual best
    print(f"\nPredicted rank of actual best (in regressions):")
    rank_hist = defaultdict(int)
    for b in regressions:
        rank_hist[b['actual_rank']] += 1
    for r in sorted(rank_hist):
        bar = '█' * rank_hist[r]
        print(f"  rank {r:3d}: {rank_hist[r]:3d}  {bar}")

    # Chosen rank
    print(f"\nHeuristic chosen predicted rank (in regressions):")
    cho_hist = defaultdict(int)
    for b in regressions:
        cho_hist[b['chosen_rank']] += 1
    for r in sorted(cho_hist):
        bar = '█' * cho_hist[r]
        print(f"  rank {r:3d}: {cho_hist[r]:3d}  {bar}")

    # Factor breakdown for in-pool wrong choices
    print(f"\nFactor analysis for in-pool-but-wrong-choice regressions:")
    factor_issues = defaultdict(int)
    for b in [b for b in regressions if b['actual_in_pool']]:
        factors = b['chosen_factors']
        bw_m  = re.search(r'BW=([\d.]+)', factors)
        occ_m = re.search(r'Occup=([\d.]+)', factors)
        grid_m= re.search(r'Grid=([\d.]+)', factors)
        bw    = float(bw_m.group(1)) if bw_m else 1.0
        occ   = float(occ_m.group(1)) if occ_m else 1.0
        grid  = float(grid_m.group(1)) if grid_m else 1.0
        if occ > 0.95 and b['actual_nw'] < b['chosen_nw']:
            factor_issues['Occupancy over-scored for high-nw'] += 1
        if grid > 0.99 and b['actual_xb'] > b['chosen_xb']:
            factor_issues['Grid over-scored for small XBLOCK'] += 1
        if bw > 0.99 and b['actual_xb'] > b['chosen_xb']:
            factor_issues['BW over-scored for small XBLOCK'] += 1
    for issue, cnt in sorted(factor_issues.items(), key=lambda x: -x[1]):
        print(f"  {cnt:3d}  {issue}")


if __name__ == '__main__':
    main()

#!/usr/bin/env bash
# run_benchmark_production.sh — Benchmark heuristic modes WITHOUT real-bench.
#
# In this mode each heuristic variant compiles only its top-N candidates and
# benchmarks them normally (no exhaustive search).  This reflects actual
# production behaviour: faster startup, no validation overhead.
#
# Output tables
# ─────────────
#  1. Per-op speedup geomean   (rows = op, cols = mode)   → op_speedup.txt + .csv
#  2. Per-bench execution time (rows = op/shape, cols = mode) → exec_time.txt + .csv
#  3. Summary                  geomean vs max-autotune + total run time per mode
#
# Usage:
#   ./run_benchmark_production.sh [options]
#
# Options:
#   --modes    "1 3 4 5 6"   Modes to run (default: 1 2 3 4 5 6)
#   --top-n    "1 5"          TOP_N values for heuristic modes (default: 1 5)
#   --iters    N              Benchmark iterations per shape (default: 5)
#   --warmup   N              Warmup iterations (default: 10)
#   --quick                   Use fewer shapes (faster)
#   --outdir   DIR            Output directory (default: bench_prod_<timestamp>)
#   --bench-rep N             do_bench rep count for multi-candidate selection
#                             (TORCHINDUCTOR_HEURISTICS_BENCH_REP, default: 40)

set -uo pipefail

# ─── colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
BLU='\033[0;34m'; CYN='\033[0;36m'; BLD='\033[1m'; RST='\033[0m'

log()  { echo -e "${BLU}[suite]${RST} $*"; }
ok()   { echo -e "${GRN}[  ok ]${RST} $*"; }
warn() { echo -e "${YLW}[ warn]${RST} $*"; }
err()  { echo -e "${RED}[error]${RST} $*"; }
sep()  { echo -e "${CYN}$(printf '═%.0s' {1..80})${RST}"; }

# ─── defaults ────────────────────────────────────────────────────────────────
MODES="1 2 3 4 5 6"
TOP_N_VALUES="1 3 5"
ITERS=5
WARMUP=10
QUICK=""
OUTDIR=""
BENCH_REP=40

# ─── argument parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --modes)    MODES="$2";        shift 2 ;;
        --top-n)    TOP_N_VALUES="$2"; shift 2 ;;
        --iters)    ITERS="$2";        shift 2 ;;
        --warmup)   WARMUP="$2";       shift 2 ;;
        --quick)    QUICK="--quick";   shift ;;
        --outdir)   OUTDIR="$2";       shift 2 ;;
        --bench-rep) BENCH_REP="$2";   shift 2 ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$OUTDIR" ]] && OUTDIR="bench_prod_$(date +%Y%m%d_%H%M%S)"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${REPO_DIR}/benchmark_pointwise.py"

[[ -f "$BENCH_SCRIPT" ]] || { err "benchmark_pointwise.py not found in ${REPO_DIR}"; exit 1; }

mkdir -p "$OUTDIR"

# ─── GPU / ROCm detection ────────────────────────────────────────────────────
GPU="unknown"; ROCM_VER="unknown"
if command -v rocminfo &>/dev/null; then
    _rocminfo="$(rocminfo 2>/dev/null)" || true
    GPU="$(echo "$_rocminfo" | awk '
        /Agent Type.*:.*GPU/ { in_gpu=1 }
        in_gpu && /Marketing Name/ { sub(/.*: */, ""); print; exit }
        /^HSA Agent/ { in_gpu=0 }
    ')" || true
    ROCM_VER="$(echo "$_rocminfo" | grep 'Runtime Version' | head -1 | sed 's/.*: *//')" || true
fi
[[ -z "$GPU" ]]      && GPU="unknown"
[[ -z "$ROCM_VER" ]] && ROCM_VER="unknown"

# ─── python runner (strips local repo from sys.path) ─────────────────────────
run_py() {
    local script="$1"; shift
    local repo_dir="$REPO_DIR"
    python3 - "$@" <<PYEOF
import sys, os, runpy

sys.path = [p for p in sys.path if '${repo_dir}' not in p]

_BENCH_COMPILE_MODE = os.environ.get('_BENCH_COMPILE_MODE', 'max-autotune-no-cudagraphs')

import torch as _torch
_orig_compile = _torch.compile
def _patched_compile(fn=None, *args, **kwargs):
    if kwargs.get('mode') == 'max-autotune':
        kwargs['mode'] = _BENCH_COMPILE_MODE
    if fn is None:
        return _orig_compile(*args, **kwargs)
    return _orig_compile(fn, *args, **kwargs)
_torch.compile = _patched_compile

sys.argv = ['${script}'] + sys.argv[1:]
runpy.run_path('${script}', run_name='__main__')
PYEOF
}
export -f run_py

# ─── mode definitions (REAL_BENCH=0 throughout) ───────────────────────────────
declare -A mode_name mode_env mode_is_heuristic

mode_name[1]="Max Autotune"
mode_env[1]="TORCHINDUCTOR_POINTWISE_HEURISTICS=0 _BENCH_COMPILE_MODE=max-autotune-no-cudagraphs"
mode_is_heuristic[1]=0

mode_name[2]="No Tune"
mode_env[2]="TORCHINDUCTOR_MAX_AUTOTUNE=0 TORCHINDUCTOR_POINTWISE_HEURISTICS=0 _BENCH_COMPILE_MODE=default"
mode_is_heuristic[2]=0

# All heuristic modes: REAL_BENCH=0 → only top-N configs compiled & benchmarked
mode_name[3]="Heuristics WPEU+Diversity"
mode_env[3]="TORCHINDUCTOR_POINTWISE_HEURISTICS=1 TORCHINDUCTOR_POINTWISE_WAVES_PER_EU=1 TORCHINDUCTOR_HEURISTICS_DIVERSITY=1 TORCHINDUCTOR_HEURISTICS_REAL_BENCH=0"
mode_is_heuristic[3]=1

mode_name[4]="Heuristics Diversity"
mode_env[4]="TORCHINDUCTOR_POINTWISE_HEURISTICS=1 TORCHINDUCTOR_POINTWISE_WAVES_PER_EU=0 TORCHINDUCTOR_HEURISTICS_DIVERSITY=1 TORCHINDUCTOR_HEURISTICS_REAL_BENCH=0"
mode_is_heuristic[4]=1

mode_name[5]="Heuristics WPEU"
mode_env[5]="TORCHINDUCTOR_POINTWISE_HEURISTICS=1 TORCHINDUCTOR_POINTWISE_WAVES_PER_EU=1 TORCHINDUCTOR_HEURISTICS_DIVERSITY=0 TORCHINDUCTOR_HEURISTICS_REAL_BENCH=0"
mode_is_heuristic[5]=1

mode_name[6]="Heuristics"
mode_env[6]="TORCHINDUCTOR_POINTWISE_HEURISTICS=1 TORCHINDUCTOR_POINTWISE_WAVES_PER_EU=0 TORCHINDUCTOR_HEURISTICS_DIVERSITY=0 TORCHINDUCTOR_HEURISTICS_REAL_BENCH=0"
mode_is_heuristic[6]=1

# ─── header ──────────────────────────────────────────────────────────────────
sep
echo -e "${BLD}  POINTWISE PRODUCTION BENCHMARK SUITE${RST}  (REAL_BENCH=0)"
sep
echo -e "  Output dir  : ${BLD}${OUTDIR}${RST}"
echo -e "  Modes       : ${BLD}${MODES}${RST}"
echo -e "  TOP_N vals  : ${BLD}${TOP_N_VALUES}${RST}  (heuristic modes only)"
echo -e "  Iters       : ${BLD}${ITERS}${RST}"
echo -e "  Warmup      : ${BLD}${WARMUP}${RST}"
echo -e "  Bench rep   : ${BLD}${BENCH_REP}${RST}  (do_bench rep for multi-candidate selection)"
echo -e "  Quick mode  : ${BLD}${QUICK:-no}${RST}"
echo -e "  GPU         : ${BLD}${GPU}${RST}"
echo -e "  ROCm        : ${BLD}${ROCM_VER}${RST}"
sep

SUITE_START=$SECONDS

# run_keys / metadata arrays
run_keys=()
declare -A rk_label rk_log rk_csv rk_rc rk_is_heuristic rk_elapsed

# ─── run_one KEY LABEL ENV_VARS TOP_N IS_HEURISTIC ───────────────────────────
run_one() {
    local key="$1" label="$2" env_vars="$3" top_n="$4" is_heur="$5"
    local log_file="${OUTDIR}/${key}.log"
    local csv_file="${OUTDIR}/${key}.csv"

    rk_label[$key]="$label"
    rk_log[$key]="$log_file"
    rk_csv[$key]="$csv_file"
    rk_is_heuristic[$key]="$is_heur"

    sep
    echo -e "${BLD}  ${key}: ${label}${RST}"
    sep
    log "iters=${ITERS} warmup=${WARMUP} top_n=${top_n} bench_rep=${BENCH_REP}"

    local T0=$SECONDS
    (
        for pair in $env_vars; do export "$pair"; done
        [[ "$top_n" != "-" ]] && export TORCHINDUCTOR_HEURISTICS_TOP_N="$top_n"
        export TORCHINDUCTOR_HEURISTICS_BENCH_REP="${BENCH_REP}"
        run_py "$BENCH_SCRIPT" \
            --iters   "$ITERS" \
            --warmup  "$WARMUP" \
            --csv     "$csv_file" \
            ${QUICK}
    ) 2>&1 | tee "$log_file"
    local RC=${PIPESTATUS[0]}
    rk_rc[$key]=$RC
    rk_elapsed[$key]=$(( SECONDS - T0 ))

    if [[ $RC -eq 0 ]]; then
        ok "${key} finished in ${rk_elapsed[$key]}s  →  ${log_file}"
    else
        err "${key} FAILED (rc=${RC}) after ${rk_elapsed[$key]}s"
    fi
}

# ─── dispatch ────────────────────────────────────────────────────────────────
for M in $MODES; do
    [[ -z "${mode_name[$M]+x}" ]] && { warn "Unknown mode $M, skipping."; continue; }

    if [[ "${mode_is_heuristic[$M]}" -eq 0 ]]; then
        key="mode${M}"
        run_keys+=("$key")
        run_one "$key" "${mode_name[$M]}" "${mode_env[$M]}" "-" 0
    else
        for N in $TOP_N_VALUES; do
            key="mode${M}_n${N}"
            run_keys+=("$key")
            run_one "$key" "${mode_name[$M]} [top_n=${N}]" "${mode_env[$M]}" "$N" 1
        done
    fi
done

# ─── build elapsed map for the analysis script ───────────────────────────────
ELAPSED_MAP=""
for key in "${run_keys[@]}"; do
    ELAPSED_MAP="${ELAPSED_MAP}'${key}':${rk_elapsed[$key]},"
done

# ─── analysis & tables ───────────────────────────────────────────────────────
sep
echo -e "${BLD}  ANALYSIS  —  generating tables${RST}"
sep

PY_KEYS=""
PY_LABELS=""
for key in "${run_keys[@]}"; do
    PY_KEYS="${PY_KEYS}'${key}',"
    PY_LABELS="${PY_LABELS}'${key}':'${rk_label[$key]}',"
done

python3 - <<PYEOF
import sys, os, csv, math
from collections import defaultdict
from statistics import geometric_mean, mean

sys.path = [p for p in sys.path if '${REPO_DIR}' not in p]

outdir   = '${OUTDIR}'
keys     = [${PY_KEYS}]
labels   = {${PY_LABELS}}
elapsed  = {${ELAPSED_MAP}}   # wall-clock seconds per mode

# ── load CSVs ────────────────────────────────────────────────────────────────
# data[key][(op,shape)] = {'eager_ms': float, 'compile_ms': float}
data = {}
for key in keys:
    path = os.path.join(outdir, f'{key}.csv')
    data[key] = {}
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                bench = (row['op'], row['shape'])
                data[key][bench] = {
                    'eager_ms':   float(row['eager_ms']),
                    'compile_ms': float(row['compile_ms']),
                    'numel':      int(row.get('numel', 0)),
                }
            except (KeyError, ValueError):
                pass

all_benches = sorted({b for k in data for b in data[k]}, key=lambda x: (x[0], x[1]))

# shared eager baseline: average eager_ms across all keys
eager_avg = {}
for bench in all_benches:
    samples = [data[k][bench]['eager_ms'] for k in keys if bench in data[k]]
    if samples:
        eager_avg[bench] = mean(samples)

# ── helper: speedup relative to eager ────────────────────────────────────────
def speedup(key, bench):
    if bench not in data[key] or bench not in eager_avg:
        return None
    ctime = data[key][bench]['compile_ms']
    if ctime <= 0:
        return None
    return eager_avg[bench] / ctime

# ── helper: speedup relative to mode1 (max autotune) ─────────────────────────
mode1_key = 'mode1' if 'mode1' in keys else None

def vs_maxautotune(key, bench):
    if mode1_key is None or bench not in data.get(mode1_key, {}):
        return None
    base = data[mode1_key][bench]['compile_ms']
    ctime = data[key][bench]['compile_ms'] if bench in data[key] else None
    if base is None or ctime is None or base <= 0 or ctime <= 0:
        return None
    return base / ctime   # >1 means faster than max-autotune

# ── TABLE 1: per-op speedup geomean (vs eager) ───────────────────────────────
ops = sorted({b[0] for b in all_benches})
op_speedups = {}   # op -> key -> [speedup values]
for bench in all_benches:
    op = bench[0]
    for key in keys:
        s = speedup(key, bench)
        if s:
            op_speedups.setdefault(op, {}).setdefault(key, []).append(s)

def short(k):
    return k.replace('mode', 'm').replace('_n', '/n')

col_hdrs = [short(k) for k in keys]
cw = max(len(h) for h in col_hdrs) + 1
ow = 22

divider = '  ' + '-'*ow + '-'*(cw * len(keys))

header_line  = f"\n  {'Op':<{ow}}" + ''.join(f'{h:>{cw}}' for h in col_hdrs)

print('\n')
print('═'*80)
print('  TABLE 1 — Per-Op Speedup vs Eager Baseline  (geomean across shapes)')
print('═'*80)
print(header_line)
print(divider)
for op in ops:
    row = f'  {op:<{ow}}'
    for key in keys:
        vals = op_speedups.get(op, {}).get(key, [])
        row += f'{geometric_mean(vals):>{cw}.3f}' if vals else f'{"–":>{cw}}'
    print(row)
print(divider)
# overall geomean row
all_speedups = {key: [] for key in keys}
for bench in all_benches:
    for key in keys:
        s = speedup(key, bench)
        if s:
            all_speedups[key].append(s)
row = f'  {"OVERALL (geomean)":<{ow}}'
for key in keys:
    v = all_speedups[key]
    row += f'{geometric_mean(v):>{cw}.3f}' if v else f'{"–":>{cw}}'
print(row)

# save table 1 to csv
t1_csv = os.path.join(outdir, 'op_speedup.csv')
with open(t1_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['op'] + keys)
    for op in ops:
        row_vals = [op]
        for key in keys:
            vals = op_speedups.get(op, {}).get(key, [])
            row_vals.append(f'{geometric_mean(vals):.4f}' if vals else '')
        w.writerow(row_vals)
    overall_row = ['OVERALL']
    for key in keys:
        v = all_speedups[key]
        overall_row.append(f'{geometric_mean(v):.4f}' if v else '')
    w.writerow(overall_row)
print(f'\n  Saved → {t1_csv}')

# ── TABLE 2: per-bench execution time in µs (compile_ms × 1000) ──────────────
print('\n')
print('═'*80)
print('  TABLE 2 — Per-Benchmark Execution Time (µs, compiled kernel, lower = better)')
print('═'*80)

bw = 38   # bench name column width
t2_header = f"  {'op / shape':<{bw}}" + ''.join(f'{short(k):>{cw}}' for k in keys)
print(t2_header)
print('  ' + '-'*bw + '-'*(cw*len(keys)))

prev_op = None
for bench in all_benches:
    op, shape = bench
    if op != prev_op:
        if prev_op is not None:
            print(f'  {"":.<{bw}}' + '.'*(cw*len(keys)))
        prev_op = op

    name = f'{op}/{shape}'
    row  = f'  {name:<{bw}}'
    for key in keys:
        if bench in data[key]:
            us = data[key][bench]['compile_ms'] * 1000
            row += f'{us:>{cw}.1f}'
        else:
            row += f'{"–":>{cw}}'
    print(row)

# save table 2 to csv
t2_csv = os.path.join(outdir, 'exec_time.csv')
with open(t2_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['op', 'shape', 'numel'] + keys)
    for bench in all_benches:
        op, shape = bench
        numel = next((data[k][bench]['numel'] for k in keys if bench in data[k]), '')
        row_vals = [op, shape, numel]
        for key in keys:
            if bench in data[key]:
                row_vals.append(f'{data[key][bench]["compile_ms"]*1000:.2f}')
            else:
                row_vals.append('')
        w.writerow(row_vals)
print(f'\n  Saved → {t2_csv}')

# ── SUMMARY: geomean speedup vs max-autotune + run time ──────────────────────
print('\n')
print('═'*80)
print('  SUMMARY — Geomean Speedup vs Max Autotune  +  Total Run Time')
print('═'*80)

# vs-max-autotune geomean
vma = {}
for key in keys:
    ratios = []
    for bench in all_benches:
        r = vs_maxautotune(key, bench)
        if r:
            ratios.append(r)
    vma[key] = geometric_mean(ratios) if ratios else None

mode1_elapsed = elapsed.get('mode1', 0)

# column widths for summary table
lw = 36   # label column
nw = 9    # numeric columns

print(f"\n  {'Mode':<{lw}} {'vs Eager':>{nw}} {'vs MaxAuto':>{nw}} {'Run time':>{nw}} {'vs MaxAuto':>{nw}}")
print(f"  {'':.<{lw}} {'-'*nw} {'-'*nw} {'-'*nw} {'-'*nw}")
for key in keys:
    label    = labels.get(key, key)
    spd      = geometric_mean(all_speedups[key]) if all_speedups[key] else None
    vma_val  = vma[key]
    wall     = elapsed.get(key, 0)
    wall_rel = wall / mode1_elapsed if mode1_elapsed > 0 else None

    spd_s     = f'{spd:.4f}x'      if spd      else '–'
    vma_s     = f'{vma_val:+.1%}'  if vma_val  else '–'
    wall_s    = f'{wall}s'
    wrel_s    = f'{wall_rel:.2f}x' if wall_rel else '–'

    # mark best/worst in the vs-MaxAuto column
    flag = ''
    if vma_val and vma_val > 1.02:
        flag = '  ✓ faster'
    elif vma_val and vma_val < 0.98:
        flag = '  ✗ slower'

    print(f"  {label:<{lw}} {spd_s:>{nw}} {vma_s:>{nw}} {wall_s:>{nw}} {wrel_s:>{nw}}{flag}")

# regressions vs max autotune (>5% slower)
if mode1_key:
    print(f'\n  Regressions vs max-autotune  (compile_ms > 1.05 × mode1):')
    print(f"  {'Mode':<{lw}} {'Count':>6}  {'Worst regression':>18}")
    print(f"  {'':.<{lw}} {'------':>6}  {'------------------':>18}")
    for key in keys:
        if key == mode1_key:
            continue
        regressions = []
        for bench in all_benches:
            r = vs_maxautotune(key, bench)
            if r and r < 1/1.05:   # more than 5% slower than max-autotune
                regressions.append((1/r - 1, bench))
        regressions.sort(reverse=True)
        cnt = len(regressions)
        worst_s = f'{regressions[0][1][0]}/{regressions[0][1][1]}  ({regressions[0][0]:+.1%})' \
                  if regressions else '(none)'
        print(f"  {labels.get(key, key):<{lw}} {cnt:>6}  {worst_s:>18}")

# save summary csv
sum_csv = os.path.join(outdir, 'summary.csv')
with open(sum_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['key', 'label', 'geomean_vs_eager', 'geomean_vs_maxautotune',
                'run_time_s', 'run_time_vs_maxautotune'])
    for key in keys:
        spd     = geometric_mean(all_speedups[key]) if all_speedups[key] else ''
        vma_val = vma[key] or ''
        wall    = elapsed.get(key, '')
        wrel    = (elapsed.get(key, 0) / mode1_elapsed) if mode1_elapsed > 0 else ''
        w.writerow([key, labels.get(key, key),
                    f'{spd:.4f}' if spd else '',
                    f'{vma_val:.4f}' if vma_val else '',
                    wall,
                    f'{wrel:.3f}' if wrel else ''])
print(f'\n  Saved → {sum_csv}')

PYEOF

# ─── final status ────────────────────────────────────────────────────────────
sep
echo -e "${BLD}  RESULTS${RST}"
sep
TOTAL=$(( SECONDS - SUITE_START ))
for key in "${run_keys[@]}"; do
    RC="${rk_rc[$key]}"
    EL="${rk_elapsed[$key]}"
    if [[ $RC -eq 0 ]]; then
        echo -e "  ${key}  ${GRN}PASS${RST}  ${EL}s  log=${rk_log[$key]}"
    else
        echo -e "  ${key}  ${RED}FAIL (rc=${RC})${RST}  ${EL}s  log=${rk_log[$key]}"
    fi
done
echo ""
echo -e "  Output files:"
echo -e "    ${OUTDIR}/op_speedup.csv    — per-op speedup geomean (Table 1)"
echo -e "    ${OUTDIR}/exec_time.csv     — per-bench execution time µs (Table 2)"
echo -e "    ${OUTDIR}/summary.csv       — summary geomean + run time"
echo ""
echo -e "  Total time : ${BLD}${TOTAL}s${RST}"
echo -e "  Results dir: ${BLD}${OUTDIR}/${RST}"
sep

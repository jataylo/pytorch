#!/usr/bin/env bash
# run_benchmark_suite.sh — Run benchmark_pointwise.py across heuristic configurations.
#
# Modes 1 & 2  — baseline (no heuristics), run once.
# Modes 3/4/5  — heuristic variants, each run with TOP_N = 1, 5, 10.
#
# Usage:
#   ./run_benchmark_suite.sh [options]
#
# Options:
#   --modes    "1 3 5"   Modes to run (default: 1 2 3 4 5)
#   --top-n    "1 5 10"  TOP_N values for heuristic modes (default: 1 5 10)
#   --iters    N         Benchmark iterations (default: 100)
#   --warmup   N         Warmup iterations (default: 20)
#   --quick              Use fewer shapes (faster)
#   --outdir   DIR       Output directory (default: bench_results_<timestamp>)
#   --skip-analysis      Skip analyze_regressions after benchmarking
#   --threshold F        Regression threshold (default: 1.05)
#   --top      N         Top-N regressions to print (default: 20)

set -uo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
BLU='\033[0;34m'; CYN='\033[0;36m'; BLD='\033[1m'; RST='\033[0m'

log()  { echo -e "${BLU}[suite]${RST} $*"; }
ok()   { echo -e "${GRN}[  ok ]${RST} $*"; }
warn() { echo -e "${YLW}[ warn]${RST} $*"; }
err()  { echo -e "${RED}[error]${RST} $*"; }
sep()  { echo -e "${CYN}$(printf '═%.0s' {1..80})${RST}"; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODES="1 2 3 4 5 6"
TOP_N_VALUES="1 5 10"
ITERS=5
WARMUP=100
QUICK=""
OUTDIR=""
SKIP_ANALYSIS=0
THRESHOLD=1.05
TOP=20

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --modes)         MODES="$2";         shift 2 ;;
        --top-n)         TOP_N_VALUES="$2";  shift 2 ;;
        --iters)         ITERS="$2";         shift 2 ;;
        --warmup)        WARMUP="$2";        shift 2 ;;
        --quick)         QUICK="--quick";    shift ;;
        --outdir)        OUTDIR="$2";        shift 2 ;;
        --skip-analysis) SKIP_ANALYSIS=1;    shift ;;
        --threshold)     THRESHOLD="$2";     shift 2 ;;
        --top)           TOP="$2";           shift 2 ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$OUTDIR" ]] && OUTDIR="bench_results_$(date +%Y%m%d_%H%M%S)"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${REPO_DIR}/benchmark_pointwise.py"
ANALYZE_SCRIPT="${REPO_DIR}/analyze_regressions.py"

[[ -f "$BENCH_SCRIPT" ]]   || { err "benchmark_pointwise.py not found"; exit 1; }
[[ -f "$ANALYZE_SCRIPT" ]] || { err "analyze_regressions.py not found"; exit 1; }

mkdir -p "$OUTDIR"

# ---------------------------------------------------------------------------
# GPU / ROCm detection
# ---------------------------------------------------------------------------
GPU="unknown"; ROCM_VER="unknown"
if command -v rocminfo &>/dev/null; then
    _rocminfo="$(rocminfo 2>/dev/null)" || true
    GPU="$(echo "$_rocminfo"     | grep 'Marketing Name'  | head -1 | sed 's/.*: *//')" || true
    ROCM_VER="$(echo "$_rocminfo" | grep 'Runtime Version' | head -1 | sed 's/.*: *//')" || true
fi
[[ -z "$GPU" ]]      && GPU="unknown"
[[ -z "$ROCM_VER" ]] && ROCM_VER="unknown"

# ---------------------------------------------------------------------------
# Python runner — strips repo root from sys.path so installed torch is used
# ---------------------------------------------------------------------------
run_py() {
    local script="$1"; shift
    local repo_dir="$REPO_DIR"
    python3 - "$@" <<PYEOF
import sys, os, runpy

# Strip repo root so the installed torch is found, not the local source tree.
sys.path = [p for p in sys.path if '${repo_dir}' not in p]

# Mode-aware torch.compile monkey-patch.
#
# benchmark_pointwise.py always calls torch.compile(mode='max-autotune'), which
# (even after the cudagraph patch below) injects {"max_autotune": True} into the
# inductor config, completely overriding TORCHINDUCTOR_MAX_AUTOTUNE=0.
#
# _BENCH_COMPILE_MODE lets each benchmark mode control the effective compile mode:
#   'max-autotune-no-cudagraphs'  →  full autotune, no CUDA graphs (modes 1, 3-6)
#   'default'                      →  no autotune at all (mode 2: true no-tune)
#
# When the var is absent, we fall back to 'max-autotune-no-cudagraphs' (safe default).
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

# ---------------------------------------------------------------------------
# Mode base definitions
# ---------------------------------------------------------------------------
declare -A mode_name mode_env mode_is_heuristic

mode_name[1]="Max Autotune  (no heuristics)"
mode_env[1]="TORCHINDUCTOR_POINTWISE_HEURISTICS=0 _BENCH_COMPILE_MODE=max-autotune-no-cudagraphs"
mode_is_heuristic[1]=0

mode_name[2]="No Tune        (no heuristics)"
# _BENCH_COMPILE_MODE=default redirects torch.compile(mode='max-autotune') → 'default',
# which does NOT set max_autotune=True. Without this, the compile call would inject
# {"max_autotune": True} directly, completely overriding TORCHINDUCTOR_MAX_AUTOTUNE=0.
mode_env[2]="TORCHINDUCTOR_MAX_AUTOTUNE=0 TORCHINDUCTOR_POINTWISE_HEURISTICS=0 _BENCH_COMPILE_MODE=default"
mode_is_heuristic[2]=0

mode_name[3]="Heuristics + WPEU + Diversity"
mode_env[3]="TORCHINDUCTOR_POINTWISE_HEURISTICS=1 TORCHINDUCTOR_POINTWISE_WAVES_PER_EU=1 TORCHINDUCTOR_HEURISTICS_DIVERSITY=1 TORCHINDUCTOR_HEURISTICS_REAL_BENCH=1 TORCHINDUCTOR_HEURISTICS_VERBOSE=1"
mode_is_heuristic[3]=1

mode_name[4]="Heuristics  No WPEU + Diversity"
mode_env[4]="TORCHINDUCTOR_POINTWISE_HEURISTICS=1 TORCHINDUCTOR_POINTWISE_WAVES_PER_EU=0 TORCHINDUCTOR_HEURISTICS_DIVERSITY=1 TORCHINDUCTOR_HEURISTICS_REAL_BENCH=1 TORCHINDUCTOR_HEURISTICS_VERBOSE=1"
mode_is_heuristic[4]=1

mode_name[5]="Heuristics + WPEU  No Diversity"
mode_env[5]="TORCHINDUCTOR_POINTWISE_HEURISTICS=1 TORCHINDUCTOR_POINTWISE_WAVES_PER_EU=1 TORCHINDUCTOR_HEURISTICS_DIVERSITY=0 TORCHINDUCTOR_HEURISTICS_REAL_BENCH=1 TORCHINDUCTOR_HEURISTICS_VERBOSE=1"
mode_is_heuristic[5]=1

mode_name[6]="Heuristics  No WPEU  No Diversity"
mode_env[6]="TORCHINDUCTOR_POINTWISE_HEURISTICS=1 TORCHINDUCTOR_POINTWISE_WAVES_PER_EU=0 TORCHINDUCTOR_HEURISTICS_DIVERSITY=0 TORCHINDUCTOR_HEURISTICS_REAL_BENCH=1 TORCHINDUCTOR_HEURISTICS_VERBOSE=1"
mode_is_heuristic[6]=1

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
sep
echo -e "${BLD}  POINTWISE BENCHMARK SUITE${RST}"
sep
echo -e "  Output dir  : ${BLD}${OUTDIR}${RST}"
echo -e "  Modes       : ${BLD}${MODES}${RST}"
echo -e "  TOP_N vals  : ${BLD}${TOP_N_VALUES}${RST}  (for heuristic modes 3/4/5)"
echo -e "  Iters       : ${BLD}${ITERS}${RST}"
echo -e "  Warmup      : ${BLD}${WARMUP}${RST}"
echo -e "  Quick mode  : ${BLD}${QUICK:-no}${RST}"
echo -e "  Analysis    : ${BLD}$( [[ $SKIP_ANALYSIS -eq 1 ]] && echo no || echo yes )${RST}"
echo -e "  GPU         : ${BLD}${GPU}${RST}"
echo -e "  ROCm        : ${BLD}${ROCM_VER}${RST}"
sep

SUITE_START=$SECONDS

# run_keys tracks every (mode, n) combination we actually ran, in order.
# Format: "M" for baseline modes, "M_nN" for heuristic modes.
run_keys=()
declare -A rk_label rk_log rk_csv rk_rc rk_is_heuristic

# ---------------------------------------------------------------------------
# run_one KEY LABEL ENV_VARS TOP_N IS_HEURISTIC
# ---------------------------------------------------------------------------
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
    log "iters=${ITERS} warmup=${WARMUP} top_n=${top_n}"

    local T0=$SECONDS
    (
        for pair in $env_vars; do export "$pair"; done
        [[ "$top_n" != "-" ]] && export TORCHINDUCTOR_HEURISTICS_TOP_N="$top_n"
        run_py "$BENCH_SCRIPT" \
            --iters "$ITERS" \
            --warmup "$WARMUP" \
            --csv "$csv_file" \
            ${QUICK}
    ) 2>&1 | tee "$log_file"
    local RC=${PIPESTATUS[0]}
    rk_rc[$key]=$RC

    local ELAPSED=$(( SECONDS - T0 ))
    if [[ $RC -eq 0 ]]; then
        ok "${key} finished in ${ELAPSED}s  →  ${log_file}"
    else
        err "${key} FAILED (rc=${RC}) after ${ELAPSED}s"
    fi
}

# ---------------------------------------------------------------------------
# Dispatch all runs
# ---------------------------------------------------------------------------
for M in $MODES; do
    [[ -z "${mode_name[$M]+x}" ]] && { warn "Unknown mode $M, skipping."; continue; }

    if [[ "${mode_is_heuristic[$M]}" -eq 0 ]]; then
        # Baseline mode — single run, no TOP_N sweep
        key="mode${M}"
        run_keys+=("$key")
        run_one "$key" "${mode_name[$M]}" "${mode_env[$M]}" "-" 0
    else
        # Heuristic mode — run once per TOP_N value
        for N in $TOP_N_VALUES; do
            key="mode${M}_n${N}"
            run_keys+=("$key")
            run_one "$key" "${mode_name[$M]}  [top_n=${N}]" "${mode_env[$M]}" "$N" 1
        done
    fi
done

# ---------------------------------------------------------------------------
# Regression analysis (heuristic runs only)
# ---------------------------------------------------------------------------
if [[ $SKIP_ANALYSIS -eq 0 ]]; then
    sep
    echo -e "${BLD}  REGRESSION ANALYSIS${RST}"
    sep

    for key in "${run_keys[@]}"; do
        [[ "${rk_is_heuristic[$key]}" -eq 0 ]] && continue
        local_log="${rk_log[$key]}"
        [[ ! -f "$local_log" ]] && { warn "No log for ${key}, skipping."; continue; }

        echo ""
        echo -e "${BLD}── ${key}: ${rk_label[$key]} ──${RST}"
        analysis_out="${OUTDIR}/${key}_analysis.txt"
        run_py "$ANALYZE_SCRIPT" "$local_log" \
            --top "$TOP" \
            --threshold "$THRESHOLD" \
            2>&1 | tee "$analysis_out"
        ok "Analysis → ${analysis_out}"
    done
fi

# ---------------------------------------------------------------------------
# Cross-run speedup comparison (shared eager baseline)
# ---------------------------------------------------------------------------
sep
echo -e "${BLD}  CROSS-RUN SPEEDUP SUMMARY${RST}"
sep

# Build Python lists from bash arrays
PY_KEYS=""; PY_LABELS=""
for key in "${run_keys[@]}"; do
    PY_KEYS="${PY_KEYS}'${key}',"
    PY_LABELS="${PY_LABELS}'${key}': '${rk_label[$key]}',"
done

python3 - <<PYEOF
import sys, os, csv, math
from collections import defaultdict
from statistics import geometric_mean, mean

sys.path = [p for p in sys.path if '${REPO_DIR}' not in p]

outdir = '${OUTDIR}'
keys   = [${PY_KEYS}]
labels = {${PY_LABELS}}

# ── Load all CSVs ────────────────────────────────────────────────────────────
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
                }
            except (KeyError, ValueError):
                pass

# ── Shared eager baseline: average eager_ms per bench across all runs ────────
all_benches = sorted({b for k in data for b in data[k]},
                     key=lambda x: (x[0], x[1]))

eager_avg = {}
for bench in all_benches:
    samples = [data[k][bench]['eager_ms'] for k in keys if bench in data[k]]
    if samples:
        eager_avg[bench] = mean(samples)

# ── Geomean summary table ─────────────────────────────────────────────────────
print(f"\n  {'Run':<20} {'Label':<34} {'Benches':>7}  {'Geomean speedup':>17}")
print(f"  {'-'*20} {'-'*34} {'-'*7}  {'-'*17}")
for key in keys:
    speedups = []
    for bench, baseline in eager_avg.items():
        if bench in data[key] and data[key][bench]['compile_ms'] > 0 and baseline > 0:
            speedups.append(baseline / data[key][bench]['compile_ms'])
    label = labels.get(key, key)
    if speedups:
        gm = geometric_mean(speedups)
        print(f"  {key:<20} {label:<34} {len(speedups):>7}  {gm:>17.4f}x")
    else:
        print(f"  {key:<20} {label:<34} {'N/A':>7}  {'N/A':>17}")

# ── Per-benchmark table ───────────────────────────────────────────────────────
# Saved to CSV and printed to terminal (condensed: op geomean across shapes)
per_bench_csv = os.path.join(outdir, 'per_benchmark_comparison.csv')

# Full per-(op,shape) CSV
with open(per_bench_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['op', 'shape', 'numel', 'avg_eager_ms'] + keys
    writer.writerow(header)
    for bench in all_benches:
        op, shape = bench
        baseline = eager_avg.get(bench)
        if baseline is None:
            continue
        # numel from any run that has this bench
        numel = next((int(data[k][bench].get('numel', 0))
                      for k in keys if bench in data[k]), '')
        row_vals = []
        for key in keys:
            if bench in data[key] and data[key][bench]['compile_ms'] > 0:
                row_vals.append(f"{baseline / data[key][bench]['compile_ms']:.4f}")
            else:
                row_vals.append('')
        writer.writerow([op, shape, numel, f"{baseline:.6f}"] + row_vals)

print(f"\n  Full per-benchmark table → {per_bench_csv}")

# ── Per-op geomean terminal table (rows=op, cols=run) ─────────────────────────
ops = sorted({b[0] for b in all_benches})
op_speedups = {}  # op -> {key -> [speedups]}
for bench in all_benches:
    op, shape = bench
    baseline = eager_avg.get(bench)
    if baseline is None:
        continue
    for key in keys:
        if bench in data[key] and data[key][bench]['compile_ms'] > 0 and baseline > 0:
            op_speedups.setdefault(op, {}).setdefault(key, []).append(
                baseline / data[key][bench]['compile_ms'])

if op_speedups:
    # Truncate long key names for column headers
    def short(k): return k.replace('mode', 'm').replace('_n', '/n')
    col_keys = keys
    col_hdrs = [short(k) for k in col_keys]
    cw = max(len(h) for h in col_hdrs) + 1

    print(f"\n  Per-op geomean speedup  (shared eager baseline, higher = better)")
    hdr = f"  {'Op':<22}" + "".join(f"{h:>{cw}}" for h in col_hdrs)
    print(hdr)
    print("  " + "-"*22 + "-"*(cw * len(col_keys)))
    for op in ops:
        row = f"  {op:<22}"
        for key in col_keys:
            vals = op_speedups.get(op, {}).get(key, [])
            if vals:
                row += f"{geometric_mean(vals):>{cw}.3f}"
            else:
                row += f"{'N/A':>{cw}}"
        print(row)
PYEOF

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
sep
echo -e "${BLD}  SUMMARY${RST}"
sep
TOTAL=$(( SECONDS - SUITE_START ))
for key in "${run_keys[@]}"; do
    RC="${rk_rc[$key]}"
    if [[ $RC -eq 0 ]]; then
        echo -e "  ${key}  ${GRN}PASS${RST}  log=${rk_log[$key]}"
    else
        echo -e "  ${key}  ${RED}FAIL (rc=${RC})${RST}  log=${rk_log[$key]}"
    fi
done
echo ""
echo -e "  Total time  : ${BLD}${TOTAL}s${RST}"
echo -e "  Results dir : ${BLD}${OUTDIR}/${RST}"
sep

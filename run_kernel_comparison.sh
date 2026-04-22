#!/usr/bin/env bash
# run_kernel_comparison.sh — Per-kernel autotune vs heuristics comparison.
#
# ┌─ Phase 1 ───────────────────────────────────────────────────────────────────┐
# │  Run benchmark_pointwise.py ONCE with full autotuning and no heuristics     │
# │  into a single shared TORCHINDUCTOR_CACHE_DIR.                              │
# │  TORCHINDUCTOR_BENCHMARK_KERNEL=1 appends a standalone benchmark harness    │
# │  to every compiled Triton kernel .py file.                                  │
# │  The autotune cache is keyed by hash(config_list) — so Phase 2 heuristic   │
# │  runs (different config lists) use different cache slots automatically.     │
# └─────────────────────────────────────────────────────────────────────────────┘
# ┌─ Phase 2 ───────────────────────────────────────────────────────────────────┐
# │  run_phase2_benchmarks.py scans the shared cache for .py files with the     │
# │  benchmark harness.  For each kernel it:                                    │
# │    1. Patches device=hip/cuda → device='cuda:0'  (codegen bug fix)         │
# │    2. For no_tune mode: patches 'max_autotune': True → False               │
# │    3. Runs `python <patched_kernel>` for every mode                         │
# │  autotune mode = cache hit from Phase 1 → zero re-autotuning               │
# │  heuristic modes = mini-autotune of top-N configs in isolation              │
# └─────────────────────────────────────────────────────────────────────────────┘
# ┌─ Analysis ──────────────────────────────────────────────────────────────────┐
# │  analyze_kernel_comparison.py pivots the flat JSONL into a wide CSV.        │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   ./run_kernel_comparison.sh [options]
#
# Options:
#   --top-n      "1 5 10"  TOP_N sweep for heuristic modes    (default: 1 5 10)
#   --iters      N         benchmark_pointwise outer loop reps (default: 3)
#   --rep        N         triton do_bench reps in Phase 2     (default: 100)
#   --timeout    N         per-kernel Phase 2 timeout (sec)    (default: 180)
#   --quick                pass --quick to benchmark_pointwise
#   --outdir     DIR       output directory (default: kern_cmp_<timestamp>)
#   --from-cache DIR       run Phase 2 only, using an existing cache dir
#   --distributed          benchmark kernels in parallel across all GPUs
#   --gpus       N         override GPU count for --distributed
#   --verbose              print full stdout/stderr for every kernel run
#   --skip-phase1          skip Phase 1 (reuse existing cache in --outdir)
#   --phase1-only          stop after Phase 1

set -uo pipefail

# ---------------------------------------------------------------------------
# Colours / helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
BLU='\033[0;34m'; CYN='\033[0;36m'; BLD='\033[1m'; RST='\033[0m'

log()  { echo -e "${BLU}[kcomp]${RST} $*"; }
ok()   { echo -e "${GRN}[  ok ]${RST} $*"; }
warn() { echo -e "${YLW}[ warn]${RST} $*"; }
fail() { echo -e "${RED}[ FAIL]${RST} $*"; }
die()  { fail "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
TOP_N="1 5 10"
ITERS=3
REP=100
TIMEOUT=180
QUICK_FLAG=""
OUTDIR=""
FROM_CACHE=""
SKIP_PHASE1=0
PHASE1_ONLY=0
DISTRIBUTED=""
GPUS_FLAG=""
VERBOSE_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --top-n)       TOP_N="$2";          shift 2 ;;
        --iters)       ITERS="$2";           shift 2 ;;
        --rep)         REP="$2";             shift 2 ;;
        --timeout)     TIMEOUT="$2";         shift 2 ;;
        --quick)       QUICK_FLAG="--quick"; shift ;;
        --outdir)      OUTDIR="$2";          shift 2 ;;
        --from-cache)  FROM_CACHE="$2";      shift 2 ;;
        --distributed) DISTRIBUTED="--distributed"; shift ;;
        --gpus)        GPUS_FLAG="--gpus $2"; shift 2 ;;
        --verbose)     VERBOSE_FLAG="--verbose"; shift ;;
        --skip-phase1) SKIP_PHASE1=1;        shift ;;
        --phase1-only) PHASE1_ONLY=1;        shift ;;
        *) die "Unknown option: $1" ;;
    esac
done

# --from-cache: jump straight to Phase 2 using the given cache dir
if [[ -n "$FROM_CACHE" ]]; then
    if [[ ! -d "$FROM_CACHE" ]]; then
        die "--from-cache directory not found: $FROM_CACHE"
    fi
    SKIP_PHASE1=1
    # Use a fresh outdir alongside the cache, or honour --outdir if set
    if [[ -z "$OUTDIR" ]]; then
        OUTDIR="$(dirname "$FROM_CACHE")/p2_$(date +%Y%m%d_%H%M%S)"
    fi
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${OUTDIR:-kern_cmp_${TIMESTAMP}}"
mkdir -p "$OUTDIR"

if [[ -n "$FROM_CACHE" ]]; then
    SHARED_CACHE="$(realpath "$FROM_CACHE")"
else
    SHARED_CACHE="$(realpath "$OUTDIR")/cache"
fi
JSONL_OUT="$(realpath "$OUTDIR")/phase2_timings.jsonl"
CSV_OUT="$(realpath "$OUTDIR")/kernel_comparison.csv"
TXT_OUT="$(realpath "$OUTDIR")/kernel_comparison_summary.txt"
P1_LOG="$(realpath "$OUTDIR")/phase1_autotune.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_PY="$SCRIPT_DIR/benchmark_pointwise.py"
P2_PY="$SCRIPT_DIR/run_phase2_benchmarks.py"
ANALYZE_PY="$SCRIPT_DIR/analyze_kernel_comparison.py"

for f in "$BENCH_PY" "$P2_PY" "$ANALYZE_PY"; do
    [[ -f "$f" ]] || die "Required script not found: $f"
done

log "Output dir   : $OUTDIR"
log "Shared cache : $SHARED_CACHE"
log "TOP_N sweep  : $TOP_N"
log "Phase-2 reps : $REP"
log "P2 timeout   : ${TIMEOUT}s"

# ---------------------------------------------------------------------------
# run_py helper — runs benchmark_pointwise.py with sys.path fixed for install
# ---------------------------------------------------------------------------
run_py() {
    local label="$1"
    shift
    local logfile="$1"
    shift
    # rest of $@ = env var assignments in KEY=VALUE form

    # Build env override string
    local env_str=""
    for kv in "$@"; do
        env_str+="$kv "
    done

    log "  Running: $label"
    log "  Env   : $env_str"
    log "  Log   : $logfile"

    # Monkey-patch torch.compile to avoid CUDA graph capture issues and to
    # correctly honour _BENCH_COMPILE_MODE.
    local preamble
    preamble=$(cat <<'PYEOF'
import torch as _torch
def _make_patched(_orig):
    def _patched(*a, **kw):
        import os as _os
        kw["mode"] = _os.environ.get("_BENCH_COMPILE_MODE", "max-autotune-no-cudagraphs")
        return _orig(*a, **kw)
    return _patched
_torch.compile = _make_patched(_torch.compile)
del _make_patched
PYEOF
)

    # Run with the specified env vars
    env \
        TORCHINDUCTOR_BENCHMARK_KERNEL=1 \
        TORCHINDUCTOR_CACHE_DIR="$SHARED_CACHE" \
        _BENCH_COMPILE_MODE="max-autotune-no-cudagraphs" \
        $env_str \
        python - "$BENCH_PY" ${QUICK_FLAG:+"$QUICK_FLAG"} --iters "$ITERS" \
        <<WRAPPER 2>&1 | tee "$logfile"
import sys, os

# Remove the repo root from sys.path so 'import torch' picks up the install.
_repo = os.path.dirname(os.path.abspath(sys.argv[1]))
sys.path = [p for p in sys.path if os.path.abspath(p) != _repo]

${preamble}

sys.argv = [sys.argv[1]] + sys.argv[2:]
with open(sys.argv[0]) as _f:
    exec(compile(_f.read(), sys.argv[0], "exec"), {"__name__": "__main__"})
WRAPPER

    local rc=${PIPESTATUS[0]}
    return $rc
}

# ---------------------------------------------------------------------------
# Phase 1 — single autotune run, shared cache
# ---------------------------------------------------------------------------

echo ""
echo -e "${BLD}╔══════════════════════════════════════════════════════════════════╗${RST}"
echo -e "${BLD}║  PHASE 1 — Compile + autotune all kernels (single shared cache) ║${RST}"
echo -e "${BLD}╚══════════════════════════════════════════════════════════════════╝${RST}"
echo ""

if [[ $SKIP_PHASE1 -eq 1 ]]; then
    warn "Skipping Phase 1 (--skip-phase1)"
    if [[ ! -d "$SHARED_CACHE" ]]; then
        die "Shared cache not found at $SHARED_CACHE — cannot skip Phase 1"
    fi
else
    mkdir -p "$SHARED_CACHE"
    log "Phase 1: autotune, no heuristics → $SHARED_CACHE"

    run_py "Phase-1: autotune (no heuristics)" "$P1_LOG" \
        TORCHINDUCTOR_POINTWISE_HEURISTICS=0 \
        TORCHINDUCTOR_MAX_AUTOTUNE=1 \
        TORCHINDUCTOR_HEURISTICS_REAL_BENCH=0

    rc=$?
    if [[ $rc -ne 0 ]]; then
        fail "Phase 1 FAILED (rc=$rc) — check $P1_LOG"
        fail "Continuing to Phase 2 with whatever was compiled"
    else
        ok "Phase 1 complete"
    fi

    # Count kernel files generated
    n_kernels=$(find "$SHARED_CACHE" -name "*.py" \
        -exec grep -l "def get_args():" {} \; 2>/dev/null | wc -l)
    ok "Kernel files with harness: $n_kernels"
fi

if [[ $PHASE1_ONLY -eq 1 ]]; then
    log "Stopping after Phase 1 (--phase1-only)"
    exit 0
fi

# ---------------------------------------------------------------------------
# Phase 2 — benchmark every kernel file × every mode
# ---------------------------------------------------------------------------

echo ""
echo -e "${BLD}╔══════════════════════════════════════════════════════════════════╗${RST}"
echo -e "${BLD}║  PHASE 2 — Standalone kernel benchmarks (all modes)             ║${RST}"
echo -e "${BLD}╚══════════════════════════════════════════════════════════════════╝${RST}"
echo ""

log "Phase 2: running all modes against shared cache"
log "  JSONL output → $JSONL_OUT"

python "$P2_PY" \
    --cache-dir "$SHARED_CACHE" \
    --out       "$JSONL_OUT" \
    --top-n     "$TOP_N" \
    --rep       "$REP" \
    --timeout   "$TIMEOUT" \
    ${DISTRIBUTED:+"$DISTRIBUTED"} \
    ${GPUS_FLAG:+$GPUS_FLAG} \
    ${VERBOSE_FLAG:+"$VERBOSE_FLAG"}

p2_rc=$?
if [[ $p2_rc -ne 0 ]]; then
    fail "Phase 2 FAILED (rc=$p2_rc)"
    exit $p2_rc
fi
ok "Phase 2 complete"

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

echo ""
echo -e "${BLD}╔══════════════════════════════════════════════════════════════════╗${RST}"
echo -e "${BLD}║  ANALYSIS                                                        ║${RST}"
echo -e "${BLD}╚══════════════════════════════════════════════════════════════════╝${RST}"
echo ""

python "$ANALYZE_PY" \
    --jsonl   "$JSONL_OUT" \
    --out-csv "$CSV_OUT" \
    --out-txt "$TXT_OUT"

ok "Analysis complete"
echo ""
echo -e "${GRN}${BLD}Results:${RST}"
echo -e "  CSV     → ${CYN}$CSV_OUT${RST}"
echo -e "  Summary → ${CYN}$TXT_OUT${RST}"
echo -e "  JSONL   → ${CYN}$JSONL_OUT${RST}"
echo -e "  P1 log  → ${CYN}$P1_LOG${RST}"

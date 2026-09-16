#!/usr/bin/env bash
# Every table in this directory, one after another into one log.
#
# Sequentially on purpose: these all measure the same GPU, and two of them at once measure
# each other. Compile and autotune time dominates the wall clock, so budget an hour.
#
#   bash benchmarks/transformer/flydsl/run_all.sh out.log
set -u
cd "$(dirname "$0")"
log="${1:-/tmp/flydsl_bench.log}"
: >"$log"

run() {
    echo "=============== $* ===============" | tee -a "$log"
    # Warnings and autotune chatter go to the log only; the table is what is worth seeing.
    timeout 5400 python "$@" 2>&1 |
        grep -av "occupancy target\|Autotune Choices\|num_choices\|UserWarning\|current_out_size\|profiler_st\|TypedStorage" |
        tee -a "$log"
}

run decode_gap.py --check
run sparse_gaps.py --decode
run sparse_gaps.py --mask-cost
run mod_matrix.py
run mod_matrix.py --bwd
run walk_cost.py --d 128
run shape_ladder.py
echo "log: $log"

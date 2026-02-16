#!/bin/bash
# Compare performance with and without new pointwise heuristics

echo "================================================================================"
echo "  A/B COMPARISON: Original vs New Pointwise Heuristics"
echo "================================================================================"
echo ""

# Clear cache before each run
clear_cache() {
    echo "Clearing Inductor cache..."
    rm -rf /tmp/torchinductor_*
}

# Run with original behavior
echo ">>> Running with ORIGINAL behavior (heuristics OFF)..."
echo ""
clear_cache
TORCHINDUCTOR_POINTWISE_HEURISTICS=0 python /root/benchmark_pointwise.py \
    --csv /tmp/results_original.csv \
    --warmup 5 \
    --iters 30 \
    2>&1 | tee /tmp/benchmark_original.log

echo ""
echo "================================================================================"
echo ""

# Run with new heuristics
echo ">>> Running with NEW heuristics (heuristics ON)..."
echo ""
clear_cache
TORCHINDUCTOR_POINTWISE_HEURISTICS=1 python /root/benchmark_pointwise.py \
    --csv /tmp/results_new.csv \
    --warmup 5 \
    --iters 30 \
    2>&1 | tee /tmp/benchmark_new.log

echo ""
echo "================================================================================"
echo "  COMPARISON COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - /tmp/results_original.csv (heuristics OFF)"
echo "  - /tmp/results_new.csv (heuristics ON)"
echo "  - /tmp/benchmark_original.log"
echo "  - /tmp/benchmark_new.log"
echo ""
echo "To compare geomean speedups:"
echo "  grep 'OVERALL' /tmp/benchmark_original.log"
echo "  grep 'OVERALL' /tmp/benchmark_new.log"
echo ""


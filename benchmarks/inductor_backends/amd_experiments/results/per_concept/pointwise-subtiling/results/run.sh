#!/usr/bin/env bash
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
python "$repo_root/benchmarks/inductor_backends/amd_experiments/benchmark.py" --output "$repo_root/benchmarks/inductor_backends/amd_experiments/results" --rounds 9 --repeats 20

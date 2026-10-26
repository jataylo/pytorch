#!/bin/bash
# Run the scatter benchmark suite across all GPUs; reruns resume from <out.jsonl>.
#   ./run_suite.sh results/suite_quick.jsonl --quick
#   ./run_suite.sh results/suite_full.jsonl
#   ./run_suite.sh out.jsonl --quick --modes preagg --ops index_add_dim0 emb_bwd
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: run_suite.sh <out.jsonl> [suite.py args...]}
shift
python "$here/suite.py" --out "$out" "$@"
python "$here/report.py" "$out"

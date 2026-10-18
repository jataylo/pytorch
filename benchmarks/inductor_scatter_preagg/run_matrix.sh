#!/bin/bash
# Run bench_matrix.py over op x V x dist x D x mode, sharded across GPUs.
# Usage: ./run_matrix.sh <out.txt> [modes...]   (default modes: base ps preagg)
# NUM_GPUS (default 8) workers; worker k runs every NUM_GPUS-th job on GPU k.
set -u
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: run_matrix.sh <out.txt> [modes...]}
shift
modes=${*:-base ps preagg}
ngpu=${NUM_GPUS:-8}
tmp=$(mktemp -d)

jobs=$tmp/jobs.txt
for op in emb_bwd index_add; do for V in 24 128 6048 100000; do for dist in hot50 zipf uniform; do
  for D in 32 128 512; do for m in $modes; do
    echo "$m $op $V $dist $D" >> "$jobs"
  done; done
done; done; done

for ((k = 0; k < ngpu; k++)); do
  (
    n=0
    while read -r line; do
      if [ $((n % ngpu)) -eq $k ]; then
        HIP_VISIBLE_DEVICES=$k CUDA_VISIBLE_DEVICES=$k python "$here/bench_matrix.py" $line 2>&1 \
          | grep -E "^RESULT" || echo "FAILED $line"
      fi
      n=$((n + 1))
    done < "$jobs"
  ) > "$tmp/out_$k.txt" &
done
wait
cat "$tmp"/out_*.txt > "$out"
rm -rf "$tmp"
echo "$(grep -c ^RESULT "$out") results, $(grep -c ^FAILED "$out") failed -> $out"

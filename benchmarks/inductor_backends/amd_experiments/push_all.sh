#!/usr/bin/env bash
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
git -C "$repo_root" push --atomic -u https://github.com/jataylo/pytorch.git \
  experiment/amd-lowering-bench-base \
  experiment/amd-lowering-wide-reduction-splits \
  experiment/amd-lowering-persistent-selection \
  experiment/amd-lowering-reduction-indexing \
  experiment/amd-lowering-transpose-tiling \
  experiment/amd-lowering-kernel-selection \
  experiment/amd-lowering-pointwise-subtiling \
  experiment/amd-lowering-row-packing \
  experiment/amd-lowering-hierarchical-scan \
  experiment/amd-lowering-two-pass-scan \
  experiment/amd-lowering-results

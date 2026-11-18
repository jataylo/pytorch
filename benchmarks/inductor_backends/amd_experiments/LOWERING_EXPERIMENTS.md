# Independent AMD Triton lowering experiments

Nine sibling feature branches isolate the lowering concepts from the combined native Triton branch. All are based on upstream `dddd90223f0` plus the benchmark-only support branch `experiment/amd-lowering-bench-base` (`a6f1fe74111`). No feature branch depends on another feature branch, and every feature is default-off. This is an experimental decomposition for human review, not a claim that these branches are all ready to submit upstream.

The benchmark-only support branch and this results catalog are additional publishing branches. Each feature branch includes its compiler changes, focused unit tests, operator workload cases, generated sources, raw timing samples, CSV comparisons, validation logs and `RESULTS.md`. This catalog duplicates those result artifacts so the independent results are also available in one checkout.

## Priority and independent uplift

Priority combines measured benefit, size of regressions, implementation scope and numerical semantics. The primary metric, chosen for every branch, is warm captured GPU kernel-sequence time. Speedup is `(stock_us / feature_us - 1) * 100`; it differs from percent time saved. Geomeans weight each unit equally. Best/worst include unchanged controls, and gains below 2% should be treated cautiously.

| Priority | Branch suffix | Valid / exercised cases | All-case geomean | Exercised-only geomean | Best | Worst | Wins / regressions >2% |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `wide-reduction-splits` | 8 / 4 | +21.62% | +47.03% | +57.08% | -0.93% | 5 / 0 |
| 2 | `persistent-selection` | 8 / 8 | +9.31% | +9.31% | +32.44% | -1.27% | 4 / 0 |
| 3 | `reduction-indexing` | 8 / 8 | +7.74% | +7.74% | +23.85% | -14.35% | 5 / 1 |
| 4 | `transpose-tiling` | 16 / 10 | +0.64% | +1.11% | +3.25% | -1.63% | 5 / 0 |
| 5 | `kernel-selection` | 8 / 8 | +0.28% | +0.28% | +2.94% | -2.18% | 2 / 1 |
| 6 | `pointwise-subtiling` | 12 / 12 | -0.07% | -0.07% | +13.70% | -12.64% | 2 / 2 |
| 7 | `row-packing` | 12 / 12 | +0.18% | +0.18% | +3.13% | -3.08% | 2 / 2 |
| 8 | `hierarchical-scan` | 8 / 6 | -0.08% | -0.18% | +2.45% | -2.60% | 1 / 1 |
| 9 | `two-pass-scan` | 8 / 8 | +91.13% | +91.13% | +178.08% | +27.53% | 8 / 0 |

The full branch name is `experiment/amd-lowering-<suffix>`. There are 88 valid stock/feature operator pairs and no failing benchmark cases. A source marker or changed kernel body establishes whether the concept was exercised; unmatched cases stay visible as controls. A marker proves that candidates were exposed, not that the winning configuration used the transformed algorithm. The six unmatched transpose cases, four unchanged wide-reduction cases and two unmatched hierarchical-scan cases are not credited as lowering improvements.

These independent numbers deliberately exclude the new kernel-selection logic from source-transform branches. They therefore do not reproduce every gain from the earlier combined branch, where transformations and selection interacted. Do not add branch speedups together. E2E model results were not rerun for this decomposition; these measurements do not establish model-level gains.

## Measurement and validation

All experiments ran sequentially on one AMD Instinct MI355X (gfx950), physical GPU 3, with nine paired rotating rounds and twenty event intervals per round. The retained manifest records PyTorch/Triton versions, source fingerprints, cache policy, GPU identity and protocol details. Each arm compiles a static full graph, then runs its frozen backend callable. Each graph is validated on three seeds before timing; cumulative sums are checked against prefixes accumulated in FP64. Positive scan performance inputs avoid counting cancellation failures in the stock tree as speedups. Unit tests separately cover signed, nonfinite, overflow, stride, tail and exclusion cases where relevant.

Four protocols are recorded: warm/flushed-cache captured GPU sequences, and warm/flushed-cache host-issued graph replay. The primary capture mode excludes host launch time. Warm/cold cache preparation and any input restoration occur outside event intervals. Each case retains all timing samples and paired bootstrap intervals. These intervals cover timing-round variation only, not independent recompilation or configuration-selection variance. Small effects need repeated compile/timing trials.

Every compiler source fingerprint and generated-source hash was checked against the measured snapshots. Source rewrites are guarded by device support and deterministic/batch-invariant exclusions; numerical benchmark validation covers both disabled and enabled arms. Focused CPU and GPU unit suites and changed-file lint pass. PYREFLY and CLANGTIDY_EXECUTORCH_COMPATIBILITY were excluded from lint because their dependencies/build are unavailable in this environment. No C++ or PyTorch rebuild was needed for this Python-only split. Compile elapsed times use reused compiler caches and are diagnostic, not cold-compile measurements. Generated kernel counts include candidate alternatives, not just executed launches.

`results/all_comparisons.csv` contains all 352 per-case/protocol comparisons, source hashes, uncertainty, compile times and kernel counts. `results/all_summaries.csv` contains per-concept aggregates for all protocols. `results/per_concept/<suffix>/` preserves each complete measured result set and validation receipts. Each feature's `RESULTS.md` lists its primary per-case results.

## Review and eventual upstream extraction

The support branch contains only the benchmark harness. Each feature branch contains one lowering concept in the next commit, with experiments and receipts included as requested. For an upstream PR, retain the `torch/_inductor/` changes and focused `test/inductor/` files from that feature; remove the experimental result artifacts and decide where to place permanent benchmark coverage. Shared hardware-sizing and AST utility definitions are repeated in independent branches so none requires another optimization. Deduplicate those helpers when stacking or merging multiple concepts; do not blindly cherry-pick all sibling commits into one tree.

## 1) wide-reduction-splits

Create or refine parallel partial reductions for underfilled wide rows with coupled broadcast epilogues; retain the existing reduction arithmetic.

Highest priority: clear gain in the four exercised cases; four unchanged controls. Broaden coupled-epilogue and cross-device coverage.

Full branch: `experiment/amd-lowering-wide-reduction-splits`.

Local checkout: `/dockerx/pytorch-flex/agent_space/amd_lowering_experiments/wide-reduction-splits`.

Focused results: [RESULTS.md](results/per_concept/wide-reduction-splits/RESULTS.md).

## 2) persistent-selection

Use live-value and hardware resource estimates to admit persistent kernels, then compare with looped MultiKernel alternatives.

High priority: clear selected-case benefit. Review MultiKernel compile overhead and broader memory-pressure behavior.

Full branch: `experiment/amd-lowering-persistent-selection`.

Local checkout: `/dockerx/pytorch-flex/agent_space/amd_lowering_experiments/persistent-selection`.

Focused results: [RESULTS.md](results/per_concept/persistent-selection/RESULTS.md).

## 3) reduction-indexing

Remove an integer modulo only when the load mask proves that its nonnegative index is already below the modulus.

Useful narrow proof, but one substantial regression. Investigate launch selection before upstream default enablement.

Full branch: `experiment/amd-lowering-reduction-indexing`.

Local checkout: `/dockerx/pytorch-flex/agent_space/amd_lowering_experiments/reduction-indexing`.

Focused results: [RESULTS.md](results/per_concept/reduction-indexing/RESULTS.md).

## 4) transpose-tiling

Compare two-dimensional transpose tiles and read orders with the original flat copy, proving strides and masks.

Experimental: small gain, largely within the 2% band; six cases do not match the transform. Needs stronger independent evidence.

Full branch: `experiment/amd-lowering-transpose-tiling`.

Local checkout: `/dockerx/pytorch-flex/agent_space/amd_lowering_experiments/transpose-tiling`.

Focused results: [RESULTS.md](results/per_concept/transpose-tiling/RESULTS.md).

## 5) kernel-selection

Expand generic reduction configurations and use bounded, mutation-safe paired tuning and confirmation; leave emitted kernel math unchanged.

Reusable infrastructure, but this stock-math-only sample is nearly flat and adds tuning cost. Evaluate broader workloads before upstreaming.

Full branch: `experiment/amd-lowering-kernel-selection`.

Local checkout: `/dockerx/pytorch-flex/agent_space/amd_lowering_experiments/kernel-selection`.

Focused results: [RESULTS.md](results/per_concept/kernel-selection/RESULTS.md).

## 6) pointwise-subtiling

Process several smaller pointwise tiles per program, reusing scalar loads and arithmetic; retain flat candidates.

Hold: near-flat aggregate masks both a 13.70% win and a 12.64% regression. Further geometry selection and separate recompilation trials are needed.

Full branch: `experiment/amd-lowering-pointwise-subtiling`.

Local checkout: `/dockerx/pytorch-flex/agent_space/amd_lowering_experiments/pointwise-subtiling`.

Focused results: [RESULTS.md](results/per_concept/pointwise-subtiling/RESULTS.md).

## 7) row-packing

Prove row-tail masks before admitting larger row groups and device-bounded warp configurations to persistent reductions.

Hold: independently near-flat. Any synergy with the new selector requires a separate factorial experiment.

Full branch: `experiment/amd-lowering-row-packing`.

Local checkout: `/dockerx/pytorch-flex/agent_space/amd_lowering_experiments/row-packing`.

Focused results: [RESULTS.md](results/per_concept/row-packing/RESULTS.md).

## 8) hierarchical-scan

Compare additive FP32 chunk scans plus preceding-chunk offsets with the original associative scan, keeping launch geometry fixed.

Hold: no convincing isolated speedup on these units, despite retaining original-tree candidates. Do not credit combined-branch scan gains to this pass.

Full branch: `experiment/amd-lowering-hierarchical-scan`.

Local checkout: `/dockerx/pytorch-flex/agent_space/amd_lowering_experiments/hierarchical-scan`.

Focused results: [RESULTS.md](results/per_concept/hierarchical-scan/RESULTS.md).

## 9) two-pass-scan

Compute chunk totals and prefixes in FP64 across two kernels, then cast to FP32. This changes accumulation precision and is experimental.

Last for upstream readiness despite strong performance: explicit FP64 accumulation changes overflow, cancellation and rounding semantics.

Full branch: `experiment/amd-lowering-two-pass-scan`.

Local checkout: `/dockerx/pytorch-flex/agent_space/amd_lowering_experiments/two-pass-scan`.

Focused results: [RESULTS.md](results/per_concept/two-pass-scan/RESULTS.md).

## Committed branch heads

| Feature | Commit | GPU-suite tests |
|---|---|---:|
| `pointwise-subtiling` | `c5f46fe6fda8` | 2 |
| `row-packing` | `aca7003a37d9` | 16 |
| `transpose-tiling` | `7b72ef33ae38` | 89 |
| `reduction-indexing` | `d76c1f96ece9` | 5 |
| `persistent-selection` | `ef9e80393969` | 7 |
| `wide-reduction-splits` | `aa3b9d7c1ade` | 58 |
| `hierarchical-scan` | `ba5862b444e4` | 47 |
| `two-pass-scan` | `16caa13b2259` | 26 |
| `kernel-selection` | `90da7ded07aa` | 10 |

The support head is `a6f1fe74111`. `branches.json` records complete hashes, worktrees and compiler file lists. All feature worktrees are committed independently.

## Reproduction and publication

In a matching built PyTorch checkout/environment, run `bash benchmarks/inductor_backends/amd_experiments/results/run.sh` from the feature branch. Local overlay commands used on this machine are retained in `local_run.sh` and `validation/benchmark_command.json`; the overlay loads the complete feature checkout's Inductor package, not a partial mix of versions. The benchmark GEMM/conv backend choices are ATEN. The stopped E2E suite remains stopped.

Regenerate CSV summaries from all sibling worktrees with:

```sh
python benchmarks/inductor_backends/amd_experiments/report.py --root /dockerx/pytorch-flex/agent_space/amd_lowering_experiments --output benchmarks/inductor_backends/amd_experiments/results
```

`push_all.sh` pushes the support branch, nine feature branches and results catalog to `https://github.com/jataylo/pytorch.git` without force. It has been syntax-checked but not executed. No branch was pushed and no issue or PR was opened. Review code and results before submitting upstream.

## Agent details

Prepared with an AI assistant for human review. Upstream issue/PR text must follow the repository AI policy and include human commentary.

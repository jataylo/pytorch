# Inductor benchmark evidence catalog

This directory packages the benchmark scripts, measurements, charts, technical
reports and raw evidence produced during the FlyDSL/Triton work. Accepted final
results and historical, failed or paused runs are kept separately. This catalog
was prepared with an AI assistant at the user's request.

The bundle contains **76,188 raw files**
(32.93 GiB before compression), plus
**1,215 directly accessible reports**. Raw XZ archive parts total
1101.28 MiB; report copies total 179.78 MiB.
Each raw archive part is at most 48 MiB, below GitHub's per-file size limit.
Large report CSVs are provided as `.csv.gz`; ordinary CSVs, JSON summaries,
Markdown, SVG charts and HTML reports remain directly accessible.

## Accepted native Triton results

The source changes and PR plan are described in
[the branch change description](../TRITON_CHANGE_DESCRIPTION_AND_PR_PLAN.md).
The final V5 operation run passed all **142 cases / 426 workers / 1,704 rows**;
358 focused tests and 8 additional wide-reduction checks passed.

| Evidence | Branch-local links |
| --- | --- |
| Operation summary and source audit | [summary](payload/reports/native_triton_v2/results_final_v5/summary.json), [audit](payload/reports/native_triton_v2/results_final_v5/VALIDATION.json) |
| Per-operation and per-case measurements | [per operation](payload/reports/native_triton_v2/results_final_v5/per_op.csv), [per case](payload/reports/native_triton_v2/results_final_v5/per_case.csv), [raw rows](payload/reports/native_triton_v2/results_final_v5/raw.csv) |
| Captured GPU charts | [runtime](payload/reports/native_triton_v2/results_final_v5/runtime_chart.svg), [per operation](payload/reports/native_triton_v2/results_final_v5/per_op_chart.svg) |
| Inline replay charts | [runtime](payload/reports/native_triton_v2/results_final_v5/graph_replay_runtime_chart.svg), [per operation](payload/reports/native_triton_v2/results_final_v5/graph_replay_per_op_chart.svg) |
| Resource and setup evidence | [kernels](payload/reports/native_triton_v2/results_final_v5/per_kernel.csv), [candidates](payload/reports/native_triton_v2/results_final_v5/per_candidate.csv), [setup costs](payload/reports/native_triton_v2/results_final_v5/compile_tune_costs.csv), [resource report](payload/reports/native_triton_v2/results_final_v5/resource_evidence.json) |
| Full training-step comparisons | [per model](payload/reports/native_triton_v2/models_v5/per_model.csv), [summary and accuracy limits](payload/reports/native_triton_v2/models_v5/summary.json) |
| Validation | [focused tests](payload/reports/native_triton_v2/validation_final_v5/VALIDATION_COMPLETE.json) |
| Compiler-cache audit | [audit](payload/reports/native_triton_v2/results_final_v5/compiler_cache_audit.json) |

Captured operation timings measured +12.43% geometric-mean speedup versus
standard Inductor and -0.54% versus the preceding native policy. The two full
training-step workloads measured +0.65% and +0.89% versus standard. These combine
multiple changes; they are not individual-rewrite ablations. Model runs use a
synthetic objective and sampled numerical checks with documented accuracy limits.
Compiler artifacts were preseeded; setup timings are not cold-compilation comparisons.

## Excel workbooks

- [Kernel timing aggregations](payload/reports/kernel_routes_v9/aggregation/kernel_timing_aggregations.xlsx).
- [Three-backend comparison](payload/reports/three_backend_report_v7/three_backend_comparison.xlsx).
- [Later comparison workbook](payload/reports/three_backend_preview_v8/three_backend_preview.xlsx).

## Earlier model, backend and hill-climb work

All earlier measurements remain historical evidence with their original status.
In particular, a paused run or a failed numerical/model result must not be
interpreted as a successful final comparison.

- [Four-mode model comparison](payload/reports/training_results_support_v4/common_four_mode_comparison.csv).
- [Three-backend reports](payload/reports/three_backend_report_v7) and
  [later report/preview tables](payload/reports/three_backend_preview_v8).
- [Mixed single-GPU results](payload/reports/training_results_mixed_single_v5)
  and [multi-GPU recovery results](payload/reports/training_results_mixed_multi_v6).
- [Kernel route/timing analysis](payload/reports/kernel_routes_v9) and
  [operation codegen comparison](payload/reports/triton_flydsl_codegen_comparison_v10).
- [Earlier hill climbs](payload/reports/inductor_hillclimb_v16),
  [native initial experiment](payload/reports/native_triton_experiment),
  and [native follow-up history](payload/reports/native_triton_v2).

The table below includes every packaged experiment group. Scripts, complete raw
worker records, generated kernel sources, frozen compiler snapshots, validation
logs and raw traces are in each group's archive, even when there is no separate
report directory.

## Verification and extraction

From the PyTorch checkout root, verify every compressed part and browsable report:

```bash
python benchmarks/inductor_backends/benchmark_artifacts/verify_artifacts.py
```

Verify every uncompressed raw file against its recorded SHA-256, without writing
out the 32.93 GiB payload:

```bash
python benchmarks/inductor_backends/benchmark_artifacts/verify_artifacts.py --deep
```

Extract a selected experiment to a new directory. Extraction verifies content,
refuses path traversal/symlinks and never overwrites existing files:

```bash
python benchmarks/inductor_backends/benchmark_artifacts/verify_artifacts.py \
  --group native_triton_v2 --extract /tmp/inductor-benchmark-evidence
```

`--group` can be repeated; omit it to extract all groups. Archive parts are ordered
fragments of one XZ-compressed tar stream per group. The verifier joins them;
individual numbered fragments are not independent archives.

Historical scripts and manifests retain their original absolute runtime paths
and hashes. A raw path originally below
`/dockerx/pytorch-flex/agent_space/simd_bench/` becomes the same relative path
under the extraction directory. Dependencies, built PyTorch and the original
machine's virtual environments are external prerequisites; the archive is
measurement/source evidence rather than a relocatable software environment.

The current branch's reusable two-policy operation runner accepts the packaged
142-case list from a built checkout matching this branch:

```bash
python benchmarks/inductor_backends/amd_native_triton_hillclimb.py \
  --cases benchmarks/inductor_backends/benchmark_artifacts/payload/reports/native_triton_v2/results_final_v5/cases.json \
  --output /tmp/native-triton-reproduction --rounds 5 --repeats 5
```

Use an empty output directory. The archived isolated three-policy workers retain
the original frozen source overlays and their provenance; running those historical
harnesses requires adapting their environment paths explicitly.

## Contents and integrity

[payload/manifest.json](payload/manifest.json) records archive parts, sizes,
checksums, report copies and exclusions.
[payload/raw_file_manifest.jsonl.gz](payload/raw_file_manifest.jsonl.gz) records
every raw relative path, original size, SHA-256 and archive group.
Original evidence bytes are preserved. The generated payload is exempt from
whitespace/spelling transformations and marked as generated binary data in Git
attributes; active code, the verifier and this catalog remain linted normally.

Excluded material: duplicate full repository checkouts, virtual environments,
installed/downloaded dependencies, compiler/build caches, model weights and tensor
checkpoints, symlinks, and internal execution journals. Benchmark text/code,
reports, raw traces, saved kernel IR/assembly, Excel workbooks and frozen source
snapshots are included. No model
checkpoints are bundled. The manifest records the directory and file-type policy.

| Experiment | Raw files | Original MiB | Archive MiB | Reports | Ordered archive parts |
| --- | ---: | ---: | ---: | --- | --- |
| baseline_flydsl | 2 | 0.31 | 0.08 | - | [1](payload/raw/baseline_flydsl.part000.tar.xz) |
| codegen_bench_v13 | 786 | 10.70 | 0.46 | [reports](payload/reports/codegen_bench_v13) | [1](payload/raw/codegen_bench_v13.part000.tar.xz) |
| codegen_startup_v13 | 1 | 0.00 | 0.00 | - | [1](payload/raw/codegen_startup_v13.part000.tar.xz) |
| fallback_micro_v3 | 34 | 0.14 | 0.01 | - | [1](payload/raw/fallback_micro_v3.part000.tar.xz) |
| fallback_micro_v3_axis_search | 36 | 0.25 | 0.02 | [reports](payload/reports/fallback_micro_v3_axis_search) | [1](payload/raw/fallback_micro_v3_axis_search.part000.tar.xz) |
| fallback_micro_v3_tunefix | 33 | 0.24 | 0.01 | - | [1](payload/raw/fallback_micro_v3_tunefix.part000.tar.xz) |
| fallback_micro_validation_v3 | 2 | 0.01 | 0.00 | - | [1](payload/raw/fallback_micro_validation_v3.part000.tar.xz) |
| fallback_review_v2 | 127 | 9.81 | 0.19 | [reports](payload/reports/fallback_review_v2) | [1](payload/raw/fallback_review_v2.part000.tar.xz) |
| fallback_validation_v2 | 2 | 0.01 | 0.00 | - | [1](payload/raw/fallback_validation_v2.part000.tar.xz) |
| fallback_validation_v2_round2 | 6 | 1.94 | 0.03 | - | [1](payload/raw/fallback_validation_v2_round2.part000.tar.xz) |
| fallback_validation_v2_round3 | 2 | 0.00 | 0.00 | - | [1](payload/raw/fallback_validation_v2_round3.part000.tar.xz) |
| fallback_validation_v3_axis | 8 | 1.94 | 0.03 | - | [1](payload/raw/fallback_validation_v3_axis.part000.tar.xz) |
| fallback_validation_v3_combined | 8 | 1.94 | 0.03 | - | [1](payload/raw/fallback_validation_v3_combined.part000.tar.xz) |
| fallback_validation_v3_ocml | 10 | 0.04 | 0.00 | - | [1](payload/raw/fallback_validation_v3_ocml.part000.tar.xz) |
| fallback_validation_v3_tail | 12 | 0.04 | 0.00 | - | [1](payload/raw/fallback_validation_v3_tail.part000.tar.xz) |
| fallback_validation_v4_dino | 14 | 0.09 | 0.01 | - | [1](payload/raw/fallback_validation_v4_dino.part000.tar.xz) |
| fallback_validation_v4_dino_1793979303192090950 | 15 | 0.06 | 0.01 | - | [1](payload/raw/fallback_validation_v4_dino_1793979303192090950.part000.tar.xz) |
| gluon_commit_validation | 11 | 0.01 | 0.00 | - | [1](payload/raw/gluon_commit_validation.part000.tar.xz) |
| inductor_hillclimb_v14 | 2,025 | 33.21 | 1.53 | [reports](payload/reports/inductor_hillclimb_v14) | [1](payload/raw/inductor_hillclimb_v14.part000.tar.xz) |
| inductor_hillclimb_v15 | 2,067 | 33.64 | 2.60 | [reports](payload/reports/inductor_hillclimb_v15) | [1](payload/raw/inductor_hillclimb_v15.part000.tar.xz) |
| inductor_hillclimb_v16 | 4,375 | 139.63 | 12.35 | [reports](payload/reports/inductor_hillclimb_v16) | [1](payload/raw/inductor_hillclimb_v16.part000.tar.xz) |
| kernel_inventory_audit_v3 | 184 | 890.65 | 38.39 | [reports](payload/reports/kernel_inventory_audit_v3) | [1](payload/raw/kernel_inventory_audit_v3.part000.tar.xz) |
| kernel_inventory_audit_v3_pre_review | 182 | 848.74 | 33.69 | [reports](payload/reports/kernel_inventory_audit_v3_pre_review) | [1](payload/raw/kernel_inventory_audit_v3_pre_review.part000.tar.xz) |
| kernel_ir_and_workbooks | 1,159 | 28.73 | 2.11 | - | [1](payload/raw/kernel_ir_and_workbooks.part000.tar.xz) |
| kernel_routes_v9 | 23 | 99.59 | 5.21 | [reports](payload/reports/kernel_routes_v9) | [1](payload/raw/kernel_routes_v9.part000.tar.xz) |
| mixed_gpu_fault_review_v6 | 17 | 0.54 | 0.05 | [reports](payload/reports/mixed_gpu_fault_review_v6) | [1](payload/raw/mixed_gpu_fault_review_v6.part000.tar.xz) |
| native_triton_experiment | 228 | 5.33 | 0.30 | [reports](payload/reports/native_triton_experiment) | [1](payload/raw/native_triton_experiment.part000.tar.xz) |
| native_triton_v2 | 7,951 | 320.75 | 24.26 | [reports](payload/reports/native_triton_v2) | [1](payload/raw/native_triton_v2.part000.tar.xz) |
| native_triton_v4 | 9 | 0.17 | 0.02 | - | [1](payload/raw/native_triton_v4.part000.tar.xz) |
| native_triton_v5 | 6 | 0.08 | 0.01 | - | [1](payload/raw/native_triton_v5.part000.tar.xz) |
| outlier_recheck_1 | 24 | 1.56 | 0.06 | - | [1](payload/raw/outlier_recheck_1.part000.tar.xz) |
| outlier_recheck_2 | 24 | 1.56 | 0.06 | - | [1](payload/raw/outlier_recheck_2.part000.tar.xz) |
| primary_results | 473 | 2.67 | 0.17 | [reports](payload/reports/primary_results) | [1](payload/raw/primary_results.part000.tar.xz) |
| root | 366 | 34.13 | 0.85 | [reports](payload/reports) | [1](payload/raw/root.part000.tar.xz) |
| smoke | 8 | 0.67 | 0.03 | - | [1](payload/raw/smoke.part000.tar.xz) |
| smoke_hf | 8 | 0.04 | 0.00 | - | [1](payload/raw/smoke_hf.part000.tar.xz) |
| smoke_hf2 | 8 | 0.18 | 0.01 | - | [1](payload/raw/smoke_hf2.part000.tar.xz) |
| support_startup_v3 | 1 | 0.00 | 0.00 | - | [1](payload/raw/support_startup_v3.part000.tar.xz) |
| support_startup_v4 | 1 | 0.00 | 0.00 | - | [1](payload/raw/support_startup_v4.part000.tar.xz) |
| three_backend_preview_v8 | 16 | 1.36 | 0.12 | [reports](payload/reports/three_backend_preview_v8) | [1](payload/raw/three_backend_preview_v8.part000.tar.xz) |
| three_backend_report_v7 | 16 | 1.38 | 0.12 | [reports](payload/reports/three_backend_report_v7) | [1](payload/raw/three_backend_report_v7.part000.tar.xz) |
| training_results | 1,092 | 52.24 | 2.06 | [reports](payload/reports/training_results) | [1](payload/raw/training_results.part000.tar.xz) |
| training_results_mixed_multi_v6 | 9,340 | 1294.01 | 23.46 | [reports](payload/reports/training_results_mixed_multi_v6) | [1](payload/raw/training_results_mixed_multi_v6.part000.tar.xz) |
| training_results_mixed_single_v5 | 31,686 | 10046.12 | 296.05 | [reports](payload/reports/training_results_mixed_single_v5) | [1](payload/raw/training_results_mixed_single_v5.part000.tar.xz) [2](payload/raw/training_results_mixed_single_v5.part001.tar.xz) [3](payload/raw/training_results_mixed_single_v5.part002.tar.xz) [4](payload/raw/training_results_mixed_single_v5.part003.tar.xz) [5](payload/raw/training_results_mixed_single_v5.part004.tar.xz) [6](payload/raw/training_results_mixed_single_v5.part005.tar.xz) [7](payload/raw/training_results_mixed_single_v5.part006.tar.xz) |
| training_results_purefly_v7 | 1,638 | 2582.89 | 78.64 | [reports](payload/reports/training_results_purefly_v7) | [1](payload/raw/training_results_purefly_v7.part000.tar.xz) [2](payload/raw/training_results_purefly_v7.part001.tar.xz) |
| training_results_support_v3 | 2,671 | 4902.73 | 163.45 | [reports](payload/reports/training_results_support_v3) | [1](payload/raw/training_results_support_v3.part000.tar.xz) [2](payload/raw/training_results_support_v3.part001.tar.xz) [3](payload/raw/training_results_support_v3.part002.tar.xz) [4](payload/raw/training_results_support_v3.part003.tar.xz) |
| training_results_support_v4 | 2,672 | 5145.01 | 168.62 | [reports](payload/reports/training_results_support_v4) | [1](payload/raw/training_results_support_v4.part000.tar.xz) [2](payload/raw/training_results_support_v4.part001.tar.xz) [3](payload/raw/training_results_support_v4.part002.tar.xz) [4](payload/raw/training_results_support_v4.part003.tar.xz) |
| training_results_v2 | 3,057 | 5239.28 | 179.31 | [reports](payload/reports/training_results_v2) | [1](payload/raw/training_results_v2.part000.tar.xz) [2](payload/raw/training_results_v2.part001.tar.xz) [3](payload/raw/training_results_v2.part002.tar.xz) [4](payload/raw/training_results_v2.part003.tar.xz) |
| training_results_v2_initial_metadata_error | 126 | 4.13 | 0.18 | - | [1](payload/raw/training_results_v2_initial_metadata_error.part000.tar.xz) |
| training_results_v2_periodic_dump | 95 | 17.67 | 0.98 | - | [1](payload/raw/training_results_v2_periodic_dump.part000.tar.xz) |
| training_results_v2_single | 1,716 | 1163.81 | 37.50 | - | [1](payload/raw/training_results_v2_single.part000.tar.xz) |
| training_results_v2_steady | 1,141 | 696.51 | 23.77 | [reports](payload/reports/training_results_v2_steady) | [1](payload/raw/training_results_v2_steady.part000.tar.xz) |
| triton_flydsl_codegen_comparison_v10 | 12 | 0.26 | 0.01 | - | [1](payload/raw/triton_flydsl_codegen_comparison_v10.part000.tar.xz) |
| triton_hillclimb_v11 | 92 | 1.88 | 0.09 | [reports](payload/reports/triton_hillclimb_v11) | [1](payload/raw/triton_hillclimb_v11.part000.tar.xz) |
| triton_hillclimb_v12 | 412 | 5.08 | 0.19 | [reports](payload/reports/triton_hillclimb_v12) | [1](payload/raw/triton_hillclimb_v12.part000.tar.xz) |
| triton_native_api_tail_review | 4 | 0.40 | 0.08 | - | [1](payload/raw/triton_native_api_tail_review.part000.tar.xz) |
| v2_frozen_sources | 6 | 0.46 | 0.12 | - | [1](payload/raw/v2_frozen_sources.part000.tar.xz) |
| validation_fsdp_v2 | 16 | 0.05 | 0.00 | - | [1](payload/raw/validation_fsdp_v2.part000.tar.xz) |
| validation_fsdp_v2_final | 24 | 0.80 | 0.03 | - | [1](payload/raw/validation_fsdp_v2_final.part000.tar.xz) |
| validation_fsdp_v2b | 28 | 0.81 | 0.04 | - | [1](payload/raw/validation_fsdp_v2b.part000.tar.xz) |
| validation_v2 | 80 | 98.74 | 3.81 | [reports](payload/reports/validation_v2) | [1](payload/raw/validation_v2.part000.tar.xz) |

# FlyDSL SIMD tuning and full-model training comparison

The modified checkout is `pytorch/`, based on `937d7fd7cdb37e3136c90e86ca71b57d8e5193be`.
`source_fingerprint_v2.json` identifies exact sources and harness files.
`flydsl_autotune_v2.patch` contains intentional source changes, excluding build-generated ROCm hipification.

Enable supported FlyDSL SIMD lowering with `torch._inductor.config.flydsl.simd = True`.
Unsupported kernels retain the existing Triton fallback. To additionally compare the two backends for
compatible supported kernels, set `flydsl.simd_autotune_backend = True`; its default is false.
The generated graph wrapper contains both launchers and caches the measured winner.
Dynamic, atomic, workspace, combo and incompatible launch ABIs retain their existing choice.

FlyDSL launch tuning is controlled by `TORCHINDUCTOR_FLYDSL_SIMD_TUNE=autotune=1` or `autotune=0`.
The expanded search varies threads, vector widths, unrolling, tiled geometry, column layout,
row groups and persistent versus looped reduction algorithms. The configurable limits are
`flydsl.simd_autotune_min_numel` and `flydsl.simd_autotune_max_configs`.
Each candidate has separate shared-memory helper names. Trials preserve aliased output storage,
restore mutations, compare outputs against the reference candidate, and use repeated measurements.
Backend comparisons use Triton as the numerical reference; FlyDSL geometry comparisons use its heuristic.

Set `TORCHINDUCTOR_FLYDSL_SIMD_DIAGNOSTICS=/path/decisions.jsonl` to record lowering and tuning decisions.
The same data is available through structured tracing as `flydsl_kernel_decision` artifacts.
Refused FlyDSL schedules and named native replacements carry graph ID, forward/backward status,
operation origins and buffer metadata. Late compilation/execution failures are separate events.

The benchmark matrix is in `benchmark_matrix_v2.json`; its manifest contains 131 models and 524 variants.
All variants force GEMM and convolution to ATen. SIMD tuning varies independently of template lowering.
The `autotune0` label disables the explicit expanded search and pointwise autotuning. Native Triton
reduction heuristics retain their limited baseline configuration selection; `autotune1` enables the
expanded pointwise/reduction search. Exact flags are recorded in every result.
Training uses full architectures with random initialization, AMP BF16, FP32 parameters and SGD updates.
No model checkpoints are requested or saved. Timing includes forward, backward and optimizer work,
with compilation/warmup excluded. Diffusion list entries follow the source loader's denoiser scope.
Models exceeding aggregate memory receive explicit capacity outcomes rather than reduced architectures.
FSDP retries preserve the original single-GPU results and use identical GPU groups for all four variants.

Run commands from `/dockerx/pytorch-flex`:

```bash
bash agent_space/simd_bench/run_env.sh agent_space/simd_bench/freeze_sources_v2.py
bash agent_space/simd_bench/run_env.sh agent_space/simd_bench/run_suite_v2.py --gpus 0,1,2,3,4,5,6,7
bash agent_space/simd_bench/run_env.sh agent_space/simd_bench/run_fsdp_v2.py
bash agent_space/simd_bench/run_env.sh agent_space/simd_bench/analyze_v2.py
```

`training_results_v2/report.html` provides a filterable comparison and links to CSV/JSONL artifacts.
Validated comparisons exclude numerical warnings and training-only results without an eager reference.
All measured comparisons remain available separately. Ratios above one favor FlyDSL.
Kernel inventory counts describe generated schedules, not dynamic GPU launch counts.

Validation: all 17 focused autotune tests pass. Of 128 existing SIMD tests, 120 pass and eight error
identically with both the original branch sources and modified sources on this gfx950/ROCm10 system.
The existing errors cover gfx942-only FNUZ operations and device-library/scan compiler failures.
See `test_autotune_v2_final.log`, `test_flydsl_full_v2_math.log`, and `test_flydsl_genuine_original.log`.

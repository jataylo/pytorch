# AMD Inductor lowering and tuning experiments

Both new options are experimental and default to `False`:

```python
from torch._inductor import config

with config.patch({"triton.amd_kernel_optimizations": True}):
    compiled = torch.compile(model)
    compiled(*inputs)
```

`triton.amd_kernel_optimizations` enables gfx950 layout/codegen improvements
and execution-only autotuning. `triton.amd_approximate_math` additionally enables
the guarded shared-exponential GELU-backward approximation and approximate
softmax exponential/reciprocal. Approximate math requires the optimization
option as well. It is a separate numerical policy, not an implicit default.

The implementation includes:

- Explicit Triton/Gluon layouts for short row reductions and saved statistics;
  multi-wave row alternatives for larger reductions.
- Input-linear and output-linear traversal alternatives for static transpose/copy.
- Vector-width, workgroup, unroll and occupancy choices for cast/add and
  broadcast arithmetic.
- Guarded removal of redundant split-reduction modulo operations.
- Alias-preserving scratch storage, restored before GPU timing; producer-cache
  modeling and input-mutation warming; rotating candidate measurements.
- A bounded hill climb over compatible launch/layout parameters. Candidate
  outputs are checked, including poisoned outputs to detect partial writes.
- Conservative scratch-memory fallback and exclusions for unsupported
  cooperative/TMA/atomic/combo cases. Deterministic/batch-invariant modes retain
  the existing path.

Set `triton.amd_autotune_diagnostics_path` to retain JSONL input geometry,
selected configurations and candidate times. Structured traces also include
`amd_kernel_autotune` artifacts. `warm_arg_names` records the producer estimate;
mutated input storage is additionally restored and touched before timing.

## Benchmark

Use an environment built from this checkout. Compilation is serial in the
accepted experiment and compilation time is separate from GPU execution time.

```sh
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python benchmarks/inductor_backends/amd_kernel_hillclimb.py \
  --case-manifest benchmarks/inductor_backends/amd_kernel_reference_cases.json \
  --output /tmp/amd-reference --rounds 15 --repeats 5
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python benchmarks/inductor_backends/amd_kernel_hillclimb.py \
  --output /tmp/amd-broad --rounds 15 --repeats 5
python benchmarks/inductor_backends/plot_amd_kernel_hillclimb_v2.py /tmp/amd-reference
python benchmarks/inductor_backends/plot_amd_kernel_hillclimb_v2.py /tmp/amd-broad
```

The reference manifest has 18 recorded model/kernel geometries. The broader
matrix has 96 cases: eight families, six shapes and BF16/FP32. Variants are
original Triton, optimized Triton, FlyDSL, and optimized Triton with approximate
math. Backend competition is disabled for the FlyDSL variant; unsupported
FlyDSL operations can still fall back, with routes retained in diagnostics.
GEMM and convolution remain configured for ATen.

Add `--quick` for one shape per family/dtype, or `--families mean,transpose`.
Completed cases are reused. Use a fresh directory after changing compiler code
or the protocol. Each result includes source fingerprints.

The benchmark freezes the `compile_fx` callable and its input mapping for each
variant. Subsequent Dynamo resets cannot silently replace a measured variant.
It validates outputs and checks that each variant compiled exactly once. Both
direct compiled execution and CUDA graph replay are measured separately.

Timing uses 15 rotating rounds and five invocations per round by default. A
256 MiB flush precedes each sample. Warm measurements touch input storage after
the flush. Mandatory mutation restores occur outside timing in both cache
modes, and can warm those buffers. The in-place GELU reference restores its
original gradient before every invocation. Graph replay includes graph launch
cost; direct GPU event measurements do not include Python wall time. Neither
is full-model training throughput.

Accuracy is checked on three input seeds. Transpose, broadcast and cast/add
require exact values. Other FP32 outputs use `atol=rtol=1e-4`; low precision
outputs use `0.02`. Additional regression tests use tighter tolerances for the
approximate math implementations, nonfinite inputs, tails and saved statistics.
Passing these checks is not training-convergence validation.

## Outputs

- `per_case.csv`: all variants/cache/invocation modes, runtimes, p10/p90 round
  statistics, time reductions, speedup factors, compilation time and coverage.
- `summary.csv`: family geomeans, retaining regressions.
- `<case>/result.json`: raw samples, source hashes, frozen-backend checks and
  failures. Maximum error fields describe the initial input seed; other seeds
  are also validated.
- `<case>/<variant>.py`: generated wrappers and kernels.
- `<case>/<variant>.tuning.jsonl`: new tuner selections and candidate timings.
- `failures.json`: compilation and numerical failures, without filtering.
- `direct/` and `graph_replay/`: separate CSVs, common-four-variant summaries,
  HTML reports, and per-family linear/log runtime SVGs. Log bars start at 0.1 us.
- `report.html`: navigation between invocation modes.

Original hill-climb kernel replays are reported separately from newly compiled
operation graphs. The replays preserve the original kernel contract and modeled
cache state, which may differ from an out-of-place function or a graph with
cold external inputs. Broad case geomeans are not invocation-weighted model
speedups. Both gains and cache-dependent regressions remain visible.

The accepted local run completed 18 reference and 96 broad cases in all four
variants, with no compilation or numerical failures. Validation used a Python
source overlay over the existing built checkout. Asynchronous source-overlay
compiler workers failed during exploratory runs; the successful measurements
use one compiler thread. `spin` is absent from the existing environment, so
repository lint was not run. Changes remain uncommitted for review.

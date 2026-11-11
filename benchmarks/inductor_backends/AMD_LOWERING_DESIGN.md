# AMD lowering and autotuning design

This document describes the experimental Inductor AMD paths and their fallback
contracts. Generated kernels remain Triton or Gluon; no model names or benchmark
case identities participate in dispatch. GPU correctness and performance work
has used MI355X (`gfx950`, wave64). Wave32 and other AMD devices have CPU tests
for capability checks, emitted layouts, and legal candidate generation; those
tests do not establish GPU correctness or a performance improvement there.

## Configuration and entry points

All four boolean options in `torch/_inductor/config.py::triton` default to
`False`. Enable the base option before enabling an additional tier.

| Option | Additional behavior |
| --- | --- |
| `amd_kernel_optimizations` | Guarded pointwise/row source conversion, masked-modulo simplification, reduction candidate expansion, and bounded autotuning. |
| `amd_extended_kernel_optimizations` | Flat elementwise Gluon conversion, larger persistent row reductions, and producer/input-warm cache scoring. |
| `amd_further_kernel_optimizations` | Outer-row layout alternatives, the retained additive BF16 wide-scan path, selective finalist confirmation, eligible atomic tuning, and captured candidate measurement. |
| `amd_approximate_math` | Separate numerical policy for matched GELU backward and matched softmax exponential/reciprocal approximations. Layout tiers alone do not enable this policy. |

`torch/_inductor/codegen/triton.py::TritonKernel.codegen_kernel` constructs
metadata and applies these source passes in order:

```text
simplify_masked_modulo
optimize_pointwise_source
optimize_row_source
optimize_gelu_source             [approximate math only]
optimize_flat_pointwise_source   [extended tier only]
```

The passes live in `torch/_inductor/codegen/amd_kernel_optimizations.py`.
Unmatched source returns unchanged. A pass that has already changed the
decorator or signature also prevents a later pointwise matcher from treating
the result as an ordinary unconverted kernel. Deterministic algorithms,
deterministic Inductor configuration, and batch-invariant mode bypass the AMD
entry point. Cooperative reductions retain the existing launcher/tuning path.

## Static pointwise arithmetic, casts, and transpose

`optimize_pointwise_source` recognizes exact static signatures, address
expressions, supported floating pointer types, masks, and arithmetic expression
trees. It provides explicit vector/warp layout and unroll parameters for two
arithmetic forms: flat addition and scalar-times-vector broadcast arithmetic.
For example, an ordinary addition with a Triton-assigned layout becomes:

```python
# Before: one vector per program, layout chosen implicitly.
a = tl.load(in_ptr0 + x0, xmask).to(tl.float32)
b = tl.load(in_ptr1 + x0, xmask).to(tl.float32)
tl.store(out_ptr0 + x0, a + b, xmask)

# After: explicit layout and disjoint unrolled subtiles.
lane = gl.arange(0, XBLOCK // AMD_UNROLL, layout=layout)
for u in gl.static_range(AMD_UNROLL):
    i = gl.program_id(0) * XBLOCK + u * (XBLOCK // AMD_UNROLL) + lane
    a = gl.load(in_ptr0 + i, i < N, other=0).to(gl.float32)
    b = gl.load(in_ptr1 + i, i < N, other=0).to(gl.float32)
    gl.store(out_ptr0 + i, a + b, i < N)
```

The broadcast matcher preserves the expression
`(scalar * (1 / N)) * (vector * 2.0)` and loads its scalar once before the
unrolled loop. The matcher checks the exact constants and expression structure;
it does not authorize algebraic reassociation of unrelated expressions.

Transpose/copy matching requires equal input/output types and proven index
decompositions with positive, divisible extents. The emitted
`AMD_READ_ORDER` selects output-linear traversal with a strided input address
or input-linear traversal with a permuted output address. Both alternatives
implement the same checked permutation. Transpose unrolling remains one;
other unroll factors are rejected by candidate generation.

`optimize_flat_pointwise_source` covers the extended tier's straight-line
elementwise fusions, including supported casts, dropout-backward arithmetic,
SiLU expressions, and optimizer arithmetic that already fits the original
flat matcher. It converts `tl` names to `gl`, adds explicit layouts to vector
construction, and recreates masks for each unrolled subtile. Supported scalar
loads remain scalar. It preserves casts and the existing sigmoid formula
`1 / (1 + exp(-x))`; it does not select a new approximate-math policy.

Vector addresses must be `pointer + x0`, with `x0 = xindex`. Stores cannot use
arbitrary addresses. Existing explicit vector masks must be `xmask` or `None`;
the converted vector operations receive the newly computed `xmask`. This keeps
tails safe when a tuned tile extends beyond the static element count.

The broader indirect gather and `sqrt_rn`/Adam extension is excluded. Gather
bounds assertions, indirect addressing, and unsupported flat operations keep
their original Triton code. An optimizer graph that fits the existing supported
flat operations can still convert; exclusion applies to the proposed broader
path, not to every graph containing an optimizer update.

## Split reductions and masked modulo

`simplify_masked_modulo` targets a split-reduction load of the form:

```python
tl.load(ptr + ((r0_1 + R * x0) % M), r0_mask & tmp_upper & xmask)
```

It replaces that address with `ptr + (r0_1 + R * x0)` only after checking
positive static extents, the nonoverflowing signed-32-bit flattened range, the
standard row/reduction index construction, the reduction loop bounds, both
dimension masks, and a resolved strict predicate `r0_1 + R * x0 < M`.
The load mask remains. Missing predicates, non-strict upper bounds, altered
index construction, and larger index products leave the modulo intact.

`runtime/triton_heuristics.py::_amd_reduction_configs` adds bounded alternatives
to the existing persistent or split-reduction configurations. Existing
configurations remain candidates. Extra row/reduction block sizes and warp
counts are filtered using the device's shared resource limits. The extended
persistent override in `TritonKernel.should_use_persistent_reduction` applies
only within those limits and retains the normal choice otherwise.

## Persistent rows, saved statistics, and outer layouts

`optimize_row_source` converts eligible persistent reductions into an explicit
two-dimensional blocked layout. Row reductions, normalization statistics,
saved statistics, and their elementwise consumers retain a common layout.
Original reduction operators, output addresses, and masks are preserved.
`triton_helpers.fmax2`, `max2`, and `any` receive explicit Gluon counterparts;
the custom maximum helper retains its NaN-aware choice.

For the usual inner layout, emitted structure is:

```python
layout = gl.BlockedLayout(
    [1, AMD_VEC], [wave_size // AMD_LANES, AMD_LANES],
    [1, AMD_WARPS] if AMD_SPLIT_WARPS else [AMD_WARPS, 1], [1, 0],
)
rows = gl.arange(0, XBLOCK, layout=gl.SliceLayout(1, layout))[:, None]
cols = gl.arange(0, R0_BLOCK, layout=gl.SliceLayout(0, layout))[None, :]
```

The source contains the device's serialized wave size, not a hardcoded wave64
layout. `persistent_reduction` in `runtime/triton_heuristics.py` supplies
`AMD_LANES`, `AMD_VEC`, `AMD_WARPS`, and `AMD_SPLIT_WARPS`. Its row counts must
divide the actual row count and fit the persistent tile budget.

With the further tier and `OUTER`/`OUTER_TINY` reduction hints,
`AMD_ROW_ORDER` also permits the opposite layout order: vector ownership along
the row axis and complementary lanes along the reduction axis. This changes
ownership/traversal, not the normalization or saved-statistics formula.
Unsupported helper functions, unavailable Gluon/libdevice operations, loops,
and explicitly excluded operations return the original source.

## Retained additive BF16 scan

The retained scan conversion is deliberately narrow: a persistent reduction,
BF16 input, reduction size hint 32768, further tier enabled, and a local
two-argument helper whose whole body assigns `left + right` and returns that
assignment. Scan calls must reference a validated helper. Additional argument
kinds/defaults, different arithmetic, bare returns, and unvalidated combine
functions are rejected.

```python
# Before
@triton.jit
def combine(left, right):
    result = left + right
    return result
result, = tl.associative_scan((value,), 1, combine)

# After: imports precede the lifted helper; arithmetic and scan axis remain.
@gluon.jit
def combine(left, right):
    result = left + right
    return result
result, = gl.associative_scan((value,), 1, combine)
```

Scan metadata requests finalist confirmation. Existing row/reduction masks
remain, including a tail below the size hint. FP32 and smaller BF16 scan
candidates keep their original Triton implementation. The 32768 restriction
records the retained MI355X performance experiment; it is not a hardware
requirement or a claim that this is the best scan tile on every device.
The property-derived register budget can additionally reject this branch on a
smaller device. Such rejection falls back instead of expanding scan eligibility.

## Explicit approximation policy

`optimize_gelu_source` is called only with `amd_approximate_math`. It matches
the precise GELU-backward derivative expression, compatible flat addresses,
static extent, and supported input/output types before replacing its erf/CDF
calculation with a polynomial sharing one Gaussian exponential. It also uses
an AMD reciprocal instruction. This is an approximate numerical path and is
identified by `amd_approximate_math=True` in diagnostics.

Within `optimize_row_source`, the separate flag permits matched softmax to use
`gl.exp` and reciprocal multiplication in place of libdevice exponential and
division. Without that flag those replacements do not occur. The layout tiers,
hardware generalization, atomic eligibility, and timing protocol do not enable
the approximation flag implicitly.

## Runtime tuning and mutation contracts

`runtime/triton_heuristics.py::CachingAutotuner.benchmark_all_configs` enters
`runtime/amd_autotune.py::tune` only for eligible AMD metadata and compatible
`XBLOCK` launchers. Cooperative kernels are excluded at codegen; TMA, host
descriptor, and combo-grid metadata exclude the runtime path. Atomic kernels
require the further tier's `amd_allow_atomic_tuning` permission and known
mutated arguments. Otherwise existing Inductor autotuning remains active.

`benchmark_configs` copies entire mutated storages, then rebuilds scratch views
with the original dtype, storage offset, shape, and stride. Arguments sharing a
mutated storage share its scratch copy. Before each invocation it restores the
scratch storage and applies reset-to-zero arguments. Candidate timing therefore
does not repeatedly accumulate into the user's mutated inputs. Ordinary output
buffers are still outputs and must be overwritten by a valid launcher.

Every candidate is checked against the first launcher's output. Floating
outputs are poisoned before checking to expose missing stores; integer outputs
receive a distinct bitwise pattern. These checks compare launch configurations,
so independent eager-reference tests remain necessary for source rewrites.

Producer-cache modeling uses
`codegen/triton.py::TritonKernel._amd_warm_arg_names` to estimate recent inputs
from scheduler dependencies. `benchmark_configs` flushes 256 MiB, restores
mutations, and touches those storages before timing. Extended layout families
also measure an all-inputs-warm context. Candidate/context order rotates;
round scores are geometric means across contexts, with a median across rounds.
This is an explicit cache-state model rather than an observation of a full
application's steady-state cache contents.

`_neighbors` varies compatible block sizes, vector widths, unrolling, lanes,
layout order, split-warps, warp counts, and `waves_per_eu`. `tune` explores at
most three rounds with ten new neighbors per round. The scratch-memory budget
can decline this tuner and return control to the existing path. Configuration
identity includes launch parameters so candidates are not silently conflated.

Further-tier metadata selects confirmation for atomic kernels and selected
reduction counts; retained scans force it. The best six finite finalists receive
nine rounds of five repetitions. Final selection and the returned timing map
use confirmed finalists only, preventing a stale exploratory measurement from
overriding confirmation. JSONL diagnostics and the `amd_kernel_autotune`
structured-trace artifact retain configurations, raw/context samples, numerical
policy, timing protocol, and the selected configuration.

## GPU measurement boundaries

Further-tier candidate tuning sets `amd_graph_tuning` and reports
`captured_gpu_protocol`. `benchmark_configs` captures preparation, external
start event, launcher, and external end event. Preparation executes before the
start event. It resolves the launcher's raw stream inside the capture context,
so the captured launcher uses the capture stream. Replaying this protocol
removes CPU submission gaps from the candidate event interval.

Operation benchmarks use
`benchmarks/inductor_backends/amd_benchmark_protocol.py::measure` and distinguish
two protocols explicitly:

| Invocation | Protocol | Host overhead can contribute |
| --- | --- | --- |
| `direct` | `captured_gpu_kernel_sequence`: capture the frozen callable's actual GPU work and external events. | No |
| `graph_replay` | `inline_graph_replay_events`: inline GPU events around an actual host-issued graph replay, with preparation outside the interval. | Yes; `host_overhead_included=True` |

The second protocol does not nest a graph replay inside another capture. ROCm
can omit that nested replay; it is unsupported as a measurement substitute.
Preparation is outside both timed intervals. Compile time is recorded separately.
Frozen callables, source fingerprints, protocol metadata, numerical checks, and
mutation probes establish what was measured. Timing data from different source
fingerprints or protocols must remain separate. These measurements do not
establish full-model training throughput or convergence.

## Hardware capability and property contracts

`supports_amd_kernel_optimizations` requires HIP, a `gfx` ISA identifier, wave
size 32 or 64, and a reported block-thread limit that can hold an integral
positive number of waves. Other or incomplete capabilities use existing codegen.
`_source_limits` decodes the serialized `DeviceProperties` call; malformed or
unsupported explicit device metadata leaves source unchanged. Legacy
source-only fixtures without device metadata retain their historical wave64
fallback; real codegen supplies checked device properties.

`runtime/hints.py::AMDConfigLimits.from_device` is shared by source conversion,
initial configurations, and neighbor search. It bounds warp count by both
`max_threads_per_block // warp_size` and the supported compiler-search ceiling
of 16. Its persistent tile budget is `min(32768, regs_per_multiprocessor // 4)`,
with a 32768 fallback when register capacity is unavailable. This budget is a
conservative search heuristic, not a register-occupancy proof.

Codegen serializes `amd_warp_size` and `amd_max_threads_per_block`; runtime also
receives the authoritative `DeviceProperties`. Row lane choices scale with wave
size, layouts use that same wave size, and warp/tile filtering uses the shared
limits. Row source conversion and the persistent override decline tiles beyond
the budget. Capability checks provide structural eligibility; only MI355X has
GPU validation in this experiment. Other devices still require their own
numerical, compilation, and performance validation before making benefit claims.

## Relevant tests

CPU source contracts are covered by
`test/inductor/test_amd_kernel_optimizations.py::TestAMDSourceOptimizations`,
`test_amd_extended_kernel_optimizations.py::TestAMDFlatSource`,
`test_amd_further_kernel_optimizations.py::TestAMDIndirectMutation` and
`TestAMDScanSource`, and `test_amd_kernel_hardware.py::TestAMDHardwareSource`.
Runtime hardware tests cover legal initial/neighbor configurations.
Device tests separately cover numerics, tails, saved statistics, rejected
scan/indirect paths, and mutable outputs. Passing CPU tests does not replace
those device checks.

Captured candidate measurement and confirmation target eligible reductions and atomics.
Flat pointwise kernels and three-component Welford statistics retain their
earlier selection protocol; the retained wide BF16 scan explicitly enables
captured measurement. This restriction avoids extending a measured regression
to unrelated forward-normalization and elementwise paths.

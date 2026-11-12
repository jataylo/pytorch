# Branch context

The original checkout is detached at `937d7fd7cdb37e3136c90e86ca71b57d8e5193be`
from `jataylo/pytorch`, branch `flydsl-simd-worker-preload`.

The branch implements an opt-in FlyDSL SIMD generator inside Inductor. It inherits
Triton's scheduler, fusion and tiling analysis, tries a FlyDSL kernel for each SIMD
schedule, and regenerates refused schedules through Triton. Each compiled graph's
wrapper can therefore contain both launcher types. Normal Dynamo graph breaks create
additional compiled graph wrappers and eager regions.

The recent commit groups explain the main design:

- `f686e2cbb41`, `d5bbb66e33c`, `2040661765f`: dynamic shape support, tiled layout
  handling and wider indexing. Buffer descriptors and raw global pointers have different
  size constraints; oversized or unprovable indexing still needs a safe fallback.
- `86a9e09b0c3`, `6feb1230de1`, `d786cdb5859`: scans, mixed-order reductions, combo
  kernels, short sorts and launch improvements. These paths have layout and workspace
  constraints beyond ordinary row reductions.
- `b27416b61a1`, `afd70469e6a`: arithmetic, reduction and dtype coverage, including
  chunked Welford, softmax, integer atomics and bitcasts.
- `e8ab14613eb`, `93884d0b6a5`: focus the branch on SIMD using FlyDSL 0.3 and gate
  supported AMD architectures. The FNUZ FP8 gate is narrower than general CDNA4 support.
- `2d7d05c40e5`: make compiler-worker preloading FlyDSL-specific, avoiding unnecessary
  FlyDSL setup for ordinary Triton jobs. The last three commits are cleanup and typing
  changes; `937d7fd7cdb` removes benchmark numbers from compiler comments.

The v2 changes add broader bounded autotuning, safe candidate benchmarking, optional
per-kernel FlyDSL/Triton selection and lowering diagnostics. Backend selection defaults
off, so selecting FlyDSL prefers it for every supported SIMD schedule. GEMM and conv
are explicitly forced to ATen in the training comparison.

The separate support worktree adds the observed multi-axis layouts, tuple softmax
combining and a mixed-order layout correction. Its bounded search can try the opposite
inner output axis for two-output-axis reductions. It is kept separate from the frozen
v2 measurement checkout, with its own source hashes, tests and performance artifacts.

The original backend's 128-test control has eight errors on this gfx950/ROCm10 setup.
The validated support revisions have retained those same error names. Focused tests
and per-model results distinguish new regressions from the original hardware/library
limitations. Kernel probe timings are kept outside compiler source comments and are
not used as claims about full training-model performance.

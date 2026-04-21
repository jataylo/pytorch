# Stage 3a — Kernel Dispatch Overhead (T_overhead)

**Source:** `BottleneckAnalysis.analyze_bottleneck()` in `triton_heuristics_adaptive.py`

**Role:** Estimates the fixed time cost of *launching* the kernel before any thread
executes a single floating-point instruction.  This cost is **config-dependent** — it
changes with `num_warps` and the number of blocks launched.

---

## Technical

### The equation

```
T_overhead = K_launch
           + (n_args − 3) × K_arg
           + (num_warps − 1) × K_warp
           + grid_overhead(num_blocks)
           + masking_overhead

where:
  K_launch  = 3.0 µs     fixed CP dispatch cost (empirically measured, AMD ROCm)
  K_arg     = 0.1 µs     per extra tensor pointer beyond baseline of 3
  K_warp    = 0.2 µs     per extra wavefront declared beyond first
  n_args    = num_inputs + num_outputs
  num_warps = config parameter
  num_blocks = ceil(xnumel / XBLOCK) × ceil(ynumel / YBLOCK) × …
```

### Reading each term

#### `K_launch = 3.0 µs` — fixed CP dispatch cost

When the host calls `hipLaunchKernel` (ROCm) or `cuLaunchKernel` (CUDA), a chain of
hardware events must complete before any wavefront begins executing:

1. **DMA to CP:** The kernel descriptor (containing the GDS pointer, SGPR setup, VGPR
   allocation requirements, and the code object address) is DMA'd from CPU-visible memory
   (pinned host or GPU-accessible memory) into the **Command Processor (CP)** input FIFO.
   The CP is a small embedded microcontroller on the GPU die.

2. **CP parsing:** The CP reads and parses the kernel descriptor, setting up the dispatch
   packet.

3. **CP → SPI signal:** The CP signals the **Shader Processor Input (SPI)** unit with the
   wavefront parameters.

4. **SPI allocation:** The SPI allocates compute units, assigns VGPR/SGPR register banks,
   and begins wavefront enqueuing.

The entire sequence takes ~3 µs on AMD RDNA/CDNA hardware measured empirically.  This is
the **irreducible minimum** — it occurs even for a kernel that does a single addition on
a single element.

#### `(n_args − 3) × K_arg = (n_args − 3) × 0.1 µs` — pointer setup

Each tensor pointer must be copied from CPU-visible memory into **Scalar General-Purpose
Registers (SGPRs)** on the GPU before the wavefront begins.  SGPRs are shared across all
threads in a wavefront and hold kernel arguments (pointer base addresses, scalar strides,
etc.).

The baseline of 3 represents a typical two-input one-output pointwise kernel.  Each
additional tensor pointer (`n_args − 3` extra) requires one additional SGPR load from
the kernel descriptor, adding ~0.1 µs to the setup phase.

**Practical example:** A kernel fusing 4 inputs + 2 outputs:
```
extra_args = 6 − 3 = 3
extra cost = 3 × 0.1 µs = 0.3 µs
```

#### `(num_warps − 1) × K_warp = (num_warps − 1) × 0.2 µs` — SPI wavefront init

For each wavefront declared in a block, the SPI must:
1. Allocate a VGPR register bank (a contiguous slice of the 65,536 VGPRs on that CU)
2. Allocate an SGPR register bank
3. Write the initial SGPR values (kernel arguments, wavefront ID, etc.)
4. Enqueue the wavefront into the CU's wavefront pool

Each of these steps adds overhead.  With `num_warps=16`, the SPI initialises 16 wavefronts
per block — adding `15 × 0.2 = 3.0 µs` on top of the base `K_launch = 3.0 µs`.  This
**doubles total overhead** from 3 µs to 6 µs before the first thread runs.

This is why the occupancy scoring model (Stage 4e) penalises high `num_warps` for
launch-dominated kernels: the SPI init cost can easily exceed the kernel's entire useful
execution time.

#### `grid_overhead(num_blocks)` — CP batch dispatch

The AMD CP maintains a finite dispatch queue.  When `num_blocks > 1000`, the CP can no
longer enqueue all blocks in a single batch and must dispatch in multiple rounds,
stalling between rounds while waiting for CUs to report completion.

The overhead is modelled as logarithmic:
```
if num_blocks > 1000:
    grid_overhead = 0.5 × log₂(num_blocks / 1000) µs
else:
    grid_overhead = 0
```

At `num_blocks = 8000`, `grid_overhead ≈ 0.5 × log₂(8) = 1.5 µs`.
At `num_blocks = 65536`, `grid_overhead ≈ 0.5 × log₂(65.5) ≈ 3 µs`.

#### `masking_overhead`

When `total_elements` is not a multiple of `XBLOCK`, tail blocks must execute with a mask
(`tl.load(..., mask=offs < n)`).  The predicate evaluation adds a small overhead per block
for the masked comparison instruction:
```
masking_overhead = 0 if total_elements % XBLOCK == 0
                 = 0.05 µs otherwise
```

### Output fractions

After computing `T_overhead`, `T_memory`, and `T_compute` (Stages 3a–3c), the fraction
each component contributes to the total gross cost is computed:

```
gross        = T_overhead + T_memory + T_compute   # (sum, not max — measures "effort")
overhead_frac = T_overhead / gross
memory_frac   = T_memory   / gross
compute_frac  = T_compute  / gross
```

Note: `gross` uses the sum, not `max(T_memory, T_compute)`, so fractions always sum to
1.0 and accurately reflect each component's *relative share of effort* even when memory
and compute overlap on the GPU.

`overhead_frac > 0.50` sets the `launch_bound` flag, which triggers the
`🚀 LAUNCH-BOUND` label in verbose output and switches the occupancy scoring from the
wavefront sweet-spot model to the first-principles overhead ratio (see Stage 4e).

---

## Simple explanation

### What "overhead" means

Every time the GPU starts a kernel — before any useful work begins — it must complete a
startup sequence.  Think of it like a factory conveyor belt: before any product comes out,
you have to power up the machine, load the programme, and prepare the workstations.  That
setup time is overhead.

On AMD hardware, this minimum startup cost is **~3 microseconds** (µs) — measured
empirically.  For a kernel that processes 512 numbers and finishes in 1–2 µs, the startup
cost is *larger than the actual work*.  For a kernel processing 100 million numbers and
running for 1,000 µs, the 3 µs startup is negligible.

### The three config-controlled costs

**1. Fixed cost (3 µs, always):**
The GPU's Command Processor must read the kernel description, set up registers, and tell
the GPU's scheduler to start.  This happens no matter what config you pick.

**2. Extra tensor pointers (+0.1 µs each):**
Each tensor the kernel reads from or writes to needs its memory address loaded into a
special register before any thread starts.  More tensors = more setup time.  Two inputs
and one output is the baseline; every additional tensor adds ~0.1 µs.

**3. More wavefronts (+0.2 µs each):**
`num_warps` controls how many wavefronts (groups of 64 threads) are declared per block.
Each wavefront the GPU declares must be initialised: registers allocated, IDs assigned,
arguments written.  With `num_warps=16`, this adds `15 × 0.2 = 3 µs` — **doubling
the total overhead** to 6 µs before a single thread executes useful code.

### Key insight

For tiny kernels, overhead is the dominant cost and `num_warps=1` is almost always the
right choice.  For large kernels, overhead is irrelevant and more warps help hide memory
latency.  Stage 3 measures how big `T_overhead` is relative to `T_memory` and
`T_compute` so the scoring stage knows which one matters.


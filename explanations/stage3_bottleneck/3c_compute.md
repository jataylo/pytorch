# Stage 3c — Compute Throughput Time (T_compute)

**Source:** `BottleneckAnalysis.analyze_bottleneck()` → `estimate_compute_time_us()`
in `triton_heuristics_adaptive.py`

**Role:** Estimates how long the kernel would take if memory bandwidth were infinite and
the ALUs were the only constraint — the pure compute ceiling.  For most pointwise kernels
this is far smaller than `T_memory`, but it becomes dominant for kernels heavy in
transcendental operations (`exp`, `log`, `sin`, etc.).

---

## Technical

### The equation

```
T_compute = total_ops / (TFLOPS_peak × η_instr)

where:
  total_ops    = total_elements × ops_per_element
  TFLOPS_peak  = device peak FP32 throughput (e.g. 200 TFLOPS on MI300X)
  η_instr      = instruction-mix efficiency factor (0.60 – 0.80)
```

### Reading each term

#### `total_ops = total_elements × ops_per_element`

`ops_per_element` is the weighted average number of floating-point operations per output
element, derived from the instruction-mix scan in Stage 2:

```
ops_per_element = (n_fast × w_fast + n_med × w_med + n_slow × w_slow) / total_elements

where the weights approximate the relative computational cost:
  w_fast = 1    (FMA, add, mul — 1 cycle each on the FP32 pipeline)
  w_med  = 4    (sqrt, abs, int div — 4–16 cycles)
  w_slow = 32   (exp, log, sin, FP div — 16–64 cycles via SFU)
```

The use of cycle-count weights rather than raw instruction counts is important: a kernel
with 10 `tl.exp` calls is not equivalent to a kernel with 10 `tl.add` calls.  The
weighted sum correctly captures the ALU load each instruction class contributes.

#### `TFLOPS_peak` — device peak FP32 throughput

Read from `torch.cuda.get_device_properties()`.  On AMD CDNA3 (MI300X):

```
TFLOPS_peak = 304 CUs × 64 SIMD-lanes × 2 FLOPs/FMA × 2 GHz clock
            ≈ 200 TFLOPS
```

The `2 FLOPs/FMA` factor counts the multiply and the add in one FMA instruction as two
floating-point operations (the standard FLOP accounting convention).

This peak assumes 100% FMA pipeline occupancy — every SIMD lane executing an FMA on
every clock cycle.  Real workloads never achieve this because of:
- Data dependencies (a later instruction needs the result of an earlier one)
- Instruction latency (FMA has a 4-cycle latency; back-to-back FMAs stall unless
  independent instructions fill the pipeline)
- Wavefront context switching (reduces effective utilisation when wavefronts stall)

#### `η_instr` — instruction-mix efficiency

Not all instructions execute at the FMA rate.  The `η_instr` factor accounts for the
fraction of pipeline slots consumed by slow instructions:

| Instruction mix | η_instr | Reason |
|---|---|---|
| Mostly FMA / add / mul (n_slow / total < 0.10) | **0.80** | FMA pipeline stays nearly full; minor stalls from data dependencies |
| Mixed (0.10 ≤ n_slow / total < 0.50) | **0.70** | SFU instructions interleave with FMA, creating pipeline bubbles |
| Transcendental-heavy (n_slow / total ≥ 0.50) | **0.60** | SFU is the bottleneck; the FMA pipeline stalls waiting for SFU results |

**Why the SFU causes stalls:**

The Special Function Unit (SFU / Transcendental Unit) is a separate hardware block
from the FMA pipeline.  It handles `exp`, `log`, `sin`, `cos`, `rcp` (reciprocal):

1. The SFU receives the input register value and begins a table-lookup + polynomial
   approximation sequence.
2. This takes 16–32 clock cycles during which the issuing wavefront cannot issue a
   dependent instruction.
3. If more than half the instructions are SFU-bound, the FMA pipeline runs dry waiting
   for SFU results — effective throughput drops to ~60% of peak.

A kernel that computes `sigmoid(x) = 1 / (1 + exp(-x))` performs both `exp()` (SFU)
and `1 / (...)` (SFU again via `rcp`).  This is ~50% SFU load → `η_instr = 0.70`.

A kernel computing `gelu(x) = x × Φ(x) ≈ x × sigmoid(1.702x)` similarly routes through
the SFU for the sigmoid term.

**A100 / CUDA note:** NVIDIA hardware has a wider SFU pipeline than AMD CDNA, so the
penalty for transcendental-heavy kernels is somewhat smaller on NVIDIA.  The efficiency
coefficients above are calibrated for AMD CDNA2/CDNA3.

### Complete example — MI300X, softmax inner loop, FP32, N=10M

A simplified softmax kernel reads one tensor, computes `exp(x)` per element, then divides
by a scalar sum.  Per element: 1 `exp` (slow) + 1 `div` (slow) + 1 add for the sum.

```
n_fast          = 1 (the add)
n_med           = 0
n_slow          = 2 (exp + div)
ops_per_element = (1×1 + 0×4 + 2×32) / 1 = 65 FLOPs/element

total_ops       = 10⁷ × 65 = 6.5 × 10⁸ FLOPs
TFLOPS_peak     = 200 × 10¹² FLOPs/s
η_instr         = 0.60  (>50% SFU instructions)

T_compute       = 6.5 × 10⁸ / (200 × 10¹² × 0.60)
                ≈ 5.4 × 10⁻⁶ s
                ≈ 5.4 µs
```

Comparing with the memory time for the same kernel (reading 2 tensors at 12 bytes/elem):
```
T_memory ≈ 10⁷ × 8 / (5300 × 10⁹ × 0.80) ≈ 18.9 µs
```

Since `T_memory (18.9 µs) > T_compute (5.4 µs)`, even this transcendental-heavy kernel
is **still memory-bound** on MI300X.  `T_compute` is hidden inside `T_memory` in the
roofline combination (Stage 3d).

This illustrates why HBM bandwidth is almost always the bottleneck for pointwise kernels:
the MI300X's enormous bandwidth (5,300 GB/s) means even compute-heavy pointwise ops are
data-starved.

---

## Simple explanation

### What "compute time" means

`T_compute` asks: *if we could magically fetch all data instantly with no memory delay,
how long would the math itself take?*

The GPU's math units (ALUs) can do an incredible number of additions and multiplications
per second — 200 trillion floating-point operations per second (200 TFLOPS) on MI300X.
`T_compute` divides the kernel's total operation count by that rate to get the minimum
time the math alone would take.

### Why it's almost always smaller than T_memory

A simple `a + b` kernel does 1 addition per 12 bytes of memory traffic.  At 200 TFLOPS,
the ALU can do 200 × 10¹² additions per second.  At 5,300 GB/s, the memory bus can
deliver 5300 × 10⁹ / 12 ≈ 440 × 10⁹ elements per second.

The ALU rate is 200,000 × 10⁹ elements/sec, while the memory rate is only 440 × 10⁹
elements/sec.  The ALUs are **450× faster** than the memory bus for this kernel —
so the ALUs are always waiting for data, never the bottleneck.

### When compute time does matter

Kernels with many expensive operations per element change the balance:

- **`exp(x)` takes ~32 cycles** on AMD hardware (through the Special Function Unit).
  If the kernel calls `exp` on every element, the ALU is busy for much longer per element,
  and the memory bus has time to catch up.

- **A very compute-heavy kernel** (many `exp`, `log`, `sin` in sequence) can shift from
  memory-bound to compute-bound.  At that point, increasing `num_warps` to hide memory
  latency stops helping — the bottleneck is now the ALU pipeline, not waiting for data.

### The efficiency factor

Not all instructions run at the same speed.  The "peak TFLOPS" rating assumes every
instruction is a single-cycle FMA.  In practice:
- **FMA (multiply-add):** 1 cycle — full speed
- **`sqrt`, `abs`:** 4–16 cycles — slower
- **`exp`, `log`, `sin`, FP division:** 16–64 cycles through a dedicated hardware unit

The `η_instr` factor (0.60–0.80) reduces the peak TFLOPS to account for this mix.
A kernel that is 50%+ transcendentals uses `η_instr = 0.60`, meaning the ALU pipeline
is only 60% as effective as it would be with pure FMA code.


# Stage 4e — Occupancy Score

**Source:** `score_occupancy()` in `triton_heuristics_pointwise.py`

**Role:** Scores whether the number of resident wavefronts per Compute Unit (controlled
by `num_warps`) is appropriate for this kernel's regime.  The scoring model has three
distinct branches depending on problem saturation and block count, because the optimal
wavefront count differs fundamentally across these regimes.

---

## Technical

### Background — VGPRs, occupancy, and latency hiding

AMD CDNA hardware provides **65,536 Vector General-Purpose Registers (VGPRs)** per CU.
These 65,536 VGPRs are shared equally among all wavefronts resident on that CU:

```
vgprs_per_wavefront = 65,536 / num_resident_wavefronts
vgprs_per_thread    = vgprs_per_wavefront / warp_size   (warp_size = 64 on AMD)
```

Hardware VGPR allocation happens in **8-register granules** — the compiler rounds the
required VGPRs up to the next multiple of 8:

```
max_vgprs_per_thread = min(256, floor(65536 / (num_warps × 64) / 8) × 8)
```

| num_warps | threads/block | VGPRs/wavefront | VGPRs/thread (granule-aligned) |
|---|---|---|---|
| 1 | 64 | 65,536 | 256 (max) |
| 2 | 128 | 32,768 | 256 (max) |
| 4 | 256 | 16,384 | 248 |
| 8 | 512 | 8,192 | 128 |
| 16 | 1,024 | 4,096 | 64 |

**Latency hiding:** The GPU CU executes instructions in a wavefront-interleaved pipeline.
When one wavefront issues a load instruction (`tl.load`), it enters a stall for ~300 cycles
while the HBM round-trip completes.  The CU immediately switches context to another
wavefront that has ready operands.  If enough wavefronts are resident, the CU stays busy
executing different wavefronts' instructions while earlier wavefronts wait for memory.

**The fundamental trade-off:**
- More wavefronts → better latency hiding → higher ALU utilisation
- More wavefronts → fewer VGPRs per thread → higher spill risk → slower execution

### Saturation parameter

The occupancy regime is selected by:

```
saturation = num_blocks / (4 × num_CUs)    # fraction of "full occupancy"
```

`4 × num_CUs` is used as the normalisation (not `num_CUs`) because empirically, AMD
hardware achieves maximum throughput when each CU has ~4 resident blocks (4 wavefronts
× 64 threads × 4 blocks = 1024 threads/CU, close to the hardware maximum of 2048).

### Regime 1 — Memory-bound or well-saturated (`saturation ≥ 0.25`)

The **sweet-spot Gaussian** model applies:

```
sweet_min = 4   wavefronts   (minimum for effective latency hiding)
sweet_max = 8   wavefronts   (maximum before register pressure dominates)

if sweet_min ≤ num_warps ≤ sweet_max:
    Occ_score = 1.00

elif num_warps < sweet_min:
    Occ_score = 0.75 + 0.25 × exp(−0.5 × ((num_warps − sweet_min) / 2.0)²)
    # Falls from 1.00 at sweet_min=4 toward 0.75 at num_warps=1

elif num_warps > sweet_max:
    Occ_score = 0.75 + 0.25 × exp(−0.5 × ((num_warps − sweet_max) / 4.0)²)
    # Falls from 1.00 at sweet_max=8 toward 0.75 as num_warps → 16+
```

**Derivation of [4, 8]:**

From Little's Law with HBM latency `L = 300 cycles` and issue gap `λ = 40 cycles`:
```
wavefronts_needed = ceil(L / λ) = ceil(300 / 40) = 8
```
This gives the upper end.  The lower end (4) is the minimum where:
1. L2 hits (latency ~30 cycles) can be hidden with `ceil(30/40) = 1–2` wavefronts, so
   a cache-warm kernel works fine at 4.
2. The register budget at num_warps=4 is 256 VGPRs/thread — well above the typical
   kernel's 40–100 VGPR need, so no spill risk.

At `num_warps=1`: `Occ_score = 0.75 + 0.25×exp(−0.5×(3/2)²) = 0.75 + 0.25×0.105 ≈ 0.776`
At `num_warps=16`: `Occ_score = 0.75 + 0.25×exp(−0.5×(8/4)²) = 0.75 + 0.25×0.135 ≈ 0.784`

Both extremes score ~0.78, correctly penalising them but not catastrophically — a kernel
might legitimately need num_warps=16 for a specific high-register use case.

### Regime 2 — Launch-bound, single-block (`saturation < 0.25, num_blocks = 1`)

One block on one CU.  All wavefronts in the block share that CU's memory latency hiding.
The sweet-spot model applies identically to Regime 1, because the physics is the same:
the single block needs 4–8 wavefronts to keep the CU's ALU busy between memory stalls.

### Regime 3 — Launch-bound, multi-block (`saturation < 0.25, num_blocks > 1`)

Multiple blocks each run on their own separate CU.  Latency hiding is provided by the
grid parallelism itself (CU #0 works on block 0 while CU #1 works on block 1), not by
per-block wavefront switching.  Therefore, additional wavefronts within a single block
**only add overhead without benefit**:

```
T(nw) = K_launch + (nw − 1) × K_warp       K_launch=3.0 µs, K_warp=0.2 µs
Occ_score = K_launch / T(nw)
           = 3.0 / (3.0 + (nw − 1) × 0.2)
```

| num_warps | T(nw) | Occ_score |
|---|---|---|
| 1 | 3.0 µs | **1.00** |
| 2 | 3.2 µs | 0.938 |
| 4 | 3.6 µs | 0.833 |
| 8 | 4.4 µs | 0.682 |
| 16 | 6.0 µs | 0.500 |

No tuned coefficients — the formula uses the same hardware constants (`K_launch`,
`K_warp`) as the Stage 3a overhead model, ensuring internal consistency.

**Why `nw=1` is ideal in this regime:**

When each block runs on a separate CU, declaring `num_warps=4` means:
- 4 wavefronts initialised per block (adding `3 × 0.2 = 0.6 µs` overhead per block)
- 4 wavefronts share the CU's resources — but the CU could have been fully occupied
  by *other blocks* from the same or adjacent kernels instead
- Latency hiding from the 3 other wavefronts is marginal when the block is launch-dominated

So declaring extra warps in this regime is pure overhead with near-zero benefit.

### Per-CU wavefront limit check

The hardware imposes a maximum of 32 wavefronts per CU (CDNA2/3):

```
max_wavefronts_per_cu = 32
wavefronts_this_block = num_warps
if num_warps > max_wavefronts_per_cu:
    # Illegal config (shouldn't reach here after Stage 1 pruning)
    return 0.0
```

Stage 1 pruning (Constraint 2: `num_warps × warp_size ≤ XBLOCK × YBLOCK`) typically
eliminates configs with excessive num_warps before they reach scoring.

---

## Simple explanation

### What "occupancy" means

Occupancy = how many wavefronts (groups of 64 threads) are simultaneously "in residence"
on a Compute Unit (CU).

Think of a CU as a multi-window computer.  It can have multiple programs open (wavefronts)
and switches between them instantly.  When one program is frozen waiting for something
(memory load), it switches to another program that's ready to continue.

**Occupancy is the number of open programs.**  More programs = better chance that one is
always ready to run, keeping the processor busy.

### The fundamental trade-off

**More wavefronts (higher occupancy):**
- ✓ More programs to switch between → fewer idle cycles waiting for memory
- ✗ Each wavefront needs registers (VGPRs) to hold its variables.  The 65,536 VGPRs
  on a CU are split equally.  With 16 wavefronts, each thread gets only 64 VGPRs —
  barely enough for simple kernels.

**Fewer wavefronts (lower occupancy):**
- ✗ When the one wavefront stalls waiting for memory, the processor has nothing to do
  → goes dark for ~300 cycles per load
- ✓ More VGPRs per thread → no register spills → clean, fast execution

### The three situations

**Situation 1 — Normal (large or medium kernel):**
You need 4–8 wavefronts to keep the ALU busy.  The Little's Law math says: HBM takes
~300 cycles to respond; a wavefront issues a new load request every ~40 cycles; you need
`300/40 = 8` wavefronts in flight to keep the pipeline full.  4 wavefronts works when
data is partially cached in L2 (lower latency).

Score is 1.00 for num_warps in [4, 8], falls gracefully outside that range.

**Situation 2 — Tiny kernel, one block, one CU:**
Same reasoning as Situation 1 — the single block needs wavefronts to hide memory latency.
Same sweet-spot model applies.

**Situation 3 — Tiny kernel, multiple blocks on multiple CUs:**
When each block runs on a different CU, latency hiding comes from the grid itself (all
CUs work in parallel), not from wavefronts within one block.  In this case, extra
wavefronts per block only add startup overhead without any benefit.

Score = `3.0 µs / (3.0 µs + extra_warp_overhead)`.  The ideal is `num_warps=1` with a
score of 1.0, and higher warp counts score progressively worse:
- `num_warps=1`: score 1.00 (no wasted overhead)
- `num_warps=4`: score 0.83 (0.6 µs wasted for zero benefit)
- `num_warps=16`: score 0.50 (half the block's total time is just starting wavefronts)

### Why 4–8 wavefronts is the "ideal" for normal kernels

This comes directly from the physics of HBM:
- **Too few (< 4):** Every memory load stalls the wavefront for ~300 cycles.  With 1
  wavefront, the ALU is idle 300/40 = 7.5× as much as it's working.  Effectively only
  ~13% efficient.
- **Too many (> 8):** Each thread gets fewer than 128 VGPRs.  Most kernels need 40–80
  VGPRs per thread.  Complex fused kernels need 80–100+.  Once the register budget is
  exceeded, the compiler spills registers to local memory — adding extra load/store
  instructions that ironically *increase* the memory traffic the kernel generates.
- **4–8:** Enough wavefronts to fill the memory pipeline, while keeping register budgets
  comfortable (128–256 VGPRs/thread).


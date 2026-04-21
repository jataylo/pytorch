# Stage 4d — Grid Granularity Score

**Source:** `estimate_grid_granularity()` in `triton_heuristics_pointwise.py`

---

## Building up from basics

### 1. Blocks and CUs

When Triton launches a kernel it creates a **grid of blocks**.  Each block is an
independent chunk of work — it owns its own registers, shared memory, and wavefronts.

The GPU hardware has **Compute Units (CUs)** — the physical execution engines.  The
hardware scheduler (Command Processor) assigns one block per CU slot.  MI300X has
**304 CUs**.

```
Grid of blocks:  [ B0 | B1 | B2 | B3 | ... | Bn ]
                    ↓    ↓    ↓    ↓
GPU CUs:        [ CU0  CU1  CU2  CU3  ... CU303 ]
```

### 2. The mismatch problems

**Too few blocks — CUs starve:**
```
10 blocks, 304 CUs → 294 CUs sit completely idle the entire kernel
→ you're using 3% of the GPU
```

**Too many blocks — scheduling stalls:**

The Command Processor dispatches at most `num_CUs` blocks at a time (one wave).
When wave 1 finishes, it dispatches wave 2.  During the gap between waves (~0.5 µs),
CUs are idle.  With 10,000 blocks on 304 CUs that's ~33 wave-gaps of wasted time.

**The sweet spot:** enough blocks to keep all CUs busy, but not so many that
inter-wave gaps accumulate.

### 3. Why 2 × num_CUs is the target for large problems

With exactly `num_CUs = 304` blocks:

```
Wave 1:  304 blocks start on CU0..CU303
         CU0 finishes first → sits idle waiting for CP to schedule next block
         CP notices, picks block 305 → dispatches → ~0.5 µs delay
```

With `2 × num_CUs = 608` blocks:

```
Wave 1:  304 blocks start on CU0..CU303
         Block 304 is already queued in the CP's ready list
         CU0 finishes → block 304 starts immediately, zero idle gap
```

`optimal_blocks_grid = 2 × num_CUs` is queried directly from device properties:
```python
optimal_blocks_grid = props.multi_processor_count * 2   # = 608 on MI300X
```

### 4. How XBLOCK controls block count

```
num_blocks = ceil(total_elements / XBLOCK)
```

Doubling `XBLOCK` halves `num_blocks`.  So the grid score is really a constraint
on `XBLOCK`: it must be small enough to create enough blocks, but not so small
that blocks proliferate into thousands.

---

## The scoring logic

### Band selection — target scales with problem size

Not every problem can afford 608 blocks.  A 100-element problem split into 608
blocks would give each block fewer than 1 element — the launch overhead would
dwarf the compute.  The target is scaled down for small problems:

| Problem size | Code | MI300X target | Rationale |
|---|---|---|---|
| `N < 2,048` | discrete lookup | see table below | Too tiny for Gaussian — only 1–8 blocks possible anyway |
| `2,048 ≤ N < 16,384` | `hardware_optimal // 32` | **19 blocks** | Partial saturation — fill only ~6% of CUs to keep EPB healthy |
| `16,384 ≤ N < 262,144` | `hardware_optimal // 2` | **304 blocks** | One block per CU — full first wave |
| `N ≥ 262,144` | `hardware_optimal` | **608 blocks** | Full 2-wave target — no inter-wave idle gaps |

> **Note:** `hardware_optimal = 2 × num_CUs = 608` on MI300X, so `// 32 = 19`,
> `// 2 = 304`, and the full value is `608`.  All three targets are derived from
> a single arch constant — they adapt automatically to any GPU.

### Gaussian score — general case

Once the target is known, the score is a Gaussian centred on it:

```
σ_grid = target_blocks × 0.5

Grid_score = clamp(0.70 + 0.30 × exp(−0.5 × ((num_blocks − target) / σ_grid)²),
                   min=0.70, max=1.00)
```

| num_blocks | Score |
|---|---|
| `= target` | 1.00 |
| `= 0.5 × target` or `1.5 × target` | ~0.88 |
| `→ 0` or `→ ∞` | 0.70 (floor) |

**Why floor at 0.70?**  A config with the wrong block count may still have excellent
bandwidth and occupancy.  Flooring at 0.70 (not 0.0) prevents the grid score from
eliminating otherwise good configs — it applies a moderate penalty, not a veto.

**Hard floor for large problems:** if `num_blocks < 4` and `N ≥ 262,144`, the score
is hard-clamped to `0.70` regardless of the Gaussian, because a 1–3 block grid for
millions of elements is severely under-saturated.

### Discrete lookup — tiny kernels (`N < 2,048`)

For tiny problems the Gaussian has no signal — the entire feasible range is only
1–8 blocks.  A hand-tuned table is used instead:

```
num_blocks ≤ 1  →  0.85   (single block: only 1 CU used, but registers uncontested)
num_blocks ≤ 2  →  0.93
num_blocks ≤ 4  →  1.00   ← optimal: 4 independent register budgets, 4× L1 cache
num_blocks ≤ 8  →  0.90
num_blocks  > 8 →  0.70   (over-dispatched for this problem size)
```

**Why does the optimum peak at 4, not 1?**

A single block on one CU must fit all its wavefronts within one CU's 65,536 VGPRs.
With four blocks on four separate CUs, each block gets its own full VGPR budget —
allowing more resident wavefronts per CU for latency hiding, plus 4× L1 cache
capacity.  The extra dispatch overhead for 4 vs 1 block is ~0.01 µs — negligible.

### 3-D kernel override

For 3-D kernels the target is further adjusted based on shape:

- **Cube-like** (`max_dim / min_dim < 2`, e.g. `64×64×64`): regular stride → excellent
  L2 reuse → target is `hardware_optimal // 2 = 304` (fewer, larger blocks exploit reuse)
- **Non-cube** (`max_dim / min_dim ≥ 2`, e.g. `256×128×32`): irregular HBM traffic →
  target is doubled to `min(optimal × 2, hardware_optimal × 4)` (more wavefronts needed
  to hide memory latency from mixed-stride access)

---

## Worked example — MI300X (304 CUs, target = 608)

Problem: `N = 10,000,000` → band `≥ 262,144` → target = 608, σ = 304

```
Config A: XBLOCK = 16,384  →  num_blocks = ceil(10M / 16384) = 611
  diff = (611 − 608) / 304 = 0.010
  Grid_score = 0.70 + 0.30 × exp(−0.00005) ≈ 1.00   ✓

Config B: XBLOCK = 256     →  num_blocks = 39,063
  diff = (39063 − 608) / 304 = 126
  Grid_score → 0.70   (floor — massively over-scheduled)

Config C: XBLOCK = 1,048,576 → num_blocks = 10
  diff = (10 − 608) / 304 = −1.97
  Grid_score = 0.70 + 0.30 × exp(−1.94) ≈ 0.74   (under-scheduled)
```

Config A wins: it hits almost exactly the 2-wave target.

---

## Simple summary

| | |
|---|---|
| **What it measures** | Does this config launch the right number of blocks to keep all CUs busy? |
| **Too few blocks** | Most CUs idle → wasted hardware |
| **Too many blocks** | Scheduling stalls between dispatch waves |
| **Sweet spot** | 2 × num_CUs blocks for large problems; scaled down for small ones |
| **Score range** | 0.70 (wrong) → 1.00 (perfect) |
| **Arch-adaptive?** | Yes — all targets derived from `props.multi_processor_count` |

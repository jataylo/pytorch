# Stage 4d — Grid Granularity Score

**Source:** `score_grid_granularity()` in `triton_heuristics_pointwise.py`

**Role:** Scores whether the number of blocks launched is well-matched to the GPU's
Compute Unit (CU) count.  Too few blocks leave most CUs idle; too many blocks create
scheduling overhead.  The optimal block count scales with problem size.

---

## Technical

### What a "grid" is

A Triton kernel launch produces a **grid** of blocks.  The hardware scheduler assigns
blocks to CUs:

- **MI300X:** 304 CUs.  Each CU can run multiple resident blocks (wavefront pool size ~
  8–16 blocks depending on register usage).
- **MI250X:** 220 CUs.
- **A100:** 108 SMs (equivalent of CUs).

The first wave of up to `num_CUs × max_resident_blocks` blocks starts immediately.  As
CUs complete blocks and become free, the Command Processor dispatches the next wave.

**Under-saturation** (`num_blocks << num_CUs`): most CUs are idle for the entire kernel.
One block running on CU #0 while CUs #1–303 are completely idle means 99.7% of compute
capacity is wasted.

**Over-saturation** (`num_blocks >> num_CUs`): the CP must dispatch in multiple rounds.
After dispatching the first wave, it waits for completion signals, re-evaluates the
queue, and dispatches the next wave.  At very large grids (>100× num_CUs), this
scheduling overhead becomes measurable.

### Problem-size-adaptive target

The optimal block count scales with problem size because small problems cannot afford
many blocks (each block has overhead cost, and small blocks have poor EPB):

| Problem size | Target blocks | Rationale |
|---|---|---|
| `N < 2,048` | `4` | Just 4 CUs; more blocks cost more overhead than they save in parallelism |
| `2,048 ≤ N < 16,384` | `num_CUs / 8 = 38` on MI300X | Partial saturation; don't over-schedule tiny workloads |
| `16,384 ≤ N < 262,144` | `num_CUs = 304` on MI300X | Half the CUs saturated |
| `N ≥ 262,144` | `2 × num_CUs = 608` on MI300X | Full saturation: 2 waves keeps CUs busy while first wave completes |

**Why 2×num_CUs for large problems?**  When the first wave of 304 blocks completes on
their respective CUs, there should already be blocks queued and ready to start without
the CU going idle.  With exactly 1×num_CUs, the CU goes idle for the re-dispatch delay
(~0.5–1 µs) between waves.  With 2×num_CUs, when a CU finishes block 0, block 304 is
already queued and starts immediately.

### Gaussian scoring — general case

For all but the tiny band:

```
target_blocks = (see table above for problem size band)
σ_grid        = target_blocks × 0.5

Grid_score = 0.70 + 0.30 × exp(−0.5 × ((num_blocks − target_blocks) / σ_grid)²)
```

The score is bounded to `[0.70, 1.00]`:
- `num_blocks = target_blocks` → `Grid_score = 1.00`
- `num_blocks = 0.5 × target_blocks` or `1.5 × target_blocks` → score ≈ `0.70 + 0.30×0.61 = 0.88`
- `num_blocks → ∞` or `num_blocks = 0` → `Grid_score → 0.70`

**Why floor at 0.70?** A config with a wrong block count might still be selected if the
other three factors are excellent.  The floor prevents the grid score from annihilating
configs that are merely sub-optimal in this dimension.

### Discrete lookup — tiny kernels (`N < 2,048`)

For tiny problems, the element count is too small for a meaningful Gaussian (the
problem's range of block counts is only 1–16 anyway):

```python
if num_blocks == 4:    score = 1.00   # optimal: 4 CUs, minimal overhead
elif num_blocks == 3:  score = 0.93
elif num_blocks == 2:  score = 0.90
elif num_blocks == 1:  score = 0.85   # single CU: penalised but not catastrophic
elif num_blocks <= 8:  score = 0.80
else:                  score = 0.70   # many tiny blocks: overhead dominated
```

**Why peak at 4, not 1?**

A single block running on one CU serialises all work on a single CU.  AMD CUs have an
internal wavefront multiplexer that can switch between wavefronts each cycle — but all
wavefronts in the block share the same 65,536 VGPRs.  With register pressure, the single
CU may not be able to fit enough wavefronts.

Four blocks on four CUs each have the full 65,536 VGPRs to themselves.  Even for a tiny
2048-element problem, dispatching 4 blocks of 512 elements each to 4 CUs takes negligible
extra overhead (~0.01 µs) and delivers ~2–3% higher throughput by:
1. Allowing 4 independent register pressure budgets
2. Enabling 4× the L1 cache capacity (4 CUs × 32 KB)
3. Hiding inter-wavefront dependencies across CUs

### Worked examples — MI300X (num_CUs=304)

```
Problem: N = 10,000,000 elements  → band: N > 256K → target = 2×304 = 608

Config A: XBLOCK=16,384   → num_blocks = 611   ≈ target
  Grid_score = 0.70 + 0.30 × exp(−0.5 × ((611−608)/304)²)
             ≈ 0.70 + 0.30 × exp(0) ≈ 1.00

Config B: XBLOCK=256      → num_blocks = 39,063 >> target
  Grid_score = 0.70 + 0.30 × exp(−0.5 × ((39063−608)/304)²)
             → exp(...) ≈ 0
             ≈ 0.70 + 0.30 × 0.0 = 0.70   (floor)

Config C: XBLOCK=1,048,576 → num_blocks = 10  << target
  Grid_score = 0.70 + 0.30 × exp(−0.5 × ((10−608)/304)²)
             → very small → ≈ 0.70   (floor)
```

Config A wins because it lands almost exactly at the target block count.

---

## Simple explanation

### The concept: matching work to workers

A GPU is like a factory with 304 identical assembly stations (Compute Units on MI300X).
Each block of work (kernel block) is assigned to one station.

- **Too few blocks:** Most stations sit idle.  If you send only 10 blocks to a 304-station
  factory, 294 stations do nothing — you're using 3% of your capacity.

- **Too many blocks:** The factory manager (Command Processor) can't hand out work fast
  enough.  Instead of sending all 10,000 blocks at once, the manager dispatches the first
  304, waits for them to finish, then dispatches the next 304, and so on.  The stations
  have brief idle periods between waves.

- **Sweet spot:** Send enough blocks to keep all stations busy, but not so many that the
  manager's scheduling overhead becomes a problem.

### Why the sweet spot depends on problem size

For small problems, dispatching many blocks is wasteful — each block has a startup cost.
If you have 1,000 elements and create 1,000 blocks of 1 element each, you spend almost
all your time on startup, not work.

For large problems, you need at least 2× the number of CUs in blocks.  If you have
exactly 304 blocks for 304 CUs, when the first wave of blocks finish, each CU is idle
for a moment while the GPU waits to schedule the next block.  Having 608 blocks means
the second set is already queued — CUs never idle between waves.

### The score table (tiny kernels)

For very small problems (< 2,048 elements), the math is simple enough to use a table:

- **4 blocks (score 1.0):** Spread across 4 CUs — best mix of parallelism and low overhead
- **1 block (score 0.85):** All work on one CU — some register budget saved, but 303 CUs wasted
- **8+ blocks (score 0.70–0.80):** Probably over-dispatched for a tiny problem

### How `XBLOCK` affects the grid

`num_blocks = ceil(N / XBLOCK)`.  Doubling `XBLOCK` halves the number of blocks.

- Want fewer blocks for a tiny problem? → increase `XBLOCK`
- Want more blocks to saturate all CUs for a large problem? → decrease `XBLOCK`

The scoring balances this against the launch overhead score (4c) — because fewer blocks
means better launch amortisation but worse CU saturation, and vice versa.


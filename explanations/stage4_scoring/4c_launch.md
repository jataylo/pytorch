# Stage 4c — Launch Overhead Score

**Source:** `score_launch_overhead()` in `triton_heuristics_pointwise.py`

**Role:** Scores how well each block amortises the fixed per-block dispatch cost
(~3 µs on AMD).  Configs that create many tiny blocks waste the majority of their
runtime on kernel startup rather than useful computation.  A secondary large-grid
penalty captures additional Command Processor scheduling latency.

---

## Technical

### Core equation — Gaussian on elements-per-block (EPB)

```
EPB = total_elements / num_blocks

Launch_score = 0.75 + 0.25 × exp(−0.5 × ((EPB − optimal_EPB) / σ_launch)²)
               floor: max(Launch_score, 0.75) for EPB ≥ 64
                      Launch_score = 0.70     for EPB < 64
```

### What is EPB and why does it matter?

**Elements-per-block (EPB)** measures the ratio of useful work per block to fixed startup
cost per block.  The startup cost is essentially constant (`K_launch ≈ 3 µs`), so the
only way to reduce its fractional overhead is to increase the useful work per block.

```
overhead_fraction ≈ K_launch / (K_launch + compute_time_per_block)
                  = 3 µs / (3 µs + EPB × time_per_element)
```

As EPB increases, overhead_fraction decreases towards zero.  As EPB decreases towards 1
(single-element blocks), overhead_fraction approaches 100%.

### Deriving `optimal_EPB`

The optimal EPB is hardware-derived from the launch amortisation break-even point: the
EPB at which the overhead fraction drops below a target threshold (typically 5%):

```
K_launch / (K_launch + EPB × time_per_element) = 0.05
→ EPB = K_launch × (1/0.05 − 1) / time_per_element
      = 3 µs × 19 / time_per_element
```

For a fast streaming kernel on MI300X where `time_per_element ≈ 30 ps` (30 ps × 1M
elements = 30 µs kernel runtime at peak BW):

```
optimal_EPB ≈ 57 µs / 0.030 ps ≈ 1900 elements/block
```

In practice, `optimal_EPB` is set from empirical tuning data to the value that produces
a 5% overhead fraction for typical problem+device combinations, typically in the range
**512–2048** for AMD MI300X.

### Floor at EPB < 64

When EPB < 64, each block processes fewer than 64 elements — less than one full wavefront's
worth of data.  Two problems compound:

1. **Overhead dominates completely:** A block handling 16 elements at ~30 ps/element
   takes ~0.5 ns of useful work but ~3 µs of startup overhead.  Overhead is 6000× the
   useful work.

2. **Wavefront underutilisation:** `num_warps × warp_size > XBLOCK` is illegal (pruned in
   Stage 1), so with EPB < 64 the config also has `num_warps=1`.  A single wavefront
   cannot pipeline memory requests.

Hard floor: `Launch_score = 0.70` for EPB < 64.

### Large-grid penalty

When `num_blocks > 4 × num_CUs`, the AMD Command Processor (CP) cannot dispatch all
blocks in a single batch.  The CP maintains a finite hardware queue; once it's full, the
CP must wait for currently-running blocks to complete before enqueueing the next batch.
This adds measurable scheduling stall:

```
if num_blocks > 4 × num_CUs:
    grid_excess = log₂(num_blocks / (4 × num_CUs))
    penalty     = 1.0 − 0.01 × grid_excess
    penalty     = max(penalty, 0.88)          # floor: never below 12% penalisation
    Launch_score *= penalty
```

At `num_blocks = 4 × num_CUs` (= 4 × 304 = 1216 blocks on MI300X): `penalty = 1.0`
At `num_blocks = 2 × 4 × num_CUs = 2432 blocks`: `penalty = 0.99` (1% reduction)
At `num_blocks = 16 × num_CUs = 4864 blocks`: `penalty = 0.98` (2% reduction)
At very large grids (`num_blocks = 256 × num_CUs`): `penalty ≈ 0.94` (clamped at 0.88)

The 0.01 × log₂ coefficient is calibrated from empirical measurements of CP scheduling
latency on MI300X for grid sizes ranging from 1× to 256× num_CUs.

**Why logarithmic?** The CP pipeline fills quadratically but the stall time grows
logarithmically with queue depth — once the pipeline is full, additional blocks queue
without increasing the per-block stall significantly.

### How config parameters affect Launch_score

The key relationship is `EPB = total_elements / num_blocks` and
`num_blocks = ceil(N / XBLOCK) × ceil(M / YBLOCK) × …`:

| Config change | Effect on EPB | Effect on Launch_score |
|---|---|---|
| `XBLOCK`: 64 → 256 | ×4 increase | Increases score toward peak |
| `XBLOCK`: 256 → 1024 | ×4 increase | Continues toward or past optimal EPB |
| `YBLOCK`: 1 → 4 (2-D) | ÷4 decrease (more blocks) | Decreases score |
| `num_warps`: 1 → 8 | No effect on EPB directly | Affects T_overhead (see 3a), not EPB |

`num_warps` does not directly affect EPB (same total elements, same num_blocks regardless
of warp count).  However, it affects `T_overhead` in Stage 3a, which changes the
`overhead_frac` and therefore changes the **weight** placed on `Launch_score` in the
composite.  Indirectly: more warps → more overhead → higher `launch_weight` → Launch_score
matters more in the final composite.

### Worked example — 1M elements, MI300X (num_CUs=304)

```
Problem: N = 1,000,000 elements,  optimal_EPB ≈ 1024

Config A: XBLOCK=16,  num_blocks = 1M/16 = 62,500
  EPB = 16  → floor → Launch_score = 0.70
  grid_excess = log₂(62500 / (4×304)) = log₂(51.4) ≈ 5.68
  penalty = 1 − 0.01×5.68 = 0.943
  Final Launch_score = 0.70 × 0.943 = 0.660

Config B: XBLOCK=256, num_blocks = 1M/256 = 3,906
  EPB = 256
  base = 0.75 + 0.25 × exp(−0.5 × ((256−1024)/512)²)
       = 0.75 + 0.25 × exp(−0.5 × 2.25)  = 0.75 + 0.25 × 0.325  ≈ 0.831
  grid_excess = log₂(3906 / 1216) ≈ log₂(3.21) ≈ 1.68
  penalty = 1 − 0.01×1.68 = 0.983
  Final Launch_score = 0.831 × 0.983 ≈ 0.817

Config C: XBLOCK=1024, num_blocks = 1M/1024 = 977
  EPB = 1024 ≈ optimal_EPB
  base = 0.75 + 0.25 × exp(0) = 0.75 + 0.25 = 1.00
  grid_excess = 0 (977 < 4 × 304 = 1216)
  penalty = 1.0
  Final Launch_score = 1.00
```

Config C correctly wins — it creates exactly enough blocks to fill ~3 CUs per CU, with
full cache-line alignment and near-zero grid overhead.

---

## Simple explanation

### What this score measures

Every time the GPU starts executing a block, it pays a fixed startup cost of about 3
microseconds (µs) — the time for the GPU's scheduler to allocate registers and hand off
the work.

The launch score asks: **does this block do enough useful work to justify that 3 µs
startup cost?**

Think of it like a delivery truck: if each trip delivers only 1 package, most of the time
is spent driving and unloading — the useful "payload fraction" is tiny.  If each trip
delivers 1000 packages, the driving overhead is shared across all of them.

**Elements-per-block (EPB)** = total elements ÷ number of blocks.  It measures how many
"packages" each block delivers.

### When the score is high

A high launch score means each block processes a lot of elements, so the 3 µs startup
is a small fraction of the block's total runtime.

Example: `XBLOCK=1024` on a 1M-element problem → 977 blocks, each processing 1024
elements.  At peak bandwidth, 1024 elements takes about 10 µs to process.  The 3 µs
startup is 30% of total time — acceptable.

### When the score is low

A low launch score means blocks are tiny and the startup cost dominates.

Example: `XBLOCK=16` on a 1M-element problem → 62,500 blocks, each processing 16
elements.  At peak bandwidth, 16 elements takes ~0.1 µs.  The 3 µs startup is 97% of
each block's total time — catastrophically wasteful.

### The large-grid penalty

Even when individual blocks are well-sized, having an enormous total number of blocks
adds a separate cost: the GPU's Command Processor must dispatch blocks in multiple
batches once its queue fills up.  This adds scheduling latency.  The penalty is small
(1–12% reduction) but measurable for pathological cases like millions of tiny blocks.

### Config guidance

- **Bigger `XBLOCK`** → fewer blocks → higher EPB → higher launch score ✓
- **More dimensions (`YBLOCK`)** → more blocks in the grid → lower EPB → lower launch score ✗
- **`num_warps`** → doesn't affect EPB directly, but more warps increase Stage 3 overhead
  fractions, which increases the *weight* placed on this score


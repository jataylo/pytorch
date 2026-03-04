# Stage 4b — Bandwidth Score

**Source:** `score_bandwidth_utilisation()` in `triton_heuristics_pointwise.py`

**Role:** Estimates how efficiently this config uses the HBM memory bus.  Thread count
determines how many simultaneous memory requests are in flight; 2-D coalescing determines
how efficiently each 64-byte cache-line fetch is utilised.

---

## Technical

### What "threads" means here — per block, not total GPU

`threads` in the formula is **threads per block** (`num_warps × warp_size`), **not** the
total number of threads running across the whole GPU.

The GPU executes many blocks in parallel — on MI300X there are 256 CUs and each CU runs
one or more blocks simultaneously.  A kernel with 512 total blocks and 1024 threads/block
has ~512,000 threads in flight across the GPU, but the bandwidth score only cares about
the **per-block thread count** because:

- Each block runs on exactly **one CU** and uses that CU's local SIMD pipelines.
- The HBM pipeline that block must keep busy is the *per-CU* memory pipeline — not the
  aggregate across all 256 CUs.
- The memory system is distributed: MI300X has 8 independent HBM stacks with independent
  channels.  Blocks on different CUs issue independent requests in parallel; they do not
  compete for a single shared pipeline in the way threads *within* a block do.

So `threads = num_warps × warp_size` is asking: *does this one block issue enough
concurrent memory requests to keep its CU's slice of the memory pipeline busy?*

On AMD CDNA `warp_size = 64`.  Typical candidate configs are `num_warps ∈ {1,2,4,8,16}`,
giving `threads ∈ {64, 128, 256, 512, 1024}`.

---

### Base score — Gaussian on thread count

```
threads_per_block = num_warps × warp_size

BW_score = 0.75 + 0.25 × exp(−0.5 × ((threads_per_block − optimal_threads) / σ_bw)²)

where:
  optimal_threads = arch.optimal_threads_bandwidth   (hardware-derived; ≈1536 on MI300X)
  σ_bw            = optimal_threads × 1.5            (≈2304 on MI300X)
  floor:  BW_score = max(BW_score, 0.75)   for threads_per_block ≥ 64
          BW_score = 0.60                   for threads_per_block < 64
```

#### Term-by-term explanation

**`threads_per_block = num_warps × warp_size`**

- `warp_size` — the SIMD width of one wavefront: 64 on AMD CDNA, 32 on NVIDIA.
  Read directly from `torch.cuda.get_device_properties(device).warp_size`.
- `num_warps` — the config parameter controlling how many wavefronts are in this block.
  Candidate values are typically {1, 2, 4, 8, 16}.
- Together these give the **actual hardware thread count active in one block at once**.
  This is the per-CU concurrency number — not total GPU threads (see the section above).

**`0.75 + 0.25 × Gaussian`**

The formula is a raised Gaussian, not a pure Gaussian.  The `0.75` floor ensures that
even a config far from optimal (e.g. `num_warps=1`) still scores 0.75 rather than 0.0
— because even one wavefront issuing loads is not *zero* bandwidth.  The `0.25` headroom
is the reward for getting close to the optimal thread count.  This keeps the score in
`[0.75, 1.00]` for valid configs (≥ 64 threads), with a hard floor of 0.60 for
sub-wavefront configs.

**`optimal_threads` — hardware-derived, not hardcoded**

The equation is:

```
optimal_threads = sqrt(simd_units × effective_latency / instructions_per_load) × warp_size

                = sqrt(         4 ×              275 /                       2) × 64

                = sqrt(550) × 64  ≈  24 × 64  =  1536   (on MI300X)
```

**What this equation is saying:**

Adding more wavefronts to a block does two opposing things simultaneously:
1. It gives the CU more work to overlap with outstanding loads (good for latency hiding)
2. It lengthens each wavefront's issue gap — each one issues instructions less frequently
   — which means more compute work happens between consecutive loads from the same wavefront
   (which *reduces* how many wavefronts are needed to hide that latency)

These two effects push in opposite directions as N grows.  The `sqrt` is the fixed point
where they balance: the number of wavefronts you *have* equals the number you *need*.

Full derivation of where the equation comes from is in the next section.  The inputs and
where each comes from:

*Queried live from the driver* (`torch.cuda.get_device_properties(device)`):

| Property | Attribute | MI300X | Role |
|---|---|---|---|
| Warp / wavefront size | `props.warp_size` | 64 | Converts wavefront count → thread count at the end |
| Max threads per CU | `props.max_threads_per_multi_processor` | 2048 | Hard ceiling: `max_wavefronts = 2048/64 = 32`; result is clamped to this |

*Architecture-spec or engineering estimates* (not queryable at runtime):

| Constant | Value | Source | Sensitivity |
|---|---|---|---|
| `simd_units` | 4 | AMD CDNA ISA: 4 SIMD-32 units per CU | High — appears under sqrt; halving it halves N |
| `l2_hit_latency` | 50 cycles | CDNA3 arch guide (~25 ns at 2 GHz) | Low — blended with HBM latency |
| `hbm_latency` | 500 cycles | MI300X HBM3 round-trip (~100–200 ns at 2 GHz) | Medium — dominates effective_latency at 50% hit rate |
| `l2_hit_rate` | 0.50 | Conservative; streaming kernels are lower | Medium — shifts effective_latency between 50 and 500 |
| `instructions_per_load` | 2 | Typical pointwise: load → compute → store | Medium — appears under sqrt; doubling it reduces N by √2 |

`effective_latency` is the L2-blended average latency:
```
effective_latency = l2_hit_rate × l2_hit_latency + (1 − l2_hit_rate) × hbm_latency
                  = 0.50 × 50  +  0.50 × 500
                  = 275 cycles
```

The full computation chain:
```
N_min             = sqrt(4 × 275 / 2)        = sqrt(550) ≈ 23.4  →  ceil = 24
wavefronts_needed = max(8, min(24, 32))       = 24    ← clamped to hardware limits
optimal_threads   = 24 × warp_size            = 24 × 64 = 1536
```

`warp_size` is the only queried value that feeds directly into the final number.
Everything else is fixed per architecture — so `optimal_threads` automatically adapts
to different GPU generations through `warp_size` and the hardware ceiling, but the
latency/simd constants would need updating for a fundamentally different memory
subsystem.

**`σ_bw = 1.5 × optimal_threads`**

The width of the Gaussian controls how steeply score falls as you move away from optimal.
A narrow sigma (1×) caused a ≈6% score gap between `num_warps=4` (256 threads) and
`num_warps=16` (1024 threads), which systematically flooded the top-N pool with
`XBLOCK=1024` variants even when a smaller XBLOCK gave better grid coverage.  The
wider 1.5× sigma shrinks that gap to ~3%, allowing the Grid score (Stage 4d) to
make the block-count decision without the bandwidth score pre-biasing it.

**Floor 0.75 (threads ≥ 64) and hard floor 0.60 (threads < 64)**

A sub-wavefront config (`threads < warp_size = 64`) means the SIMD unit is never fully
utilised — half the lanes are idle on every instruction.  That warrants a harsher 0.60
floor rather than the 0.75 of a valid-but-thin wavefront.

---

### How optimal_threads is derived — self-consistent Little's Law

There are **two separate** Little's Law analyses in this codebase addressing different
questions.  It is important not to confuse them:

| Analysis | Question answered | MI300X result |
|---|---|---|
| **Bandwidth optimal** (here) | How many wavefronts per block to keep the CU's SIMD units continuously busy? | 24 wf → **1536 threads** |
| **Occupancy sweetspot** (Stage 4e) | How many wavefronts so the CU scheduler can hide stalls by switching between them? | 8 wf → **512 threads** |

The bandwidth question is about *throughput* (never leaving an issue slot empty);
the occupancy question is about *latency hiding* (always having a ready wavefront to
switch to while another waits for data).  These have different answers because the
bandwidth calculation must account for the fact that the issue rate *depends on N*.

#### Full derivation — the self-consistent fixed point

**Step 1 — issue gap**

Each CU has `simd_units = 4` SIMD units, each issuing one wavefront-instruction per
cycle.  With N wavefronts sharing 4 issue slots, the hardware round-robins them.
Each wavefront waits its turn, so the gap between consecutive instructions from the
**same wavefront** is:

```
issue_gap(N) = N / simd_units = N / 4   cycles
```

This is N-dependent: more wavefronts → each individual wavefront issues less often.

**Step 2 — compute gap between loads**

A pointwise kernel does `I` arithmetic instructions between consecutive memory loads
(a load, some compute, then the next load).  The time between two loads from the same
wavefront is:

```
compute_gap(N) = I × issue_gap(N) = I × N / simd_units   cycles
```

**Step 3 — how many wavefronts are needed to hide the latency?**

To avoid stalling, the previous load must have returned before the next one is issued.
Given a compute gap of `compute_gap(N)` cycles, the number of wavefronts the hardware
needs to keep itself busy during one latency period is:

```
N_needed(N) = effective_latency / compute_gap(N)
            = effective_latency / (I × N / simd_units)
            = simd_units × effective_latency / (I × N)
```

This is also N-dependent: more wavefronts → longer compute gap → fewer additional
wavefronts needed to absorb the latency.

**Step 4 — the self-consistent fixed point**

We want to find the N where the number of wavefronts we *have* equals the number we
*need*:

```
N = N_needed(N)
N = simd_units × effective_latency / (I × N)
N² = simd_units × effective_latency / I
N  = sqrt(simd_units × effective_latency / I)
```

This is the equilibrium point.  Below it: `N_needed > N` — the wavefronts stall
waiting for data.  Above it: `N_needed < N` — more than enough, but VGPR pressure
rises.

**Why this is circular (and why it gives sqrt, not a linear answer):**

The naive approach would set `compute_gap ≥ effective_latency` and solve for N
directly, giving `N ≥ simd_units × effective_latency / I = 550 wavefronts` — a
physically impossible number for a single block.  The reason that's wrong is that
it ignores the feedback: more wavefronts lengthen the compute gap, which reduces
the latency-hiding requirement.  The fixed point captures both sides of that
feedback simultaneously, which is why it produces a square root and a sensible
answer (~24 wavefronts) instead of 550.

**The constants and where they come from:**

```
simd_units = 4
  → AMD CDNA ISA spec: each CU contains exactly 4 SIMD-32 units (each handles half a
    wave64 per cycle, so effectively 4 independent issue slots for a full wave64)
  → Not queryable at runtime; hardcoded from architecture documentation

l2_hit_latency = 50 cycles
  → CDNA3 architecture estimate (~25 ns at 2 GHz)
  → Tier 3 estimate; validated broadly against AMD profiling tools

hbm_latency = 500 cycles
  → MI300X HBM3 round-trip; ~100–200 ns at 2 GHz depending on bank/channel state
  → Tier 3 estimate; conservative (can be 400–600 in practice)

l2_hit_rate = 0.50
  → Conservative assumption for streaming pointwise kernels
  → Real kernels can be 10–20% (pure streaming) to 80%+ (small tensor reuse)
  → 50% gives a safe middle estimate; the score is not very sensitive to this

instructions_per_load = 2
  → Typical for pointwise: load → compute → store = 2 arithmetic ops per load
  → Hardcoded assumption; does not vary per-kernel

effective_latency = l2_hit_rate × l2_hit_latency + (1 − l2_hit_rate) × hbm_latency
                  = 0.50 × 50 + 0.50 × 500
                  = 275 cycles

N_min = sqrt(simd_units × effective_latency / instructions_per_load)
      = sqrt(4 × 275 / 2)
      = sqrt(550)
      ≈ 23.4  →  ceil = 24

optimal_threads = 24 × warp_size
```

`warp_size` is queried from `props.warp_size` — 64 on AMD, 32 on NVIDIA — so
`optimal_threads` is automatically correct for both architectures:

```
MI300X (AMD):  24 × 64  = 1536 threads
A100   (NVIDIA): 24 × 32 = 768 threads  (hbm_latency and simd_units may differ)
```

The result is clamped: `max(8, min(ceil(N_min), max_wavefronts_per_cu))` where
`max_wavefronts_per_cu = props.max_threads_per_multi_processor / props.warp_size`
is queried directly from the device.

**Why not the simpler 300/40 = 8 wavefronts formula?**

The simpler formula (`ceil(latency / issue_gap) = 8`) treats the issue gap as a fixed
40-cycle constant.  That 40-cycle gap is appropriate for a *single* wavefront running
alone on the CU (no sharing).  The self-consistent formula accounts for the fact that
once you have N wavefronts sharing 4 SIMD units, the effective gap is N/4 — much shorter
— so you need more wavefronts than the naive estimate suggests.  At N=8, the gap is
8/4 = 2 cycles, far too short to provide 275 cycles of compute hiding.  N=24 gives
24/4 = 6 cycles per issue slot, and with I=2 arithmetic ops that's 12 cycles compute
between loads — still short of 275, which is why ILP (see below) and L2 hits are
doing most of the real work.

---

### ILP correction (Stage 4e interaction)

The BW score measures wavefront concurrency — but there is a second mechanism for hiding
memory latency: **Instruction-Level Parallelism (ILP)** within a single thread.

When `EPT = XBLOCK / threads_per_block ≥ 8`, each thread processes 8+ elements
in a loop.  The compiler (Triton's LLVM backend) can software-pipeline this loop:

```
Iteration 0:  issue load[0]         ← in-flight
Iteration 1:  issue load[1]         ← in-flight simultaneously
…
Iteration 7:  issue load[7]
              compute[0]            ← load[0] has returned by now
              compute[1]
              store[0]
```

Eight loads are in-flight simultaneously from a *single* thread without needing a second
wavefront.  This provides per-thread latency hiding that partially substitutes for the
inter-thread wavefront switching that the BW score measures.

**Where the ILP correction lives:** Stage 4e (`estimate_occupancy_impact`), not the BW
score itself.  The BW score always penalises low `num_warps`.  The occupancy score then
*credits back* that penalty when ILP is available and CU utilisation is adequate:

```
if not need_intra_block_warps:          # enough block-level wavefront switching
    if elems_per_thread >= 8:
        occupancy_score = 1.00          # full credit: CU-level + ILP covers everything
    elif elems_per_thread >= 4:
        occupancy_score = 0.93          # partial credit
    else:
        occupancy_score = 0.82          # no compensation: CU starved
```

**The CU utilisation gate:**

ILP only helps *per-block*.  If very few blocks are active (`num_blocks << num_cus`),
the majority of CUs are completely idle — ILP inside the active blocks does nothing for
those idle CUs.  The correction is therefore scaled by CU utilisation:

```
cu_util   = min(1.0, num_blocks / num_cus)
ilp_scale = min(1.0, cu_util / 0.25)     # full benefit at ≥ 25% CU utilisation

if elems_per_thread >= 8:
    ilp_ceiling = 0.766 + 0.234 × ilp_scale   # 0.766 (no CUs) → 1.000 (≥25% CUs)
    occupancy_score = max(base_score, ilp_ceiling)
```

**Net effect on the composite score:**

A config with `num_warps=1, EPT=16, many blocks` gets:
- BW score = 0.75 (floor — only 1 wavefront)
- Occupancy score = 1.00 (ILP + CU-level switching fully compensates)
- The composite geometric mean reflects the genuine tradeoff — neither fully penalised
  nor fully rewarded

A config with `num_warps=1, EPT=16, few blocks` gets:
- BW score = 0.75 (floor)
- Occupancy score = 0.82 (ILP doesn't compensate for CU idleness)
- Composite is genuinely lower — correctly reflecting that leaving 80% of CUs idle is
  a real performance loss regardless of per-thread pipelining

---

#### Intuition — what ILP is actually doing and where `num_stages` fits

The BW score penalises `num_warps=1` because a single wavefront normally stalls:
it issues a load, waits ~275 cycles for the data, and the CU sits idle the whole time.
The fix the hardware expects is more wavefronts — while one waits for its load, another
runs compute.

ILP is an alternative fix done entirely in software.  When a thread processes multiple
elements in a loop, the compiler can issue the *next* iteration's load before finishing
the *current* iteration's compute:

```
iter 0: issue load[0]
iter 1: issue load[1]          ← load[0] still in flight, but we don't need it yet
iter 2: issue load[2]
...
iter 0: compute(load[0])       ← data finally arrives; we've moved on to load[2] already
```

Instead of stalling for 275 cycles, the thread stays busy doing compute on earlier
iterations while later iterations' loads are in flight.  The single wavefront is
effectively hiding its own latency — no second wavefront required.

**How many iterations need to be in-flight to fully hide latency?**

```
cycles hidden = pipeline_depth × compute_time_per_iter
pipeline_depth = num_stages × elems_per_thread

At full hiding:  pipeline_depth × compute_time ≥ effective_latency (275 cycles)
```

`elems_per_thread` (EPT) is how many elements each thread processes.  `num_stages` is
how many of those iterations the Triton compiler actually keeps in-flight simultaneously
— the prefetch depth.  Both multiply together to give the total latency-hiding capacity.

**Current state — why EPT is used instead of `num_stages`:**

On AMD (HIP + Triton > 3.2), `num_stages=2` is injected into every pointwise element
loop at codegen time — it is not a tunable config parameter, so it is the same for
every candidate config.  Since it never varies, it adds no signal to scoring.  EPT
*does* vary (XBLOCK=1024 with num_warps=1 gives EPT=16; XBLOCK=64 with num_warps=1
gives EPT=1), so EPT is used as a proxy for the combined pipeline depth.

If `num_stages` were ever a tunable parameter for pointwise kernels, the threshold
should become:

```python
pipeline_depth = config.get('num_stages', 1) * elems_per_thread
# full hiding: pipeline_depth >= 8  (≈275 cycles / ~35 cycles per iter)
if pipeline_depth >= 8:
    occupancy_score = 1.00
```

**The CU-utilisation gate — why ILP alone is not enough:**

ILP hides latency *inside* one block.  It does nothing for CUs that have no block
assigned to them at all.  If only 64 out of 256 CUs have work (few blocks), the other
192 are completely dark — no amount of per-thread pipelining changes that.  The
`ilp_scale = cu_util / 0.25` gate zeroes out the ILP credit when CU utilisation is
low, correctly reflecting that the bottleneck has shifted from memory-latency to
CU-idleness.

### Hard floor at `threads < 64` (< 1 wavefront)

```
if threads < warp_size:   # < 64 on AMD
    BW_score = 0.60
```

A sub-wavefront thread count means the kernel can never issue a full SIMD width of
loads simultaneously.  The memory pipeline receives sparse, fragmented requests that
the HBM controller cannot burst-combine efficiently.  Score hard-floored at 0.60 to
reflect this fundamental inability to saturate the bus.

### 2-D coalescing correction

For 2-D kernels (`YBLOCK > 0`), `XBLOCK` controls how wide each tile row is in the
contiguous (row-major) memory layout.

The GPU L2 cache and HBM controller issue **64-byte cache line** fetches.  At FP32
(4 bytes/element), one cache line holds 16 elements.  If `XBLOCK < 16`, each tile
row does not fill a full cache line:

```
XBLOCK=4:  4 elements × 4 bytes = 16 bytes used, 48 bytes fetched but discarded
           coalescing efficiency = 4/16 = 0.25 → 75% of HBM fetch wasted
```

The correction formula:
```
coalescing  = min(1.0, XBLOCK / 16)
BW_score   *= (0.65 + 0.35 × coalescing)
```

| XBLOCK | coalescing | multiplier | Combined effect |
|---|---|---|---|
| 4 | 0.25 | ×0.739 | 74% of base BW_score |
| 8 | 0.50 | ×0.825 | 83% of base BW_score |
| 16 | 1.00 | ×1.000 | Full base BW_score |
| 32+ | 1.00 | ×1.000 | Full base BW_score |

**Why the floor at 0.65?** Even `XBLOCK=4` doesn't make bandwidth zero — the cache line
is fetched once and the entire 16-element row will be consumed by 4 consecutive blocks.
The 65% floor reflects the amortised efficiency when adjacent blocks reuse each other's
cache lines (temporal locality within the L2).

### Full formula — 2-D case

```
threads       = num_warps × warp_size           # threads PER BLOCK (not total GPU threads)
optimal       = arch.optimal_threads_bandwidth  # e.g. 1536 on MI300X
sigma         = optimal × 1.5                   # e.g. 2304 on MI300X

base_score    = 0.75 + 0.25 × exp(−0.5 × ((threads − optimal) / sigma)²)
base_score    = max(base_score, 0.75)           # floor for threads ≥ 64
base_score    = 0.60 if threads < 64            # hard floor for sub-wavefront

coalescing    = min(1.0, XBLOCK / 16)           # only in 2-D
BW_score      = base_score × (0.65 + 0.35 × coalescing)   # 2-D only
BW_score      = base_score                      # 1-D: no correction needed
```

### Effect of config parameters on BW_score

Scores below are for MI300X (`optimal=1536`, `sigma=2304`):

| Config change | threads/block | BW_score | Reason |
|---|---|---|---|
| `num_warps=1` | 64 | 0.75 | Hard floor — only 1 wavefront, pipeline mostly idle |
| `num_warps=4` | 256 | 0.78 | Gaussian tail, still far below 1536 optimal |
| `num_warps=8` | 512 | 0.83 | Closer; still ~3 wavefronts short of optimal |
| `num_warps=16` | 1024 | 0.92 | Near optimal; Gaussian peak region |
| `num_warps=24` | 1536 | 1.00 | Peak — exactly 24 wavefronts per block |
| `XBLOCK`: 4 → 16 (2-D) | — | ×0.74 → ×1.00 | Coalescing correction improves as row fills cache line |

Note: `num_warps=16` (1024 threads) scores 0.92, not 0.89 as the simpler 512-optimal
model predicts.  The Gaussian is wider (sigma=2304 vs old 256), so configs between
1024 and 1536 threads are only mildly penalised rather than heavily.

---

## Simple explanation

### What "bandwidth score" measures

The HBM memory bus is like a motorway with a speed limit.  To achieve peak speed on a
motorway, you need enough cars (memory requests) on the road simultaneously.  Too few
cars and the road is underutilised.  Too many and they start to interfere with each other.

The bandwidth score measures: **how well does this config keep the memory bus busy?**

### What "threads" means here

`threads` is **threads per block** — not the total number of threads running across the
whole GPU.  If a kernel has 512 blocks and 1024 threads/block, the GPU is running
~512,000 threads total, but the bandwidth score only cares about the 1024 per block.

Why?  Each block runs on one CU.  The question this score answers is: *does this one
block issue enough concurrent memory requests to keep that CU's slice of the memory
pipeline busy?*  Blocks on other CUs run independently and don't compete for the same
pipeline.

### The optimal thread count (1536 on MI300X)

Think of each CU's memory pipeline as a factory assembly line.  The line takes 275 "time
units" to fulfil an order (effective latency — a mix of fast L2 hits and slow HBM misses).
The factory has 4 production stages (SIMD units) and each wavefront needs 2 compute
steps between placing consecutive orders (instructions_per_load = 2).

To keep all 4 stages busy continuously, the factory needs:

```
N = sqrt(stages × latency / compute_steps) = sqrt(4 × 275 / 2) ≈ 24 wavefronts in flight
```

At 64 threads per wavefront that is `24 × 64 = 1536 threads/block` — the peak of the
Gaussian.  Below that the pipeline has idle cycles; above that you start competing for
the fixed VGPR budget.

### How config parameters affect the score (MI300X)

- **`num_warps=1` (64 threads):** 1 wavefront issuing loads; pipeline almost always idle
  waiting for data.  Score = 0.75 (hard floor).

- **`num_warps=16` (1024 threads):** 16 wavefronts — close to optimal but ~33% short of
  1536.  Still well into the wide Gaussian.  Score ≈ 0.92.

- **`num_warps=24` (1536 threads):** Exactly 24 wavefronts — all 4 SIMD units always have
  an active wavefront ready to issue.  Score = 1.00.

The wide sigma (2304 = 1.5 × 1536) keeps `num_warps=16` scoring 0.92 rather than
being heavily penalised — allowing the Grid score to pick the right XBLOCK without the
bandwidth score pre-biasing toward the largest possible block.

### The 2-D coalescing penalty

In 2-D kernels, the GPU fetches memory in 64-byte chunks (cache lines) = 16 FP32
elements along the X axis.  If your X tile is only 4 elements wide (`XBLOCK=4`), you
fetch 64 bytes but only use 16 of them — 75% waste.

Widening the X tile to 16+ elements means each cache-line fetch is fully utilised.  The
coalescing penalty multiplies the base bandwidth score:
- `XBLOCK=4` → ×0.74 (26% penalty for poor coalescing)
- `XBLOCK=16+` → ×1.00 (no penalty — full cache-line fills)


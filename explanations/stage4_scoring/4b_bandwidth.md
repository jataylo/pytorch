# Stage 4b — Bandwidth Score

**Source:** `score_bandwidth_utilisation()` in `triton_heuristics_pointwise.py`

**Role:** Estimates how efficiently this config uses the HBM memory bus.  Thread count
determines how many simultaneous memory requests are in flight; 2-D coalescing determines
how efficiently each 64-byte cache-line fetch is utilised.

---

## Technical

### Base score — Gaussian on thread count

```
threads = num_warps × warp_size

BW_score = 0.75 + 0.25 × exp(−0.5 × ((threads − optimal_threads) / σ_bw)²)

where:
  optimal_threads = 512    (8 wavefronts × 64 threads/wavefront on AMD)
  σ_bw            = 256    (half-width at ~0.61 score)
  floor:  BW_score = max(BW_score, 0.75)   for threads ≥ 64
          BW_score = 0.60                   for threads < 64
```

### Why 512 threads is optimal — Little's Law

The optimal thread count is derived from **Little's Law** applied to the HBM memory
subsystem:

```
L = λ × W

where:
  L = number of outstanding requests in the system (concurrency)
  λ = request arrival rate
  W = average time to service one request (latency)
```

For HBM on AMD CDNA hardware:
- HBM round-trip latency `W ≈ 300 cycles` at typical clock (~2 GHz → ~150 ns)
- Wavefront instruction issue gap `λ_issue ≈ 40 cycles` (a wavefront issues one load
  every ~40 cycles due to instruction-level dependencies and the scoreboard)
- Required concurrent outstanding requests: `L = W / λ_issue = 300 / 40 = 7.5 → 8`

Each wavefront can have at most 1–2 outstanding HBM requests in flight (limited by the
per-wavefront scoreboard size).  To maintain 8 concurrent requests, we need ≥ 8 wavefronts
simultaneously resident on the CU:

```
wavefronts_needed = ceil(300 / 40) = 8
threads_needed    = 8 × warp_size  = 8 × 64 = 512
```

At exactly 512 threads, the memory pipeline is theoretically fully saturated — every
cycle, some wavefront is returning with data while others are issuing new loads.

**Below 512 threads:** Fewer outstanding requests → HBM pipeline not full → memory
bus underutilised → effective bandwidth below peak.

**Above 512 threads:** More wavefronts compete for the fixed 65,536 VGPRs per CU.
Each thread gets fewer VGPRs → register pressure increases → compiler likely to spill
more registers to local memory → additional memory traffic → net bandwidth decreases.

The Gaussian shape reflects the smooth degradation in both directions.

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
threads       = num_warps × warp_size
base_score    = 0.75 + 0.25 × exp(−0.5 × ((threads − 512) / 256)²)
base_score    = max(base_score, 0.75)           # floor for threads ≥ 64
base_score    = 0.60 if threads < 64            # hard floor for sub-wavefront

coalescing    = min(1.0, XBLOCK / 16)           # only in 2-D
BW_score      = base_score × (0.65 + 0.35 × coalescing)   # 2-D only
BW_score      = base_score                      # 1-D: no correction needed
```

### Effect of config parameters on BW_score

| Config change | Effect on BW_score | Reason |
|---|---|---|
| `num_warps`: 1 → 4 | 0.76 → 0.98 | Threads 64→256: moving toward optimal 512 |
| `num_warps`: 4 → 8 | 0.98 → 1.00 | Threads 256→512: reaches optimal |
| `num_warps`: 8 → 16 | 1.00 → 0.89 | Threads 512→1024: past optimal, register pressure rises |
| `XBLOCK`: 4 → 16 (2-D) | ×0.74 → ×1.00 | Coalescing correction improves as row fills cache line |

---

## Simple explanation

### What "bandwidth score" measures

The HBM memory bus is like a motorway with a speed limit.  To achieve peak speed on a
motorway, you need enough cars (memory requests) on the road simultaneously.  Too few
cars and the road is underutilised.  Too many and they start to interfere with each other.

The bandwidth score measures: **how well does this config keep the memory bus busy?**

### The optimal thread count (512 on AMD)

Imagine a restaurant (the GPU's memory system) that takes 10 minutes to prepare and
deliver each dish (memory latency ≈ 300 cycles).  The kitchen can start preparing a
new dish every 2 minutes (issue interval ≈ 40 cycles).

To keep the kitchen running at full capacity, you need `10 / 2 = 5` orders in the queue
at all times.  With fewer orders, the kitchen sits idle between dishes.  With many more,
orders start piling up and the system gets confused.

On AMD, each "order" is a wavefront issuing a memory load.  You need 8 wavefronts
in flight simultaneously to saturate the HBM pipeline.  At 64 threads per wavefront,
that's 512 threads.  This is why the bandwidth score peaks at 512 threads.

### How config parameters affect the score

- **`num_warps=1` (64 threads):** Only 1 wavefront can be loading at a time.  While it
  waits 300 cycles for its data, the memory bus sits idle.  Score ≈ 0.76.

- **`num_warps=8` (512 threads):** 8 wavefronts all loading simultaneously.  The bus
  stays continuously busy.  Score ≈ 1.00.

- **`num_warps=16` (1024 threads):** 16 wavefronts compete for 65,536 VGPRs — each thread
  gets only 64 VGPRs.  Complex kernels start spilling registers, adding extra memory
  traffic that *hurts* bandwidth.  Score falls to ≈ 0.89.

### The 2-D coalescing penalty

In 2-D kernels, the GPU fetches memory in 64-byte chunks (cache lines) = 16 FP32
elements along the X axis.  If your X tile is only 4 elements wide (`XBLOCK=4`), you
fetch 64 bytes but only use 16 of them — 75% waste.

Widening the X tile to 16+ elements means each cache-line fetch is fully utilised.  The
coalescing penalty multiplies the base bandwidth score:
- `XBLOCK=4` → ×0.74 (26% penalty for poor coalescing)
- `XBLOCK=16+` → ×1.00 (no penalty — full cache-line fills)


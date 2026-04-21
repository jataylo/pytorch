# Stage 4f — 2-D Tile Tie-breaker

**Source:** `score_tile_shape()` in `triton_heuristics_pointwise.py`, applied as a
secondary multiplier inside `_score_one_config()` for 2-D configs only.

**Role:** For 2-D configs (`YBLOCK > 0`) that achieve similar composite scores from the
four primary factors, a secondary shape multiplier breaks ties by rewarding:
1. **Square-ish aspect ratios** (aligned with hardware prefetcher expectations)
2. **Larger `XBLOCK`** (contiguous-axis width for better cache-line utilisation)

This multiplier is applied *after* the primary composite score and does not alter the
primary factor weights.

---

## Technical

### When the tie-breaker applies

The tie-breaker is applied to all 2-D configs (those with `YBLOCK > 0`).  It is not
a binary tie-breaking rule — it is a small continuous multiplier that shifts the ranking
of similarly-scored configs.  The magnitude is intentionally small (maximum ±3%) to avoid
overriding the primary scoring signals.

### Factor 1 — Aspect ratio penalty

```
aspect_ratio = max(XBLOCK, YBLOCK) / min(XBLOCK, YBLOCK)
AR_score     = 1.0 − 0.005 × log₂(aspect_ratio)
```

| XBLOCK | YBLOCK | aspect_ratio | log₂(ratio) | AR_score |
|---|---|---|---|---|
| 64 | 64 | 1.0 | 0.0 | 1.000 |
| 64 | 32 | 2.0 | 1.0 | 0.995 |
| 128 | 16 | 8.0 | 3.0 | 0.985 |
| 256 | 4 | 64.0 | 6.0 | 0.970 |
| 512 | 4 | 128.0 | 7.0 | 0.965 |

The 0.005 coefficient means a 2:1 aspect ratio costs 0.5%, a 64:1 ratio costs 3%.  This
is intentionally small because aspect ratio is a secondary concern — a very elongated tile
may still be optimal if the primary scores (bandwidth, grid granularity) are much better
than the square alternative.

**Why penalise elongated tiles?**

The GPU hardware prefetcher in the L2 cache attempts to predict future memory accesses
and pre-load them before they are requested.  For row-major tensor storage:
- A **wide, flat tile** (`XBLOCK=64, YBLOCK=8`) accesses consecutive rows in the X
  direction.  The access pattern along X is sequential — cache line after cache line
  in order.  The prefetcher detects the sequential stride and pre-loads the next
  cache line before it's needed.
- A **tall, thin tile** (`XBLOCK=4, YBLOCK=128`) accesses 128 consecutive Y positions,
  each 4 elements wide.  Each Y step jumps `row_stride` bytes — which may be thousands
  of bytes for wide matrices.  This stride pattern confuses the prefetcher, and each Y
  step results in a new cache-line miss.

The prefetcher benefit is most pronounced when the access pattern has small, predictable
strides.  Wide X = small stride = prefetcher-friendly.

Additionally, the memory coalescing correction in Stage 4b already penalises small
`XBLOCK` separately.  The aspect-ratio penalty here is a *complementary* signal
specifically about the tile *shape* (ratio), not just the absolute X width.

### Factor 2 — Contiguous-axis width bonus

```
x_score = 1.0 − 0.008 × max(0, 5 − log₂(XBLOCK))
```

| XBLOCK | log₂(XBLOCK) | max(0, 5 − log₂) | x_score |
|---|---|---|---|
| 4 | 2.0 | 3.0 | 0.976 |
| 8 | 3.0 | 2.0 | 0.984 |
| 16 | 4.0 | 1.0 | 0.992 |
| 32 | 5.0 | 0.0 | 1.000 |
| 64+ | ≥6.0 | 0.0 | 1.000 |

`XBLOCK ≥ 32` scores 1.000 — no penalty.  `XBLOCK = 4` scores 0.976 — a 2.4% reduction.

**Why the threshold at log₂(XBLOCK) = 5 (XBLOCK=32)?**

A cache line holds 64 bytes = 16 FP32 elements = `XBLOCK = 16` elements.  `XBLOCK = 32`
covers *two* full cache lines per tile row — guaranteed coalescing with one extra cache
line of spatial prefetch.  Below 32, each tile row is a fraction of a cache line or at
most one cache line, which is the minimum threshold for good coalescing.

The `max(0, ...)` ensures the penalty only applies below the threshold (log₂ < 5), not
above.

### Combined tie-breaker multiplier

```
shape_score  = AR_score × x_score
               (bounded to [0.94, 1.00] in practice)

final_score  = composite_score × shape_score
```

For two configs with `composite_score = 0.92`:

```
Config A: XBLOCK=64, YBLOCK=64   → AR=1.0, x_score=1.000 → shape=1.000 → final=0.920
Config B: XBLOCK=4,  YBLOCK=256  → AR=64,  x_score=0.976 → shape=0.965 → final=0.888
Config C: XBLOCK=32, YBLOCK=16   → AR=2.0, x_score=1.000 → shape=0.995 → final=0.915
```

Config A (square tile with wide X) wins; Config B (tall thin tile) is penalised by 3.5%.
In a typical config pool where scores are within 0.01–0.05 of each other, this is
sufficient to break ties without overwhelming the primary signals.

---

## Simple explanation

### Why tile shape matters in 2-D kernels

In a 2-D kernel, data is stored in row-major order — all elements in row 0 are consecutive
in memory, then all elements in row 1, etc.  A tile with `XBLOCK=64` covers 64 consecutive
elements in a row — that's `64 × 4 = 256 bytes` of continuous memory, filling 4 complete
cache lines end-to-end.

A tile with `XBLOCK=4, YBLOCK=64` covers 4 elements per row over 64 rows.  Each row is
only 16 bytes — 1/4 of a cache line.  After processing those 4 elements, the kernel jumps
to the next row, which may be thousands of bytes away.  This "jumping" access pattern is
hard for the GPU's memory prefetcher to predict.

### Two shape penalties

**1. Elongated tile penalty (aspect ratio):**
Very elongated tiles (e.g. `XBLOCK=256, YBLOCK=4` → 64:1 ratio) are penalised because:
- The wide direction (X) accesses many columns in sequence → good coalescing
- The narrow direction (Y) means many row-jumps → hard for the prefetcher

Square tiles (XBLOCK=YBLOCK) avoid row-jumping and let the prefetcher work well along
both axes.  The penalty is small (~0.5% per doubling of the ratio) — it only decides
ties, it doesn't override the primary factors.

**2. Small XBLOCK penalty:**
`XBLOCK < 32` means each tile row doesn't even cover two full cache lines.  The GPU
fetches 64 bytes at a time from memory.  An `XBLOCK=4` tile uses only 16 bytes of each
64-byte fetch — the other 48 bytes are fetched but may not be used before eviction.

Once `XBLOCK ≥ 32`, two cache lines are always filled per row — no further benefit from
wider X in this scoring factor (coalescing correction in 4b handles the 16-element
threshold already).

### How to read this in practice

If you see two 2-D configs with nearly identical primary scores, the one with the more
square tile shape and larger X dimension will win.  The maximum advantage is ~3.5%, which
is decisive for near-ties but invisible when there's a real primary-factor winner.


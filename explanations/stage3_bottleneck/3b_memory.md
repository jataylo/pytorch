# Stage 3b — Memory Transfer Time (T_memory)

**Source:** `BottleneckAnalysis.analyze_bottleneck()` → `estimate_memory_time_us()`
in `triton_heuristics_adaptive.py`

**Role:** Estimates how long it would take to stream all of the kernel's input and output
data through the HBM memory hierarchy, assuming compute were free.  For the vast majority
of pointwise kernels, this is the dominant execution cost.

---

## Technical

### The equation

```
T_memory = total_bytes / (BW_effective)

where:
  total_bytes   = total_elements × bytes_per_element
  BW_effective  = BW_peak × η_hbm × η_coalesce

  BW_peak       = device peak HBM bandwidth (e.g. 5300 GB/s on MI300X, 2000 GB/s on A100)
  η_hbm         = HBM efficiency factor (0.80 clean, 0.65 masked)
  η_coalesce    = coalescing factor (1.00 for 1-D, 0.65–1.00 for 2-D depending on XBLOCK)
```

### Reading each term

#### `total_bytes = total_elements × bytes_per_element`

Every byte that crosses the HBM bus must be counted — both reads and writes.

For a simple `c = a + b` kernel on FP32:
```
bytes_per_element = 4 (read a) + 4 (read b) + 4 (write c) = 12 bytes
total_bytes       = N × 12
```

For an in-place scale `a *= scalar` on FP16:
```
bytes_per_element = 2 (read a) + 2 (write a) = 4 bytes
total_bytes       = N × 4
```

Broadcast tensors contribute far fewer bytes: if a tensor of size 1 is broadcast to a
size-N output, it is read once (or cached in L2 after the first wavefront) rather than N
times.  `extract_kernel_metadata()` detects broadcast patterns and reduces
`bytes_per_element` accordingly.

#### `BW_peak` — device peak bandwidth

Read from `torch.cuda.get_device_properties().memory_bandwidth_gb_per_s` at problem-setup
time.  Representative values:

| GPU | BW_peak |
|---|---|
| AMD MI300X | 5,300 GB/s |
| AMD MI250X | 3,276 GB/s |
| NVIDIA A100 (80 GB) | 2,000 GB/s |
| NVIDIA H100 SXM | 3,350 GB/s |

This is the absolute ceiling set by HBM DRAM width × clock speed.  No real workload
achieves 100% of this number.

#### `η_hbm` — HBM efficiency derating

Achievable bandwidth is always below peak for three hardware reasons:

1. **ECC overhead:** High-bandwidth memory uses inline ECC — error-correcting codes are
   computed and checked on every DRAM burst.  This consumes ~5–8% of raw bandwidth.

2. **DRAM refresh:** HBM must periodically pause reads/writes to refresh its capacitors.
   During refresh windows (every ~64 ms, each row takes ~100 ns), the memory controller
   must stall any pending transactions.  This contributes ~1–3% overhead.

3. **Row-buffer conflicts:** When consecutive accesses land on different DRAM rows in the
   same bank, the memory controller must precharge and activate a new row, adding ~30 ns
   of latency and reducing effective throughput.  Sequential streaming patterns minimise
   this; random access patterns maximise it.

Combined, clean sequential streaming achieves ~80% of peak:
```
η_hbm = 0.80  (clean streaming, no masking)
```

For **masked kernels** (`has_mask = True`), predicate instructions cause partial
cache-line stores.  When a wavefront writes only some of the 64 bytes in a cache line,
the memory controller cannot use write-combining.  It must instead:
1. Read the existing cache line from HBM
2. Merge the predicated bytes (write only the non-masked elements)
3. Write the full cache line back to HBM

This read-modify-write cycle roughly doubles the write traffic and breaks the
write-combining path that coalesces multiple stores into a single HBM burst.  The
efficiency drops to:
```
η_hbm = 0.65  (masked / partial-store kernels)
```

#### L2 vs. HBM — small-problem shortcut

For problems smaller than the L2 cache (typically 4 MB on MI300X), the model substitutes
L2 bandwidth (~1 TB/s on MI300X) for HBM bandwidth:

```
if total_bytes < l2_size_bytes:
    BW_effective = L2_bandwidth × η_l2         # η_l2 ≈ 0.85
else:
    BW_effective = HBM_bandwidth × η_hbm
```

L1 cache (32 KB per CU, replicated across 304 CUs) is **not** treated as a global cache.
Data that "fits in L1" for one block is still fetched from HBM by all other CUs running
in parallel — the L1 benefit is local to a single block, not the whole kernel.

#### `η_coalesce` — 2-D memory coalescing

For 2-D kernels, the X dimension maps to the **contiguous (row-major) axis** in memory.
The GPU memory controller issues 64-byte cache-line fetches.  At FP32 (4 bytes/element),
one cache line holds 16 elements.

If `XBLOCK < 16`, each tile row spans less than one full cache line:

```
         ← XBLOCK=4 elements = 16 bytes ─►
Cache line: [████ OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO OOOO]
             used   ←  fetched but unused (48 bytes)  →
```

The memory controller fetches the full 64 bytes but only 16 bytes are useful.  The
remaining 48 bytes are read over HBM but discarded — 75% bus waste.

```
coalescing  = min(1.0, XBLOCK / 16)
η_coalesce  = 0.65 + 0.35 × coalescing

XBLOCK= 4  → coalescing=0.25 → η_coalesce=0.74
XBLOCK= 8  → coalescing=0.50 → η_coalesce=0.82
XBLOCK=16  → coalescing=1.00 → η_coalesce=1.00   (full cache-line fill)
XBLOCK=64  → coalescing=1.00 → η_coalesce=1.00
```

Note: `η_coalesce` modifies `BW_score` in Stage 4b rather than `T_memory` directly.
The memory-time model uses the 1-D `η_hbm` for simplicity; the scoring factor
accounts for 2-D coalescing separately.

### Complete example — MI300X, `c = relu(a + b)`, FP32, N=10M

```
bytes_per_element = 4 × 3 = 12 bytes
total_bytes       = 10⁷ × 12 = 120 MB

BW_peak           = 5300 GB/s = 5.3 × 10¹² bytes/s
η_hbm             = 0.80       (no masking assumed)
BW_effective      = 5.3 × 10¹² × 0.80 = 4.24 × 10¹² bytes/s

T_memory          = (120 × 10⁶) / (4.24 × 10¹²)
                  ≈ 28.3 × 10⁻⁶ s
                  ≈ 28.3 µs
```

This is the **floor** on kernel runtime — no matter how fast the config, we cannot move
120 MB through HBM in less than ~28 µs on MI300X.

---

## Simple explanation

### What "memory time" means

Most AI kernels spend most of their time waiting for data to arrive from memory, not
doing arithmetic.  The GPU's math units (ALUs) are fast — they can finish a multiply-add
in 1 clock cycle (~4 ns).  But loading the input data from High Bandwidth Memory (HBM —
the main GPU memory) takes ~300 clock cycles (~1,200 ns) for each load.

`T_memory` estimates: *if arithmetic were free, how long would it take just to move all
the input and output data through the memory system?*  This is the hard lower bound on
how fast the kernel can possibly run.

### Why config choices affect memory time

**Total bytes = problem size × bytes per element**

Each tensor (input and output) is one full pass over all elements.  A kernel that reads
two tensors and writes one reads 3× the problem data.  At FP32 (4 bytes), 100 million
elements means 3 × 400 MB = 1.2 GB of memory traffic per kernel call.

**Efficiency losses:**

1. **ECC, refresh, row-buffer conflicts (η_hbm = 0.80):** The hardware cannot achieve
   100% of its advertised peak bandwidth.  Real streaming peaks around 80%.  Think of it
   like a motorway that's rated for 100 mph but always has some slow patches — you
   realistically average 80 mph.

2. **Masking (η_hbm = 0.65):** If the problem size isn't a multiple of the block size,
   partial writes to memory are needed.  These partial writes force the memory controller
   to read the existing cache line, patch it, then write it back — doubling write traffic
   and dropping effective bandwidth to 65% of peak.

3. **2-D coalescing:** For 2-D kernels, if the X tile width (`XBLOCK`) is small (like 4),
   each row of the tile only covers 16 bytes — but the GPU always fetches 64 bytes at a
   time from memory.  The other 48 bytes are fetched but discarded.  A wider X tile
   (`XBLOCK=16` or more) fills the full 64-byte fetch, wasting nothing.

### Where this number goes

`T_memory` is compared with `T_compute` in Stage 3d.  Whichever is larger is the real
bottleneck — the other is hidden "for free" because the GPU's memory system and math
units run in parallel.  `T_overhead` is then added on top (it can't run in parallel with
anything, because the kernel hasn't started yet).


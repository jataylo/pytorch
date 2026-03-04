# XBLOCK / YBLOCK — 1-D and 2-D Kernel Examples

This document walks through how Triton Inductor maps a tensor operation onto a
grid of blocks, using `XBLOCK` (and `YBLOCK` for 2-D) as tile sizes.

---

## 1-D example — element-wise add

### The operation

```python
out = a + b          # a, b, out are 1-D tensors of length N = 1,048,576
```

### What Inductor generates

```python
@triton.jit
def add_kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Which block am I?
    pid = tl.program_id(0)                         # block index in the grid

    # Which elements does this block own?
    xoffset = pid * XBLOCK                         # first element index
    xindex  = xoffset + tl.arange(0, XBLOCK)      # vector [pid*XBLOCK .. pid*XBLOCK+XBLOCK-1]
    xmask   = xindex < xnumel                      # boundary guard for last block

    # Load, compute, store
    a   = tl.load(in_ptr0 + xindex, mask=xmask)
    b   = tl.load(in_ptr1 + xindex, mask=xmask)
    out = a + b
    tl.store(out_ptr0 + xindex, out, mask=xmask)
```

### Grid

```
grid = (ceil(N / XBLOCK),)     # number of blocks launched
```

### How blocks tile the tensor

```
N = 1,048,576    XBLOCK = 1024    →  grid = (1024 blocks,)

Block 0:    elements   0 ..  1023   (xoffset =    0)
Block 1:    elements  1024 ..  2047  (xoffset = 1024)
Block 2:    elements  2048 ..  3071  (xoffset = 2048)
...
Block 1023: elements 1047552..1048575
```

Each block is an independent unit of work.  The GPU scheduler assigns blocks to
free Compute Units (CUs).  All 1024 blocks can run in parallel if 1024 CUs are
available.

### What XBLOCK controls

| XBLOCK | num_blocks | threads/block (num_warps×64) | Effect |
|---|---|---|---|
| 256 | 4096 | 256 (4 warps) | Many small blocks — high grid parallelism, lower EPB |
| 1024 | 1024 | 256 (4 warps) | Balanced — matches 2×num_CUs on MI300X (608) |
| 4096 | 256 | 256 (4 warps) | Few large blocks — high EPB, low grid parallelism |

`threads/block` is `num_warps × warp_size`.  `XBLOCK` is the number of
**elements** per block, not the number of threads — one thread can process
multiple elements if `XBLOCK > threads/block`.

```
elements_per_thread (EPT) = XBLOCK / (num_warps × warp_size)

e.g. XBLOCK=1024, num_warps=4, warp_size=64 → EPT = 1024/256 = 4
     XBLOCK=4096, num_warps=4, warp_size=64 → EPT = 4096/256 = 16
```

`EPT > 1` means each thread processes multiple elements in a loop.  Higher EPT
enables software pipelining (`num_stages`) to hide per-thread memory latency.

---

## 2-D example — element-wise add on a matrix

### The operation

```python
out = a + b          # a, b, out are 2-D tensors, shape [M, N] = [1024, 1024]
```

Inductor flattens to 1-D internally (`total_elements = M × N = 1,048,576`) for
simple pointwise ops, so the kernel structure is identical to the 1-D case.
YBLOCK only appears when Inductor **keeps the 2-D structure**, which happens for
certain reduction + pointwise fusions or when it can exploit row-locality.

### When YBLOCK appears

```python
@triton.jit
def add_kernel_2d(in_ptr0, in_ptr1, out_ptr0,
                  xnumel, ynumel,
                  XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):

    # Each block gets a unique (pid_x, pid_y) pair from the hardware scheduler.
    # pid_x identifies which column-tile this block owns (0 .. ceil(N/XBLOCK)-1).
    # pid_y identifies which row-tile this block owns    (0 .. ceil(M/YBLOCK)-1).
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    # Convert the block index into the starting element index for this tile.
    # Block pid_x=3, XBLOCK=64 → this block starts at column 192.
    xoffset = pid_x * XBLOCK
    yoffset = pid_y * YBLOCK

    # Build a vector of all column indices this block will touch.
    # tl.arange(0, XBLOCK) = [0, 1, 2, ..., XBLOCK-1]
    # [None, :] reshapes it to [1, XBLOCK] so it broadcasts with yindex [YBLOCK, 1].
    # Result: every row in this tile shares the same column indices.
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]   # shape [1,     XBLOCK]

    # Build a vector of all row indices this block will touch.
    # [:, None] reshapes to [YBLOCK, 1] so it broadcasts with xindex [1, XBLOCK].
    # Result: every column in this tile shares the same row indices.
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]   # shape [YBLOCK, 1    ]

    # Guard against out-of-bounds access for the last (partial) block.
    # e.g. M=1000, YBLOCK=32: the last row-block covers rows 992..1023,
    # but rows 1000..1023 don't exist → ymask marks them False → no load/store.
    xmask = xindex < xnumel                             # shape [1,     XBLOCK]
    ymask = yindex < ynumel                             # shape [YBLOCK, 1    ]
    mask  = xmask & ymask   # broadcast → shape [YBLOCK, XBLOCK]; True = valid element

    # Compute the flat (1-D) memory offset for each element in the [YBLOCK×XBLOCK] tile.
    # Row-major: element at (row, col) lives at address base + row*N + col.
    # Broadcasting: yindex [YBLOCK,1] + xindex [1,XBLOCK] → [YBLOCK, XBLOCK] grid of offsets.
    flat = yindex * xnumel + xindex                     # shape [YBLOCK, XBLOCK]

    # Load YBLOCK×XBLOCK elements from each input tensor in one vectorised operation.
    # mask=mask ensures out-of-bounds lanes return 0 instead of faulting.
    a   = tl.load(in_ptr0 + flat, mask=mask)
    b   = tl.load(in_ptr1 + flat, mask=mask)
    out = a + b
    # Write results back; masked lanes are silently skipped (no store issued).
    tl.store(out_ptr0 + flat, out, mask=mask)
```

### Grid for 2-D kernel

```
grid = (ceil(N / XBLOCK), ceil(M / YBLOCK))

e.g. M=1024, N=1024, XBLOCK=64, YBLOCK=32:
     grid = (16, 32)  →  512 blocks total
```

### How the tile covers the matrix

```
Matrix [M=1024 rows × N=1024 cols]

        col 0..63   col 64..127  ...  col 960..1023
row 0..31:  [block(0,0)] [block(1,0)]  ...  [block(15,0)]
row 32..63: [block(0,1)] [block(1,1)]  ...  [block(15,1)]
...
row 992..1023:                              [block(15,31)]
```

Each block owns a `YBLOCK × XBLOCK` rectangle of the matrix.

### XBLOCK vs YBLOCK — what each controls

| Dimension | Controls | Contiguous in memory? |
|---|---|---|
| `XBLOCK` | number of **columns** per block (innermost) | ✓ Yes — row-major, columns are adjacent |
| `YBLOCK` | number of **rows** per block | ✗ No — rows are `N` elements apart |

**Why this matters for memory:**

```
Row-major layout:  [ row0_col0, row0_col1, ..., row0_colN-1,
                     row1_col0, row1_col1, ..., row1_colN-1, ... ]
                     ↑ contiguous                ↑ stride = N elements apart
```

Loading `XBLOCK` consecutive columns → one contiguous HBM burst → good coalescing.
Loading `YBLOCK` consecutive rows → stride-N accesses → separate HBM transactions.

This is why `XBLOCK` tends to be larger than `YBLOCK` in 2-D configs — the
heuristic keeps X aligned with the fast (contiguous) dimension.

---

## Key relationships — cheat sheet

```
num_blocks         = ceil(total_elements / (XBLOCK × YBLOCK))
threads_per_block  = num_warps × warp_size           (e.g. 4 × 64 = 256)
elements_per_thread = (XBLOCK × YBLOCK) / threads_per_block

Scoring impact:
  XBLOCK↑  →  num_blocks↓  →  Grid score↓ (if below target)
                             →  EPB↑        →  Launch score↑
                             →  EPT↑        →  ILP benefit possible
  XBLOCK↓  →  num_blocks↑  →  Grid score↑ (more CUs active)
                             →  EPB↓        →  Launch score↓
```

The four scoring factors (Bandwidth, Launch, Grid, Occupancy) jointly navigate
this trade-off to find the `XBLOCK` (and `YBLOCK`) that best balances
parallelism against per-block efficiency.


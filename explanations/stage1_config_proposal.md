# Stage 1 — Exhaustive Configuration Proposal

**Source:** `triton_heuristics_pointwise.py` → `generate_all_candidate_configs(problem_metadata)`

**Role in pipeline:** Produces every *legally valid* `(XBLOCK, YBLOCK, ZBLOCK, num_warps)`
combination for the given problem, forming the candidate pool that all later stages score
and prune.

---

## Technical

### What is a Triton config?

A Triton `Config` is a dictionary of compile-time constants injected into the kernel via
`tl.constexpr`.  For pointwise kernels the relevant fields are:

| Field | Meaning | Effect on codegen |
|---|---|---|
| `XBLOCK` | Tile width along the X (fastest-moving) dimension | Loop trip count = `ceil(xnumel / XBLOCK)` |
| `YBLOCK` | Tile height (2-D kernels only) | Adds a second tiled loop |
| `ZBLOCK` | Tile depth (3-D kernels only) | Adds a third tiled loop |
| `num_warps` | Wavefronts per block | Governs VGPR allocation, launch cost, occupancy |

The combination of block sizes and warp count fully determines:
- How many GPU blocks are launched (`num_blocks = ceil(xnumel/XBLOCK) × ceil(ynumel/YBLOCK) × …`)
- How many threads per block (`threads = num_warps × warp_size`)
- How many VGPRs each thread receives (`65536 // threads`, capped at 256)

### Candidate sets

```
Dimensionality   XBLOCK candidates                        Warp candidates
─────────────    ─────────────────────────────────────    ───────────────
1-D              {16, 32, 64, 128, 256, 512, 1024}        {1, 2, 4, 8, 16}
2-D              {4, 8, 16, 32, 64, 128, 256, 512, 1024}  {1, 2, 4, 8, 16}
                 (same set for YBLOCK)
3-D              {4, 8, 16, 32, 64}                       {1, 2, 4, 8, 16}
                 (same set for YBLOCK, ZBLOCK)
```

3-D uses a smaller maximum block size because the tile *volume* (`XBLOCK × YBLOCK × ZBLOCK`)
already grows cubically; allowing 1024 per axis would produce tiles with 10⁹ elements —
far more than any problem's total size.

### Pruning constraints

Before any scoring, two hard constraints discard illegal configs from the raw Cartesian
product:

#### Constraint 1 — Dimension cap
```
XBLOCK ≤ xnumel
YBLOCK ≤ ynumel   (2-D / 3-D only)
ZBLOCK ≤ znumel   (3-D only)
```
A block dimension larger than the problem dimension would allocate threads that have no
element to process.  Those threads still consume registers and launch overhead, but
contribute zero useful work.  Triton masks them out with `tl.load(..., mask=...)`, but
the register allocation and SPI initialisation still happen — pure waste.

#### Constraint 2 — Warp–thread coherence
```
num_warps × warp_size ≤ XBLOCK × YBLOCK × ZBLOCK
```
`warp_size` is the minimum schedulable thread group — 64 on AMD CDNA, 32 on NVIDIA.
`num_warps × warp_size` is the number of threads the **compiler** assumes are in the
block.  If this exceeds the number of elements the tile covers, the hardware would
silently clamp the thread count, making the declared `num_warps` inconsistent with what
the kernel actually launches.  This breaks the compiler's register-allocation model and
produces incorrect or unpredictable spill behaviour.

```python
# Simplified excerpt — 1-D
for xblock in [16, 32, 64, 128, 256, 512, 1024]:
    if xblock > xnumel:
        continue                              # Constraint 1
    for num_warps in [1, 2, 4, 8, 16]:
        if num_warps * warp_size > xblock:
            continue                          # Constraint 2
        configs.append({'XBLOCK': xblock, 'num_warps': num_warps})
```

Both constraint values (`warp_size`, `max_threads_per_block`) are read from
`problem_metadata` which was populated from `torch.cuda.get_device_properties()` at
call time.  The generator is therefore **device-agnostic** — it generates correct legal
configs on MI300X (`warp_size=64`), A100 (`warp_size=32`), and future hardware without
modification.

### Output size

| Kernel type | Typical surviving configs |
|---|---|
| 1-D | 30–80 |
| 2-D | 60–200 |
| 3-D | 15–60 |

Every surviving config is guaranteed to be **compilable and correct** — it will produce
valid output.  The question answered by Stages 2–4 is which one will be *fastest*.

### Why powers of two?

Triton's vectoriser operates on power-of-two tile widths for two reasons:

1. **Alignment:** HBM cache lines are 64 bytes = 16 FP32 elements.  Power-of-two block
   sizes align naturally to cache-line boundaries, maximising bus utilisation.

2. **Loop unrolling:** Triton unrolls the innermost loop `XBLOCK / warp_size` times.  A
   non-power-of-two unroll count forces the compiler to emit a partial loop epilogue with
   scalar fallback code, breaking the vectorisation pipeline.  Powers of two allow the
   compiler to unroll completely with no remainder.

---

## Simple explanation

### What this stage does

Think of it as **writing a menu** before you cook.  A chef doesn't start cooking without
first deciding what dishes are possible given the ingredients available.  Stage 1 does the
same: before any GPU runs, it lists every possible way to divide the work.

### What a "config" is

A GPU kernel doesn't process all data at once — it divides it into rectangular tiles and
processes one tile at a time.

- **Block size (`XBLOCK`, `YBLOCK`)** — how big each tile is.  `XBLOCK=256` means each
  block processes 256 elements along the X axis.
- **`num_warps`** — how many groups of threads work on that tile simultaneously.  Each
  "warp" (called a *wavefront* on AMD) is a group of 64 threads that move in lockstep.

Different combinations give wildly different performance:
- A small block size creates many small tiles → lots of overhead launching them.
- A large block size with many warps → uses lots of registers → possible register spill.
- Too few warps → the GPU can't hide memory latency.

### What gets filtered out and why

The generator creates all combinations but throws away two kinds of invalid configs:

1. **Tile bigger than the problem:** If the problem is 64 elements and you make a 128-element
   block, half your threads would have nothing to do.  The GPU still allocates registers
   for them — pure wasted resources.

2. **More thread-groups than threads:** If you declare 8 warps but your tile is only 32
   elements wide, you're asking for 512 threads but only 32 exist.  The hardware clamps
   this silently, making the compiler's assumptions about registers incorrect.

### Result

After filtering, you have 30–200 valid configurations — like a shortlist of recipes that
will all produce a correct result.  Stages 2–4 figure out which one is fastest without
cooking all of them.


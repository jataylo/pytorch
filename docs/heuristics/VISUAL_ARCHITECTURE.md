# Pointwise Heuristics V4 - Visual Architecture

## 🎯 High-Level System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                         USER APPLICATION                              │
│                                                                        │
│  @torch.compile                                                       │
│  def kernel(x, y, w):                                                 │
│      return x + y * w                                                 │
│                                                                        │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ↓
┌───────────────────────────────────────────────────────────────────────┐
│                      PYTORCH INDUCTOR                                 │
│                                                                        │
│  • Generates Triton kernel code                                       │
│  • Detects operation type: POINTWISE                                  │
│  • Extracts metadata: dimensions, num_inputs, element_size            │
│                                                                        │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ↓
┌───────────────────────────────────────────────────────────────────────┐
│                 runtime/triton_heuristics.py                          │
│                    (INTEGRATION LAYER)                                │
│                                                                        │
│  pointwise() entry point                                              │
│    ↓                                                                   │
│  _apply_pointwise_heuristics()                                        │
│    ├─ Generate configs                                                │
│    ├─ Score with V4 adaptive                                          │
│    ├─ Select top 5                                                    │
│    └─ Return to autotuner                                             │
│                                                                        │
└─────┬────────────────────────────────────────┬─────────────────┬──────┘
      │                                        │                 │
      ↓                                        ↓                 ↓
┌──────────────────┐    ┌────────────────────────┐    ┌──────────────────┐
│   HARDWARE       │    │     ADAPTIVE           │    │   POINTWISE      │
│   MODULE         │    │     MODULE             │    │   MODULE         │
│                  │    │                        │    │                  │
│  hardware.py     │    │  adaptive.py           │    │  pointwise.py    │
│                  │    │                        │    │                  │
│  • Query GPU     │    │  • Analyze bottleneck  │    │  • Generate      │
│  • Derive        │    │  • Estimate times      │    │    configs       │
│    optimal       │    │  • Get adaptive        │    │  • Calculate     │
│    values        │    │    weights             │    │    factors       │
│                  │    │                        │    │  • Score config  │
│                  │    │                        │    │                  │
└──────────────────┘    └────────────────────────┘    └──────────────────┘
```

---

## 🔄 V4 Adaptive Scoring Flow (Per Config)

```
┌────────────────────────────────────────────────────────────────┐
│ INPUT: Config + Problem Metadata                               │
│   Config: {XBLOCK: 256, num_warps: 4}                         │
│   Problem: {dimensions: (65536,), num_inputs: 3, ...}         │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 1: Hardware Query (hardware.py)                          │
│                                                                 │
│  get_architecture_config()                                     │
│    └─> {num_cus: 304, warp_size: 64,                         │
│         optimal_threads: 256,                                  │
│         optimal_elements: 2048,                                │
│         optimal_blocks: 608}                                   │
│                                                                 │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 2: Bottleneck Analysis (adaptive.py) [V4 INNOVATION]     │
│                                                                 │
│  analyze_bottleneck(config, problem)                           │
│                                                                 │
│    Calculate grid & threads:                                   │
│      threads_per_block = 256                                   │
│      num_blocks = ceil(65536 / 256) = 256                     │
│                                                                 │
│    Estimate times:                                             │
│      overhead_us = 3.0μs (kernel launch)                      │
│      memory_us = 12,288 bytes / (3500 GB/s * 0.8)            │
│                = 10.7μs                                        │
│      compute_us = (65536 * 2 ops) / 1000 GFLOPS              │
│                 = 0.13μs                                       │
│                                                                 │
│    Calculate fractions:                                        │
│      total_us = 3.0 + 10.7 + 0.13 = 13.83μs                  │
│      overhead_frac = 3.0 / 13.83 = 22%                       │
│      memory_frac = 10.7 / 13.83 = 77%  ← DOMINANT             │
│      compute_frac = 0.13 / 13.83 = 1%                        │
│                                                                 │
│    Identify bottleneck:                                        │
│      memory_frac > 40% → bottleneck = 'MEMORY'               │
│                                                                 │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 3: Adaptive Weights (adaptive.py)                        │
│                                                                 │
│  get_adaptive_weights(bottleneck='memory')                     │
│                                                                 │
│    Base weights for MEMORY-bound:                             │
│      bandwidth:  40%  (maximize throughput)                   │
│      launch:     25%  (amortize overhead)                     │
│      grid:       20%  (saturate GPU)                          │
│      occupancy:  15%  (latency hiding)                        │
│                                                                 │
│    Fine-tune (overhead 22% > 20% threshold):                  │
│      boost launch by +2%                                       │
│                                                                 │
│    Final weights:                                              │
│      bandwidth:  39%                                           │
│      launch:     27%                                           │
│      grid:       19%                                           │
│      occupancy:  15%                                           │
│                                                                 │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 4: Convert Weights → Exponents (adaptive.py)             │
│                                                                 │
│  get_adaptive_exponents(weights)                               │
│                                                                 │
│    Map weights [10%, 50%] → exponents [0.5, 3.0]:            │
│      bandwidth:  39% → 2.4                                     │
│      launch:     27% → 1.6                                     │
│      grid:       19% → 1.1                                     │
│      occupancy:  15% → 0.8                                     │
│                                                                 │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 5: Calculate Factor Scores (pointwise.py)                │
│                                                                 │
│  estimate_memory_bandwidth(config, problem)                    │
│    threads = 256, optimal = 256                               │
│    score = 0.75 + 0.25 * exp(-0.5 * ((256-256)/256)²)       │
│         = 1.000                                                │
│                                                                 │
│  estimate_launch_overhead(grid, problem)                       │
│    elems_per_block = 65536 / 256 = 256                       │
│    optimal = 2048                                              │
│    score = 0.75 + 0.25 * exp(-0.5 * ((256-2048)/1024)²)     │
│         = 0.783                                                │
│                                                                 │
│  estimate_grid_granularity(grid, problem)                      │
│    num_blocks = 256, optimal = 608/2 = 304 (medium problem)  │
│    score = 0.75 + 0.25 * exp(-0.5 * ((256-304)/152)²)       │
│         = 0.994                                                │
│                                                                 │
│  estimate_occupancy_impact(config, problem)                    │
│    wavefronts = 256/64 = 4, num_warps = 4                    │
│    base = 1.0 (in sweet spot 4-8)                            │
│    warp_match = 1.0 (num_warps == wavefronts)                │
│    score = 1.000                                               │
│                                                                 │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 6: Weighted Geometric Mean (pointwise.py)                │
│                                                                 │
│  score_config()                                                │
│                                                                 │
│    Factors: bw=1.000, launch=0.783, grid=0.994, occ=1.000    │
│    Exponents: 2.4, 1.6, 1.1, 0.8                              │
│                                                                 │
│    score = (bw^2.4 * launch^1.6 * grid^1.1 * occ^0.8)^φ      │
│          = (1.0^2.4 * 0.783^1.6 * 0.994^1.1 * 1.0^0.8)^φ     │
│          = (1.0 * 0.690 * 0.993 * 1.0)^(1/5.9)               │
│          = 0.685^0.169                                         │
│          = 0.935                                               │
│                                                                 │
│  OUTPUT: Config score = 0.935                                 │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Top-5 Selection Flow

```
┌────────────────────────────────────────────────────────────────┐
│ ALL Configs Scored (15 configs)                               │
│                                                                 │
│  #1:  XBLOCK=256, nw=4   score=0.935  ← Top predicted        │
│  #2:  XBLOCK=256, nw=2   score=0.928                         │
│  #3:  XBLOCK=256, nw=1   score=0.921                         │
│  #4:  XBLOCK=512, nw=8   score=0.916                         │
│  #5:  XBLOCK=512, nw=4   score=0.912  ← Top 5 cutoff         │
│  #6:  XBLOCK=512, nw=2   score=0.905  ← Outside top 5        │
│  ...                                                           │
│  #15: XBLOCK=16,  nw=1   score=0.701                         │
│                                                                 │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ↓
┌────────────────────────────────────────────────────────────────┐
│ IF HEURISTICS_REAL_BENCH=1: Benchmark ALL                     │
│                                                                 │
│  benchmark_all_configs()                                       │
│                                                                 │
│    Results (actual timings):                                   │
│      #1:  0.0064ms                                            │
│      #2:  0.0065ms                                            │
│      #3:  0.0066ms                                            │
│      #4:  0.0067ms                                            │
│      #5:  0.0068ms                                            │
│      #6:  0.0063ms  ← ACTUAL FASTEST!                        │
│      ...                                                       │
│      #15: 0.0095ms                                            │
│                                                                 │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ↓
┌────────────────────────────────────────────────────────────────┐
│ autotune_to_one_config() - SELECT FROM TOP 5 ONLY             │
│                                                                 │
│  Filter to top 5 predicted configs:                           │
│    {#1: 0.0064ms, #2: 0.0065ms, #3: 0.0066ms,               │
│     #4: 0.0067ms, #5: 0.0068ms}                              │
│                                                                 │
│  Select fastest from filtered:                                │
│    Winner: Config #1 (0.0064ms)                              │
│                                                                 │
│  Note: Config #6 (0.0063ms) was actual fastest               │
│        but EXCLUDED (outside top 5)                           │
│        Performance gap: 1.6% - acceptable!                    │
│                                                                 │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ↓
┌────────────────────────────────────────────────────────────────┐
│ _print_heuristics_validation_summary()                        │
│                                                                 │
│  Compare:                                                      │
│    Predicted best: #1 (0.0064ms)                             │
│    Actual best: #6 (0.0063ms) [from ALL 15]                  │
│    Gap: 1.6%                                                   │
│                                                                 │
│  Analysis:                                                     │
│    ⚠️  Actual best ranked #6 (outside top 5)                  │
│    ✅ Selected config within 1.6% of optimal                  │
│    ✅ Top-5 restriction provided safety net                   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 📊 Bottleneck Decision Tree

```
                    ┌─────────────────────┐
                    │ For Config X        │
                    │ Estimate Times:     │
                    │ • overhead_us       │
                    │ • memory_us         │
                    │ • compute_us        │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴───────────────┐
                │ Calculate Fractions          │
                │   frac = time_us / total_us  │
                └──────────────┬───────────────┘
                               │
                     ┌─────────┴─────────┐
                     │  overhead_frac    │
                     │     > 40%?        │
                     └─────────┬─────────┘
                          Yes  │  No
                     ┌─────────┴─────────┐
                     │                   │
                     ↓                   ↓
          ┌──────────────────┐   ┌──────────────────┐
          │ OVERHEAD-BOUND   │   │  memory_frac     │
          │                  │   │    > 40%?        │
          │ Weights:         │   └────────┬─────────┘
          │ • launch: 50%    │       Yes  │  No
          │ • grid: 30%      │   ┌────────┴────────┐
          │ • bandwidth: 10% │   │                 │
          │ • occupancy: 10% │   ↓                 ↓
          │                  │ ┌───────────┐  ┌──────────┐
          │ Favors:          │ │MEMORY-    │  │COMPUTE-  │
          │ • 1 block        │ │BOUND      │  │BOUND     │
          │ • Minimize       │ │           │  │          │
          │   overhead       │ │Weights:   │  │Weights:  │
          └──────────────────┘ │• bw: 40%  │  │• occ:35% │
                               │• lch:25%  │  │• grd:30% │
                               │• grd:20%  │  │• bw: 20% │
                               │• occ:15%  │  │• lch:15% │
                               │           │  │          │
                               │Favors:    │  │Favors:   │
                               │• 256 thr  │  │• Many    │
                               │• Saturate │  │  warps   │
                               └───────────┘  └──────────┘
```

---

## 🔄 Complete System Data Flow

```
┌───────────────────────────────────────────────────────────────┐
│                      USER CODE                                │
│  result = torch.compile(kernel)(x, y, w)                     │
└─────────────────────────┬─────────────────────────────────────┘
                          │
                          ↓
         ╔════════════════════════════════════════╗
         ║        PYTORCH INDUCTOR                ║
         ║  • FX graph optimization               ║
         ║  • Detects pointwise pattern           ║
         ║  • Generates Triton kernel             ║
         ╚═══════════════════╤════════════════════╝
                             │
                             ↓
         ┌────────────────────────────────────────┐
         │ runtime/triton_heuristics.py           │
         │   pointwise() entry                    │
         └───────────────┬────────────────────────┘
                         │
         ┌───────────────┴────────────────┐
         │ Problem Metadata:              │
         │ • dimensions: (65536,)         │
         │ • num_inputs: 3                │
         │ • element_size: 4 bytes        │
         │ • ops_per_element: 2           │
         └───────────────┬────────────────┘
                         │
                         ↓
         ╔════════════════════════════════════════╗
         ║  _apply_pointwise_heuristics()         ║
         ╠════════════════════════════════════════╣
         ║                                        ║
         ║  PHASE 1: Generate Configs             ║
         ║  ├─ generate_all_candidate_configs()   ║
         ║  └─> 15 configs                        ║
         ║                                        ║
         ║  PHASE 2: Score ALL (V4 Adaptive)      ║
         ║  FOR EACH config:                      ║
         ║    ├─ Query hardware (optimal values)  ║
         ║    ├─ Analyze bottleneck              ║
         ║    ├─ Get adaptive weights            ║
         ║    ├─ Calculate factor scores         ║
         ║    └─ Weighted geometric mean         ║
         ║                                        ║
         ║  PHASE 3: Rank & Select                ║
         ║  ├─ Sort by score                      ║
         ║  ├─ Select top 5                       ║
         ║  └─ Store predictions                  ║
         ║                                        ║
         ╚════════════════╤═══════════════════════╝
                          │
                          ↓
         ┌────────────────────────────────────────┐
         │ Return top 5 configs to autotuner:     │
         │  [cfg1, cfg2, cfg3, cfg4, cfg5]        │
         └───────────────┬────────────────────────┘
                         │
         ┌───────────────┴────────────────────────┐
         │ IF HEURISTICS_REAL_BENCH=1:            │
         │   Benchmark ALL 15 configs             │
         │ ELSE:                                   │
         │   Benchmark top 5 only                 │
         └───────────────┬────────────────────────┘
                         │
                         ↓
         ╔════════════════════════════════════════╗
         ║  autotune_to_one_config()              ║
         ╠════════════════════════════════════════╣
         ║                                        ║
         ║  • Filter benchmarks to top 5          ║
         ║  • Select fastest from top 5           ║
         ║  • Store actual timings                ║
         ║                                        ║
         ╚════════════════╤═══════════════════════╝
                          │
         ┌────────────────┴────────────────────────┐
         │ IF HEURISTICS_REAL_BENCH=1:             │
         │   _print_heuristics_validation_summary()│
         │   • Compare predicted vs actual         │
         │   • Show factor breakdown               │
         │   • Calculate accuracy                  │
         └────────────────┬────────────────────────┘
                          │
                          ↓
         ┌────────────────────────────────────────┐
         │ COMPILED KERNEL READY                  │
         │  Config: XBLOCK=256, num_warps=4       │
         │  Performance: 0.0064ms                 │
         │  Accuracy: 98.4% of optimal            │
         └────────────────────────────────────────┘
```

---

**Legend:**
- `┌─┐` - Regular process step
- `╔═╗` - Major subsystem
- `→` - Data flow
- `├─` - Subprocess/sub-step
- `⚡` - Critical path
- `✅` - Success condition
- `⚠️` - Warning/note

---

**Created:** 2026-02-18  
**Version:** V4 (Adaptive Bottleneck Analysis)


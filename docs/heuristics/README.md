# PyTorch Inductor Pointwise Heuristics Documentation

📚 **Complete documentation for the V4 Adaptive Bottleneck Analysis heuristics system**

---

## 🚀 Start Here

**New to the system?**

1. **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Master index & navigation
2. **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - What this is & what was accomplished
3. **[QUICK_REFERENCE_V4.md](QUICK_REFERENCE_V4.md)** - 5-minute essential facts

---

## 📖 All Documentation

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** | Master navigation | 3 min |
| **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** | Complete summary | 15 min |
| **[QUICK_REFERENCE_V4.md](QUICK_REFERENCE_V4.md)** | Quick lookup | 5 min |
| **[VISUAL_ARCHITECTURE.md](VISUAL_ARCHITECTURE.md)** | Diagrams & flow | 10 min |
| **[HEURISTICS_FLOW.md](HEURISTICS_FLOW.md)** | Complete reference | 30 min |
| **[IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md)** | Math & algorithms | 60 min |
| **[REAL_BENCH_MODE_FLOW.md](REAL_BENCH_MODE_FLOW.md)** | Autotuning integration | 20 min |
| **[AUTOTUNER_RESULTS_LIMITATION.md](AUTOTUNER_RESULTS_LIMITATION.md)** | Troubleshooting | 5 min |

**Total:** 141 KB of focused, cross-referenced documentation

---

## 🎯 What Is This?

The **V4 Adaptive Bottleneck Analysis** system automatically selects optimal GPU kernel configurations for PyTorch Inductor pointwise operations.

**Key Innovation (V4):**
- Analyzes each kernel to identify bottleneck (overhead/memory/compute)
- Dynamically adjusts scoring weights based on bottleneck
- Achieves 95%+ accuracy across all kernel sizes (tiny to large)

**Previous versions (V1-V3):** Fixed weights, failed on tiny kernels

---

## 🧪 Tests & Benchmarks

**Tests:** (in `pytorch/` root)
- `test_adaptive_scoring.py` - Test V4 adaptive weights
- `test_hardware_aware_heuristics.py` - Test hardware detection
- `test_top_n_selection.py` - Test selection logic

**Benchmark:**
- `benchmark_pointwise.py` - Full benchmark suite

**Run from:** `pytorch/` directory

---

## 🔧 Code Modules

**Location:** `pytorch/torch/_inductor/`

- **`config.py`** - Configuration (heuristics_real_bench)
- **`codegen/triton_heuristics_pointwise.py`** - Scoring factors
- **`codegen/triton_heuristics_adaptive.py`** - Bottleneck analysis (V4)
- **`codegen/triton_heuristics_hardware.py`** - Hardware queries
- **`runtime/triton_heuristics.py`** - Integration & autotuning

---

## 📞 Quick Access

| I want to... | Read this |
|--------------|-----------|
| Understand the system | [HEURISTICS_FLOW.md](HEURISTICS_FLOW.md) |
| Get quick facts | [QUICK_REFERENCE_V4.md](QUICK_REFERENCE_V4.md) |
| See diagrams | [VISUAL_ARCHITECTURE.md](VISUAL_ARCHITECTURE.md) |
| Understand math | [IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md) |
| Debug results | [AUTOTUNER_RESULTS_LIMITATION.md](AUTOTUNER_RESULTS_LIMITATION.md) |
| Find anything | [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) |

---

## 🗂️ Historical Documentation

**Location:** `/root/old_docs/` (84 archived docs)  
**Purpose:** Historical context, not for current use

---

**Version:** V4 (Adaptive Bottleneck Analysis)  
**Status:** Production-ready ✅  
**Last Updated:** 2026-02-18



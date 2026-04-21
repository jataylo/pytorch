# Stage 6 — Validation & Verbose Logging

**Sources:**
- `_print_heuristics_validation_summary()` in `triton_heuristics.py`
- `bench()` in `triton_heuristics.py`
- `_HEURISTICS_VALIDATION_DATA` global dict

**Role:** Optionally compares heuristic predictions against real GPU benchmark results,
produces a structured validation report, and tracks hit rates across a benchmark suite.
This stage is the primary tool for iterating on scoring formula accuracy.

---

## Technical

### Two modes of operation

| Mode | `heuristics_real_bench` | What is compiled | What is benchmarked | Winner selection |
|---|---|---|---|---|
| **Heuristics-only** | `False` (default prod) | Top-N (+buffer) configs | Top-N compiled configs | Top-N by heuristic score |
| **Real-bench / validation** | `True` | All 30–200 generated configs | All compiled configs | Top-N heuristic pool only |

In real-bench mode, `_score_and_prune_heuristic_configs` does not prune `self.configs`
— all configs are passed to Triton.  However, `_TOP_N_CONFIGS_FOR_SELECTION` is
populated with the heuristic top-N, and `autotune_to_one_config` restricts the winner
to that pool:

```python
# Real-bench winner selection
top_n_set     = {key(h) for h in _TOP_N_CONFIGS_FOR_SELECTION[problem_key]}
top_n_timings = {cfg: t for cfg, t in timings.items() if key(cfg) in top_n_set}
best          = min(top_n_timings, key=top_n_timings.get)
```

This means real-bench mode answers: *"If we had selected from the heuristic top-N, would
we have gotten a config close to the global best?"*  The global best timing is also
recorded for comparison.

### Data flow through `_HEURISTICS_VALIDATION_DATA`

```
_score_and_prune_heuristic_configs():
    _HEURISTICS_VALIDATION_DATA[key] = {
        'problem_metadata': {...},
        'kernel_code':      <source string>,
        'predicted_scores': [(score, effective_dict, detail_dict, bottleneck_dict), ...],
        'actual_timings':   {},       # filled by bench()
    }

bench(launcher, cfg, ...):
    _store_actual_timing(key, cfg_dict, timing_us)
    # timing_us = float("inf") for spilled configs (stored BEFORE early-return)
    # timing_us = measured microseconds for successful benchmarks
```

`actual_timings` maps `effective_dict_key → timing_us` where the key is a canonical
tuple of sorted `(field, value)` pairs that uniquely identifies the config:

```python
def _timing_key(hdict):
    return tuple(sorted(hdict.items()))
```

### Validation summary — `_print_heuristics_validation_summary()`

Called at the end of `autotune_to_one_config` when `heuristics_verbose=True`.

#### Step 1 — Identify the "heuristic pick"

```python
predicted_sorted = sorted(
    predicted_scores, key=lambda x: x[0], reverse=True  # descending by score
)

# Walk predicted in score order; find first config with finite actual timing
for rank, (score, hdict, detail, bottleneck) in enumerate(predicted_sorted, start=1):
    timing = actual_timings.get(_timing_key(hdict))
    if timing is not None and timing < float("inf"):
        chosen_rank    = rank
        chosen_hdict   = hdict
        chosen_timing  = timing
        break

# If all predicted spill, use best finite timing across all configs
if chosen_timing is None:
    finite = {k: v for k, v in actual_timings.items() if v < float("inf")}
    if finite:
        best_key = min(finite, key=finite.get)
        chosen_timing = finite[best_key]
```

#### Step 2 — Identify the actual best

```python
# Among all benchmarked configs (not just heuristic pool)
finite_timings = {k: v for k, v in actual_timings.items() if v < float("inf")}
actual_best_key = min(finite_timings, key=finite_timings.get)
actual_best_timing = finite_timings[actual_best_key]
```

#### Step 3 — Hit determination

```python
is_top1_hit = (chosen_rank == 1)            # heuristic #1 matches actual best
is_top3_hit = (chosen_rank <= 3)            # actual best is within heuristic top-3
is_top5_hit = (chosen_rank <= 5)            # actual best is within heuristic top-5
speedup_gap = chosen_timing / actual_best_timing    # 1.0 = perfect, >1 = suboptimal
```

#### Step 4 — Factor delta table

For each of the four scoring factors, compute the gap between the predicted score and
what a "perfect" prediction would have scored given the actual best config:

```
Factor  Pred    Actual  Delta   Weight  Note
BW      0.962   0.971   -0.009  55%     predicted LOWER
Launch  0.810   0.804   +0.006  10%     predicted HIGHER
Grid    0.900   1.000   -0.100  15%     predicted LOWER  ← systematic gap
Occ     1.000   0.900   +0.100  20%     predicted HIGHER
```

The "Actual" column is what the factor would score if we knew the actual best config.
The "Delta" = Pred − Actual.  Positive delta = heuristic over-scored this factor for
the predicted config vs. the actual best; negative delta = under-scored.

**Systematic deltas across many kernels indicate scoring formula bugs.**  For example:
- Consistently negative Grid delta → grid Gaussian is centred too low (underscoring
  large-grid configs)
- Consistently positive Occ delta → occupancy sweet-spot is too aggressive (overscoring
  high-warp configs that don't help in practice)

This table is the primary iterative tool used across tuning log versions 4–11.

#### Step 5 — Spill-aware rank labels

When the top-predicted configs all spill:

```
📊 PREDICTED #3  (ranks 1–2 spill-skipped, using next available)
⏱  timing = 18.4 µs   speedup_gap = 1.07×   (actual best: 17.2 µs)
```

The label reflects the actual rank used (3, not 1) and explains why the top predictions
were skipped.  `actual_inf_count` (number of `inf` timings in `actual_timings`) is shown
in the header to quantify how many configs spilled.

### Hit-rate metrics across a benchmark suite

When running `heuristics_real_bench=True` across a suite of kernels (e.g. a full model
forward pass), the validation summaries from each kernel accumulate hit counts:

```
top1_hits / total_kernels    → Top-1 hit rate   (target: >90%)
top3_hits / total_kernels    → Top-3 hit rate   (target: >97%)
top5_hits / total_kernels    → Top-5 hit rate   (target: >99%)
```

When the heuristic misses top-1, the miss is logged with:
- The actual vs. predicted config diff (`XBLOCK: 256 vs 512`, `num_warps: 4 vs 8`)
- The speedup gap (`actual_time / predicted_time` — e.g. `1.23×` means actual is 23%
  faster than heuristic pick)
- The kernel name and `size_hints` for reproduction

### Verbose table format

When `heuristics_verbose=True`, a full scoring table is printed before compilation:

```
[HEURISTICS] Scoring table — problem_key=(32768,)  N=32768  dtype=fp32  device=MI300X
──────────────────────────────────────────────────────────────────────────────────────
Rank  XBLOCK  YBLOCK  nw   Score   BW     Launch  Grid   Occ    Blocks  Spill?  Regime
   1    512      –     8  0.9487  1.000   0.921  0.900  1.000     64    low    MEM
   2    256      –     4  0.9241  0.998   0.840  0.874  1.000    128    low    MEM
   3   1024      –     4  0.9104  0.986   0.960  0.770  0.833     32    low    MEM
   4    128      –     8  0.8812  0.976   0.700  0.900  1.000    256    MED    MEM
   5    512      –    16  0.8631  0.882   0.921  0.900  0.681     64    HIGH   MEM
...
```

`Spill?` shows the Mechanism A estimate label.  `Regime` shows the bottleneck from
Stage 3.  This table is the first place to look when debugging a misprediction.

---

## Simple explanation

### What this stage does

Stage 6 is the **report card** — it compares what the heuristic predicted to what
actually happens when the GPU runs the kernels.

In **production mode** (real_bench=False), Stage 6 just prints a quick confirmation:
which config was selected, and any spill warnings.  Fast and minimal.

In **validation mode** (real_bench=True), Stage 6 is a full audit:
1. Compile every single candidate config (30–200 of them)
2. Run each one on the GPU and measure its speed
3. Compare the heuristic's top-5 against the actual best
4. Print a detailed report showing where the heuristic was right, where it was wrong,
   and which factor scores contributed to any mistake

### The four-column factor table

The factor delta table is the most important diagnostic tool:

```
Factor  Pred    Actual  Delta   Weight
BW      0.96    0.97    -0.01   55%     ← very close, on-target
Grid    0.90    1.00    -0.10   15%     ← heuristic under-scored grid factor
```

If the `Grid` row consistently shows a negative delta across many kernels, it means
the grid scoring formula is systematically undervaluing configs with more blocks than
the current target.  That's actionable feedback to adjust the target block count formula.

### The spill-aware ranking

When the top-predicted configs all fail due to register spills (they return `infinity`
timing), the system doesn't just give up.  It walks down the predicted ranking, finds
the first config that successfully ran on the GPU, and reports that rank explicitly:

```
📊 PREDICTED #3  (ranks 1–2 spill-skipped, using next available)
```

This tells the developer: "Config #3 ran, but configs #1 and #2 both triggered register
spills."  This is actionable — it means the scoring should penalise high-warp configs
more when the estimated VGPR demand is close to the budget.

### How the heuristic improves over time

Real-bench mode turns every forward pass into a labelled training dataset:
- **Input:** problem shape, kernel code, all 30–200 candidate configs
- **Output:** actual timings for every config

By running real-bench mode on a representative model (e.g. LLaMA, Stable Diffusion),
then analysing the systematic deltas in the factor tables across all kernels, specific
scoring formula adjustments can be targeted.  Each improvement is validated by checking
whether the top-1 hit rate increases without regressions on previously-correct kernels.

The target metrics are:
- **Top-1 hit rate > 90%** — the single best prediction matches the actual best
- **Top-3 hit rate > 97%** — the actual best is within 3 predictions
- **Top-5 hit rate > 99%** — the actual best is within 5 predictions (effectively no
  regression over standard 5-config autotuning, just faster startup)


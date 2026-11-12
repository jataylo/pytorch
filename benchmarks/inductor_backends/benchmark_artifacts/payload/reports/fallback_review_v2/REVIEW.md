# Observed FlyDSL fallback review

Snapshot: 2026-11-05 14:25:49 UTC, while the primary training sweep is running.
Source fingerprint: `a45478b9dd4b8dde05b7e605604bbf08931dfab992b388a7bba667d69d5f3f8b`.
The backend and primary benchmark sources have not been changed for this investigation.

The main gap is layout coverage for reductions, especially backward graphs. This is
independent of autotuning: all three reasons have identical event counts with tuning
enabled and disabled. These are generated schedule events, including repeated layers
and separate validation/training graphs, rather than physical GPU launches or a measure
of runtime cost. Profiling is still needed to rank their performance impact.

| Recorded reason | Schedule events | Models | Backward events | Finding |
| --- | ---: | ---: | ---: | --- |
| transposed access needs a 2D tile | 2,036 | 18 | 1,810 | 1,880 events match emitted Triton kernels with x/y output axes plus r0; 156 match x/y/z pointwise kernels |
| online_softmax_reduce of a tuple | 16 | 1 | 0 | CoaT combines partial softmax states; the scalar input path exists, but tuple input is explicitly refused |
| mix-order reduction over a looped row | 12 | 1 | 12 | ConvNeXt LayerNorm backward hits a restricted mixed-order fusion layout |

All 2,036 layout events were matched to saved generated Triton source. The index contains
120 kernel source variants across the three reasons. Multiple schedules can reuse one
kernel definition. No `tl.dot` or `tl.dot_scaled` calls occur in these captured fallback
sources. GEMM/conv policy audits remain separate; an original ATen source mapping that
mentions bmm does not establish that the fallback kernel contains a matrix template.

## Layout coverage

The shared reason is emitted at two different sites in `flydsl_simd.py`:

- `_choose_tiling` rejects a preferred `z` axis, before constructing the FlyDSL kernel.
- `finalize_indexing` rejects a transposed reduction unless the existing row-interleaved
  layout was selected. `_transposed_rows` recognizes a narrow pattern with adjacent rows
  stepping by one memory element. Flattened multi-axis row indices can miss that pattern.

The kernel constructor accepts `{x}`, `{x,r0_}` and `{x,y}`; it does not accept
`{x,y,r0_}` or `{x,y,z}`. Its existing x/y tiled path is pointwise. Ordinary transpose,
LayerNorm and softmax arithmetic are already implemented; these failures concern layout
and coalescing. The guards also avoid known slow flat layouts. Removing them alone is
not evidence of a performance improvement.

Highest event counts are CAiT 728, Qwen3-8B 288, Gemma4-31B 240, LTX-Video 224,
ConViT 120 and Qwen3-0.6B 112. LayerNorm backward appears in 682 layout events;
fused RMSNorm backward appears in 224. These operation counts overlap and describe
source provenance, not distinct unsupported operators.

Recommended first implementation: preserve separate output coordinates for tiled
reductions, with a coalesced inner axis, explicit masks and an appropriate row/column
combine. Start with real Qwen3/CAiT backward shapes. Address x/y/z pointwise schedules
separately by retaining an outer grid axis around the existing transposing tile where
the access pattern permits it. Avoid forcing a generic gather layout globally.

Representative emitted Triton dimensions:

- Qwen3-8B: y=128, x=32, r0=256; and y=1,024, x=8, r0=128.
- CAiT: y=768, x=10, r0=231; and y=2,304, x=3, r0=256.
- MobileViT: y=144, x=512, r0=256, with fused normalization and layout conversion.

The emitted Triton dimensions are evidence of the replacement's geometry. The current
diagnostic string does not capture the exact rejection site or chosen FlyDSL geometry;
that distinction should be logged in the next source revision rather than inferred
solely from the reason text.

## Tuple softmax

CoaT falls back during forward softmax. Representative reductions have x=106,496,
r0=242 for a looped kernel, and x=8,192, r0=13 for a persistent kernel. Code generation
explicitly refuses a tuple in `online_softmax_reduce`.

The runtime already implements `online_softmax_merge((m1,s1),(m2,s2))` and group-level
combining of those states. The missing codegen path must preserve each partial sum,
rather than treating every partial maximum as a raw logit with implicit sum one.
This is the smallest contained support addition in the current snapshot.

The first split stage also receives tuples: masked logits paired with weight one or
zero, over an actual extent of 3,137 tokens. Tuple support must handle both that stage
and the subsequent combination of the 13 partial states.

Validation must cover both looped and persistent paths, masks, partial final tiles,
very different maxima, and all-negative-infinity states. The existing runtime merge
and Triton helpers handle some infinity cases differently internally; reuse requires
checking the resulting semantics rather than deleting the tuple guard alone.

## Mixed-order reduction

ConvNeXt LayerNorm backward produces x=6,272, r0=1,024. `_finalize_mix_order` requires
a persistent row, stride one, no row interleaving, and enough threads across the split
to cover a wave. Its error text says "looped row" when any of those conditions fails.
The actual fallback here is a Triton persistent kernel; the reason does not prove
that the reduction was too long for a persistent implementation.

Extending this path needs a combine that preserves both row-local reductions and
column partial accumulators under the transposed layout. Splitting the fusion is a
possible correctness-first alternative, with extra memory traffic and launches that
must be measured. This is more invasive than tuple softmax support.

Before introducing a new combine algorithm, test a layout-heuristic correction: the
generic row-interleaving heuristic can choose a geometry that mixed-order finalization
then explicitly forbids. Keeping mixed-order kernels on their existing supported row
geometry may recover this case. This needs measured correctness and performance tests.

## Evidence and next validation

- `summary.json`, `reasons.csv`, `models.csv`: counts, affected models, directions,
  dtypes, source operation associations and current model status.
- `events.jsonl`, `examples.json`: full captured diagnostics and source graph fragments.
- `triton_source_index.json`, `triton_sources/`: actual replacement source, dimensions,
  source hashes and original wrapper paths.
- `review_fallbacks_live_v2.py` and `inspect_fallback_sources_v2.py` in the parent
  directory regenerate these snapshots without running GPU kernels.

After the frozen measurements finish, isolate representative kernels, compare output
and gradients with eager/Triton, then measure each prospective FlyDSL path on the same
device and inputs. Frequency alone is insufficient to claim these gaps dominate full
training time. New backend code requires a new source fingerprint and separate results.

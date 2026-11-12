# Coverage audit snapshot

Captured 2026-11-05T20:13:53.751751+00:00. No GPU work ran. The collector reads source through AST; it never imports or executes cached wrappers. All counts below are static source counts, not dynamic GPU launches or kernel timings.

**No unrecorded Triton path was found in the current original-v2 requested-FlyDSL wrapper corpus.** All2972 static Triton callsites match a named lowered/reason diagnostic in the same attempt, model, suite, autotune mode, rank, world size, AOT graph and direction. Reused kernel names make this a many-to-many coverage match; it is not exact per-node or per-call diagnostic attribution.

| Source inventory | Count |
| --- | ---: |
| Wrapper paths | 1272 |
| Unique wrapper source hashes | 1266 |
| FlyDSL compiled definitions, including unused candidates | 272254 |
| Triton compiled definitions | 328 |
| FlyDSL static callsites | 341486 |
| Triton static callsites | 2972 |
| Unresolved launch references / parse errors | 0 / 0 |

The independent current-v2 audit used306 FlyDSL cache paths and1272 wrapper files. This supplemental audit additionally reads the old metadata-error/periodic-dump roots for attribution, giving1247 record files,360 requested-FlyDSL contexts and314 cache paths. Shared wrapper source is counted once, while all associated contexts are retained. No retained snapshot source or cache directory was missing.118 contexts have no wrapper source; all are capacity/setup/eager failures or still-running precompile phases. None is a completed training context.

## Historical cache ambiguity

There are610 unknown context-to-cache associations, all in historical metadata-error/periodic-dump roots. Every one shares a cache with another source fingerprint, and its wrapper file was written after all associated old record files. They are not evidence that those old attempts launched these kernels, and are not newly discovered missing FlyDSL lowerings. No old timings were used or combined.

| Historical model/context | Unknown associations |
| --- | ---: |
| Lightricks/LTX-Video (training_results_v2_initial_metadata_error) | 224 |
| Qwen/Qwen3-8B (training_results_v2_periodic_dump) | 144 |
| google/gemma-4-31B (training_results_v2_initial_metadata_error) | 120 |
| google/gemma-4-31B (training_results_v2_periodic_dump) | 120 |
| openai/gpt-oss-20b (training_results_v2_initial_metadata_error) | 2 |

## Native calls

Mandated ATen GEMM/convolution calls are separate from other native operations:137560 mandated callsites,1984 other ATen callsites, and4 other external callsites. Other operations include FFT, histogram, indexing, sort/top-k, RNG seed generation and view/shape operations. A native callsite may be a metadata-only view; it does not automatically mean a GPU launch or unsupported FlyDSL operation. The exact operation list is in native_operations.csv and native_calls.jsonl.

## Reproduction and scope

Collector SHA256: `a81aba9ae5320ed0482fd57d29c915bb400bbf45121ab4d6ed6371fd94617918`. Its17 CPU tests pass, including alias/branch/lambda/entrypoint checks and historical cache attribution. CLI accepts --results, --additional-results and --output; the actual read-only audit runs with run_env.sh -S and requires no Torch or GPU imports.

Definitions and candidates are listed separately from launch sites. NamedExpr/shared-source variants, function/method/import aliases, Runner.call and reachable helpers are retained. Source graphs behind eager graph breaks, arbitrary external libraries, or removed caches cannot be inferred as GPU launches by this audit. The original/recovery/full-v3 benchmark pipelines remain incomplete at this snapshot; this is not a final full-model coverage or performance claim.

See summary.json, wrappers.jsonl, record_contexts.jsonl and triton_diagnostic_matches.jsonl for exact provenance. Triton source bodies are preserved under triton_sources/. No frozen backend, original collector, benchmark process or shared dependency was changed.

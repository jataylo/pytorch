# GPT-OSS-120B standard mixed-mode GPU fault: CPU review

This is a CPU-only review of retained logs, result JSON, diagnostic JSONL, generated Python wrappers and source. No benchmark source, active harness, raw result, GPU runtime, GPU compiler, GPU core file or benchmark process was changed. The applicable PT2 debugging and CUDA stream/lifetime skills were read. Their GPU reproduction steps are explicitly deferred until the parent's benchmark phase is idle; the existing approved environment is reused.

## Observed failure and limits

The failed attempt is `openai/gpt-oss-120b`, batch8/global8, FSDP world4 on physical GPUs0,1,2,3, local batch2, BF16 computation/FP32 reduction and master parameters, standard tuning0 with `flydsl.simd_autotune_backend=True`. It has116,829,156,672 trainable parameters, config hash `555fa29e40643e7668ba56aba2b758f3f8fa041360d1dce64e304c710c1f85d9`, source fingerprint `6fb45422d9e0f72c768669b7f9e3879c19655d124248cbae62ba17c39c73255e`, and attempt `1794307806144290748`.

The hard failure evidence is a **null-address fault in `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)`**, reported by ROCm on physical GPU0. This is not an unsupported FlyDSL lowering, nor an allocator OOM exception. The following watchdog failure is all-gather sequence6089 (`_ALLGATHER_BASE`, input803,270,048 elements, output3,213,080,192 elements), in FSDP parameter unsharding before a layer forward. The recorded collective stack ends at the worker's timed `step()` at line470.

All four rank records reached phase`timing`, after a successful first compiled step (17.81seconds), initial finite loss/sampled-gradient checks, five regular warmups, and30 additional isolated timing warmups. All have eight compiled graphs before timing. There is no completed timing sample or final timing-recompile counter for this failed attempt. The result stays `process_failed`; the rank1/2/3 `started` files must not be invented as successful terminal records.

Every one of the four rank diagnostic files has159 events, all phase`warmup`:55lowered events,51backend-candidate events,49backend-selection events,4selection-skip events. There are49unique candidate names and49unique selected names, with none lacking a recorded choice. Rank0 selected43FlyDSL and6Triton; ranks1/2/3 selected40/9,41/8,43/6. Per-rank differences are legitimate independent timing choices, not proof of an error. The last recorded decision is a completed FlyDSL softmax-backward selection. There are no new selection records during timing. Combined with the faulting collective kernel and timed collective stack, the evidence locates the observed failure in **steady timed execution**, not a documented in-progress candidate comparison. It does not exclude earlier memory corruption or an unrecorded asynchronous cause.

Earlier at10:56:15, HIPCachingAllocator printed failed3,214,934,016-byte allocation attempts on all four ranks while only about2.60–2.74GB was free. The local allocator source at3935 prints this warning on a failed `hipMalloc`, clears the error and may retry. The process then continued, completed warmup and began timing at10:59:37.602 UTC. These earlier memory-pressure warnings are retained, but they neither classify the later HSA fault as allocator OOM nor establish that allocator pressure caused it. The GPU fault's exact timestamp is not printed; the 11:31:52 watchdog says the failed collective had run for30minutes, suggesting an enqueue around11:01:52. That approximate time is an inference, not an exact HSA-fault timestamp.

The other three configurations completed actual full training at this same world4/config: native standard5507.330ms, native expanded5472.140ms and mixed expanded5202.149ms. All are training-only/finite-check results without an eager numerical oracle. Their success narrows the observed problem but does not prove the mixed standard selector is the root cause.

## Selector binding, stream and lifetime review

The49rank0 selector definitions in11retained AOT wrapper sources were checked by AST. All49 alternatives have identical lambda parameter lists, forward the supplied `stream` keyword in both backend calls, and mark the exact named`in_out_ptr`/`out_ptr`indices as mutated. Both direct native and FlyDSL call sources, mutated indices, parameter lists, wrapper hashes and line numbers are saved in`selector_binding_review.json`. This establishes those source properties only; it is not a proof of all runtime tensor aliasing or descriptor bounds.

The wrapper obtains its raw stream through`get_raw_stream(0)`, which is imported from`torch._C._cuda_getCurrentRawStream`, then passes that stream to kernel launchers. The FlyDSL binary loader passes that raw stream through to the static/fast C++ launcher. It keeps loaded modules in the global`_LOADED`list. `BackendSelector`holds the alternative callable objects and replaces its instance`.run`with the selected callable; it does not retain the scratch tensor list in object state. The tuning scratch storages and temporary copies are local to`pick_fastest`. On the inspected source there is no demonstrated dropped-stream argument, code-object lifetime loss, persistent19GBscratch attachment or mismatched selector parameter list.

FSDP's own unshard/post-backward streams and their synchronization/recording remain unchanged shared PyTorch code. A collectives-stream lifetime bug, earlier SIMD write corruption, a Triton-selected data-dependent indexing problem, or an RCCL/runtime/hardware fault cannot be distinguished by the retained sources alone. No speculative `record_stream()`change or production workaround is proposed from this CPU review.

## Separate proven cold-cache defect

`pick_fastest`currently loads a cached choice with:

```python
with open(path) as f:
    best = next(m for m in mains if m.__name__ == f.read().strip())
```

A cached second/non-first winner is consumed while testing candidate0. Candidate1 then compares its name toEOF, so the generator raises`StopIteration`and falls through to re-tuning. Candidate0cache hits work. This is a concrete cached-choice reuse defect shared by SIMD tuning and backend selection; it is separate from the NCCL fault during steady timed execution, where the selector callable was already chosen during warmup.

The minimal isolated fix reads the winner once before iterating. `cache_read_once.patch` contains only that production hunk plus a focused CPU regression test. It has **not been applied**. A read-only`git apply --check`against the frozen source passed. The test executes the exact full`pick_fastest`function extracted byAST with a stub Torch module, CPU launchers, a temporary choice file and captured diagnostics; it performs no GPU calls. Existing source: cached first winnerPASS, cached second winnerFAIL. Isolated fixed source: bothPASS. Formal common_utils fixtures ran two cases: original exit1(one error), fixed exit0(two passes). The original expected failure and fixed logs are preserved separately.

This patch should be reviewed/applied only in a new isolated source revision after active frozen benchmark work, with fresh source hashes and a standalone cache test. It does not justify changing or reinterpreting already measured performance numbers. It should not be advertised as a fix for the null-address collective fault.

## Preserved evidence

- `evidence.json`: seven current/rank result summaries with hashes, four rank diagnostic summaries/selections,11rank0 wrapper hashes, exact log lines and source hashes.
- `selector_binding_review.json`:49selector definitions, argument/mutation/stream checks and calls.
- `cache_read_once_cpu_controls.json`: original versus isolated-fixed control outcomes.
- `cache_read_once.patch`: unapplied minimal production/test patch.
- `cache_regression_original.log` and`cache_regression_isolated_fixed.log`: formal two-case regression runs.
- `REPRODUCTION_PLAN.md`: bounded post-benchmark GPU investigation plan; nothing in it has been launched.

The core`/dockerx/pytorch-flex/agent_space/simd_bench/gpucore.3897880.gpu`is preserved unchanged at185,313,709,376bytes. Its first64KiBsha256 is`bcc84020652ad1697a68455933264c239e1430170212497d6ecbc248d0c45155`; the full185GBfile was deliberately not read or hashed during active GPU benchmarking. The evidence JSON retains stat/header metadata and does not pretend that a header digest is a full-file digest.

**Conclusion:** the observed failure is a timed RCCL all-gather GPU fault with unproven root cause. The new cache-read-once defect is proven independently and has an isolated tested patch, but it does not explain this timed fault. A bounded reproduction is required after active jobs finish. No frozen source or raw result has changed.

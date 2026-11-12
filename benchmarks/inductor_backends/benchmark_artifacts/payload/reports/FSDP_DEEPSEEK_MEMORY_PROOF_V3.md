# Verified DeepseekV4 FSDP storage bound

The current full architecture cannot fit the current FP32-master/BF16-compute/FP32-reduction FSDP protocol on eight MI355X GPUs, independently of CPU initialization speed. This is a model/protocol capacity limitation, not a missing FlyDSL lowering. It does not claim that a different precision/offload/sharding protocol is impossible.

## Actual model and partition

The official native DeepseekV4ForCausalLM at DeepSeek-V4-Flash-0731 revision7872f01b1d1fe23eabc4c98b48bffcef5a386062 has284332230231 unique trainable parameters in1242 parameter tensors. No parameter aliases were found. The actual `_no_split_modules` policy gives43 independent DeepseekV4DecoderLayer handles plus the root handle. Root owns separate embedding and LM-head weights; they are not tied. Every verified decoder participates in backward. The capacity decision does not assume that gradient completion follows reverse ModuleList order. Each block invokes packed expert tensors: selecting a few experts still produces dense gradients for the packed parameter storage. FSDP makes these parameters differentiable views of one flat parameter, so the flat gradient has the handle's full extent even if some elements are zero. No checkpointing, CPU offload, custom communication hook, optimizer-in-backward, or early gradient release is used in the frozen worker.

`fallback_fixes/agent_space/deepseek_flash_fsdp_meta_counts.json` records the exact actual metadata, parameter ownership, wrapping construction order, source/config hashes and padded counts. It was constructed on meta under64GiB RLIMIT_AS, using under1GiB RSS, with no GPU initialization. The real worker casts rank0 weights and rank>0 meta parameters to FP32 before FSDP; the constructor's initial BF16 dtype was not treated as the master dtype.

## Order-independent conservative bound

Let W>=2, n_i be the actual unique parameter count of decoder handle i, and g_i=4*floor(n_i/W). Let P be the sum of FP32 master-shard byte counts, rounded down separately for every handle including root. Let G=sum_i(g_i), over decoders only.

Consider whichever decoder j reaches its last gradient cast/reduction event last. Every other decoder retains either its completed FP32 gradient shard or an unreduced full gradient buffer. A full BF16 buffer has2*n_i bytes, which is at least4*floor(n_i/W) for W>=2. A full FP32 buffer is also at least that large. Thus the other handles collectively retain at least G-g_j bytes, regardless of completion order or communication overlap. Their storage persists until optimizer.step; no optimizer-in-backward or early gradient release is used.

At the last decoder's cast, the old BF16 and new FP32 full gradients coexist, adding6*n_j bytes. At its reduction, the FP32 input and distinct FP32 output shard coexist, adding at least4*n_j+g_j bytes. Therefore guaranteed bounds are:

```
cast   >= P + G + min_i(6*n_i - 4*floor(n_i/W))
reduce >= P + G + 4*min_i(n_i)
```

The capacity decision uses the maximum of these two guaranteed bounds. Expected reverse-order stages are retained only under expected_reverse_order_diagnostics, explicitly marked used_for_capacity_decision=False. No compiler/autograd completion ordering is required.

For actual world8, discard all alignment/world padding and round each shard DOWN. Exclude all root gradients, all full parameters, activations, buffers, and allocator/communication workspaces:

| Guaranteed component | Bytes | GiB |
| --- | ---: | ---: |
| P: all FP32 master shards |142166114984|132.402512230|
| G: all decoder FP32 gradient shards |141636549288|131.909315743|
| Minimum last-decoder cast increment above G |36170290060|33.686207663|
| **Order-independent cast bound** |**319972954332**|**297.998035636**|
| **Order-independent reduction bound** |**310108329768**|**288.810888089**|

The minimizing handle has6576416374 parameters (the same minimum occurs in layers0/1). This is a minimization over all possible last decoders, not a claim that layer0 must finish last. Recorded physical GPU capacity is309220868096bytes,287.984375GiB. Both guarantees exceed that capacity before any excluded costs. Actual meta recomputation gives1081.746373GiB/rank at world2 and559.247481GiB/rank at world4 as well.

## Why these allocations coexist

- Frozen `torch/distributed/fsdp/_flat_param.py`1257 retains the full-precision master shard in `_local_shard`. During computation, `_use_unsharded_flat_param`1492 installs the BF16 full parameter and `_use_unsharded_views`1959 exposes differentiable views; it does not eliminate the original master shard.
- Frozen `_runtime_utils.py`745-779 first reshards/frees the current full parameter, then saves `autograd_computed_grad = flat_param.grad.data`. It converts the gradient to the configured FP32 reduce dtype and calls `_reduce_grad` while that old BF16 tensor remains referenced. Thus the BF16 and FP32 full-gradient storages coexist. The proof does not also count the already-freed full parameter.
- `_reduce_grad`831-886 retains the FP32 input and `_get_reduce_scatter_tensors`889-905 allocates a distinct `empty_like` output shard. Padding, when necessary, adds storage; it cannot reduce this lower bound.
- `_accumulate_sharded_grad`908-928 stores the completed FP32 shard in each handle's `_saved_grad_shard`. `_cast_grad_to_param_dtype`1030 restores master dtype when needed. These different handles' gradient storages do not alias their master shards and persist until the end-of-backward optimizer path. If a reduction is still in flight instead of completed, its full input/output storage supplies at least the same conservative shard term. Frozen train_worker_v2.py calls optimizer.step only after loss.backward completes.
- MixedPrecision API117-141 defines BF16 forward/backward parameters and FP32 sharded parameters outside computation, with full-precision optimizer gradients by default. The worker explicitly requests BF16 param dtype and FP32 reduce dtype, use_orig_params=True and sync_module_states=True.

`test_deepseek_meta_init_feasibility_v3.py` has9 passing bounded CPU controls. They include an actual official tiny DeepseekV4 packed-expert forward/backward using only expert0: the packed gradient has full shape, unused-expert zeros and BF16 dtype. The same actual gradient is retained while a separate FP32 cast is allocated. Additional controls call the actual FSDP reduction tensor helper to prove separate input/output storages and actual accumulation helper to prove retained, separate FP32 gradient shards. These tests establish storage semantics; they are not GPU timings or a substitute for the full model metadata.

## Versioned helper and integration

`fsdp_memory_bound_v3.py` recomputes the bound from the actual FP32 meta model for any world size>=2. It accepts only the verified native class/module/source hashes and topology, exact FP32-master/BF16-compute/FP32-reduce FULL_SHARD protocol, unique all-trainable meta parameters, expected wrapping boundaries and no checkpointing/offload. Unknown models/protocols returnNone. It records all proof terms and official model/FSDP source hashes. No parameter count is hardcoded. The bound is independent of decoder completion order. Free memory below the guaranteed bound yields an explicit insufficient-capacity reason; free memory above the bound is only absence of this proof, not a claim that training fits.

Future versioned workers should call `meta.to(spec.param_dtype)` before this helper (metadata-only), then pass the actual per-rank capacity/free-memory list after their existing all_gather. Apply equally to all four variants before deleting meta or constructing real CPU weights. Original source and the active frozen run must remain untouched. Full2/4/8 placement evidence is in `fsdp_memory_bound_deepseek_flash0731_v3.json`. The updated helper has23 passing CPU tests. Its new controls use an actual tiny native three-decoder meta model, enumerate all6 completion orders and all9 combinations of prior shard/full-BF16/full-FP32 states per world size, and check both guarantees against deduplicated actual CPU tensor storage at cast and the actual FSDP reduce-tensor helper. These tests cover162 order/state/world combinations in addition to the nine existing initializer/gradient lifetime controls. The prior order-assumed helper/proof/artifacts are archived with an _order_assumed suffix.

## Actual first distributed attempt

The first FlyDSL0 world8 attempt failed after1416.9seconds. Its log has explicit GPU allocator OOMs on all eight ranks before the final NCCL watchdog HIPLaunchOutOfResources abort. Ranks0/1 failed a13161725952byte allocation; other ranks failed3298820096byte allocations. This was not the30minute process-group timeout. The final asynchronous NCCL error alone would hide the preceding capacity evidence; the chronology does not prove its precise cause. No FlyDSL-generated-kernel traceback was identified here.

## Initialization and timeout assessment

A future90minute process-group timeout can prevent healthy long CPU initialization from timing out peers; the external watchdog should allow additional time beyond that for loading, synchronization, compilation and training. It does not fix the proven memory bound or the observed OOMs.

Blockwise rank0 GPU initialization using official `_init_weights` is technically possible for a validated model, with memoized parameter/buffer aliases and correct nested HF initializer ownership. FSDP callback order is breadth-first while official HF initialization is depth-first; GPU and CPU RNG sequences also differ. Thus it requires a separately documented deterministic initialization protocol across all four variants and cannot preserve bitwise original CPU-initialized weights merely by using the same seed. Current native DeepseekV4 meta construction skips initializers and leaves module `_is_hf_initialized` unset (CPU-tested); remote HF implementations may differ. No GPU initialization implementation was added because it cannot make this current protocol fit.

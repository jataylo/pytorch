# Qwen4Exp single-embedding capacity guard

New helper: fsdp_large_embedding_bound_v3.fsdp_large_embedding_memory_bound(model, world_size, free_bytes_per_rank=...). Unknown architectures, source versions, aliasing, shapes or protocols return None. It performs no GPU queries or tensor materialization. Existing helpers and original workers are untouched.

The helper recognizes only the exact official Qwen4ExpForConditionalGeneration class from the installed, hashed modeling source. It requires the actual177392830576 unique FP32 meta parameters, no shared objects/storages, and the official dense trainable contiguous FP32 meta embedding at model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight, shape[320001536,160]. The exact official decoder/PLE classes and native nn.Embedding are checked. Sparse/max-norm embeddings, gradient checkpointing, CPU offload, other precision/reduction settings, non-FULL_SHARD, non-use_orig_params or low-precision gradients are unsupported by this proof.

The conservative decision counts ONLY n=51200245760 embedding elements, independent of other handles or backward order:

```
BF16 full dense gradient           2*n
FP32 full reduction cast           4*n
persistent FP32 master shard       4*floor(n/world_size)
bound                             6*n + 4*floor(n/world_size)
```

Any FSDP flat handle owning the embedding has M>=n elements and a persistent FP32 shard of ceil(M/world_size) elements (padding only increases storage). Thus its master storage is at least4*floor(n/world_size), even when the embedding is wrapped alone. This counts storage size, not which embedding indices happen to reside on a rank. No other master shard or gradient is counted.

At world8 the bound is332801597440bytes=309.945640564GiB, exceeding actual287.984375GiB device capacity. At world4 and2 it is333.787612915GiB and381.471557617GiB. The earlier368.708642058GiB audit included all model master shards; that stronger topology-specific diagnostic remains historical evidence, while the NEW helper uses this simpler single-embedding bound.

Native runtime source _runtime_utils.py:756 saves autograd_computed_grad=flat_param.grad.data; line764 allocates flat_param.grad.to(handle._reduce_dtype) while retaining the old BF16 gradient. The distinct FP32 cast and original BF16 storage coexist. The verified _flat_param.py retains persistent FP32 master storage. These runtime files, the official Qwen modeling source, and native nn.Embedding implementation are source-hashed in the returned proof.

The official PLE constructor creates dense nn.Embedding, and its forward uses that weight. Native CPU control tests actually run BF16 embedding backward with only two selected rows: the resulting gradient has full embedding shape, BF16 dtype, dense storage and zero unused rows. The control then retains it while making the FP32 cast and verifies distinct storage pointers and combined6*n bytes. Shape/handle-refinement checks confirm the own-master inequality. This is a source-backed storage lower bound, not a measured GPU peak or evidence that any model execution succeeded.

Thirty-three common_utils CPU tests pass, including actual full official meta construction under64GiB RLIMIT_AS and offline cache-only loading, world2/4/8 proofs, source mismatch, unknown protocols/world sizes/classes, alias/shape/dtype/stride/trainability/sparse guards, malformed capacities, and native gradient/cast lifetime. No GPU initialization, checkpoint or full real model weights were used.

Actual helper proof output: fsdp_large_embedding_bound_qwen4exp_v3.json. CPU test log: test_fsdp_large_embedding_bound_v3.log. The helper returns per_rank_lower_bound_bytes, all three proof terms, exact embedding/model identity, actual source hashes, and (when capacities are supplied) insufficient_capacity plus explicit reason. A lower bound below capacity never proves a model fits.

## Forward-reachability guard audit

A concrete same-parameter-count counterexample existed before the config guard: changing text_config.num_hidden_layers from48 to1 leaves all allocated parameters unchanged, while TextModel.forward iterates only layers[:num_hidden_layers] and bypasses the second decoder's PLE. The helper now requires exact configSHA d6ea711f30719d348cbd790808f78c63f8358abf46f2b3a46b3e958da06519ab, calculated with the worker's exact hashlib.sha256(json.dumps(model.config.to_dict(),sort_keys=True,default=str).encode()) algorithm. It also requires the exact official TextModel,48 actual layers, and identity between the runtime text-model config and outer config.text_config, preventing a detached runtime config from bypassing the outer hash. The actual cached official model satisfies these checks. Three added negative CPU cases retain the same parameter count while reducing executed layers, disabling PLE configuration, or replacing the runtime config; all are rejected. The final CPU suite has33 passing tests.

The verified forward chain is:

1. common.py:1374 constructs positive-size text input_ids and all-ones attention_mask; multimodal augmentation retains text and adds placeholders/pixels. _disable_cache at1411 sets use_cache=False. No generation/decode-only route or model-side checkpointing is selected.
2. Qwen4ExpForConditionalGeneration.forward around2505 calls self.model and computes LM logits from its returned hidden states.
3. Qwen4ExpModel.forward around2317 prepares ple_input_ids from the original token IDs when config.text_config.ple_layer_ids=[2], then calls language_model around2368 with those IDs and the embedded multimodal input.
4. TextModel.forward around1477 iterates all48 configured layers. Decoder layer1 was constructed with self.ple non-None because layer_idx+1=2 is in ple_layer_ids. Its forward around1277 unconditionally adds self.ple(...) when that object is present.
5. PLE.forward at1236 unconditionally calls ple_embedding, feeds its result to BOTH key_proj and value_proj, and adds the gated value plus convolution to the decoder hidden states. A zero-valued mask or gradient still produces a dense full-shape embedding gradient; it does not change sparse=False or detach that graph.
6. NGramEmbedding.forward at1174 calls the single dense nn.Embedding. There is no input-ID predicate that skips this call. This output flows through the remaining decoder stack and LM head to logits. Worker objective around224 chooses logits and computes their float squared mean; backward is enabled and the embedding weight requires gradients.

Source/config/runtime guards therefore cover the identified skip predicate under the fixed benchmark training protocol. Unknown future configuration digests return None, rather than extrapolating the capacity decision. No real full-model forward or GPU run was performed by this audit.

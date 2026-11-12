# Opt-in FSDP state-only initialization recovery

The original worker, backend sources, frozen tests and active jobs are unchanged. The recovery generators and future controllers now include this opt-in path and its mandatory reserved probe. GPU validation remains pending.

The retained fsdp_probe.py obtains an independent eager gradient reference before compiled comparisons. Its two/eight-GPU logs demonstrate that recipe works, not that eager backward is a requirement. Earlier DistillGPT2 timeout stacks in validation_fsdp_v2 point to joint_graph.lazy_init -> attention pattern initialization under compilation. Keep the existing explicit joint_graph.lazy_init prewarm.

In this source checkout public FSDP.check_is_root (fully_sharded_data_parallel.py:542) calls _is_fsdp_root -> _lazy_init (_runtime_utils.py:100-143). It initializes all nested FSDP root flags, streams, buffer casts, execution-order metadata and parameter-handle attributes without executing the model. The new helper initialize_fsdp_state_for_compile(model) requires an actual root FSDP object and initialized process group, calls that public method, synchronizes and barriers. It does not execute forward/backward or alter parameters, precision, inputs or architecture.

A no_grad model forward can also initialize FSDP, but still consumes forward activation/workspace memory and populates execution-order state. The state-only public method is the smaller opt-in candidate. First backward normally completes execution-order bookkeeping (_post_backward_final_callback calls next_iter); state-only initialization leaves first forward/backward to the normal compiled training path. Do not fake those bookkeeping transitions or invoke private cleanup callbacks manually.

## Integration

Use BENCHMARK_FSDP_STATE_INIT_ONLY=1 only for explicitly selected diffusion eager-initialization OOM recoveries, equally for both backends and both autotuning modes. Record phase fsdp_state_initialization and fsdp_initialization_mode=state_only. Retain first compiled-step finite loss, gradient and optimizer checks; no eager-reference claim follows from this recovery. The helper returns initialization method/rank/world metadata. Default protocol stays unchanged.

## CPU tests

The standalone test_fsdp_state_init_v3.py uses common_utils TestCase/run_tests. Nine cases cover call order, no model execution, invalid model/process-group/root rejection, initialization errors, synchronization errors barrier errors, rank-local cache setup before Torch import, and actual Inductor configuration keys. These mocked controls establish the helper contract; they do not validate real GPU FSDP state.

## Reserved two-GPU probe

Root must hold the existing exclusive entry and measurement locks. The probe takes no locks and performs no timing measurements. Run only after original work completes. Example argv, using the existing virtualenv Python:

```
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  /dockerx/pytorch-flex/agent_space/simd_bench/probe_fsdp_state_init_v3.py \
  --output /ABS/PATH/fsdp_state_init_probe.json
```

Root sets CUDA_VISIBLE_DEVICES=0,1 (or the reserved pair). For support validation retain the parent support_startup_v3 PYTHONPATH and BENCHMARK_SUPPORT_SOURCE_ROOT so spawned compiler imports use the support overlay. Launch the probe directly, not through run_overlay.py: it configures rank-local Inductor/Triton/Fly caches before its first Torch import and rejects a launcher that already imported Torch. Set a fresh TORCHINDUCTOR_CACHE_DIR base; the probe appends rankN. A baseline probe runs directly under its intended original import environment; the result records actual loaded module paths and hashes. Parent should compare these against the expected source version. Do not silently accept an earlier source-version probe.

All four modes construct fresh Linear/ReLU/Linear models with an additional root output-bias parameter, using supported baseline operations. Wrapping the Linear modules exercises both nested handles and the root's own parameter handle. Rank0 constructs real CPU weights; rank1 constructs meta weights, then FSDP broadcasts rank0 values. The separate eager reference is an ordinary non-FSDP GPU model with the same complete initial parameters. Inputs differ deterministically by rank. Reference gradients are reduced and averaged across ranks to match FSDP semantics. No FSDP eager forward/backward occurs before the first compiled training invocation. A forward hook verifies that state initialization executes no model forward and exactly one FSDP forward occurs for that first compiled invocation.

The probe compares the complete local loss, every full parameter gradient (via public summon_full_params with_grads), and every full parameter after one SGD update. Fly modes require nonzero actual FlyDSL kernels and zero Triton fallback; mode1 additionally requires actual autotuning. Any unavailable backend, compilation failure, incomplete parameter coverage, numerical mismatch, source drift or missing variant raises an error and torchrun fails. No silent skip or successful fallback marker is emitted.

Both native pointwise tuning switches follow each variant's tuning mode. GEMM and convolution use ATEN, conv_1x1_as_mm is false, and backend selection remains false. The tiny probe lowers the Fly tuning-size threshold to zero to exercise tuning; that threshold change applies only to the probe and is recorded in its configuration.

Output schema: rank0 writes --output with status=passed, world_size=2, four variants, ranks=[both rank payloads], source_hashes, config, config_sha256 and correctness-only protocol. Each rank also writes OUTPUT_STEM.rankN.json. The aggregate is written only after collecting and validating both ranks' complete four-mode results and matching sources/configuration. No latency or throughput claims are produced.

The toy should require far less than 1GiB live tensors per GPU, with runtime/compiler overhead additional. Allow a several-minute compile window; no measured duration is available yet. After the toy passes for the intended source version, the actual diffusion world8 recovery still needs all four modes to execute compiled training successfully. This helper cannot establish that a model fits memory or that compilation reduces its activation memory.

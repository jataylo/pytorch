# Optional expandable-allocator experiment: prepared, not scheduled

No allocator policy was applied to active jobs. No GPU probe, supplemental controller or model rerun was launched. Original/frozen sources, active workers/controllers, manifests and benchmark inputs remain unchanged. The final fixed-source benchmark will retain the existing allocator policy.

## Actual recovery evidence

The exact records reviewed are in training_results_v2_steady/raw, not the original training_results_v2/raw. allocator_fragmentation_snapshot_v3.json preserves each record path, source hash, mode, world size, phase, allocator error excerpt and parsed evidence.

Hunyuan's four state-only world8 recoveries all fail in compiled warmup. Fly0, Triton0 and Triton1 each request18.93GiB with approximately9GiB free and19.33GiB reserved but unallocated. These individual messages indicate plausible fragmentation, not proof of full training fit. Fly1 requests18.94GiB with6.63GiB free and2.66GiB reserved, which lacks nominal reclaimable headroom.

The subsequent source audit in fallback_fixes/agent_space/hunyuan_attention_memory_review/REVIEW.md explains why an allocator retry is not justified by the earlier messages alone: normalized Fly0/Fly1 Runner.call ASTs have identical allocation and deletion behavior. Fly1 progresses from the earlier attention-score allocation to a later softmax-output allocation, explaining the18.77GiB difference in live allocations exactly. The later point requires272.92+18.9433=291.8633GiB of live storage, above287.98GiB device capacity. This is not evidence of retained tuning scratch. Changing fragmentation does not establish that this later live-storage requirement fits.

All four Wan2.2 state-only world8 recoveries likewise fail compiled warmup, requesting29.39GiB with5.87-8.19GiB free and only2.66-4.94GiB reserved. None has sufficient nominal reclaimable headroom. These outcomes do not motivate a speculative allocator rerun.

The optional experiment below is retained for a future independently justified candidate. It is not part of the current benchmark protocol or a claim of successful recovery.

## New CPU-only evidence helper

allocator_fragmentation_v3.py provides fragmentation_evidence(record). It requires terminal-unmeasured status and a complete CUDA/HIP OOM stanza containing requested/free/allocated/reserved sizes. Positive or invalid milliseconds/ms fields are excluded. It preserves display rounding uncertainty; binary/decimal units are distinguished. The nominal test request>free and free+reserved>=request is explicitly only an indication, not a capacity proof or predictor of full training success. A generic allocator warning without the real size report is insufficient.

The same module provides require_expandable_allocation(snapshot,pointer,nbytes,device). It fails unless allocator_settings.expandable_segments is exactlyTrue, the tested allocation exceeds2MiB, and its entire contiguous interval lies inside one matching-device is_expandable segment AND one active_allocated block. This prevents environment-variable presence or unrelated expandable segments from masquerading as actual VMM use. Forty CPU parser/synthetic snapshot tests pass.

## Separate optional two-GPU probe

probe_expandable_allocator_v3.py is new and unlaunched. It must run directly under torchrun, not a wrapper that already imported Torch. Before Torch import it sets PYTORCH_ALLOC_CONF=expandable_segments:True and clears the higher-precedence PYTORCH_CUDA_ALLOC_CONF and PYTORCH_HIP_ALLOC_CONF aliases. CUDAAllocatorConfig.h explicitly checks CUDA then HIP then generic config; the probe also requires the actual native allocator and snapshot evidence. A fresh Inductor cache base is required and every rank receives distinct Inductor/Triton/Fly cache directories. Choose a fresh --output path to preserve earlier evidence.

Root must own the normal exclusive GPU scheduling/timing reservation and process-group timeout cleanup. The probe takes no locks and must never run alongside automatic recovery jobs. Example only, not executed:

```
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  /dockerx/pytorch-flex/agent_space/simd_bench/probe_expandable_allocator_v3.py \
  --output /ABS/FRESH/allocator_probe.json
```

For an overlay source version the parent must set support_startup_v3 PYTHONPATH and BENCHMARK_SUPPORT_SOURCE_ROOT before rank startup. The probe records actual loaded scheduler/Fly/runtime paths and hashes, native memory/config source hashes, ROCm version, architecture and per-variant configuration/source hashes. All ranks must agree on source/config versions.

Each rank tests F32 and BF16 all_gather_into_tensor, reduce_scatter_tensor and all_reduce using power-of-two2^21-element tensors. Every actual input/output allocation is larger than2MiB and receives live expandable-segment/block proof. Exact rank-dependent values verify all collective results.

It then tests eight backend/tuning/dtype variants: Triton/FlyDSL x autotune0/1 x F32/BF16. The pointwise expression(weight*input+input).float() uses small integer values; complete gradients and SGD updates at lr0.125 have exact representable results. It checks all output/gradient/updated-parameter elements against independent eager execution, performs an actual RCCL all_reduce on the compiled full gradient, and proves actual parameter/input/output/gradient allocations live in expandable blocks. Updates must be nontrivial. Fly usage must be nonzero with zero Triton fallback, Fly mode1 must exercise tuning, and native modes must emit Triton code. GEMM/conv remain ATEN, max_autotune remainsFalse, native pointwise tuning toggles match the selected mode. No performance measurements are made.

Per-rank JSON and a rank0 aggregate are written only after successful checks. Aggregate keys include status=passed, world_size=2, collectives(two dtype records), variants(eight records), ranks(two full payloads), source_hashes, config/config_sha256, allocator environment/backend, ROCm architecture/version and the correctness-only protocol. Missing backend support, allocation proof, collective results, exact numerics, tuning coverage or source agreement raises an error. There are no silent skips.

Seven additional CPU controls validate pre-import environment ordering/cache isolation, actual Inductor config keys and four-mode policy, and nontrivial exact F32/BF16 reference gradients/updates. These CPU controls do not validate GPU VMM or RCCL behavior. Logs: test_allocator_fragmentation_v3.log and test_probe_expandable_allocator_v3.log.

## Limits and decision

No supported current candidate remains established by the full evidence. Therefore no retry controller or declarative rerun manifest was created, and the allocator probe remains optional/unrun. If future evidence justifies an experiment, first reserve and validate the small probe for the intended source version; only then consider one explicitly versioned all-four-mode model rerun through the existing frozen runner, with fresh cache/result provenance, command/env/manifest hashes and normal attempt archives. That requires a new parent decision. No checkpoint, architecture, input or precision substitutions are proposed.

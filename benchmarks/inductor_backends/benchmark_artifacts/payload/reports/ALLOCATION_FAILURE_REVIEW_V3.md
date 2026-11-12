# GPU allocation status classification

CPU-only audit; no GPU work, retry launches, live controller/worker/manifest edits or frozen backend changes.

## Actual Qwen3.6-35B-A3B eligibility

All four latest records in training_results_v2_steady/raw/priority--Qwen_Qwen3.6-35B-A3B--bs8 were terminal world1 warmup errors on physical GPU3. The full architecture has35107181936 parameters and140428727744 FP32 parameter bytes.

- Triton0: HIPBLAS_STATUS_ALLOC_FAILED during hipblasCreate(handle) in backward.
- Triton1: literal CUDA out of memory,2.11GiB requested with28MiB free.
- Both Fly modes: previously identified missing OCML legacy-library linker error.

The active controller selects memory cases when ANY of the four records is capacity_skip or contains literal out of memory. Triton1 already makes this full group eligible for its normal all-four-mode FSDP retry. A manual supplemental retry controller would duplicate that work and was not created. A read-only snapshot of23 current steady model groups found no complete terminal group newly memory-eligible under the precise classifier but missed by the old predicate. This snapshot is not a final-run completeness claim.

## Bounded implementation

New allocation_failures_v3.py provides:

- is_gpu_allocation_failure(message): precise CUDA/HIP OOM strings and BLAS/DNN allocation status tokens, including HIPBLAS_STATUS_ALLOC_FAILED and CUBLAS_STATUS_ALLOC_FAILED.
- is_terminal_gpu_memory_failure(record): only terminal error/timeout/process_failed records with such error/traceback evidence, or an explicit capacity_skip. Started and measured outcomes do not trigger a retry from stale error text.

LaunchOutOfResources, too many resources requested for launch, unsupported BLAS operations, illegal memory accesses and CPU allocator messages are not classified as GPU allocation failures. Token boundaries reject strings that merely contain an allocation token as part of another identifier.

Unfrozen merge_steady_v2.py now classifies exact allocation statuses found in error OR traceback as memory instead of compiler_or_execution, while preserving existing generic out-of-memory and external-termination classification. No merge was run on live results. Root owns integrating the terminal-record helper into newly generated final controllers; active/fingerprinted run_steady_v2.py remains unchanged.

## CPU evidence

41 common_utils classifier tests pass. Updated test_merge_raw_failures_v3.py has18 passing CPU tests, including six real merge subprocesses on temporary fixtures that preserve HIPBLAS/CUBLAS error or traceback provenance, deduplicate raw+diagnostic records, classify allocation as memory, and leave launch-resource errors outside memory. Existing atomic four-mode adoption and external-failure tests remain passing.

Logs: test_allocation_failures_v3.log and test_merge_raw_failures_v3.log. No performance or successful-retry claim follows from these classifications.

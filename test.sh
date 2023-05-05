#!/bin/bash

# Define the test suites

test_suites=(
inductor/test_config
inductor/test_coordinate_descent_tuner
inductor/test_cpp_wrapper
inductor/test_cpu_repro
inductor/test_cuda_repro
inductor/test_cudagraph_trees
inductor/test_fused_attention
inductor/test_fx_fusion
inductor/test_indexing
inductor/test_kernel_benchmark
inductor/test_max_autotune
inductor/test_minifier
inductor/test_minifier_isolate
inductor/test_mkldnn_pattern_matcher
inductor/test_pattern_matcher
inductor/test_perf
inductor/test_profiler
inductor/test_select_algorithm
inductor/test_smoke
inductor/test_standalone_compile
inductor/test_torchinductor
inductor/test_torchinductor_codagen_dynamic_shapes
inductor/test_torchinductor_dynamic_shapes
inductor/test_torchinductor_opinfo
inductor/test_triton_wrapper
dynamo/test_after_aot
dynamo/test_aot_autograd
dynamo/test_backends
dynamo/test_comptime
dynamo/test_ctx_manager
dynamo/test_cudagraphs
dynamo/test_decorators
dynamo/test_dynamic_shapes
dynamo/test_export
dynamo/test_export_mutations
dynamo/test_functions
dynamo/test_global
dynamo/test_global_declaration
dynamo/test_higher_order_ops
dynamo/test_interop
dynamo/test_logging
dynamo/test_minifier
dynamo/test_misc
dynamo/test_model_output
dynamo/test_modules
dynamo/test_nops
dynamo/test_optimizers
dynamo/test_profiler
dynamo/test_python_autograd
dynamo/test_recompile_ux
dynamo/test_recompiles
dynamo/test_replay_record
dynamo/test_repros
dynamo/test_skip_non_tensor
dynamo/test_subgraphs
dynamo/test_unspec
dynamo/test_verify_correctness
)

# Loop through the test suites
for suite in "${test_suites[@]}"
do
    # Run the test suite and redirect the output to a file with the suite name
    PYTORCH_TEST_WITH_ROCM=1 CI=1 python test/run_test.py -v --continue-through-error -i "$suite" 2>&1 | tee "test_${suite//\//-}.log"
done

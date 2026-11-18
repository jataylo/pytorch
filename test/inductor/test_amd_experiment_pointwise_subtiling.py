# Owner(s): ["module: inductor"]

import ast
from unittest.mock import patch

from torch._inductor.codegen.amd_kernel_optimizations import (
    AMDCodegenSource,
    optimize_flat_pointwise_source,
    optimize_pointwise_source,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestAMDNativeCodegen(TestCase):
    def test_flat_subtiling_and_rejection(self):
        from amd_native_test_utils import pointwise_source

        from torch._inductor.runtime.hints import DeviceProperties

        device = DeviceProperties(
            "hip",
            0,
            256,
            "gfx950",
            warp_size=64,
            max_threads_per_block=1024,
            regs_per_multiprocessor=131072,
        )
        source = pointwise_source(device)
        result = optimize_flat_pointwise_source(source)
        self.assertIn("AMD_UNROLL", result)
        unsupported = source.replace(
            "tmp0 + tmp1", "tl.atomic_add(in_ptr0 + x0, tmp1, xmask)"
        )
        self.assertEqual(optimize_flat_pointwise_source(unsupported), unsupported)

    def test_shared_ast_parses_unmatched_source_once(self):
        source = "# Preserve formatting.\n@triton.jit\ndef kernel(value):\n    return value\n"
        with patch("ast.parse", wraps=ast.parse) as parse:
            context = AMDCodegenSource(source)
            for transform in (
                optimize_pointwise_source,
                optimize_flat_pointwise_source,
            ):
                transform(source, _context=context)
            self.assertEqual(context.emit(), source)
        parse.assert_called_once_with(source)


if __name__ == "__main__":
    run_tests()

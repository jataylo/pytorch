# Owner(s): ["module: inductor"]

import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch._dynamo.utils import guard_failures
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


class TestAMDNativeDynamic(TestCase):
    hw_classification = HardwareClassification.CUDA

    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("width", (127, 128))
    @parametrize("family", ("softmax", "softmax_backward", "norm"))
    def test_odd_dynamic_rows(self, device, dtype, width, family):
        if torch.version.hip is None:
            self.skipTest("AMD native kernel specialization")
        if family == "softmax":

            def fn(x):
                return torch.softmax(x.float(), -1)
        elif family == "softmax_backward":

            def fn(x):
                probability = torch.softmax(x.float(), -1)
                product = probability * x.float()
                return product - probability * product.sum(-1, keepdim=True)
        else:

            def fn(x):
                return torch.native_layer_norm(x.float(), (width,), None, None, 1e-5)

        self._check_dynamic_rows(
            fn,
            [(rows, width) for rows in (37, 39, 127, 129)],
            device,
            dtype,
            native=True,
        )

    @parametrize("dtype", (torch.float32, torch.bfloat16))
    def test_dynamic_pointwise_tails(self, device, dtype):
        if torch.version.hip is None:
            self.skipTest("AMD native kernel specialization")

        def fn(x):
            return torch.nn.functional.silu(x.float()).to(x.dtype)

        self._check_dynamic_rows(
            fn, [(size,) for size in (4099, 8197, 65537)], device, dtype, native=True
        )

    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("batched", (False, True))
    def test_dynamic_transpose(self, device, dtype, batched):
        if torch.version.hip is None:
            self.skipTest("AMD native kernel specialization")
        if batched:
            shapes = [(batch, 5, 37, 127) for batch in (2, 3, 7)]

            def fn(x):
                return x.permute(0, 2, 1, 3).contiguous()
        else:
            shapes = [(rows, 127) for rows in (37, 39, 127, 129)]

            def fn(x):
                return x.t().contiguous()

        self._check_dynamic_rows(fn, shapes, device, dtype, native=False)

    def _check_dynamic_rows(self, fn, shapes, device, dtype, *, native):
        torch._dynamo.reset()
        guard_failures.clear()
        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        sources = []
        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "triton.amd_further_kernel_optimizations": True,
                "triton.amd_use_gluon": False,
                "fx_graph_cache": False,
                "autotune_local_cache": False,
                "max_autotune_pointwise": True,
            }
        ):
            for index, shape in enumerate(shapes):
                value = torch.randn(shape, device=device, dtype=dtype)
                torch._dynamo.mark_dynamic(value, 0)
                expected = fn(value)
                if index == 0:
                    actual, sources = run_and_get_code(compiled, value)
                else:
                    actual = compiled(value)
                tolerance = 0.02 if dtype == torch.bfloat16 else 2e-4
                self.assertEqual(actual, expected, atol=tolerance, rtol=tolerance)
                self.assertEqual(
                    compiled(value), expected, atol=tolerance, rtol=tolerance
                )
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(sum(len(items) for items in guard_failures.values()), 0)
        source = "\n".join(sources)
        self.assertNotIn("@gluon.jit", source)
        self.assertNotIn("triton.experimental.gluon", source)
        if native:
            self.assertIn("'amd_native_family':", source)


instantiate_device_type_tests(TestAMDNativeDynamic, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()

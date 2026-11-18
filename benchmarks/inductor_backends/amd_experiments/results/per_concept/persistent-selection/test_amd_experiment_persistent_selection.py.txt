# Owner(s): ["module: inductor"]


import torch
from torch._inductor import config
from torch._inductor.codegen.amd_kernel_optimizations import (
    amd_persistent_reduction_eligible,
)
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestAMDNativeCodegen(TestCase):
    @parametrize("registers", [32768, 65536, 131072])
    def test_persistent_budget_uses_live_values_and_rows(self, registers):
        device = DeviceProperties(
            "hip",
            0,
            16,
            "gfx_future",
            warp_size=64,
            max_threads_per_block=1024,
            regs_per_multiprocessor=registers,
        )
        self.assertTrue(
            amd_persistent_reduction_eligible(
                device, 1, 128, input_count=1, reduction_count=1
            )
        )
        self.assertFalse(
            amd_persistent_reduction_eligible(
                device, 1, 65536, input_count=1, reduction_count=1
            )
        )
        self.assertFalse(
            amd_persistent_reduction_eligible(
                device, 1, 32768, input_count=8, reduction_count=8
            )
        )
        self.assertFalse(
            amd_persistent_reduction_eligible(
                device, 4096, 8192, input_count=3, reduction_count=3
            )
        )
        if registers == 131072:
            self.assertTrue(
                amd_persistent_reduction_eligible(
                    device, 8, 32768, input_count=1, reduction_count=2, compare=True
                )
            )
            self.assertFalse(
                amd_persistent_reduction_eligible(
                    device, 8, 32768, input_count=1, reduction_count=2, compare=False
                )
            )
        expected = registers >= 32768
        self.assertEqual(
            amd_persistent_reduction_eligible(
                device, 4096, 8192, input_count=3, reduction_count=3, compare=True
            ),
            expected,
        )


class TestAMDNativePersistence(TestCase):
    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("width", [2048, 8192])
    def test_persistent_and_looped_choices(self, device, dtype, width):
        if torch.version.hip is None:
            self.skipTest("AMD optimization")
        torch._dynamo.reset()
        rows = 1024 if width == 8192 else 37
        x = torch.randn(rows, width, device=device, dtype=dtype)

        def fn(x):
            return torch.softmax(x.float(), -1).to(x.dtype)

        with config.patch(
            {
                "triton.amd_persistent_selection": True,
                "triton.amd_persistent_reduction_autotune": True,
                "fx_graph_cache": False,
                "autotune_local_cache": False,
            }
        ):
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertEqual(
            actual,
            fn(x),
            atol=0.02 if dtype == torch.bfloat16 else 2e-05,
            rtol=0.02 if dtype == torch.bfloat16 else 2e-05,
        )
        source = "\n".join(code)
        self.assertIn("async_compile.multi_kernel", source)
        self.assertIn("triton_heuristics.reduction", source)
        self.assertIn("triton_heuristics.persistent_reduction", source)
        self.assertNotIn("experimental.gluon", source)


instantiate_device_type_tests(TestAMDNativePersistence, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()

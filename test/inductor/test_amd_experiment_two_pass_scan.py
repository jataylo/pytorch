# Owner(s): ["module: inductor"]

import torch
from torch._inductor import config
from torch._inductor.runtime.hints import amd_scan_parameters, DeviceProperties
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    largeTensorTest,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestAMDScanParameters(TestCase):
    @parametrize("wave", (32, 64))
    def test_resource_scaling(self, wave):
        device = DeviceProperties(
            "hip", 0, 256, "gfx_future", warp_size=wave, max_threads_per_block=1024
        )
        block, warps = amd_scan_parameters(device, 1, 262144)
        self.assertEqual(block, 32 * wave)
        self.assertEqual(warps, 8)
        self.assertLessEqual(
            (262144 + block - 1) // block, 4 * device.multi_processor_count
        )
        self.assertEqual(amd_scan_parameters(device, 64, 262144), None)
        self.assertEqual(amd_scan_parameters(device, 1, 127), None)
        self.assertEqual(amd_scan_parameters(device, 1, 2**28), None)

    @parametrize("backend", ("cpu", "cuda", "xpu"))
    def test_other_backend(self, backend):
        self.assertEqual(
            amd_scan_parameters(DeviceProperties(backend, 0, 256, "other"), 1, 262144),
            None,
        )


class TestAMDTwoPassScan(TestCase):
    @parametrize("shape", ((1, 131073), (1, 262145), (8, 1048576), (1, 4194304)))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    def test_exact_input_reference(self, device, shape, dtype):
        if torch.version.hip is None:
            self.skipTest("AMD native scan lowering")

        def fn(x):
            return x.float().cumsum(-1)

        torch.manual_seed(1701)
        x = torch.randn(shape, dtype=dtype, device=device)
        with config.patch(
            {
                "triton.amd_two_pass_scan": True,
                "fx_graph_cache": False,
            }
        ):
            compiled = torch.compile(fn, fullgraph=True, dynamic=False)
            actual, source = run_and_get_code(compiled, x)
            self.assertIn("j < chunk", "\n".join(source))
            self.assertEqual(
                actual, x.double().cumsum(-1).float(), atol=0.0002, rtol=0.0001
            )
            for kind in ("constant", "offset", "nonfinite", "finite_overflow"):
                value = torch.full(
                    shape,
                    4096.0 if kind == "offset" else 1.0,
                    device=device,
                    dtype=dtype,
                )
                if kind == "finite_overflow":
                    value.zero_()
                    value[:, :2] = torch.finfo(dtype).max
                    value[:, 2] = -torch.finfo(dtype).max
                if kind == "nonfinite":
                    value[:, 0] = float("inf")
                    value[:, 2048] = -float("inf")
                    value[:, -1] = float("nan")
                self.assertEqual(
                    compiled(value),
                    value.double().cumsum(-1).float(),
                    atol=0.0002,
                    rtol=0.0001,
                    equal_nan=True,
                )

    @parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
    def test_strides_and_fusion(self, device, dtype):
        if torch.version.hip is None:
            self.skipTest("AMD native scan lowering")
        x = torch.randn((262146, 2), dtype=dtype, device=device).t()[:, ::2]

        def fn(x):
            return x.cumsum(-1, dtype=torch.float32) * 0.125 + 1

        with config.patch(
            {
                "triton.amd_two_pass_scan": True,
                "fx_graph_cache": False,
            }
        ):
            actual, source = run_and_get_code(
                torch.compile(fn, fullgraph=True, dynamic=False), x
            )
        self.assertIn("j < chunk", "\n".join(source))
        self.assertEqual(
            actual, x.double().cumsum(-1).float() * 0.125 + 1, atol=0.0002, rtol=0.0001
        )

    @largeTensorTest("9GB", device="cuda", inductor=True)
    @parametrize("layout", ("row_stride", "column_stride"))
    def test_large_storage_offsets(self, device, layout):
        if torch.version.hip is None:
            self.skipTest("AMD native scan lowering")
        width = 131073
        stride = 1 << 30
        storage = torch.empty(2 * stride + width, device=device)
        x = (
            storage.as_strided((3, width), (stride, 1))
            if layout == "row_stride"
            else storage.as_strided((1, width), (width * 16384, 16384))
        )
        self.assertGreater(2 * stride, torch.iinfo(torch.int32).max)
        x.fill_(1)
        if layout == "row_stride":
            x[1].fill_(2)
            x[2].fill_(3)

        def fn(x):
            return x.cumsum(-1)

        with config.patch(
            {
                "triton.amd_two_pass_scan": True,
                "fx_graph_cache": False,
            }
        ):
            actual, source = run_and_get_code(
                torch.compile(fn, fullgraph=True, dynamic=False), x
            )
        self.assertIn("INDEX_DTYPE : tl.constexpr = tl.int64", "\n".join(source))
        expected = (
            torch.arange(1, width + 1, device=device).float()[None, :]
            * torch.arange(1, x.shape[0] + 1, device=device).float()[:, None]
        )
        self.assertEqual(actual, expected)

    @parametrize(
        "setting",
        (
            "disabled",
            "integer",
            "outer_axis",
            "deterministic",
            "batch_invariant",
            "torch_deterministic",
            "small",
            "dynamic",
        ),
    )
    def test_scope(self, device, setting):
        if torch.version.hip is None:
            self.skipTest("AMD native scan lowering")
        shape = (1, 262144) if setting not in ("small", "outer_axis") else (37, 127)
        dtype = torch.int32 if setting == "integer" else torch.float32
        x = torch.ones(shape, device=device, dtype=dtype)
        axis = 0 if setting == "outer_axis" else -1

        def fn(x):
            return x.cumsum(axis)

        settings = {
            "fx_graph_cache": False,
            "triton.amd_two_pass_scan": setting != "disabled",
            "deterministic": setting == "deterministic",
            "batch_invariant": setting == "batch_invariant",
        }
        with (
            DeterministicGuard(setting == "torch_deterministic"),
            config.patch(settings),
        ):
            if setting == "dynamic":
                torch._dynamo.mark_dynamic(x, 1)
            actual, source = run_and_get_code(
                torch.compile(fn, fullgraph=True, dynamic=setting == "dynamic"), x
            )
            self.assertNotIn("j < chunk", "\n".join(source))
            self.assertEqual(actual, fn(x))


instantiate_device_type_tests(TestAMDTwoPassScan, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()

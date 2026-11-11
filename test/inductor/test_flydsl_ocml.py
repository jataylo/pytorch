# Owner(s): ["module: inductor"]
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codegen.flydsl import flydsl_simd
from torch._inductor.codegen.flydsl.flydsl_simd import FlyDSLKernel
from torch._inductor.codegen.flydsl.flydsl_utils import (
    _flydsl_runtime_unavailable_reason,
    _simd_arch_supported,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestFlyDSLOCMLNumerics(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.version.hip:
            self.skipTest("requires ROCm")
        reason = _flydsl_runtime_unavailable_reason()
        if reason is not None:
            self.skipTest(reason)
        if not _simd_arch_supported(torch.cuda.current_device()):
            self.skipTest("requires a supported CDNA device")
        self.enterContext(
            config.patch(
                {
                    "flydsl.simd": True,
                    "flydsl.simd_strict": True,
                    "flydsl.simd_autotune_backend": False,
                    "fx_graph_cache": False,
                }
            )
        )
        self.enterContext(
            patch.object(flydsl_simd, "_tune_table", return_value={"autotune": 0})
        )
        torch._dynamo.reset()
        counters.clear()

    @onlyCUDA
    @parametrize("mode", ["log1p", "expm1"])
    def test_small_values_and_vjp(self, device, mode):
        values = torch.tensor(
            [
                -0.9,
                -0.5,
                -1e-4,
                -1e-8,
                -1e-20,
                -1e-38,
                0.0,
                1e-38,
                1e-20,
                1e-8,
                1e-4,
                0.5,
                1.0,
                10.0,
            ],
            dtype=torch.float32,
        )
        reference_input = values.clone().requires_grad_(True)
        actual_input = values.to(device).requires_grad_(True)
        cotangent = torch.linspace(0.1, 1.0, values.numel())
        function = getattr(torch, mode)
        expected = function(reference_input)
        expected_gradient = torch.autograd.grad(expected, reference_input, cotangent)[0]
        actual = torch.compile(function, dynamic=False)(actual_input)
        actual_gradient = torch.autograd.grad(
            actual, actual_input, cotangent.to(device)
        )[0]
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=5e-6)
        self.assertEqual(actual_gradient.cpu(), expected_gradient, atol=1e-5, rtol=1e-4)
        self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_qwen_softplus_chunk_scan_and_vjp(self, device, dtype):
        def fn(projected_ba, log_rate, bias):
            _, activation = projected_ba.split([2, 2], dim=-1)
            activation = activation.reshape(2, 128, 32)
            decay = -log_rate.float().exp() * F.softplus(
                activation.float() + bias.float()
            )
            return decay.transpose(1, 2).reshape(2, 32, 2, 64).cumsum(-1)

        torch.manual_seed(2026)
        args = [
            torch.randn(
                (2, 128, 16, 4), dtype=dtype, device=device, requires_grad=True
            ),
            torch.linspace(0.01, 4.0, 32, device=device)
            .log()
            .to(dtype)
            .requires_grad_(True),
            torch.ones(32, dtype=dtype, device=device, requires_grad=True),
        ]
        reference = [value.detach().clone().requires_grad_(True) for value in args]
        cotangent = torch.randn((2, 32, 2, 64), device=device)
        expected = fn(*reference)
        expected_gradients = torch.autograd.grad(expected, reference, cotangent)
        generated = []
        original = FlyDSLKernel.codegen_kernel

        def observe(kernel, *args, **kwargs):
            source = original(kernel, *args, **kwargs)
            generated.append({"scan": kernel.has_scan, "log1p": "fdp.log1p(" in source})
            return source

        with patch.object(FlyDSLKernel, "codegen_kernel", new=observe):
            actual = torch.compile(fn, dynamic=False)(*args)
            actual_gradients = torch.autograd.grad(actual, args, cotangent)
        self.assertEqual(actual, expected, atol=1e-3, rtol=1e-4)
        for gradient, ref in zip(actual_gradients, expected_gradients):
            tolerance = 0.02 if dtype == torch.bfloat16 else 1e-3
            self.assertEqual(gradient, ref, atol=tolerance, rtol=tolerance)
        self.assertTrue(
            any(item["scan"] and item["log1p"] for item in generated), generated
        )
        self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)


instantiate_device_type_tests(TestFlyDSLOCMLNumerics, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()

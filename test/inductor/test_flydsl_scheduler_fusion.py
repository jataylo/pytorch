# Owner(s): ["module: inductor"]

from unittest import mock

import torch
from torch._dynamo.utils import counters
from torch._inductor import config, metrics
from torch._inductor.scheduler import (
    BaseScheduling,
    FusedMixOrderReductions,
    MixOrderReduction,
    NestedReduction,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestExistingMixedOrderFusion(TestCase):
    @parametrize("other_mixed", [False, True])
    def test_existing_group_uses_append_fusion(self, other_mixed):
        group = object.__new__(FusedMixOrderReductions)
        group.is_foreach = mock.Mock(return_value=False)
        group.fuse_with = mock.Mock(return_value=mock.sentinel.fused)
        other = object.__new__(FusedMixOrderReductions) if other_mixed else mock.Mock()
        other.is_foreach = mock.Mock(return_value=False)
        with (
            mock.patch.object(
                NestedReduction, "_is_dependent_reduction_pair", return_value=False
            ),
            mock.patch.object(
                MixOrderReduction,
                "are_mix_order_reductions",
                side_effect=AssertionError(
                    "An existing mixed-order group must not be nested in a new group"
                ),
            ),
        ):
            result = BaseScheduling.fuse(mock.sentinel.backend, group, other)
        self.assertIs(result, mock.sentinel.fused)
        group.fuse_with.assert_called_once_with(other)


class TestMixedOrderLayerNorm(TestCase):
    @parametrize("flydsl", [False, True])
    def test_layer_norm_backward_multiple_reductions(self, device, flydsl):
        if flydsl:
            from torch._inductor.codegen.flydsl.flydsl_utils import (
                _flydsl_runtime_unavailable_reason,
                _simd_arch_supported,
            )

            reason = _flydsl_runtime_unavailable_reason()
            if reason is not None:
                self.skipTest(reason)
            if not _simd_arch_supported(torch.cuda.current_device()):
                self.skipTest("FlyDSL requires a supported CDNA device")

        def fn(x, weight, bias, scale):
            y = torch.nn.functional.layer_norm(x, (768,), weight, bias)
            return torch.tanh(y + x * scale)

        torch.manual_seed(42)
        args = [
            torch.randn(shape, device=device, requires_grad=True)
            for shape in [(8192, 768), (768,), (768,), (768,)]
        ]
        reference = [x.detach().clone().requires_grad_(True) for x in args]
        cotangent = torch.randn((8192, 768), device=device)
        expected = fn(*reference)
        expected.backward(cotangent)
        torch._dynamo.reset()
        metrics.reset()
        counters.clear()
        with config.patch(
            {
                "triton.mix_order_reduction": True,
                "triton.cooperative_reductions": False,
                "flydsl.simd": flydsl,
                "fx_graph_cache": False,
                "flydsl.simd_autotune_backend": False,
            }
        ):
            actual = torch.compile(fn)(*args)
            actual.backward(cotangent)
        self.assertGreater(metrics.codegen_mix_order_reduction, 0)
        if flydsl:
            self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        for result, ref in zip(args, reference):
            self.assertEqual(result.grad, ref.grad, atol=2e-3, rtol=2e-3)

    @parametrize("flydsl", [False, True])
    def test_xcit_strided_layer_norm_partials(self, device, flydsl):
        if flydsl:
            from torch._inductor.codegen.flydsl.flydsl_utils import (
                _flydsl_runtime_unavailable_reason,
                _simd_arch_supported,
            )

            reason = _flydsl_runtime_unavailable_reason()
            if reason is not None:
                self.skipTest(reason)
            if not _simd_arch_supported(torch.cuda.current_device()):
                self.skipTest("FlyDSL requires a supported CDNA device")

        def fn(gradient, weight, activation, mean, rstd):
            normalized = (activation.reshape(12544, 768) - mean[:, None]) * rstd[
                :, None
            ]
            return (
                (gradient * weight).sum(-1),
                (gradient * normalized).sum(0),
                gradient.sum(0),
            )

        args = (
            torch.randn(12544, 768, device=device),
            torch.randn(768, device=device),
            torch.randn(16, 768, 784, device=device).transpose(1, 2),
            torch.randn(12544, device=device),
            torch.rand(12544, device=device) + 0.5,
        )
        expected = fn(*args)
        torch._dynamo.reset()
        metrics.reset()
        counters.clear()
        with config.patch(
            {
                "triton.mix_order_reduction": True,
                "triton.cooperative_reductions": False,
                "flydsl.simd": flydsl,
                "flydsl.simd_strict": flydsl,
                "flydsl.simd_autotune_backend": False,
                "fx_graph_cache": False,
            }
        ):
            actual = torch.compile(fn)(*args)
        self.assertGreater(metrics.codegen_mix_order_reduction, 0)
        self.assertEqual(actual, expected, atol=2e-3, rtol=2e-3)
        if flydsl:
            self.assertGreater(counters["flydsl_simd"]["mix_order"], 0)
            self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)


instantiate_device_type_tests(TestMixedOrderLayerNorm, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()

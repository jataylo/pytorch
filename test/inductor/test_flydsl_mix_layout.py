# Owner(s): ["module: inductor"]
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import sympy

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codegen.flydsl.flydsl_simd import FlyDSLKernel
from torch._inductor.codegen.flydsl.flydsl_utils import (
    _flydsl_runtime_unavailable_reason,
    _simd_arch_supported,
)
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
from torch.utils._sympy.functions import FloorDiv, ModularIndexing


@instantiate_parametrized_tests
class TestFlyDSLMixLayout(TestCase):
    def test_mixed_order_keeps_supported_row_layout(self):
        x = sympy.Symbol("xindex", integer=True, nonnegative=True)
        r = sympy.Symbol("rindex", integer=True, nonnegative=True)
        indices = [
            r + 1024 * x,
            r,
            ModularIndexing(x, 1, 49) + 49 * r + 50176 * FloorDiv(x, 49),
        ]
        kernel = SimpleNamespace(
            features=SimpleNamespace(contains_op=lambda name: False),
            tile2d=False,
            in_combo=False,
            is_reduction_kernel=True,
            xnumel=6272,
            rnumel=1024,
            rnumel_dyn=None,
            mix_order_reduction=True,
            rsplit_size=16,
            stride=1,
            interleaved=False,
            persistent=False,
            one_step=False,
            unroll=1,
            xunroll=1,
            _buffer_itemsizes=lambda: [2, 4],
            _root_expr=lambda value: value,
            _x_sym=lambda: x,
            _r_sym=lambda: r,
            _has_atomic=lambda **kwargs: False,
            _memory_roots=lambda **kwargs: indices,
            _has_gather=lambda: False,
            _row_inputs=lambda: 3,
            _r_pow2_dividing=lambda width: width,
            _pure_reduction=lambda: False,
            _reduction_types=lambda: ["sum"],
            _order_output_axes=lambda: None,
        )
        kernel._transposed_rows = partial(FlyDSLKernel._transposed_rows, kernel)
        kernel._mix_cols = partial(FlyDSLKernel._mix_cols, kernel)
        kernel._finalize_mix_order = partial(FlyDSLKernel._finalize_mix_order, kernel)
        with (
            patch(
                "torch._inductor.codegen.flydsl.flydsl_simd._num_cus", return_value=256
            ),
            patch(
                "torch._inductor.codegen.flydsl.flydsl_simd._tune_table",
                return_value={},
            ),
        ):
            FlyDSLKernel.finalize_indexing(kernel, indices)
        self.assertFalse(kernel.interleaved)
        self.assertEqual(kernel.stride, 1)
        self.assertTrue(kernel.persistent)
        self.assertEqual(kernel.tpr, 64)

    @parametrize(
        "persistent,stride,interleaved,tpr,split,reason",
        [
            (False, 1, False, 64, 16, "requires a persistent row"),
            (True, 8, True, 64, 16, "requires non-interleaved row groups"),
            (True, 1, False, 1, 16, "split spans less than a wave"),
        ],
    )
    def test_constraint_diagnostics(
        self, persistent, stride, interleaved, tpr, split, reason
    ):
        kernel = SimpleNamespace(
            rnumel_dyn=None,
            rsplit_size=split,
            persistent=persistent,
            stride=stride,
            interleaved=interleaved,
            tpr=tpr,
        )
        with self.assertRaisesRegex(RuntimeError, reason):
            FlyDSLKernel._finalize_mix_order(kernel)


@unittest.skipUnless(torch.version.hip, "requires ROCm")
class TestFlyDSLMixNumerics(TestCase):
    def setUp(self):
        super().setUp()
        reason = _flydsl_runtime_unavailable_reason()
        if reason is not None:
            self.skipTest(reason)
        if not _simd_arch_supported(torch.cuda.current_device()):
            self.skipTest("requires a supported CDNA device")

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_transposed_layer_norm_partials(self, device, dtype):
        def fn(gradient, weight, activation, mean, rstd):
            grad = gradient.float()
            values = activation.permute(0, 2, 3, 1).reshape(6272, 1024).float()
            normalized = (values - mean[:, None]) * rstd[:, None]
            return (grad * weight).sum(-1), (grad * normalized).sum(0), grad.sum(0)

        args = (
            torch.randn(6272, 1024, dtype=dtype, device=device),
            torch.randn(1024, device=device),
            torch.randn(128, 1024, 7, 7, dtype=dtype, device=device),
            torch.randn(6272, device=device),
            torch.rand(6272, device=device) + 0.5,
        )
        reference = fn(*args)
        torch._dynamo.reset()
        counters.clear()
        with (
            config.patch(
                {
                    "flydsl.simd": True,
                    "flydsl.simd_strict": True,
                    "fx_graph_cache": False,
                    "triton.mix_order_reduction": True,
                }
            ),
            patch(
                "torch._inductor.codegen.flydsl.flydsl_simd._tune_table",
                return_value={"autotune": 0},
            ),
        ):
            actual = torch.compile(fn, dynamic=False)(*args)
        self.assertEqual(actual, reference, atol=1e-3, rtol=1e-3)
        self.assertGreater(counters["flydsl_simd"]["mix_order"], 0)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)


instantiate_device_type_tests(TestFlyDSLMixNumerics, globals(), only_for=("cuda",))


if __name__ == "__main__":
    run_tests()

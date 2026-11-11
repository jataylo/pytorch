# Owner(s): ["module: inductor"]
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters
from torch._inductor import config, inductor_prims, ir
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.flydsl.flydsl_simd import FlyDSLKernel
from torch._inductor.codegen.flydsl.flydsl_utils import (
    _flydsl_runtime_unavailable_reason,
    _simd_arch_supported,
)
from torch._inductor.runtime.hints import ReductionHint
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


RUNTIME_UNAVAILABLE_REASON = _flydsl_runtime_unavailable_reason()
if RUNTIME_UNAVAILABLE_REASON is None:
    from torch._inductor.codegen.flydsl import flydsl_simd_runtime as fdp
else:
    fdp = None


@unittest.skipIf(
    RUNTIME_UNAVAILABLE_REASON is not None, RUNTIME_UNAVAILABLE_REASON or ""
)
@instantiate_parametrized_tests
class TestFlyDSLSoftmaxPairs(TestCase):
    def setUp(self):
        super().setUp()

        def maximum(a, b):
            return float("nan") if math.isnan(a) or math.isnan(b) else max(a, b)

        self.enterContext(
            patch.multiple(
                fdp,
                const=lambda value, kind: float(value),
                add=lambda a, b: a + b,
                sub=lambda a, b: a - b,
                mul=lambda a, b: a * b,
                maximum=maximum,
                eq=lambda a, b: a == b,
                exp=math.exp,
                where=lambda pred, a, b: a if pred else b,
                _derive=lambda value, hint: value,
                _masked_items=lambda items, kind, fill: [
                    value if mask is None or mask else fill for value, mask in items
                ],
                group_reduce=lambda op, value, *geometry: value,
            )
        )

    @parametrize("persistent", [False, True])
    @parametrize(
        "case", ["finite", "negative_infinity", "masked", "positive_infinity", "nan"]
    )
    def test_weighted_states(self, persistent, case):
        if case == "finite":
            items = [((-1000.0, 0.25), True), ((4.0, 0.5), True), ((-2.0, 0.125), True)]
            expected = (4.0, 0.5 + 0.125 * math.exp(-6.0))
        elif case == "negative_infinity":
            items = [((float("-inf"), 2.0), True), ((float("-inf"), 3.0), True)]
            expected = (float("-inf"), 5.0)
        elif case == "masked":
            items = [((float("nan"), 7.0), False), ((1.0, 2.0), False)]
            expected = (float("-inf"), 0.0)
        elif case == "positive_infinity":
            items = [((float("inf"), 1.0), True), ((0.0, 1.0), True)]
            expected = (float("inf"), float("nan"))
        else:
            items = [((0.0, 1.0), True), ((float("nan"), 1.0), True)]
            expected = (float("nan"), float("nan"))
        if persistent:
            actual = fdp.group_softmax_pair_twopass(items, 1, None, 1)
        else:
            actual = (float("-inf"), 0.0)
            for item in items:
                actual = fdp.online_softmax_pair_block(*actual, [item])
        for value, reference in zip(actual, expected):
            if math.isnan(reference):
                self.assertTrue(math.isnan(value))
            else:
                self.assertEqual(value, reference)

    def test_pair_merge_keeps_negative_infinity_weights(self):
        self.assertEqual(
            fdp.online_softmax_pair_merge((float("-inf"), 2.0), (float("-inf"), 3.0)),
            (float("-inf"), 5.0),
        )

    def test_partial_sum_has_no_lower_bound_hint(self):
        with (
            patch.object(fdp, "_group_reduce", return_value=(0.0, 0.125)),
            patch.object(
                fdp, "_derive", side_effect=lambda value, hint: value
            ) as derive,
        ):
            self.assertEqual(
                fdp.group_online_softmax(0.0, 0.125, 1, None, 1, partial=True),
                (0.0, 0.125),
            )
        self.assertEqual([call.args[1] for call in derive.call_args_list], ["rowmax"])


@instantiate_parametrized_tests
class TestFlyDSLSoftmaxCodegen(TestCase):
    @parametrize("persistent", [False, True])
    def test_tuple_codegen(self, persistent):
        kernel = SimpleNamespace(
            inside_reduction=True,
            reduction_cache={},
            tpr=1,
            stride=1,
            threads=64,
            uses_smem=False,
            persistent=persistent,
            max_components=1,
            _rmask=lambda: "rmask",
            _new_acc=lambda: "acc",
            body=IndentedBuffer(),
            compute=IndentedBuffer(),
            chunk_pre=IndentedBuffer(),
            chunk_post=IndentedBuffer(),
            post_loop_combine=IndentedBuffer(),
            _post=lambda source, count, dtype: [source, "sum"],
        )
        result = FlyDSLKernel.reduction(
            kernel,
            torch.float32,
            torch.float32,
            "online_softmax_reduce",
            ("value_max", "value_sum"),
        )
        self.assertIn("((value_max, value_sum), rmask)", kernel.compute.getvalue())
        if persistent:
            self.assertIn("group_softmax_pair_twopass", result[0])
        else:
            self.assertIn("online_softmax_pair_block", kernel.chunk_post.getvalue())
            self.assertIn("partial=True", result[0])


@unittest.skipIf(
    RUNTIME_UNAVAILABLE_REASON is not None, RUNTIME_UNAVAILABLE_REASON or ""
)
@instantiate_parametrized_tests
class TestFlyDSLSoftmaxPairIR(TestCase):
    @parametrize("persistent", [False, True])
    def test_masked_states_trace_to_valid_ir(self, persistent):
        from flydsl._mlir import ir as mlir
        from flydsl._mlir.dialects import func

        with mlir.Context(), mlir.Location.unknown():
            module = mlir.Module.create()
            with mlir.InsertionPoint(module.body):
                function = func.FuncOp("softmax_pair", mlir.FunctionType.get([], []))
                block = function.add_entry_block()
                with mlir.InsertionPoint(block):
                    fdp.kernel_begin()
                    items = [
                        (
                            (fdp.const(-2, "f32"), fdp.const(1, "f32")),
                            fdp.const(True, "b1"),
                        ),
                        (
                            (fdp.const(4, "f32"), fdp.const(0.5, "f32")),
                            fdp.const(False, "b1"),
                        ),
                    ]
                    if persistent:
                        fdp.group_softmax_pair_twopass(items, 1, None, 1)
                    else:
                        state = fdp.online_softmax_pair_block(
                            *fdp.online_softmax_default(), items
                        )
                        fdp.group_online_softmax(*state, 1, None, 1, partial=True)
                    func.ReturnOp([])
            self.assertTrue(module.operation.verify())


class _SplitSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(value):
        wide = value.float()
        maximum, total = inductor_prims.prepare_softmax_online(wide, -1)
        return torch.exp(wide - maximum) / total

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.input_dtype = inputs[0].dtype
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, gradient):
        (result,) = ctx.saved_tensors
        wide, gradient_wide = result.float(), gradient.float()
        return (
            wide * (gradient_wide - (gradient_wide * wide).sum(-1, keepdim=True))
        ).to(ctx.input_dtype)


class TestFlyDSLSoftmaxPairsGPU(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.version.hip:
            self.skipTest("requires ROCm")
        if RUNTIME_UNAVAILABLE_REASON is not None:
            self.skipTest(RUNTIME_UNAVAILABLE_REASON)
        if not _simd_arch_supported(torch.cuda.current_device()):
            self.skipTest("requires CDNA3 or CDNA4")

    @onlyCUDA
    @parametrize("looped", [False, True])
    def test_split_primitive(self, device, looped):
        torch._dynamo.reset()
        counters.clear()
        x = torch.randn((8, 3137, 64), device=device).transpose(1, 2)
        expected_maximum = x.amax(-1, keepdim=True)
        expected_total = torch.exp(x - expected_maximum).sum(-1, keepdim=True)
        pairs = []
        reduction = FlyDSLKernel.reduction

        def track(kernel, dtype, src_dtype, reduction_type, value):
            if reduction_type == "online_softmax_reduce" and isinstance(value, tuple):
                pairs.append(kernel.persistent)
            return reduction(kernel, dtype, src_dtype, reduction_type, value)

        tuning = {"autotune": 0, "persistent": int(not looped)}
        with (
            config.patch(
                {
                    "flydsl.simd": True,
                    "flydsl.simd_strict": True,
                    "fx_graph_cache": False,
                }
            ),
            patch(
                "torch._inductor.codegen.flydsl.flydsl_simd._tune_table",
                return_value=tuning,
            ),
            patch.object(
                ir.OnlineSoftmaxReduction,
                "num_splits",
                return_value=(ReductionHint.OUTER, 13),
            ),
            patch.object(FlyDSLKernel, "reduction", new=track),
        ):
            fn = torch.compile(
                lambda value: inductor_prims.prepare_softmax_online(value, -1),
                dynamic=False,
            )
            maximum, total = fn(x)
        self.assertEqual(maximum, expected_maximum)
        self.assertEqual(total, expected_total, atol=2e-4, rtol=3e-4)
        self.assertTrue(pairs)
        self.assertIn(not looped, pairs)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)

    @onlyCUDA
    @parametrize("looped", [False, True])
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_split_softmax_forward_backward(self, device, looped, dtype):
        torch._dynamo.reset()
        counters.clear()
        x = (
            torch.randn((8, 3137, 64), device=device, dtype=dtype)
            .transpose(1, 2)
            .requires_grad_()
        )
        probe = torch.randn_like(x)
        reference = torch.softmax(x.float(), dim=-1)
        reference_gradient = torch.autograd.grad((reference * probe).sum(), x)[0]
        tuning = {"autotune": 0, "persistent": int(not looped)}
        pairs = []
        reduction = FlyDSLKernel.reduction

        def track(kernel, dtype, src_dtype, reduction_type, value):
            if reduction_type == "online_softmax_reduce" and isinstance(value, tuple):
                pairs.append(kernel.persistent)
            return reduction(kernel, dtype, src_dtype, reduction_type, value)

        with (
            config.patch(
                {
                    "flydsl.simd": True,
                    "flydsl.simd_strict": True,
                    "fx_graph_cache": False,
                    "online_softmax": True,
                }
            ),
            patch(
                "torch._inductor.codegen.flydsl.flydsl_simd._tune_table",
                return_value=tuning,
            ),
            patch.object(
                ir.OnlineSoftmaxReduction,
                "num_splits",
                return_value=(ReductionHint.OUTER, 13),
            ),
            patch.object(FlyDSLKernel, "reduction", new=track),
        ):
            actual = torch.compile(_SplitSoftmax.apply, dynamic=False)(x)
            gradient = torch.autograd.grad((actual * probe).sum(), x)[0]
        atol, rtol = (1e-5, 3e-2) if dtype == torch.bfloat16 else (1e-6, 3e-4)
        self.assertEqual(actual, reference, atol=atol, rtol=rtol)
        self.assertEqual(gradient, reference_gradient, atol=atol, rtol=rtol)
        self.assertTrue(pairs)
        self.assertIn(not looped, pairs)
        self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)


instantiate_device_type_tests(TestFlyDSLSoftmaxPairsGPU, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()

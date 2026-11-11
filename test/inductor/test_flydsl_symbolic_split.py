# Owner(s): ["module: inductor"]
import operator
import os
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codegen import simd
from torch._inductor.codegen.flydsl import flydsl_simd
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import ShapeEnv
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
from torch.utils._sympy.functions import FloorDiv, Mod
from torch.utils._sympy.value_ranges import ValueRanges


def model_factor(sign=1):
    symbol = sympy.Symbol("s16", integer=True)
    length = sign * symbol
    padded = length + Mod(64 - Mod(length, 64), 64)
    factor = FloorDiv(padded, FloorDiv(padded, 64))
    env = ShapeEnv()
    env.backed_var_to_val[symbol] = sympy.Integer(sign * 569)
    env.var_to_range[symbol] = (
        ValueRanges(2, sympy.oo) if sign == 1 else ValueRanges(-sympy.oo, -2)
    )
    return factor, SizeVarAllocator(env)


def mark_nonsequence_axes_static(values):
    for value in values:
        for axis in range(value.ndim):
            if axis != 1:
                torch._dynamo.mark_static(value, axis)


@instantiate_parametrized_tests
class TestFlyDSLSymbolicSplit(TestCase):
    @parametrize("tokens", [65, 129])
    def test_shape_constraints_keep_only_sequence_dynamic(self, tokens):
        def scaling(query):
            return query * query.shape[-1] ** -0.5

        graph_shapes = []
        graph_has_pow = []
        for constrained in [False, True]:
            torch._dynamo.reset()
            graphs = []

            def record_graph(graph, inputs):
                graphs.append(graph)
                return graph.forward

            value = torch.randn((2, tokens, 2, 8))
            if constrained:
                mark_nonsequence_axes_static([value])
            with torch._dynamo.config.patch(suppress_errors=False):
                actual = torch.compile(
                    scaling, backend=record_graph, dynamic=True, fullgraph=True
                )(value)
            self.assertEqual(actual, scaling(value))
            self.assertEqual(len(graphs), 1)
            examples = [
                node.meta.get("example_value")
                for node in graphs[0].graph.nodes
                if node.op == "placeholder"
            ]
            tensor = next(
                example for example in examples if isinstance(example, torch.Tensor)
            )
            graph_shapes.append(tensor.shape)
            graph_has_pow.append(
                any(
                    node.op == "call_function" and node.target is operator.pow
                    for node in graphs[0].graph.nodes
                )
            )
        self.assertTrue(graph_has_pow[0])
        self.assertFalse(graph_has_pow[1])
        self.assertIsInstance(graph_shapes[0][-1], torch.SymInt)
        self.assertIsInstance(graph_shapes[1][1], torch.SymInt)
        self.assertTrue(
            all(
                not isinstance(size, torch.SymInt)
                for axis, size in enumerate(graph_shapes[1])
                if axis != 1
            )
        )
        torch._dynamo.reset()

    @parametrize("sign", [1, -1])
    def test_padded_qwen_chunk_factor(self, sign):
        factor, sizevars = model_factor(sign)
        self.assertEqual(sizevars.guarding_hint_or_throw(factor), 64)
        self.assertFalse(sizevars.statically_known_equals(factor, 64))
        with V.set_graph_handler(SimpleNamespace(sizevars=sizevars)):
            ranges, getters = simd.SIMDKernel._split_iteration_ranges(
                [factor**2], [[factor, factor], []]
            )
        self.assertEqual(ranges, [[factor, factor]])
        indices = sympy.symbols("i j", integer=True)
        self.assertEqual(
            [getter(list(indices)) for getter in getters[0]], list(indices)
        )
        self.assertEqual(getters[1], [])

    @parametrize("power", [2, 3, 4])
    def test_repeated_integer_power(self, power):
        factor, sizevars = model_factor()
        with V.set_graph_handler(SimpleNamespace(sizevars=sizevars)):
            ranges, getters = simd.SIMDKernel._split_iteration_ranges(
                [factor**power], [[factor] * power, []]
            )
        self.assertEqual(ranges, [[factor] * power])
        indices = list(sympy.symbols(f"i:{power}", integer=True))
        self.assertEqual([getter(indices) for getter in getters[0]], indices)

    @parametrize("combine", [False, True])
    def test_flat_tiled_mapping_getters(self, combine):
        factor, sizevars = model_factor()
        if combine:
            factor = sympy.Symbol("q", integer=True, positive=True)
            env = ShapeEnv()
            env.backed_var_to_val[factor] = sympy.Integer(64)
            env.var_to_range[factor] = ValueRanges(2, sympy.oo)
            sizevars = SizeVarAllocator(env)
        groups, lengths = (
            ([factor, factor], [[factor**2], []])
            if combine
            else ([factor**2], [[factor, factor], []])
        )
        with V.set_graph_handler(SimpleNamespace(sizevars=sizevars)):
            ranges, getters = simd.SIMDKernel._split_iteration_ranges(groups, lengths)
        i, j = sympy.symbols("i j", integer=True)
        self.assertEqual(
            ranges, [[factor], [factor]] if combine else [[factor, factor]]
        )
        actual = [getter([i, j]) for getter in getters[0]]
        self.assertEqual(actual, [factor * i + j] if combine else [i, j])
        for row, column in [(0, 0), (1, 63), (63, 1), (63, 63)]:
            substitutions = {i: row, j: column, **sizevars.shape_env.backed_var_to_val}
            self.assertEqual(
                [int(x.subs(substitutions)) for x in actual],
                [row * 64 + column] if combine else [row, column],
            )

    def test_three_tiled_groups_reconstruct_flat_index(self):
        factor = sympy.Symbol("q", integer=True, positive=True)
        env = ShapeEnv()
        env.backed_var_to_val[factor] = sympy.Integer(64)
        env.var_to_range[factor] = ValueRanges(2, sympy.oo)
        with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator(env))):
            ranges, getters = simd.SIMDKernel._split_iteration_ranges(
                [factor] * 3, [[factor**3], []]
            )
        self.assertEqual(ranges, [[factor]] * 3)
        i, j, k = sympy.symbols("i j k", integer=True)
        self.assertEqual(
            sympy.expand(getters[0][0]([i, j, k])), factor**2 * i + factor * j + k
        )

    def test_unproved_complex_inverse_order_remains_rejected(self):
        factor, sizevars = model_factor()
        self.assertFalse(sizevars.statically_known_gt(factor, 1))
        with (
            V.set_graph_handler(SimpleNamespace(sizevars=sizevars)),
            self.assertRaises(simd.CantSplit),
        ):
            simd.SIMDKernel._split_iteration_ranges([factor, factor], [[factor**2], []])

    def test_nondivisible_extent_remains_rejected(self):
        factor, sizevars = model_factor()
        with (
            V.set_graph_handler(SimpleNamespace(sizevars=sizevars)),
            self.assertRaises(simd.CantSplit),
        ):
            simd.SIMDKernel._split_iteration_ranges(
                [factor**2], [[factor, factor + 1], []]
            )

    def test_zero_divisor_is_not_accepted(self):
        with (
            V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator())),
            self.assertRaises(ZeroDivisionError),
        ):
            simd.SIMDKernel._split_iteration_ranges(
                [sympy.S.Zero], [[sympy.S.Zero], []]
            )

    def test_unfactored_rational_quotient_keeps_floordiv(self):
        symbol = sympy.Symbol("n", integer=True, positive=True)
        env = ShapeEnv()
        env.backed_var_to_val[symbol] = sympy.Integer(7)
        env.var_to_range[symbol] = ValueRanges(2, sympy.oo)
        sizevars = SizeVarAllocator(env)
        numerator, divisor = symbol**2 - 1, symbol - 1
        self.assertIsNone((numerator / divisor).is_integer)
        self.assertTrue(sizevars.statically_known_multiple_of(numerator, divisor))
        with (
            V.set_graph_handler(SimpleNamespace(sizevars=sizevars)),
            mock.patch.object(simd, "FloorDiv", wraps=FloorDiv) as floor,
        ):
            ranges, _ = simd.SIMDKernel._split_iteration_ranges(
                [numerator], [[divisor, symbol + 1], []]
            )
        self.assertEqual(ranges, [[divisor, symbol + 1]])
        self.assertIn(mock.call(numerator, divisor), floor.call_args_list)


class TestOfficialGatedDeltaSymbolicSplit(TestCase):
    def setUp(self):
        super().setUp()
        from torch._inductor.codegen.flydsl.flydsl_utils import (
            _flydsl_runtime_unavailable_reason,
            _simd_arch_supported,
        )

        if not torch.version.hip:
            self.skipTest("requires ROCm")
        reason = _flydsl_runtime_unavailable_reason()
        if reason is not None:
            self.skipTest(reason)
        if not _simd_arch_supported(torch.cuda.current_device()):
            self.skipTest("requires a supported CDNA device")

    @onlyCUDA
    @parametrize("flydsl", [False, True])
    @parametrize("autotune", [False, True])
    def test_official_delta_rule_forward_and_vjp(self, device, flydsl, autotune):
        import inspect

        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_chunk_gated_delta_rule,
        )

        torch_chunk_gated_delta_rule = inspect.unwrap(torch_chunk_gated_delta_rule)
        self.assertTrue(
            torch_chunk_gated_delta_rule.__code__.co_filename.endswith(
                "models/qwen3_5/modeling_qwen3_5.py"
            )
        )

        observed = []
        original_split = simd.SIMDKernel._split_iteration_ranges

        def observe_split(groups, lengths):
            groups = tuple(groups)
            observed.extend(
                sympy.sympify(extent)
                for extent in [
                    *groups,
                    *(extent for group in lengths for extent in group),
                ]
            )
            return original_split(groups, lengths)

        def official(query, key, value, decay, beta):
            return torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                decay,
                beta,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        options = {
            "flydsl.simd": flydsl,
            "flydsl.simd_strict": flydsl,
            "flydsl.simd_autotune_backend": False,
            "max_autotune": False,
            "max_autotune_gemm": False,
            "max_autotune_gemm_backends": "ATEN",
            "max_autotune_conv_backends": "ATEN",
            "conv_1x1_as_mm": False,
            "max_autotune_pointwise": autotune,
            "triton.autotune_pointwise": autotune,
            "fx_graph_cache": False,
            "triton.cudagraphs": False,
        }
        for tokens in [65, 129]:
            with self.subTest(tokens=tokens):
                torch.manual_seed(2026)
                shape = (2, tokens, 2, 8)
                inputs = [
                    (
                        torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.25
                    ).requires_grad_()
                    for _ in range(3)
                ]
                inputs += [
                    (-torch.rand((2, tokens, 2), device=device) * 0.1).requires_grad_(),
                    torch.full(
                        (2, tokens, 2),
                        0.5,
                        device=device,
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    ),
                ]
                mark_nonsequence_axes_static(inputs)
                reference = [
                    value.detach().clone().requires_grad_(True) for value in inputs
                ]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = official(*reference)
                cotangents = [torch.randn_like(value) * 0.125 for value in expected]
                expected_gradients = torch.autograd.grad(
                    expected, reference, cotangents
                )
                torch._dynamo.reset()
                counters.clear()
                flydsl_simd._tune_table.cache_clear()
                try:
                    with (
                        torch._dynamo.config.patch(suppress_errors=False),
                        config.patch(options),
                        mock.patch.dict(
                            os.environ,
                            {
                                "TORCHINDUCTOR_FLYDSL_SIMD_TUNE": f"autotune={int(autotune)}"
                            },
                        ),
                        mock.patch.object(
                            simd.SIMDKernel,
                            "_split_iteration_ranges",
                            side_effect=observe_split,
                        ),
                    ):
                        self.assertEqual(
                            flydsl_simd._tune_table().get("autotune"), int(autotune)
                        )
                        self.assertEqual(config.flydsl.simd, flydsl)
                        compiled = torch.compile(official, dynamic=True)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            actual = compiled(*inputs)
                        actual_gradients = torch.autograd.grad(
                            actual, inputs, cotangents
                        )
                finally:
                    flydsl_simd._tune_table.cache_clear()
                self.assertGreater(counters["stats"]["unique_graphs"], 0)
                self.assertEqual(actual, expected, atol=2e-3, rtol=2e-2)
                self.assertEqual(len(actual_gradients), 5)
                for actual_gradient, expected_gradient in zip(
                    actual_gradients, expected_gradients
                ):
                    self.assertTrue(torch.isfinite(actual_gradient).all().item())
                    self.assertEqual(
                        actual_gradient, expected_gradient, atol=2e-3, rtol=2e-2
                    )
                if flydsl:
                    self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
                    self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)
                else:
                    self.assertEqual(counters["flydsl_simd"]["flydsl_kernels"], 0)
        self.assertTrue(any(extent.has(FloorDiv) for extent in observed))
        if flydsl:
            self.assertTrue(
                any(
                    power.base.has(FloorDiv) and power.exp.is_Integer and power.exp >= 2
                    for extent in observed
                    for power in extent.atoms(sympy.Pow)
                )
            )


instantiate_device_type_tests(
    TestOfficialGatedDeltaSymbolicSplit, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()

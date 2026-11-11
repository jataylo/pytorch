# Owner(s): ["module: inductor"]
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import sympy

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codegen.flydsl import flydsl_diagnostics, flydsl_simd
from torch._inductor.codegen.flydsl.flydsl_simd import (
    FlyDSLKernel,
    FlyDSLSIMDScheduling,
    FlyDSLUnsupported,
)
from torch._inductor.codegen.flydsl.flydsl_utils import (
    _flydsl_runtime_unavailable_reason,
    _simd_arch_supported,
)
from torch._inductor.utils import IndentedBuffer
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
class TestFlyDSLLayoutGrid(TestCase):
    @parametrize("third_axis", [False, True])
    def test_pointwise_unit_reduction_axis_is_removed(self, third_axis):
        preferred = {"y": sympy.Integer(17), "x": sympy.Integer(5), "r0_": sympy.S.One}
        if third_axis:
            preferred["z"] = sympy.Integer(11)
        features = SimpleNamespace(
            node_schedule=[],
            numel=sympy.Integer(935 if third_axis else 85),
            reduction_numel=sympy.S.One,
            coalesce_analysis=None,
            is_reduction=lambda: False,
        )
        scheduling = object.__new__(FlyDSLSIMDScheduling)
        scheduling.get_tiling_and_scores = lambda *args: (preferred, {})
        with patch(
            "torch._inductor.codegen.flydsl.flydsl_simd._tune_table", return_value={}
        ):
            actual = scheduling._choose_tiling(features)
        self.assertEqual(
            actual, {name: value for name, value in preferred.items() if name != "r0_"}
        )

    @parametrize("reduction", [False, True])
    def test_real_reduction_axis_is_not_discarded(self, reduction):
        preferred = {
            "z": sympy.Integer(11),
            "y": sympy.Integer(17),
            "x": sympy.Integer(5),
            "r0_": sympy.Integer(7),
        }
        features = SimpleNamespace(
            node_schedule=[],
            numel=sympy.Integer(935),
            reduction_numel=sympy.Integer(7),
            coalesce_analysis=None,
            is_reduction=lambda: reduction,
        )
        scheduling = object.__new__(FlyDSLSIMDScheduling)
        scheduling.get_tiling_and_scores = lambda *args: (preferred, {})
        reason = (
            "reduction tiling needs three output axes"
            if reduction
            else "pointwise tiling has non-unit reduction axes"
        )
        with patch(
            "torch._inductor.codegen.flydsl.flydsl_simd._tune_table", return_value={}
        ):
            with self.assertRaisesRegex(FlyDSLUnsupported, reason):
                scheduling._choose_tiling(features)

    @parametrize("reduction", [False, True])
    def test_preferred_axes_are_preserved(self, reduction):
        preferred = {"y": sympy.Integer(17), "x": sympy.Integer(5)}
        preferred["r0_" if reduction else "z"] = sympy.Integer(231)
        features = SimpleNamespace(
            node_schedule=[],
            numel=sympy.Integer(85 if reduction else 19635),
            reduction_numel=sympy.Integer(231 if reduction else 1),
            coalesce_analysis=None,
            is_reduction=lambda: reduction,
        )
        scheduling = object.__new__(FlyDSLSIMDScheduling)
        scheduling.get_tiling_and_scores = lambda *args: (preferred, {})
        with patch(
            "torch._inductor.codegen.flydsl.flydsl_simd._tune_table", return_value={}
        ):
            self.assertEqual(scheduling._choose_tiling(features), preferred)

    def test_multiple_reduction_axes_keep_flattened_coverage(self):
        preferred = {
            "y": sympy.Integer(17),
            "x": sympy.Integer(5),
            "r0_": sympy.Integer(11),
            "r1_": sympy.Integer(7),
        }
        features = SimpleNamespace(
            node_schedule=[],
            numel=sympy.Integer(85),
            reduction_numel=sympy.Integer(77),
            coalesce_analysis=None,
            is_reduction=lambda: True,
        )
        scheduling = object.__new__(FlyDSLSIMDScheduling)
        scheduling.get_tiling_and_scores = lambda *args: (preferred, {})
        with patch(
            "torch._inductor.codegen.flydsl.flydsl_simd._tune_table", return_value={}
        ):
            self.assertEqual(scheduling._choose_tiling(features), {"x": 85, "r0_": 77})
        self.assertTrue(features.flydsl_transposed)

    @parametrize("reduction", [False, True])
    @parametrize("output_axis", [-1, 0, 1])
    def test_contiguous_axis_is_inside_grid(self, reduction, output_axis):
        symbols = {
            dim: sympy.Symbol(f"{dim}index", integer=True, nonnegative=True)
            for dim in ("x", "y", "z")
        }
        x, y, z = (symbols[dim] for dim in ("x", "y", "z"))
        loads = (
            [y + 32 * x, y + 64 * x, y + 128 * x]
            if reduction
            else [z + 32 * x, z + 64 * x, z + 128 * x, y + 64 * z]
        )
        stores = [x + 32 * y]
        kernel = SimpleNamespace(
            numels={"x": 17, "y": 11},
            is_reduction_kernel=reduction,
            output_extents={"x": (17, None), "y": (11, None), "z": (19, None)},
            axis_order={"x": "x", "y": "y", "z": "z"},
            range_trees=[
                SimpleNamespace(prefix=dim, symbol=lambda value=value: value)
                for dim, value in symbols.items()
            ],
            _memory_roots=lambda loads=True, stores=True: loads_exprs
            if loads
            else stores_exprs,
        )
        loads_exprs, stores_exprs = loads, stores
        if not reduction:
            kernel.numels["z"] = 19
        with patch.object(
            flydsl_simd, "_tune_table", return_value={"output_axis": output_axis}
        ):
            FlyDSLKernel._order_output_axes(kernel)
            if reduction:
                expected = (
                    {"x": "x", "y": "y", "z": "z"}
                    if output_axis == 0
                    else {"x": "y", "y": "x", "z": "z"}
                )
            else:
                expected = {"x": "z", "y": "x", "z": "y"}
            self.assertEqual(kernel.axis_order, expected)
            FlyDSLKernel._order_output_axes(kernel)
            self.assertEqual(kernel.axis_order, expected)
        self.assertEqual(
            kernel.xnumel,
            17 if reduction and output_axis == 0 else 11 if reduction else 19,
        )

    @parametrize("inner", ["x", "y"])
    @parametrize("override", [None, 0, 1])
    def test_alternate_axis_precedes_geometry_search(self, inner, override):
        kernel = SimpleNamespace(
            stage="row",
            xnumel=17,
            ynumel=11,
            znumel=1,
            rnumel=231,
            is_reduction_kernel=True,
            numels={"x": 17, "y": 11, "r0_": 231},
            axis_order={"x": inner, "y": "y" if inner == "x" else "x", "z": "z"},
            in_combo=False,
            _has_atomic=lambda: False,
            ept=4,
            threads=256,
            tpr=64,
            persistent=True,
            cpair=1,
        )
        tuning = {} if override is None else {"output_axis": override}
        with (
            patch.object(flydsl_simd, "_tune_table", return_value=tuning),
            config.patch(
                {
                    "flydsl.simd_autotune_min_numel": 0,
                    "flydsl.simd_autotune_max_configs": 2,
                }
            ),
        ):
            candidates = FlyDSLSIMDScheduling._candidates(kernel)
        if override is None:
            self.assertEqual(candidates[0], {"output_axis": 1 if inner == "x" else 0})
        else:
            self.assertTrue(candidates)
            self.assertTrue(
                all("output_axis" not in candidate for candidate in candidates)
            )

    @parametrize("outer", [1, 7])
    def test_outer_grid_preserves_inner_tile(self, outer):
        kernel = object.__new__(FlyDSLKernel)
        kernel.__dict__.update(
            tile2d=False,
            is_reduction_kernel=True,
            mix_order_reduction=False,
            split_chunks=0,
            xnumel=65,
            ynumel=outer,
            znumel=1,
            rpb=8,
            xunroll=1,
            _pair_lines=None,
        )
        self.assertEqual(FlyDSLKernel.num_blocks.fget(kernel), 9 * outer)

    def test_outer_coordinates_are_emitted_before_inner_body(self):
        kernel = SimpleNamespace(
            tile2d=False,
            is_reduction_kernel=True,
            ynumel=11,
            znumel=1,
            ynumel_dyn=None,
            znumel_dyn=None,
            axis_order={"x": "y", "y": "x", "z": "z"},
            _inner_blocks_expr=lambda: sympy.Integer(5),
            numels={"x": 11, "y": 5, "r0_": 32},
        )
        code = IndentedBuffer()
        FlyDSLKernel._emit_outer_indices(kernel, code)
        source = code.getvalue()
        self.assertIn("xindex = _b // 5", source)
        self.assertIn("_b = _b % 5", source)

    @parametrize("order", list(itertools.permutations(("x", "y", "z"))))
    @parametrize("shape", [(17, 11, 19), (16, 11, 32)])
    @parametrize("fast", ["x", "y"])
    def test_three_axis_tiles_cover_each_element_once(self, order, shape, fast):
        sizes = dict(zip(("x", "y", "z"), shape))
        kernel = object.__new__(FlyDSLKernel)
        kernel.__dict__.update(
            tile2d=True,
            is_reduction_kernel=False,
            mix_order_reduction=False,
            numels=sizes,
            axis_order=dict(zip(("x", "y", "z"), order)),
            xnumel=sizes[order[0]],
            ynumel=sizes[order[1]],
            znumel=sizes[order[2]],
            xnumel_dyn=None,
            ynumel_dyn=None,
            znumel_dyn=None,
            fast=fast,
            tile_f=16,
            tile_s=16,
            threads=64,
            ept=2,
            unroll=2,
            rpb=8,
            staged=[],
            range_tree_nodes={},
            uses_smem=False,
            lds_tables={},
            max_components=1,
            scan_components=0,
            sort_words=0,
            split_chunks=0,
            prefetch=IndentedBuffer(),
            prefetch_step=0,
            body=IndentedBuffer(),
            _load_mask=None,
        )
        condition = " and ".join(kernel._masks()) or "True"
        kernel.body.writeline(
            f"if {condition}: visited.append((xindex, yindex, zindex))"
        )
        code = IndentedBuffer()
        kernel.emit_inner(code)
        executable = compile(code.getvalue(), "generated_tile_indices", "exec")
        visited = []
        for block in range(kernel.num_blocks):
            for thread in range(kernel.threads):
                exec(
                    executable,
                    {
                        "_b": block,
                        "_t": thread,
                        "range_constexpr": range,
                        "visited": visited,
                    },
                )
        expected = set(
            itertools.product(range(sizes["x"]), range(sizes["y"]), range(sizes["z"]))
        )
        self.assertEqual(len(visited), len(expected))
        self.assertEqual(set(visited), expected)

    @parametrize("outer_axis", [False, True])
    def test_split_scan_workspace_needs_flat_rows(self, outer_axis):
        node = SimpleNamespace(is_split_scan=lambda: True, is_reduction=lambda: True)
        kernel = SimpleNamespace(
            features=SimpleNamespace(scheduler_nodes=lambda: [node]),
            rnumel_dyn=None,
            xnumel_dyn=None,
            numels={"x": 17, "r0_": 8192},
        )
        if outer_axis:
            kernel.numels["y"] = 11
        self.assertEqual(FlyDSLKernel._split_scan_ok(kernel), not outer_axis)

    @parametrize("inner", ["x", "y"])
    def test_two_axis_rows_cover_each_output_once(self, inner):
        outer = "y" if inner == "x" else "x"
        sizes = {"x": 17, "y": 11}
        kernel = object.__new__(FlyDSLKernel)
        kernel.__dict__.update(
            tile2d=False,
            is_reduction_kernel=True,
            mix_order_reduction=False,
            numels=sizes,
            axis_order={"x": inner, "y": outer, "z": "z"},
            xnumel=sizes[inner],
            ynumel=sizes[outer],
            znumel=1,
            xnumel_dyn=None,
            ynumel_dyn=None,
            znumel_dyn=None,
            threads=64,
            tpr=8,
            stride=1,
            rpb=8,
            xunroll=2,
            cpair=1,
            persistent=False,
            _pair_lines=None,
            staged=[],
            uses_smem=False,
            lds_tables={},
            max_components=1,
            scan_components=0,
            sort_words=0,
            split_chunks=0,
            preload_lines=[],
            body=IndentedBuffer(),
            _insert_widen=lambda: None,
        )
        kernel.body.writeline("if _lane0 and xmask: visited.append((xindex, yindex))")
        code = IndentedBuffer()
        kernel.emit_inner(code)
        executable = compile(code.getvalue(), "generated_row_indices", "exec")
        visited = []
        for block in range(kernel.num_blocks):
            for thread in range(kernel.threads):
                exec(
                    executable,
                    {
                        "_b": block,
                        "_t": thread,
                        "range_constexpr": range,
                        "visited": visited,
                    },
                )
        expected = set(itertools.product(range(sizes["x"]), range(sizes["y"])))
        self.assertEqual(len(visited), len(expected))
        self.assertEqual(set(visited), expected)


class TestFlyDSLTiledLayouts(TestCase):
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
                    "max_autotune_gemm_backends": "ATEN",
                    "fx_graph_cache": False,
                }
            )
        )
        self.enterContext(
            patch(
                "torch._inductor.codegen.flydsl.flydsl_simd._tune_table",
                return_value={"autotune": 0},
            )
        )
        torch._dynamo.reset()
        counters.clear()
        self.layouts = []
        finalize = FlyDSLKernel.finalize_indexing

        def observe(kernel, indices):
            result = finalize(kernel, indices)
            self.layouts.append(
                {
                    "reduction": kernel.is_reduction_kernel,
                    "axes": frozenset(kernel.numels),
                    "order": dict(kernel.axis_order),
                    "dynamic_outer": kernel.znumel_dyn is not None
                    if kernel.tile2d
                    else kernel.ynumel_dyn is not None,
                }
            )
            return result

        self.enterContext(patch.object(FlyDSLKernel, "finalize_indexing", new=observe))

    def preferred_tiling(self, build):
        original = FlyDSLSIMDScheduling.get_tiling_and_scores

        def choose(
            cls, nodes, numel, reduction_numel=sympy.S.One, coalesce_analysis=None
        ):
            tiling = build(numel, reduction_numel)
            if tiling is not None and cls.tiling_is_compatible(
                nodes, numel, reduction_numel, tiling
            ):
                return tiling, None
            return original(nodes, numel, reduction_numel, coalesce_analysis)

        return patch.object(
            FlyDSLSIMDScheduling, "get_tiling_and_scores", new=classmethod(choose)
        )

    @onlyCUDA
    @parametrize("shape", [(17, 5, 231), (128, 32, 256)])
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_two_output_axes_reduction(self, device, shape, dtype):
        y, x, r = shape
        first = torch.randn((y, x, r), device=device, dtype=dtype, requires_grad=True)
        second = torch.randn((x, r, y), device=device, dtype=dtype, requires_grad=True)

        def fn(a, b):
            return (a * b.permute(2, 0, 1)).sum(-1)

        bf16_control = dtype == torch.bfloat16 and shape == (17, 5, 231)
        expected = fn(first, second)
        probe = torch.randn_like(expected)
        expected_grads = torch.autograd.grad(
            expected, (first, second), probe, retain_graph=bf16_control
        )
        if bf16_control:
            expected_mse_grads = torch.autograd.grad(
                expected.float().square().mean(), (first, second)
            )
            torch._dynamo.reset()
            counters.clear()
            with config.patch({"flydsl.simd": False}):
                triton_output = torch.compile(fn, dynamic=False)(first, second)
                triton_grads = torch.autograd.grad(
                    triton_output, (first, second), probe, retain_graph=True
                )
                triton_mse_grads = torch.autograd.grad(
                    triton_output.float().square().mean(), (first, second)
                )
            self.assertEqual(counters["flydsl_simd"]["flydsl_kernels"], 0)
            self.assertEqual(triton_output, expected, atol=0.125, rtol=2e-2)
            self.assertEqual(triton_grads, expected_grads, atol=1e-4, rtol=3e-2)
            torch._dynamo.reset()
            counters.clear()
            self.layouts.clear()
        with self.preferred_tiling(
            lambda n, rn: {
                "y": sympy.Integer(y),
                "x": sympy.Integer(x),
                "r0_": sympy.Integer(r),
            }
            if n == y * x and rn == r
            else None
        ):
            actual = torch.compile(fn, dynamic=False)(first, second)
            actual_grads = torch.autograd.grad(
                actual, (first, second), probe, retain_graph=bf16_control
            )
            if bf16_control:
                actual_mse_grads = torch.autograd.grad(
                    actual.float().square().mean(), (first, second)
                )
        if dtype == torch.bfloat16:
            self.assertEqual(actual, expected, atol=0.125, rtol=2e-2)
            self.assertEqual(actual_grads, expected_grads, atol=1e-4, rtol=3e-2)
        else:
            self.assertEqual(actual, expected, atol=2e-4, rtol=2e-4)
            self.assertEqual(actual_grads, expected_grads, atol=2e-5, rtol=2e-4)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)
        self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
        self.assertTrue(
            any(
                layout["reduction"] and {"x", "y", "r0_"} <= layout["axes"]
                for layout in self.layouts
            ),
            self.layouts,
        )
        if bf16_control:
            self.assertEqual(actual_grads, triton_grads, atol=1e-4, rtol=3e-2)
            fly_mse_error = max(
                (value.float() - reference.float()).abs().max().item()
                for value, reference in zip(actual_mse_grads, expected_mse_grads)
            )
            triton_mse_error = max(
                (value.float() - reference.float()).abs().max().item()
                for value, reference in zip(triton_mse_grads, expected_mse_grads)
            )
            print(
                {
                    "original_bf16_mse_gradient_max_error": {
                        "flydsl": fly_mse_error,
                        "triton": triton_mse_error,
                    }
                }
            )

    @onlyCUDA
    def test_swapped_inner_reduction_axis(self, device):
        inputs = [torch.randn((17, 5, 231), device=device, requires_grad=True)]
        inputs.extend(
            torch.randn((5, 231, 17), device=device, requires_grad=True)
            for _ in range(3)
        )

        def fn(a, b, c, d):
            return (
                a * b.permute(2, 0, 1) * c.permute(2, 0, 1) * d.permute(2, 0, 1)
            ).sum(-1)

        expected = fn(*inputs)
        probe = torch.randn_like(expected)
        expected_grads = torch.autograd.grad(expected, inputs, probe)
        with (
            config.patch(
                {"pick_loop_orders": False, "loop_ordering_after_fusion": False}
            ),
            self.preferred_tiling(
                lambda n, rn: {
                    "y": sympy.Integer(17),
                    "x": sympy.Integer(5),
                    "r0_": sympy.Integer(231),
                }
                if n == 85 and rn == 231
                else None
            ),
        ):
            actual = torch.compile(fn, dynamic=False)(*inputs)
            actual_grads = torch.autograd.grad(actual, inputs, probe)
        self.assertEqual(actual, expected, atol=2e-4, rtol=2e-4)
        self.assertEqual(actual_grads, expected_grads, atol=2e-4, rtol=2e-4)
        self.assertTrue(
            any(
                layout["reduction"]
                and "y" in layout["axes"]
                and layout["order"]["x"] == "y"
                for layout in self.layouts
            ),
            self.layouts,
        )
        self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)

    @onlyCUDA
    def test_dynamic_outer_reduction_axis(self, device):
        def fn(a, b):
            return (a * b.permute(2, 0, 1)).sum(-1)

        first = torch.randn((17, 5, 231), device=device)
        second = torch.randn((5, 231, 17), device=device)
        torch._dynamo.mark_dynamic(first, 0, min=2, max=32)
        torch._dynamo.mark_dynamic(second, 2, min=2, max=32)
        compiled = torch.compile(fn)
        with self.preferred_tiling(
            lambda n, rn: {
                "y": sympy.simplify(n / 5),
                "x": sympy.Integer(5),
                "r0_": sympy.Integer(231),
            }
            if rn == 231
            else None
        ):
            self.assertEqual(
                compiled(first, second), fn(first, second), atol=2e-4, rtol=2e-4
            )
            graphs = counters["stats"]["unique_graphs"]
            next_first = torch.randn((23, 5, 231), device=device)
            next_second = torch.randn((5, 231, 23), device=device)
            self.assertEqual(
                compiled(next_first, next_second),
                fn(next_first, next_second),
                atol=2e-4,
                rtol=2e-4,
            )
            self.assertEqual(counters["stats"]["unique_graphs"], graphs)
        self.assertTrue(
            any(
                layout["reduction"] and layout["dynamic_outer"]
                for layout in self.layouts
            ),
            self.layouts,
        )
        self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)

    @onlyCUDA
    def test_alternate_axis_autotune_with_two_config_budget(self, device):
        first = torch.randn((17, 5, 231), device=device, requires_grad=True)
        second = torch.randn((5, 231, 17), device=device, requires_grad=True)

        def fn(a, b):
            return (a * b.permute(2, 0, 1)).sum(-1)

        expected = fn(first, second)
        probe = torch.randn_like(expected)
        expected_grads = torch.autograd.grad(expected, (first, second), probe)
        with (
            config.patch(
                {
                    "flydsl.simd_autotune_min_numel": 0,
                    "flydsl.simd_autotune_max_configs": 2,
                    "force_disable_caches": True,
                }
            ),
            patch.object(flydsl_simd, "_tune_table", return_value={"autotune": 1}),
            patch.object(
                flydsl_simd, "_autotune_source", wraps=flydsl_simd._autotune_source
            ) as builder,
            patch.object(
                flydsl_diagnostics,
                "record_kernel_decision",
                wraps=flydsl_diagnostics.record_kernel_decision,
            ) as diagnostics,
            self.preferred_tiling(
                lambda n, rn: {
                    "y": sympy.Integer(17),
                    "x": sympy.Integer(5),
                    "r0_": sympy.Integer(231),
                }
                if n == 85 and rn == 231
                else None
            ),
        ):
            actual = torch.compile(fn, dynamic=False)(first, second)
            actual_grads = torch.autograd.grad(actual, (first, second), probe)
        self.assertEqual(actual, expected, atol=2e-4, rtol=2e-4)
        self.assertEqual(actual_grads, expected_grads, atol=2e-5, rtol=2e-4)
        self.assertTrue(builder.call_args_list)
        self.assertTrue(all(len(call.args[0]) == 2 for call in builder.call_args_list))
        alternate = "axes={'x': 'y', 'y': 'x', 'z': 'z'}"
        self.assertTrue(
            any(
                alternate in description
                for call in builder.call_args_list
                for description in call.args[3]["candidate_descriptions"][1:]
            )
        )
        events = [call.args[0] for call in diagnostics.call_args_list]
        self.assertFalse(
            [
                event
                for event in events
                if event.get("category") == "candidate_rejected"
                and event.get("reason") == "output differs from reference candidate"
            ]
        )
        self.assertTrue(
            any(
                event.get("category") == "autotune_selection"
                and event.get("cache_hit") is False
                for event in events
            )
        )
        self.assertGreater(counters["flydsl_simd"]["autotune_candidates"], 1)
        self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)

    @onlyCUDA
    @parametrize("shape", [(19, 11, 37), (64, 65, 127)])
    def test_three_pointwise_axes(self, device, shape):
        z, y, x = shape
        a = torch.randn((z, y, x), device=device)
        b = torch.randn((x, z, y), device=device)
        c = torch.randn((y, x, z), device=device)

        def fn(x, y, z):
            return torch.sin(x) + y.permute(1, 2, 0) * z.permute(2, 0, 1)

        self.assertEqual(torch.compile(fn, dynamic=False)(a, b, c), fn(a, b, c))
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)
        self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
        self.assertTrue(
            any(
                not layout["reduction"] and {"x", "y", "z"} <= layout["axes"]
                for layout in self.layouts
            ),
            self.layouts,
        )


instantiate_device_type_tests(TestFlyDSLTiledLayouts, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()

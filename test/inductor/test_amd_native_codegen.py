# Owner(s): ["module: inductor"]

import ast
import textwrap
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.codegen.amd_kernel_optimizations import (
    _row_tail_safe,
    amd_persistent_reduction_eligible,
    AMDCodegenSource,
    optimize_flat_pointwise_source,
    optimize_pointwise_source,
    optimize_row_source,
    optimize_tiled_transpose_source,
    simplify_masked_modulo,
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


def row_body():
    return textwrap.dedent("""
        def kernel(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK):
            R0_BLOCK: tl.constexpr = 128
            xoffset = tl.program_id(0) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
            rindex = tl.arange(0, R0_BLOCK)[None, :]
            xmask = xindex < xnumel
            mask = xmask & (rindex < r0_numel)
            value = tl.load(in_ptr0 + xindex * r0_numel + rindex, mask, other=0)
            result = tl.sum(value, 1)[:, None].to(tl.float32)
            tl.store(out_ptr0 + xindex, result, xmask)
    """)


@instantiate_parametrized_tests
class TestAMDNativeCodegen(TestCase):
    @parametrize("family", ["pointwise", "row", "tiled", "grid2d"])
    def test_shared_ast_matches_sequential_emission(self, family):
        from test_amd_kernel_hardware import pointwise_source, row_source

        device = DeviceProperties(
            "hip",
            0,
            16,
            "gfx_future",
            warp_size=64,
            max_threads_per_block=1024,
            regs_per_multiprocessor=131072,
        )
        if family in ("tiled", "grid2d"):
            from test_amd_tiled_transpose import grid2d_source, transpose_source

            source = transpose_source() if family == "tiled" else grid2d_source()
        else:
            source = (
                pointwise_source(device)
                if family == "pointwise"
                else row_source(device)
            )
        passes = [
            (simplify_masked_modulo, {}),
            (optimize_tiled_transpose_source, {"extended": True}),
            (optimize_pointwise_source, {"extended": True, "use_gluon": False}),
            (
                optimize_row_source,
                {"extended": True, "further": True, "use_gluon": False},
            ),
            (optimize_flat_pointwise_source, {"use_gluon": False}),
        ]
        sequential = source
        context = AMDCodegenSource(source)
        for transform, kwargs in passes:
            sequential = transform(sequential, **kwargs)
            transform(source, _context=context, **kwargs)
        self.assertTrue(context.changed)
        self.assertEqual(
            ast.dump(ast.parse(context.emit())), ast.dump(ast.parse(sequential))
        )

    def test_shared_ast_parses_unmatched_source_once(self):
        source = "# Preserve formatting.\n@triton.jit\ndef kernel(value):\n    return value\n"
        with patch("ast.parse", wraps=ast.parse) as parse:
            context = AMDCodegenSource(source)
            for transform in (
                simplify_masked_modulo,
                optimize_tiled_transpose_source,
                optimize_pointwise_source,
                optimize_row_source,
                optimize_flat_pointwise_source,
            ):
                transform(source, _context=context)
            self.assertEqual(context.emit(), source)
        parse.assert_called_once_with(source)

    @parametrize(
        "change",
        [
            "none",
            "store_mask",
            "load_mask",
            "or_mask",
            "axis",
            "transpose",
            "index",
            "upper_bound",
            "method",
            "property",
            "reassign",
            "row_slice",
            "alias_reduce",
            "alias_trans",
        ],
    )
    def test_partial_row_proof(self, change):
        source = row_body()
        if change == "store_mask":
            source = source.replace("result, xmask)", "result, rindex < r0_numel)")
        elif change == "load_mask":
            source = source.replace(
                "rindex, mask, other=0)", "rindex, rindex < r0_numel, other=0)"
            )
        elif change == "or_mask":
            source = source.replace("xmask & (", "xmask | (")
        elif change == "axis":
            source = source.replace("tl.sum(value, 1)", "tl.sum(value, 0)")
        elif change == "transpose":
            source = source.replace("tl.sum(value, 1)", "tl.sum(tl.trans(value), 1)")
        elif change == "method":
            source = source.replace(
                "tl.sum(value, 1)", "tl.sum(value.permute(1, 0), 1)"
            )
        elif change == "property":
            source = source.replace("tl.sum(value, 1)", "tl.sum(value.T, 1)")
        elif change == "reassign":
            source += "    xmask = xindex < xnumel\n"
            source = source.replace("xmask = xindex < xnumel", "xmask = xindex < 64", 1)
        elif change == "alias_reduce":
            source = source.replace(
                "result = tl.sum(value, 1)",
                "reduce = tl.sum\n    result = reduce(value, 0)",
            )
        elif change == "alias_trans":
            source = source.replace(
                "result = tl.sum(value, 1)",
                "swap = tl.trans\n    result = tl.sum(swap(value), 1)",
            )
        elif change == "row_slice":
            source = source.replace("tl.sum(value, 1)", "tl.sum(value[1:, :], 1)")
        elif change == "index":
            source = source.replace("xoffset + tl.arange", "-xoffset + tl.arange")
        elif change == "upper_bound":
            source = source.replace("xindex < xnumel", "xindex <= xnumel")
        fn = ast.parse(source).body[0]
        assignments = {
            node.targets[0].id: node.value
            for node in fn.body
            if isinstance(node, ast.Assign)
        }
        self.assertEqual(_row_tail_safe(fn, assignments), change == "none")

    @parametrize("dependent", [False, True])
    def test_row_independent_weight_load(self, dependent):
        source = row_body().replace("in_ptr0, out_ptr0", "in_ptr0, in_ptr1, out_ptr0")
        address = "xindex" if dependent else "rindex"
        source = source.replace(
            "    result =",
            f"    weight = tl.load(in_ptr1 + {address}, rindex < r0_numel, other=0)\n    result =",
        )
        source = source.replace("tl.sum(value, 1)", "tl.sum(value * weight, 1)")
        fn = ast.parse(source).body[0]
        assignments = {
            node.targets[0].id: node.value
            for node in fn.body
            if isinstance(node, ast.Assign)
        }
        self.assertEqual(_row_tail_safe(fn, assignments), not dependent)

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
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "triton.amd_further_kernel_optimizations": True,
                "triton.amd_use_gluon": False,
                "triton.amd_persistent_reduction_autotune": True,
                "fx_graph_cache": False,
                "autotune_local_cache": False,
            }
        ):
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertEqual(
            actual,
            fn(x),
            atol=0.02 if dtype == torch.bfloat16 else 2e-5,
            rtol=0.02 if dtype == torch.bfloat16 else 2e-5,
        )
        source = "\n".join(code)
        if width == 8192:
            self.assertIn("async_compile.multi_kernel", source)
            self.assertIn("triton_heuristics.reduction", source)
        else:
            self.assertNotIn("async_compile.multi_kernel", source)
        self.assertIn("triton_heuristics.persistent_reduction", source)
        self.assertNotIn("experimental.gluon", source)

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_dynamic_training_gradients(self, device, dtype):
        if torch.version.hip is None:
            self.skipTest("AMD optimization")
        torch._dynamo.reset()

        def fn(x, weight):
            normalized = torch.nn.functional.layer_norm(x.float(), (2048,), weight)
            return torch.softmax(torch.nn.functional.gelu(normalized), -1).to(x.dtype)

        compiled = torch.compile(fn, fullgraph=True)
        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "triton.amd_further_kernel_optimizations": True,
                "triton.amd_use_gluon": False,
                "fx_graph_cache": False,
                "autotune_local_cache": False,
            }
        ):
            for rows in (37, 39):
                x = torch.randn(
                    rows, 2048, device=device, dtype=dtype, requires_grad=True
                )
                weight = torch.randn(2048, device=device, requires_grad=True)
                reference_x = x.detach().clone().requires_grad_()
                reference_weight = weight.detach().clone().requires_grad_()
                torch._dynamo.mark_dynamic(x, 0)
                actual = compiled(x, weight)
                expected = fn(reference_x, reference_weight)
                gradient = torch.randn_like(actual)
                actual.backward(gradient)
                expected.backward(gradient)
                tolerance = 0.02 if dtype == torch.bfloat16 else 2e-4
                self.assertEqual(actual, expected, atol=tolerance, rtol=tolerance)
                self.assertEqual(
                    x.grad, reference_x.grad, atol=tolerance, rtol=tolerance
                )
                self.assertEqual(
                    weight.grad, reference_weight.grad, atol=tolerance, rtol=tolerance
                )


instantiate_device_type_tests(TestAMDNativePersistence, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()

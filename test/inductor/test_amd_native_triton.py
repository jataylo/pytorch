# Owner(s): ["module: inductor"]

import ast
import builtins
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.codegen.amd_kernel_optimizations import (
    optimize_flat_pointwise_source,
    optimize_row_source,
    use_triton_language,
)
from torch._inductor.runtime.amd_autotune import _neighbors
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.runtime.triton_compat import Config
from torch._inductor.runtime.triton_heuristics import (
    _amd_native_configs,
    persistent_reduction,
    pointwise,
)
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
class TestAMDNativeSource(TestCase):
    @parametrize("family", ["flat_pointwise", "row_reduction"])
    def test_native_recognizer_does_not_import_gluon(self, family):
        if family == "flat_pointwise":
            source = textwrap.dedent("""
                @triton_heuristics.pointwise(size_hints={'x': 256}, triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}}, inductor_meta={})
                @triton.jit
                def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
                    xnumel = 129
                    xoffset = tl.program_id(0) * XBLOCK
                    xindex = xoffset + tl.arange(0, XBLOCK)[:]
                    xmask = xindex < xnumel
                    x0 = xindex
                    tmp0 = tl.load(in_ptr0 + x0, xmask)
                    tmp1 = libdevice.sqrt(tmp0)
                    tl.store(out_ptr0 + x0, tmp1, xmask)
            """)
        else:
            source = textwrap.dedent("""
                @triton_heuristics.persistent_reduction(size_hints={'x': 64, 'r0_': 128}, triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}}, inductor_meta={})
                @triton.jit
                def kernel(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
                    xnumel = 37
                    r0_numel = 127
                    R0_BLOCK: tl.constexpr = 128
                    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
                    rindex = tl.arange(0, R0_BLOCK)[None, :]
                    mask = (xindex < xnumel) & (rindex < r0_numel)
                    value = tl.load(in_ptr0 + xindex * r0_numel + rindex, mask, other=0)
                    transformed = libdevice.sqrt(value)
                    result = tl.sum(transformed, 1)[:, None]
                    tl.store(out_ptr0 + xindex * r0_numel + rindex, result, mask)
            """)
        original_import = builtins.__import__

        def reject_gluon(name, *args, **kwargs):
            if "gluon" in name:
                raise AssertionError(f"native recognizer imported {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_gluon):
            if family == "flat_pointwise":
                result = optimize_flat_pointwise_source(source, use_gluon=False)
            else:
                result = optimize_row_source(source, extended=True, use_gluon=False)
        self.assertNotEqual(result, source)
        native = use_triton_language(result)
        self.assertIn(f"'amd_native_family': '{family}'", native)
        self.assertNotIn("triton.experimental", native)

    @parametrize(
        "family", ["add", "broadcast", "transpose", "flat_pointwise", "row_reduction"]
    )
    def test_native_neighbors_preserve_tail_safety(self, family):
        device = DeviceProperties(
            type="hip",
            index=0,
            multi_processor_count=8,
            cc="gfx950",
            warp_size=64,
            max_threads_per_block=1024,
            regs_per_multiprocessor=131072,
        )
        row = family == "row_reduction"
        tuner = SimpleNamespace(
            fn=SimpleNamespace(arg_names=["xnumel", "r0_numel"]),
            device_props=device,
            inductor_meta={"amd_native_family": family, "use_gluon": False},
        )
        config = Config({"XBLOCK": 1 if row else 128}, num_warps=2, num_stages=1)
        neighbors = list(
            _neighbors(tuner, config, (37 if row else 129, 128 if row else 1), {})
        )
        self.assertTrue(neighbors)
        if row:
            self.assertTrue(
                all(37 % candidate.kwargs["XBLOCK"] == 0 for candidate in neighbors)
            )
        else:
            self.assertIn(256, [candidate.kwargs["XBLOCK"] for candidate in neighbors])

    def test_removes_layout_parameters_and_preserves_nan_semantics(self):
        source = textwrap.dedent("""
            import triton
            import triton.language as tl
            from triton.experimental import gluon
            from triton.experimental.gluon import language as gl
            from triton.experimental.gluon.language.extra import libdevice
            @triton_heuristics.pointwise(
                triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr', 'AMD_VEC': 'constexpr', 'AMD_WARPS': 'constexpr', 'AMD_UNROLL': 'constexpr'}},
                inductor_meta={'use_gluon': True, 'amd_gluon_family': 'flat_pointwise'})
            @gluon.jit
            def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: gl.constexpr, AMD_VEC: gl.constexpr, AMD_WARPS: gl.constexpr, AMD_UNROLL: gl.constexpr):
                _amd_layout: gl.constexpr = gl.BlockedLayout([AMD_VEC], [64], [AMD_WARPS], [0])
                index = gl.program_id(0) * XBLOCK + gl.arange(0, XBLOCK, layout=_amd_layout)
                for iteration in range(AMD_UNROLL):
                    value = gl.load(in_ptr0 + index, index < xnumel)
                    value = gl.maximum(value, 0.0, gl.PropagateNan.ALL)
                    value = libdevice.exp(value)
                    gl.store(out_ptr0 + index, value, index < xnumel)
        """)
        result = use_triton_language(source)
        tree = ast.parse(result)
        kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        self.assertEqual(
            [argument.arg for argument in kernel.args.args],
            ["in_ptr0", "out_ptr0", "xnumel", "XBLOCK", "AMD_UNROLL"],
        )
        decorators = [
            node for node in kernel.decorator_list if isinstance(node, ast.Call)
        ]
        keywords = {keyword.arg: keyword.value for keyword in decorators[0].keywords}
        self.assertEqual(
            ast.literal_eval(keywords["inductor_meta"]),
            {"use_gluon": False, "amd_native_family": "flat_pointwise"},
        )
        signature = ast.literal_eval(keywords["triton_meta"])["signature"]
        self.assertEqual(
            list(signature), [argument.arg for argument in kernel.args.args]
        )
        self.assertIn("@triton.jit", result)
        self.assertIn("tl.maximum(value, 0.0, tl.PropagateNan.ALL)", result)
        self.assertIn("from triton.language.extra import libdevice", result)
        self.assertFalse(
            any(
                isinstance(node, ast.Name) and node.id in ("gl", "gluon")
                for node in ast.walk(tree)
            )
        )
        self.assertNotIn("triton.experimental", result)
        self.assertNotIn("_amd_layout", result)
        self.assertFalse(
            any(
                isinstance(node, ast.keyword) and node.arg == "layout"
                for node in ast.walk(tree)
            )
        )

    def test_native_source_is_unchanged(self):
        source = "@triton.jit\ndef kernel(value):\n    return tl.maximum(value, 0, tl.PropagateNan.ALL)\n"
        self.assertEqual(use_triton_language(source), source)

    def test_conversion_preserves_unrelated_imports(self):
        source = textwrap.dedent("""
            from triton.experimental import gluon, unrelated
            from triton.experimental.gluon import language as gl
            @gluon.jit
            def kernel(value):
                return gl.maximum(value, 0, gl.PropagateNan.ALL)
        """)
        result = use_triton_language(source)
        self.assertIn("from triton.experimental import unrelated", result)
        self.assertNotIn("import gluon", result)
        self.assertIn("@triton.jit", result)

    @parametrize("family", ["row_reduction", "flat_pointwise", "transpose"])
    def test_projection_deduplicates_layout_choices(self, family):
        configs = [
            Config(
                {"XBLOCK": 128, "AMD_VEC": vector, "AMD_WARPS": warps},
                num_warps=warps,
                num_stages=stages,
            )
            for vector, warps, stages in [(1, 1, 1), (4, 1, 1), (1, 2, 1), (1, 2, 2)]
        ]
        result = _amd_native_configs(configs, family)
        self.assertEqual(
            [
                (candidate.kwargs, candidate.num_warps, candidate.num_stages)
                for candidate in result
            ],
            [({"XBLOCK": 128}, 1, 1), ({"XBLOCK": 128}, 2, 1), ({"XBLOCK": 128}, 2, 2)],
        )

    @parametrize("wave", [32, 64])
    @parametrize("family", ["flat_pointwise", "transpose", "row_reduction"])
    def test_native_heuristic_configs_match_kernel_parameters(self, wave, family):
        device = DeviceProperties(
            type="hip",
            index=0,
            multi_processor_count=8,
            cc="gfx_future",
            warp_size=wave,
            max_threads_per_block=256,
            regs_per_multiprocessor=131072,
        )
        meta = {"amd_native_family": family, "amd_confirm_tuning": True}
        if family == "row_reduction":
            meta.update(amd_reduction_extent=128, amd_row_count=37)
            configs = persistent_reduction(
                {"x": 64, "r0_": 128},
                triton_meta={"device": device},
                inductor_meta=meta,
                return_configs=True,
            )
            fields = {"XBLOCK"}
        else:
            configs = pointwise(
                {"x": 4096},
                triton_meta={"device": device},
                inductor_meta=meta,
                return_configs=True,
            )
            fields = {"XBLOCK", "AMD_UNROLL", "AMD_READ_ORDER"}
        self.assertTrue(configs)
        keys = set()
        for candidate in configs:
            signature_fields = set(candidate.kwargs) - {"waves_per_eu"}
            self.assertTrue(signature_fields.issubset(fields))
            self.assertLessEqual(candidate.num_warps * wave, 256)
            key = (
                *sorted(candidate.kwargs.items()),
                candidate.num_warps,
                candidate.num_stages,
            )
            self.assertNotIn(key, keys)
            keys.add(key)


class TestAMDNativeCompiled(TestCase):
    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize(
        "family",
        [
            "softmax",
            "softmax_backward",
            "norm",
            "flat_pointwise",
            "transpose",
            "nan_minimum_maximum",
        ],
    )
    def test_compiled_paths(self, device, dtype, family):
        if torch.version.hip is None:
            self.skipTest("AMD specialization")
        torch._dynamo.reset()
        torch.manual_seed(2026)
        expected_family = "row_reduction"
        if family == "softmax":
            inputs = (torch.randn(37, 127, device=device, dtype=dtype),)

            def fn(x):
                return torch.softmax(x.float(), -1)
        elif family == "softmax_backward":
            probability = torch.softmax(torch.randn(37, 128, device=device), -1).to(
                dtype
            )
            inputs = (probability, torch.randn_like(probability))

            def fn(probability, gradient):
                value = probability.float() * gradient.float()
                return value - probability.float() * value.sum(-1, keepdim=True)
        elif family == "norm":
            x = torch.randn(37, 127, device=device, dtype=dtype)
            weight = torch.randn(127, device=device, dtype=dtype)
            inputs = (x, weight, torch.randn_like(weight))

            def fn(x, weight, bias):
                return torch.native_layer_norm(x, (127,), weight, bias, 1e-5)
        elif family == "transpose":
            inputs = (torch.randn(2, 5, 37, 64, device=device, dtype=dtype),)
            expected_family = "tiled_transpose"

            def fn(x):
                return x.permute(0, 2, 1, 3).contiguous()
        else:
            x = torch.randn(4099, device=device, dtype=dtype)
            y = torch.randn_like(x)
            expected_family = "flat_pointwise"
            if family == "nan_minimum_maximum":
                x[:4] = torch.tensor(
                    [float("nan"), float("inf"), -float("inf"), -0.0],
                    device=device,
                    dtype=dtype,
                )
                y[:4] = torch.tensor(
                    [1.0, float("nan"), 0.0, 0.0], device=device, dtype=dtype
                )

                def fn(x, y):
                    return torch.minimum(x, y), torch.maximum(x, y)
            else:

                def fn(x, y):
                    return (torch.nn.functional.silu(x.float()) * y.float()).to(x.dtype)

            inputs = (x, y)
        expected = fn(*inputs)
        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "triton.amd_further_kernel_optimizations": True,
                "triton.amd_use_gluon": False,
                "max_autotune_pointwise": True,
                "fx_graph_cache": False,
                "autotune_local_cache": False,
            }
        ):
            actual, sources = run_and_get_code(
                torch.compile(fn, fullgraph=True), *inputs
            )
        tolerance = 0.02 if dtype == torch.bfloat16 else 2e-5
        self.assertEqual(
            actual, expected, atol=tolerance, rtol=tolerance, equal_nan=True
        )
        source = "\n".join(sources)
        self.assertIn(f"'amd_native_family': '{expected_family}'", source)
        self.assertIn("@triton.jit", source)
        self.assertNotIn("triton.experimental.gluon", source)
        self.assertNotIn("from triton.experimental import gluon", source)
        self.assertNotIn("@gluon.jit", source)
        if family == "nan_minimum_maximum":
            self.assertIn("tl.PropagateNan.ALL", source)


instantiate_device_type_tests(TestAMDNativeCompiled, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()

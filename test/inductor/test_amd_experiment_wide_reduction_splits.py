# Owner(s): ["module: inductor"]

import operator

import torch
from torch._inductor.runtime.hints import amd_wide_reduction_split, DeviceProperties
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestAMDWideReductionSplit(TestCase):
    @parametrize("wave", [32, 64])
    def test_resource_relative_splits(self, wave):
        props = DeviceProperties(
            "hip", 0, 256, "gfx_future", warp_size=wave, max_threads_per_block=1024
        )
        self.assertEqual(
            amd_wide_reduction_split(props, 32768, 8, 1), 32768 // (32 * wave)
        )
        self.assertEqual(amd_wide_reduction_split(props, 32768, 37, 1), 16)
        self.assertEqual(amd_wide_reduction_split(props, 32768, 127, 1), 1)
        self.assertEqual(amd_wide_reduction_split(props, 16384, 8, 1), 1)
        self.assertEqual(amd_wide_reduction_split(props, 32768, 1, 1), 1)
        self.assertEqual(amd_wide_reduction_split(props, 32768, 8, 4), 4)

    @parametrize("wave", [32, 64])
    @parametrize("original", [2, 4, 64])
    def test_refine_existing_splits(self, wave, original):
        props = DeviceProperties(
            "hip", 0, 256, "gfx_future", warp_size=wave, max_threads_per_block=1024
        )
        target = min(65536 // (32 * wave), 64)
        self.assertEqual(
            amd_wide_reduction_split(props, 65536, 8, original, refine=True),
            max(original, target),
        )
        self.assertEqual(amd_wide_reduction_split(props, 65536, 8, original), original)
        self.assertEqual(
            amd_wide_reduction_split(props, 65536, 127, original, refine=True), original
        )

    @parametrize("backend", ["cpu", "cuda", "xpu"])
    def test_split_backend_scope(self, backend):
        props = DeviceProperties(backend, 0, 256, "other")
        self.assertEqual(amd_wide_reduction_split(props, 65536, 8, 2, refine=True), 2)

    @parametrize(
        "path", ["bare", "single", "coupled", "view", "missing_meta", "wrong_shape"]
    )
    def test_broadcast_epilogue_gate(self, path):
        from torch._inductor.ir import Reduction

        graph = torch.fx.Graph()
        full = graph.placeholder("full")
        stat = graph.call_function(torch.ops.aten.sum.dim_IntList, (full, [-1], True))
        stat.meta["val"] = torch.empty(8, 1)
        current = stat
        if path == "view":
            current = graph.call_function(torch.ops.aten.view.default, (stat, (8, 1)))
            current.meta["val"] = torch.empty(8, 1)
        if path != "bare":
            out = graph.call_function(torch.ops.aten.add.Tensor, (full, current))
            if path != "missing_meta":
                out.meta["val"] = torch.empty(
                    8, 32767 if path == "wrong_shape" else 32768
                )
            if path != "single":
                other = graph.call_function(torch.ops.aten.mean.dim, (full, [-1], True))
                other.meta["val"] = torch.empty(8, 1)
                out = graph.call_function(torch.ops.aten.sub.Tensor, (out, other))
                if path != "missing_meta":
                    out.meta["val"] = torch.empty(
                        8, 32767 if path == "wrong_shape" else 32768
                    )
            graph.output(out)
        else:
            graph.output(stat)
        self.assertEqual(
            Reduction._has_coupled_broadcast_epilogue(stat, 8 * 32768, 8, False),
            path in ("coupled", "view"),
        )
        self.assertFalse(
            Reduction._has_coupled_broadcast_epilogue(None, 8 * 32768, 8, False)
        )

    @parametrize(
        "path",
        [
            "shared_input",
            "same_stat",
            "sequential",
            "sequential_view",
            "sequential_branch",
            "long",
            "online",
            "tuple",
            "slice",
        ],
    )
    def test_statistic_locality(self, path):
        from torch._inductor.inductor_prims import prepare_softmax_online
        from torch._inductor.ir import Reduction

        graph = torch.fx.Graph()
        full = graph.placeholder("full")
        stat = graph.call_function(torch.ops.aten.sum.dim_IntList, (full, [-1], True))
        stat.meta["val"] = torch.empty(8, 1)
        current = stat
        if path in ("online", "tuple"):
            target = (
                prepare_softmax_online
                if path == "online"
                else torch.ops.aten.var_mean.correction
            )
            stat = graph.call_function(target, (full, -1))
            stat.meta["val"] = (torch.empty(8, 1), torch.empty(8, 1))
            current = graph.call_function(operator.getitem, (stat, 0))
            current.meta["val"] = torch.empty(8, 1)
            other = graph.call_function(operator.getitem, (stat, 1))
            other.meta["val"] = torch.empty(8, 1)
        elif path == "slice":
            other = graph.call_function(operator.getitem, (stat, slice(None)))
            other.meta["val"] = torch.empty(8, 1)
        elif path == "same_stat":
            other = stat
        else:
            other = graph.call_function(torch.ops.aten.mean.dim, (full, [-1], True))
            other.meta["val"] = torch.empty(8, 1)
        out = graph.call_function(torch.ops.aten.add.Tensor, (full, current))
        out.meta["val"] = torch.empty(8, 32768)
        if path == "shared_input":
            second = graph.call_function(torch.ops.aten.sub.Tensor, (full, other))
            second.meta["val"] = torch.empty(8, 32768)
            graph.output((out, second))
        else:
            if path.startswith("sequential"):
                branch = out
                if path == "sequential_view":
                    branch = graph.call_function(
                        torch.ops.aten.view.default, (out, (8, 32768))
                    )
                    branch.meta["val"] = torch.empty(8, 32768)
                elif path == "sequential_branch":
                    branch = graph.call_function(torch.ops.aten.neg.default, (out,))
                    branch.meta["val"] = torch.empty(8, 32768)
                other = graph.call_function(
                    torch.ops.aten.mean.dim, (branch, [-1], True)
                )
                other.meta["val"] = torch.empty(8, 1)
            if path == "long":
                for _ in range(70):
                    out = graph.call_function(torch.ops.aten.neg.default, (out,))
                    out.meta["val"] = torch.empty(8, 32768)
            out = graph.call_function(torch.ops.aten.sub.Tensor, (out, other))
            out.meta["val"] = torch.empty(8, 32768)
            graph.output(out)
        self.assertEqual(
            Reduction._has_coupled_broadcast_epilogue(
                stat, 8 * 32768, 8, path == "online"
            ),
            path == "online",
        )
        if path.startswith("sequential"):
            self.assertFalse(
                Reduction._has_coupled_broadcast_epilogue(other, 8 * 32768, 8, False)
            )


class TestAMDScanNumerics(TestCase):
    @parametrize(
        "family",
        [
            "softmax",
            "coupled_norm",
            "weighted_norm",
            "weighted_sum",
            "product_sum",
            "scan",
        ],
    )
    @parametrize(
        "shape", [(8, 32768), (8, 16384), (127, 32768), (8, 65536), (37, 65535)]
    )
    def test_compiled_split_decisions(self, device, family, shape):
        if torch.version.hip is None:
            self.skipTest("AMD reduction splitting")
        from unittest.mock import patch

        from torch._inductor import config, ir
        from torch._inductor.runtime.hints import amd_wide_reduction_split
        from torch._inductor.utils import run_and_get_code

        torch._dynamo.reset()
        x = torch.randint(-8, 9, shape, device=device).float() / 16
        weight = torch.randint(-8, 9, (shape[-1],), device=device).float() / 16

        def fn(x, w):
            if family == "softmax":
                return x.softmax(-1)
            if family == "coupled_norm":
                v = x * w
                return (
                    v
                    - v.sum(-1, keepdim=True) / x.shape[-1]
                    - x * (v * x).sum(-1, keepdim=True) / x.shape[-1]
                )
            if family == "weighted_norm":
                v = x * w
                return v - v.sum(-1, keepdim=True) / x.shape[-1]
            if family == "weighted_sum":
                return (x * w).sum(-1)
            if family == "product_sum":
                return x.square().sum(-1)
            return x.cumsum(-1)

        original = ir.Reduction.num_splits
        choices = {}
        for enabled in (False, True):
            torch._dynamo.reset()
            decisions = []

            def choose(*args, **kwargs):
                result = original(*args, **kwargs)
                rnumel = kwargs.get(
                    "reduction_numel", args[7] if len(args) > 7 else None
                )
                if str(rnumel) == str(shape[1]):
                    decisions.append(result)
                return result

            with (
                config.patch(
                    {
                        "triton.amd_wide_reduction_splits": True,
                        "triton.amd_split_wide_reductions": enabled,
                        "fx_graph_cache": False,
                    }
                ),
                patch.object(ir.Reduction, "num_splits", side_effect=choose),
            ):
                result, sources = run_and_get_code(
                    torch.compile(fn, fullgraph=True), x, weight
                )
            self.assertEqual(result, fn(x, weight))
            self.assertTrue(decisions)
            choices[enabled] = decisions
        props = DeviceProperties.create(torch.device(device))
        self.assertEqual(len(choices[False]), len(choices[True]))
        for baseline, actual in zip(choices[False], choices[True]):
            expected = baseline[1]
            if family in ("softmax", "coupled_norm"):
                expected = amd_wide_reduction_split(
                    props, shape[1], shape[0], expected, refine=True
                )
            self.assertEqual(actual[1], expected)
            if expected == baseline[1]:
                self.assertEqual(actual[0], baseline[0])
            else:
                self.assertGreater(
                    sum(code.count("async_compile.triton(") for code in sources), 1
                )

    @parametrize("dynamic", [False, True])
    def test_unsupported_split_inputs(self, device, dynamic):
        if torch.version.hip is None:
            self.skipTest("AMD reduction splitting")
        from unittest.mock import patch

        from torch._inductor import config

        x = torch.randint(-8, 9, (8, 32768), device=device)
        if dynamic:
            x = x.float() / 16

        def fn(x):
            return x - x.sum(-1, keepdim=True) - x.square().sum(-1, keepdim=True)

        torch._dynamo.reset()
        with (
            config.patch(
                {
                    "triton.amd_wide_reduction_splits": True,
                    "triton.amd_split_wide_reductions": True,
                    "fx_graph_cache": False,
                }
            ),
            patch(
                "torch._inductor.ir.amd_wide_reduction_split",
                wraps=amd_wide_reduction_split,
            ) as choose,
        ):
            result = torch.compile(fn, fullgraph=True, dynamic=dynamic)(x)
        self.assertEqual(result, fn(x))
        self.assertEqual(choose.call_count, 0)


instantiate_device_type_tests(TestAMDScanNumerics, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()

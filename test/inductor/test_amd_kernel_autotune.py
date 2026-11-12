# Owner(s): ["module: inductor"]

import copy
import importlib.util
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.runtime.amd_autotune import _neighbors, benchmark_configs
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.runtime.triton_compat import Config
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
class TestAMDNeighbors(TestCase):
    @parametrize("family", ["transpose", "row_reduction"])
    def test_layout_constraints(self, family):
        keywords = {
            "XBLOCK": 1 if family == "row_reduction" else 1024,
            "AMD_VEC": 4,
            "AMD_WARPS": 2,
        }
        keywords.update(
            {"AMD_LANES": 16} if family == "row_reduction" else {"AMD_UNROLL": 1}
        )
        device_props = DeviceProperties(
            type="hip",
            index=0,
            multi_processor_count=1,
            cc="gfx950",
            warp_size=64,
            max_threads_per_block=1024,
            regs_per_multiprocessor=131072,
        )
        tuner = SimpleNamespace(
            fn=SimpleNamespace(arg_names=["xnumel", "r0_numel"]),
            inductor_meta={"use_gluon": True, "amd_gluon_family": family},
            device_props=device_props,
        )
        rnumel = 128 if family == "row_reduction" else 1
        neighbors = list(
            _neighbors(
                tuner, Config(keywords, num_warps=2, num_stages=1), (37, rnumel), {}
            )
        )
        self.assertGreater(len(neighbors), 0)
        for candidate in neighbors:
            self.assertEqual(candidate.kwargs["AMD_WARPS"], candidate.num_warps)
            if family == "transpose":
                self.assertEqual(candidate.kwargs["AMD_UNROLL"], 1)
            else:
                self.assertEqual(37 % candidate.kwargs["XBLOCK"], 0)

    def test_approximate_math_opt_in(self):
        self.assertFalse(config.triton.amd_approximate_math)


class TestAMDExecutionTuning(TestCase):
    @onlyCUDA
    @parametrize("alias_output", [False, True])
    @parametrize("cache_policy", ["producer", "balanced"])
    def test_restore_strided_aliases_and_reject_partial_writer(
        self, device, alias_output, cache_policy
    ):
        if torch.version.hip is None:
            self.skipTest("AMD autotuner")
        original = torch.arange(24, device=device, dtype=torch.float32)
        expected_original = original.clone()
        x = original.as_strided((5,), (2,), 1)
        y = original.as_strided((5,), (2,), 3)
        output = x if alias_output else torch.empty_like(x)
        interface = SimpleNamespace(
            get_raw_stream=lambda index: torch.cuda.current_stream(index).cuda_stream
        )
        tuner = SimpleNamespace(
            fn=SimpleNamespace(arg_names=["in_out_ptr0", "in_ptr0", "out_ptr0"]),
            mutated_arg_names={"in_out_ptr0"},
            inductor_meta={
                "amd_warm_arg_names": ("in_ptr0",),
                "amd_cache_policy": cache_policy,
            },
            device_props=SimpleNamespace(index=torch.cuda.current_device()),
            get_device_interface=lambda: interface,
            reset_to_zero_args=lambda *args, **kwargs: None,
        )
        alias_checks = []

        def good(a, b, out, stream=None):
            alias_checks.append(
                a.untyped_storage()._cdata == b.untyped_storage()._cdata
                and a.stride() == x.stride()
                and b.storage_offset() == y.storage_offset()
            )
            value = a + b
            out.copy_(value)
            a.add_(1)

        def bad(a, b, out, stream=None):
            if alias_output:
                out.zero_()
            else:
                out[:2].copy_((a + b)[:2])
            a.add_(1)

        cache_samples = {}
        times, samples = benchmark_configs(
            tuner,
            [good, bad],
            (x, y, output),
            {},
            rounds=2,
            repeats=2,
            cache_samples=cache_samples,
        )
        self.assertEqual(
            set(cache_samples[good]),
            {"producer", "inputs_warm"} if cache_policy == "balanced" else {"producer"},
        )
        for index, value in enumerate(samples[good]):
            expected = math.exp(
                sum(math.log(values[index]) for values in cache_samples[good].values())
                / len(cache_samples[good])
            )
            self.assertAlmostEqual(value, expected)
        self.assertTrue(math.isfinite(times[good]))
        self.assertTrue(math.isinf(times[bad]))
        self.assertTrue(all(alias_checks))
        self.assertEqual(original, expected_original, atol=0, rtol=0)
        self.assertEqual(len(samples[good]), 2)
        self.assertEqual(samples[bad], [])

    @onlyCUDA
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    @parametrize("extended", [False, True])
    def test_gelu_math_accuracy(self, device, dtype, extended):
        if (
            torch.version.hip is None
            or torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            != "gfx950"
        ):
            self.skipTest("gfx950 specialization")
        x = torch.linspace(-10, 10, 4099, device=device, dtype=dtype)
        x[:4] = torch.tensor(
            [-float("inf"), float("inf"), float("nan"), -0.7517915],
            device=device,
            dtype=dtype,
        )
        g = torch.randn_like(x)

        def fn(x, g):
            return torch.ops.aten.gelu_backward(g, x, approximate="none")

        expected = fn(x, g)
        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_use_gluon": True,
                "triton.amd_approximate_math": True,
                "max_autotune_pointwise": True,
                "autotune_local_cache": False,
                "fx_graph_cache": False,
            }
        ):
            actual, source = run_and_get_code(torch.compile(fn, fullgraph=True), x, g)
            self.assertEqual(
                actual,
                expected,
                atol=0.003 if dtype == torch.bfloat16 else 2e-6,
                rtol=0.02 if dtype == torch.bfloat16 else 2e-5,
                equal_nan=True,
            )
            self.assertIn("'amd_pointwise_family': 'gelu_cdf'", "\n".join(source))

    @onlyCUDA
    @parametrize("approximate", [False, True])
    def test_saved_softmax_stats_and_all_masked(self, device, approximate):
        if (
            torch.version.hip is None
            or torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            != "gfx950"
        ):
            self.skipTest("gfx950 specialization")
        from torch._inductor import inductor_prims

        x = torch.randn(2, 3, 37, 64, device=device)
        mask = torch.rand(2, 1, 37, 64, device=device) > 0.2
        mask[:, :, 0] = False

        def fn(x, mask):
            value = x + torch.where(mask, 0.0, -float("inf"))
            maximum, total = inductor_prims.prepare_softmax_online(value, -1)
            invalid = (value == -float("inf")).all(-1, keepdim=True)
            return (
                invalid,
                maximum,
                total,
                torch.where(invalid, 0.0, torch.exp(value - maximum) / total),
            )

        expected = fn(x, mask)
        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_use_gluon": True,
                "triton.amd_approximate_math": approximate,
                "max_autotune_pointwise": True,
                "autotune_local_cache": False,
                "fx_graph_cache": False,
            }
        ):
            actual, source = run_and_get_code(
                torch.compile(fn, fullgraph=True), x, mask
            )
            self.assertEqual(actual, expected, atol=2e-5, rtol=2e-5, equal_nan=True)
            self.assertIn("'amd_gluon_family': 'row_reduction'", "\n".join(source))

    @onlyCUDA
    @parametrize("family", ["broadcast", "gelu_backward_inplace"])
    def test_frozen_backend_survives_dynamo_reset(self, device, family):
        if (
            torch.version.hip is None
            or torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            != "gfx950"
        ):
            self.skipTest("gfx950 specialization")
        path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/inductor_backends/amd_kernel_hillclimb.py"
        )
        spec = importlib.util.spec_from_file_location("amd_benchmark_for_test", path)
        benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark)
        case = dict(family=family, dtype="float32", shape=[4099], case="frozen_backend")
        if family == "gelu_backward_inplace":
            case["mutated_input_indices"] = [1]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                benchmark,
                "VARIANTS",
                ("triton", "optimized_triton", "optimized_triton_approx"),
            ),
        ):
            report = benchmark.run_case(case, Path(directory), 2, 2)
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["frozen_backend_callbacks"], [1, 1, 1])
        self.assertEqual(len(report["rows"]), 12)
        self.assertEqual(
            {row["invocation"] for row in report["rows"]}, {"direct", "graph_replay"}
        )

    @onlyCUDA
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    @parametrize("approximate", [False, True])
    def test_training_forward_and_backward(self, device, dtype, approximate):
        if (
            torch.version.hip is None
            or torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            != "gfx950"
        ):
            self.skipTest("gfx950 specialization")
        torch.manual_seed(23)
        eager = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 64),
        ).to(device=device, dtype=dtype)
        baseline = copy.deepcopy(eager)
        optimized = copy.deepcopy(eager)
        x = torch.randn(512, 128, device=device, dtype=dtype)
        expected = eager(x)
        expected.float().square().mean().backward()
        with config.patch(
            {
                "triton.amd_kernel_optimizations": False,
                "triton.amd_use_gluon": True,
                "max_autotune_pointwise": True,
                "max_autotune_gemm_backends": "ATEN",
                "fx_graph_cache": False,
                "autotune_local_cache": False,
            }
        ):
            standard = torch.compile(baseline, fullgraph=True)(x)
            standard.float().square().mean().backward()
        self.assertEqual(
            standard,
            expected,
            atol=0.02 if dtype == torch.bfloat16 else 2e-5,
            rtol=0.02 if dtype == torch.bfloat16 else 2e-5,
        )
        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_use_gluon": True,
                "triton.amd_approximate_math": approximate,
                "max_autotune_pointwise": True,
                "max_autotune_gemm_backends": "ATEN",
                "fx_graph_cache": False,
                "autotune_local_cache": False,
            }
        ):
            actual = torch.compile(optimized, fullgraph=True)(x)
            actual.float().square().mean().backward()
        self.assertEqual(
            actual,
            standard,
            atol=0.003 if dtype == torch.bfloat16 else 2e-5,
            rtol=0.02 if dtype == torch.bfloat16 else 2e-5,
        )
        for parameter, reference in zip(optimized.parameters(), baseline.parameters()):
            self.assertEqual(
                parameter.grad,
                reference.grad,
                atol=2e-5 if dtype == torch.bfloat16 else 2e-6,
                rtol=0.02 if dtype == torch.bfloat16 else 2e-4,
            )


instantiate_device_type_tests(TestAMDExecutionTuning, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()

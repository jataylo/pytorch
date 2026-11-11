# Owner(s): ["module: inductor"]

import ast
import json
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.codegen.amd_kernel_optimizations import (
    optimize_flat_pointwise_source,
    optimize_row_source,
)
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestAMDConfirmation(TestCase):
    def test_confirmed_winner_is_used_by_runtime(self):
        from torch._inductor.runtime.amd_autotune import tune
        from torch._inductor.runtime.triton_compat import Config

        class Launcher:
            def __init__(self, block):
                self.config = Config({"XBLOCK": block}, num_warps=1, num_stages=1)

        launchers = [Launcher(128 << index) for index in range(7)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "choices.jsonl"
            tuner = SimpleNamespace(
                fn=SimpleNamespace(arg_names=[]),
                device_props=SimpleNamespace(index=0),
                inductor_meta={
                    "amd_confirm_tuning": True,
                    "amd_autotune_diagnostics_path": str(path),
                },
                launchers=launchers,
                coordesc_tuner=SimpleNamespace(
                    cache_benchmark_result=lambda *args: None
                ),
            )

            def benchmark(tuner, choices, args, kwargs, **options):
                rounds = options.get("rounds", 5)
                times = {
                    launcher: (1.2 if launcher is launchers[1] else 2.0 + 0.01 * index)
                    if rounds == 9
                    else 1.0 + 0.01 * index
                    for index, launcher in enumerate(choices)
                }
                raw = {launcher: [value] * rounds for launcher, value in times.items()}
                options["cache_samples"].update(
                    {launcher: {"producer": values} for launcher, values in raw.items()}
                )
                return times, raw

            with (
                patch(
                    "torch._inductor.runtime.amd_autotune.benchmark_configs",
                    side_effect=benchmark,
                ),
                patch(
                    "torch._inductor.runtime.amd_autotune._neighbors", return_value=()
                ),
                patch("torch.cuda.mem_get_info", return_value=(1 << 40, 1 << 40)),
            ):
                timings = tune(tuner, (), {})
            self.assertIs(min(timings, key=timings.get), launchers[1])
            record = json.loads(path.read_text())
            self.assertEqual(record["selected_config"]["XBLOCK"], 256)
            self.assertEqual(record["confirmed_candidates"], 6)
            self.assertEqual(len(timings), 6)
            self.assertNotIn(launchers[-1], timings)
            self.assertEqual(
                sum(candidate["confirmed"] for candidate in record["candidates"]), 6
            )


class TestAMDIndirectMutation(TestCase):
    def test_nonidentity_x0_mutation_keeps_original(self):
        source = textwrap.dedent("""
            @triton_heuristics.pointwise(size_hints={'x':128},triton_meta={'signature':{'in_out_ptr0':'*fp32','xnumel':'i32','XBLOCK':'constexpr'}},inductor_meta={'mutated_arg_names':['in_out_ptr0']})
            @triton.jit
            def kernel(in_out_ptr0,xnumel,XBLOCK:tl.constexpr):
                xnumel=128
                xoffset=tl.program_id(0)*XBLOCK
                xindex=xoffset+tl.arange(0,XBLOCK)[:]
                xmask=xindex<xnumel
                x0=xindex^64
                x1=xindex
                tmp0=tl.load(in_out_ptr0+x0,xmask)
                tl.store(in_out_ptr0+x1,tmp0,xmask)
        """)
        self.assertEqual(optimize_flat_pointwise_source(source), source)


class TestAMDScanSource(TestCase):
    def test_lifted_scan_helper_import_order(self):
        source = textwrap.dedent("""
            @triton.jit
            def add(a,b):
                tmp=a+b
                return tmp
            @triton_heuristics.persistent_reduction(size_hints={'x':37,'r0_':32768},triton_meta={'signature':{'in_ptr0':'*bf16','out_ptr0':'*fp32','xnumel':'i32','r0_numel':'i32','XBLOCK':'constexpr'}},inductor_meta={})
            @triton.jit
            def kernel(in_ptr0,out_ptr0,xnumel,r0_numel,XBLOCK:tl.constexpr):
                xnumel=37
                r0_numel=127
                xindex=tl.program_id(0)*XBLOCK+tl.arange(0,XBLOCK)[:,None]
                r=tl.arange(0,R0_BLOCK)[None,:]
                tmp=tl.load(in_ptr0+r+127*xindex,r<127)
                result,=tl.associative_scan((tmp,),1,add)
                tl.store(out_ptr0+r+127*xindex,result,r<127)
        """)
        result = optimize_row_source(source, extended=True, further=True)
        tree = ast.parse(result)
        import_index = next(
            i
            for i, node in enumerate(tree.body)
            if isinstance(node, ast.ImportFrom) and node.module == "triton.experimental"
        )
        helper_index = next(
            i
            for i, node in enumerate(tree.body)
            if isinstance(node, ast.FunctionDef) and node.name == "add"
        )
        self.assertLess(import_index, helper_index)


class TestAMDFurtherLowering(TestCase):
    @onlyCUDA
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_cumsum_tail(self, device, dtype):
        if torch.version.hip is None:
            self.skipTest("AMD specialization")
        x = torch.randn(37, 127, device=device, dtype=dtype)

        def fn(x):
            return torch.cumsum(x.float(), -1)

        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "triton.amd_further_kernel_optimizations": True,
                "fx_graph_cache": False,
            }
        ):
            actual, source = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertEqual(actual, fn(x), atol=1e-4, rtol=1e-4)
        self.assertNotIn("gl.associative_scan", "\n".join(source))

    @onlyCUDA
    @parametrize("extent", [32767, 32768])
    def test_wide_bfloat16_scan_layout(self, device, extent):
        if torch.version.hip is None:
            self.skipTest("AMD specialization")
        x = torch.randn(37, extent, device=device, dtype=torch.bfloat16)

        def fn(x):
            return torch.cumsum(x.float(), -1)

        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "triton.amd_further_kernel_optimizations": True,
                "fx_graph_cache": False,
            }
        ):
            actual, source = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertEqual(actual, fn(x), atol=2e-4, rtol=1e-4)
        self.assertIn("gl.associative_scan", "\n".join(source))

    @onlyCUDA
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_embedding_indirect_tail_and_bounds_checks(self, device, dtype):
        if torch.version.hip is None:
            self.skipTest("AMD specialization")
        table = torch.randn(257, 127, device=device, dtype=dtype)
        indices = torch.tensor([0, 256, 4, 4, 32, 7, 256] * 5 + [1, 255], device=device)

        def fn(table, indices):
            return torch.nn.functional.embedding(indices, table)

        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "triton.amd_further_kernel_optimizations": True,
                "fx_graph_cache": False,
            }
        ):
            actual, source = run_and_get_code(
                torch.compile(fn, fullgraph=True), table, indices
            )
        self.assertEqual(actual, fn(table, indices), atol=0, rtol=0)
        code = "\n".join(source)
        self.assertNotIn("'amd_gluon_family': 'flat_pointwise'", code)
        self.assertIn("tl.device_assert", code)

    @onlyCUDA
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_adamw_multiple_mutated_outputs(self, device, dtype):
        if torch.version.hip is None:
            self.skipTest("AMD specialization")
        x = torch.randn(4099, device=device, dtype=dtype)
        g = torch.randn_like(x)
        m = torch.randn_like(x, dtype=torch.float32)
        v = torch.rand_like(m)

        def step(x, g, m, v):
            mean = m * 0.9 + g.float() * 0.1
            variance = v * 0.999 + g.float().square() * 0.001
            value = (
                x.float() * 0.9999 - 0.001 * mean / (torch.sqrt(variance) + 1e-8)
            ).to(x.dtype)
            x.copy_(value)
            m.copy_(mean)
            v.copy_(variance)
            return x, m, v

        expected = step(x.clone(), g.clone(), m.clone(), v.clone())
        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "triton.amd_further_kernel_optimizations": True,
                "fx_graph_cache": False,
            }
        ):
            actual, source = run_and_get_code(
                torch.compile(step, fullgraph=True), x, g, m, v
            )
        self.assertEqual(
            actual,
            expected,
            atol=0.02 if dtype == torch.bfloat16 else 1e-5,
            rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
        )
        self.assertEqual(actual, (x, m, v))
        self.assertIn("tl.sqrt_rn", "\n".join(source))

    @onlyCUDA
    def test_atomic_scatter_restores_original_inputs(self, device):
        if torch.version.hip is None:
            self.skipTest("AMD specialization")
        x = torch.randn(257, 127, device=device)
        indices = torch.arange(257, device=device) % 37

        def fn(x, i):
            return torch.zeros((37, 127), device=x.device).index_add(0, i, x)

        original = x.clone()
        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "triton.amd_further_kernel_optimizations": True,
                "fx_graph_cache": False,
            }
        ):
            actual, source = run_and_get_code(
                torch.compile(fn, fullgraph=True), x, indices
            )
        self.assertEqual(actual, fn(x, indices), atol=1e-4, rtol=1e-4)
        self.assertEqual(x, original, atol=0, rtol=0)
        self.assertIn("'amd_allow_atomic_tuning': True", "\n".join(source))

    @onlyCUDA
    @parametrize("extent", [8192, 32768])
    def test_long_scan_accuracy_against_original_and_float64(self, device, extent):
        if torch.version.hip is None:
            self.skipTest("AMD specialization")
        torch.manual_seed(2026)
        x = torch.randn(37, extent, device=device)

        def fn(x):
            return torch.cumsum(x, -1)

        expected = fn(x.double())
        with config.patch(
            {
                "triton.amd_kernel_optimizations": False,
                "triton.amd_extended_kernel_optimizations": False,
                "triton.amd_further_kernel_optimizations": False,
                "fx_graph_cache": False,
            }
        ):
            original = torch.compile(fn, fullgraph=True)(x).clone()
        torch._dynamo.reset()
        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "triton.amd_further_kernel_optimizations": True,
                "fx_graph_cache": False,
            }
        ):
            actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(actual, original, atol=2e-4, rtol=1e-4)
        original_error = (original.double() - expected).abs().max().item()
        actual_error = (actual.double() - expected).abs().max().item()
        self.assertLessEqual(actual_error, 1.25 * original_error + 5e-5)

    @onlyCUDA
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_cross_entropy_training_with_ignored_target(self, device, dtype):
        if torch.version.hip is None:
            self.skipTest("AMD specialization")
        x = torch.randn(37, 127, device=device, dtype=dtype, requires_grad=True)
        target = torch.arange(37, device=device) % 127
        target[0] = -100

        def fn(x, t):
            return torch.nn.functional.cross_entropy(x.float(), t)

        expected = fn(x, target)
        expected_grad = torch.autograd.grad(expected, x)[0]
        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "triton.amd_further_kernel_optimizations": True,
                "fx_graph_cache": False,
            }
        ):
            actual = torch.compile(fn, fullgraph=True)(x, target)
            actual_grad = torch.autograd.grad(actual, x)[0]
        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(
            actual_grad,
            expected_grad,
            atol=0.002 if dtype == torch.bfloat16 else 1e-5,
            rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
        )

    @onlyCUDA
    @parametrize("family", ["batch_norm", "group_norm"])
    @parametrize("channels_last", [False, True])
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_normalization_training(self, device, family, channels_last, dtype):
        if torch.version.hip is None:
            self.skipTest("AMD specialization")
        x = torch.randn(2, 32, 7, 7, device=device, dtype=dtype)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x.requires_grad_()
        if family == "batch_norm":

            def fn(x):
                return torch.nn.functional.batch_norm(
                    x.float(), None, None, training=True
                )
        else:

            def fn(x):
                return torch.nn.functional.group_norm(x.float(), 8)

        expected = fn(x)
        grad = torch.randn_like(expected)
        expected_grad = torch.autograd.grad(expected, x, grad)[0]
        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "triton.amd_further_kernel_optimizations": True,
                "fx_graph_cache": False,
            }
        ):
            actual = torch.compile(fn, fullgraph=True)(x)
            actual_grad = torch.autograd.grad(actual, x, grad)[0]
        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(
            actual_grad,
            expected_grad,
            atol=0.02 if dtype == torch.bfloat16 else 1e-4,
            rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
        )

    @onlyCUDA
    def test_mutated_input_pointer_is_validated(self, device):
        if torch.version.hip is None:
            self.skipTest("AMD specialization")
        from torch._inductor.runtime.amd_autotune import benchmark_configs

        x = torch.arange(1, 38, device=device, dtype=torch.float32)
        expected = x.clone()
        interface = SimpleNamespace(
            get_raw_stream=lambda index: torch.cuda.current_stream(index).cuda_stream
        )
        tuner = SimpleNamespace(
            fn=SimpleNamespace(arg_names=["in_ptr0"]),
            mutated_arg_names={"in_ptr0"},
            inductor_meta={"amd_graph_tuning": True},
            device_props=SimpleNamespace(index=torch.cuda.current_device()),
            get_device_interface=lambda: interface,
            reset_to_zero_args=lambda *args, **kwargs: None,
        )

        def good(value, stream=None):
            value.mul_(2)

        def bad(value, stream=None):
            value.mul_(3)

        timings, samples = benchmark_configs(
            tuner, [good, bad], (x,), {}, rounds=1, repeats=1
        )
        self.assertEqual(timings[bad], float("inf"))
        self.assertTrue(timings[good] < float("inf"))
        self.assertEqual(x, expected, atol=0, rtol=0)


instantiate_device_type_tests(TestAMDFurtherLowering, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()

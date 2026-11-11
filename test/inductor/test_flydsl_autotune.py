# Owner(s): ["module: inductor"]
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codegen.flydsl.flydsl_simd import FlyDSLSIMDScheduling
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_GPU


class TestFlyDSLSearch(TestCase):
    def candidates(self, stage, persistent=False):
        kernel = SimpleNamespace(
            stage=stage,
            xnumel=128,
            ynumel=1,
            znumel=1,
            rnumel=4096,
            is_reduction_kernel=stage in ("row", "norm", "column"),
            numels={"x": 128, "r0_": 4096}
            if stage in ("row", "norm", "column")
            else {"x": 128},
            axis_order={"x": "x", "y": "y", "z": "z"},
            in_combo=False,
            _has_atomic=lambda: False,
            ept=4,
            unroll=2,
            threads=256,
            cpair=2,
            tpr=64,
            persistent=persistent,
            one_step=False,
        )
        with patch(
            "torch._inductor.codegen.flydsl.flydsl_simd._tune_table", return_value={}
        ):
            return FlyDSLSIMDScheduling._candidates(kernel)

    def test_looped_reductions_have_candidates(self):
        values = self.candidates("row")
        self.assertIn({"row_unroll": 8}, values)
        self.assertIn({"row_tpr": 128}, values)
        self.assertIn({"threads": 512, "row_tpr": 512}, values)

    def test_persistent_algorithms_and_row_groups(self):
        values = self.candidates("norm", persistent=True)
        self.assertIn({"persistent": 0}, values)
        self.assertIn({"xunroll": 4}, values)
        self.assertIn({"ept": 2}, values)

    def test_column_geometry(self):
        values = self.candidates("column")
        self.assertIn({"threads": 64}, values)
        self.assertIn({"col_stride": 32}, values)
        self.assertIn({"col_pair": 0}, values)
        self.assertIn({"col_unroll": 8}, values)


@unittest.skipUnless(HAS_GPU and torch.version.hip, "requires ROCm")
class TestFlyDSLAutotune(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        counters.clear()
        self.enterContext(
            config.patch(
                {
                    "flydsl.simd": True,
                    "flydsl.simd_strict": True,
                    "fx_graph_cache": False,
                    "flydsl.simd_autotune_max_configs": 8,
                }
            )
        )
        self.enterContext(
            patch(
                "torch._inductor.codegen.flydsl.flydsl_simd._tune_table",
                return_value={"autotune": 1},
            )
        )

    @onlyCUDA
    @parametrize("layout", ["pointwise", "row", "norm", "column", "long_row"])
    def test_forward_backward(self, device, layout):
        shape = (32, 8192) if layout == "long_row" else (256, 512)
        x = torch.randn(shape, device=device, requires_grad=True)
        if layout == "pointwise":

            def fn(a):
                return F.silu(a) * a
        elif layout in ("row", "long_row"):

            def fn(a):
                return a.square().sum(-1)
        elif layout == "column":

            def fn(a):
                return a.square().sum(0)
        else:

            def fn(a):
                return F.layer_norm(a, (a.shape[-1],))

        reference = fn(x)
        gradient = torch.autograd.grad(reference.square().mean(), x)[0]
        actual = torch.compile(fn, dynamic=False)(x)
        actual_gradient = torch.autograd.grad(actual.square().mean(), x)[0]
        self.assertEqual(actual, reference, atol=1e-3, rtol=1e-4)
        self.assertEqual(actual_gradient, gradient, atol=1e-4, rtol=1e-4)
        self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)
        self.assertGreater(counters["flydsl_simd"]["autotuned_kernels"], 0)

    @onlyCUDA
    @parametrize("winner", [0, 1])
    @parametrize("reduction", [False, True])
    def test_backend_dispatch(self, device, winner, reduction):
        def choose(mains, args, mutated, stream=None, warm=None, metadata=None):
            mains[winner](*args, stream=stream)
            return mains[winner]

        x = torch.randn((64, 128), device=device)
        fn = (lambda a: a.sum(-1)) if reduction else (lambda a: F.silu(a) + 1)
        with (
            config.patch({"flydsl.simd_autotune_backend": True}),
            patch(
                "torch._inductor.codegen.flydsl.flydsl_simd._tune_table",
                return_value={"autotune": 0},
            ),
            patch(
                "torch._inductor.codegen.flydsl.flydsl_simd_runtime.pick_fastest",
                side_effect=choose,
            ),
        ):
            compiled = torch.compile(fn, dynamic=False)
            self.assertEqual(compiled(x), fn(x))
            self.assertEqual(compiled(x + 1), fn(x + 1))
        key = "selected_flydsl" if winner == 0 else "selected_triton"
        self.assertGreater(counters["flydsl_simd"][key], 0)

    @onlyCUDA
    def test_mutation_is_applied_once(self, device):
        x = torch.randn((64, 64), device=device)[:, ::2]
        original = x.clone()

        def fn(a):
            a.add_(2)
            return a

        with (
            config.patch(
                {"flydsl.simd_autotune_backend": True, "force_disable_caches": True}
            ),
            patch(
                "torch._inductor.codegen.flydsl.flydsl_simd._tune_table",
                return_value={"autotune": 0},
            ),
        ):
            compiled = torch.compile(fn, dynamic=False)
            self.assertEqual(compiled(x), original + 2)
            self.assertEqual(compiled(x), original + 4)
        self.assertGreater(counters["flydsl_simd"]["backend_candidates"], 0)

    @onlyCUDA
    @parametrize("triton_reference", [False, True])
    def test_reject_incorrect_candidate(self, device, triton_reference):
        from torch._inductor.codegen.flydsl.flydsl_simd_runtime import pick_fastest

        x = torch.randn(256, device=device)
        output = torch.empty_like(x)

        def correct(a, out, stream=None):
            out.copy_(a + 1)

        def incorrect(a, out, stream=None):
            out.zero_()

        mains = [incorrect, correct] if triton_reference else [correct, incorrect]
        metadata = (
            {"category": "backend_selection", "backends": ["flydsl", "triton"]}
            if triton_reference
            else None
        )
        with config.patch({"force_disable_caches": True}):
            best = pick_fastest(mains, (x, output), (1,), warm=(), metadata=metadata)
        self.assertIs(best, correct)
        self.assertEqual(output, x + 1)

    @onlyCUDA
    def test_affine_absolute_index(self, device):
        table = torch.randn((64, 16), device=device)

        def fn(a):
            row = torch.arange(a.shape[0], device=a.device)
            return a[(row[:, None] - row).abs()]

        self.assertEqual(torch.compile(fn, dynamic=False)(table), fn(table))

    @onlyCUDA
    def test_fallback_manifest(self, device):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "kernels.jsonl")
            with (
                config.patch(
                    {"flydsl.simd_strict": False, "flydsl.simd_diagnostics_path": path}
                ),
                patch.dict(
                    os.environ, {"TORCHINDUCTOR_FLYDSL_SIMD_STAGES": "pointwise"}
                ),
            ):
                x = torch.randn((64, 128), device=device)
                self.assertEqual(torch.compile(lambda a: a.sum(0))(x), x.sum(0))
            records = [json.loads(line) for line in Path(path).read_text().splitlines()]
            fallback = [
                r
                for r in records
                if r["category"] == "lowered" and r["backend"] == "triton"
            ]
            self.assertTrue(fallback)
            self.assertTrue(fallback[0]["kernel_name"])
            self.assertIn("stage column disabled", fallback[0]["reason"])
            self.assertTrue(fallback[0]["buffers"])


instantiate_device_type_tests(TestFlyDSLAutotune, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()

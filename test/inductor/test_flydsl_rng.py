# Owner(s): ["module: inductor"]
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sympy

import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters
from torch._inductor import config, inductor_prims
from torch._inductor.codegen.flydsl import flydsl_simd
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


RUNTIME_UNAVAILABLE_REASON = _flydsl_runtime_unavailable_reason()
if RUNTIME_UNAVAILABLE_REASON is None:
    from torch._inductor.codegen.flydsl import flydsl_simd_runtime as fdp
else:
    fdp = None


def philox_words(seed, offset):
    mask = (1 << 32) - 1
    key0, key1 = seed & mask, (seed >> 32) & mask
    counter0, counter1, counter2, counter3 = offset & mask, 0, 0, 0
    for _ in range(10):
        product0 = 0xD2511F53 * counter0
        product2 = 0xCD9E8D57 * counter2
        counter0, counter1, counter2, counter3 = (
            ((product2 >> 32) ^ counter1 ^ key0) & mask,
            product2 & mask,
            ((product0 >> 32) ^ counter3 ^ key1) & mask,
            product0 & mask,
        )
        key0 = (key0 + 0x9E3779B9) & mask
        key1 = (key1 + 0xBB67AE85) & mask
    return counter0, counter1, counter2, counter3


@instantiate_parametrized_tests
class TestFlyDSLRandomStreams(TestCase):
    @parametrize("fn", ["rand", "randn"])
    @parametrize("layout", ["pointwise", "reduction", "tile"])
    @parametrize("ept", [1, 2, 4, 8])
    @parametrize("shift", [0, 1])
    def test_codegen_stream_does_not_depend_on_geometry(self, fn, layout, ept, shift):
        index = sympy.Symbol("xindex", integer=True, nonnegative=True)
        offset = "logical_offset"
        kernel = SimpleNamespace(
            cse=SimpleNamespace(_cache={"offset": offset}),
            index_expr_roots={"offset": index + shift},
            _vector_var=lambda: index,
            ept=ept,
            is_reduction_kernel=layout == "reduction",
            tile2d=layout == "tile",
            load_site=0,
            _site_key=lambda *args: "site",
            _cache=lambda: "_vc",
        )
        with config.patch({"align_random_eager": False}):
            actual = FlyDSLKernel.random(kernel, fn, "seed", offset)
        packed = layout == "pointwise" and shift == 0
        suffix = "4" if ept % 4 == 0 else "4_at"
        expected = f"fdp.{fn}{suffix}(" if packed else f"fdp.{fn}("
        self.assertTrue(actual.startswith(expected), actual)

    @parametrize("mode", ["rand", "randn"])
    @parametrize("base", [-8, 0, 12, (1 << 32) - 4, (1 << 32) + 8])
    @unittest.skipIf(
        RUNTIME_UNAVAILABLE_REASON is not None, RUNTIME_UNAVAILABLE_REASON or ""
    )
    def test_scalar_packed_lookup_matches_vector_stream(self, mode, base):
        def uniform(word):
            signed = word if word < (1 << 31) else word - (1 << 32)
            return (-1 - signed if signed < 0 else signed) * 4.6566127342e-10

        arithmetic = SimpleNamespace(
            ShRUIOp=lambda value, shift: SimpleNamespace(
                result=(value & 0xFFFFFFFF) >> shift
            ),
            AndIOp=lambda a, b: SimpleNamespace(result=a & b),
        )
        with patch.multiple(
            fdp,
            _philox=philox_words,
            _uniform=uniform,
            idx=lambda value: value,
            idx32=lambda value: value & 0xFFFFFFFF,
            _val=lambda value: value,
            ArithValue=lambda value: value,
            const=lambda value, kind: value,
            _arith=arithmetic,
            sub=lambda a, b: a - b,
            mul=lambda a, b: a * b,
            eq=lambda a, b: a == b,
            lt=lambda a, b: a < b,
            bitwise_and=lambda a, b: a & b,
            where=lambda predicate, a, b: a if predicate else b,
            maximum=max,
            sqrt=math.sqrt,
            log=math.log,
            cos=math.cos,
            sin=math.sin,
        ):
            scalar = getattr(fdp, f"{mode}4_at")
            vector = getattr(fdp, f"{mode}4")
            for seed in [2026, (1 << 48) + 2026, -2026]:
                cache = {}
                for index in range(8):
                    self.assertEqual(
                        scalar(seed, base + index),
                        vector(cache, "stream", index, seed, base + index),
                    )
            for offset in [-3, 1, 5, (1 << 32) + 3, (1 << 40) + 7]:
                self.assertEqual(
                    fdp._random_counter_lane(offset),
                    ((offset & 0xFFFFFFFF) >> 2, offset & 3),
                )


@unittest.skipIf(
    RUNTIME_UNAVAILABLE_REASON is not None, RUNTIME_UNAVAILABLE_REASON or ""
)
@instantiate_parametrized_tests
class TestFlyDSLRandomIR(TestCase):
    @parametrize("mode", ["rand", "randn"])
    @parametrize("index_bits", [32, 64])
    def test_scalar_lookup_dynamic_offset_ir(self, mode, index_bits):
        from flydsl._mlir import ir as mlir
        from flydsl._mlir.dialects import func

        with mlir.Context(), mlir.Location.unknown():
            module = mlir.Module.create()
            with mlir.InsertionPoint(module.body):
                types = [
                    mlir.IntegerType.get_signless(64),
                    mlir.IntegerType.get_signless(index_bits),
                ]
                function = func.FuncOp(
                    "packed_random_at",
                    mlir.FunctionType.get(types, [mlir.F32Type.get()]),
                )
                block = function.add_entry_block()
                with mlir.InsertionPoint(block):
                    fdp.kernel_begin(wide=index_bits == 64)
                    self.addCleanup(fdp.kernel_begin)
                    value = getattr(fdp, f"{mode}4_at")(
                        *[fdp.ArithValue(arg) for arg in block.arguments]
                    )
                    func.ReturnOp([fdp._val(value)])
            self.assertTrue(module.operation.verify())


class TestFlyDSLRNG(TestCase):
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
                {"fx_graph_cache": False, "flydsl.simd_autotune_backend": False}
            )
        )

    @onlyCUDA
    @parametrize("mode", ["rand", "randn"])
    @parametrize("shape", [(256, 256), (17, 59)])
    def test_pointwise_stream_across_vector_widths(self, device, mode, shape):
        value = torch.empty(shape, device=device)
        seeds = [
            torch.tensor(seed, dtype=torch.int64, device=device)
            for seed in [2026, (1 << 48) + 2026, -2026]
        ]

        def fn(a, key):
            return inductor_prims.random(list(a.shape), key, mode)

        references = None
        for ept in [None, 1, 2, 4, 8]:
            torch._dynamo.reset()
            counters.clear()
            tuning = {"autotune": 0} if ept is None else {"autotune": 0, "ept": ept}
            finalized = []
            random_calls = []
            finalize = FlyDSLKernel.finalize_indexing
            random = FlyDSLKernel.random

            def observe_finalize(kernel, indices):
                result = finalize(kernel, indices)
                finalized.append((kernel, kernel.ept))
                return result

            def observe_random(kernel, kind, key, offset):
                source = random(kernel, kind, key, offset)
                random_calls.append((kernel, kernel.ept, source))
                return source

            with (
                config.patch(
                    {
                        "flydsl.simd": ept is not None,
                        "flydsl.simd_strict": ept is not None,
                    }
                ),
                patch.object(flydsl_simd, "_tune_table", return_value=tuning),
                patch.object(FlyDSLKernel, "finalize_indexing", new=observe_finalize),
                patch.object(FlyDSLKernel, "random", new=observe_random),
            ):
                compiled = torch.compile(fn, dynamic=False)
                actual = []
                graphs = None
                for seed in seeds:
                    actual.append(compiled(value, seed))
                    if graphs is None:
                        graphs = counters["stats"]["unique_graphs"]
                    else:
                        self.assertEqual(counters["stats"]["unique_graphs"], graphs)
            if references is None:
                references = actual
            else:
                for output, reference in zip(actual, references):
                    if mode == "rand":
                        self.assertEqual(output, reference, atol=0, rtol=0)
                    else:
                        self.assertEqual(output, reference, atol=1e-5, rtol=2e-5)
            self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)
            if ept is not None:
                self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
                self.assertTrue(finalized)
                self.assertTrue(random_calls)
                self.assertTrue(all(width == ept for _, width in finalized), finalized)
                self.assertTrue(
                    all(width == ept for _, width, _ in random_calls), random_calls
                )
                helper = f"fdp.{mode}4" + ("(" if ept % 4 == 0 else "_at(")
                self.assertTrue(
                    all(source.startswith(helper) for _, _, source in random_calls),
                    random_calls,
                )
                self.assertTrue(
                    all(
                        not kernel.is_reduction_kernel and not kernel.tile2d
                        for kernel, _, _ in random_calls
                    )
                )

    @onlyCUDA
    @parametrize("kind", ["sum", "layer_norm"])
    @parametrize("rows", [1, 32])
    def test_fused_random_reduction_stream_and_gradient(self, device, kind, rows):
        value = torch.randn((rows, 257), device=device, requires_grad=True)
        seeds = [
            torch.tensor(seed, dtype=torch.int64, device=device)
            for seed in [2026, (1 << 48) + 2026, -2026]
        ]

        def fn(a, key):
            random = inductor_prims.random(list(a.shape), key, "rand")
            mask = random >= 0.1
            dropped = torch.where(mask, a / 0.9, 0.0)
            result = (
                dropped.sum(-1)
                if kind == "sum"
                else F.layer_norm(dropped, (a.shape[-1],))
            )
            return result, mask

        probe = torch.randn((rows,) if kind == "sum" else value.shape, device=device)
        references = reference_gradients = None
        # Triton can collapse one output row to a 1D packed stream; FlyDSL keeps reduction RNG scalar.
        geometries = [None, 1, 2, 4, 8] if rows > 1 else [1, 2, 4, 8]
        for ept in geometries:
            torch._dynamo.reset()
            counters.clear()
            calls = []
            original = FlyDSLKernel.random

            def observe(kernel, fn, seed, offset):
                calls.append(kernel.is_reduction_kernel)
                return original(kernel, fn, seed, offset)

            tuning = {"autotune": 0} if ept is None else {"autotune": 0, "ept": ept}
            with (
                config.patch(
                    {
                        "flydsl.simd": ept is not None,
                        "flydsl.simd_strict": ept is not None,
                    }
                ),
                patch.object(flydsl_simd, "_tune_table", return_value=tuning),
                patch.object(FlyDSLKernel, "random", new=observe),
            ):
                compiled = torch.compile(fn, dynamic=False)
                actual, gradients = [], []
                graphs = None
                for seed in seeds:
                    output = compiled(value, seed)
                    gradients.append(torch.autograd.grad(output[0], value, probe)[0])
                    actual.append(output)
                    if graphs is None:
                        graphs = counters["stats"]["unique_graphs"]
                    else:
                        self.assertEqual(counters["stats"]["unique_graphs"], graphs)
            if references is None:
                references, reference_gradients = actual, gradients
            else:
                for output, reference, gradient, reference_gradient in zip(
                    actual, references, gradients, reference_gradients
                ):
                    self.assertEqual(output[1], reference[1])
                    self.assertEqual(output[0], reference[0], atol=2e-4, rtol=2e-4)
                    self.assertEqual(gradient, reference_gradient, atol=1e-4, rtol=2e-4)
            if ept is not None:
                self.assertTrue(
                    any(calls), "Fixture must fuse random generation into the reduction"
                )
                self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
                self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)


instantiate_device_type_tests(TestFlyDSLRNG, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()

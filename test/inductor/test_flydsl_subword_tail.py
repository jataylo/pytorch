# Owner(s): ["module: inductor"]
import re
from types import SimpleNamespace
from unittest.mock import patch

import sympy

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codegen.flydsl import flydsl_simd
from torch._inductor.codegen.flydsl.flydsl_simd import FlyDSLKernel
from torch._inductor.codegen.flydsl.flydsl_utils import (
    _flydsl_runtime_unavailable_reason,
    _simd_arch_supported,
)
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
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


DTYPES = [torch.bool, torch.int8, torch.uint8, torch.float16, torch.bfloat16]


def kernel_geometry(layout, extent, ept, dynamic=None):
    dim = {"flat": "x", "tile": "y", "reduction": "r"}[layout]
    attrs = {
        "ept": ept,
        "fast": "y" if layout == "tile" else "x",
        "is_reduction_kernel": layout == "reduction",
        "_extent": lambda name: 0,
    }
    for axis in ["x", "y", "r"]:
        attrs[f"{axis}numel"] = extent if axis == dim else 3
        attrs[f"{axis}numel_dyn"] = dynamic if axis == dim else None
    return SimpleNamespace(**attrs)


def memory_graph(storage):
    sizevars = SimpleNamespace(
        inv_precomputed_replacements={},
        statically_known_multiple_of=lambda expr, width: sympy.simplify(expr % width)
        == 0,
    )
    node = SimpleNamespace(
        get_layout=lambda: SimpleNamespace(storage_size=lambda: storage)
    )
    return SimpleNamespace(
        sizevars=sizevars,
        try_get_buffer=lambda name: node if name == "buffer" else None,
    )


@instantiate_parametrized_tests
class TestFlyDSLSubwordTailCodegen(TestCase):
    @parametrize("dtype", DTYPES)
    def test_actual_width_preserves_whole_masked_vectors(self, dtype):
        for layout in ["flat", "tile", "reduction"]:
            for extent in [256, 257, 258, 259]:
                for ept in [2, 3, 4, 8]:
                    with self.subTest(layout=layout, extent=extent, ept=ept):
                        kernel = kernel_geometry(layout, extent, ept)
                        with V.set_graph_handler(memory_graph(extent)):
                            width = FlyDSLKernel._memory_vector_width(
                                kernel, "buffer", dtype
                            )
                        expected = (
                            1 if ept == 3 or extent % 2 else (2 if extent % 4 else ept)
                        )
                        self.assertEqual(width, expected)
                        self.assertEqual(ept % width, 0)
                        self.assertEqual(extent % width, 0)

    def test_dynamic_extent_requires_proof(self):
        size = sympy.Symbol("s", integer=True, positive=True)
        for layout in ["flat", "tile", "reduction"]:
            for expression, expected in [(size, 1), (2 * size, 2), (8 * size, 8)]:
                with V.set_graph_handler(memory_graph(256)):
                    kernel = kernel_geometry(layout, 256, 8, expression)
                    self.assertEqual(
                        FlyDSLKernel._memory_vector_width(kernel, "buffer", torch.bool),
                        expected,
                    )

    def test_split_global_tail_requires_storage_proof(self):
        size = sympy.Symbol("s", integer=True, positive=True)
        for layout in ["flat", "tile", "reduction"]:
            kernel = kernel_geometry(layout, 128, 8)
            for storage, expected in [
                (257, 1),
                (258, 2),
                (259, 1),
                (256, 8),
                (size, 1),
                (2 * size, 2),
                (8 * size, 8),
            ]:
                with self.subTest(layout=layout, storage=storage):
                    with V.set_graph_handler(memory_graph(storage)):
                        self.assertEqual(
                            FlyDSLKernel._memory_vector_width(
                                kernel, "buffer", torch.bool
                            ),
                            expected,
                        )
            with V.set_graph_handler(memory_graph(256)):
                self.assertEqual(
                    FlyDSLKernel._memory_vector_width(kernel, "missing", torch.bool), 1
                )

    def test_full_word_accesses_keep_existing_width(self):
        for dtype in [torch.float32, torch.int64]:
            for layout in ["flat", "tile", "reduction"]:
                kernel = kernel_geometry(layout, 257, 8)
                self.assertEqual(
                    FlyDSLKernel._memory_vector_width(kernel, "buffer", dtype),
                    16 // dtype.itemsize,
                )


class TestFlyDSLSubwordTail(TestCase):
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
                    "fx_graph_cache": False,
                    "flydsl.simd": True,
                    "flydsl.simd_strict": True,
                    "flydsl.simd_autotune_backend": False,
                }
            )
        )

    @onlyCUDA
    @parametrize("dtype", DTYPES)
    def test_narrow_load_store_tail(self, device, dtype):
        def flat(a, destination):
            value = torch.logical_not(a) if dtype == torch.bool else a + 1
            destination.copy_(value)
            return destination

        def reduction(a, destination):
            value = torch.logical_not(a) if dtype == torch.bool else a + 1
            destination.copy_(value)
            return value.float().sum(-1), destination

        storage = {
            torch.bool: "b1",
            torch.int8: "i8",
            torch.uint8: "u8",
            torch.float16: "f16",
            torch.bfloat16: "bf16",
        }[dtype]
        for layout, fn in [("flat", flat), ("reduction", reduction)]:
            for extent in [257, 258, 259]:
                shape = (extent,) if layout == "flat" else (1, extent)
                source = (
                    (torch.arange(extent, device=device) % 7).reshape(shape).to(dtype)
                )
                wanted = (
                    torch.logical_not(source) if dtype == torch.bool else source + 1
                )
                sentinel = (
                    torch.logical_not(wanted)
                    if dtype == torch.bool
                    else torch.full_like(wanted, -7 if dtype != torch.uint8 else 247)
                )
                for ept in [2, 4, 8]:
                    with self.subTest(layout=layout, extent=extent, ept=ept):
                        torch._dynamo.reset()
                        counters.clear()
                        destination = sentinel.clone()
                        observed = []
                        original = FlyDSLKernel._memory_vector_width

                        def observe(kernel, name, actual_dtype):
                            width = original(kernel, name, actual_dtype)
                            if actual_dtype == dtype:
                                observed.append(
                                    (kernel.ept, kernel.is_reduction_kernel, width)
                                )
                            return width

                        with (
                            patch.object(
                                flydsl_simd,
                                "_tune_table",
                                return_value={"autotune": 0, "ept": ept},
                            ),
                            patch.object(
                                FlyDSLKernel, "_memory_vector_width", new=observe
                            ),
                        ):
                            actual, sources = run_and_get_code(
                                torch.compile(fn, dynamic=False), source, destination
                            )
                        self.assertEqual(destination, wanted)
                        if layout == "reduction":
                            self.assertEqual(
                                actual[0], wanted.float().sum(-1), atol=0, rtol=0
                            )
                        self.assertTrue(
                            observed,
                            "fixture must exercise narrow FlyDSL memory codegen",
                        )
                        self.assertTrue(
                            all(actual_ept == ept for actual_ept, _, _ in observed),
                            observed,
                        )
                        if layout == "reduction":
                            self.assertTrue(
                                any(is_reduction for _, is_reduction, _ in observed),
                                observed,
                            )
                        width = 2 if extent == 258 else 1
                        self.assertTrue(
                            all(
                                actual_width == width for _, _, actual_width in observed
                            ),
                            observed,
                        )
                        code = "\n".join(sources)
                        for helper in ["vload", "vstore"]:
                            lines = [
                                line
                                for line in code.splitlines()
                                if f"fdp.{helper}(" in line and repr(storage) in line
                            ]
                            if width == 1:
                                self.assertEqual(lines, [], code)
                            else:
                                self.assertTrue(lines, code)
                                self.assertTrue(
                                    all(re.search(r"_j, 2,", line) for line in lines),
                                    lines,
                                )
                        self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
                        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)


instantiate_device_type_tests(TestFlyDSLSubwordTail, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()

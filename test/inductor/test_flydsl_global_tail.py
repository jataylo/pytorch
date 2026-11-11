# Owner(s): ["module: inductor"]
import re
from types import SimpleNamespace
from unittest.mock import patch

import sympy

import torch
from torch._dynamo.utils import counters
from torch._inductor import config, ir
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


def memory_fixture(dtype, storage, extent=None, ept=4):
    node = SimpleNamespace(
        get_dtype=lambda: dtype,
        get_layout=lambda: SimpleNamespace(storage_size=lambda: storage),
    )
    graph = SimpleNamespace(
        try_get_buffer=lambda name: node if name == "buffer" else None,
        sizevars=SimpleNamespace(
            simplify=sympy.simplify,
            inv_precomputed_replacements={},
            statically_known_multiple_of=lambda expr, width: sympy.simplify(
                expr % width
            )
            == 0,
        ),
    )
    kernel = SimpleNamespace(
        ept=ept,
        fast="x",
        is_reduction_kernel=False,
        xnumel=storage,
        xnumel_dyn=extent,
        _extent=FlyDSLKernel._extent,
    )
    return kernel, graph


@instantiate_parametrized_tests
class TestFlyDSLGlobalTailCodegen(TestCase):
    @parametrize("dtype", [torch.float32, torch.int32, torch.float64, torch.int64])
    def test_large_partial_word_uses_independent_masks(self, dtype):
        count = 2**31 // dtype.itemsize + 1
        kernel, graph = memory_fixture(dtype, count)
        with V.set_graph_handler(graph):
            self.assertGreaterEqual(kernel._extent("buffer"), 2**31 - 1)
            self.assertEqual(
                FlyDSLKernel._memory_vector_width(kernel, "buffer", dtype), 1
            )

    @parametrize(
        "dtype,delta",
        [
            (torch.float32, 2),
            (torch.float32, 4),
            (torch.float64, 2),
            (torch.float64, 4),
        ],
    )
    def test_large_aligned_storage_keeps_vectorization(self, dtype, delta):
        kernel, graph = memory_fixture(dtype, 2**31 // dtype.itemsize + delta)
        with V.set_graph_handler(graph):
            self.assertEqual(
                FlyDSLKernel._memory_vector_width(kernel, "buffer", dtype),
                min(delta, 16 // dtype.itemsize),
            )

    def test_small_descriptor_keeps_existing_component_bounds(self):
        kernel, graph = memory_fixture(torch.float32, 257)
        with V.set_graph_handler(graph):
            self.assertEqual(
                FlyDSLKernel._memory_vector_width(kernel, "buffer", torch.float32), 4
            )

    def test_large_storage_alone_does_not_prove_axis(self):
        kernel, graph = memory_fixture(torch.float32, 2**29 + 4)
        kernel.xnumel = 257
        with V.set_graph_handler(graph):
            self.assertEqual(
                FlyDSLKernel._memory_vector_width(kernel, "buffer", torch.float32), 1
            )

    @parametrize("multiple", [1, 2, 4])
    def test_large_dynamic_axis_requires_divisibility(self, multiple):
        size = sympy.Symbol("s", integer=True, positive=True)
        kernel, graph = memory_fixture(torch.float32, 2**29 + 4, multiple * size)
        with V.set_graph_handler(graph):
            self.assertEqual(
                FlyDSLKernel._memory_vector_width(kernel, "buffer", torch.float32),
                multiple,
            )

    @parametrize(
        "dtype,large",
        [
            (torch.float32, False),
            (torch.float32, True),
            (torch.float64, False),
            (torch.float64, True),
        ],
    )
    def test_explicit_load_predicate_requires_descriptor(self, dtype, large):
        count = 2**31 // dtype.itemsize + 4 if large else 256
        x = sympy.Symbol("x0", integer=True)
        emitted = []
        kernel, graph = memory_fixture(dtype, count)
        graph.get_dtype = lambda name: dtype
        kernel = SimpleNamespace(
            **vars(kernel),
            args=SimpleNamespace(input=lambda name: "in_ptr0"),
            prepare_indexing=lambda expr: expr,
            kexpr=str,
            rename_indexing=lambda expr: expr,
            _root_expr=lambda expr: expr,
            uniform_vars=set(),
            _load_mask="predicate",
            _vector_var=lambda: x,
            _in_loop_region=lambda: False,
            _r_sym=lambda: None,
            compute="compute",
            body="body",
            loads="loads",
            tile2d=False,
            _xmask=lambda: "xmask",
            _rmask=lambda: None,
            _mask_arg=FlyDSLKernel._mask_arg,
            _masks=lambda **kwargs: FlyDSLKernel._masks(kernel, **kwargs),
            _memory_vector_width=lambda name,
            actual_dtype: FlyDSLKernel._memory_vector_width(kernel, name, actual_dtype),
            load_site=0,
            _site_key=lambda arg, index: "'site'",
            _cache=lambda: "_lc",
            cpair=1,
            prefetch_step=4,
            outside_loop_vars=set(),
            _persistent_region=lambda: False,
            cse=SimpleNamespace(
                generate=lambda buffer, line, **kwargs: emitted.append((buffer, line))
                or "value"
            ),
        )
        with V.set_graph_handler(graph):
            FlyDSLKernel.load(kernel, "buffer", x - 4)
        self.assertEqual(len(emitted), 1)
        buffer, line = emitted[0]
        self.assertEqual("predicate" in line, large, line)
        self.assertEqual(buffer, "compute" if large else "loads")


class TestFlyDSLGlobalTail(TestCase):
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
    @parametrize(
        "dtype,ept,delta",
        [
            (torch.float32, 2, 1),
            (torch.float32, 4, 1),
            (torch.float64, 2, 1),
            (torch.float64, 4, 1),
            (torch.float32, 4, 2),
            (torch.float32, 4, 4),
        ],
    )
    def test_large_tensor_view_preserves_adjacent_values(
        self, device, dtype, ept, delta
    ):
        def fn(source, destination):
            destination.copy_(source + 1)
            return destination

        count = 2**31 // dtype.itemsize + delta
        source_base = torch.full((count + 3,), 2, dtype=dtype, device=device)
        destination_base = torch.full((count + 3,), -7, dtype=dtype, device=device)
        source, destination = source_base[:count], destination_base[:count]
        observed = []
        mutation_stores = []
        original_store = FlyDSLKernel.store
        original = FlyDSLKernel._memory_vector_width

        def observe(kernel, name, actual_dtype):
            width = original(kernel, name, actual_dtype)
            extent = kernel._extent(name)
            if actual_dtype == dtype and extent >= 2**31 - 1:
                observed.append((name, kernel.ept, width, extent))
            return width

        def observe_store(kernel, name, index, value, mode=None):
            result = original_store(kernel, name, index, value, mode)
            if kernel._extent(name) >= 2**31 - 1:
                layout = V.graph.get_buffer(name).get_layout()
                mutation = isinstance(layout, ir.MutationLayoutSHOULDREMOVE)
                target = layout.get_buffer().get_name() if mutation else None
                mutation_stores.append(
                    {
                        "argument": kernel.args.output(name),
                        "mutation": mutation,
                        "target": target,
                        "input_target": target in V.graph.graph_inputs,
                        "size": tuple(layout.size),
                    }
                )
            return result

        torch._dynamo.reset()
        counters.clear()
        with (
            patch.object(
                flydsl_simd, "_tune_table", return_value={"autotune": 0, "ept": ept}
            ),
            patch.object(FlyDSLKernel, "_memory_vector_width", new=observe),
            patch.object(FlyDSLKernel, "store", new=observe_store),
        ):
            actual, sources = run_and_get_code(
                torch.compile(fn, dynamic=False), source, destination
            )
        indexes = torch.tensor(
            [0, count // 2, count - 3, count - 2, count - 1], device=device
        )
        self.assertEqual(
            destination[indexes],
            torch.full((5,), 3, dtype=dtype, device=device),
            atol=0,
            rtol=0,
        )
        self.assertEqual(
            destination_base[count:],
            torch.full((3,), -7, dtype=dtype, device=device),
            atol=0,
            rtol=0,
        )
        self.assertEqual(
            source_base[count:],
            torch.full((3,), 2, dtype=dtype, device=device),
            atol=0,
            rtol=0,
        )
        self.assertTrue(
            observed, "fixture must exercise global memory above the descriptor limit"
        )
        self.assertEqual(actual.data_ptr(), destination.data_ptr())
        self.assertTrue(
            any(
                store["mutation"]
                and store["input_target"]
                and store["size"] == (count,)
                for store in mutation_stores
            ),
            mutation_stores,
        )
        self.assertGreaterEqual(len({name for name, _, _, _ in observed}), 2, observed)
        self.assertTrue(
            all(actual_ept == ept for _, actual_ept, _, _ in observed), observed
        )
        wanted_width = min(delta, ept, 16 // dtype.itemsize)
        self.assertTrue(
            all(width == wanted_width for _, _, width, _ in observed), observed
        )
        self.assertTrue(
            any(
                "fdp.rsrc(" in line and str(count * dtype.itemsize) in line
                for source in sources
                for line in source.splitlines()
            ),
            sources,
        )
        self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.float64])
    def test_large_masked_cat_preserves_predicate(self, device, dtype):
        def fn(source):
            return torch.cat((source.new_zeros(4), source[:-4]))

        count = 2**31 // dtype.itemsize + 4
        source_base = torch.full((count + 4,), 2, dtype=dtype, device=device)
        source = source_base[4:]
        observed = []
        original_load = FlyDSLKernel.load

        def observe_load(kernel, name, index):
            predicate = kernel._load_mask
            buffers = [kernel.compute, kernel.loads]
            before = [buffer.getvalue() for buffer in buffers]
            value = original_load(kernel, name, index)
            if predicate is not None and kernel._extent(name) >= 2**31 - 1:
                emitted = "\n".join(
                    buffer.getvalue()[len(previous) :]
                    for buffer, previous in zip(buffers, before)
                )
                observed.append((str(predicate), emitted))
                for line in emitted.splitlines():
                    if "fdp.load(" in line or "fdp.vload(" in line:
                        self.assertIn("mask=", line)
                        self.assertRegex(
                            line.split("mask=", 1)[1],
                            rf"(?<!\w){re.escape(str(predicate))}(?!\w)",
                        )
            return value

        torch._dynamo.reset()
        counters.clear()
        with (
            patch.object(
                flydsl_simd, "_tune_table", return_value={"autotune": 0, "ept": 4}
            ),
            patch.object(FlyDSLKernel, "load", new=observe_load),
        ):
            actual, _ = run_and_get_code(torch.compile(fn, dynamic=False), source)
        self.assertEqual(actual.shape, (count,))
        self.assertEqual(
            actual[:4], torch.zeros((4,), dtype=dtype, device=device), atol=0, rtol=0
        )
        indexes = torch.tensor([4, count // 2, count - 2, count - 1], device=device)
        self.assertEqual(
            actual[indexes],
            torch.full((4,), 2, dtype=dtype, device=device),
            atol=0,
            rtol=0,
        )
        self.assertEqual(
            source_base[:4],
            torch.full((4,), 2, dtype=dtype, device=device),
            atol=0,
            rtol=0,
        )
        loads = [
            (predicate, line)
            for predicate, emitted in observed
            for line in emitted.splitlines()
            if "fdp.load(" in line or "fdp.vload(" in line
        ]
        self.assertTrue(
            loads, "fixture must emit an explicitly masked large-resource Fly load"
        )
        for predicate, line in loads:
            self.assertIn("mask=", line)
            self.assertRegex(
                line.split("mask=", 1)[1], rf"(?<!\w){re.escape(predicate)}(?!\w)"
            )
        self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)


instantiate_device_type_tests(TestFlyDSLGlobalTail, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()

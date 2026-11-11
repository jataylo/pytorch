# Owner(s): ["module: inductor"]

import os
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._dynamo.utils import counters
from torch._inductor import config, ir, metrics
from torch._inductor.codegen.simd import SIMDScheduling
from torch._inductor.dependencies import MemoryDep, ReadWrites
from torch._inductor.loop_body import MemoryUsageType
from torch._inductor.runtime.hints import ReductionHint
from torch._inductor.scheduler import (
    FusedMixOrderReductions,
    FusedSchedulerNode,
    MixOrderReduction,
    Scheduler,
    SchedulerNode,
)
from torch._inductor.virtualized import V
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.utils._ordered_set import OrderedSet


class _SizeVars:
    def simplify(self, x):
        return sympy.simplify(x)

    def optimization_hint(self, x, fallback=0):
        return int(x)

    def evaluate_expr(self, x, **kwargs):
        return bool(x)

    def statically_known_equals(self, a, b):
        return bool(sympy.simplify(a - b) == 0)

    def statically_known_leq(self, a, b):
        return bool(a <= b)

    def stride_vars(self, expr, symbols, *args):
        return [sympy.diff(expr, symbol) for symbol in symbols]


class _ReductionBuffer(ir.ComputedBuffer):
    def __init__(self):
        self._split_size = None
        self.data = SimpleNamespace(reduction_hint=ReductionHint.INNER)

    def get_reduction_type(self):
        return "sum"


class _ReductionNode(SchedulerNode):
    def __init__(self, scheduler, name, rows, cols, contiguous):
        self.scheduler = scheduler
        self.name = name
        self.node = _ReductionBuffer()
        self.group = (torch.device("cuda"), (sympy.Integer(rows), sympy.Integer(cols)))
        x, r = sympy.symbols(name + "_x " + name + "_r", integer=True, nonnegative=True)
        c = sympy.Symbol("common_linear", integer=True, nonnegative=True)
        var_ranges = {x: sympy.Integer(rows), r: sympy.Integer(cols)}
        # Dependencies use the canonical flattened index; bodies keep their loop order.
        dep = MemoryDep("common", c, (c,), (sympy.Integer(rows * cols),))
        self.read_writes = ReadWrites(
            OrderedSet([dep]), OrderedSet(), OrderedSet(), list(var_ranges), var_ranges
        )
        self.unmet_dependencies = OrderedSet([dep])
        expr = cols * x + r if contiguous else x + rows * r
        load = SimpleNamespace(buffer_name="common", index_name="load")
        self._body = SimpleNamespace(
            memory_usage={MemoryUsageType.LOAD: [load]},
            indexing_exprs={"load": expr},
            var_ranges=var_ranges,
        )
        self.ancestors = OrderedSet()
        self.min_order = self.max_order = 0
        self.min_input_distance = self.max_input_distance = 0
        self.outputs = []
        self.outputs_by_name = {}

    def get_nodes(self):
        return [self]

    def get_name(self):
        return self.name

    def get_operation_names(self):
        return OrderedSet([self.name])

    def get_buffer_names(self):
        return OrderedSet()

    def get_outputs(self):
        return []

    def get_device(self):
        return torch.device("cuda")

    def is_reduction(self):
        return True

    def is_template(self):
        return False

    def is_foreach(self):
        return False

    def is_native_matmul(self):
        return False

    def is_split_scan(self):
        return False


class _MinimalScheduler(Scheduler):
    def __init__(self):
        self.available_buffer_names = OrderedSet()
        self.mutation_renames = {}
        self.backend = SIMDScheduling(self)
        self.allow_values = []

    def can_fuse(
        self,
        a,
        b,
        can_reorder=False,
        allow_mix_order_reduction=True,
        index_equivalent_dep_names=None,
    ):
        self.allow_values.append(allow_mix_order_reduction)
        return Scheduler._can_fuse(
            self,
            a,
            b,
            can_reorder=can_reorder,
            allow_mix_order_reduction=allow_mix_order_reduction,
            index_equivalent_dep_names=index_equivalent_dep_names,
        )

    def get_backend(self, device):
        return self.backend

    def _has_multi_stream_nodes(self):
        return False

    def _has_mempool_nodes(self):
        return False

    def _nested_index_equivalent_dep_names(self, a, b):
        return None

    def dep_size_hint(self, dep, count_bytes=True):
        return int(sympy.prod(dep.size)) * (4 if count_bytes else 1)

    def can_fusion_increase_peak_memory(self, a, b):
        return False

    def fusion_prevent_too_many_reads_and_writes(self, *args):
        return False

    def are_long_distant_nodes(self, a, b):
        return False


class TestRecursiveMixedOrderCPU(TestCase):
    def setUp(self):
        super().setUp()
        self.cuda_initialized_at_entry = torch.cuda.is_initialized()
        self.gpu_calls = []
        for name in (
            "init",
            "_lazy_init",
            "current_device",
            "get_device_properties",
            "get_device_capability",
        ):
            self.enterContext(
                mock.patch.object(torch.cuda, name, side_effect=self.forbid_gpu)
            )
        self.enterContext(
            config.patch(
                {
                    "flydsl.simd": False,
                    "triton.mix_order_reduction": True,
                    "triton.mix_order_reduction_max_reads": 0,
                    "expand_dimension_for_pointwise_nodes": False,
                    "loop_index_inversion_in_fusion": False,
                    "loop_reindexing_after_fusion": False,
                }
            )
        )
        graph = SimpleNamespace(
            sizevars=_SizeVars(), cpp_wrapper=False, no_fuse_buffer_names=OrderedSet()
        )
        self.enterContext(V.set_graph_handler(graph))
        self.scheduler = _MinimalScheduler()
        self.a = _ReductionNode(self.scheduler, "row", 8192, 768, True)
        self.b = _ReductionNode(self.scheduler, "column", 768, 8192, False)
        self.c = _ReductionNode(self.scheduler, "other_reshaped_row", 768, 8192, True)
        self.d = _ReductionNode(self.scheduler, "another_column", 768, 8192, False)

    def forbid_gpu(self, *args, **kwargs):
        self.gpu_calls.append((args, kwargs))
        raise AssertionError("GPU access is forbidden in structural CPU tests")

    def tearDown(self):
        self.assertEqual(self.gpu_calls, [])
        self.assertEqual(torch.cuda.is_initialized(), self.cuda_initialized_at_entry)
        super().tearDown()

    def test_real_predicate_has_opposite_order_pair(self):
        self.assertTrue(MixOrderReduction.can_fuse(self.a, self.b))
        self.assertTrue(MixOrderReduction.can_fuse(self.a, self.c))
        self.assertTrue(MixOrderReduction.is_contiguous_node(self.a))
        self.assertTrue(MixOrderReduction.is_contiguous_node(self.c))
        self.assertFalse(MixOrderReduction.is_contiguous_node(self.b))

    def test_allow_false_rejects_inner_mixed_creation(self):
        self.assertFalse(
            self.scheduler.can_fuse(self.a, self.c, allow_mix_order_reduction=False)
        )

    def test_append_fusion_keeps_two_flat_components(self):
        group = self.scheduler.backend.fuse(self.a, self.b)
        self.assertIsInstance(group, FusedMixOrderReductions)
        self.assertTrue(group.can_fuse_with(self.c))
        fused = group.fuse_with(self.c)
        self.assertTrue(fused.can_fuse_with(self.d))
        self.assertNotIsInstance(fused.node1, FusedMixOrderReductions)
        self.assertNotIsInstance(fused.node2, FusedMixOrderReductions)
        self.assertEqual(set(fused.get_nodes()), {self.a, self.b, self.c})
        self.assertIn(False, self.scheduler.allow_values)

    def test_two_groups_merge_without_nested_components(self):
        row2 = _ReductionNode(self.scheduler, "row2", 8192, 768, True)
        column2 = _ReductionNode(self.scheduler, "column2", 768, 8192, False)
        first = self.scheduler.backend.fuse(self.a, self.b)
        second = self.scheduler.backend.fuse(row2, column2)
        self.assertTrue(first.can_fuse_with(second))
        merged = self.scheduler.backend.fuse(first, second)
        self.assertNotIsInstance(merged.node1, FusedMixOrderReductions)
        self.assertNotIsInstance(merged.node2, FusedMixOrderReductions)
        self.assertEqual(set(merged.get_nodes()), {self.a, self.b, row2, column2})

    def test_regular_same_group_still_fuses(self):
        row2 = _ReductionNode(self.scheduler, "row2", 8192, 768, True)
        self.assertTrue(
            self.scheduler.can_fuse(self.a, row2, allow_mix_order_reduction=False)
        )
        fused = self.scheduler.backend.fuse(self.a, row2)
        self.assertIsInstance(fused, FusedSchedulerNode)
        self.assertNotIsInstance(fused, FusedMixOrderReductions)


class TestRecursiveMixedOrderNumeric(TestCase):
    @parametrize("flydsl", [False, True])
    @parametrize("autotune", [False, True])
    @parametrize("shape", [(8192, 768), (172 * 1020, 768)])
    def test_row_and_two_column_reductions(self, device, flydsl, autotune, shape):
        if flydsl:
            from torch._inductor.codegen.flydsl.flydsl_utils import (
                _flydsl_runtime_unavailable_reason,
                _simd_arch_supported,
            )

            reason = _flydsl_runtime_unavailable_reason()
            if reason is not None:
                self.skipTest(reason)
            if not _simd_arch_supported(torch.cuda.current_device()):
                self.skipTest("FlyDSL requires a supported CDNA device")

        def fn(x):
            return x.sum(1), x.sum(0), x.square().sum(0)

        torch.manual_seed(2026)
        x = torch.randn(shape, device=device, requires_grad=True)
        reference = x.detach().clone().requires_grad_(True)
        expected = fn(reference)
        cotangents = tuple(torch.randn_like(value) for value in expected)
        expected_grad = torch.autograd.grad(expected, reference, cotangents)[0]
        groups = []
        original_init = FusedMixOrderReductions.__init__

        def observe_init(group, first, second):
            self.assertNotIsInstance(first, FusedMixOrderReductions)
            self.assertNotIsInstance(second, FusedMixOrderReductions)
            original_init(group, first, second)
            groups.append(len(group.get_nodes()))

        from torch._inductor.codegen.flydsl.flydsl_simd import _tune, _tune_table

        _tune_table.cache_clear()
        self.addCleanup(_tune_table.cache_clear)
        torch._dynamo.reset()
        metrics.reset()
        counters.clear()
        with (
            config.patch(
                {
                    "triton.mix_order_reduction": True,
                    "triton.mix_order_reduction_max_reads": 0,
                    "triton.cooperative_reductions": False,
                    "flydsl.simd": flydsl,
                    "flydsl.simd_strict": flydsl,
                    "flydsl.simd_autotune_backend": False,
                    "max_autotune": False,
                    "max_autotune_pointwise": autotune,
                    "triton.autotune_pointwise": autotune,
                    "conv_1x1_as_mm": False,
                    "max_autotune_gemm_backends": "ATEN",
                    "max_autotune_conv_backends": "ATEN",
                    "fx_graph_cache": False,
                }
            ),
            mock.patch.object(FusedMixOrderReductions, "__init__", observe_init),
            mock.patch.dict(
                os.environ,
                {"TORCHINDUCTOR_FLYDSL_SIMD_TUNE": f"autotune={int(autotune)}"},
            ),
        ):
            effective_tune = _tune("autotune")
            self.assertEqual(effective_tune, int(autotune))
            self.assertFalse(config.flydsl.simd_autotune_backend)
            actual = torch.compile(fn, fullgraph=True)(x)
            actual_grad = torch.autograd.grad(actual, x, cotangents)[0]
        self.assertGreater(metrics.codegen_mix_order_reduction, 0)
        self.assertTrue(groups)
        self.assertGreaterEqual(max(groups), 3)
        self.assertEqual(actual, expected, atol=2e-3, rtol=2e-3)
        self.assertEqual(actual_grad, expected_grad, atol=1e-5, rtol=1e-5)
        if flydsl:
            self.assertGreater(counters["flydsl_simd"]["flydsl_kernels"], 0)
            self.assertGreater(counters["flydsl_simd"]["mix_order"], 0)
            self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)


instantiate_device_type_tests(
    TestRecursiveMixedOrderNumeric, globals(), only_for="cuda"
)

if __name__ == "__main__":
    run_tests()

# Owner(s): ["module: inductor"]

import unittest.mock

import torch
import torch._dynamo
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


def _hot_indices(n, num_rows, device, hot_fraction=0.5, dtype=torch.int64):
    tail = torch.randint(0, num_rows, (n,), device=device, dtype=dtype)
    hot = torch.rand(n, device=device) < hot_fraction
    return torch.where(hot, torch.zeros_like(tail), tail)


def _has_native_bf16_atomics():
    from torch._inductor.kernel.scatter_preagg import _has_native_bf16_atomics

    return _has_native_bf16_atomics(torch.device(GPU_TYPE, 0))


_BF16_SCATTER_FNS = {
    "index_put": lambda x, i, v: torch.index_put(x, (i,), v, accumulate=True),
    "index_add": lambda x, i, v: x.index_add(0, i, v),
    "scatter_add": lambda x, i, v: x.scatter_add(0, i[:, None].expand_as(v), v),
    "scatter_reduce": lambda x, i, v: x.scatter_reduce(
        0, i[:, None].expand_as(v), v, "sum", include_self=False
    ),
    "scatter_reduce_add": lambda x, i, v: x.scatter(
        0, i[:, None].expand_as(v), v, reduce="add"
    ),
}


class TestScatterPreagg(TestCase):
    def setUp(self):
        super().setUp()
        patch = config.patch(scatter_preagg_enabled=True)
        patch.__enter__()
        self.addCleanup(patch.__exit__, None, None, None)
        # Deterministic mode selects a different lowering; other suites can
        # leave it on.
        guard = DeterministicGuard(False)
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)
        torch._dynamo.reset()
        counters.clear()
        torch.manual_seed(0)

    def _check(self, fn, *args, applied=True, atol=1e-3, rtol=1e-4, dynamic=False):
        expect = fn(*[a.clone() for a in args])
        actual = torch.compile(fn, fullgraph=True, dynamic=dynamic)(
            *[a.clone() for a in args]
        )
        self.assertEqual(actual, expect, atol=atol, rtol=rtol)
        self.assertEqual(
            counters["inductor"]["scatter_preagg_applied"] > 0,
            applied,
            msg=f"counters={dict(counters['inductor'])}",
        )

    def test_embedding_dense_backward(self):
        n, v, d = 50_000, 24, 32
        idx = _hot_indices(n, v, GPU_TYPE, dtype=torch.int32)
        x = torch.randn(n, 224, device=GPU_TYPE, dtype=torch.bfloat16)

        def fn(i, a):
            return torch.ops.aten.embedding_dense_backward(
                a[:, 184 : 184 + d], i, v, -1, False
            )

        self._check(fn, idx, x, atol=5e-2, rtol=2e-2)

    def test_embedding_dense_backward_padding_idx(self):
        n, v, d = 50_000, 24, 32
        idx = _hot_indices(n, v, GPU_TYPE)
        g = torch.randn(n, d, device=GPU_TYPE)

        def fn(i, a):
            return torch.ops.aten.embedding_dense_backward(a, i, v, 3, False)

        self._check(fn, idx, g)

    def test_index_put_rows(self):
        x = torch.zeros(64, 32, device=GPU_TYPE)
        idx = _hot_indices(40_000, 64, GPU_TYPE)
        v = torch.randn(40_000, 32, device=GPU_TYPE)

        def fn(x, i, v):
            return x.index_put_((i,), v, accumulate=True)

        self._check(fn, x, idx, v)

    def test_index_put_1d_target(self):
        x = torch.randn(100, device=GPU_TYPE)
        idx = _hot_indices(60_000, 100, GPU_TYPE)
        v = torch.randn(60_000, device=GPU_TYPE)

        def fn(x, i, v):
            x[i] += 0  # keep x an input mutation target
            return x.index_put_((i,), v, accumulate=True)

        self._check(fn, x, idx, v)

    def test_negative_indices(self):
        x = torch.zeros(50, 16, device=GPU_TYPE)
        idx = torch.randint(-50, 50, (30_000,), device=GPU_TYPE)
        v = torch.randn(30_000, 16, device=GPU_TYPE)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        self._check(fn, x, idx, v)

    def test_multi_index_adjacent(self):
        x = torch.zeros(8, 6, 32, device=GPU_TYPE)
        i0 = torch.randint(0, 8, (20_000,), device=GPU_TYPE)
        i1 = torch.randint(0, 6, (20_000,), device=GPU_TYPE)
        v = torch.randn(20_000, 32, device=GPU_TYPE)

        def fn(x, i0, i1, v):
            return torch.index_put(x, (i0, i1), v, accumulate=True)

        self._check(fn, x, i0, i1, v)

    def test_multi_index_non_adjacent(self):
        x = torch.zeros(8, 16, 6, device=GPU_TYPE)
        i0 = torch.randint(0, 8, (20_000,), device=GPU_TYPE)
        i1 = torch.randint(0, 6, (20_000,), device=GPU_TYPE)
        v = torch.randn(20_000, 16, device=GPU_TYPE)

        def fn(x, i0, i1, v):
            return torch.index_put(x, (i0, None, i1), v, accumulate=True)

        def fn_indexing(x, i0, i1, v):
            x[i0, :, i1] += 0
            return torch.ops.aten.index_put(x, [i0, None, i1], v, True)

        self._check(fn_indexing, x, i0, i1, v)

    def test_leading_slice(self):
        x = torch.zeros(4, 40, 16, device=GPU_TYPE)
        idx = _hot_indices(10_000, 40, GPU_TYPE)
        v = torch.randn(4, 10_000, 16, device=GPU_TYPE)

        def fn(x, i, v):
            return torch.ops.aten.index_put(x, [None, i], v, True)

        self._check(fn, x, idx, v)

    def test_multidim_index(self):
        x = torch.zeros(30, 8, device=GPU_TYPE)
        idx = torch.randint(0, 30, (100, 200), device=GPU_TYPE)
        v = torch.randn(100, 200, 8, device=GPU_TYPE)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        self._check(fn, x, idx, v)

    def test_fused_value_producer(self):
        x = torch.zeros(32, 64, device=GPU_TYPE)
        idx = _hot_indices(20_000, 32, GPU_TYPE)
        a = torch.randn(20_000, 64, device=GPU_TYPE)
        b = torch.randn(64, device=GPU_TYPE)

        def fn(x, i, a, b):
            return torch.index_put(x, (i,), (a * b).relu() + 1, accumulate=True)

        self._check(fn, x, idx, a, b)

    def test_index_add(self):
        x = torch.randn(48, 32, device=GPU_TYPE)
        idx = _hot_indices(30_000, 48, GPU_TYPE)
        src = torch.randn(30_000, 32, device=GPU_TYPE)

        def fn(x, i, s):
            return x.index_add(0, i, s, alpha=2.0)

        self._check(fn, x, idx, src)

    def test_index_add_dim1(self):
        x = torch.randn(16, 48, device=GPU_TYPE)
        idx = _hot_indices(30_000, 48, GPU_TYPE)
        src = torch.randn(16, 30_000, device=GPU_TYPE)

        def fn(x, i, s):
            return x.index_add(1, i, s)

        self._check(fn, x, idx, src)

    def test_single_destination(self):
        x = torch.zeros(1, 64, device=GPU_TYPE)
        idx = torch.zeros(65_536, dtype=torch.long, device=GPU_TYPE)
        v = torch.randn(65_536, 64, device=GPU_TYPE)

        def fn(x, i, v):
            return x.index_put_((i,), v, accumulate=True)

        self._check(fn, x, idx, v, atol=1e-2)

    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_blocks_without_duplicates(self, dtype):
        # Consecutive tokens never repeat a destination within a block, which
        # takes the path that skips the segmented scan.
        x = torch.zeros(1024, 256, device=GPU_TYPE, dtype=dtype)
        idx = torch.arange(65_536, device=GPU_TYPE) % 1024
        v = torch.randn(65_536, 256, device=GPU_TYPE, dtype=dtype)

        def fn(x, i, v):
            return x.index_put_((i,), v, accumulate=True)

        if dtype == torch.float32:
            self._check(fn, x, idx, v)
            return
        ref = fn(x.float().clone(), idx, v.float())
        out = torch.compile(fn, fullgraph=True)(x.clone(), idx, v)
        self.assertEqual(out.float(), ref, atol=ref.abs().max().item() * 8e-2, rtol=0)
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)

    def test_scatter_add_full_index(self):
        x = torch.zeros(16, 8, device=GPU_TYPE)
        idx = torch.randint(0, 16, (20_000, 8), device=GPU_TYPE)
        src = torch.randn(20_000, 8, device=GPU_TYPE)

        def fn(x, i, s):
            return x.scatter_add(0, i, s)

        self._check(fn, x, idx, src)

    def test_scatter_add_expanded_index(self):
        x = torch.zeros(16, 32, device=GPU_TYPE)
        rows = _hot_indices(20_000, 16, GPU_TYPE)
        src = torch.randn(20_000, 32, device=GPU_TYPE)

        def fn(x, r, s):
            return x.scatter_add(0, r[:, None].expand(-1, 32), s)

        self._check(fn, x, rows, src)

    def test_scatter_reduce_sum_exclude_self(self):
        x = torch.randn(16, 8, device=GPU_TYPE)
        idx = torch.randint(0, 16, (20_000, 8), device=GPU_TYPE)
        src = torch.randn(20_000, 8, device=GPU_TYPE)

        def fn(x, i, s):
            return x.scatter_reduce(0, i, s, "sum", include_self=False)

        self._check(fn, x, idx, src)

    def test_scatter_reduce_add(self):
        x = torch.randn(16, 32, device=GPU_TYPE)
        idx = _hot_indices(20_000, 16, GPU_TYPE)
        src = torch.randn(20_000, 32, device=GPU_TYPE)

        def fn(x, i, s):
            return x.scatter(0, i[:, None].expand_as(s), s, reduce="add")

        self._check(fn, x, idx, src)

    @parametrize("op", list(_BF16_SCATTER_FNS))
    def test_bf16(self, op):
        fn = _BF16_SCATTER_FNS[op]
        x = torch.randn(24, 64, device=GPU_TYPE).bfloat16()
        idx = _hot_indices(50_000, 24, GPU_TYPE)
        v = torch.randn(50_000, 64, device=GPU_TYPE).bfloat16()

        # One bf16 atomic per chunk per destination (~200 for the hot row
        # here), so expect a few percent of the row magnitude, not the single
        # rounding of an fp32 sum.
        ref = fn(x.float(), idx, v.float())
        out = torch.compile(fn, fullgraph=True)(x.clone(), idx, v)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.float(), ref, atol=ref.abs().max().item() * 8e-2, rtol=0)
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)

    @parametrize("op", ["index_put", "scatter_reduce"])
    def test_bf16_declined_falls_back(self, op):
        # Too little contention for the template: the scatter must still go to
        # the eager fallback rather than a per-element bf16 atomic kernel.
        fn = _BF16_SCATTER_FNS[op]
        x = torch.randn(10_000, 8, device=GPU_TYPE).bfloat16()
        idx = torch.randint(0, 10_000, (20_000,), device=GPU_TYPE)
        v = torch.randn(20_000, 8, device=GPU_TYPE).bfloat16()

        _, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), x.clone(), idx, v
        )
        # Without native bf16 atomics the sorted lowering takes it instead.
        applied = not _has_native_bf16_atomics()
        self._check(fn, x, idx, v, applied=applied, atol=2e-2, rtol=2e-2)
        self.assertNotIn("tl.atomic_add(", code)

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64, torch.int32])
    def test_dtypes(self, dtype):
        x = torch.zeros(20, 16, device=GPU_TYPE, dtype=dtype)
        idx = _hot_indices(20_000, 20, GPU_TYPE)
        if dtype.is_floating_point:
            v = torch.randn(20_000, 16, device=GPU_TYPE).to(dtype)
        else:
            v = torch.randint(-5, 5, (20_000, 16), device=GPU_TYPE, dtype=dtype)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        tol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-6
        if dtype in (torch.float16, torch.bfloat16):
            # Chunks are summed in fp32 but combined with output-dtype atomics,
            # so compare against an fp32 reference relative to its magnitude.
            ref = fn(x.float(), idx, v.float())
            out = torch.compile(fn, fullgraph=True)(x.clone(), idx, v)
            self.assertEqual(
                out.float(), ref, atol=ref.abs().max().item() * 8e-2, rtol=0
            )
            self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)
            return
        self._check(fn, x, idx, v, atol=tol, rtol=tol)

    @parametrize("n", [1000, 4096, 4097, 50_001])
    @config.patch(scatter_preagg_force=True)
    def test_token_tails(self, n):
        x = torch.zeros(10, 32, device=GPU_TYPE)
        idx = torch.randint(0, 10, (n,), device=GPU_TYPE)
        v = torch.randn(n, 32, device=GPU_TYPE)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        self._check(fn, x, idx, v)

    @parametrize("d", [1, 48, 37, 300])
    def test_inner_sizes(self, d):
        x = torch.zeros(12, d, device=GPU_TYPE)
        idx = _hot_indices(30_000, 12, GPU_TYPE)
        v = torch.randn(30_000, d, device=GPU_TYPE)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        self._check(fn, x, idx, v)

    def test_unchecked_out_of_range_dropped(self):
        # _unsafe_masked_index_put_accumulate clamps indices and zeroes masked values.
        x = torch.zeros(16, 8, device=GPU_TYPE)
        idx = torch.randint(-20, 20, (20_000,), device=GPU_TYPE)
        v = torch.randn(20_000, 8, device=GPU_TYPE)

        def fn(x, i, v):
            mask = ((i >= 0) & (i < 16))[:, None]
            return torch.ops.aten._unsafe_masked_index_put_accumulate(x, mask, [i], v)

        self._check(fn, x, idx, v)

    def test_codegen_uses_sort(self):
        x = torch.zeros(24, 32, device=GPU_TYPE)
        idx = _hot_indices(20_000, 24, GPU_TYPE)
        v = torch.randn(20_000, 32, device=GPU_TYPE)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        _, (code,) = run_and_get_code(torch.compile(fn), x, idx, v)
        self.assertIn("tl.sort(", code)
        self.assertIn("tl.associative_scan(", code)

    def test_configs(self):
        from torch._inductor.virtualized import V

        default = V.choices.get_scatter_preagg_configs(50_000, 32, torch.float32)
        self.assertEqual(len(default), 1)
        self.assertEqual((default[0].block_t, default[0].block_d), (256, 32))
        strided = V.choices.get_scatter_preagg_configs(
            50_000, 256, torch.float32, inner_contiguous=False
        )
        self.assertEqual((strided[0].block_t, strided[0].block_d), (512, 8))
        half = V.choices.get_scatter_preagg_configs(50_000, 64, torch.bfloat16)
        self.assertEqual((half[0].block_t, half[0].block_d), (64, 64))
        with config.patch(max_autotune=True):
            tuned = V.choices.get_scatter_preagg_configs(50_000, 32, torch.float32)
        self.assertGreater(len(tuned), 1)
        self.assertIn(default[0], tuned)
        # Ragged inner sizes must stay within the LDS-safe tile.
        for cfg in V.choices.get_scatter_preagg_configs(50_000, 37, torch.float32):
            self.assertLessEqual(cfg.block_t * cfg.block_d, 4096)

    @config.patch(max_autotune=True)
    def test_max_autotune(self):
        # Several configs are benchmarked on example inputs; the in-place
        # output must only be accumulated into once, by the chosen kernel.
        x = torch.randn(24, 64, device=GPU_TYPE)
        idx = _hot_indices(50_000, 24, GPU_TYPE)
        v = torch.randn(50_000, 64, device=GPU_TYPE)

        def fn(x, i, v):
            return x.index_put_((i,), v, accumulate=True)

        self._check(fn, x, idx, v, atol=2e-3, rtol=1e-4)

    def test_skip_low_contention(self):
        x = torch.zeros(10_000, 8, device=GPU_TYPE)
        idx = torch.randint(0, 10_000, (20_000,), device=GPU_TYPE)
        v = torch.randn(20_000, 8, device=GPU_TYPE)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        self._check(fn, x, idx, v, applied=False)
        self.assertEqual(
            counters["inductor"]["scatter_preagg_skipped_low_contention"], 1
        )

    def test_values_read_output(self):
        # The values read the scatter target, so they must be computed before
        # the template mutates it.
        def fn(x, i):
            x.index_put_((i,), x[i] * 2, accumulate=True)
            return x

        x = torch.randn(16, 64, device=GPU_TYPE)
        idx = _hot_indices(20_000, 16, GPU_TYPE)
        self._check(fn, x, idx, rtol=1e-3)

    def test_dynamic_shapes(self):
        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        compiled = torch.compile(fn, fullgraph=True)
        # Symbolic token and key counts; covers fewer tokens than a block and
        # an inner size no tile divides.
        for n, rows in [(100, 3), (20_000, 16), (30_001, 24)]:
            x = torch.zeros(rows, 72, device=GPU_TYPE)
            idx = torch.randint(0, rows, (n,), device=GPU_TYPE)
            v = torch.randn(n, 72, device=GPU_TYPE)
            expect = fn(x.clone(), idx, v)
            for t in (x, idx, v):
                torch._dynamo.mark_dynamic(t, 0)
            self.assertEqual(compiled(x, idx, v), expect, atol=1e-3, rtol=1e-3)
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)
        self.assertEqual(counters["stats"]["unique_graphs"], 1)

    def test_skip_dynamic_inner(self):
        x = torch.zeros(16, 8, device=GPU_TYPE)
        idx = torch.randint(0, 16, (20_000,), device=GPU_TYPE)
        v = torch.randn(20_000, 8, device=GPU_TYPE)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        self._check(fn, x, idx, v, applied=False, dynamic=True)
        self.assertEqual(
            counters["inductor"]["scatter_preagg_skipped_dynamic_inner"], 1
        )

    def test_output_view(self):
        def fn(i, v):
            base = v.new_zeros(64, 16)
            base[32:].t().index_add_(0, i, v)
            return base

        idx = _hot_indices(20_000, 16, GPU_TYPE)
        v = torch.randn(20_000, 32, device=GPU_TYPE)
        self._check(fn, idx, v, rtol=1e-3)

    def test_masked_value_producer_materialized(self):
        x = torch.zeros(16, 40, device=GPU_TYPE)
        idx = torch.randint(0, 16, (20_000,), device=GPU_TYPE)
        v = torch.randn(20_000, 32, device=GPU_TYPE)

        def fn(x, i, v):
            return torch.index_put(
                x, (i,), torch.nn.functional.pad(v, (4, 4)), accumulate=True
            )

        self._check(fn, x, idx, v)
        self.assertEqual(counters["inductor"]["scatter_preagg_materialized_values"], 1)

    @config.patch(partitioned_scatter_enabled=True)
    def test_partitioned_scatter_defers(self):
        x = torch.zeros(24, 32, device=GPU_TYPE)
        idx = _hot_indices(20_000, 24, GPU_TYPE)
        v = torch.randn(20_000, 32, device=GPU_TYPE)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        self._check(fn, x, idx, v)
        self.assertEqual(counters["inductor"]["partitioned_scatter_applied"], 0)
        self.assertEqual(
            counters["inductor"]["partitioned_scatter_skipped_scatter_preagg"], 1
        )

    def _check_deterministic(self, fn, *args, tol):
        ref = fn(*[a.double() if a.is_floating_point() else a.clone() for a in args])
        ref = ref.double()
        compiled = torch.compile(fn, fullgraph=True)
        with DeterministicGuard(True):
            out1, (code,) = run_and_get_code(compiled, *[a.clone() for a in args])
            out2 = compiled(*[a.clone() for a in args])
        self.assertEqual(out1.double(), ref, atol=ref.abs().max().item() * tol, rtol=0)
        self.assertTrue(torch.equal(out1, out2))
        self.assertEqual(counters["inductor"]["scatter_preagg_sorted"], 1)
        self.assertNotIn("tl.atomic_add(", code)

    @parametrize(
        "case",
        ["index_put", "index_add", "scatter_add", "one_key", "unique", "int32", "view"],
    )
    def test_deterministic(self, case):
        n, rows, d = 30_000, 50, 24
        dtype = torch.int32 if case == "int32" else torch.float32
        x = torch.zeros(rows, d, device=GPU_TYPE, dtype=dtype)
        v = torch.randn(n, d, device=GPU_TYPE).to(dtype)
        idx = _hot_indices(n, rows, GPU_TYPE)
        fn = _BF16_SCATTER_FNS.get(case, _BF16_SCATTER_FNS["index_put"])
        if case == "one_key":
            # One segment spanning every chunk.
            idx = torch.zeros_like(idx)
        elif case == "unique":
            x = torch.zeros(n, d, device=GPU_TYPE)
            idx = torch.randperm(n, device=GPU_TYPE)
        elif case == "int32":
            v = torch.randint(-5, 5, (n, d), device=GPU_TYPE, dtype=dtype)
        elif case == "view":
            x = torch.zeros(rows, 3, device=GPU_TYPE)
            v = torch.randn(n, 1, device=GPU_TYPE)

            def fn(x, i, v):
                return x[:, 1:2].index_add_(0, i, v)

        self._check_deterministic(fn, x, idx, v, tol=1e-5)

    def test_deterministic_embedding_dense_backward(self):
        idx = _hot_indices(30_000, 500, GPU_TYPE)
        grad = torch.randn(30_000, 64, device=GPU_TYPE).bfloat16()

        def fn(i, g):
            return torch.ops.aten.embedding_dense_backward(g, i, 500, -1, False)

        self._check_deterministic(fn, idx, grad, tol=1e-2)

    @parametrize("op", ["index_put", "scatter_reduce"])
    def test_bf16_without_native_atomics(self, op):
        from torch._inductor.kernel import scatter_preagg

        fn = _BF16_SCATTER_FNS[op]
        x = torch.randn(24, 64, device=GPU_TYPE).bfloat16()
        idx = _hot_indices(50_000, 24, GPU_TYPE)
        v = torch.randn(50_000, 64, device=GPU_TYPE).bfloat16()
        ref = fn(x.float(), idx, v.float())
        with unittest.mock.patch.object(
            scatter_preagg, "_has_native_bf16_atomics", lambda device: False
        ):
            _, (code,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), x.clone(), idx, v
            )
            out = torch.compile(fn, fullgraph=True)(x.clone(), idx, v)
        # One fp32 sum per destination, rounded once.
        self.assertEqual(out.float(), ref, atol=ref.abs().max().item() * 1e-2, rtol=0)
        self.assertNotIn("tl.atomic_add(", code)
        self.assertGreater(counters["inductor"]["scatter_preagg_sorted"], 0)

    @config.patch(scatter_preagg_enabled=False)
    def test_disabled(self):
        x = torch.zeros(24, 32, device=GPU_TYPE)
        idx = _hot_indices(20_000, 24, GPU_TYPE)
        v = torch.randn(20_000, 32, device=GPU_TYPE)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        self._check(fn, x, idx, v, applied=False)

    def test_backward_through_embedding(self):
        emb = torch.nn.Embedding(24, 32, device=GPU_TYPE)
        idx = _hot_indices(20_000, 24, GPU_TYPE)

        def fn(i):
            emb.zero_grad()
            emb(i).pow(2).sum().backward()
            return emb.weight.grad

        expect = fn(idx).clone()
        actual = torch.compile(fn)(idx)
        self.assertEqual(actual, expect, atol=1e-2, rtol=1e-4)
        self.assertGreater(counters["inductor"]["scatter_preagg_applied"], 0)


instantiate_parametrized_tests(TestScatterPreagg)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()

# Owner(s): ["module: inductor"]
"""FlyDSL codegen for Inductor's pointwise and reduction kernels (config.flydsl.simd).

The numerical tests run in strict mode, where a kernel FlyDSL cannot express raises
instead of falling back to Triton, so a pass means the kernel really was FlyDSL. The
fallback tests run without it and check the reason that was counted. The access
classifier tests are pure sympy and run anywhere.
"""

import contextlib
import unittest
from unittest.mock import patch

import sympy

import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.utils._sympy.functions import FloorDiv, ModularIndexing


def _skip_reason():
    if not HAS_GPU:
        return "no GPU"
    if not torch.version.hip:
        return "FlyDSL is ROCm only"
    try:
        from torch._inductor.kernel.flex.flydsl_flash_attention import (
            flydsl_unavailable_reason,
        )
    except ImportError as e:
        return f"could not import FlyDSL: {e}"
    return flydsl_unavailable_reason()


SKIP_REASON = _skip_reason()
requires_flydsl = unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")


class TestClassifyAccess(TestCase):
    """How a load's index behaves over one thread's run of consecutive elements."""

    def setUp(self):
        super().setUp()
        from torch._inductor.codegen.flydsl.flydsl_simd import classify_access

        self.classify = classify_access
        self.x = sympy.Symbol("xindex", integer=True, nonnegative=True)

    def test_contiguous(self):
        x = self.x
        self.assertEqual(self.classify(x, x, 4), "contig")
        self.assertEqual(self.classify(x + 1024, x, 4), "contig")
        self.assertEqual(
            self.classify(ModularIndexing(x, 1, 768) + 768 * FloorDiv(x, 768), x, 4),
            "contig",
        )

    def test_misaligned_base_is_not_contiguous(self):
        # Offsets base+j with base not a multiple of the width would straddle chunks.
        self.assertEqual(self.classify(self.x + 1, self.x, 4), "other")

    def test_uniform(self):
        x = self.x
        self.assertEqual(self.classify(FloorDiv(x, 64), x, 4), "uniform")
        self.assertEqual(self.classify(ModularIndexing(x, 256, 3), x, 4), "uniform")
        self.assertEqual(self.classify(sympy.Integer(7), x, 4), "uniform")

    def test_gather(self):
        x = self.x
        self.assertEqual(self.classify(2 * x, x, 4), "other")
        self.assertEqual(self.classify(ModularIndexing(x, 1, 6), x, 4), "other")
        self.assertEqual(self.classify(FloorDiv(x, 3), x, 4), "other")

    def test_split_reduction_shift(self):
        # A split reduction indexes ``r + rnumel * x`` through the input's view.
        r = self.x
        x1 = sympy.Symbol("x1", integer=True, nonnegative=True)
        e = 262144 * ModularIndexing(r + 87382 * x1, 4096, 64) + ModularIndexing(
            r + 87382 * x1, 1, 4096
        )
        self.assertEqual(self.classify(e, r, 2), "contig")
        self.assertEqual(self.classify(e, r, 4), "other")
        self.assertEqual(
            self.classify(ModularIndexing(r + 87383 * x1, 1, 4096), r, 2), "other"
        )

    def test_nested_view(self):
        # Llama's head permute reads (B, H, S, D) through the output's x % (H * D).
        x = self.x
        x0, x1 = ModularIndexing(x, 1, 1024), FloorDiv(x, 1024)
        e = (
            64 * ModularIndexing(x1, 1, 1024)
            + 65536 * FloorDiv(x0, 64)
            + 1048576 * FloorDiv(x1, 1024)
            + ModularIndexing(x0, 1, 64)
        )
        self.assertEqual(self.classify(e, x, 8), "contig")
        self.assertEqual(self.classify(ModularIndexing(x0, 1, 60), x, 8), "other")
        self.assertEqual(self.classify(FloorDiv(x0, 4), x, 8), "other")


@requires_flydsl
@instantiate_parametrized_tests
class TestFlyDSLSIMD(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        counters.clear()
        self.patch = config.patch(
            {"flydsl.simd": True, "flydsl.simd_strict": True, "fx_graph_cache": False}
        )
        self.patch.__enter__()

    def tearDown(self):
        self.patch.__exit__(None, None, None)
        super().tearDown()

    def check(self, fn, *args, atol=1e-4, rtol=1e-4, kinds=None):
        ref = fn(*args)
        out = torch.compile(fn, dynamic=False)(*args)
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol, equal_nan=True)
        c = counters["flydsl_simd"]
        self.assertGreater(c["flydsl_kernels"], 0)
        self.assertEqual(c["triton_kernels"], 0)
        for kind in kinds or ():
            self.assertGreater(c[kind], 0, kind)

    # -- pointwise -------------------------------------------------------------------

    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_pointwise_broadcast(self, dtype):
        a = torch.randn(1000, 777, device="cuda", dtype=dtype)
        b = torch.randn(777, device="cuda", dtype=dtype)
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        self.check(
            lambda x, y: F.silu(x) * y + 1,
            a,
            b,
            atol=tol,
            rtol=tol,
            kinds=["pointwise"],
        )

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_paired_half_atomics(self, dtype):
        # An even row width packs a thread's two elements into one atomic add; an odd
        # one adds an element at a time.
        from torch._inductor.utils import run_and_get_code

        for d, paired in ((64, True), (7, False)):
            torch._dynamo.reset()
            ix = torch.randint(0, 50, (300,), device="cuda")
            g = torch.randn(300, d, device="cuda").to(dtype)

            def fn(ix, g):
                z = torch.zeros(50, d, device="cuda", dtype=dtype)
                return z.index_put_((ix,), g, accumulate=True)

            out, (code,) = run_and_get_code(torch.compile(fn, dynamic=False), ix, g)
            ref = torch.zeros(50, d, device="cuda").index_put_(
                (ix,), g.float(), accumulate=True
            )
            torch.testing.assert_close(out.float(), ref, atol=0.1, rtol=0.02)
            self.assertEqual("fdp.atomic_add_pair(" in code, paired)

    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.int8])
    def test_wave_uniform_load(self, dtype):
        # A per-channel operand is the same across a wave's lanes and is read with one
        # scalar load from lane 0's offset; sub-dword elements come out of their word,
        # and 3 * 8 * 1024 elements leave the last block partly masked. With 5 channels
        # of a sub-dword type the last word is partial, out of a scalar load's reach.
        from torch._inductor.utils import run_and_get_code

        for c in (8, 5):
            torch._dynamo.reset()
            x = torch.randn(3, c, 16, 64, device="cuda").to(dtype)
            w = (torch.arange(c, device="cuda") - 2).to(dtype)
            fn = torch.compile(lambda a, b: a * b[:, None, None] + 1, dynamic=False)
            out, (code,) = run_and_get_code(fn, x, w)
            ref = (x.float() * w.float()[:, None, None] + 1).to(dtype)
            torch.testing.assert_close(out, ref)
            if c == 8:
                self.assertIn("fdp.wload(", code)

    def test_integer_and_bool(self):
        i = torch.randint(-100, 100, (5000,), device="cuda")
        self.check(lambda x: (x // 3 + x % 7) * 2, i)
        self.check(lambda x: ((x * 7) > 13) | (x < -50), i)

    def test_casts(self):
        a = torch.randn(1000, 777, device="cuda")
        self.check(
            lambda x: (x * 30).clamp(-120, 120).to(torch.int8).float()
            + (x > 0).to(torch.uint8),
            a,
        )
        # A float cast inside a kernel computes in f32, as Triton's does: Inductor
        # asks for eager's rounding only under emulate_precision_casts.
        tri, fly = self._vs_triton(lambda x: x.to(torch.bfloat16).float() * 3, a)
        self.assertEqual(tri, fly, atol=0, rtol=0)
        with config.patch({"emulate_precision_casts": True}):
            self.check(lambda x: x.to(torch.bfloat16).float() * 3, a)

    def test_bf16_rounding(self):
        # Round to nearest even, ties and the overflow to inf included; NaN stays NaN.
        x = torch.randn(1 << 16, device="cuda") * 1e3
        edges = [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 3.4e38,
                 1.00390625, 1.01171875, -2.5, 1e-40, -1e-40, 2.0**-130]  # fmt: skip
        x[: len(edges)] = torch.tensor(edges, device="cuda")
        y = torch.compile(lambda a: (a * 1.0).to(torch.bfloat16))(x)
        ref = x.to(torch.bfloat16)
        same = (y.view(torch.int16) == ref.view(torch.int16)) | (
            y.isnan() & ref.isnan()
        )
        self.assertTrue(bool(same.all()))

    def test_transcendentals(self):
        a = torch.randn(4097, 33, device="cuda")
        self.check(
            lambda x: torch.where(x > 0, x.exp(), x.sin()) + x.tanh() * x.erf(), a
        )

    def test_sin_cos(self):
        # One angle reduction shared by a sin and a cos of the same value, OCML's for
        # angles past 1e5; inf and NaN give NaN.
        for scale in (1.0, 1e3, 3e5):
            torch._dynamo.reset()
            a = torch.randn(1 << 16, device="cuda") * scale
            a[:4] = torch.tensor([float("inf"), -float("inf"), float("nan"), -0.0])
            self.check(lambda x: x.sin() * 2 + x.cos(), a, atol=1e-6, rtol=1e-5)

    def test_strided(self):
        a = torch.randn(1000, 1000, device="cuda")
        self.check(lambda x: x * 2 + 1, a[:, :777])

    def test_transposed_tiles(self):
        # {y, x} tiling: transposed loads staged through LDS; ragged edges masked.
        a = torch.randn(1000, 777, device="cuda")
        self.check(lambda x: x.t() + 1, a)
        torch._dynamo.reset()
        self.check(lambda x: x + x.t(), torch.randn(1000, 1000, device="cuda"))
        torch._dynamo.reset()
        q = torch.randn(8, 128, 12, 64, device="cuda", dtype=torch.bfloat16)
        self.check(
            lambda x: x.permute(0, 2, 1, 3).contiguous() * 2, q, atol=1e-2, rtol=1e-2
        )
        torch._dynamo.reset()
        i = torch.randint(0, 100, (300, 500), device="cuda")
        self.check(lambda x: x.t().contiguous() + 1, i)
        torch._dynamo.reset()
        # Rectangular tiles (an autotuning candidate), ragged on both edges.
        from torch._inductor.codegen.flydsl.flydsl_simd import _tune_override

        with _tune_override({"tile_f": 16, "tile_s": 64, "autotune": 0}):
            self.check(lambda x: x.t() + 1, a)
            torch._dynamo.reset()
            self.check(lambda x: x.t().contiguous() * 2, i)
        torch._dynamo.reset()
        # Blocks wider than a staging pass of full vectors covers.
        h = torch.randn(512, 768, device="cuda", dtype=torch.bfloat16)
        for threads in (256, 512):
            with _tune_override({"threads": threads, "autotune": 0}):
                self.check(lambda x: x.t().contiguous() * 2, h, atol=1e-2, rtol=1e-2)
                torch._dynamo.reset()
                self.check(lambda x: x.t().contiguous() * 2, a)
            torch._dynamo.reset()
        c, b = torch.randn(4, 64, 30, 30, device="cuda"), torch.randn(64, device="cuda")
        self.check(
            lambda c, b: (c + b.view(1, -1, 1, 1)).to(
                memory_format=torch.channels_last
            ),
            c,
            b,
        )

    def test_foreach_combo(self):
        # Foreach ops become one combo launch: sub-kernels on ranges of block ids.
        ps = [torch.randn(s, device="cuda") for s in (1000, 4096, 777, 3)]
        gs = [torch.randn_like(p) for p in ps]

        def step(ps, gs):
            ms = torch._foreach_mul(gs, 0.1)
            out = torch._foreach_addcdiv(
                ps,
                ms,
                torch._foreach_add(torch._foreach_sqrt(torch._foreach_abs(gs)), 1e-8),
            )
            return out, torch._foreach_norm(gs)

        self.check(step, ps, gs)
        self.assertGreater(counters["flydsl_simd"]["combo"], 0)

    def test_foreach_combo_merged(self):
        # Same-shaped parameters share one traced body, each choosing its buffers
        # under a switch on the member index; the odd one out keeps its own branch.
        from torch._inductor.utils import run_and_get_code

        ps = [torch.randn(s, device="cuda") for s in (4096,) * 5 + (777,)]
        gs = [torch.randn_like(p) for p in ps]

        def step(ps, gs):
            torch._foreach_add_(ps, torch._foreach_mul(gs, 0.1))
            return ps

        ref = step([p.clone() for p in ps], gs)
        out, (code,) = run_and_get_code(torch.compile(step, dynamic=False), ps, gs)
        torch.testing.assert_close(out, ref)
        self.assertIn("fdp.pick(_s, lambda: (", code)

    def test_indirect(self):
        w = torch.randn(5000, 128, device="cuda")
        ix = torch.randint(0, 5000, (64, 33), device="cuda")
        self.check(lambda w, ix: F.embedding(ix, w) * 2, w, ix)

    def test_masked_load_order(self):
        # cat loads under a computed mask: the load must follow the mask's definition.
        a = torch.randn(1000, 777, device="cuda")
        self.check(lambda x, y: torch.cat([x.sin(), y.cos()], 0), a, a[:10])

    # -- reductions ------------------------------------------------------------------

    @parametrize("shape", [(1000, 777), (8, 100000), (100000, 7), (64, 4096)])
    def test_sum_rows(self, shape):
        a = torch.randn(*shape, device="cuda")
        self.check(lambda x: x.sum(-1), a, atol=1e-2, rtol=1e-3)

    def test_column_reduction(self):
        a = torch.randn(1000, 777, device="cuda")
        self.check(lambda x: x.sum(0), a, atol=1e-3, kinds=["reduction"])

    def test_no_mix_order_reduction(self):
        # A row and a column reduction of the same input (a norm's backward) fuse into
        # one Triton-only mix-order kernel; with FlyDSL they are generated apart.
        from torch._inductor.utils import run_and_get_code

        x = torch.randn(8192, 1024, device="cuda")
        w = torch.randn(1024, device="cuda")
        fn = lambda x, w: ((x * w).sum(1), (x * x).sum(0))  # noqa: E731
        with config.patch({"flydsl.simd": False}):
            _, (code,) = run_and_get_code(torch.compile(fn, dynamic=False), x, w)
        self.assertEqual(code.count("@triton.jit"), 1)
        torch._dynamo.reset()
        out, (code,) = run_and_get_code(torch.compile(fn, dynamic=False), x, w)
        torch.testing.assert_close(out, fn(x, w), atol=1e-2, rtol=1e-4)
        self.assertNotIn("@triton.jit", code)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)

    @parametrize("rows", [1, 3, 17])
    def test_short_column_reduction(self, rows):
        # No more lanes a column than rows: 3 rows of 64 columns take 4 lanes each.
        from torch._inductor.utils import run_and_get_code

        a = torch.randn(rows, 64, device="cuda")
        b = torch.randn(rows, 64, device="cuda")
        fn = lambda a, b: (a * b).sum(0) + a.mean(0)  # noqa: E731
        with config.patch({"unroll_reductions_threshold": 1}):
            out, (code,) = run_and_get_code(torch.compile(fn, dynamic=False), a, b)
        torch.testing.assert_close(out, fn(a, b), atol=1e-4, rtol=1e-4)
        if rows == 3:
            self.assertIn("tpr=4 stride=64", code)

    @parametrize("ept", [0, 8])
    def test_stored_row_reloaded(self, ept):
        # A persistent row stores the gathered embedding, then normalizes it from the
        # registers; at 8 elements a lane the last run of 768 is masked.
        from torch._inductor.codegen.flydsl.flydsl_simd import _tune_override

        idx = torch.randint(0, 1000, (64, 32), device="cuda")
        e = torch.randn(1000, 768, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(768, device="cuda", dtype=torch.bfloat16)

        def fn(idx, e, w):
            x = torch.nn.functional.embedding(idx, e) + 1
            y, m, r = torch.ops.aten.native_layer_norm(x, [768], w, None, 1e-5)
            return x, y, m, r

        with _tune_override({"ept": ept} if ept else {}):
            out = torch.compile(fn, dynamic=False)(idx, e, w)
        for o, ref in zip(out, fn(idx, e, w)):
            torch.testing.assert_close(o, ref, atol=2e-2, rtol=2e-2)

    @parametrize(
        "op",
        ["amax", "amin", "prod", "any", "argmax", "argmin", "var_mean", "logsumexp"],
    )
    def test_reduction_types(self, op):
        a = torch.randn(1000, 777, device="cuda")
        fns = {
            "amax": lambda x: x.amax(1),
            "amin": lambda x: x.amin(1),
            "prod": lambda x: x[:, :20].prod(1),
            "any": lambda x: (x > 3.2).any(1),
            "argmax": lambda x: x.argmax(1),
            "argmin": lambda x: x.argmin(1),
            "var_mean": lambda x: torch.var_mean(x, 1),
            "logsumexp": lambda x: torch.logsumexp(x, 1),
        }
        self.check(fns[op], a, atol=1e-3, rtol=1e-3, kinds=["reduction"])

    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_softmax_layernorm(self, dtype):
        h = torch.randn(64, 4096, device="cuda", dtype=dtype)
        w = torch.randn(4096, device="cuda", dtype=dtype)
        tol = 1e-4 if dtype == torch.float32 else 5e-2
        self.check(lambda x: torch.softmax(x, -1), h, atol=tol, rtol=tol)
        torch._dynamo.reset()
        self.check(lambda x, w: F.layer_norm(x, (4096,), w), h, w, atol=tol, rtol=tol)

    @parametrize("width", [32, 8])
    def test_short_rows_stored_together(self, width):
        # Each row group's results go out in one store, a lane a row.
        x = torch.randn(1 << 17, width, device="cuda")
        self.check(lambda x: (x.sum(-1), x.amax(-1), x.argmax(-1)), x)
        torch._dynamo.reset()
        self.check(lambda x: torch.var_mean(x, -1), x)
        torch._dynamo.reset()
        i = torch.randint(-(1 << 40), 1 << 40, (1 << 17, width), device="cuda")
        self.check(lambda i: i.sum(-1), i)

    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.int8, torch.bool])
    def test_constant_index_load(self, dtype):
        # A broadcast element goes through the scalar cache, the sub-dword ones shifted
        # out of their dword.
        x = torch.randn(4096, 64, device="cuda")
        s = torch.arange(-3, 9, device="cuda").to(dtype)
        for i in (0, 3, 5):
            torch._dynamo.reset()
            self.check(lambda x, s: x * s[i] + s[i + 1], x, s)

    @parametrize("width", [1024, 700])
    def test_stored_then_reread(self, width):
        # The residual sum is stored in the first pass and read back in the second; 700
        # leaves masked lanes, whose stored register must not stand in for the reload.
        a, b = (
            torch.randn(256, width, device="cuda", dtype=torch.bfloat16) for _ in "ab"
        )
        w = torch.randn(width, device="cuda", dtype=torch.bfloat16)

        def fn(a, b, w):
            h = a + b
            return h, h * torch.rsqrt(
                h.float().pow(2).mean(-1, keepdim=True) + 1e-6
            ) * w

        self.check(fn, a, b, w, atol=5e-2, rtol=5e-2)

    @parametrize("width", [768, 700, 96])
    def test_var_mean_in_registers(self, width):
        # Rows held whole in registers take the two-pass variance; 700 leaves masked lanes.
        x = torch.randn(300, width, device="cuda") * 3 + 5
        self.check(lambda x: torch.var_mean(x, -1), x, atol=1e-3, rtol=1e-3)
        torch._dynamo.reset()
        self.check(lambda x: F.layer_norm(x, (width,)), x, atol=1e-3, rtol=1e-3)

    @parametrize("width", [256, 100])
    def test_rows_through_transpose(self, width):
        # A ViT patch embedding: LayerNorm over tokens of an NCHW conv output, whose
        # rows (channels 49 apart) interleave across a wave's lanes. 100 channels leave
        # a masked run, and 3 * 50 tokens a partly masked group of rows.
        from torch._inductor.utils import run_and_get_code

        def fn(conv, cls, pos):
            x = conv.flatten(2).transpose(1, 2)
            x = torch.cat([cls.expand(x.shape[0], -1, -1), x], 1) + pos
            return F.layer_norm(x, (width,))

        conv = torch.randn(3, width, 7, 7, device="cuda")
        cls = torch.randn(1, 1, width, device="cuda")
        pos = torch.randn(1, 50, width, device="cuda")
        out, (code,) = run_and_get_code(
            torch.compile(fn, dynamic=False), conv, cls, pos
        )
        torch.testing.assert_close(out, fn(conv, cls, pos), atol=1e-4, rtol=1e-4)
        self.assertIn(f"stride={8}", code)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)

    @parametrize("cols", [3072, 1000])
    def test_column_pairs(self, cols):
        # bf16 column sums read two adjacent columns a lane: a split reduction's
        # masked first stage (6304 rows), and 1000 columns leaving a masked pair.
        # Adding an f32 operand makes one load unpairable: one column a lane.
        from torch._inductor.utils import run_and_get_code

        g = torch.randn(6304, cols, device="cuda", dtype=torch.bfloat16)
        fn = lambda g: g.float().sum(0)  # noqa: E731
        out, (code,) = run_and_get_code(torch.compile(fn, dynamic=False), g)
        torch.testing.assert_close(out, fn(g), atol=1e-3, rtol=1e-4)
        self.assertIn("column_pairs=True", code)
        h = torch.randn(6304, cols, device="cuda")
        fn = lambda g, h: (g.float() * h).sum(0)  # noqa: E731
        out, (code,) = run_and_get_code(torch.compile(fn, dynamic=False), g, h)
        torch.testing.assert_close(out, fn(g, h), atol=1e-3, rtol=1e-4)
        self.assertNotIn("column_pairs=True", code)
        self.assertNotIn("_K_", code)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)

    def test_cross_entropy(self):
        x = torch.randn(512, 32000, device="cuda")
        t = torch.randint(0, 32000, (512,), device="cuda")
        self.check(lambda x: F.cross_entropy(x, t), x)

    def test_inplace_reload_after_store(self):
        # BatchNorm over a conv output updates the buffer in place in its last loop, and
        # the unrolled loops share loads through a trace-time cache. A store has to drop
        # that cache without misaligning a vector reload mid-chunk.
        x = torch.randn(16, 64, 16, 16, device="cuda")
        conv = torch.nn.Conv2d(64, 64, 3, padding=1).cuda()
        bn = torch.nn.BatchNorm2d(64).cuda()
        with torch.no_grad():
            self.check(lambda t: bn(conv(t)), x, atol=1e-4, rtol=1e-4)

    # -- scatter ---------------------------------------------------------------------

    def test_atomic_add_masked(self):
        # 10000 is not a multiple of the block, so the atomic is masked; an out-of-range
        # buffer atomic faults rather than being dropped.
        v = torch.randn(10000, device="cuda")
        ix = torch.randint(0, 64, (10000,), device="cuda")
        self.check(
            lambda v, ix: torch.zeros(64, device="cuda").scatter_add(0, ix, v),
            v,
            ix,
            atol=1e-3,
        )

    def test_atomic_add_of_row_reduction(self):
        # A row's sum scattered with an atomic add: every thread of the row group holds
        # the sum, and exactly one of them may add it.
        torch.manual_seed(0)
        for dtype, d in ((torch.float32, 1024), (torch.bfloat16, 1024), (torch.float32, 96)):
            torch._dynamo.reset()
            x = torch.randn(512, d, device="cuda", dtype=dtype)
            ix = torch.randint(0, 64, (512,), device="cuda")

            def fn(x, ix):
                z = torch.zeros(64, device="cuda", dtype=dtype)
                return z.index_put_((ix,), x.sum(-1), accumulate=True)

            out = torch.compile(fn, dynamic=False)(x, ix)
            ref = torch.zeros(64, device="cuda", dtype=torch.float64).index_put_(
                (ix,), x.double().sum(-1), accumulate=True
            )
            # bf16 partial sums near 90 round to 0.5, in an atomic order that varies.
            tol = 2.0 if dtype == torch.bfloat16 else 1e-3
            torch.testing.assert_close(out.double(), ref, atol=tol, rtol=2e-2)

    # -- training --------------------------------------------------------------------

    def test_transformer_block_train_step(self):
        torch.manual_seed(0)
        d = 256
        ln = torch.nn.LayerNorm(d).cuda()
        fc1, fc2 = torch.nn.Linear(d, 4 * d).cuda(), torch.nn.Linear(4 * d, d).cuda()
        x = torch.randn(8, 128, d, device="cuda", requires_grad=True)

        def f(x):
            y = x + fc2(F.gelu(fc1(ln(x)), approximate="tanh"))
            return torch.softmax(y, -1).sum()

        ref = torch.autograd.grad(f(x), x)[0]
        out = torch.autograd.grad(torch.compile(f)(x), x)[0]
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)

    # -- atomics ---------------------------------------------------------------------

    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
    def test_atomic_add(self, dtype):
        # 16-bit atomics are packed two-lane adds with -0.0 in the neighbour's lane, so
        # bins nobody hits must keep their exact bits, -0.0 included.
        v = torch.randn(1000, device="cuda", dtype=dtype)
        ix = torch.randint(0, 64, (1000,), device="cuda")
        base = torch.zeros(65, device="cuda", dtype=dtype)
        base[1::2] = -0.0
        out = torch.compile(
            lambda b, v, ix: b.clone().index_put_((ix,), v, accumulate=True)
        )(base, v, ix)
        ref = base.double().index_put_((ix,), v.double(), accumulate=True)
        # ~16 adds a bin, landing in any order, each rounded to dtype.
        tol = {torch.bfloat16: 1e-1, torch.float16: 2e-2}.get(dtype, 1e-5)
        torch.testing.assert_close(out.double(), ref, atol=tol, rtol=tol)
        miss = torch.ones(65, dtype=torch.bool, device="cuda")
        miss[ix] = False
        self.assertEqual(
            torch.signbit(out[miss]).tolist(), torch.signbit(base[miss]).tolist()
        )
        c = counters["flydsl_simd"]
        self.assertGreater(c["flydsl_kernels"], 0)
        self.assertEqual(c["triton_kernels"], 0)

    @unittest.skipIf(
        not HAS_GPU or torch.cuda.get_device_properties(0).total_memory < 32 << 30,
        "needs 32 GiB",
    )
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_buffer_past_2gib(self, dtype):
        # Past a buffer descriptor's 2 GiB range loads, stores and atomics take 64-bit
        # global pointers; the odd sizes leave a masked tail.
        cols = 151936
        rows = (3 << 30) // (cols * dtype.itemsize) + 1
        x = torch.randn(rows, cols, device="cuda", dtype=dtype)
        self.check(lambda x: x * 2, x, atol=0, rtol=0, kinds=["pointwise"])
        self.check(
            lambda x: torch.logsumexp(x.float(), -1), x, atol=1e-4, kinds=["reduction"]
        )
        del x
        n = rows * cols + 7
        ix = torch.randint(n - 3000, n, (5000,), device="cuda")
        v = torch.randn(5000, device="cuda", dtype=dtype)
        base = torch.zeros(n, device="cuda", dtype=dtype)
        out = torch.compile(lambda b, v, ix: b.index_put_((ix,), v, accumulate=True))(
            base, v, ix
        )
        ref = torch.zeros(3000, device="cuda", dtype=torch.float64).index_put_(
            (ix - (n - 3000),), v.double(), accumulate=True
        )
        tol = 1e-1 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(out[-3000:].double(), ref, atol=tol, rtol=tol)
        self.assertEqual(out[: n - 3000].count_nonzero().item(), 0)
        self.assertEqual(counters["flydsl_simd"]["triton_kernels"], 0)

    # -- scans -----------------------------------------------------------------------

    def test_group_widths(self):
        # Rows of 8 to 4096 give row groups of 2 to 256 threads: DPP within a wave
        # (with and without an identity), then LDS across waves. Argmax ties check the
        # combine order.
        for r in (8, 24, 64, 100, 300, 1000, 4096):
            x = torch.randn(129, r, device="cuda")
            ties = torch.randint(0, 5, (129, r), device="cuda").float()
            for fn, a in (
                (lambda t: t.sum(-1), x),
                (lambda t: t.amax(-1), x),
                (lambda t: t.argmax(-1), ties),
                (lambda t: torch.var_mean(t, -1), x),
                (lambda t: t.double().sum(-1), x),
            ):
                torch._dynamo.reset()
                counters.clear()
                self.check(fn, a)

    def test_scan_runs(self):
        # Runs of several elements a thread: scanned in registers, then across lanes.
        for fn, a in (
            (lambda x: (x * 0.01 + 1).cumprod(-1), torch.randn(33, 640)),
            (lambda x: torch.cummax(x, -1)[0], torch.randn(50, 8192)),
        ):
            torch._dynamo.reset()
            counters.clear()
            self.check(fn, a.cuda(), atol=1e-3, rtol=1e-3)
        # Eager accumulates a bf16 cumsum in bf16; Inductor in f32 either way.
        tri, fly = self._vs_triton(
            lambda x: x.cumsum(-1),
            torch.randn(20, 4096, device="cuda", dtype=torch.bfloat16),
        )
        self.assertEqual(fly, tri, atol=0.25, rtol=1e-2)  # one bf16 ulp

    def test_chained_scans_fall_back(self):
        a = torch.randn(16, 1024, device="cuda")
        fn = lambda x: x.cumsum(-1).cumsum(-1)  # noqa: E731
        with config.patch({"flydsl.simd_strict": False}):
            out = torch.compile(fn, dynamic=False)(a)
        torch.testing.assert_close(out, fn(a), atol=1e-2, rtol=1e-3)
        self.assertGreater(counters["flydsl_simd"]["fallback: chained scans"], 0)

    def test_cumsum(self):
        # One run per step, scanned across a wave with the carry of earlier runs: short
        # rows (one run), several runs, a dynamic loop, and a column scan.
        for shape, dim in (
            ((1000, 20), -1),
            ((100, 300), -1),
            ((64, 5000), -1),
            ((300, 500), 0),
        ):
            torch._dynamo.reset()
            a = torch.randn(*shape, device="cuda")
            self.check(lambda x: x.cumsum(dim), a, atol=1e-3, rtol=1e-3)
        torch._dynamo.reset()
        self.check(
            lambda x: x.cumsum(-1), torch.randint(0, 5, (50, 777), device="cuda")
        )

    def test_tuple_scans(self):
        a = torch.randn(64, 513, device="cuda")
        self.check(lambda x: torch.cummax(x, -1), a)
        torch._dynamo.reset()
        self.check(lambda x: torch.logcumsumexp(x, -1), a, atol=1e-3, rtol=1e-3)
        torch._dynamo.reset()
        self.check(
            lambda x: x.exp().cumsum(-1) / x.exp().sum(-1, keepdim=True),
            torch.randn(128, 1000, device="cuda"),
            atol=1e-3,
            rtol=1e-3,
        )

    # -- random numbers --------------------------------------------------------------

    def _compiled_random(self, fn, *args):
        out = torch.compile(fn, dynamic=False)(*args)
        c = counters["flydsl_simd"]
        self.assertGreater(c["flydsl_kernels"], 0)
        self.assertEqual(c["triton_kernels"], 0)
        return out

    def test_rand_distributions(self):
        x = torch.empty(1 << 20, device="cuda")
        u = self._compiled_random(lambda a: torch.rand_like(a), x)
        self.assertAlmostEqual(u.mean().item(), 0.5, delta=5e-3)
        self.assertGreaterEqual(u.min().item(), 0.0)
        self.assertLess(u.max().item(), 1.0)
        torch._dynamo.reset()
        n = self._compiled_random(lambda a: torch.randn_like(a), x)
        self.assertAlmostEqual(n.mean().item(), 0.0, delta=1e-2)
        self.assertAlmostEqual(n.std().item(), 1.0, delta=1e-2)
        torch._dynamo.reset()
        i = self._compiled_random(
            lambda a: torch.randint(3, 17, a.shape, device="cuda"), x
        )
        self.assertEqual((i.min().item(), i.max().item()), (3, 16))

    def test_dropout(self):
        x = torch.randn(4096, 1024, device="cuda")
        y = self._compiled_random(
            lambda a: torch.nn.functional.dropout(a, 0.25, True), x
        )
        kept = y != 0
        self.assertAlmostEqual(kept.float().mean().item(), 0.75, delta=5e-3)
        torch.testing.assert_close(y[kept], x[kept] / 0.75)

    def _vs_triton(self, fn, x, patches=()):
        outs = []
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            for fly in (False, True):
                torch._dynamo.reset()
                torch.manual_seed(0)
                with config.patch({"flydsl.simd": fly, "flydsl.simd_strict": fly}):
                    outs.append(torch.compile(fn)(x))
        return outs

    def test_rand_matches_triton_philox(self):
        # The same Philox-4x32-10 stream as Triton: one call per four consecutive
        # offsets as rand4x, and one per element as tl.rand where an odd size leaves
        # FlyDSL scalar elements.
        from torch._inductor.codegen.triton import TritonOverrides

        x = torch.empty(1000, 1000, device="cuda")
        t, f = self._vs_triton(lambda a: torch.rand_like(a), x)
        self.assertTrue(torch.equal(t, f))
        t, f = self._vs_triton(
            lambda a: (
                torch.rand_like(a),
                torch.randint(0, 1 << 40, a.shape, device="cuda"),
            ),
            torch.empty(999, 1001, device="cuda"),
            [
                patch.object(
                    TritonOverrides, "_can_use_4x_random", staticmethod(lambda: False)
                )
            ],
        )
        for a, b in zip(t, f):
            self.assertTrue(torch.equal(a, b))

    # -- plumbing --------------------------------------------------------------------

    def test_source_carries_runtime_fingerprint(self):
        # FlyDSL's binary cache does not hash the runtime modules a kernel calls.
        from torch._inductor.codegen.flydsl.flydsl_simd import _runtime_fingerprint
        from torch._inductor.utils import run_and_get_code

        _, (code,) = run_and_get_code(
            torch.compile(lambda x: x + 1), torch.randn(1024, device="cuda")
        )
        self.assertIn(f"# runtime {_runtime_fingerprint()}", code)
        # The wrapper names the code object the worker saves, so a warm start can load
        # it without the pool.
        import os
        import re

        (binary,) = re.findall(r"''', '([^']+\.hsaco)', precompile_metadata", code)
        self.assertIn("fdp_rt.compiled(async_compile, ", code)
        self.assertTrue(os.path.exists(binary), binary)

    def test_compiles_at_codegen_once_pool_is_up(self):
        # With the pool up, codegen hands each kernel to a worker; the module's load
        # then waits for that job instead of submitting its own.
        import os
        from unittest import mock

        from torch._inductor.async_compile import AsyncCompile
        from torch._inductor.codegen.flydsl import flydsl_simd_runtime as rt

        if not AsyncCompile.use_process_pool() and not (
            AsyncCompile.wait_process_pool_ready() and AsyncCompile.use_process_pool()
        ):
            self.skipTest("no compile worker pool")
        submitted = []

        def record(name, source, binaries):
            compile_early(name, source, binaries)
            submitted.extend(b for b in binaries if b in rt._EARLY)

        compile_early = rt.compile_early
        x = torch.randn(1000, 1000, device="cuda")
        with mock.patch.object(rt, "compile_early", record):
            out = torch.compile(lambda t: (t * 3.25).sin() + 0.5)(x)
        torch.testing.assert_close(out, (x * 3.25).sin() + 0.5)
        self.assertTrue(submitted)
        self.assertFalse(set(submitted) & set(rt._EARLY))
        for binary in submitted:
            self.assertTrue(os.path.exists(binary), binary)

    def test_autotuned_in_place(self):
        # A pointwise kernel this large times its candidates on its first call; one that
        # updates a buffer in place must be timed on a copy and applied exactly once.
        from torch._inductor.utils import run_and_get_code

        x = torch.randn(64, 64, 32, 32, device="cuda")
        conv = torch.nn.Conv2d(64, 64, 3, padding=1).cuda()
        with torch.no_grad():
            ref = conv(x)
            fn = torch.compile(conv)
            out, (code,) = run_and_get_code(fn, x)
            torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(fn(x), ref, atol=1e-4, rtol=1e-4)
        self.assertIn("fdp.pick_fastest(", code)
        self.assertIn("in_out_ptr0", code)


@requires_flydsl
class TestFlyDSLSIMDFallback(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        counters.clear()

    def _fallback(self, fn, *args, reason, tol=1e-2):
        with config.patch({"flydsl.simd": True, "fx_graph_cache": False}):
            ref = fn(*args)
            out = torch.compile(fn)(*args)
        torch.testing.assert_close(out, ref, atol=tol, rtol=tol)
        c = counters["flydsl_simd"]
        self.assertGreater(c["triton_kernels"], 0)
        self.assertTrue(any(reason in k for k in c), dict(c))

    def test_strict_raises(self):
        with config.patch(
            {"flydsl.simd": True, "flydsl.simd_strict": True, "fx_graph_cache": False}
        ):
            with self.assertRaisesRegex(Exception, "FlyDSL SIMD kernel refused"):
                torch.compile(lambda x: x + 1, dynamic=True)(
                    torch.randn(100, device="cuda")
                )


if __name__ == "__main__":
    run_tests()

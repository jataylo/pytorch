# Owner(s): ["module: inductor"]
"""FlyDSL codegen for Inductor's pointwise and reduction kernels (config.flydsl.simd).

The numerical tests run in strict mode, where a kernel FlyDSL cannot express raises
instead of falling back to Triton, so a pass means the kernel really was FlyDSL. The
fallback tests run without it and check the reason that was counted. The access
classifier tests are pure sympy and run anywhere.
"""

import unittest

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

    def test_integer_and_bool(self):
        i = torch.randint(-100, 100, (5000,), device="cuda")
        self.check(lambda x: (x // 3 + x % 7) * 2, i)
        self.check(lambda x: ((x * 7) > 13) | (x < -50), i)

    def test_casts(self):
        a = torch.randn(1000, 777, device="cuda")
        self.check(
            lambda x: (x * 30).clamp(-120, 120).to(torch.int8).float()
            + x.to(torch.bfloat16).float()
            + (x > 0).to(torch.uint8),
            a,
        )

    def test_transcendentals(self):
        a = torch.randn(4097, 33, device="cuda")
        self.check(
            lambda x: torch.where(x > 0, x.exp(), x.sin()) + x.tanh() * x.erf(), a
        )

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

    @parametrize("width", [768, 700, 96])
    def test_var_mean_in_registers(self, width):
        # Rows held whole in registers take the two-pass variance; 700 leaves masked lanes.
        x = torch.randn(300, width, device="cuda") * 3 + 5
        self.check(lambda x: torch.var_mean(x, -1), x, atol=1e-3, rtol=1e-3)
        torch._dynamo.reset()
        self.check(lambda x: F.layer_norm(x, (width,)), x, atol=1e-3, rtol=1e-3)

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

    # -- plumbing --------------------------------------------------------------------

    def test_source_carries_runtime_fingerprint(self):
        # FlyDSL's binary cache does not hash the runtime modules a kernel calls.
        from torch._inductor.codegen.flydsl.flydsl_simd import _runtime_fingerprint
        from torch._inductor.utils import run_and_get_code

        _, (code,) = run_and_get_code(
            torch.compile(lambda x: x + 1), torch.randn(1024, device="cuda")
        )
        self.assertIn(f"# runtime {_runtime_fingerprint()}", code)
        self.assertIn("async_compile.flydsl(", code)


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

    def test_scan(self):
        self._fallback(
            lambda x: x.cumsum(1), torch.randn(100, 300, device="cuda"), reason="scan"
        )

    def test_random(self):
        self._fallback(
            lambda x: x + torch.rand_like(x) * 0,
            torch.randn(1000, device="cuda"),
            reason="load_seed",
        )

    def test_bf16_atomic(self):
        v = torch.randn(1000, device="cuda", dtype=torch.bfloat16)
        ix = torch.randint(0, 64, (1000,), device="cuda")
        self._fallback(
            lambda v, ix: torch.zeros(
                64, device="cuda", dtype=torch.bfloat16
            ).scatter_add(0, ix, v),
            v,
            ix,
            reason="atomic_add of torch.bfloat16",
            # ~16 bf16 atomics per bin, accumulated in whatever order they land.
            tol=1e-1,
        )

    def test_strict_raises(self):
        with config.patch(
            {"flydsl.simd": True, "flydsl.simd_strict": True, "fx_graph_cache": False}
        ):
            with self.assertRaisesRegex(Exception, "FlyDSL SIMD kernel refused"):
                torch.compile(lambda x: x.cumsum(0))(torch.randn(100, device="cuda"))


if __name__ == "__main__":
    run_tests()

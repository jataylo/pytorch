# Owner(s): ["module: inductor"]

import ast
import textwrap

import torch
from torch._inductor import config
from torch._inductor.codegen.amd_kernel_optimizations import (
    optimize_flat_pointwise_source,
)
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


def flat_source():
    return textwrap.dedent("""
        @triton_heuristics.pointwise(size_hints={'x':8192}, triton_meta={'signature':{'in_ptr0':'*fp32','out_ptr0':'*fp32','xnumel':'i32','XBLOCK':'constexpr'}}, inductor_meta={})
        @triton.jit
        def kernel(in_ptr0,out_ptr0,xnumel,XBLOCK:tl.constexpr):
            xnumel=4099
            xoffset=tl.program_id(0)*XBLOCK
            xindex=xoffset+tl.arange(0,XBLOCK)[:]
            xmask=xindex<xnumel
            x0=xindex
            tmp0=tl.load(in_ptr0+x0,xmask)
            tmp1=tl.sigmoid(tmp0)
            tl.store(out_ptr0+x0,tmp1,xmask)
    """)


@instantiate_parametrized_tests
class TestAMDFlatSource(TestCase):
    def test_default_off(self):
        self.assertFalse(config.triton.amd_extended_kernel_optimizations)

    def test_preserves_sigmoid_definition_and_guards_tails(self):
        source = (
            flat_source()
            .replace("xmask=xindex<xnumel", "xmask=tl.full([XBLOCK],True,tl.int1)[:]")
            .replace("+x0,xmask", "+x0,None")
        )
        actual = optimize_flat_pointwise_source(source)
        ast.parse(actual)
        self.assertIn("1 / (1 + gl.exp(-tmp0))", actual)
        self.assertIn("xmask = xindex < 4099", actual)
        self.assertIn("gl.load(in_ptr0 + x0, xmask)", actual)
        self.assertIn("gl.store(out_ptr0 + x0, tmp1, xmask)", actual)

    @parametrize(
        "change",
        [
            ("xnumel=4099", "xnumel=xnumel"),
            ("in_ptr0+x0", "in_ptr0+2*x0"),
            ("tl.sigmoid(tmp0)", "tl.rand(1,x0)"),
            ("tl.sigmoid(tmp0)", "tl.sum(tmp0,0)"),
            ("tl.sigmoid(tmp0)", "libdevice.erf(tmp0)"),
            ("tl.sigmoid(tmp0)", "triton_helpers.bucketize(tmp0)"),
            ("tl.sigmoid(tmp0)", "tl.atomic_add(in_ptr0+x0,tmp0)"),
            ("tl.sigmoid(tmp0)", "tl.program_id(1)"),
            ("*fp32", "*fp64"),
        ],
    )
    def test_unsupported_source_unchanged(self, change):
        source = flat_source().replace(*change)
        self.assertEqual(optimize_flat_pointwise_source(source), source)


class TestAMDExtendedLowering(TestCase):
    @onlyCUDA
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    @parametrize(
        "op", ["dropout_backward", "silu_gate", "silu_backward", "gelu_forward"]
    )
    def test_elementwise_tails_and_nonfinites(self, device, dtype, op):
        if (
            torch.version.hip is None
            or torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            != "gfx950"
        ):
            self.skipTest("gfx950 specialization")
        x = torch.linspace(-10, 10, 4099, device=device, dtype=dtype)
        x[:4] = torch.tensor(
            [float("inf"), -float("inf"), float("nan"), -0.0],
            device=device,
            dtype=dtype,
        )
        y = torch.randn_like(x)
        if op == "dropout_backward":
            y = y > 0

            def fn(x, y):
                return (x.float() * y.float() * 1.1111111111111112).to(x.dtype)
        elif op == "silu_gate":

            def fn(x, y):
                return (torch.nn.functional.silu(x.float()) * y.float()).to(x.dtype)
        elif op == "silu_backward":

            def fn(x, y):
                return torch.ops.aten.silu_backward(y, x)
        else:

            def fn(x, y):
                return torch.nn.functional.gelu(x, approximate="none") + y

        expected = fn(x, y)
        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "fx_graph_cache": False,
            }
        ):
            actual, source = run_and_get_code(torch.compile(fn, fullgraph=True), x, y)
        self.assertEqual(
            actual,
            expected,
            atol=0.02 if dtype == torch.bfloat16 else 1e-5,
            rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
            equal_nan=True,
        )
        if op == "gelu_forward":
            self.assertNotIn("'amd_gluon_family': 'flat_pointwise'", "\n".join(source))
        else:
            self.assertIn("'amd_gluon_family': 'flat_pointwise'", "\n".join(source))

    @onlyCUDA
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    @parametrize("extent", [127, 8192, 16384])
    def test_normalization_statistics(self, device, dtype, extent):
        if (
            torch.version.hip is None
            or torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            != "gfx950"
        ):
            self.skipTest("gfx950 specialization")
        x = torch.randn(37, extent, device=device, dtype=dtype)
        x[0].fill_(1.5)
        weight = torch.randn(extent, device=device, dtype=dtype)
        bias = torch.randn_like(weight)

        def fn(x, w, b):
            return torch.native_layer_norm(x, (extent,), w, b, 1e-05)

        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "fx_graph_cache": False,
            }
        ):
            actual, source = run_and_get_code(
                torch.compile(fn, fullgraph=True), x, weight, bias
            )
        self.assertEqual(
            actual,
            fn(x, weight, bias),
            atol=0.02 if dtype == torch.bfloat16 else 1e-4,
            rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
        )
        self.assertIn("'amd_gluon_family': 'row_reduction'", "\n".join(source))

    @onlyCUDA
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_training_forward_backward(self, device, dtype):
        if (
            torch.version.hip is None
            or torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            != "gfx950"
        ):
            self.skipTest("gfx950 specialization")
        torch.manual_seed(1701)
        inputs = [
            torch.randn(37, 128, device=device, dtype=dtype, requires_grad=True),
            torch.randn(128, 128, device=device, dtype=dtype, requires_grad=True),
            torch.randn(128, device=device, dtype=dtype, requires_grad=True),
        ]

        def fn(x, w, b):
            value = torch.nn.functional.linear(x, w, b)
            value = torch.nn.functional.silu(value.float())
            inverse = torch.rsqrt(value.square().mean(-1, True) + 1e-5)
            return (value * inverse).square().mean()

        expected = fn(*inputs)
        expected_grad = torch.autograd.grad(expected, inputs)
        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
                "max_autotune_gemm_backends": "ATEN",
                "fx_graph_cache": False,
            }
        ):
            actual = torch.compile(fn, fullgraph=True)(*inputs)
            actual_grad = torch.autograd.grad(actual, inputs)
        self.assertEqual(
            actual,
            expected,
            atol=0.002 if dtype == torch.bfloat16 else 1e-5,
            rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
        )
        self.assertEqual(
            actual_grad,
            expected_grad,
            atol=2e-5 if dtype == torch.bfloat16 else 1e-6,
            rtol=0.03 if dtype == torch.bfloat16 else 1e-3,
        )

    @onlyCUDA
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_optimizer_multiple_mutated_outputs(self, device, dtype):
        if (
            torch.version.hip is None
            or torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            != "gfx950"
        ):
            self.skipTest("gfx950 specialization")
        x = torch.randn(4099, device=device, dtype=dtype)
        g = torch.randn_like(x)
        m = torch.randn(4099, device=device)
        v = torch.rand_like(m)

        def step(x, g, m, v):
            mean = m * 0.9 + g.float() * 0.1
            variance = v * 0.999 + g.float().square() * 0.001
            value = x.float() - 0.001 * mean * torch.rsqrt(variance + 1e-8)
            m.copy_(mean)
            v.copy_(variance)
            x.copy_(value.to(x.dtype))
            return x, m, v

        expected = step(x.clone(), g.clone(), m.clone(), v.clone())
        with config.patch(
            {
                "triton.amd_kernel_optimizations": True,
                "triton.amd_extended_kernel_optimizations": True,
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
        self.assertIn("'amd_gluon_family': 'flat_pointwise'", "\n".join(source))


instantiate_device_type_tests(TestAMDExtendedLowering, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()

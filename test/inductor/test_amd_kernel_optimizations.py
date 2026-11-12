# Owner(s): ["module: inductor"]

import ast
import textwrap

import torch
from torch._inductor import config
from torch._inductor.codegen.amd_kernel_optimizations import (
    optimize_pointwise_source,
    simplify_masked_modulo,
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


def broadcast_source(n=4099):
    return textwrap.dedent(f"""
        @triton_heuristics.pointwise(size_hints={{'x':8192}}, triton_meta={{'signature':{{'in_ptr0':'*fp32','in_ptr1':'*bf16','out_ptr0':'*bf16','xnumel':'i32','XBLOCK':'constexpr'}}}}, inductor_meta={{}})
        @triton.jit
        def kernel(in_ptr0,in_ptr1,out_ptr0,xnumel,XBLOCK:tl.constexpr):
            xnumel={n}
            xoffset=tl.program_id(0)*XBLOCK
            xindex=xoffset+tl.arange(0,XBLOCK)[:]
            xmask=xindex<xnumel
            x0=xindex
            tmp0=tl.load(in_ptr0+0)
            tmp1=tl.broadcast_to(tmp0,[XBLOCK])
            tmp2=tl.full([1],{1.0 / n!r},tl.float32)
            tmp3=tmp1*tmp2
            tmp4=tl.load(in_ptr1+x0,xmask).to(tl.float32)
            tmp5=tmp4.to(tl.float32)
            tmp6=tl.full([1],2.0,tl.float32)
            tmp7=tmp5*tmp6
            tmp8=tmp3*tmp7
            tl.store(out_ptr0+x0,tmp8,xmask)
    """)


def reduction_source():
    return textwrap.dedent("""
        @triton_heuristics.reduction()
        @triton.jit
        def kernel(in_ptr0,out_ptr0,xnumel,r0_numel,XBLOCK:tl.constexpr,R0_BLOCK:tl.constexpr):
            xnumel=257
            r0_numel=127
            xoffset=tl.program_id(0)*XBLOCK
            xindex=xoffset+tl.arange(0,XBLOCK)[:,None]
            xmask=xindex<xnumel
            r0_base=tl.arange(0,R0_BLOCK)[None,:]
            x0=xindex
            for r0_offset in tl.range(0,r0_numel,R0_BLOCK,num_stages=2):
                r0_index=r0_offset+r0_base
                r0_mask=r0_index<r0_numel
                r0_1=r0_index
                tmp0=(r0_1+127*x0).to(tl.int32)
                tmp1=tl.full([1,1],32632,tl.int32)
                tmp2=tmp0<tmp1
                tmp3=tmp2.to(tl.int1)
                tmp4=tl.load(in_ptr0+((r0_1+127*x0)%32632),r0_mask&tmp3&xmask,other=0.)
                tl.store(out_ptr0+x0,tl.sum(tmp4,1)[:,None],xmask)
    """)


@instantiate_parametrized_tests
class TestAMDSourceOptimizations(TestCase):
    @parametrize("n", [4099, 65537, 1048608768])
    def test_broadcast_recognized(self, n):
        source = broadcast_source(n)
        output = optimize_pointwise_source(source)
        self.assertNotEqual(source, output)
        self.assertIn("gluon.jit", output)
        self.assertIn("'amd_gluon_family': 'broadcast'", output)
        ast.parse(output)

    @parametrize(
        "change", ["factor", "vector_offset", "mask", "integer", "size", "extra_store"]
    )
    def test_unrelated_kernels_unchanged(self, change):
        source = broadcast_source()
        if change == "factor":
            source = source.replace(repr(1.0 / 4099), repr(2.0 / 4099))
        elif change == "vector_offset":
            source = source.replace("in_ptr1+x0", "in_ptr1+x0+1")
        elif change == "mask":
            source = source.replace("tmp8,xmask", "tmp8,tmp0>0")
        elif change == "integer":
            source = source.replace("'*bf16'", "'*int32'")
        elif change == "size":
            source = source.replace("xnumel=4099", "xnumel=ks0")
        else:
            source += "    tl.store(out_ptr0+x0,tmp8,xmask)\n"
        self.assertEqual(optimize_pointwise_source(source), source)

    def test_mask_proves_modulo_range(self):
        source = reduction_source()
        output = simplify_masked_modulo(source)
        self.assertNotEqual(source, output)
        self.assertNotIn("% 32632", output)
        self.assertIn("r0_mask & tmp3 & xmask", output)

    @parametrize(
        "change",
        ["upper_bound", "row_mask", "reduction_mask", "overflow", "negative_index"],
    )
    def test_modulo_requires_complete_proof(self, change):
        source = reduction_source()
        if change == "upper_bound":
            source = source.replace("tmp0<tmp1", "tmp0<=tmp1")
        elif change == "row_mask":
            source = source.replace("r0_mask&tmp3&xmask", "r0_mask&tmp3")
        elif change == "reduction_mask":
            source = source.replace("r0_mask&tmp3&xmask", "tmp3&xmask")
        elif change == "overflow":
            source = source.replace("xnumel=257", "xnumel=2147483647")
        else:
            source = source.replace("r0_1+127*x0", "r0_1+127*x0-1")
        self.assertEqual(simplify_masked_modulo(source), source)


class TestAMDCompiledOptimizations(TestCase):
    @onlyCUDA
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    @parametrize(
        "family",
        ["broadcast", "transpose", "add", "mean", "softmax_backward", "norm_backward"],
    )
    def test_compiled_paths(self, device, dtype, family):
        if (
            torch.version.hip is None
            or torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            != "gfx950"
        ):
            self.skipTest("wave64 gfx950 optimization")
        torch.manual_seed(17)
        if family == "transpose":
            x = torch.randn(2, 5, 37, 64, device=device, dtype=dtype)
            args = (x,)

            def fn(x):
                return x.permute(0, 2, 1, 3).contiguous()
        elif family == "broadcast":
            x = torch.randn(4099, device=device, dtype=dtype)
            s = torch.tensor(0.75, device=device)
            args = (x, s)

            def fn(x, s):
                return (s * (1.0 / x.numel()) * (x.float() * 2.0)).to(x.dtype)
        elif family == "add":
            x = torch.randn(4099, device=device, dtype=dtype)
            args = (x, torch.randn_like(x))

            def fn(x, y):
                return x.float() + y.float()
        elif family == "mean":
            x = torch.randn(37, 1027, device=device, dtype=dtype)
            args = (x,)

            def fn(x):
                return x.float().square().mean()
        elif family == "softmax_backward":
            p = torch.softmax(torch.randn(37, 128, device=device), -1)
            g = torch.randn_like(p)
            keep = torch.rand_like(p) > 0.1
            args = (p, g, keep)

            def fn(p, g, keep):
                value = (g * (keep.float() * 1.1111111111111112)) * p
                return torch.addcmul(value, -p, value.sum(-1, keepdim=True))
        else:
            g = torch.randn(37, 1536, device=device, dtype=dtype)
            h = torch.randn_like(g, dtype=torch.float32)
            w = torch.randn(1536, device=device)
            args = (g, h, w)

            def fn(g, h, w):
                v = g.float() * w
                return (
                    (v * 1536 - v.sum(-1, keepdim=True))
                    - h * (v * h).sum(-1, keepdim=True)
                ) / 1536

        expected = fn(*args)
        for enabled in (False, True):
            torch._dynamo.reset()
            with config.patch(
                {
                    "triton.amd_kernel_optimizations": enabled,
                    "triton.amd_use_gluon": True,
                    "max_autotune_pointwise": True,
                    "fx_graph_cache": False,
                }
            ):
                actual, code = run_and_get_code(
                    torch.compile(fn, fullgraph=True), *args
                )
                self.assertEqual(
                    actual,
                    expected,
                    atol=0
                    if family in ("transpose", "broadcast", "add")
                    else 0.02
                    if dtype == torch.bfloat16
                    else 1e-4,
                    rtol=0
                    if family in ("transpose", "broadcast", "add")
                    else 0.02
                    if dtype == torch.bfloat16
                    else 1e-4,
                )
                if enabled and family in ("transpose", "broadcast", "add"):
                    self.assertIn("'use_gluon': True", "\n".join(code))


instantiate_device_type_tests(TestAMDCompiledOptimizations, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()

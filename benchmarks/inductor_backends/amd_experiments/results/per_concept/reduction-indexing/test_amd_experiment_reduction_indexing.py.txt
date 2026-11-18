# Owner(s): ["module: inductor"]

import textwrap

from torch._inductor.codegen.amd_kernel_optimizations import simplify_masked_modulo
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestAMDReductionIndexing(TestCase):
    @parametrize(
        "proof",
        [
            "valid",
            "missing_upper",
            "wrong_period",
            "wrong_row_mask",
            "wrong_reduction_mask",
        ],
    )
    def test_load_mask_proves_modulo_range(self, proof):
        source = textwrap.dedent("""
            @triton_heuristics.reduction()
            def kernel(in_ptr0, out_ptr0, XBLOCK, R0_BLOCK):
                xnumel = 4
                r0_numel = 4099
                xoffset = tl.program_id(0) * XBLOCK
                xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
                xmask = xindex < xnumel
                r0_base = tl.arange(0, R0_BLOCK)[None, :]
                x0 = xindex
                for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
                    r0_index = r0_offset + r0_base
                    r0_mask = r0_index < r0_numel
                    r0_1 = r0_index
                    tmp0 = r0_1 + 4099 * x0
                    tmp1 = tmp0 < 12314
                    value = tl.load(in_ptr0 + ((r0_1 + 4099 * x0) % 12314), xmask & r0_mask & tmp1, other=0)
        """)
        if proof == "missing_upper":
            source = source.replace(" & tmp1", "")
        elif proof == "wrong_period":
            source = source.replace("tmp0 < 12314", "tmp0 < 12315")
        elif proof == "wrong_row_mask":
            source = source.replace("xmask & r0_mask & tmp1", "r0_mask & tmp1")
        elif proof == "wrong_reduction_mask":
            source = source.replace("xmask & r0_mask & tmp1", "xmask & tmp1")
        result = simplify_masked_modulo(source)
        if proof == "valid":
            self.assertNotIn("% 12314", result)
        else:
            self.assertEqual(result, source)


if __name__ == "__main__":
    run_tests()

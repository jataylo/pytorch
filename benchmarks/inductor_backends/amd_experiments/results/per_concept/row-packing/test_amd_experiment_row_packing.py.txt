# Owner(s): ["module: inductor"]

import ast
import textwrap

from torch._inductor.codegen.amd_kernel_optimizations import _row_tail_safe
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def row_body():
    return textwrap.dedent(
        "\n        def kernel(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK):\n            R0_BLOCK: tl.constexpr = 128\n            xoffset = tl.program_id(0) * XBLOCK\n            xindex = xoffset + tl.arange(0, XBLOCK)[:, None]\n            rindex = tl.arange(0, R0_BLOCK)[None, :]\n            xmask = xindex < xnumel\n            mask = xmask & (rindex < r0_numel)\n            value = tl.load(in_ptr0 + xindex * r0_numel + rindex, mask, other=0)\n            result = tl.sum(value, 1)[:, None].to(tl.float32)\n            tl.store(out_ptr0 + xindex, result, xmask)\n    "
    )


@instantiate_parametrized_tests
class TestAMDNativeCodegen(TestCase):
    @parametrize(
        "change",
        [
            "none",
            "store_mask",
            "load_mask",
            "or_mask",
            "axis",
            "transpose",
            "index",
            "upper_bound",
            "method",
            "property",
            "reassign",
            "row_slice",
            "alias_reduce",
            "alias_trans",
        ],
    )
    def test_partial_row_proof(self, change):
        source = row_body()
        if change == "store_mask":
            source = source.replace("result, xmask)", "result, rindex < r0_numel)")
        elif change == "load_mask":
            source = source.replace(
                "rindex, mask, other=0)", "rindex, rindex < r0_numel, other=0)"
            )
        elif change == "or_mask":
            source = source.replace("xmask & (", "xmask | (")
        elif change == "axis":
            source = source.replace("tl.sum(value, 1)", "tl.sum(value, 0)")
        elif change == "transpose":
            source = source.replace("tl.sum(value, 1)", "tl.sum(tl.trans(value), 1)")
        elif change == "method":
            source = source.replace(
                "tl.sum(value, 1)", "tl.sum(value.permute(1, 0), 1)"
            )
        elif change == "property":
            source = source.replace("tl.sum(value, 1)", "tl.sum(value.T, 1)")
        elif change == "reassign":
            source += "    xmask = xindex < xnumel\n"
            source = source.replace("xmask = xindex < xnumel", "xmask = xindex < 64", 1)
        elif change == "alias_reduce":
            source = source.replace(
                "result = tl.sum(value, 1)",
                "reduce = tl.sum\n    result = reduce(value, 0)",
            )
        elif change == "alias_trans":
            source = source.replace(
                "result = tl.sum(value, 1)",
                "swap = tl.trans\n    result = tl.sum(swap(value), 1)",
            )
        elif change == "row_slice":
            source = source.replace("tl.sum(value, 1)", "tl.sum(value[1:, :], 1)")
        elif change == "index":
            source = source.replace("xoffset + tl.arange", "-xoffset + tl.arange")
        elif change == "upper_bound":
            source = source.replace("xindex < xnumel", "xindex <= xnumel")
        fn = ast.parse(source).body[0]
        assignments = {
            node.targets[0].id: node.value
            for node in fn.body
            if isinstance(node, ast.Assign)
        }
        self.assertEqual(_row_tail_safe(fn, assignments), change == "none")

    @parametrize("dependent", [False, True])
    def test_row_independent_weight_load(self, dependent):
        source = row_body().replace("in_ptr0, out_ptr0", "in_ptr0, in_ptr1, out_ptr0")
        address = "xindex" if dependent else "rindex"
        source = source.replace(
            "    result =",
            f"    weight = tl.load(in_ptr1 + {address}, rindex < r0_numel, other=0)\n    result =",
        )
        source = source.replace("tl.sum(value, 1)", "tl.sum(value * weight, 1)")
        fn = ast.parse(source).body[0]
        assignments = {
            node.targets[0].id: node.value
            for node in fn.body
            if isinstance(node, ast.Assign)
        }
        self.assertEqual(_row_tail_safe(fn, assignments), not dependent)


if __name__ == "__main__":
    run_tests()

import textwrap


def row_source(device, *, extent=128, scan=False, outer=False):
    helper = (
        "\n        @triton.jit\n        def combine(left, right):\n            result = left + right\n            return result\n    "
        if scan
        else ""
    )
    operation = (
        "result, = tl.associative_scan((value,), 1, combine)"
        if scan
        else "result = tl.sum(value, 1)[:, None]"
    )
    return textwrap.dedent(helper) + textwrap.dedent(
        f"\n        @triton_heuristics.persistent_reduction(size_hints={{'x': 2, 'r0_': {extent}}}, reduction_hint=ReductionHint.{('OUTER' if outer else 'INNER')}, triton_meta={{'device': {device!r}, 'signature': {{'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}}}}, inductor_meta={{}})\n        @triton.jit\n        def kernel(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):\n            xnumel = 2\n            r0_numel = {extent - 1}\n            R0_BLOCK: tl.constexpr = {extent}\n            xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]\n            rindex = tl.arange(0, R0_BLOCK)[None, :]\n            mask = (xindex < xnumel) & (rindex < r0_numel)\n            value = tl.load(in_ptr0 + xindex * r0_numel + rindex, mask, other=0).to(tl.float32)\n            {operation}\n            tl.store(out_ptr0 + xindex * r0_numel + rindex, result, mask)\n    "
    )

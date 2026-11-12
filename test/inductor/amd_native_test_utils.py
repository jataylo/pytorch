import textwrap


def pointwise_source(device):
    return textwrap.dedent(
        f"\n        @triton_heuristics.pointwise(size_hints={{'x': 256}}, triton_meta={{'device': {device!r}, 'signature': {{'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}}}}, inductor_meta={{}})\n        @triton.jit\n        def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):\n            xnumel = 129\n            xoffset = tl.program_id(0) * XBLOCK\n            xindex = xoffset + tl.arange(0, XBLOCK)[:]\n            xmask = xindex < xnumel\n            x0 = xindex\n            tmp0 = tl.load(in_ptr0 + x0, xmask)\n            tmp1 = tl.load(in_ptr1 + x0, xmask)\n            tmp2 = tmp0 + tmp1\n            tl.store(out_ptr0 + x0, tmp2, xmask)\n    "
    )


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

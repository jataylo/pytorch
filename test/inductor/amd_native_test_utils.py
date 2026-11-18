import textwrap


def pointwise_source(device):
    return textwrap.dedent(
        f"\n        @triton_heuristics.pointwise(size_hints={{'x': 256}}, triton_meta={{'device': {device!r}, 'signature': {{'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}}}}, inductor_meta={{}})\n        @triton.jit\n        def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):\n            xnumel = 129\n            xoffset = tl.program_id(0) * XBLOCK\n            xindex = xoffset + tl.arange(0, XBLOCK)[:]\n            xmask = xindex < xnumel\n            x0 = xindex\n            tmp0 = tl.load(in_ptr0 + x0, xmask)\n            tmp1 = tl.load(in_ptr1 + x0, xmask)\n            tmp2 = tmp0 + tmp1\n            tl.store(out_ptr0 + x0, tmp2, xmask)\n    "
    )

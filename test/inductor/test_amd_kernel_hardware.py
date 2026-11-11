# Owner(s): ["module: inductor"]

import ast
import textwrap

from torch._inductor.codegen.amd_kernel_optimizations import (
    optimize_flat_pointwise_source,
    optimize_pointwise_source,
    optimize_row_source,
    supports_amd_kernel_optimizations,
)
from torch._inductor.runtime.hints import DeviceProperties
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def pointwise_source(device):
    return textwrap.dedent(f"""
        @triton_heuristics.pointwise(size_hints={{'x': 256}}, triton_meta={{'device': {device!r}, 'signature': {{'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}}}}, inductor_meta={{}})
        @triton.jit
        def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
            xnumel = 129
            xoffset = tl.program_id(0) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:]
            xmask = xindex < xnumel
            x0 = xindex
            tmp0 = tl.load(in_ptr0 + x0, xmask)
            tmp1 = tl.load(in_ptr1 + x0, xmask)
            tmp2 = tmp0 + tmp1
            tl.store(out_ptr0 + x0, tmp2, xmask)
    """)


def row_source(device, *, extent=128, scan=False, outer=False):
    helper = (
        """
        @triton.jit
        def combine(left, right):
            result = left + right
            return result
    """
        if scan
        else ""
    )
    operation = (
        "result, = tl.associative_scan((value,), 1, combine)"
        if scan
        else "result = tl.sum(value, 1)[:, None]"
    )
    return textwrap.dedent(helper) + textwrap.dedent(f"""
        @triton_heuristics.persistent_reduction(size_hints={{'x': 2, 'r0_': {extent}}}, reduction_hint=ReductionHint.{"OUTER" if outer else "INNER"}, triton_meta={{'device': {device!r}, 'signature': {{'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}}}}, inductor_meta={{}})
        @triton.jit
        def kernel(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
            xnumel = 2
            r0_numel = {extent - 1}
            R0_BLOCK: tl.constexpr = {extent}
            xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
            rindex = tl.arange(0, R0_BLOCK)[None, :]
            mask = (xindex < xnumel) & (rindex < r0_numel)
            value = tl.load(in_ptr0 + xindex * r0_numel + rindex, mask, other=0).to(tl.float32)
            {operation}
            tl.store(out_ptr0 + xindex * r0_numel + rindex, result, mask)
    """)


@instantiate_parametrized_tests
class TestAMDHardwareSource(TestCase):
    @parametrize("wave", [32, 64])
    @parametrize(
        "optimizer", [optimize_pointwise_source, optimize_flat_pointwise_source]
    )
    def test_pointwise_layout_uses_device_wave(self, wave, optimizer):
        device = DeviceProperties(
            "hip", 0, 8, "gfx_future", max_threads_per_block=256, warp_size=wave
        )
        source = pointwise_source(device)
        result = optimizer(source)
        self.assertNotEqual(result, source)
        self.assertIn(
            f"gl.BlockedLayout([AMD_VEC], [{wave}], [AMD_WARPS], [0])", result
        )
        kernel = next(
            node for node in ast.parse(result).body if isinstance(node, ast.FunctionDef)
        )
        memory_calls = [
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) in ("gl.load", "gl.store")
        ]
        for call in memory_calls:
            mask_index = 2 if ast.unparse(call.func) == "gl.store" else 1
            self.assertIn(ast.unparse(call.args[mask_index]), ("xmask", "i < 129"))

    @parametrize("wave", [32, 64])
    @parametrize("outer", [False, True])
    def test_row_layout_uses_device_wave(self, wave, outer):
        device = DeviceProperties(
            "hip", 0, 8, "gfx_future", max_threads_per_block=256, warp_size=wave
        )
        source = row_source(device, outer=outer)
        result = optimize_row_source(source, extended=True, further=True)
        self.assertNotEqual(result, source)
        self.assertIn(f"[{wave} // AMD_LANES, AMD_LANES]", result)
        if outer:
            self.assertIn(f"[AMD_LANES, {wave} // AMD_LANES]", result)
        self.assertIn("rindex < r0_numel", result)
        self.assertIn("xindex < xnumel", result)

    @parametrize("wave", [32, 64])
    def test_scan_rejects_tile_above_register_budget(self, wave):
        device = DeviceProperties(
            "hip",
            0,
            8,
            "gfx_future",
            regs_per_multiprocessor=65536,
            max_threads_per_block=256,
            warp_size=wave,
        )
        source = row_source(device, extent=32768, scan=True)
        self.assertEqual(
            optimize_row_source(source, extended=True, further=True), source
        )

    @parametrize("change", ["subtract", "return", "combine", "defaults"])
    def test_scan_requires_additive_helper(self, change):
        device = DeviceProperties(
            "hip",
            0,
            8,
            "gfx_future",
            regs_per_multiprocessor=131072,
            max_threads_per_block=256,
            warp_size=64,
        )
        source = row_source(device, extent=32768, scan=True)
        if change == "subtract":
            source = source.replace("left + right", "left - right")
        elif change == "return":
            source = source.replace("return result", "return")
        elif change == "combine":
            source = source.replace("1, combine)", "1, other_combine)")
        else:
            source = source.replace("left, right):", "left, right=0):")
        self.assertEqual(
            optimize_row_source(source, extended=True, further=True), source
        )

    @parametrize(
        "device",
        [
            DeviceProperties(
                "cuda", 0, 8, 90, max_threads_per_block=1024, warp_size=32
            ),
            DeviceProperties(
                "hip", 0, 8, "other", max_threads_per_block=1024, warp_size=64
            ),
            DeviceProperties(
                "hip", 0, 8, "gfx_future", max_threads_per_block=1024, warp_size=16
            ),
            DeviceProperties(
                "hip", 0, 8, "gfx_future", max_threads_per_block=32, warp_size=64
            ),
            DeviceProperties(
                "hip", 0, 8, "gfx_future", max_threads_per_block=96, warp_size=64
            ),
            DeviceProperties(
                "hip", 0, 8, "gfx_future", max_threads_per_block=None, warp_size=64
            ),
        ],
    )
    def test_unsupported_device_keeps_source(self, device):
        self.assertFalse(supports_amd_kernel_optimizations(device))
        source = pointwise_source(device)
        self.assertEqual(optimize_pointwise_source(source), source)
        self.assertEqual(optimize_flat_pointwise_source(source), source)
        source = row_source(device)
        self.assertEqual(optimize_row_source(source), source)


if __name__ == "__main__":
    run_tests()

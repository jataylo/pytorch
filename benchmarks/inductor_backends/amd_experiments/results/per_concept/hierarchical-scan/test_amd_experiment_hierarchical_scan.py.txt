# Owner(s): ["module: inductor"]

import ast

from amd_native_test_utils import row_source

import torch
from torch._inductor.codegen.amd_kernel_optimizations import optimize_scan_source
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.runtime.triton_compat import Config
from torch._inductor.runtime.triton_heuristics import _amd_scan_configs
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestAMDScanCodegen(TestCase):
    @parametrize("looped", [False, True])
    def test_additive_scan_rewrite(self, looped):
        device = DeviceProperties(
            "hip", 0, 256, "gfx950", warp_size=64, max_threads_per_block=1024
        )
        source = row_source(device, extent=32768, scan=True)
        if looped:
            source = source.replace("persistent_reduction(", "reduction(")
            source = source.replace(
                "'XBLOCK': 'constexpr'",
                "'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'",
            )
            source = source.replace(
                "XBLOCK: tl.constexpr):",
                "XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):",
            )
        changed = optimize_scan_source(source)
        self.assertIn("triton_helpers.associative_scan_in_chunks", changed)
        self.assertIn("AMD_SCAN_TILE", changed)
        self.assertEqual(changed, optimize_scan_source(changed))
        ast.parse(changed)

    @parametrize("change", ["small", "non_additive", "tuple", "axis"])
    def test_reject_unsupported_scan(self, change):
        device = DeviceProperties(
            "hip", 0, 256, "gfx950", warp_size=64, max_threads_per_block=1024
        )
        source = row_source(
            device, extent=128 if change == "small" else 32768, scan=True
        )
        if change == "non_additive":
            source = source.replace("left + right", "left * right")
        elif change == "tuple":
            source = source.replace("(value,), 1", "(value, value), 1")
        elif change == "axis":
            source = source.replace("(value,), 1", "(value,), 0")
        self.assertEqual(source, optimize_scan_source(source))

    @parametrize("wave", [32, 64])
    def test_wave_relative_algorithm_candidates(self, wave):
        device = DeviceProperties(
            "hip", 0, 256, "gfx_future", warp_size=wave, max_threads_per_block=1024
        )
        original = [
            Config({"XBLOCK": 1, "R0_BLOCK": 1024}, num_warps=4),
            Config({"XBLOCK": 1, "R0_BLOCK": 8192}, num_warps=4),
        ]
        actual = _amd_scan_configs(
            original,
            {"x": 8, "r0_": 32768},
            {"device": device},
            {"amd_hierarchical_scan": True},
        )
        self.assertEqual(
            [c.kwargs["AMD_SCAN_TILE"] for c in actual if c.kwargs["R0_BLOCK"] == 1024],
            [0],
        )
        self.assertEqual(
            [c.kwargs["AMD_SCAN_TILE"] for c in actual if c.kwargs["R0_BLOCK"] == 8192],
            [0, wave, 4 * wave],
        )
        self.assertEqual(original[0].kwargs, {"XBLOCK": 1, "R0_BLOCK": 1024})

    @parametrize(
        "flag",
        [
            "deterministic",
            "force_filter_reduction_configs",
            "are_deterministic_algorithms_enabled",
        ],
    )
    def test_scan_deterministic_tree(self, flag):
        device = DeviceProperties(
            "hip", 0, 256, "gfx950", warp_size=64, max_threads_per_block=1024
        )
        original = Config({"XBLOCK": 1, "R0_BLOCK": 8192}, num_warps=4)
        actual = _amd_scan_configs(
            [original],
            {"x": 8, "r0_": 32768},
            {"device": device},
            {"amd_hierarchical_scan": True, flag: True},
        )
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0].kwargs, dict(original.kwargs, AMD_SCAN_TILE=0))


class TestAMDScanNumerics(TestCase):
    @parametrize("width", [127, 4095, 32768])
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("tile", [0, 64, 256])
    @parametrize("nonfinite", [False, True])
    def test_hierarchical_prefix(self, device, width, dtype, tile, nonfinite):
        if torch.version.hip is None:
            self.skipTest("AMD scan algorithm")
        import triton
        import triton.language as tl

        from torch._inductor.runtime import triton_helpers

        @triton.jit
        def kernel(x, out, N: tl.constexpr, BLOCK: tl.constexpr, TILE: tl.constexpr):
            row = tl.program_id(0)
            col = tl.arange(0, BLOCK)
            value = tl.load(x + row * N + col, col < N, other=0).to(tl.float32)[None, :]
            (result,) = triton_helpers.associative_scan_in_chunks((value,), 1, TILE)
            tl.store(out + row * N + col, tl.reshape(result, (BLOCK,)), col < N)

        torch.manual_seed(2026)
        x = torch.randn(3, width, device=device, dtype=dtype)
        if nonfinite:
            x[0, 1] = float("inf")
            x[0, width // 2] = -float("inf")
            x[1, width // 2] = float("nan")
        out = torch.empty_like(x, dtype=torch.float32)
        kernel[3,](x, out, width, triton.next_power_of_2(width), tile, num_warps=8)
        self.assertEqual(out, x.double().cumsum(-1).float(), atol=0.0002, rtol=0.0001)


instantiate_device_type_tests(TestAMDScanNumerics, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()

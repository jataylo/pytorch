# Owner(s): ["module: inductor"]

import importlib.util
from pathlib import Path

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._triton import has_triton_package


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/inductor_backends/amd_kernel_hillclimb.py"
)
SPEC = importlib.util.spec_from_file_location("amd_scan_reference", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load benchmark reference helper: {SOURCE}")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)

if has_triton_package():
    import triton
    import triton.language as tl

    @triton.jit
    def _looped_scan(x, out, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0) + tl.arange(0, 1)[:, None]
        lane = tl.arange(0, BLOCK)[None, :]
        carry = tl.full((1, 1), float("nan"), tl.float32)
        for offset in tl.range(0, WIDTH, BLOCK, num_stages=2):
            column = offset + lane
            value = tl.load(x + row * WIDTH + column, column < WIDTH, other=0)
            prefix = tl.cumsum(value, 1)
            last = tl.sum(
                prefix.to(tl.uint32, bitcast=True) * (lane == BLOCK - 1),
                1,
                keep_dims=True,
            ).to(tl.float32, bitcast=True)
            result = tl.where(offset > 0, carry + prefix, prefix)
            carry = tl.where(offset > 0, carry + last, last)
            tl.store(out + row * WIDTH + column, result, column < WIDTH)

    @triton.jit
    def _persistent_scan(x, out, WIDTH: tl.constexpr):
        row = tl.program_id(0) + tl.arange(0, 1)[:, None]
        column = tl.arange(0, WIDTH)[None, :]
        value = tl.load(x + row * WIDTH + column)
        tl.store(out + row * WIDTH + column, tl.cumsum(value, 1))


class TestAMDBenchmarkReference(TestCase):
    hw_classification = HardwareClassification.CUDA

    @parametrize("seed", (2026, 1701, 2027))
    def test_scan_gold_uses_high_precision(self, device, seed):
        if not has_triton_package():
            self.skipTest("Triton is not available")
        if torch.version.hip is None or not torch.cuda.get_device_properties(
            device
        ).gcnArchName.startswith("gfx950"):
            self.skipTest("Pinned scan schedule fixture requires gfx950")
        torch.manual_seed(seed)
        value = torch.randn((257, 32768), device=device, dtype=torch.float32)

        def fn(x):
            return torch.cumsum(x.float(), -1)

        eager = fn(value)
        gold = benchmark.reference_output(fn, (value,), "cumsum")
        stock, native = torch.empty_like(value), torch.empty_like(value)
        _looped_scan[(257,)](value, stock, 32768, 2048, num_warps=8, num_stages=1)
        _persistent_scan[(257,)](value, native, 32768, num_warps=4, num_stages=1)
        self.assertEqual(gold.dtype, eager.dtype)
        self.assertEqual(gold.shape, eager.shape)
        self.assertEqual(stock, gold, atol=2e-4, rtol=1e-4)
        self.assertEqual(native, gold, atol=2e-4, rtol=1e-4)
        if seed == 1701:
            mismatch = (stock - eager).abs() > 2e-4 + 1e-4 * eager.abs()
            self.assertGreater(mismatch.count_nonzero().item(), 0)


instantiate_device_type_tests(TestAMDBenchmarkReference, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()

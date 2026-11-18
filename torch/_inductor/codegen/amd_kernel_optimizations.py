from __future__ import annotations

from typing import TYPE_CHECKING

from ..runtime.hints import AMDConfigLimits, DeviceProperties


if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch


def supports_amd_kernel_optimizations(device: DeviceProperties) -> bool:
    return (
        device.type == "hip"
        and isinstance(device.cc, str)
        and device.cc.startswith("gfx")
        and isinstance(device.warp_size, int)
        and (device.warp_size in (32, 64))
        and isinstance(device.max_threads_per_block, int)
        and (device.max_threads_per_block >= device.warp_size)
        and (device.max_threads_per_block % device.warp_size == 0)
    )


def amd_reduction_value_bytes(
    input_dtypes: Iterable[torch.dtype], reduction_dtypes: Iterable[torch.dtype]
) -> int:
    """Estimate reduction value width without charging integer labels as full tiles."""
    inputs = list(input_dtypes)
    widths = [
        dtype.itemsize
        for dtype in inputs
        if dtype.is_floating_point or dtype.is_complex
    ]
    if not widths:
        widths = [dtype.itemsize for dtype in inputs]
    return max([4, *widths, *(dtype.itemsize for dtype in reduction_dtypes)])


def amd_persistent_reduction_eligible(
    device: DeviceProperties,
    rows: int,
    width: int,
    *,
    input_count: int,
    reduction_count: int,
    dtype_bytes: int = 4,
    compare: bool = False,
) -> bool:
    """Bound persistent live values and desired residency using device resources."""
    if not supports_amd_kernel_optimizations(device) or rows < 1 or width < 1:
        return False
    limits = AMDConfigLimits.from_device(device)
    padded = 1 << (width - 1).bit_length()
    if padded > limits.max_tile:
        return False
    registers = device.regs_per_multiprocessor or limits.max_tile * 4
    live_values = 2 + min(4, max(1, input_count)) + min(4, max(1, reduction_count))
    live_registers = padded * live_values * max(1, (dtype_bytes + 3) // 4)
    resident_blocks = (
        1
        if compare
        else min(
            4,
            max(
                1,
                (rows + device.multi_processor_count - 1)
                // max(1, device.multi_processor_count),
            ),
        )
    )
    budget = registers * 2 if compare else registers // resident_blocks
    return live_registers <= budget

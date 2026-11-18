from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch

    from ..runtime.hints import DeviceProperties


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

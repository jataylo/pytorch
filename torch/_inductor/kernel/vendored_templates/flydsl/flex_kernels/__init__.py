# SPDX-License-Identifier: Apache-2.0
"""FlexAttention-capable FlyDSL flash-attention kernels (forward only).

Vendored snapshot of the external ``flex_kernels`` package. Derived copies of FlyDSL's
flash kernels with ``score_mod`` / ``mask_mod`` hooks, log-sum-exp output, block-skip, and
captured-tensor reads.
"""

from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_generic import (
    build_flex_flash_generic_module,
)
from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_interface import (
    block_mask_tensors,
    flex_flash_attn,
)

__all__ = [
    "block_mask_tensors",
    "build_flex_flash_generic_module",
    "flex_flash_attn",
]

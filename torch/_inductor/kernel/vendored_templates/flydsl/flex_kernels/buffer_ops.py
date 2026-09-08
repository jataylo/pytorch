# SPDX-License-Identifier: Apache-2.0
"""Whichever `buffer_ops` the installed FlyDSL wants.

These helpers (buffer resources, raw-pointer buffer load/store) shipped as
`flydsl.expr.buffer_ops` up to FlyDSL 0.2.4. In 0.3.1 that module was **removed** from
the library and the same code moved kernel-side, so a kernel is now expected to carry
its own copy -- `kernels/common/buffer_ops.py` in the upstream tree.

Our kernels have to build against both: 0.2.4 is what the shared environment has and
what the flex path is validated against, while every parity phase needs 0.3.1. So the
library module wins when it is there, and `_vendored_buffer_ops` covers it when it is
not. Importing `flydsl.expr.buffer_ops` directly is what made the whole flex suite skip
under 0.3.1 with `cannot import name 'buffer_ops'`, so import this instead.

The two differ only in import style and a couple of op-construction details; the API
surface this uses (`create_buffer_resource`, `buffer_load`, `buffer_store`,
`create_llvm_ptr`, `extract_base_index`, `get_element_ptr`) is identical.
"""

from __future__ import annotations


try:
    from flydsl.expr.buffer_ops import *  # noqa: F403
    from flydsl.expr import buffer_ops as _impl

    SOURCE = "flydsl.expr.buffer_ops"
except ImportError:
    from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels._vendored_buffer_ops import *  # noqa: F403
    from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels import (
        _vendored_buffer_ops as _impl,
    )

    SOURCE = "vendored"


# `import *` skips the underscore-prefixed helpers, and the kernels use a few of them
# (`_unwrap_value`, `_get_buffer_flags`, `_create_i32_constant`), so re-export by name.
def __getattr__(name):
    try:
        return getattr(_impl, name)
    except AttributeError:
        raise AttributeError(f"{SOURCE} has no attribute {name!r}") from None

import functools
import logging
from importlib.machinery import PathFinder
from importlib.util import find_spec

from torch._native.common_utils import _available_version
from torch.backends import cuda as _cuda


log = logging.getLogger(__name__)
_pathfinder_find_spec = PathFinder.find_spec

# Note [the FlyDSL releases these kernels are validated against]
#
# This is a record of what has been run, not a guess at what should work: a release
# joins the set once the flex suite has passed against it. 0.2.x and 0.3.x differ in
# ways the kernels have to absorb -- `flydsl.expr.buffer_ops` is gone in 0.3.x, hence
# the shim in flex_kernels/buffer_ops.py -- so neither is inferable from the other.
# Both were run at 0.2.4 and 0.3.2 on gfx942; 0.3.2 is the faster of the two by
# 10-22% on the forward, so prefer it where there is a choice.
#
# The GEMM templates on the parity branch require 0.3.x, because `@fx.struct` values
# only expose `__cache_signature__()` from that release. A tree carrying both would
# therefore floor at 0.3.x. The only `@fx.struct` on this side is in
# flex_flash_950.py, which the tree carries but does not dispatch to, so it is
# unexercised on either release.
_FLYDSL_SUPPORTED_RELEASES = frozenset({(0, 2), (0, 3)})


def _flydsl_runtime_unavailable_reason() -> str | None:
    try:
        flydsl_spec = find_spec("flydsl")
    except (ImportError, ValueError):
        flydsl_spec = None
    if flydsl_spec is None or flydsl_spec.submodule_search_locations is None:
        return "missing optional dependency `flydsl`"

    # Query the package paths directly so this availability check does not
    # import flydsl as a side effect during regular torch imports.
    try:
        mlir_spec = _pathfinder_find_spec(
            "_mlir",
            list(flydsl_spec.submodule_search_locations),
        )
    except (ImportError, ValueError):
        mlir_spec = None
    if mlir_spec is None:
        return "missing optional dependency `flydsl._mlir` (runtime is not built)"

    flydsl_version = _available_version("flydsl")
    if flydsl_version is None:
        return "missing or invalid FlyDSL version metadata"
    if flydsl_version.release[:2] not in _FLYDSL_SUPPORTED_RELEASES:
        supported = ", ".join(
            f"{major}.{minor}.x" for major, minor in sorted(_FLYDSL_SUPPORTED_RELEASES)
        )
        return f"unsupported FlyDSL version `{flydsl_version}` (expected one of {supported})"

    return None


@functools.cache
def _check_runtime_available() -> bool:
    import torch

    if not _cuda.is_built():
        return False

    if torch.version.hip is None:
        return False

    reason = _flydsl_runtime_unavailable_reason()
    if reason is not None:
        log.debug("FlyDSL Inductor templates are unavailable: %s", reason)
        return False

    return True


def runtime_available() -> bool:
    return _check_runtime_available()

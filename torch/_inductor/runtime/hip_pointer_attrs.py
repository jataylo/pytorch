"""Draft IS_SLC bit from hipPointerGetAttributes.

Upstream hipPointerAttribute_t has no such field. IS_SLC_BIT in
allocationFlags stands in for it. mark_pointer_slc and
TORCHINDUCTOR_FAKE_SLC=1 force the bit without a driver change. A missing
HIP runtime or a failed query reads as not SLC, in Python and in C++.

config.rocm.pointer_slc (TORCHINDUCTOR_POINTER_SLC=1):
Triton launches pass distribution_tile = <first tile> if any pointer input is
SLC, else 0. The first tile is the first name matching BLOCK(_|$) in the
signature, then in the config (XBLOCK, BLOCK_M, XBLOCK_0, ...). Templates
take it from inductor_meta["config_args"]. inductor_meta["distribution_tile"]
is "launcher" when Inductor adds the argument, or "caller" when a user kernel
declares it and the launcher overwrites the placeholder.
"""

from __future__ import annotations

import ctypes
import functools
import os
import re
from dataclasses import dataclass


IS_SLC_BIT = 1 << 31
DISTRIBUTION_TILE = "distribution_tile"
CPP_DISTRIBUTION_TILE = "inductor_distribution_tile"
LAUNCHER = "launcher"
CALLER = "caller"
_BLOCK = re.compile(r"BLOCK(?:_|$)")
_CTYPE = {"i32": "int32_t", "i64": "int64_t"}
_overrides: dict[int, bool] = {}


@dataclass(frozen=True)
class HipPointerAttributes:
    memory_type: int
    device: int
    device_pointer: int
    host_pointer: int
    is_managed: bool
    allocation_flags: int

    @property
    def is_slc(self) -> bool:
        return bool(self.allocation_flags & IS_SLC_BIT)


class _Raw(ctypes.Structure):
    _fields_ = [
        ("memory_type", ctypes.c_int),
        ("device", ctypes.c_int),
        ("device_pointer", ctypes.c_void_p),
        ("host_pointer", ctypes.c_void_p),
        ("is_managed", ctypes.c_int),
        ("allocation_flags", ctypes.c_uint),
    ]


def mark_pointer_slc(ptr: int, slc: bool = True) -> None:
    _overrides[int(ptr)] = bool(slc)


def clear_pointer_slc_overrides() -> None:
    _overrides.clear()


@functools.cache
def _load_hip() -> ctypes.CDLL | None:
    for name in ("libamdhip64.so", "libamdhip64.so.6"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        lib.hipPointerGetAttributes.argtypes = [ctypes.POINTER(_Raw), ctypes.c_void_p]
        lib.hipPointerGetAttributes.restype = ctypes.c_int
        lib.hipGetLastError.restype = ctypes.c_int
        return lib
    return None


def _view(ptr: int, slc: bool) -> HipPointerAttributes:
    return HipPointerAttributes(0, -1, ptr, 0, False, IS_SLC_BIT if slc else 0)


def hip_pointer_get_attributes(ptr: int) -> HipPointerAttributes:
    ptr = int(ptr)
    if ptr in _overrides:
        return _view(ptr, _overrides[ptr])
    if ptr == 0:
        return _view(ptr, False)
    if os.environ.get("TORCHINDUCTOR_FAKE_SLC") == "1":
        return _view(ptr, True)
    lib = _load_hip()
    if lib is None:
        return _view(ptr, False)
    raw = _Raw()
    if lib.hipPointerGetAttributes(ctypes.byref(raw), ctypes.c_void_p(ptr)) != 0:
        lib.hipGetLastError()  # clear the sticky error so the next launch is clean
        return _view(ptr, False)
    return HipPointerAttributes(
        int(raw.memory_type),
        int(raw.device),
        int(raw.device_pointer or 0),
        int(raw.host_pointer or 0),
        bool(raw.is_managed),
        int(raw.allocation_flags),
    )


def pointer_is_slc(value: object) -> bool:
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        return hip_pointer_get_attributes(data_ptr()).is_slc
    if isinstance(value, int) and not isinstance(value, bool):
        return hip_pointer_get_attributes(value).is_slc
    return False


def first_tile_name(signature: dict, config: dict | None = None) -> str | None:
    """First tiling name: XBLOCK, BLOCK_M, XBLOCK_0, and so on."""
    return next((n for n in (*signature, *(config or ())) if _BLOCK.search(n)), None)


def first_tile(signature: dict, config: dict, constants: dict | None = None) -> int:
    """Value of the first tiling name in this config. 0 if there is none."""
    name = first_tile_name(signature, config)
    for source in (config, constants or {}):
        value = source.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def omit_launcher_distribution_tile(meta: dict, args: list[str]) -> list[str]:
    if meta.get(DISTRIBUTION_TILE) == LAUNCHER:
        return [a for a in args if a != DISTRIBUTION_TILE]
    return args


def prepare_distribution_tile_launcher(
    scope: dict,
    def_args: list[str],
    runner_args: list[str],
    meta: dict,
    signature: dict,
    tile: int,
) -> tuple[list[str], list[str]]:
    """Bind distribution_tile in a generated launcher. Returns (def_args, lines)."""
    mode = meta.get(DISTRIBUTION_TILE)
    if not mode:
        return def_args, []
    if DISTRIBUTION_TILE not in runner_args:
        raise AssertionError(f"{mode} kernel missing distribution_tile")
    scope["pointer_is_slc"] = pointer_is_slc
    names = [n for n in def_args if str(signature.get(n, "")).startswith("*")]
    cond = " or ".join(f"pointer_is_slc({n})" for n in names) or "False"
    return omit_launcher_distribution_tile(meta, def_args), [
        f"_any_slc = {cond}",
        f"distribution_tile = {tile} if _any_slc else 0",
    ]


def _lazy_block_expr(name: str, kernel: str) -> str | None:
    """C++ expression for a generated-kernel block on the lazy launch result."""
    arrays = {"XBLOCK": "xblocks", "YBLOCK": "yblocks", "ZBLOCK": "zblocks"}
    if name == "R0_BLOCK":
        return f"{kernel}_result.r0blocks[0]"
    prefix, _, index = name.rpartition("_")
    if name in arrays:
        prefix, index = name, "0"
    if prefix in arrays and index.isdigit():
        return f"{kernel}_result.{arrays[prefix]}[{index}]"
    return None


def tile_cpp_expr(
    signature: dict,
    *,
    config: dict | None = None,
    constants: dict | None = None,
    lazy_kernel: str | None = None,
    inductor_meta: dict | None = None,
) -> str:
    """C++ expression for the first tile. A literal once the config is known."""
    meta = inductor_meta or {}
    known = {**(meta.get("config_args") or {}), **(config or {})}
    name = first_tile_name(signature, known)
    if name is None:
        return "0"
    if lazy_kernel and meta.get("grid_type") == "PrecomputedGrid":
        grids = meta.get("precomputed_grids") or []
        values = [first_tile(signature, g.get("config") or {}) for g in grids]
        if len(set(values)) <= 1:
            return str(values[0] if values else 0)
        cases = " ".join(f"case {i}: return {v};" for i, v in enumerate(values))
        index = f"{lazy_kernel}_result.config_index"
        return f"[&] {{ switch ({index}) {{ {cases} default: return 0; }} }}()"
    if config is None and lazy_kernel:
        expr = _lazy_block_expr(name, lazy_kernel)
        if expr:
            return expr
    return str(first_tile(signature, known, constants))


def cpp_distribution_tile_decl(exprs: list[str], sig: str, tile: str) -> str:
    """C++ wrapper: declares inductor_distribution_tile. tile is a C++ expression."""
    t, name = _CTYPE[sig], CPP_DISTRIBUTION_TILE
    cond = " || ".join(f"inductor_ptr_is_slc({e})" for e in exprs) or "false"
    return f"""\
#ifdef USE_ROCM
auto inductor_ptr_is_slc = [](const void* p) -> bool {{
    if (!p) return false;
    hipPointerAttribute_t a{{}};
    if (hipPointerGetAttributes(&a, p) != hipSuccess) {{
        (void)hipGetLastError();
        return false;
    }}
    return (a.allocationFlags & {IS_SLC_BIT}u) != 0;
}};
{t} {name} = {cond}
    ? static_cast<{t}>({tile}) : 0;
#else
{t} {name} = 0;
#endif"""

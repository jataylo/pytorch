# mypy: allow-untyped-defs
"""FlyDSL codegen for Inductor's pointwise and reduction kernels.

Opt-in with ``torch._inductor.config.flydsl.simd = True`` (or
``TORCHINDUCTOR_FLYDSL_SIMD=1``) on ROCm. ``FlyDSLSIMDScheduling`` replaces
``TritonScheduling`` for non-template nodes but *inherits* everything from it --
fusion, tiling decisions, templates, combo kernels -- and only changes how a fused
pointwise/reduction node schedule is turned into a kernel. Each schedule is first tried
as a :class:`FlyDSLKernel`; anything that kernel cannot express raises
:class:`FlyDSLUnsupported` and the schedule is regenerated as a Triton kernel, so
coverage can grow without ever breaking a graph. ``config.flydsl.simd_strict`` turns the
fallback into an error, which is what the tests use to prove a kernel really is FlyDSL.

Kernel shape. Everything is one-dimensional over the flattened ``x`` range, optionally
with one reduction range, which is the tiling Inductor itself falls back to; a schedule
whose chosen tiling has ``y``/``z`` or several reduction dims is re-tiled flat rather
than refused. A block is 256 threads.

* Pointwise: each thread owns ``EPT`` consecutive x elements (16 bytes of the narrowest
  dtype). The body is traced once per element under ``range_constexpr``; loads/stores
  whose index is contiguous over that run become one buffer access of up to 16 bytes,
  loads whose index is constant over it are issued once, and everything else is a
  per-element gather.
* Reduction: ``TPR`` threads cooperate on a row (``RPB = 256 / TPR`` rows per block).
  When the row is contiguous in memory each thread takes ``EPT`` consecutive elements per
  step; when the *x* dimension is the contiguous one (a column reduction) ``TPR = 1`` and
  each thread walks its own row, so a wave's loads are coalesced across rows. Short rows
  are unrolled at trace time ("persistent"), long ones run an ``scf.for``. Cross-thread
  combines are shuffles within a wave plus one LDS round across waves.

Indices are 32-bit and masking goes through AMD buffer descriptors (see
``flydsl_simd_runtime``); buffers of 2 GiB or more, past a descriptor's range, take
64-bit global pointers instead. Dynamic shapes are refused: a buffer descriptor needs an
exact byte extent, and with symbolic sizes it would have to be computed on the device
with no way to refuse an oversized one.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import itertools
import logging
import math
import re
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, Identity, ModularIndexing
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT

from ... import config, ir, metrics
from ...codecache import code_hash, get_path
from ...dependencies import MemoryDep
from ...utils import (
    DeferredLineBase,
    expr_fits_within_32bit,
    get_fused_kernel_name,
    get_kernel_metadata,
    Placeholder,
    sympy_product,
    upcast_compute_type,
)
from ...virtualized import ops, OpsWrapper, V
from ..common import (
    BackendFeature,
    CSE,
    DeferredLine,
    IndentedBuffer,
    OpOverrides,
    PythonPrinter,
)
from ..simd import IterationRangesEntry, PartialAccumulate, SIMDKernel
from ..triton import TritonScheduling


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...ops_handler import ReductionType, StoreMode
    from ..simd_kernel_features import SIMDKernelFeatures


log = logging.getLogger(__name__)

THREADS = 256
# Pointwise kernels this large time a few launch configurations on their first call.
# At 2M elements the candidates tied on every Llama and GPT kernel, doubling their
# compile time for nothing.
AUTOTUNE_MIN_NUMEL = 1 << 22
# Transposing tiles, rarer, from here: GPT-2's 3M-element V-head select 11.1us ->
# 9.9us at its candidate (Triton 9.8us).
AUTOTUNE_MIN_TILE_NUMEL = 1 << 20
# A transposing tile's block: 50us vs 55us at 256 threads on a 4096^2 transpose.
TILE_THREADS = 128
# A column reduction launching fewer blocks than CUs: Qwen3's per-head weight
# gradient's second stage (128 columns x 319 partial sums) took 37.6us end to end at
# 256 threads, 33.8us at this many (Triton 32.5us); 34.8us at 512.
SMALL_COLUMN_THREADS = 1024
# Buffers written or read within this many bytes of traffic before a kernel count as
# still in the 256 MB last-level cache when its autotune candidates are timed.
WARM_BYTES = 128 << 20
WAVE = 64
# Rows a wave interleaves when they are read through a transpose (see
# finalize_indexing): 101us at 8 (8 lanes a row), 107us at 16, 124us at 4.
ROW_INTERLEAVE = 8
# Only rows this short: 8 lanes on a ResNet BatchNorm backward's 200704-long rows took
# 2.1 ms rather than 0.1 ms.
ROW_INTERLEAVE_MAX = 4096
# A trace-time-unrolled ("persistent") reduction body is at most this many copies of the
# loop body per loop; beyond it the code size and register pressure outgrow the reuse.
PERSISTENT_MAX_UNROLL = 16
# The row length a dynamic one is laid out for: long enough for the runtime loop.
DYNAMIC_RNUMEL = 1 << 16
# Elements of a bucketize table staged in LDS (two words each): 16 KB at most.
LDS_TABLE_MAX = 2048
# Columns a scan along a strided dimension needs before each gets a thread of its own.
SCAN_COLUMN_MIN = 8192
# Waves per CU a row scan aims for before rows widen past a wave.
SCAN_TARGET_WAVES = 16
# The widest such row, in threads (the block grows to it): 8x100000 f32 cumsum takes
# 68 us at 256, 44 us at 512, 42 us at 1024 (Triton 27 us).
SCAN_MAX_THREADS = 1024
# Chunks a row needs before Inductor's split scan takes a block per chunk.
SPLIT_SCAN_MIN_CHUNKS = 4
# Elements a thread of a split scan takes: a row's prefix crosses the chunks one
# look-back at a time (~0.5 us each), so the fewer the better. 1M f32 cumsum: ept 4
# 134 us, 8 74 us, 16 48 us, 32 44 us (Triton 46 us).
SPLIT_SCAN_EPT = 16
SPLIT_SLOT_WORDS = 8  # a chunk's workspace slot: fdp.SPLIT_SLOT_WORDS
# A sort's row across at most this many threads (the rest in runs of each thread).
SORT_MAX_THREADS = 256
SORT_MAX_LDS_BYTES = 64 * 1024
SORT_REG_RUN = 2  # consecutive slots a thread sorts in registers: see fdp._bitonic
ONE_STEP_MAX_RUNS = 8  # vector runs per thread in a one-step looped row reduction
# Preloaded f16/bf16 elements a thread converts to f32 ahead of the reduction (fdp.widen).
WIDEN_MAX_ELEMENTS = 32
# A mix-order reduction's block (see FlyDSLKernel._finalize_mix_order): its threads,
# rows a row group unrolls, and the LDS its groups' column partials may take.
MIX_ORDER_THREADS = 256
MIX_ORDER_XUNROLL = 4
MIX_ORDER_LDS_WORDS = 4096
# Its second stage (_mix_order_finalize_source): lanes splitting a column's partials,
# and the partials a lane loads at once.
MIX_FINALIZE_SLICES = 16
MIX_FINALIZE_UNROLL = 16
SCAN_PHASE = "__scan_phase__"  # replaced per traced pass of a multi-element scan
# Edge of the square tile a block covers in a {y, x} (transposing) pointwise kernel. A
# 64-wide f32 tile's 16.6 KiB of LDS leaves room for three blocks a CU: 54us vs 47us
# on an NCHW -> NHWC copy (Triton 47us), 54us vs 50us on a 4096^2 transpose.
TILE = 32
# LDS words a staged tile row takes: one word of padding spreads the transposed reads
# over the banks.
TILE_PITCH = TILE + 1
# LDS budget for staged tiles (of 64 KiB): three f32 tiles.
MAX_TILE_WORDS = 3 * TILE * TILE_PITCH
_I32_LIMIT = 2**31 - 1

_STORAGE = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.int64: "i64",
    torch.int32: "i32",
    torch.int16: "i16",
    torch.int8: "i8",
    torch.uint8: "u8",
    torch.bool: "b1",
    torch.float8_e4m3fnuz: "f8e4",
    torch.float8_e5m2fnuz: "f8e5",
}
_COMPUTE = {
    "f8e4": "f32",
    "f8e5": "f32",
    "f32": "f32",
    "f16": "f32",
    "bf16": "f32",
    "f64": "f64",
    "i64": "i64",
    "i32": "i32",
    "i16": "i32",
    "i8": "i32",
    "u8": "i32",
    "b1": "b1",
}


class FlyDSLUnsupported(RuntimeError):
    """The schedule needs something the FlyDSL SIMD kernel does not lower."""


def storage_kind(dtype: torch.dtype) -> str:
    try:
        return _STORAGE[dtype]
    except KeyError:
        raise FlyDSLUnsupported(f"dtype {dtype}") from None


def compute_kind(dtype: torch.dtype) -> str:
    return _COMPUTE[storage_kind(dtype)]


def _pow2_floor(n: int) -> int:
    return 1 << (max(1, n).bit_length() - 1)


def _pow2_ceil(n: int) -> int:
    return 1 << (max(1, n) - 1).bit_length()


def _largest_pow2_dividing(n: int, cap: int) -> int:
    w = _pow2_floor(cap)
    while w > 1 and n % w:
        w //= 2
    return w


# ---------------------------------------------------------------------------------------
# Index printing
# ---------------------------------------------------------------------------------------


class FlyDSLSIMDPrinter(PythonPrinter):
    """sympy index -> FlyDSL i32 arithmetic.

    Divisions go through ``fdp.idiv``/``fdp.imod`` (unsigned, which is exact for the
    non-negative terms Inductor indexes with); an indirect index (a ``tmp`` symbol) is
    narrowed to i32 at its use.
    """

    def _print_Symbol(self, expr):
        if symbol_is_type(expr, SymT.TMP):
            return f"fdp.idx({expr.name})"
        if symbol_is_type(expr, (SymT.SIZE, SymT.PRECOMPUTED_SIZE, SymT.UNBACKED_INT)):
            raise FlyDSLUnsupported("dynamic shapes")
        return expr.name

    def _print_Integer(self, expr):
        v = int(expr)
        if not -_I32_LIMIT <= v <= _I32_LIMIT:
            kernel = V.kernel
            if not isinstance(kernel, FlyDSLKernel) or kernel.is_dynamic:
                raise FlyDSLUnsupported("index constant outside int32")
            kernel.wide = True
            return f"fdp.ic({v})"
        return str(v)

    def _print_FloorDiv(self, expr):
        x, div = expr.args
        # pyrefly: ignore [missing-attribute]
        return f"fdp.idiv({self._print(x)}, {self._print(div)})"

    _print_CleanDiv = _print_FloorDiv

    def _print_floor(self, expr):
        # `(s0 + 15) // 16` written in sympy: floor(s0/16 + 15/16).
        num, den = sympy.fraction(sympy.together(expr.args[0]))
        if not den.is_Integer or den <= 0:
            raise FlyDSLUnsupported(f"index expression {expr}")
        if den == 1:
            # pyrefly: ignore [missing-attribute]
            return self._print(num)
        # pyrefly: ignore [missing-attribute]
        return f"fdp.idiv({self._print(sympy.expand(num))}, {int(den)})"

    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        inner = (
            # pyrefly: ignore [missing-attribute]
            self._print(x)
            if div == 1
            # pyrefly: ignore [missing-attribute]
            else f"fdp.idiv({self._print(x)}, {self._print(div)})"
        )
        # pyrefly: ignore [missing-attribute]
        return f"fdp.imod({inner}, {self._print(mod)})"

    def _print_PythonMod(self, expr):
        x, y = expr.args
        # pyrefly: ignore [missing-attribute]
        return f"fdp.imod({self._print(x)}, {self._print(y)})"

    def _print_Min(self, expr):
        # pyrefly: ignore [missing-attribute]
        out = self._print(expr.args[0])
        for a in expr.args[1:]:
            # pyrefly: ignore [missing-attribute]
            out = f"fdp.imin({out}, {self._print(a)})"
        return out

    def _print_Max(self, expr):
        # pyrefly: ignore [missing-attribute]
        out = self._print(expr.args[0])
        for a in expr.args[1:]:
            # pyrefly: ignore [missing-attribute]
            out = f"fdp.imax({out}, {self._print(a)})"
        return out

    def _print_Where(self, expr):
        # pyrefly: ignore [missing-attribute]
        c, p, q = (self._print(a) for a in expr.args)
        return f"fdp.where({c}, {p}, {q})"

    def _print_Identity(self, expr):
        # pyrefly: ignore [missing-attribute]
        return self._print(expr.args[0])

    def _print_Pow(self, expr):
        base, exp = expr.args
        if exp.is_Integer and 0 < int(exp) <= 4:
            # pyrefly: ignore [missing-attribute]
            b = self._print(base)
            return "(" + " * ".join([f"({b})"] * int(exp)) + ")"
        raise FlyDSLUnsupported(f"index power {expr}")

    def _print_Relational(self, expr):
        op = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}[
            expr.rel_op
        ]
        # pyrefly: ignore [missing-attribute]
        return f"fdp.{op}({self._print(expr.lhs)}, {self._print(expr.rhs)})"

    def _print_And(self, expr):
        # pyrefly: ignore [missing-attribute]
        return f"fdp.land({', '.join(self._print(a) for a in expr.args)})"

    def emptyPrinter(self, expr):
        raise FlyDSLUnsupported(f"index expression {type(expr).__name__}: {expr}")


_printer = FlyDSLSIMDPrinter()


# ---------------------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------------------


def _a(x: Any) -> str:
    if isinstance(x, bool):
        return f"fdp.const({x}, 'b1')"
    if isinstance(x, (int, float)):
        return (
            repr(x)
            if not isinstance(x, float) or math.isfinite(x)
            else f"float({str(x)!r})"
        )
    return str(x)


def _call(fn: str, *args: Any) -> str:
    return f"fdp.{fn}({', '.join(_a(x) for x in args)})"


_UNARY = [
    "neg",
    "abs",
    "sign",
    "signbit",
    "isnan",
    "isinf",
    "relu",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "floor",
    "ceil",
    "trunc",
    "round",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sin",
    "cos",
    "tan",
    "tanh",
    "sigmoid",
    "sinh",
    "cosh",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
    "erf",
    "erfc",
    "logical_not",
    "bitwise_not",
    "identity",
    # ocml's, as Triton's libdevice on ROCm.
    "lgamma",
    "erfinv",
    "erfcx",
    "bessel_j0",
    "bessel_j1",
    "bessel_y0",
    "bessel_y1",
    "i0",
    "i1",
    "modified_bessel_i0",
    "modified_bessel_i1",
]
_BINARY = [
    "add",
    "sub",
    "mul",
    "truediv",
    "div_rn",
    "truncdiv",
    "floordiv",
    "mod",
    "fmod",
    "remainder",
    "maximum",
    "minimum",
    "pow",
    "atan2",
    "hypot",
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "ne",
    "logical_and",
    "logical_or",
    "logical_xor",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "copysign",
    "nextafter",
    "ldexp",
]
_ALIASES = {
    "square": lambda x: _call("mul", x, x),
    "mul_rn": lambda a, b: _call("mul", a, b),
    "int_truediv": lambda a, b: _call("truediv", a, b),
    "and_": lambda a, b: _call("bitwise_and", a, b),
    "or_": lambda a, b: _call("bitwise_or", a, b),
    "xor": lambda a, b: _call("bitwise_xor", a, b),
    "lshift": lambda a, b: _call("bitwise_left_shift", a, b),
    "rshift": lambda a, b: _call("bitwise_right_shift", a, b),
    "fma": lambda a, b, c: _call("add", _call("mul", a, b), c),
    "isfinite": lambda x: _call("isfinite", x),
    "rand": lambda seed, offset: V.kernel.random("rand", seed, offset),
    "randn": lambda seed, offset: V.kernel.random("randn", seed, offset),
    "randint64": lambda seed, offset, lo, hi: _call("randint64", seed, offset, lo, hi),
    "rand_eager": lambda seed, base, threads, tid, vec: _call(
        "rand_eager", seed, base, int(threads), tid, int(vec)
    ),
    "load_seed": lambda name, offset: V.kernel.load(name, sympy.Integer(offset)),
}
# Inherited from OpOverrides and still correct here: they emit no source of their own.
_KEEP = OrderedSet(
    [
        "indirect_indexing",
        "check_bounds",
        "paren",
        "load",
        "store",
        "reduction",
        "store_reduction",
        "scan",
        "sort",
        "bucketize",
        "placeholder",
        "output",
        "partial_accumulate",
        "device_assert_async",
        "floor_to_int",
        "ceil_to_int",
        "trunc_to_int",
        "round_to_int",
    ]
)


class FlyDSLKernelOverrides(OpOverrides):
    """Every pointwise op is a call into ``flydsl_simd_runtime``, which owns promotion."""

    @staticmethod
    def constant(value, dtype):
        k = compute_kind(dtype)
        if isinstance(value, float) and not math.isfinite(value):
            return f"fdp.const(float({str(value)!r}), {k!r})"
        if k == "b1":
            return f"fdp.const({bool(value)}, 'b1')"
        if k in ("i32", "i64"):
            v = int(value)
            if k == "i32" and not -(2**31) <= v < 2**31:
                k = "i64"
            return f"fdp.const({v}, {k!r})"
        return f"fdp.const({float(value)!r}, {k!r})"

    @staticmethod
    def index_expr(expr, dtype):
        index = V.kernel.prepare_indexing(expr)
        line = f"fdp.to({V.kernel.kexpr(V.kernel.rename_indexing(index))}, {compute_kind(dtype)!r})"
        V.kernel.index_expr_roots[line] = V.kernel._root_expr(index)
        return line

    @staticmethod
    def value_expr(expr, dtype):
        # Shapes are static here, so a value expression is an index expression whose
        # dtype is honoured -- which index_expr already does.
        return FlyDSLKernelOverrides.index_expr(expr, dtype)

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None, use_compute_types=True):
        # Narrow integers stay in i32 between ops, so a cast *to* one has to truncate and
        # re-extend here or `x.to(int8)` would keep bits torch drops.
        k = storage_kind(dtype)
        if k in ("i8", "u8", "i16"):
            return f"fdp.to(fdp.to({x}, {k!r}), 'i32')"
        if k in ("f8e4", "f8e5"):
            # Always rounded (to the f32 the FP8 value stands for): unlike bf16's,
            # FP8's rounding is what a quantizing graph computes with.
            return f"fdp.to({x}, {k!r})"
        if k in ("f16", "bf16"):
            # Computed in f32, as Triton does: Inductor asks for a rounding explicitly
            # (use_compute_types=False, then an upcast) where eager's must be kept --
            # under emulate_precision_casts, or for a saved tensor. Rounding every cast
            # cost a third more bf16 conversions on Llama's RMSNorm: 25us vs 20us.
            if not use_compute_types:
                return f"fdp.to(fdp.to({x}, {k!r}), 'f32')"
            return f"fdp.to({x}, 'f32')"
        return f"fdp.to({x}, {_COMPUTE[k]!r})"

    @staticmethod
    def to_dtype_bitcast(x, dtype, src_dtype):
        if dtype.itemsize != src_dtype.itemsize or torch.bool in (dtype, src_dtype):
            raise FlyDSLUnsupported("bitcast across widths")
        src, dst = storage_kind(src_dtype), storage_kind(dtype)
        return f"fdp.bitcast({x}, {src!r}, {dst!r})"

    @staticmethod
    def inline_asm_elementwise(
        *inputs,
        asm,
        constraints=None,
        dtype=torch.float32,
        is_pure=True,
        pack=1,
        input_dtypes=None,
    ):
        if pack != 1 or dtype == torch.bool:
            raise FlyDSLUnsupported("inline asm with pack > 1")
        if constraints is None:
            constraints = ",".join(["=v"] + ["v"] * len(inputs))
        out = storage_kind(dtype)
        args = ", ".join(_a(x) for x in inputs)
        return f"fdp.inline_asm({asm!r}, {constraints!r}, {out!r}, {bool(is_pure)}, {args})"

    @staticmethod
    def frexp(x):
        key = f"frexp({x})"
        if cached := V.kernel.cse.try_get(key):
            return cached
        mantissa = V.kernel.cse.newvar(dtype=x.dtype)
        exponent = V.kernel.cse.newvar(dtype=torch.int32)
        V.kernel.compute.writeline(f"{mantissa}, {exponent} = fdp.frexp({x})")
        V.kernel.cse.put(key, (mantissa, exponent))
        return (mantissa, exponent)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def where(cond, a, b):
        return _call("where", cond, a, b)

    @staticmethod
    def clamp(x, lo, hi):
        return _call("clamp", x, lo, hi)

    @staticmethod
    def masked(mask, body, other):
        with V.kernel.mask_loads(mask, other) as new_mask:
            result = body()
        inner = getattr(result, "value", result)
        dtype = getattr(inner, "dtype", None)
        other_v = ops.constant(other, dtype) if dtype is not None else other
        out = ops.where(new_mask, result, other_v)
        return getattr(out, "value", out)


def _make(name, arity):
    if arity == 1:
        return staticmethod(lambda x: _call(name, x))
    return staticmethod(lambda a, b: _call(name, a, b))


def _unsupported(name):
    def fn(*args, **kwargs):
        raise FlyDSLUnsupported(f"op {name}")

    return staticmethod(fn)


def _install_overrides():
    implemented = (
        OrderedSet(_UNARY)
        | OrderedSet(_BINARY)
        | OrderedSet(_ALIASES)
        | OrderedSet(
            [
                n
                for n, v in vars(FlyDSLKernelOverrides).items()
                if isinstance(v, staticmethod)
            ]
        )
    )
    for n in _UNARY:
        setattr(FlyDSLKernelOverrides, n, _make(n, 1))
    for n in _BINARY:
        setattr(FlyDSLKernelOverrides, n, _make(n, 2))
    for n, f in _ALIASES.items():
        # pyrefly: ignore [bad-argument-type]
        setattr(FlyDSLKernelOverrides, n, staticmethod(f))
    for n in dir(OpOverrides):
        if n.startswith("_") or n in implemented or n in _KEEP:
            continue
        if callable(getattr(OpOverrides, n, None)):
            setattr(FlyDSLKernelOverrides, n, _unsupported(n))


_install_overrides()


# ---------------------------------------------------------------------------------------
# Access classification
# ---------------------------------------------------------------------------------------


@functools.cache
def _runtime_fingerprint() -> str:
    import os

    from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels import buffer_ops

    from . import flydsl_mod_runtime

    h = hashlib.sha256()
    here = os.path.dirname(__file__)
    for path in (
        os.path.join(here, "flydsl_simd_runtime.py"),
        flydsl_mod_runtime.__file__,
        buffer_ops.__file__,
    ):
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


_STRING = re.compile(r"'[^']*'|\"[^\"]*\"")
_NAME = re.compile(r"(?<![\w.])([A-Za-z_]\w*)")
_BUILTIN_NAMES = frozenset(OrderedSet(["fdp", "float", "True", "False", "None"]))


def _only_names(expr: str, allowed: OrderedSet[str]) -> bool:
    """Whether a generated expression reads no variable outside ``allowed``."""
    names = OrderedSet(_NAME.findall(_STRING.sub("", expr)))
    return names <= allowed | _BUILTIN_NAMES


@functools.cache
def _tune_table() -> dict[str, int]:
    """Developer overrides for the launch heuristics, for tuning experiments:
    ``TORCHINDUCTOR_FLYDSL_SIMD_TUNE="col_unroll=8,col_stride=32,force_flat=1"``."""
    import os

    out = {}
    for item in os.environ.get("TORCHINDUCTOR_FLYDSL_SIMD_TUNE", "").split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = int(v)
    return out


_TUNE_OVERRIDE: dict[str, int] = {}


def _tune(name: str, default: int = 0) -> int:
    if name in _TUNE_OVERRIDE:
        return _TUNE_OVERRIDE[name]
    return _tune_table().get(name, default)


@contextlib.contextmanager
def _tune_override(values: dict[str, int]):
    """Launch parameters for one autotuning candidate, over the env's."""
    saved = dict(_TUNE_OVERRIDE)
    _TUNE_OVERRIDE.update(values)
    try:
        yield
    finally:
        _TUNE_OVERRIDE.clear()
        _TUNE_OVERRIDE.update(saved)


def _stage_enabled(stage: str) -> bool:
    """``TORCHINDUCTOR_FLYDSL_SIMD_STAGES="pointwise,row"`` limits FlyDSL to those
    kernel classes (see ``FlyDSLKernel.stage``; ``combo`` for combo kernels), the rest
    going to Triton: what each class is worth to a model on its own."""
    import os

    stages = os.environ.get("TORCHINDUCTOR_FLYDSL_SIMD_STAGES")
    return not stages or stage in stages.split(",")


def _num_cus() -> int:
    try:
        return torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    except Exception:
        return 304


@functools.cache
def _gpu_arch() -> str | None:
    """The device's gfx target, handed to the compile workers (see fdp._set_arch)."""
    try:
        name = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    except Exception:
        return None
    return name.split(":", 1)[0] if name.startswith("gfx") else None


def _int_or_none(e) -> int | None:
    try:
        return int(e)
    except (TypeError, ValueError):
        return None


# Dynamic sizes a kernel variant may take to be multiples of 16, as Triton specializes
# integer arguments divisible by 16; ``_main`` picks that variant when they are.
SIZE_ALIGN = 16
_ASSUME_ALIGNED = [False]


@contextlib.contextmanager
def _assume_aligned():
    _ASSUME_ALIGNED[0] = True
    try:
        yield
    finally:
        _ASSUME_ALIGNED[0] = False


def _multiple(e, w: int) -> bool:
    """Whether ``e`` is a multiple of ``w``: provably, for a size expression."""
    v = _int_or_none(e)
    if v is not None:
        return v % w == 0
    try:
        sizevars = V.graph.sizevars
        # A precomputed size (``ps0``) stands for its expression: `64 * s0`, say.
        e = sympy.sympify(e).xreplace(sizevars.inv_precomputed_replacements)
        if _ASSUME_ALIGNED[0]:
            e = e.xreplace(
                {
                    s: SIZE_ALIGN * s
                    for s in e.free_symbols
                    if symbol_is_type(s, SymT.SIZE)
                }
            )
        return bool(sizevars.statically_known_multiple_of(e, w))
    except Exception:
        return False


def _run_uniform(expr: sympy.Expr, var: sympy.Symbol, w: int) -> bool:
    """Whether ``expr`` is constant over an aligned run of ``w`` values of ``var``: every
    occurrence of ``var`` sits in a ``var // d`` or ``ModularIndexing(var, d, m)`` with
    ``d`` a multiple of ``w``."""

    def fixed(e) -> bool:
        if isinstance(e, (FloorDiv, ModularIndexing)) and e.args[0] == var:
            return _multiple(e.args[1], w)
        return False

    stand_ins = {
        e: sympy.Dummy() for e in expr.atoms(FloorDiv, ModularIndexing) if fixed(e)
    }
    return var not in expr.xreplace(stand_ins).free_symbols


def _aligned_shift(arg: sympy.Expr, var: sympy.Symbol, w: int) -> bool:
    """``arg`` is ``var`` plus terms free of it that are multiples of ``w``: over an
    aligned run of ``var`` it then moves exactly as ``var`` does. A split reduction's
    ``r + 87382 * x`` is one, for runs of 2."""
    if arg == var:
        return True
    rest = sympy.expand(arg - var)
    if var in rest.free_symbols:
        return False
    for term in sympy.Add.make_args(rest):
        coeff, factor = term.as_coeff_Mul()
        c = _int_or_none(coeff)
        if c is None:
            return False
        if c % w and not _multiple(c * factor, w):
            return False
    return True


def classify_access(
    expr: sympy.Expr,
    var: sympy.Symbol,
    w: int,
    uniform: frozenset[str] = frozenset(),
    any_base: bool = False,
) -> str:
    """How ``expr`` (in terms of the root index symbols) behaves over an aligned run of
    ``w`` consecutive values of ``var``.

    ``"contig"``  element offsets are ``base + j`` with ``base`` a multiple of ``w``;
    ``"uniform"`` the offset is the same for all ``w``;
    ``"other"``   anything else, which is gathered element by element.

    ``uniform`` names indirect-index symbols (``tmp`` values) known to be constant over
    the run -- an index loaded once per row, say -- which may appear as terms of the
    base; anything else computed makes the access a gather.
    """
    tmps = OrderedSet(
        [s.name for s in expr.free_symbols if symbol_is_type(s, SymT.TMP)]
    )
    if tmps - uniform:
        return "other"
    if w <= 1:
        return "uniform" if var not in expr.free_symbols else "other"
    expr = sympy.expand(expr)

    def runs(arg):
        # An aligned run of ``arg`` -- ``var`` shifted, or itself such a run, as in a
        # view's ``(x % 1024) // 64`` -- divides and wraps exactly as ``var`` would.
        return _aligned_shift(arg, var, w) or (
            classify_access(arg, var, w, uniform) == "contig"
        )

    linear = 0
    aligned = True
    for term in sympy.Add.make_args(expr):
        coeff, factor = term.as_coeff_Mul()
        c = _int_or_none(coeff)
        if c is None:
            return "other"
        if var not in factor.free_symbols:
            # A uniform term moves the run's base; it must keep it aligned.
            if c % w and not _multiple(c * factor, w):
                aligned = False
            continue
        if factor == var:
            linear += c
            continue
        if isinstance(factor, ModularIndexing) and runs(factor.args[0]):
            d, m = factor.args[1:]
            if _multiple(d, w):
                if c % w:
                    aligned = False
                continue
            if d == 1 and _multiple(m, w):
                linear += c
                continue
            return "other"
        if isinstance(factor, FloorDiv) and runs(factor.args[0]):
            if _multiple(factor.args[1], w):
                if c % w:
                    aligned = False
                continue
            return "other"
        if _run_uniform(factor, var, w):
            # var only enters through quotients fixed over the run, as in a view's
            # `768 * ((r + 253 * (x // 768)) % 6304)`.
            if c % w:
                aligned = False
            continue
        return "other"
    if linear == 0:
        return "uniform"
    if linear == 1 and (aligned or any_base):
        return "contig"
    return "other"


def _linear_in(expr: sympy.Expr, var: sympy.Symbol) -> bool:
    """Whether ``expr`` steps by one element with ``var``, its runs aligned or not (a
    row pitch of 8197, or ``ks0``)."""
    return classify_access(expr, var, 2, any_base=True) == "contig"


# ---------------------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------------------


class FlyDSLKernel(SIMDKernel):
    """A fused pointwise or reduction kernel emitted as a FlyDSL module: one flat ``x``
    tile, or ``x`` by ``r0_``, over a fixed 256-thread block. See SIMD_BACKEND.md."""

    overrides = FlyDSLKernelOverrides  # type: ignore[assignment]
    kexpr = staticmethod(_printer.doprint)  # type: ignore[assignment]

    def __init__(self, tiling: dict[str, sympy.Expr], **kwargs) -> None:
        keys = OrderedSet(tiling)
        if "x" not in keys or not (
            keys <= OrderedSet(["x", "r0_"]) or keys == OrderedSet(["x", "y"])
        ):
            raise FlyDSLUnsupported(f"tiling {sorted(tiling)}")
        super().__init__(tiling, **kwargs)
        # A dynamic x extent (batch, sequence): the kernel takes it as an argument, its
        # bounds mask always on, and the static choices use its size hint.
        self.xnumel_dyn: sympy.Expr | None = None
        xs = V.graph.sizevars.simplify(self.numels["x"])
        if _int_or_none(xs) is None:
            self.xnumel_dyn = xs
            self.xnumel = int(V.graph.sizevars.optimization_hint(xs, fallback=8192))
        else:
            self.xnumel = int(xs)
        self.is_reduction_kernel = "r0_" in self.numels
        # A dynamic row length: a runtime loop over it, masked; the layout choices see a
        # long row (DYNAMIC_RNUMEL), which keeps them off the unrolled forms.
        self.rnumel_dyn: sympy.Expr | None = None
        self.rnumel: int = 1
        if self.is_reduction_kernel:
            rs = V.graph.sizevars.simplify(self.numels["r0_"])
            if _int_or_none(rs) is None:
                self.rnumel_dyn = rs
                self.rnumel = DYNAMIC_RNUMEL
            else:
                self.rnumel = int(rs)
        # A {y, x} pointwise kernel runs tile_f x tile_s tiles: see _finalize_tile2d.
        self.tile2d = "y" in self.numels
        self.ynumel_dyn: sympy.Expr | None = None
        self.ynumel = 1
        if self.tile2d:
            ys = V.graph.sizevars.simplify(self.numels["y"])
            if _int_or_none(ys) is None:
                self.ynumel_dyn = ys
                self.ynumel = int(V.graph.sizevars.optimization_hint(ys, fallback=8192))
            else:
                self.ynumel = int(ys)
        self.fast = "x"  # the dimension a thread's vector runs follow
        # Tile edges along the fast and slow dimensions.
        self.tile_f = self.tile_s = TILE
        self.staged: list[dict[str, Any]] = []
        # i64 index values, for 2**31 elements or more (a buffer or index constant past
        # them sets it too); buffers under 2 GiB still take i32 descriptor offsets.
        self.wide = False
        if not self.is_dynamic:
            if self.xnumel * self.rnumel * self.ynumel > _I32_LIMIT:
                self.wide = True
        else:
            # i32 offsets, as Inductor guards for Triton's 32-bit indexing; the bound
            # leaves room for a mask's block-sized overshoot.
            total = V.graph.sizevars.simplify(
                self.numels["x"]
                * self.numels.get("r0_", 1)
                * self.numels.get("y", 1)
            )
            if not expr_fits_within_32bit(total):
                raise FlyDSLUnsupported("dynamic extent not known to fit int32")
            V.graph.sizevars.check_leq(total, _I32_LIMIT // 2)
        self.outside_loop_vars: OrderedSet[Any] = OrderedSet()
        self.post_loop_combine = IndentedBuffer()
        # Per loop step, around the run of a thread's elements: chunked combines.
        self.chunk_pre = IndentedBuffer()
        self.chunk_post = IndentedBuffer()
        # Pointwise only: the indexing and loads alone, replayed ahead of the body.
        self.prefetch = IndentedBuffer()
        # Elements between those of the prefetch that can issue a load: the gcd of its
        # loads' runs (0: none yet).
        self.prefetch_step = 0
        self._node_index_vars: list[list[list[sympy.Expr]]] = []
        self.acc_counter = 0
        self.uses_smem = False
        # Search tables staged in LDS by the block before its body: arg -> (kind, size).
        self.lds_tables: dict[str, tuple[str, int]] = {}
        self.max_components = 1
        self.reduction_cache: dict[Any, Any] = {}
        self.loop_stored_args: dict[str, str] = {}
        self.helpers: dict[str, str] = {}  # module-level functions: scan combines
        # Loads of later unrolled passes issued before the first: see _queue_preload.
        self._preload_at: int | None = None
        self._preload_indent = 0
        self.preload_lines: list[Any] = []
        # Where the first pass's loop ends: see _insert_widen.
        self._widen_at: int | None = None
        self.stored_args: OrderedSet[str] = OrderedSet()
        self.index_expr_roots: dict[str, sympy.Expr] = {}
        # Decided in finalize_indexing. A thread owns `unroll` runs of `ept` contiguous
        # elements per step; in a reduction, `tpr` threads `stride` apart share a row,
        # and a block covers `rpb` rows `xunroll` times over.
        self.ept = 1
        self.unroll = 1
        self.tpr = 1
        self.stride = 1
        self.interleaved = False  # rows `stride` lanes apart within a wave
        self.threads = THREADS  # block size
        self.in_combo = False
        self.rpb = self.threads
        self.xunroll = 1
        # Adjacent columns a lane in a column reduction (see finalize_indexing).
        self.cpair = 1
        self.cpair_failed = False  # a load of the row could not be paired
        self._pair_lines: IndentedBuffer | None = None
        self.persistent = False
        self.one_step = False  # a looped row reduction that covers the row in one step
        self.scan_runs = (
            False  # a scan over runs of ept > 1 elements: two traced passes
        )
        self.has_scan = False
        self.scan_components = 0  # of the widest scan whose rows span several waves
        self.split_chunks = 0  # blocks a row of a split scan takes (_finalize_scan)
        self.has_sort = False
        self.sort_words = 0  # LDS words of the kernel's sorts, after _scan_base
        self.rstep = 1
        self.load_site = 0
        self.region = 0  # bumped per flushed scf.for loop: see _site_key
        # Values constant over one run of a thread's elements (see classify_access).
        self.uniform_vars: OrderedSet[str] = OrderedSet()
        generate = self.cse.generate

        def tracked_generate(buffer, expr, *args, **kwargs):
            var = generate(buffer, expr, *args, **kwargs)
            if isinstance(expr, str) and _only_names(expr, self.uniform_vars):
                self.uniform_vars.add(str(var))
            return var

        self.cse.generate = tracked_generate  # type: ignore[method-assign]

    # -- configuration ------------------------------------------------------------------

    @staticmethod
    def _static(expr, what) -> int:
        v = _int_or_none(V.graph.sizevars.simplify(expr))
        if v is None:
            raise FlyDSLUnsupported(f"dynamic {what}")
        return v

    def should_use_persistent_reduction(self) -> bool:
        # Keep every reduction tree a loop so DisableReduction flushes the body; whether
        # the loop is unrolled at trace time is decided separately (self.persistent).
        return False

    def should_use_cooperative_reduction(self) -> bool:
        return False

    def want_no_x_dim(self):
        return False

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return storage_kind(dtype)

    def _tree_sym(self, prefix: str):
        return next(t for t in self.range_trees if t.prefix == prefix).symbol()

    def _x_sym(self):
        return self._tree_sym("x")

    def _r_sym(self):
        return self.range_trees[-1].symbol() if self.is_reduction_kernel else None

    def _root_expr(self, index: sympy.Expr) -> sympy.Expr:
        """``index`` with every range-tree entry replaced by its root-index expression."""
        subs = {
            s: e.expr
            for s, e in self.range_tree_nodes.items()
            if s in index.free_symbols
        }
        if index.has(Identity):
            # torch.cat's offsets arrive wrapped, which would hide them from
            # classify_access.
            index = index.replace(Identity, lambda a: a)
        return sympy.expand(index.xreplace(subs)) if subs else index

    def _buffer_itemsizes(self) -> list[int]:
        sizes = []
        for node in self.features.scheduler_nodes():
            for dep in node.read_writes.reads_and_writes():
                try:
                    sizes.append(V.graph.get_dtype(dep.name).itemsize)
                except Exception:
                    continue
        return sizes or [4]

    def _row_inputs(self) -> int:
        """Kernel inputs read in full, one element per (row, column)."""
        full = self.xnumel * self.rnumel
        written = OrderedSet(
            d.name
            for n in self.features.scheduler_nodes()
            for d in n.read_writes.writes
        )
        names: OrderedSet[str] = OrderedSet()
        for n in self.features.scheduler_nodes():
            for d in n.read_writes.reads:
                if d.name in written or not isinstance(d, MemoryDep):
                    continue
                try:
                    if int(d.get_numel()) == full:
                        names.add(d.name)
                except TypeError:
                    continue
        return len(names)

    def _describe(self) -> str:
        if self.tile2d:
            return (
                f"tiled pointwise ynumel={self.ynumel} xnumel={self.xnumel} fast={self.fast} "
                f"ept={self.ept} staged={[(s['arg'], s['w']) for s in self.staged]}"
            )
        if not self.is_reduction_kernel:
            return f"pointwise xnumel={self.xnumel} ept={self.ept} unroll={self.unroll}"
        mode = "column" if self.stage == "column" else "row"
        pairs = " column_pairs=True" if self._pair_lines is not None else ""
        return (
            f"{mode} reduction xnumel={self.xnumel} rnumel={self.rnumel} ept={self.ept} "
            f"unroll={self.unroll} tpr={self.tpr} stride={self.stride} xunroll={self.xunroll} "
            f"persistent={self.persistent}{pairs}"
        )

    def _finalize_tile2d(self, widest_vec: int) -> None:
        """Layout of a {y, x} pointwise kernel: each block covers a tile_f x tile_s tile.

        A thread's vector runs follow the ``fast`` dimension, chosen so the stores are
        contiguous along it. Loads contiguous along the other dimension are staged:
        read coalesced along that dimension into an LDS tile, then read back
        transposed (see ``load`` and ``emit_inner``).
        """
        stores = self._memory_roots(loads=False) or []
        x, y = self._x_sym(), self._tree_sym("y")
        x_st = sum(_linear_in(e, x) for e in stores)
        y_st = sum(_linear_in(e, y) for e in stores)
        self.fast = "y" if y_st > x_st else "x"
        f = self._tree_sym(self.fast)
        # A dynamic row pitch leaves the runs unaligned: ept falls to what is provable,
        # down to 1, where a wave's lanes still write consecutive elements.
        fast_roots = [e for e in (self._memory_roots() or []) if _linear_in(e, f)]
        ept = widest_vec
        while ept > 1 and not all(
            classify_access(e, f, ept) == "contig" for e in fast_roots
        ):
            ept //= 2
        self.ept = _tune("ept", ept)
        # A tile narrow along the stores and long along the loads when the stores' rows
        # are close together, as when the fast dimension is a few channels, so the
        # neighbouring tiles fill each other's cache lines: an NCHW -> NHWC f32 copy
        # 52us -> 48us (Triton 46us), where a 4096^2 transpose (rows 16 KiB apart) goes
        # 52.6us -> 57.5us.
        s = self._tree_sym("x" if self.fast == "y" else "y")
        pitch = [
            e.subs({f: 0, s: 1}) - e.subs({f: 0, s: 0})
            for e in stores
            if _linear_in(e, f)
        ]
        narrow = bool(pitch) and all(p.is_Integer and 0 < p <= 64 for p in pitch)
        self.tile_f = _tune("tile_f", TILE // 2 if narrow else TILE)
        self.tile_s = _tune("tile_s", TILE * 2 if narrow else TILE)
        while self.ept > 1 and self.threads * self.ept > self.tile_f * self.tile_s:
            self.ept //= 2
        # Runs of a pass: tile_f // ept threads across the fast dimension, the block's
        # remaining threads down the slow one; `unroll` passes cover the tile.
        self.unroll = self.tile_f * self.tile_s // (self.threads * self.ept)
        self.rpb = self.threads * self.ept // self.tile_f

    def _finalize_scan(self, column: bool, widest_vec: int) -> None:
        """Layout of a scan: each run of the row gives a thread ``ept`` contiguous
        elements, scanned in registers and then across the row's lanes (at most a wave:
        shuffles only), and combined with the carry from the runs before it. A scan along a
        non-contiguous dimension gives each thread a whole column instead, scanned
        sequentially with the lanes coalesced along x."""
        self.ept = 1
        if column:
            self.stride, self.tpr = self.threads, 1
            self.rpb = self.threads
        else:
            self.stride = 1
            # Runs of ept contiguous elements: one vector load each, scanned in
            # registers, with one cross-lane scan per run.
            self.ept = _tune("ept", self._r_pow2_dividing(min(4, widest_vec)))
            runs = -(-self.rnumel // self.ept)
            # Too few rows to fill the GPU with a wave each: rows of several waves,
            # their totals exchanged through LDS.
            lanes = SCAN_TARGET_WAVES * _num_cus() * WAVE
            span = _pow2_floor(max(1, lanes // self.xnumel))
            span = max(WAVE, min(_tune("scan_max_threads", SCAN_MAX_THREADS), span))
            self.tpr = _tune("row_tpr", min(span, _pow2_ceil(runs)))
            self.threads = max(self.threads, self.tpr)
            self.rpb = self.threads // self.tpr
            split_ept = _tune("ept", SPLIT_SCAN_EPT)
            chunks = -(-self.rnumel // (self.tpr * split_ept))
            if self._split_scan_ok() and self.tpr > WAVE and chunks >= _tune(
                "split_scan_min_chunks", SPLIT_SCAN_MIN_CHUNKS
            ):
                # A block a chunk of the row, chained by fdp's decoupled look-back.
                self.split_chunks, self.ept = chunks, split_ept
                self.persistent, self.unroll = True, 1
                self.rstep = self.tpr * self.ept
                return
        iters = -(-self.rnumel // (self.tpr * self.ept))
        # A dynamic row (DYNAMIC_RNUMEL long here) loops: masked, its carry passed on.
        self.persistent = (
            self.rnumel_dyn is None and iters * self.ept <= PERSISTENT_MAX_UNROLL
        )
        if not self.persistent:
            self.unroll = _tune(
                "row_unroll",
                max(1, min(4, PERSISTENT_MAX_UNROLL // self.ept, iters // 2)),
            )
        self.rstep = self.tpr * self.ept * self.unroll

    def _finalize_sort(self) -> None:
        """Layout of a sort: the whole row in one persistent step, an element a thread
        a run, for fdp.row_sort to stage it in LDS."""
        if self.rnumel_dyn is not None:
            raise FlyDSLUnsupported("sort over a dynamic row")
        self.stride = self.ept = 1
        self.tpr = min(_tune("sort_tpr", SORT_MAX_THREADS), _pow2_ceil(self.rnumel))
        self.threads = _tune("sort_threads", max(self.threads, self.tpr))
        self.rpb = self.threads // self.tpr
        # Every one of the row's power-of-two slots some element's: see fdp.row_sort.
        self.unroll = _pow2_ceil(self.rnumel) // self.tpr
        if self.unroll > PERSISTENT_MAX_UNROLL:
            raise FlyDSLUnsupported("sort of a long row")
        self.persistent = True
        self.rstep = self.tpr * self.unroll

    def _split_scan_ok(self) -> bool:
        """Inductor split this scan (``ir.SplitScan``: one value, a long row), and the
        kernel has nothing a chunk of the row could not compute alone."""
        nodes = list(self.features.scheduler_nodes())
        return (
            self.rnumel_dyn is None
            and self.xnumel_dyn is None
            and any(n.is_split_scan() for n in nodes)
            and all(n.is_split_scan() or not n.is_reduction() for n in nodes)
        )

    @property
    def _r_base(self) -> str:
        return "_rc0 + " if self.split_chunks else ""

    @property
    def _r_extent(self) -> int:
        """The part of the row a block's persistent loop covers."""
        return self.rstep if self.split_chunks else self.rnumel

    def _tile_counts(self) -> tuple[int, int]:
        """Tiles along (fast, slow)."""
        nf = self.xnumel if self.fast == "x" else self.ynumel
        ns = self.ynumel if self.fast == "x" else self.xnumel
        return -(-nf // self.tile_f), -(-ns // self.tile_s)

    def _tile_edge(self, dim: str) -> int:
        return self.tile_f if dim == self.fast else self.tile_s

    @property
    def _tile_pitch(self) -> int:
        # LDS words a staged tile row (one fast index) takes: see TILE_PITCH.
        return self.tile_s + 1

    def _pure_reduction(self) -> bool:
        """Every node is a reduction: the kernel stores only row results."""
        return all(n.is_reduction() for n in self.features.scheduler_nodes())

    def _reduction_types(self) -> list[str]:
        return [
            t
            for n in self.features.reduction_nodes()
            if isinstance(n.node, ir.ComputedBuffer)
            and (t := n.node.get_reduction_type()) is not None
        ]

    def split_and_set_ranges(self, lengths):
        index_vars = super().split_and_set_ranges(lengths)
        self.__dict__.setdefault("_node_index_vars", []).append(index_vars)
        return index_vars

    def _memory_roots(self, loads=True, stores=True) -> list[sympy.Expr] | None:
        """Root-symbol indices of the loads and/or stores alone -- ``finalize_indexing``
        also sees index_expr values such as a split reduction's bounds check -- or None
        if the per-node ranges recorded by ``split_and_set_ranges`` do not line up."""
        from ...loop_body import MemoryUsageType

        nodes = list(self.features.scheduler_nodes())
        # The first pass over the schedule records one call per node, in order.
        calls = self._node_index_vars
        if len(calls) < len(nodes):
            return None
        kinds = (MemoryUsageType.LOAD,) if loads else ()
        if stores:
            kinds += (MemoryUsageType.STORE, MemoryUsageType.STORE_REDUCTION)
        roots = []
        for node, index_vars in zip(nodes, calls):
            body = getattr(node, "_body", None)
            if body is None:
                return None
            exprs = body.indexing_from_args(index_vars)
            for t in kinds:
                roots.extend(
                    self._root_expr(exprs[e.index_name]) for e in body.memory_usage[t]
                )
        return roots

    def _has_gather(self) -> bool:
        """Some load's index goes through another load (an embedding, a loss's target)."""
        from ...loop_body import MemoryUsageType

        for node in self.features.scheduler_nodes():
            body = getattr(node, "_body", None)
            if body is None:
                continue
            for e in body.memory_usage[MemoryUsageType.LOAD]:
                if free_symbol_is_type(
                    body.indexing_exprs[e.index_name], SymT.INDIRECT
                ):
                    return True
        return False

    def _stream_itemsizes(self, loads: bool = True) -> list[int]:
        """Element sizes of the buffers a pointwise kernel streams: those whose index
        runs contiguously with x, which broadcast operands do not. Only the stored
        ones without ``loads``."""
        from ...loop_body import MemoryUsageType

        nodes = list(self.features.scheduler_nodes())
        calls = self._node_index_vars
        x = self._x_sym()
        if len(calls) < len(nodes) or x is None:
            return []
        sizes = []
        kinds = [MemoryUsageType.STORE]
        if loads:
            kinds.insert(0, MemoryUsageType.LOAD)
        for node, index_vars in zip(nodes, calls):
            body = getattr(node, "_body", None)
            if body is None:
                return []
            exprs = body.indexing_from_args(index_vars)
            for t in kinds:
                for e in body.memory_usage[t]:
                    root = self._root_expr(exprs[e.index_name])
                    if classify_access(root, x, 2) == "contig":
                        sizes.append(V.graph.get_dtype(e.buffer_name).itemsize)
        return sizes

    def _transposed_rows(self, roots) -> bool:
        """Whether some load strides along the row but is contiguous across rows, as a
        row of an NCHW tensor read as NHWC."""
        x, r = self._x_sym(), self._r_sym()
        for e in roots:
            if r not in e.free_symbols or classify_access(e, r, 2) == "contig":
                continue
            # Away from x == 0, where a shifted index such as (x0 - 1) % 196 wraps.
            step = e.subs({r: 0, x: 2}) - e.subs({r: 0, x: 1})
            if step == 1:
                return True
        return False

    def _has_atomic(self, per_element: bool = False) -> bool:
        """Whether a node stores atomically; with ``per_element``, not counting a
        reduction kernel's nodes that run once a row (no reduction range)."""
        from ...loop_body import MemoryUsageType

        for node in self.features.scheduler_nodes():
            body = getattr(node, "_body", None)
            if body is None or not any(
                e.mode == "atomic_add" for e in body.memory_usage[MemoryUsageType.STORE]
            ):
                continue
            if per_element and self.is_reduction_kernel and not node.is_reduction():
                # A pointwise node's reduction range is always empty; one per row
                # spans just the kernel's rows.
                sizes, _ = node.get_ranges()
                if V.graph.sizevars.statically_known_equals(
                    sympy_product(sizes), self.numels["x"]
                ):
                    continue
            return True
        return False

    def finalize_indexing(self, indices: Sequence[sympy.Expr]) -> None:
        """Choose the thread layout -- ``ept``, ``unroll``, ``tpr``, ``stride``,
        ``xunroll``, persistence -- from the kernel's shape and access patterns."""
        widest_vec = max(1, min(8, 16 // min(self._buffer_itemsizes())))
        roots = [self._root_expr(i) for i in indices]
        tiled = self.tile2d and not self.in_combo
        self.threads = _tune("threads", TILE_THREADS if tiled else THREADS)
        self.rpb = self.threads
        if self.tile2d:
            self._finalize_tile2d(widest_vec)
            return
        # Enough blocks to fill the GPU a few times over before a thread takes more work.
        enough_blocks = 4 * _num_cus()
        atomic = self._has_atomic()
        atomic_elems = self._has_atomic(per_element=True)
        if atomic_elems:
            # Atomics are issued an element at a time, and a wave's spread over more
            # than two elements a lane goes slower: an embedding-gradient scatter
            # takes 58us at 4 a lane, 32us at 2 in f32 (Triton 36us); 34us at 4,
            # 23us at 2 in bf16 (Triton 21us). Not one atomic a row, after its
            # reduction, which leaves the row's loads and stores at full width.
            widest_vec = min(widest_vec, 2)
        if not self.is_reduction_kernel:
            if not atomic and not self.in_combo:
                # One-wave blocks: a [32, 512, 768] bf16 heads permute 18.6us -> 14.3us
                # (Triton 18.1us), a 16M f32 add 45.8 -> 41.2us, an NCHW bias add
                # 38.7 -> 36.0us; none slower.
                self.threads = self.rpb = _tune("threads", WAVE)
            streams = self._stream_itemsizes()
            if streams:
                # Runs as wide as the widest stream's 16-byte vector: wider ones split
                # its loads into strided halves, each instruction a wave's 64 lanes over
                # twice the lines -- 136us vs 105us on ResNet's f32 threshold backward,
                # whose bool mask alone would allow eight a lane (Triton 100us).
                widest_vec = min(widest_vec, max(1, 16 // max(streams)))
                # Unless that leaves every store short of a 16-byte vector: twice as
                # wide, an f32 -> bf16 cast stores 16 bytes a lane for two strided
                # loads, 44.7us -> 41.3us on 24M elements (Triton 41.9us). Not past the
                # atomics' two: the f32 scatter of a standalone embedding gradient went
                # back to 100us at four (Triton 51us, 48us at two).
                stores = self._stream_itemsizes(loads=False)
                if stores and not atomic_elems and 16 // max(stores) > widest_vec:
                    widest_vec = min(2 * widest_vec, 16 // max(stores))
            self.ept = _tune("ept", self._x_pow2_dividing(widest_vec))
            # Two runs per thread keep two vector loads in flight per input, as Triton's
            # default pointwise config does, once the grid is big enough to afford it.
            per_block = self.threads * self.ept
            self.unroll = _tune(
                "unroll", 2 if self.xnumel >= 2 * per_block * enough_blocks else 1
            )
            return
        x, r = self._x_sym(), self._r_sym()
        # One vote per load: `indices` holds each distinct index once, so three inputs
        # read at one x-contiguous index tied a small r-contiguous one and kept the
        # row layout -- 184us vs Triton's 28us on Llama's RMSNorm weight gradient.
        votes = self._memory_roots(stores=False) or roots
        r_contig = sum(
            classify_access(e, r, 2) == "contig" for e in votes if r in e.free_symbols
        )
        x_contig = sum(
            classify_access(e, x, 2) == "contig" for e in votes if r in e.free_symbols
        )
        if self.features.contains_op("sort"):
            self._finalize_sort()
            return
        if self.features.contains_op("scan"):
            # A column per thread only with columns to fill the GPU: otherwise the
            # row's lanes share each column, its loads gathered. cumsum(0) of 8192 x 64
            # f32 2854us -> 100us (Triton 43us), 100000 x 8 22.4ms -> 0.68ms, 8192 x
            # 4096 3.3ms -> 2.5ms (Triton 2.7ms); 64 x 65536 stays a column's (25us
            # against 100us).
            column = x_contig > r_contig and self.xnumel >= _tune(
                "scan_column_min", SCAN_COLUMN_MIN
            )
            self._finalize_scan(column, widest_vec)
            return
        gather = False
        if x_contig > r_contig:
            # Column reduction: x is the contiguous dimension. Lanes run along x (`stride`
            # of them, so a wave's loads coalesce) and the block's remaining threads split
            # the row range, combining through LDS at the end.
            # One wave per row slice measured best by far on an 8192^2 column sum: 85us
            # against 140us for 32 or 16 lanes, though those launch 2-4x the blocks.
            # Coalescing beats occupancy here, and Inductor's split reduction already
            # handles the few-columns, very-long-rows case at the graph level.
            self.ept = 1
            self.stride = _tune("col_stride", WAVE)
            if (
                not self.in_combo
                and not _tune("threads", 0)
                and -(-self.xnumel // self.stride) < _num_cus()
            ):
                # A few blocks over the whole GPU (a split reduction's second stage):
                # more of each block's waves on the rows.
                self.threads = SMALL_COLUMN_THREADS
            self.tpr = self.threads // self.stride
            if not _tune("threads", 0):
                # No more lanes a column than it has rows: ResNet's BatchNorm stats
                # finalize (64 channels x 3 partial sums) ran 1024 threads, 192 of them
                # loading, and combined through LDS across all of them.
                while self.tpr > 1 and V.graph.sizevars.statically_known_geq(
                    self.tpr // 2, self.rnumel
                ):
                    self.tpr //= 2
                self.threads = self.tpr * self.stride
            self.rpb = self.stride
        else:
            self.ept = self._r_pow2_dividing(widest_vec)
            # Narrower vectors when they tile a one-wave row exactly where the widest
            # leave a partly masked last run: 768 = 64 lanes x 3 runs of 4. Not with
            # many row-sized inputs, where the wider loads' fewer instructions win
            # (these kernels are vector-ALU-bound): 49us vs 63us on a 4-input add +
            # LayerNorm over 768, against 20us vs 19.6us with 2 inputs. Nor with a
            # gathered row, whose loads wait on the index load: GPT's embedding +
            # LayerNorm 15.3us -> 14.3us at 8 a lane (Triton 12.9us).
            ept = self.ept
            while (
                ept > 1
                and not self._has_gather()
                and self._row_inputs() < 4
                and self.rnumel % (WAVE * ept)
                and WAVE * self.ept < self.rnumel <= WAVE * PERSISTENT_MAX_UNROLL
            ):
                ept //= 2
                if self.rnumel % (WAVE * ept) == 0:
                    self.ept = ept
                    break
            r_roots = [
                e for e in (self._memory_roots() or roots) if r in e.free_symbols
            ]
            if r_roots and not any(
                classify_access(e, r, self.ept) == "contig" for e in r_roots
            ):
                # Nothing vectorizes (an unaligned split-reduction offset, say), so a
                # thread's elements are scalar loads anyway: one per thread keeps a
                # wave's loads adjacent, and deeper unrolling keeps them in flight.
                # 70us vs 127us on a BatchNorm backward reduction over NCHW.
                self.ept = 1
                gather = True
            self.ept = _tune("ept", self.ept)
            runs = -(-self.rnumel // self.ept)
            if runs >= 8 and (self._pure_reduction() or 32 <= runs <= WAVE):
                # Give each thread two runs of a short row (and, for a bare reduction,
                # two rows below), trading threads for loads in flight: 47us vs 55us
                # on a 512-wide bf16 softmax backward. A normalization over a 64-wide
                # row (16 runs) instead lost 89us -> 105us at 8 threads a row.
                runs //= 2
            tpr = min(self.threads, _pow2_ceil(runs))
            if (
                tpr >= WAVE
                and not self.in_combo
                and not _tune("threads", 0)
                and self.xnumel < _num_cus()
            ):
                # Fewer rows than CUs: each row's block as wide as its runs allow, for
                # more of its loads in flight at once. BERT's loss tail (one row of
                # 8192 gathered logits) 36.3us -> 25.3us at 1024 threads (Triton 26.4us);
                # Llama's (4096) 15.0us at one wave, 14.1us at 512 threads and 12.2us
                # at 1024 of 4 elements (Triton 12.1us). Narrower runs only pay for
                # gathers: a contiguous 4096-wide sum went 1.8us -> 2.3us.
                gathers = self._has_gather()
                while (
                    gathers
                    and self.ept > 4
                    and not _tune("ept", 0)
                    and -(-self.rnumel // self.ept) < SMALL_COLUMN_THREADS
                ):
                    self.ept //= 2
                full_runs = -(-self.rnumel // self.ept)
                self.threads = tpr = min(SMALL_COLUMN_THREADS, _pow2_ceil(full_runs))
            elif (
                tpr > WAVE
                and not atomic
                and -(-self.rnumel // (WAVE * self.ept)) * self.ept
                <= PERSISTENT_MAX_UNROLL
            ):
                # One wave still holds the row in registers: shuffles only, no LDS
                # round trip and barriers per reduction. 31us vs 39us on a 768-wide
                # add+layernorm, where 128 threads also left a quarter of them idle.
                tpr = WAVE
            elif (
                tpr > WAVE
                and self.mix_order_reduction
                and self.rnumel % self.ept == 0
                and (
                    fit := next(
                        (
                            t
                            for t in range(self.threads // WAVE * WAVE, WAVE, -WAVE)
                            if (self.rnumel // self.ept) % t == 0
                        ),
                        None,
                    )
                )
            ):
                # A mix-order row as wide as whole waves that tile it exactly go: its
                # column partials take a register a lane per element, so more lanes,
                # not fewer, beat more elements a lane. A 16384 x 1536 RMSNorm backward
                # 76us -> 60us at 192 lanes of 8, 3072 wide 174us -> 153us at 192 of
                # 16; 2048 wide 172us at 256 lanes of 8 against 192us at 128 of 16.
                tpr = fit
            elif (
                tpr > WAVE
                and not self.mix_order_reduction
                and not self._pure_reduction()
                and self.rnumel % PERSISTENT_MAX_UNROLL == 0
                and WAVE < (fit := self.rnumel // PERSISTENT_MAX_UNROLL) <= self.threads
                and fit % (WAVE // 2) == 0
            ):
                # Rows that waves (or half waves) hold in registers, a full unroll a
                # lane: a normalization reads its row more than once, which the
                # one-step loop goes back to L2 for. A 4096-wide bf16 RMSNorm 107us ->
                # 93us (Triton 147us) and its backward 95us -> 57us; 2048-wide 53us ->
                # 47us and 74us -> 65us; 3072-wide (three waves) 77us -> 80us but its
                # backward 124us -> 93us; 1536-wide (96 lanes) backward 55us -> 47us.
                tpr = fit
            elif (
                tpr > WAVE
                and not self.mix_order_reduction
                and self.rnumel % (WAVE * self.ept) == 0
                and self.rnumel <= WAVE * self.ept * ONE_STEP_MAX_RUNS
                and not any(t.startswith("welford") for t in self._reduction_types())
            ):
                # One wave per row, all of it in a single loop step: the second pass
                # re-reads the row from L2 rather than holding it, and the reduction
                # needs no LDS or barrier. 154us vs 165us on a 4096-wide bf16 RMSNorm,
                # 99us vs 112us on an amax. A welford instead pays for its looped
                # (online) form: 254us vs 147us on a LayerNorm of the same shape. Nor
                # a mix-order reduction, which needs its row held: a 32768 x 2048 RMSNorm
                # backward went to Triton's otherwise, 235us vs 184us.
                tpr = WAVE
                self.one_step = True
            self.tpr = _tune("row_tpr", tpr)
            short = (
                -(-self.rnumel // (self.tpr * self.ept)) * self.ept
                <= PERSISTENT_MAX_UNROLL
            )
            if (
                self.tpr <= WAVE
                and not self.in_combo
                and not _tune("threads", 0)
                and not (short and self._pure_reduction())
            ):
                # Rows that stay within a wave share nothing across it, and one-wave
                # blocks schedule and retire independently: 40us vs 45us on a 512-wide
                # bf16 softmax backward, as Triton's num_warps=1 there. Not a bare
                # reduction of short rows, each thread group several of them: a sum of
                # 32-wide rows 38.5us -> 35.7us in 256-thread blocks (Triton 35.5us).
                self.threads = WAVE
            elif self.threads % self.tpr:
                # Whole rows a block: the spare waves would take a part of the next.
                self.threads = self.tpr * max(1, self.threads // self.tpr)
            interleave = _tune(
                "row_stride",
                ROW_INTERLEAVE
                if self.rnumel <= ROW_INTERLEAVE_MAX and self._transposed_rows(votes)
                else 0,
            )
            if interleave and not self.in_combo and not _tune("threads", 0):
                # Rows read through a transpose (an NCHW conv output as tokens): the
                # rows of a wave interleave, `interleave` lanes apart, so its loads of
                # one element of adjacent rows are adjacent. One row a wave reads 64
                # lines for a strided load. ViT's patch embedding + LayerNorm (rows of
                # 768 f16, 196 apart) 255us -> 101us (Triton 141us), the widest vectors
                # along the row-contiguous inputs counting: 144us at 4 a lane.
                self.threads, self.stride = WAVE, interleave
                self.tpr = WAVE // interleave
                self.interleaved = True
                ept = self._r_pow2_dividing(widest_vec)
                contig = [e for e in r_roots if classify_access(e, r, 2) == "contig"]
                while ept > 1 and not all(
                    classify_access(e, r, ept) == "contig" for e in contig
                ):
                    ept //= 2
                self.ept = _tune("ept", ept)
            self.rpb = self.threads // self.tpr
        if not self.interleaved and getattr(self.features, "flydsl_transposed", False):
            raise FlyDSLUnsupported("transposed access needs a 2D tile")
        iters = -(-self.rnumel // (self.tpr * self.ept))
        self.persistent = iters * self.ept <= PERSISTENT_MAX_UNROLL
        if self.rnumel_dyn is not None and (self.persistent or self.one_step):
            raise FlyDSLUnsupported("dynamic row length laid out unrolled")
        if self.persistent:
            # Short rows: the whole row is unrolled already, so let each thread group take
            # several rows instead, while the grid stays large.
            # Normalization kernels (several passes, a full-size store) measured best
            # with one row per group: 89us vs 97us at two rows on a 64-wide softmax. A
            # bare reduction wants about 64 elements a thread, every row's loads issued
            # up front (see emit_inner): 40us vs 43us at two rows on a 32-wide sum.
            self.xunroll = (
                max(2, min(8, 64 // (iters * self.ept)))
                if self._pure_reduction()
                else 1
            )
            while (
                self.xunroll > 1
                and -(-self.xnumel // (self.rpb * self.xunroll)) < enough_blocks
            ):
                self.xunroll //= 2
            self.xunroll = _tune("xunroll", self.xunroll)
            if self.mix_order_reduction:
                # Each row group loops over its rows, so their column partials
                # accumulate in its registers: see _emit_mix_order.
                self.xunroll = 1
        else:
            # Long rows: several runs per loop iteration, for loads in flight.
            self.unroll = max(
                1,
                min(8 if gather else 4, PERSISTENT_MAX_UNROLL // self.ept, iters // 2),
            )
            if self.one_step and self.tpr == WAVE:
                self.unroll = iters
            self.unroll = _tune(
                "col_unroll" if self.stride > 1 else "row_unroll", self.unroll
            )
        if (
            self.stride > 1
            and not self.interleaved
            and not self.persistent
            and not self.in_combo
            and not atomic
            and self._x_multiple(2)
            and _tune("col_pair", 1)
        ):
            r_roots = [e for e in votes if r in e.free_symbols]
            if r_roots and all(classify_access(e, x, 2) == "contig" for e in r_roots):
                # Two adjacent columns a lane, one load for both: a bf16 column
                # reads a dword a lane, not a short. ViT's MLP bias gradient (first
                # stage) 33.8us -> 23.1us (Triton 26.8us). Settled in emit_inner,
                # which falls back to one column if the body has other control flow.
                self.cpair = 2
        self.rstep = self.tpr * self.ept * self.unroll
        if self.mix_order_reduction:
            self._finalize_mix_order()

    # -- range trees --------------------------------------------------------------------

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry):
        line = f"{entry.name} = {self.kexpr(self.rename_indexing(entry.expr))}"
        if entry.root.is_loop:
            self.indexing_code.writeline(line)
        else:
            self.body.writeline(line)
        if self._entry_is_uniform(entry):
            self.uniform_vars.add(entry.name)

    def _entry_is_uniform(self, entry: IterationRangesEntry) -> bool:
        if self.is_reduction_kernel and not entry.root.is_reduction:
            return True  # one row per thread: x entries are fixed across its r runs
        var = self._r_sym() if self.is_reduction_kernel else self._tree_sym(self.fast)
        return classify_access(entry.expr, var, self.ept) == "uniform"

    def _xmask(self) -> str | None:
        if self.tile2d:
            return None if self._dim_multiple("x", self._tile_edge("x")) else "xmask"
        if self.mix_order_reduction:
            return None if self._x_multiple(self.rsplit_size) else "xmask"
        if self.is_reduction_kernel:
            block = self.rpb * self.xunroll * self.cpair
        else:
            block = self.threads * self.ept * self.unroll
        return None if self._x_multiple(block) else "xmask"

    @property
    def is_dynamic(self) -> bool:
        return not (
            self.xnumel_dyn is None
            and self.rnumel_dyn is None
            and self.ynumel_dyn is None
        )

    def _dim_multiple(self, dim: str, n: int) -> bool:
        """Whether the ``dim`` extent is a multiple of ``n`` (a dynamic one: provably)."""
        dyn = self.xnumel_dyn if dim == "x" else self.ynumel_dyn
        if dyn is None:
            return (self.xnumel if dim == "x" else self.ynumel) % n == 0
        return _multiple(dyn, n)

    def _dim_n(self, dim: str) -> str:
        """The ``dim`` extent in the kernel: a constant, or computed from size arguments."""
        dyn = self.xnumel_dyn if dim == "x" else self.ynumel_dyn
        if dyn is None:
            return str(self.xnumel if dim == "x" else self.ynumel)
        return self.kexpr(self.rename_indexing(dyn))

    def _x_multiple(self, n: int) -> bool:
        return self._dim_multiple("x", n)

    def _x_pow2_dividing(self, cap: int) -> int:
        p = 1
        while p * 2 <= cap and self._x_multiple(p * 2):
            p *= 2
        return p

    @property
    def _xn(self) -> str:
        return self._dim_n("x")

    def _rmask(self) -> str | None:
        if not self.is_reduction_kernel or not self.inside_reduction:
            return None
        if self.rnumel_dyn is not None:
            return None if _multiple(self.rnumel_dyn, self.rstep) else "r0_mask"
        return None if self.rnumel % self.rstep == 0 else "r0_mask"

    def _r_pow2_dividing(self, cap: int) -> int:
        if self.rnumel_dyn is None:
            return _largest_pow2_dividing(self.rnumel, cap)
        p = 1
        while p * 2 <= cap and _multiple(self.rnumel_dyn, p * 2):
            p *= 2
        return p

    @property
    def _rn_loop(self) -> str:
        return f"fx.Int32({self.rnumel})" if self.rnumel_dyn is None else self._rn

    @property
    def _rn(self) -> str:
        """The row length in the kernel: a constant, or from the size arguments."""
        if self.rnumel_dyn is None:
            return str(self.rnumel)
        return self.kexpr(self.rename_indexing(self.rnumel_dyn))

    def _masks(self, *, include_r: bool = True, load_mask: bool = True) -> list[str]:
        ms = []
        if (m := self._xmask()) is not None:
            ms.append(m)
        if self.tile2d and not self._dim_multiple("y", self._tile_edge("y")):
            ms.append("ymask")
        if include_r and (m := self._rmask()) is not None:
            ms.append(m)
        if load_mask and self._load_mask is not None:
            ms.append(str(self._load_mask))
        return ms

    @staticmethod
    def _mask_arg(ms: list[str]) -> str:
        if not ms:
            return ""
        return (
            f", mask={ms[0]}" if len(ms) == 1 else f", mask=fdp.land({', '.join(ms)})"
        )

    def _vector_var(self):
        """The root index that varies across the elements one thread owns in this region."""
        if not self.is_reduction_kernel:
            return self._tree_sym(self.fast)
        if self.inside_reduction and self.ept > 1:
            return self._r_sym()
        return None

    # -- memory -------------------------------------------------------------------------

    def _rsrc(self, name: str, *, output: bool) -> str:
        arg = self.args.output(name) if output else self.args.input(name)
        return arg

    def load(self, name: str, index: sympy.Expr):
        """Emit a load of ``name`` at ``index``, choosing scalar, wave-uniform,
        vectorized or gathered access from how the index varies with the lane."""
        arg = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        st = storage_kind(dtype)
        index = self.prepare_indexing(index)
        index_str = self.kexpr(self.rename_indexing(index))
        root = self._root_expr(index)
        uniform = frozenset(self.uniform_vars)
        indirect = free_symbol_is_type(index, SymT.TMP)
        load_mask = self._load_mask
        if load_mask is not None and not indirect:
            # Under ops.masked the value is always passed through where(mask, ., other),
            # and a buffer load never faults (out of range reads 0), so a load with a
            # static index can ignore the mask -- and vectorize: 95us -> 27us on RoPE,
            # where torch.cat's halves are masked loads.
            var = self._vector_var()
            if var is not None and classify_access(
                root, var, min(self.ept, max(1, 16 // dtype.itemsize)), uniform
            ) in ("contig", "uniform"):
                load_mask = None
        dependent = indirect or load_mask is not None
        r = self._r_sym()
        hoist = (
            not dependent
            and self._in_loop_region()
            and r is not None
            and r not in root.free_symbols
        )
        if dependent:
            # The index or mask is computed: the load has to follow it.
            buffer = self.compute
        elif hoist:
            # Row-uniform: issue it once, before the reduction loop.
            buffer = self.body
        else:
            buffer = self.loads
        ms = self._masks(include_r=not hoist, load_mask=load_mask is not None)
        mask = self._mask_arg(ms)
        var = None if hoist else self._vector_var()
        kind = "other"
        w = 1
        if var is not None and load_mask is None:
            w = min(self.ept, max(1, 16 // dtype.itemsize))
            if classify_access(root, var, self.ept, uniform) == "uniform":
                kind = "uniform"
            elif w > 1 and classify_access(root, var, w, uniform) == "contig":
                kind = "contig"
        elif hoist:
            kind = "hoisted"
        if kind == "other" and self.tile2d and not dependent:
            staged = self._stage(arg, index_str, root, dtype)
            if staged is not None:
                return self.cse.generate(self.loads, staged, dtype=dtype)
        self.load_site += 1
        key = self._site_key(arg, index_str)
        # Scalar loads read whole dwords at dword-aligned offsets, and a dword reaching
        # past the descriptor reads as 0: a 5-channel bf16 scale lost its last channel.
        nbytes = self._extent(name)
        scalar_ok = (
            arg.startswith("in_ptr")
            and dtype.itemsize <= 4
            and nbytes > 0
            and nbytes % 4 == 0
        )
        if root.is_Integer and not dependent and not ms and scalar_ok:
            skey = repr(("s", arg, index_str)) if hoist else f"('s', {key})"
            line = f"fdp.memo({self._cache()}, {skey}, lambda: fdp.sload(_r_{arg}, {int(root)}, {st!r}))"
            result = self.cse.generate(buffer, line, dtype=dtype)
            self.uniform_vars.add(str(result))
            if hoist or not self._in_loop_region():
                self.outside_loop_vars.add(result)
            return result
        if kind == "contig":
            line = f"fdp.vload({self._cache()}, {key}, _j, {w}, _r_{arg}, lambda: {index_str}, {st!r}{mask})"
        elif (
            kind == "uniform"
            and not self.is_reduction_kernel
            and not self.tile2d
            and scalar_ok
            and _tune("wload", 1)
            and classify_access(root, var, WAVE * self.ept) == "uniform"
        ):
            # A wave's lanes cover an aligned WAVE * ept elements, all reading one
            # address; a masked lane's value is unused, so the mask is not needed.
            line = f"fdp.wload({self._cache()}, {key}, _j // {self.ept}, _r_{arg}, lambda: {index_str}, {st!r})"
        elif kind == "uniform":
            line = f"fdp.uload({self._cache()}, {key}, _j // {self.ept}, _r_{arg}, lambda: {index_str}, {st!r}{mask})"
        elif self.cpair > 1 and not hoist and r is not None and r in root.free_symbols:
            x = self._x_sym()
            # Only 2-byte loads: f32 pairs (dwordx2) took an 8192^2 column sum 85us ->
            # 119us.
            if (
                not indirect
                and dtype.itemsize == 2
                and nbytes % 4 == 0
                and classify_access(root, x, 2) == "contig"
            ):
                # Column `_K_` of the lane's pair: see emit_inner.
                line = (
                    f"fdp.pair_load(_K_, _r_{arg}, {index_str}, {st!r}{mask}, "
                    f"cache=_pc, key={self.load_site})"
                )
            else:
                self.cpair_failed = True
                line = f"fdp.load(_r_{arg}, {index_str}, {st!r}{mask})"
        else:
            line = f"fdp.load(_r_{arg}, {index_str}, {st!r}{mask})"
            if self._persistent_region() and not hoist and not dependent:
                line = f"fdp.memo(_lc, ({key}, _j), lambda: {line})"
        if buffer is self.loads:
            run = w if kind == "contig" else self.ept if kind == "uniform" else 1
            self.prefetch_step = math.gcd(self.prefetch_step, run)
        result = self.cse.generate(buffer, line, dtype=dtype)
        if kind in ("uniform", "hoisted") or (
            self.is_reduction_kernel and not self._in_loop_region()
        ):
            self.uniform_vars.add(str(result))
        if hoist or not self._in_loop_region():
            self.outside_loop_vars.add(result)
        return result

    def _stage(self, arg: str, index_str: str, root, dtype) -> str | None:
        """Route a load contiguous along the slow dimension through an LDS tile; returns
        the expression reading this thread's element, or None if it does not qualify."""
        slow = self._tree_sym("y" if self.fast == "x" else "x")
        w = min(8, max(1, 16 // dtype.itemsize))
        while w > 1 and classify_access(root, slow, w) != "contig":
            w //= 2
        # A staging pass's threads must fit the tile, or it loads nothing.
        while w > 1 and (self.tile_s // w) * self.tile_f < self.threads:
            w //= 2
        if w == 1 and not _linear_in(root, slow):
            return None
        ck = compute_kind(dtype)
        words = 2 if ck in ("i64", "f64") else 1
        for s in self.staged:
            if (s["arg"], s["index"]) == (arg, index_str):
                break
        else:
            plane = self.tile_f * self._tile_pitch
            base = sum(s["words"] * plane for s in self.staged)
            if base + words * plane > MAX_TILE_WORDS:
                return None
            s = {
                "arg": arg,
                "index": index_str,
                "st": storage_kind(dtype),
                "w": w,
                "base": base,
                "words": words,
            }
            self.staged.append(s)
        plane, pitch = self.tile_f * self._tile_pitch, self._tile_pitch
        return f"fdp.tile_get(_smem, {s['base']}, {plane}, _fl * {pitch} + _sl, {ck!r})"

    def _in_loop_region(self) -> bool:
        return self.is_reduction_kernel and self.inside_reduction

    def _persistent_region(self) -> bool:
        return self._in_loop_region() and self.persistent

    def _cache(self) -> str:
        return "_lc" if self._persistent_region() else "_vc"

    def _site_key(self, arg: str, index_str: str) -> str:
        """Trace-time cache key for a load.

        Keys are semantic -- buffer and index -- so two loads of the same element share a
        register. In an ``scf.for`` region the key also names the region, or a later loop
        would pick up a value that only exists inside an earlier loop's body; in an
        unrolled (persistent) region it names the iteration instead, so a later loop
        reading the same element reuses the register rather than reloading it.
        """
        if self._persistent_region():
            return f"({arg!r}, {index_str!r}, r0_offset)"
        return repr((arg, index_str, self.region))

    def store(
        self, name: str, index: sympy.Expr, value, mode: StoreMode = None
    ) -> None:
        """Emit a store, or an atomic add, of ``value`` to ``name`` at ``index``."""
        if mode not in (None, "atomic_add"):
            raise FlyDSLUnsupported(f"store mode {mode}")
        arg = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        st = storage_kind(dtype)
        if mode == "atomic_add" and st not in ("f32", "f64", "f16", "bf16"):
            raise FlyDSLUnsupported(f"atomic_add of {dtype}")
        index = self.prepare_indexing(index)
        index_str = self.kexpr(self.rename_indexing(index))
        in_loop = self._in_loop_region()
        if mode == "atomic_add":
            ms = self._masks(include_r=in_loop)
            if self.is_reduction_kernel and not in_loop and self.tpr > 1:
                # Every thread of the row group holds the row's value; one adds it.
                ms.append("_lane0")
            fn, target = (
                ("atomic_add_global", arg)
                if st == "bf16"
                else ("atomic_add", f"_r_{arg}")
            )
            line = (
                f"fdp.{fn}({target}, {index_str}, {value}, {st!r}{self._mask_arg(ms)})"
            )
            var = self._vector_var()
            if (
                st in ("f16", "bf16")
                and not self.is_reduction_kernel
                and not self.tile2d
                and self.ept % 2 == 0
                # An even extent keeps both elements of an aligned pair under one mask.
                and self._x_multiple(2)
                and var is not None
                and classify_access(
                    self._root_expr(index), var, 2, frozenset(self.uniform_vars)
                )
                == "contig"
                and _tune("atomic_pair", 1)
            ):
                self.load_site += 1
                key = repr(f"ap{self.load_site}_{arg}")
                line = (
                    f"fdp.atomic_add_pair(_sc, {key}, _j, {target}, {index_str}, "
                    f"{value}, {st!r}{self._mask_arg(ms)})"
                )
            self.stores.writeline(DeferredLine(name, line))
            self.store_buffer_names.add(name)
            self.stored_args.add(arg)
            return
        ms = self._masks(include_r=in_loop)
        merged = False
        if self.is_reduction_kernel and not in_loop and self.tpr > 1:
            merged = self.merged_rows and not ms
            # Every thread of the row group holds the value; one of them writes it.
            ms.append("_lane0")
        var = self._vector_var()
        w = min(self.ept, max(1, 16 // dtype.itemsize))
        contig = (
            var is not None
            and self._load_mask is None
            and w > 1
            and classify_access(self._root_expr(index), var, w) == "contig"
        )
        mask = self._mask_arg(ms)
        buf = self.stores
        if merged:
            line = (
                f"fdp.xstore(_xs, {arg!r}, _xr, {self.xunroll}, {self.tpr}, _rl, "
                f"_r_{arg}, {index_str}, {value}, {st!r})"
            )
        elif contig:
            seed = ""
            if self._persistent_region():
                seed = f", seed=(_lc, {self._site_key(arg, index_str)})"
            line = (
                f"fdp.vstore(_sc, {repr(f'st{self.load_site}_{arg}')}, _j, {w}, _r_{arg}, "
                f"{index_str}, {value}, {st!r}{mask}{seed})"
            )
            self.load_site += 1
        else:
            line = f"fdp.store(_r_{arg}, {index_str}, {value}, {st!r}{mask})"
        # Deferred: a buffer that turns out to be kernel-local is removed after the body
        # is written, and its store has to disappear with it.
        buf.writeline(DeferredLine(name, line))
        self.stored_args.add(arg)
        if self.is_reduction_kernel and self.persistent:
            # Loads a later unrolled loop shares through _lc must not outlive a store to
            # the same buffer. Within one loop Inductor's store cache already forwards
            # the stored value, so dropping them once the loop ends is enough.
            if in_loop:
                self.loop_stored_args[arg] = name
            else:
                buf.writeline(DeferredLine(name, f"fdp.forget(_lc, {arg!r})"))
        self.store_buffer_names.add(name)

    def check_bounds(self, expr, size, lower, upper):
        # No device assert: an out-of-range buffer access is dropped (store) or reads 0
        # (load) by the descriptor, so a bad index is a wrong value rather than a fault.
        pass

    def random(self, fn: str, seed, offset) -> str:
        """``fdp.rand``/``fdp.randn``, or their 4-wide forms -- one Philox call for four
        consecutive offsets, as Triton's rand4x -- when a thread's elements have
        consecutive offsets in aligned groups of four."""
        line = next(
            (
                k
                for k, v in self.cse._cache.items()
                if v is offset and isinstance(k, str)
            ),
            None,
        )
        root = self.index_expr_roots.get(line) if line is not None else None
        var = self._vector_var()
        if (
            root is not None
            and var is not None
            and self.ept % 4 == 0
            and classify_access(root, var, 4) == "contig"
        ):
            self.load_site += 1
            key = self._site_key(f"rng{self.load_site}", str(offset))
            return f"fdp.{fn}4({self._cache()}, {key}, _j, {seed}, {offset})"
        return _call(fn, seed, offset)

    def scan(self, dtypes, combine_fn, values):
        """One run of the row per step (``_finalize_scan``): scan it across the row's
        lanes and fold in the carry of the runs before it."""
        if not self.inside_reduction:
            raise FlyDSLUnsupported("scan outside a reduction loop")
        self.has_scan = True
        if self._load_mask:
            raise FlyDSLUnsupported("scan inside ops.masked")
        if self.features.contains_op("sort"):
            raise FlyDSLUnsupported("sort with a scan")
        if self.tpr * self.stride > self.threads or (
            self.tpr > WAVE and self.tpr % WAVE
        ):
            raise FlyDSLUnsupported("scan layout")
        dtypes = tuple(upcast_compute_type(d) for d in dtypes)
        fn = self._lift_combine(combine_fn, len(values), dtypes)
        self.acc_counter += 1
        n = len(values)
        waves = ""
        if self.tpr > WAVE:
            self.scan_components = max(self.scan_components, n)
            waves = ", smem=_smem, base=_sb"
        if self.split_chunks:
            nbytes = self.xnumel * self.split_chunks * SPLIT_SLOT_WORDS * 4
            ws, _, off = self.args.workspace(nbytes, True)
            if off % (4 * SPLIT_SLOT_WORDS):
                raise FlyDSLUnsupported("split scan workspace offset")
            slot = f"_b + {off // (4 * SPLIT_SLOT_WORDS)}"
            waves += f", split=({ws}, {slot}, _chunk)"
        carries = [f"_scan{self.acc_counter}_c{i}" for i in range(n)]
        kinds = ", ".join(repr(compute_kind(d)) for d in dtypes)
        self.body.writeline(f"{', '.join(carries)}, = fdp.scan_init({kinds})")
        results = [self.cse.newvar(dtype=d) for d in dtypes]
        lhs = ", ".join(str(v) for v in [*results, *carries])
        vals = "".join(f"{v}, " for v in values)
        cs = "".join(f"{c}, " for c in carries)
        head = f"{lhs}, = fdp.group_scan" + "_runs" * (self.ept > 1)
        args = f"{fn}, ({vals}), ({cs}), fdp.scan_first(r0_offset, _j), {self.tpr}, _rl"
        if self.ept > 1:
            # Traced in two phases: see codegen_body.
            self.scan_runs = True
            key = f"scan{self.acc_counter}"
            self.compute.writeline(
                f"{head}({args}, _sc, {key!r}, _j, {self.ept}, {SCAN_PHASE}{waves})"
            )
        else:
            self.compute.writeline(f"{head}({args}{waves})")
        return tuple(results)

    def _lift_combine(self, combine_fn, n: int, dtypes) -> str:
        """``combine_fn`` as a module-level function of two tuples, named by its body."""
        from ...ops_handler import DefaultHandler

        helper = IndentedBuffer()
        cse = CSE()
        args = [
            tuple(cse.namedvar(f"arg{i}_{k}", dtype=dtypes[k]) for k in range(n))
            for i in range(2)
        ]
        overrides = FlyDSLKernelOverrides()

        class Proxy(DefaultHandler):
            def _default(self, name, args_, kwargs):
                return cse.generate(helper, getattr(overrides, name)(*args_, **kwargs))

        with helper.indent():
            for i in range(2):
                helper.writeline(f"{''.join(f'{a}, ' for a in args[i])}= _{'ab'[i]}")
            with V.set_ops_handler(Proxy()):
                outs = combine_fn(*args)
            helper.writeline(f"return ({''.join(f'{o}, ' for o in outs)})")
        body = helper.getvalue()
        name = f"_combine_{hashlib.sha256(body.encode()).hexdigest()[:8]}"
        self.helpers[name] = f"def {name}(_a, _b):\n{body}"
        return name

    def sort(self, dtypes, values, stable, descending):
        """A row's (key, index) pairs sorted, through LDS (``_finalize_sort``)."""
        if not self.inside_reduction:
            raise FlyDSLUnsupported("sort outside a reduction loop")
        if self._load_mask:
            raise FlyDSLUnsupported("sort inside ops.masked")
        if len(values) != 2:
            raise FlyDSLUnsupported("sort of other than a (key, index) pair")
        if self.has_scan or not self.features.contains_op("sort"):
            raise FlyDSLUnsupported("sort with a scan")
        dtypes = tuple(upcast_compute_type(d) for d in dtypes)
        kinds = tuple(compute_kind(d) for d in dtypes)
        # Inductor sorts a row once for its values and once for its indices.
        cache_key = (tuple(str(v) for v in values), kinds, stable, descending)
        cache = self.__dict__.setdefault("_sorts", {})
        if cache_key in cache:
            return cache[cache_key]
        p = _pow2_ceil(self.rnumel)
        # Mirrors fdp.sort_smem_words.
        words = self.rpb * p * (1 + sum(2 if k in ("i64", "f64") else 1 for k in kinds))
        at = self.sort_words
        self.sort_words += words
        if 4 * (self.sort_words + self._scan_base) > SORT_MAX_LDS_BYTES:
            raise FlyDSLUnsupported("sort beyond LDS")
        self.has_sort = self.scan_runs = True
        self.acc_counter += 1
        results = [self.cse.newvar(dtype=d) for d in dtypes]
        vals = "".join(f"{v}, " for v in values)
        self.compute.writeline(
            f"{', '.join(str(r) for r in results)}, = fdp.row_sort(_sc, "
            f"'sort{self.acc_counter}', {SCAN_PHASE}, ({vals}), r0_index, "
            f"_t // {self.tpr}, _smem, _sb + {at}, {self.rnumel}, {p}, {self.rpb}, "
            f"{self.threads}, {kinds!r}, {bool(stable)}, {bool(descending)}, "
            f"{_tune('sort_run', SORT_REG_RUN)})"
        )
        cache[cache_key] = tuple(results)
        return cache[cache_key]

    def bucketize(
        self,
        values,
        boundaries,
        boundary_indices,
        indexing_dtype,
        right,
        sorter=None,
        sorter_indices=None,
    ):
        """See [Note: Inductor bucketize op]: the binary search unrolled, a gather per
        step, as ``triton_helpers.bucketize_binary_search``."""
        if indexing_dtype not in (torch.int32, torch.int64):
            raise FlyDSLUnsupported("bucketize indexing dtype")
        sizes = [boundaries[1], boundaries[2], boundaries[3]]
        if sorter:
            sizes.append(sorter[1])
        mask = self._mask_arg(self._masks())
        barg = self.args.input(boundaries[0])
        bst = storage_kind(V.graph.get_dtype(boundaries[0]))
        if not all(sympy.sympify(s).is_Integer for s in sizes):
            return self._bucketize_dynamic(
                values, barg, bst, boundary_indices, sizes, right, sorter,
                sorter_indices, indexing_dtype, mask,
            )
        size, numel, stride = (int(s) for s in sizes[:3])
        if (
            not sorter
            and numel == size
            and stride == 1
            and size <= LDS_TABLE_MAX
            and not self.tile2d
            and not self.mix_order_reduction
        ):
            # One table every element searches: read from LDS, staged by the block.
            self.lds_tables[barg] = (bst, size)
            return self.cse.generate(
                self.compute,
                f"fdp.bucketize_lds(_smem, _tb_{barg}, {size}, {_COMPUTE[bst]!r}, "
                f"{values}, {bool(right)}, {compute_kind(indexing_dtype)!r})",
                dtype=indexing_dtype,
            )
        args = [
            f"_r_{barg}",
            repr(bst),
            str(values),
            str(boundary_indices),
            str(size),
            str(numel),
            str(stride),
            str(bool(right)),
        ]
        if sorter:
            args += [
                f"_r_{self.args.input(sorter[0])}",
                repr(storage_kind(V.graph.get_dtype(sorter[0]))),
                str(int(sorter[1])),
                str(sorter_indices),
            ]
        else:
            args += ["None", "None", "None", "None"]
        args.append(repr(compute_kind(indexing_dtype)))
        return self.cse.generate(
            self.compute, f"fdp.bucketize({', '.join(args)}{mask})", dtype=indexing_dtype
        )

    def _bucketize_dynamic(
        self, values, barg, bst, boundary_indices, sizes, right, sorter,
        sorter_indices, indexing_dtype, mask,
    ):
        """``bucketize`` over a table sized at run time: a loop of ``bit_length(size)``
        search steps, as Triton's ``while`` over the range."""
        size, numel, stride = (
            self.kexpr(self.rename_indexing(sympy.sympify(s))) for s in sizes[:3]
        )
        srt = ["None", "None", "None", "None"]
        if sorter:
            srt = [
                f"_r_{self.args.input(sorter[0])}",
                repr(storage_kind(V.graph.get_dtype(sorter[0]))),
                self.kexpr(self.rename_indexing(sympy.sympify(sorter[1]))),
                str(sorter_indices),
            ]
        lo, hi = self.cse.newvar(dtype=torch.int64), self.cse.newvar(dtype=torch.int64)
        step_args = ", ".join(
            [f"_r_{barg}", repr(bst), str(values), str(boundary_indices), size, numel,
             stride, str(bool(right)), *srt]
        )
        self.compute.writeline(f"{lo}, {hi} = fdp.bsearch_init({size})")
        self.compute.writeline(
            f"for _bk in range(fx.Int32(0), fdp.bsearch_steps({size}), fx.Int32(1)):"
        )
        with self.compute.indent():
            self.compute.writeline(
                f"{lo}, {hi} = fdp.bsearch_step({lo}, {hi}, {step_args}{mask})"
            )
        return self.cse.generate(
            self.compute,
            f"fdp.to({lo}, {compute_kind(indexing_dtype)!r})",
            dtype=indexing_dtype,
        )

    def _finalize_mix_order(self) -> None:
        """A block runs ``rsplit_size`` rows over ``rpb`` row groups, each of its rows
        unrolled ``xunroll`` at a time with every row's loads issued first, then
        combines the groups' column partials through LDS into one workspace row."""
        split = self.rsplit_size
        if self.rnumel_dyn is not None:
            raise FlyDSLUnsupported("mix-order reduction over a dynamic row")
        # A last block's rows past xnumel are masked, their partials the identity.
        if not (
            self.persistent
            and self.stride == 1
            and not self.interleaved
            and self.tpr * split >= WAVE
        ):
            raise FlyDSLUnsupported("mix-order reduction over a looped row")
        groups = max(1, _tune("mix_threads", MIX_ORDER_THREADS) // self.tpr)
        while groups > 1 and (
            split % groups
            or groups * self._mix_cols(groups * self.tpr) > MIX_ORDER_LDS_WORDS
            or groups * self.tpr < WAVE
        ):
            groups //= 2
        if groups * self.tpr < WAVE:
            raise FlyDSLUnsupported("mix-order reduction row groups under a wave")
        self.rpb = groups
        self.threads = groups * self.tpr
        rows = split // groups
        self.xunroll = min(rows, _tune("mix_xunroll", MIX_ORDER_XUNROLL))
        while rows % self.xunroll:
            self.xunroll -= 1

    def _mix_cols(self, threads: int) -> int:
        """A partial's LDS row: the columns, padded to the block's 4-wide stores and to
        the row's loop steps."""
        span = 4 * threads
        return max(
            -(-self.rnumel // span) * span,
            -(-self.rnumel // self.rstep) * self.rstep,
        )

    def partial_accumulate(self, name, reduction_type, value, extra_meta) -> None:
        """A mix-order reduction's column partial: the row's elements, collected in
        order for _emit_mix_order to fold into the row group's accumulators."""
        if not self.mix_order_reduction or reduction_type not in ("sum", "prod"):
            raise FlyDSLUnsupported(f"partial_accumulate {reduction_type}")
        i = len(self.saved_partial_accumulate)
        self.saved_partial_accumulate.append(
            PartialAccumulate(name, reduction_type, value)
        )
        ms = self._masks()
        mask = ms[0] if len(ms) == 1 else f"fdp.land({', '.join(ms)})" if ms else None
        self.body.writeline(f"_pv{i} = []")
        self.compute.writeline(f"_pv{i}.append(({value}, {mask}))")

    # -- reductions ---------------------------------------------------------------------

    def _new_acc(self) -> str:
        self.acc_counter += 1
        return f"_acc{self.acc_counter}"

    def _post(self, line: str, n: int, dtype):
        """Emit a post-loop line that defines ``n`` new values; returns them."""
        vars_ = [self.cse.newvar(dtype=dtype) for _ in range(n)]
        lhs = ", ".join(str(v) for v in vars_)
        self.post_loop_combine.writeline(f"{lhs} = {line}")
        for v in vars_:
            self.outside_loop_vars.add(v)
            self.uniform_vars.add(str(v))
        return vars_

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value,
    ):
        """Accumulate per thread in the loop body (chunked where the combine allows),
        then combine across the row's threads after the loop."""
        if not self.inside_reduction:
            raise FlyDSLUnsupported("reduction outside a reduction loop")
        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.reduction_cache:
            return self.reduction_cache[cache_key]
        rmask = self._rmask()
        mask = f", mask={rmask}" if rmask else ""
        tpr = self.tpr
        cross_wave = tpr * self.stride > WAVE
        smem = "_smem" if cross_wave else "None"
        self.uses_smem |= cross_wave
        geo = f"{tpr}, {smem}, {self.threads}, {self.stride}"

        item = f"({value}, {rmask or 'None'})"
        if reduction_type == "sum":
            # A chunk at a time, as a balanced tree (fdp.sum_block).
            k = compute_kind(dtype)
            acc = self._new_acc()
            self.body.writeline(f"{acc} = fdp.default('sum', {k!r})")
            self.chunk_pre.writeline(f"{acc}_items = []")
            self.compute.writeline(f"{acc}_items.append({item})")
            self.chunk_post.writeline(f"{acc} = fdp.sum_block({acc}, {acc}_items)")
            (result,) = self._post(f"fdp.group_reduce('sum', {acc}, {geo})", 1, dtype)
        elif reduction_type in ("prod", "max", "min", "fmax", "any", "xor_sum"):
            if reduction_type == "fmax" and not dtype.is_floating_point:
                raise FlyDSLUnsupported("fmax on integers")
            k = compute_kind(dtype)
            acc = self._new_acc()
            self.body.writeline(f"{acc} = fdp.default({reduction_type!r}, {k!r})")
            self.compute.writeline(
                f"{acc} = fdp.combine({reduction_type!r}, {acc}, {value}{mask})"
            )
            (result,) = self._post(
                f"fdp.group_reduce({reduction_type!r}, {acc}, {geo})", 1, dtype
            )
        elif reduction_type.startswith(("argmax", "argmin")):
            # argmax (the index), argmax_value (the value) and argmax_with_value (both:
            # torch.max(x, dim)), the value in its accumulator type as Triton's.
            logical_index = None
            if isinstance(value, tuple):
                value, logical_index = value
            op = "max" if reduction_type.startswith("argmax") else "min"
            k = compute_kind(src_dtype)
            acc = self._new_acc()
            self.body.writeline(f"{acc}_v, {acc}_i = fdp.arg_default({op!r}, {k!r})")
            index_var = str(logical_index) if logical_index is not None else "r0_index"
            self.compute.writeline(
                f"{acc}_v, {acc}_i = fdp.arg_combine({op!r}, {acc}_v, {acc}_i, {value}, {index_var}{mask})"
            )
            half = src_dtype in (torch.float16, torch.bfloat16)
            val = self.cse.newvar(dtype=torch.float32 if half else src_dtype)
            idx = self.cse.newvar(dtype=torch.int64)
            self.post_loop_combine.writeline(
                f"{val}, {idx} = fdp.group_arg_reduce({op!r}, {acc}_v, {acc}_i, {geo})"
            )
            for v in (val, idx):
                self.outside_loop_vars.add(v)
                self.uniform_vars.add(str(v))
            result = (
                (val, idx)
                if reduction_type.endswith("_with_value")
                else val
                if reduction_type.endswith("_value")
                else idx
            )
            self.max_components = max(self.max_components, 2)
        elif reduction_type == "welford_reduce" and self.persistent:
            acc = self._new_acc()
            self.body.writeline(f"{acc}_items = []")
            self.compute.writeline(f"{acc}_items.append({item})")
            result = tuple(
                self._post(
                    # Every element of the row counts, as in Triton's persistent
                    # welford, which divides by rnumel.
                    f"fdp.group_welford_twopass({acc}_items, {geo}, count={self.rnumel})",
                    3,
                    torch.float32,
                )
            )
            self.max_components = max(self.max_components, 2)
        elif reduction_type in ("welford_reduce", "welford_combine"):
            acc = self._new_acc()
            self.body.writeline(f"{acc}_m, {acc}_s, {acc}_w = fdp.welford_default()")
            if reduction_type == "welford_reduce":
                self.compute.writeline(
                    f"{acc}_m, {acc}_s, {acc}_w = fdp.welford_update({acc}_m, {acc}_s, {acc}_w, {value}{mask})"
                )
            else:
                m, s, w = value
                self.compute.writeline(
                    f"{acc}_m, {acc}_s, {acc}_w = fdp.welford_merge(({acc}_m, {acc}_s, {acc}_w), "
                    f"({m}, {s}, {w}){mask})"
                )
            result = tuple(
                self._post(
                    f"fdp.group_welford({acc}_m, {acc}_s, {acc}_w, {geo})",
                    3,
                    torch.float32,
                )
            )
            self.max_components = max(self.max_components, 3)
        elif reduction_type == "online_softmax_reduce":
            if isinstance(value, tuple):
                raise FlyDSLUnsupported("online_softmax_reduce of a tuple")
            acc = self._new_acc()
            if self.persistent:
                # The whole row is in registers: max pass, then exp-sum pass.
                self.body.writeline(f"{acc}_items = []")
                self.compute.writeline(f"{acc}_items.append({item})")
                result = tuple(
                    self._post(
                        f"fdp.group_softmax_twopass({acc}_items, {geo})",
                        2,
                        torch.float32,
                    )
                )
            else:
                # Online across loop steps, one rescale per chunk within a step.
                self.body.writeline(f"{acc}_m, {acc}_s = fdp.online_softmax_default()")
                self.chunk_pre.writeline(f"{acc}_items = []")
                self.compute.writeline(f"{acc}_items.append({item})")
                self.chunk_post.writeline(
                    f"{acc}_m, {acc}_s = fdp.online_softmax_block({acc}_m, {acc}_s, {acc}_items)"
                )
                result = tuple(
                    self._post(
                        f"fdp.group_online_softmax({acc}_m, {acc}_s, {geo})",
                        2,
                        torch.float32,
                    )
                )
                self.max_components = max(self.max_components, 2)
        else:
            raise FlyDSLUnsupported(f"reduction {reduction_type}")
        self.reduction_cache[cache_key] = result
        return result

    def store_reduction(self, name: str, index: sympy.Expr, value) -> None:
        # Written after the loop with the row-uniform index; `store` sees
        # inside_reduction=False and adds the lane-0 mask.
        self.inside_reduction = False
        try:
            prior, self.stores = self.stores, self.post_loop_combine
            self.store(name, index, value)
        finally:
            self.stores = prior
            self.inside_reduction = True

    # -- body ---------------------------------------------------------------------------

    @staticmethod
    def _scan_inputs(lines: list[Any]) -> list[Any]:
        """The compute lines a multi-element scan's first traced pass needs: the scan
        calls and what their inputs derive from. Anything else -- an accumulator
        update above all -- would run twice."""
        ident = re.compile(r"\b[A-Za-z_]\w*\b")
        defines = re.compile(r"^\s*([\w\s,]+?)\s*=(?!=)")
        keep: list[Any] = []
        needed: OrderedSet[str] = OrderedSet()
        for li in reversed(lines):
            text = str(li)
            m = defines.match(text)
            outs = OrderedSet(ident.findall(m.group(1))) if m else OrderedSet()
            if SCAN_PHASE not in text and not (outs & needed):
                continue
            rhs = text[m.end() :] if m else text
            needed |= OrderedSet(ident.findall(rhs))
            keep.append(li)
        return keep[::-1]

    @staticmethod
    def _scan_depths(lines: list[Any]) -> dict[int, int]:
        """Each multi-element scan line's nesting (by ``id``): 1 for a scan of no other
        scan's result, one more than the deepest scan its inputs derive from else."""
        ident = re.compile(r"\b[A-Za-z_]\w*\b")
        defines = re.compile(r"^\s*([\w\s,]+?)\s*=(?!=)")
        depth_of: dict[str, int] = {}
        out: dict[int, int] = {}
        for li in lines:
            text = str(li)
            m = defines.match(text)
            if not m:
                continue
            rhs = text[m.end() :]
            d = max((depth_of.get(v, 0) for v in ident.findall(rhs)), default=0)
            if SCAN_PHASE in text:
                d += 1
                out[id(li)] = d
            for v in ident.findall(m.group(1)):
                if not v.startswith("_scan"):
                    depth_of[v] = d
        return out

    def _queue_preload(self) -> None:
        """Issue this unrolled pass's loads ahead of the first pass.

        Every pass reads through the ``_lc`` memo, so loading early only moves the load:
        the pass finds it cached. A later pass's loads (a LayerNorm's weight and bias,
        say) then need not wait out the reduction before them: 135us vs 151us on a
        4096-wide bf16 LayerNorm. Skipped for
        buffers already stored to, and for a pass whose indexing uses names the body
        only defines after the first pass.
        """
        if self._preload_at is None:
            self._preload_at = len(self.body.get_lines_ref())
            self._preload_indent = self.body._indent
        later = "\n".join(
            str(li) for li in self.body.get_lines_ref()[self._preload_at :]
        )
        defined = OrderedSet(re.findall(r"^\s*(\w+)\s*=", later, re.MULTILINE))
        stored = [f"_r_{a}" for a in self.stored_args]
        loads = [
            li
            for li in self.loads.get_lines_ref()
            if isinstance(li, str) and not any(s in li for s in stored)
        ]
        if not loads:
            return
        pre = IndentedBuffer(initial_indent=self._preload_indent)
        pre.writeline(
            f"for r0_offset in range_constexpr(0, {self._r_extent}, {self.rstep}):"
        )
        with pre.indent():
            pre.writeline(f"for _j in range_constexpr({self.ept * self.unroll}):")
            with pre.indent():
                pre.writeline(
                    f"r0_index = {self._r_base}r0_offset + (_j // {self.ept}) * {self.tpr * self.ept} "
                    f"+ _rl * {self.ept} + _j % {self.ept}"
                )
                if self._rmask():
                    pre.writeline(f"r0_mask = r0_index < {self._rn}")
                pre.splice(self.indexing_code)
                for li in loads:
                    pre.writeline(li.strip())
        text = pre.getvalue()
        own = OrderedSet(re.findall(r"^\s*(\w+)\s*=", text, re.MULTILINE))
        used = OrderedSet(re.findall(r"\b[A-Za-z_]\w*\b", text))
        if (used - own) & defined:
            return
        self.preload_lines.extend(pre.get_lines_ref())

    def _insert_widen(self) -> None:
        """After the first pass, convert the preloaded f16/bf16 loads the same for every
        row (a norm's weight and bias) to f32 (see fdp.widen), as long as the f32
        copies stay few: they are live across the reduction. Not a row's own inputs:
        the converted copies of those only add pressure (Qwen's q/k RMSNorm + RoPE
        67us -> 79us)."""
        if not self.preload_lines or self._widen_at is None:
            return
        text = "\n".join(str(li) for li in self.preload_lines)
        keys = [
            (a, i)
            for a, i in OrderedSet(
                re.findall(r"fdp\.vload\(_lc, \('(\w+)', '([^']*)'", text)
            )
            if not re.search(r"\bx|\btmp", i)
        ]
        per_thread = -(-self.rnumel // (self.tpr * self.ept)) * self.ept
        if not keys or len(keys) * per_thread > _tune("widen_max", WIDEN_MAX_ELEMENTS):
            return
        line = IndentedBuffer(initial_indent=self._preload_indent)
        line.writeline(f"fdp.widen(_lc, {tuple(keys)!r})")
        lines = self.body.get_lines_ref()
        lines[self._widen_at : self._widen_at] = line.get_lines_ref()
        self._widen_at = None

    def _splice_preload(self) -> None:
        if self.preload_lines and self._preload_at is not None:
            lines = self.body.get_lines_ref()
            lines[self._preload_at : self._preload_at] = self.preload_lines
            self.preload_lines = []

    def codegen_body(self):
        if not (
            self.indexing_code
            or self.loads
            or self.stores
            or self.compute
            or self.post_loop_combine
        ):
            return
        if self.is_reduction_kernel and self.inside_reduction:
            if self.persistent:
                self._queue_preload()
                self.body.writeline(
                    f"for r0_offset in range_constexpr(0, {self._r_extent}, {self.rstep}):"
                )
            else:
                # FlyDSL loops count in i32; a wide kernel's index values are i64.
                self.body.writeline(
                    f"for _ro in range(fx.Int32(0), {self._rn_loop}, fx.Int32({self.rstep})):"
                )
                with self.body.indent():
                    self.body.writeline("r0_offset = fdp.idx(_ro)")
            with self.body.indent():
                self.body.splice(self.chunk_pre)
                # A multi-element scan first traces the step for its inputs alone: see
                # fdp.group_scan_runs.
                lines = self.compute.get_lines_ref()
                depths = self._scan_depths(lines) if self.scan_runs else {}
                # A scan of depth d records its inputs in pass d - 1 and has its
                # results from pass d on: one pass more than the deepest.
                last = max(depths.values(), default=0)
                for phase in range(last + 1) if self.scan_runs else (None,):
                    self.body.writeline(
                        f"for _j in range_constexpr({self.ept * self.unroll}):"
                    )
                    with self.body.indent():
                        # Run `_j // ept` of this step: runs are tpr*ept apart, so each
                        # vector instruction covers one contiguous span across the row's
                        # threads.
                        self.body.writeline(
                            f"r0_index = {self._r_base}r0_offset + (_j // {self.ept}) * {self.tpr * self.ept} "
                            f"+ _rl * {self.ept} + _j % {self.ept}"
                        )
                        if self._rmask():
                            self.body.writeline(f"r0_mask = r0_index < {self._rn}")
                        self.body.splice(self.indexing_code)
                        self.body.splice(self.loads)
                        if phase is None:
                            self.body.splice(self.compute)
                            self.body.splice(self.stores)
                        else:
                            part = lines if phase == last else self._scan_inputs(lines)
                            traced = IndentedBuffer()
                            traced.get_lines_ref().extend(
                                li.replace(
                                    SCAN_PHASE, str(phase - depths.get(id(li), 1) + 1)
                                )
                                if isinstance(li, str)
                                else li
                                for li in part
                            )
                            self.body.splice(traced)
                            if phase == last:
                                self.body.splice(self.stores)
                self.body.splice(self.chunk_post)
            for arg, name in self.loop_stored_args.items():
                self.body.writeline(DeferredLine(name, f"fdp.forget(_lc, {arg!r})"))
            if self.persistent and self._widen_at is None:
                self._widen_at = len(self.body.get_lines_ref())
            self.loop_stored_args.clear()
            self.cse.invalidate(self.outside_loop_vars)
            self.range_trees[-1].cache_clear()
            self.__dict__.pop("_sorts", None)
            self.region += 1
        else:
            # The body so far holds the pointwise index entries.
            self.prefetch.splice(self.body)
            self.prefetch.splice(self.indexing_code)
            self.prefetch.splice(self.loads)
            self.body.splice(self.indexing_code)
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)
        self.body.splice(self.post_loop_combine)
        self.indexing_code.clear()
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.post_loop_combine.clear()
        self.chunk_pre.clear()
        self.chunk_post.clear()

    # -- module -------------------------------------------------------------------------

    @staticmethod
    def _extent(name: str) -> int:
        """Byte extent of buffer ``name``, or 0 when unknown."""
        node = V.graph.try_get_buffer(name)
        if node is None:
            return 0
        n = _int_or_none(V.graph.sizevars.simplify(node.get_layout().storage_size()))
        return 0 if n is None else n * node.get_dtype().itemsize

    def _arg_extents(self, arg_defs) -> dict[str, int | str]:
        """Exact byte extent of each tensor argument, for its buffer descriptor: an int,
        or for a dynamic one the kernel expression computing it from the size
        arguments."""
        name_of = {inner: outer for outer, inner in self.args.input_buffers.items()}
        for outer, inner in self.args.output_buffers.items():
            if not isinstance(inner, str) or inner.startswith("REMOVED"):
                continue
            name_of[inner] = outer
        for outer, inner in self.args.inplace_buffers.items():
            if hasattr(inner, "inner_name"):
                name_of[inner.inner_name] = outer
        extents = {}
        workspaces = {ws.inner_name: ws for ws in self.args.workspace_args}
        sizes = OrderedSet(self.args.sizevars.values())
        for a in arg_defs:
            if a in sizes:
                continue
            if a in workspaces:
                ws = workspaces[a]
                count = V.graph.sizevars.simplify(ws.count)
                n = _int_or_none(count)
                if n is None:
                    nbytes = count * ws.dtype.itemsize
                    if not expr_fits_within_32bit(nbytes):
                        raise FlyDSLUnsupported("dynamic workspace not known to fit")
                    V.graph.sizevars.check_leq(nbytes, _I32_LIMIT - 1)
                    # Unsimplified: simplify turns its FloorDiv into a rational floor.
                    extents[a] = self.kexpr(
                        self.rename_indexing(ws.count * ws.dtype.itemsize)
                    )
                    continue
                extents[a] = max(1, n) * ws.dtype.itemsize
                continue
            buf = name_of.get(a)
            if buf is None:
                raise FlyDSLUnsupported(f"argument {a} without a buffer")
            node = V.graph.try_get_buffer(buf)
            if node is None:
                raise FlyDSLUnsupported(f"no layout for {buf}")
            layout = node.get_layout()
            size = V.graph.sizevars.simplify(layout.storage_size())
            n = _int_or_none(size)
            if n is None:
                nbytes = size * node.get_dtype().itemsize
                # The descriptor's extent is 32-bit: guarded as Triton's i32 indexing.
                if not expr_fits_within_32bit(nbytes):
                    raise FlyDSLUnsupported("dynamic buffer extent not known to fit")
                V.graph.sizevars.check_leq(nbytes, _I32_LIMIT - 1)
                extents[a] = self.kexpr(self.rename_indexing(nbytes))
                continue
            if n > _I32_LIMIT:
                if self.is_dynamic:
                    raise FlyDSLUnsupported("buffer of 2**31 elements or more")
                self.wide = True
            extents[a] = max(1, n) * node.get_dtype().itemsize
        return extents

    @property
    def stage(self) -> str:
        """The kernel's class, for ``TORCHINDUCTOR_FLYDSL_SIMD_STAGES``."""
        if self.tile2d:
            return "tile"
        if not self.is_reduction_kernel:
            return "pointwise"
        if self.has_scan:
            return "scan"
        if self.stride > 1 and not self.interleaved:
            return "column"
        return "row" if self._pure_reduction() else "norm"

    @property
    def merged_rows(self) -> bool:
        """Whether a row group's ``xunroll`` results are stored together (fdp.xstore),
        every lane writing one, instead of one store a row from lane 0: short rows
        otherwise take as many partial store instructions as loads -- 42us vs 37us on
        a 32-wide sum (Triton 35us), 52us vs 38us at 8 lanes a row."""
        return bool(
            self.is_reduction_kernel
            and self.persistent
            and self.stride == 1
            and self.tpr > 1
            and self.xunroll > 1
            and self._pure_reduction()
            and not self._xmask()
            and _tune("merge_rows", 1)
        )

    @property
    def slp(self) -> bool:
        """Whether LLVM's SLP vectorizer runs. It pairs a reduction's f32 math into
        ``v_pk_*`` ops whose operands need a ``v_mov`` each to line up -- 44us vs 49us
        on a 512-wide bf16 softmax backward, and 5-10% on LayerNorm backward and
        RMSNorm -- while pointwise kernels gain from it overall (gelu: 0.79x vs 0.94x
        of Triton)."""
        return bool(_tune("slp", int(not self.is_reduction_kernel or self.tile2d)))

    @property
    def smem_words(self) -> int:
        n = self.scan_components
        if self.sort_words:
            return self._scan_base + self.sort_words
        if self.split_chunks:
            # Mirrors fdp.split_smem_words.
            return self._scan_base + 2 * (self.threads // WAVE) * n + 2 * n
        return self._scan_base + 2 * (self.threads // WAVE) * n

    @property
    def _scan_base(self) -> int:
        # Wave totals of a scan wider than a wave: after the body and the tables.
        return self._smem_body_words + sum(
            2 * size for _, size in self.lds_tables.values()
        )

    @property
    def _smem_body_words(self) -> int:
        if self.staged:
            return sum(s["words"] * self.tile_f * self._tile_pitch for s in self.staged)
        # Mirrors fdp.smem_words: a slot per thread per 32-bit word of each component.
        words = self.threads * 2 * self.max_components if self.uses_smem else 0
        if self.mix_order_reduction and self.rpb > 1:
            words += self.rpb * self._mix_cols(self.threads)
        return words

    def _emit_lds_tables(self, code: IndentedBuffer) -> None:
        """Each table into LDS by the whole block, then one barrier: ``_tb_<arg>`` its
        first word, a 64-bit element's second word ``size`` further on."""
        base = self._smem_body_words
        for arg, (st, size) in self.lds_tables.items():
            code.writeline(f"_tb_{arg} = {base}")
            for k in range(-(-size // self.threads)):
                i = f"fdp.imin(_t + {k * self.threads}, {size - 1})"
                code.writeline(
                    f"fdp.tile_put(_smem, _tb_{arg}, {size}, {i}, fdp.load(_r_{arg}, {i}, {st!r}))"
                )
            base += 2 * size
        if self.lds_tables:
            code.writeline("fdp.barrier()")

    @property
    def num_blocks(self) -> int:
        if self.tile2d:
            nf, ns = self._tile_counts()
            return nf * ns
        if not self.is_reduction_kernel:
            return -(-self.xnumel // (self.threads * self.ept * self.unroll))
        if self.mix_order_reduction:
            return -(-self.xnumel // self.rsplit_size)
        if self.split_chunks:
            return self.xnumel * self.split_chunks
        pairs = 2 if self._pair_lines is not None else 1
        return -(-self.xnumel // (self.rpb * self.xunroll * pairs))

    def _blocks_expr(self) -> sympy.Expr:
        """``num_blocks`` as an expression in the (renamed) size arguments."""

        def cdiv(n, d: int) -> sympy.Expr:
            return FloorDiv(n + d - 1, d)

        if self.tile2d:
            if self.xnumel_dyn is None and self.ynumel_dyn is None:
                return sympy.Integer(self.num_blocks)
            ns = {
                d: dyn if dyn is not None else sympy.Integer(n)
                for d, dyn, n in (
                    ("x", self.xnumel_dyn, self.xnumel),
                    ("y", self.ynumel_dyn, self.ynumel),
                )
            }
            return self.rename_indexing(
                cdiv(ns["x"], self._tile_edge("x")) * cdiv(ns["y"], self._tile_edge("y"))
            )
        if self.xnumel_dyn is None:
            return sympy.Integer(self.num_blocks)
        if self.mix_order_reduction:
            span = self.rsplit_size
        elif self.is_reduction_kernel:
            pairs = 2 if self._pair_lines is not None else 1
            span = self.rpb * self.xunroll * pairs
        else:
            span = self.threads * self.ept * self.unroll
        return self.rename_indexing(cdiv(self.xnumel_dyn, span))

    def _grid(self) -> int | str:
        """``num_blocks``, or for a dynamic extent the expression ``_main`` computes it
        by from its size arguments."""
        blocks = self._blocks_expr()
        if blocks.is_Integer:
            return int(blocks)
        return f"max(1, {PythonPrinter().doprint(blocks)})"

    def arg_names(self) -> list[str]:
        arg_defs, _, _, _ = self.args.python_argdefs()
        return [a.full_name() for a in arg_defs]

    def size_args(self) -> OrderedSet[str]:
        return OrderedSet(self.args.sizevars.values())

    def codegen_kernel(self, name=None):
        """The module source: the ``@flyc.kernel``, its ``@flyc.jit`` launcher, and the
        ``_main`` / ``_precompile`` entry points the wrapper calls."""
        self.codegen_body()
        self._pair_lines = self._column_pairs()
        # Extents and the grid may name sizes the body does not: register them first.
        grid = self._grid()
        extents = self._arg_extents(self.arg_names())
        arg_names = self.arg_names()
        extents = self._arg_extents(arg_names)
        # Emitted first: an index constant past 2**31 printed there makes it wide.
        inner = IndentedBuffer()
        inner.writeline("_b = fdp.bid()")
        self.emit_inner(inner)
        code = _module_prologue(
            arg_names,
            extents,
            self.smem_words,
            [self._describe()],
            self.helpers,
            self.threads,
            self.size_args(),
            self.wide,
        )
        with code.indent():
            code.splice(inner.getvalue())
        code.splice(
            _module_epilogue(
                arg_names, grid, self.threads, self.slp, self.size_args()
            )
        )
        return code.getvalue()

    def _column_pairs(self) -> IndentedBuffer | None:
        """A column reduction's body for two adjacent columns a lane (``cpair``), or
        None for one.

        Each column gets its own copy of the body's statements, names suffixed
        ``_k0``/``_k1`` (accumulators are loop-carried by name), sharing the row loops:
        within a step column 0's ``pair_load`` issues the load both columns read.
        Bodies with any other control flow keep one column a lane."""
        lines = self.body.get_lines_ref()

        def text_of(li) -> str:
            return li.line if isinstance(li, DeferredLineBase) else li

        def unpaired() -> None:
            for i, li in enumerate(lines):
                text = text_of(li)
                if not isinstance(text, str) or "_K_" not in text:
                    continue
                text = re.sub(r", cache=_pc, key=\d+\)", ")", text)
                text = text.replace("fdp.pair_load(_K_, ", "fdp.load(")
                lines[i] = (
                    li._new_line(text) if isinstance(li, DeferredLineBase) else text
                )

        if self.cpair < 2:
            return None
        if self.cpair_failed or self.xunroll > 1 or self.preload_lines:
            unpaired()
            return None
        lane_x = f"(_t % {self.stride}) * 2 + _K_"
        template: list[tuple[int, Any]] = [
            (0, line) for line in ("_vc = {}", "_sc = {}", "_lc = {}")
        ]
        template.append((0, f"xindex = _b * {self.rpb * 2} + {lane_x}"))
        if self._xmask():
            template.append((0, f"xmask = xindex < {self._xn}"))
        body = [
            li
            for li in lines
            if not isinstance(text_of(li), str) or text_of(li).strip()
        ]
        if any(not isinstance(text_of(li), str) for li in body):
            unpaired()
            return None
        indents = [len(text_of(li)) - len(text_of(li).lstrip()) for li in body]
        base = min(indents, default=0)
        template.extend(((i - base) // 4, li) for i, li in zip(indents, body))
        shared = ("for r0_offset in ", "for _ro in ", "for _j in range_constexpr(")

        # Parse into statements and shared loops by indentation.
        def parse(i: int, d: int):
            nodes = []
            while i < len(template):
                di, li = template[i]
                if di < d:
                    break
                stmt = text_of(li).strip()
                if stmt.endswith(":"):
                    if not stmt.startswith(shared):
                        return None, i
                    kids, i = parse(i + 1, d + 1)
                    if kids is None:
                        return None, i
                    nodes.append((li, kids))
                    continue
                nodes.append(li)
                i += 1
            return nodes, i

        tree, _ = parse(0, 0)
        if tree is None:
            unpaired()
            return None
        # Every target, a tuple's too: an arg reduction's value and index (or a
        # welford's three) shared by both columns took the pair's combined result.
        assigned = OrderedSet(
            name
            for _, li in template
            for m in [re.match(r"\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*=(?!=)", text_of(li))]
            if m
            for name in re.split(r"\s*,\s*", m.group(1))
            if name not in ("r0_offset", "_j", "_pc")
        )
        names = re.compile(r"\b(" + "|".join(map(re.escape, assigned)) + r")\b")

        def copy(li, k: int):
            text = names.sub(lambda m: f"{m.group(1)}_k{k}", text_of(li).strip())
            text = text.replace("_K_", str(k))
            return li._new_line(text) if isinstance(li, DeferredLineBase) else text

        out = IndentedBuffer()

        def emit(nodes) -> None:
            run: list[Any] = []

            def flush() -> None:
                for k in (0, 1):
                    for li in run:
                        out.writeline(copy(li, k))
                run.clear()

            for node in nodes:
                if not isinstance(node, tuple):
                    run.append(node)
                    continue
                flush()
                head, kids = node
                out.writeline(text_of(head).strip())
                with out.indent():
                    if text_of(head).strip().startswith("for _j"):
                        out.writeline("_pc = {}")
                    emit(kids)
            flush()

        emit(tree)
        return out

    def _pass_range(self, part: IndentedBuffer) -> str:
        """The elements a pointwise or tile pass traces: all of a thread's for the body,
        only those that can issue a load for the prefetch -- the rest of a 64-element
        RoPE variant's prefetch traced index math for nothing (0.76 s to compile)."""
        n = self.ept * self.unroll
        step = math.gcd(self.prefetch_step, self.ept) if part is self.prefetch else 1
        return f"range_constexpr(0, {n}, {step})" if step > 1 else f"range_constexpr({n})"

    def emit_inner(self, code: IndentedBuffer) -> None:
        """The thread layout and body, given ``_t``, ``_b``, the descriptors and ``_smem``."""
        self._emit_lds_tables(code)
        if self.scan_components or self.sort_words:
            code.writeline(f"_sb = {self._scan_base}")
        if self.tile2d:
            self._emit_tile2d(code)
        elif not self.is_reduction_kernel:
            code.writeline("_vc = {}")
            code.writeline("_sc = {}")
            ept, span = self.ept, self.threads * self.ept
            # With several runs a thread, issue every run's loads (into the trace-time
            # vector cache) before the first store: the backend cannot prove a store
            # leaves a later run's input alone, so it would otherwise wait out each
            # store before the next load -- 141us vs 95us on an in-place bias add.
            passes = (
                [self.prefetch, self.body]
                if self.unroll > 1 and self.prefetch
                else [self.body]
            )
            for part in passes:
                code.writeline(f"for _j in {self._pass_range(part)}:")
                with code.indent():
                    code.writeline(
                        f"xindex = _b * {span * self.unroll} + (_j // {ept}) * {span} + _t * {ept} + _j % {ept}"
                    )
                    if self._xmask():
                        code.writeline(f"xmask = xindex < {self._xn}")
                    code.splice(part)
        else:
            if self.stride == 1:
                code.writeline(f"_rl = _t % {self.tpr}")
                lane_x = f"_t // {self.tpr}"
            else:
                code.writeline(f"_rl = _t // {self.stride}")
                lane_x = f"_t % {self.stride}"
            code.writeline("_lane0 = _rl == 0")
            if self.mix_order_reduction:
                self._emit_mix_order(code, lane_x)
                return
            if self._pair_lines is not None:
                code.splice(self._pair_lines)
                return
            if self.merged_rows:
                code.writeline("_xs = {}")

            def row_start() -> None:
                if self.split_chunks:
                    code.writeline(f"_chunk = _b % {self.split_chunks}")
                    code.writeline(f"_rc0 = fdp.idx(_chunk) * {self.rstep}")
                    code.writeline(f"xindex = _b // {self.split_chunks}")
                else:
                    code.writeline(
                        f"xindex = _b * {self.rpb * self.xunroll} + _xr * {self.rpb} + {lane_x}"
                    )
                if self._xmask():
                    code.writeline(f"xmask = xindex < {self._xn}")

            self._insert_widen()
            if self.xunroll > 1 and self.preload_lines:
                # Every row's preloads first, into per-row trace-time caches: a row's
                # stores would otherwise hold back the next row's loads, which the
                # backend cannot prove independent -- one row's loads in flight at a
                # time where Triton issues all of its rows'.
                code.writeline("_caches = []")
                code.writeline(f"for _xr in range_constexpr({self.xunroll}):")
                with code.indent():
                    code.writeline("_vc, _sc, _lc = {}, {}, {}")
                    row_start()
                    pre = IndentedBuffer()
                    pre.get_lines_ref().extend(
                        li
                        for li in self.body.get_lines_ref()[: self._preload_at]
                        if isinstance(li, str)
                    )
                    pre.get_lines_ref().extend(self.preload_lines)
                    code.splice(pre)
                    code.writeline("_caches.append((_vc, _sc, _lc))")
                self.preload_lines = []
                code.writeline(f"for _xr in range_constexpr({self.xunroll}):")
                with code.indent():
                    code.writeline("_vc, _sc, _lc = _caches[_xr]")
                    row_start()
                    code.splice(self.body)
                return
            code.writeline(f"for _xr in range_constexpr({self.xunroll}):")
            with code.indent():
                # Trace-time load caches are per row.
                code.writeline("_vc = {}")
                code.writeline("_sc = {}")
                code.writeline("_lc = {}")
                row_start()
                self._splice_preload()
                code.splice(self.body)

    def _emit_mix_order(self, code: IndentedBuffer, lane_x: str) -> None:
        """A mix-order reduction (Inductor's MixOrderReduction): each row group runs
        ``rsplit_size`` rows one after another, the row reduction as usual, while
        each lane sums the column values it saw into one register per element it
        owns. The group then writes those partials as row ``_b * rpb + group`` of the
        workspace, which the wrapper sums over its rows."""
        split, ept = self.rsplit_size, self.ept
        steps = -(-self.rnumel // self.rstep)
        per_step = ept * self.unroll
        slots = steps * per_step
        accs = self.saved_partial_accumulate
        for i, pa in enumerate(accs):
            for s in range(slots):
                code.writeline(
                    f"_pa{i}_{s} = fdp.default({pa.reduction_type!r}, 'f32')"
                )
        groups, xr = self.rpb, self.xunroll
        iters = split // groups // xr
        xc = "_xc" if iters > 1 else "0"

        def row_start() -> None:
            code.writeline(
                f"xindex = _b * {split} + ({xc} * {xr} + _xr) * {groups} + {lane_x}"
            )
            if self._xmask() is not None:
                code.writeline(f"xmask = xindex < {self._xn}")

        def row_end() -> None:
            for i, pa in enumerate(accs):
                for s in range(slots):
                    code.writeline(
                        f"_pa{i}_{s} = fdp.combine({pa.reduction_type!r}, _pa{i}_{s}, *_pv{i}[{s}])"
                    )

        loop = contextlib.ExitStack()
        if iters > 1:
            code.writeline(
                f"for _xci in range(fx.Int32(0), fx.Int32({iters}), fx.Int32(1)):"
            )
            loop.enter_context(code.indent())
            code.writeline("_xc = fdp.idx(_xci)")
        self._insert_widen()
        if xr > 1 and self.preload_lines:
            code.writeline("_caches = []")
            code.writeline(f"for _xr in range_constexpr({xr}):")
            with code.indent():
                code.writeline("_vc, _sc, _lc = {}, {}, {}")
                row_start()
                pre = IndentedBuffer()
                pre.get_lines_ref().extend(
                    li
                    for li in self.body.get_lines_ref()[: self._preload_at]
                    if isinstance(li, str)
                )
                pre.get_lines_ref().extend(self.preload_lines)
                code.splice(pre)
                code.writeline("_caches.append((_vc, _sc, _lc))")
            self.preload_lines = []
            code.writeline(f"for _xr in range_constexpr({xr}):")
            with code.indent():
                code.writeline("_vc, _sc, _lc = _caches[_xr]")
                row_start()
                code.splice(self.body)
                row_end()
        else:
            code.writeline(f"for _xr in range_constexpr({xr}):")
            with code.indent():
                code.writeline("_vc, _sc, _lc = {}, {}, {}")
                row_start()
                self._splice_preload()
                code.splice(self.body)
                row_end()
        loop.close()

        (ws,) = self.args.workspace_args
        rnumel = self.rnumel
        if self.xnumel_dyn is None:
            partials = str(self.num_blocks)
        else:
            split = self.rsplit_size
            partials = self.kexpr(
                self.rename_indexing(FloorDiv(self.xnumel_dyn + split - 1, split))
            )
        code.writeline("_ws = {}")

        def ws_base(i: int) -> str:
            return f"{i} * {partials} * {rnumel} + " if i else ""

        def col_of(step: int, j: int) -> str:
            return f"{step * self.rstep + (j // ept) * self.tpr * ept + j % ept} + _rl * {ept}"

        if groups == 1:
            w = min(ept, 4)
            for i in range(len(accs)):
                for step in range(steps):
                    for j in range(per_step):
                        col = col_of(step, j)
                        mask = (
                            f", mask=({col}) < {rnumel}" if rnumel % self.rstep else ""
                        )
                        code.writeline(
                            f"fdp.vstore(_ws, 'pa{i}_{step}_{j // w}', {j % w}, {w}, "
                            f"_r_{ws.inner_name}, {ws_base(i)}_b * {rnumel} + {col}, "
                            f"_pa{i}_{step * per_step + j}, 'f32'{mask})"
                        )
            return
        # The groups' partials through LDS, a row each, then summed down the groups
        # by every thread, four columns a thread.
        base = self.threads * 2 * self.max_components if self.uses_smem else 0
        cols = self._mix_cols(self.threads)
        span = 4 * self.threads
        w = 4 if rnumel % 4 == 0 else 1
        for i, pa in enumerate(accs):
            if i:
                code.writeline("fdp.barrier()")
            for step in range(steps):
                for j in range(per_step):
                    code.writeline(
                        f"fdp.tile_put(_smem, {base}, 0, {lane_x} * {cols} + {col_of(step, j)}, "
                        f"_pa{i}_{step * per_step + j})"
                    )
            code.writeline("fdp.barrier()")
            for c in range(-(-rnumel // span)):
                for e in range(4):
                    col = f"{c * span + e} + _t * 4"
                    code.writeline(f"_v = fdp.tile_get(_smem, {base}, 0, {col}, 'f32')")
                    for g in range(1, groups):
                        code.writeline(
                            f"_v = fdp.combine({pa.reduction_type!r}, _v, "
                            f"fdp.tile_get(_smem, {base}, 0, {g * cols} + {col}, 'f32'))"
                        )
                    mask = f", mask=({col}) < {rnumel}" if rnumel % span else ""
                    code.writeline(
                        f"fdp.vstore(_ws, 'f{i}_{c}_{e // w}', {e % w}, {w}, "
                        f"_r_{ws.inner_name}, {ws_base(i)}_b * {rnumel} + {col}, "
                        f"_v, 'f32'{mask})"
                    )

    def _emit_tile2d(self, code: IndentedBuffer) -> None:
        f = self.fast
        s = "y" if f == "x" else "x"
        tf, ts, pitch = self.tile_f, self.tile_s, self._tile_pitch
        fdyn = self.xnumel_dyn if f == "x" else self.ynumel_dyn
        if fdyn is None:
            nf = str(self._tile_counts()[0])
        else:
            code.writeline(
                f"_nf = {self.kexpr(self.rename_indexing(FloorDiv(fdyn + tf - 1, tf)))}"
            )
            nf = "_nf"
        code.writeline(f"_bf = _b % {nf}")
        code.writeline(f"_bs = _b // {nf}")
        masks = [
            f"{d}mask = {d}index < {self._dim_n(d)}"
            for d in ("x", "y")
            if not self._dim_multiple(d, self._tile_edge(d))
        ]
        entries = [
            f"{e.name} = {self.kexpr(self.rename_indexing(e.expr))}"
            for e in self.range_tree_nodes.values()
        ]
        mask_arg = self._mask_arg([m.split(" ")[0] for m in masks])

        def coords(fl: str, sl: str) -> None:
            code.writeline(f"_fl = {fl}")
            code.writeline(f"_sl = {sl}")
            code.writeline(f"{f}index = _bf * {tf} + _fl")
            code.writeline(f"{s}index = _bs * {ts} + _sl")
            for m in masks:
                code.writeline(m)

        # Staging: threads run along the slow dimension, so each staged load is a
        # coalesced vector read; the tile keeps (fast, slow) coordinates.
        for i, st in enumerate(self.staged):
            w = st["w"]
            tps = ts // w
            rps = self.threads // tps
            code.writeline(f"_vs{i} = {{}}")
            code.writeline(f"for _j in range_constexpr({w * (tf // rps)}):")
            with code.indent():
                coords(
                    f"_t // {tps} + (_j // {w}) * {rps}",
                    f"(_t % {tps}) * {w} + _j % {w}",
                )
                for e in entries:
                    code.writeline(e)
                if w == 1:
                    code.writeline(
                        f"_v = fdp.load(_r_{st['arg']}, {st['index']}, {st['st']!r}{mask_arg})"
                    )
                else:
                    code.writeline(
                        f"_v = fdp.vload(_vs{i}, 'stage', _j, {w}, _r_{st['arg']}, "
                        f"{st['index']}, {st['st']!r}{mask_arg})"
                    )
                code.writeline(
                    f"fdp.tile_put(_smem, {st['base']}, {tf * pitch}, "
                    f"_fl * {pitch} + _sl, _v)"
                )
        if self.staged:
            code.writeline("fdp.barrier()")

        code.writeline("_vc = {}")
        code.writeline("_sc = {}")
        ept, tpf = self.ept, tf // self.ept
        passes = [self.prefetch, self.body] if self.prefetch else [self.body]
        for part in passes:
            code.writeline(f"for _j in {self._pass_range(part)}:")
            with code.indent():
                coords(
                    f"(_t % {tpf}) * {ept} + _j % {ept}",
                    f"_t // {tpf} + (_j // {ept}) * {self.rpb}",
                )
                code.splice(part)

    def call_kernel(self, name: str, node=None, deallocate_ws: bool = True):
        wrapper = V.graph.wrapper_code
        _, call_args, _, arg_types = self.args.python_argdefs()
        for ws in self.args.workspace_args:
            wrapper.generate_workspace_allocation(ws)
        wrapper.generate_kernel_call(name, call_args, triton=True, arg_types=arg_types)
        if deallocate_ws:
            self.deallocate_workspaces()


# ---------------------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------------------


def _module_prologue(
    arg_names: list[str],
    extents: dict[str, int],
    smem_words: int,
    notes: list[str],
    helpers: dict[str, str],
    threads: int = THREADS,
    sizes: OrderedSet[str] = OrderedSet(),
    wide: bool = False,
) -> IndentedBuffer:
    """Imports, the LDS struct, and the ``@flyc.kernel`` header up to ``_t``; the caller
    writes the rest of the body at the returned buffer's next indent."""
    code = IndentedBuffer()
    code.splice(
        """
        import flydsl.compiler as flyc
        import flydsl.expr as fx
        from flydsl.expr import range_constexpr
        from torch._inductor.codegen.flydsl import flydsl_simd_runtime as fdp
        """
    )
    if smem_words:
        code.splice(
            f"""
            @fx.struct
            class _Smem:
                words: fx.Array[fx.Int32, {smem_words}, 16]
            """
        )
    for src in helpers.values():
        code.writeline("")
        code.splice(src)
    sig = ", ".join(f"{a}: fx.Int64" for a in arg_names)
    code.writeline("")
    # Undecorated: _module_epilogue's {name}_flydsl() applies flyc.kernel on first use.
    code.writeline(f"def {Placeholder.KERNEL_NAME}_kernel({sig}):")
    with code.indent():
        # FlyDSL's binary cache keys on this function's source and on helpers in the
        # same directory -- not on the runtime modules it calls -- so stamp them in.
        code.writeline(f"# runtime {_runtime_fingerprint()}")
        for note in notes:
            code.writeline(f"# {note}")
        code.writeline(f"fdp.kernel_begin({'wide=True' if wide else ''})")
        for a in arg_names:
            if a in sizes:
                code.writeline(f"{a} = fdp.size_arg({a})")
        for a in arg_names:
            if a not in sizes:
                code.writeline(f"_r_{a} = fdp.rsrc({a}, {extents[a]})")
        if smem_words:
            code.writeline(
                f"_smem = fx.SharedAllocator().allocate(_Smem).peek().words.view(fx.make_layout({smem_words}, 1))"
            )
        code.writeline("_t = fdp.tid()")
    return code


def _module_epilogue(
    arg_names: list[str],
    grid: int | str,
    threads: int,
    slp: bool,
    sizes: OrderedSet[str] = OrderedSet(),
) -> str:
    kname = str(Placeholder.KERNEL_NAME)
    # Contracting a multiply and add into an FMA, as Triton's default enable_fp_fusion:
    # LayerNorm over 16384 x 4096 bf16 118us -> 112us.
    hint: dict[str, Any] = {}
    if _tune("fma", 1):
        hint["fastmath"] = "contract"
    if not slp:
        hint["llvm_options"] = {"vectorize-slp": False}
    hints = f"\n                {kname}_launch.compile_hints = {hint!r}" if hint else ""
    decorator = (
        f"flyc.kernel(known_block_size=[{threads}, 1, 1])"
        if threads > THREADS
        else "flyc.kernel"
    )
    launch_sig = ", ".join([f"{a}: fx.Int64" for a in arg_names] + ["_grid: fx.Int32"])
    return f"""

        def {kname}_launch_fn({launch_sig}, stream: fx.Stream = fx.Stream(None)):
            {kname}_kernel({", ".join(arg_names)}).launch(grid=(_grid, 1, 1), block=({threads}, 1, 1), stream=stream)


        {kname}_launch = None


        def {kname}_flydsl():
            # Decorating rewrites the kernel's AST, ~15 ms a kernel: only the compile
            # worker, or a process without the saved code object, needs it.
            global {kname}_kernel, {kname}_launch
            if {kname}_launch is None:
                {kname}_kernel = {decorator}({kname}_kernel)
                {kname}_launch = flyc.jit({kname}_launch_fn){hints}
            return {kname}_launch


        {kname}_compiled = None
        {kname}_binary = __file__.rsplit(".", 1)[0] + ".{kname}.hsaco"


        def {kname}_main({", ".join(arg_names)}, stream=None):
            global {kname}_compiled
            args = ({"".join(f"{a}, " if a in sizes else f"{a}.data_ptr(), " for a in arg_names)}{grid}, 0 if stream is None else stream)
            if {kname}_compiled is None:
                {kname}_compiled = fdp.load_binary(
                    {kname}_binary, "{kname}_kernel_0", {threads}, {len(arg_names)}
                )
                if {kname}_compiled is None:
                    # flyc.compile launches once while compiling, so this call is done.
                    {kname}_compiled = flyc.compile({kname}_flydsl(), *args)
                    return
            {kname}_compiled(*args)


        def {kname}_precompile(**_):
            # Compiles into FlyDSL's disk cache, launching nothing; the code object is
            # kept beside this module for _main.
            if fdp.have_binary({kname}_binary):
                return
            launch = {kname}_flydsl()
            fdp.compile_only(launch, {"0, " * len(arg_names)}0, 0, arch={_gpu_arch()!r})
            fdp.save_binary(launch, {kname}_binary, "{kname}_kernel_0")
        """


def _mix_order_finalize_source(
    nsplit: int, rnumel: int, outs: list[tuple[str, torch.dtype]]
) -> str:
    """A mix-order reduction's second stage: each of ``outs`` (reduction type,
    storage) reduces its ``nsplit`` x ``rnumel`` f32 slice of the workspace over its
    rows. A one-wave block takes four columns a lane over its column lanes, the
    ``slices`` lanes of each splitting the rows, then combined by shuffles. ATen's
    ``sum(dim=0)`` and cast took 19.7us on 512 x 1024 partials."""
    slices = _tune("mix_fin_slices", MIX_FINALIZE_SLICES)
    lanes = WAVE // slices
    cols = 4 * lanes
    steps = -(-nsplit // slices)
    unroll = min(steps, _tune("mix_fin_unroll", MIX_FINALIZE_UNROLL))
    iters = -(-steps // unroll)
    w = 4 if rnumel % 4 == 0 else 1
    args = ["ws_ptr", *(f"out_ptr{i}" for i in range(len(outs)))]
    extents = {"ws_ptr": len(outs) * nsplit * rnumel * 4}
    for i, (_, dtype) in enumerate(outs):
        extents[f"out_ptr{i}"] = rnumel * dtype.itemsize
    code = _module_prologue(
        args,
        extents,
        0,
        [f"mix-order finalize nsplit={nsplit} rnumel={rnumel}"],
        {},
        WAVE,
    )
    with code.indent():
        code.writeline("_b = fdp.bid()")
        code.writeline(f"_c = _b * {cols} + (_t // {slices}) * 4")
        code.writeline(f"_s = _t % {slices}")
        for i, (rt, dtype) in enumerate(outs):
            for e in range(4):
                code.writeline(f"_a{e} = fdp.default({rt!r}, 'f32')")
            loop = contextlib.ExitStack()
            k = "0"
            if iters > 1:
                code.writeline(
                    f"for _k in range(fx.Int32(0), fx.Int32({iters}), fx.Int32(1)):"
                )
                loop.enter_context(code.indent())
                k = "_k"
            code.writeline("_lc = {}")
            for u in range(unroll):
                row = f"({k} * {unroll} + {u}) * {slices} + _s"
                ms = []
                if iters * unroll * slices != nsplit:
                    ms.append(f"({row}) < {nsplit}")
                if rnumel % cols:
                    ms.append(f"_c < {rnumel}")
                for e in range(4):
                    m = ms + ([f"_c + {e} < {rnumel}"] if rnumel % 4 else [])
                    mask = (
                        "None"
                        if not m
                        else m[0]
                        if len(m) == 1
                        else f"fdp.land({', '.join(m)})"
                    )
                    off = f"{i * nsplit * rnumel} + ({row}) * {rnumel} + _c + {e}"
                    load = (
                        f"fdp.vload(_lc, ('ws', {u}), {e}, 4, _r_ws_ptr, {off}, 'f32', {mask})"
                        if w == 4
                        else f"fdp.load(_r_ws_ptr, {off}, 'f32', {mask})"
                    )
                    code.writeline(
                        f"_a{e} = fdp.combine({rt!r}, _a{e}, {load}, {mask})"
                    )
            loop.close()
            code.writeline("_os = {}")
            for e in range(4):
                code.writeline(
                    f"_v = fdp.group_reduce({rt!r}, _a{e}, {slices}, None, {WAVE})"
                )
                m = ["_s == 0"]
                if rnumel % cols or rnumel % 4:
                    m.append(f"_c + {e} < {rnumel}")
                mask = m[0] if len(m) == 1 else f"fdp.land({', '.join(m)})"
                code.writeline(
                    f"fdp.vstore(_os, 'o', {e % w}, {w}, _r_out_ptr{i}, _c + {e}, _v, {_STORAGE[dtype]!r}, {mask})"
                )
    code.splice(_module_epilogue(args, -(-rnumel // cols), WAVE, False))
    return code.getvalue()


_PTR_RE = re.compile(r"(?<![A-Za-z0-9])(_r_)?((?:in_out_ptr|in_ptr|out_ptr)\d+)\b")


def _merge_identical(
    kernels, inners, flat: OrderedSet[str] = OrderedSet()
) -> list[tuple[list[int], str, list[tuple[str, ...]]]]:
    """Group a combo's sub-kernels whose bodies differ only in their buffers, as a
    foreach's per-parameter kernels do (6 distinct of GPT's 22 AdamW sub-kernels).
    Each group is ``(each member's blocks, body, members)``: the body names its j-th
    buffer ``_g{j}`` (descriptor ``_r__g{j}``), a member lists its buffers in that
    order. One traced body a group: GPT's largest AdamW kernel compiles in 2.4 s
    rather than 4.7 s, and runs faster for its smaller code. Members of different
    sizes share a body too when it does not depend on the size (Llama's 1024^2 and
    256 x 1024 weights)."""
    groups: dict[tuple, tuple[list[int], str, list[tuple[str, ...]]]] = {}
    for k, inner in zip(kernels, inners):
        names: list[str] = []

        def rename(m):
            if m.group(2) not in names:
                names.append(m.group(2))
            return f"{m.group(1) or ''}_g{names.index(m.group(2))}"

        body = _PTR_RE.sub(rename, inner)
        # A buffer of 2 GiB or more is a raw address rather than a descriptor (see
        # fdp._Flat): members switch between buffers of one kind.
        kinds = tuple(n in flat for n in names)
        key = (kinds, body) if _tune("combo_merge", 1) else (k.num_blocks, inner)
        if key not in groups:
            groups[key] = ([], body, [])
        groups[key][0].append(k.num_blocks)
        groups[key][2].append(tuple(names))
    out = []
    for blocks, body, members in groups.values():
        if len(members) == 1:
            # Unchanged, under its own buffer names.
            names = members[0]
            body = re.sub(
                r"(?<![A-Za-z0-9])(_r_)?_g(\d+)\b",
                lambda m: f"{m.group(1) or ''}{names[int(m.group(2))]}",
                body,
            )
        out.append((blocks, body, members))
    return out


def _combo_source_dynamic(kernels: list[FlyDSLKernel]) -> str:
    """``_combo_source`` with sizes known at run time: each kernel a branch of its own
    (no merged bodies), its block range ending where the size arguments put it."""
    inners, blocks = [], []
    for k in kernels:
        with V.set_kernel_handler(k):
            k.codegen_body()
            buf = IndentedBuffer()
            k.emit_inner(buf)
            inners.append(buf.getvalue())
            blocks.append(k._blocks_expr())
    k0 = kernels[0]
    arg_names = k0.arg_names()
    extents = k0._arg_extents(arg_names)
    code = _module_prologue(
        arg_names,
        extents,
        max(k.smem_words for k in kernels),
        [k._describe() for k in kernels],
        {n: s for k in kernels for n, s in k.helpers.items()},
        k0.threads,
        k0.size_args(),
        wide=any(k.wide for k in kernels),
    )
    ends = list(itertools.accumulate(blocks))
    with code.indent():
        code.writeline("_bid = fdp.bid()")
        for i, inner in enumerate(inners):
            if i == len(inners) - 1:
                code.writeline("else:")
            else:
                end = k0.kexpr(ends[i])
                code.writeline(f"{'if' if i == 0 else 'elif'} _bid < {end}:")
            with code.indent():
                if i:
                    code.writeline(f"_b = _bid - ({k0.kexpr(ends[i - 1])})")
                else:
                    code.writeline("_b = _bid")
                code.splice(inner)
    grid = f"max(1, {PythonPrinter().doprint(ends[-1])})"
    code.splice(
        _module_epilogue(
            arg_names, grid, k0.threads, all(k.slp for k in kernels), k0.size_args()
        )
    )
    return code.getvalue()


def _combo_source(kernels: list[FlyDSLKernel]) -> str:
    """One launch for several kernels sharing a KernelArgs: each owns a contiguous range
    of block ids and runs, with ``_b`` rebased, under a branch on the block id."""
    if any(k.is_dynamic for k in kernels):
        return _combo_source_dynamic(kernels)
    inners = []
    for k in kernels:
        with V.set_kernel_handler(k):
            k.codegen_body()
            buf = IndentedBuffer()
            k.emit_inner(buf)
            # Deferred lines resolve against the current kernel's removed buffers.
            inners.append(buf.getvalue())
    arg_names = kernels[0].arg_names()
    extents = kernels[0]._arg_extents(arg_names)
    smem = max(k.smem_words for k in kernels)
    helpers = {n: s for k in kernels for n, s in k.helpers.items()}
    code = _module_prologue(
        arg_names,
        extents,
        smem,
        [k._describe() for k in kernels],
        helpers,
        kernels[0].threads,
        wide=any(k.wide for k in kernels),
    )
    groups = _merge_identical(
        kernels, inners, OrderedSet(a for a, n in extents.items() if n >= _I32_LIMIT)
    )
    start = 0
    with code.indent():
        code.writeline("_bid = fdp.bid()")
        if len(groups) == 1 and len(groups[0][2]) == 1:
            code.writeline("_b = _bid")
            code.splice(inners[0])
            start = kernels[0].num_blocks
        for i, (blocks, inner, members) in enumerate(
            groups if len(groups) > 1 or len(groups[0][2]) > 1 else []
        ):
            ends = list(itertools.accumulate(blocks))
            end = start + ends[-1]
            if len(groups) == 1:
                pass
            elif i == len(groups) - 1:
                code.writeline("else:")
            else:
                code.writeline(f"{'if' if i == 0 else 'elif'} _bid < {end}:")
            with code.indent() if len(groups) > 1 else contextlib.nullcontext():
                if len(members) == 1:
                    code.writeline(f"_b = _bid - {start}" if start else "_b = _bid")
                else:
                    even = len(OrderedSet(blocks)) == 1
                    if even:
                        code.writeline(f"_s = fdp.idiv(_bid - {start}, {blocks[0]})")
                        code.writeline(f"_b = _bid - {start} - _s * {blocks[0]}")
                    else:
                        code.writeline(
                            f"_s = fdp.bucket(_bid - {start}, {tuple(ends[:-1])})"
                        )
                    picks = []
                    for j in range(len(members[0])):
                        if f"_r__g{j}" in inner:
                            picks.append(
                                (
                                    f"_r__g{j}",
                                    lambda n: f"fdp.rsrc({n}, {extents[n]})",
                                    j,
                                )
                            )
                        if re.search(rf"(?<!['_])\b_g{j}\b(?!')", inner):
                            picks.append((f"_g{j}", lambda n: n, j))
                    tuples = ", ".join(
                        "lambda: ("
                        + "".join(f"{ref(m[j])}, " for _, ref, j in picks)
                        + ("" if even else f"fdp.idx({first}), ")
                        + ")"
                        for m, first in zip(members, [0, *ends[:-1]])
                    )
                    targets = "".join(f"{t}, " for t, _, _ in picks)
                    if not even:
                        targets += "_first, "
                    code.writeline(f"{targets}= fdp.pick(_s, {tuples})")
                    if not even:
                        code.writeline(f"_b = _bid - {start} - _first")
                code.splice(inner)
            start = end
    code.splice(
        _module_epilogue(
            arg_names, start, kernels[0].threads, all(k.slp for k in kernels)
        )
    )
    return code.getvalue()


def _autotune_source(
    srcs: list[str], arg_names: list[str], warm: tuple[int, ...] = ()
) -> str:
    """One module holding each candidate's kernel (``_c0`` the heuristic's) and a
    ``_main`` that times them on the first call and keeps the fastest."""
    kname = str(Placeholder.KERNEL_NAME)
    names = [f"{kname}_c{i}" for i in range(len(srcs))]
    parts = [s.replace(kname, n) for s, n in zip(srcs, names)]
    args = ", ".join(arg_names)
    mutated = tuple(i for i, a in enumerate(arg_names) if a.startswith("in_out_ptr"))
    precompiles = "".join(
        f"\n            if only in (None, {i}):\n                {n}_precompile(**kw)"
        for i, n in enumerate(names)
    )
    parts.append(
        f"""
        {kname}_best = None


        def {kname}_main({args}, stream=None):
            global {kname}_best
            if {kname}_best is None:
                {kname}_best = fdp.pick_fastest(
                    [{", ".join(f"{n}_main" for n in names)}], ({args},), {mutated!r}, stream,
                    warm={warm!r},
                )
                return
            {kname}_best({args}, stream=stream)


        def {kname}_precompile(only=None, **kw):{precompiles}
        """
    )
    code = IndentedBuffer()
    for p in parts:
        code.splice(p)
    return code.getvalue()


def _dispatch_source(generic: str, aligned: str, arg_names: list[str], sizes) -> str:
    """One module holding a dynamic kernel built for any sizes (``_c0``) and for sizes
    all multiples of SIZE_ALIGN (``_c1``), and a ``_main`` choosing by the sizes."""
    kname = str(Placeholder.KERNEL_NAME)
    names = [f"{kname}_c0", f"{kname}_c1"]
    parts = [s.replace(kname, n) for s, n in zip((generic, aligned), names)]
    args = ", ".join(arg_names)
    cond = " and ".join(f"{s} % {SIZE_ALIGN} == 0" for s in sizes)
    parts.append(
        f"""
        def {kname}_main({args}, stream=None):
            if {cond}:
                {names[1]}_main({args}, stream=stream)
            else:
                {names[0]}_main({args}, stream=stream)


        def {kname}_precompile(only=None, **kw):
            if only in (None, 0):
                {names[0]}_precompile(**kw)
            if only in (None, 1):
                {names[1]}_precompile(**kw)
        """
    )
    code = IndentedBuffer()
    for p in parts:
        code.splice(p)
    return code.getvalue()


class _DropFallbackErrors(logging.Filter):
    """The scheduler logs every exception out of a node body as fatal before re-raising;
    a FlyDSLUnsupported is the expected signal to use Triton, not an error."""

    def filter(self, record: logging.LogRecord) -> bool:
        import sys

        exc = sys.exc_info()[1]
        return not (
            record.getMessage().startswith("Error in codegen")
            and isinstance(exc, FlyDSLUnsupported)
        )


@contextlib.contextmanager
def _quiet_scheduler_codegen_errors():
    sched_log = logging.getLogger("torch._inductor.scheduler")
    f = _DropFallbackErrors()
    sched_log.addFilter(f)
    try:
        yield
    finally:
        sched_log.removeFilter(f)


def flydsl_simd_counters():
    from torch._dynamo.utils import counters

    return counters["flydsl_simd"]


class FlyDSLSIMDScheduling(TritonScheduling):
    """TritonScheduling whose pointwise/reduction kernels are FlyDSL where possible."""

    def __init__(self, scheduler) -> None:
        super().__init__(scheduler)
        # Before fusion rather than at the first kernel: ViT inference's first step
        # waited 3.8 s on workers still importing FlyDSL.
        _warm_compile_workers()

    def codegen_node_schedule(self, kernel_features: SIMDKernelFeatures):
        reason = self._precheck(kernel_features)
        if reason is None:
            try:
                return self._codegen_flydsl(kernel_features)
            except FlyDSLUnsupported as e:
                reason = str(e)
        flydsl_simd_counters()[f"fallback: {reason}"] += 1
        flydsl_simd_counters()["triton_kernels"] += 1
        if config.flydsl.simd_strict:
            raise FlyDSLUnsupported(f"FlyDSL SIMD kernel refused ({reason})")
        log.debug("FlyDSL SIMD fallback to Triton: %s", reason)
        return super().codegen_node_schedule(kernel_features)

    def codegen_combo_kernel(self, combo_kernel_node):
        """Foreach (and horizontally fused) kernels: one FlyDSL launch per group of
        sub-kernels, each owning a range of block ids."""
        reason = self._precheck(None)
        if reason is None:
            try:
                with _quiet_scheduler_codegen_errors():
                    self._codegen_flydsl_combo(combo_kernel_node)
                return
            except FlyDSLUnsupported as e:
                reason = str(e)
        flydsl_simd_counters()[f"fallback: combo: {reason}"] += 1
        flydsl_simd_counters()["triton_kernels"] += 1
        if config.flydsl.simd_strict:
            raise FlyDSLUnsupported(f"FlyDSL SIMD kernel refused ({reason})")
        return super().codegen_combo_kernel(combo_kernel_node)

    def codegen_mix_order_reduction(self, node):
        # A fixed split: autotuning it would benchmark the kernel as Triton's.
        with config.patch(
            {
                "triton.mix_order_reduction_split_size": config.triton.mix_order_reduction_split_size
                or self._mix_order_split(node.node1)
            }
        ):
            return super().codegen_mix_order_reduction(node)

    @staticmethod
    def _mix_order_split(node1) -> int:
        """Rows a row group runs: Triton's choice, about eight splits a CU of 16 to
        128 rows."""
        from ...scheduler import MixOrderReduction

        numel, _ = MixOrderReduction.get_numel_rnumel(node1)
        n = V.graph.sizevars.optimization_hint(numel, fallback=0)
        if _tune("mix_split", 0):
            return _tune("mix_split", 0)
        split = 16
        while split < 128 and n // (split * 2) >= 8 * _num_cus():
            split *= 2
        return split

    def _generate_kernel_code_for_mix_order_reduction(
        self, kernel_features, split_size, for_benchmark
    ):
        reason = self._precheck(kernel_features)
        if reason is None and not for_benchmark:
            count_before = metrics.generated_kernel_count
            try:
                with _quiet_scheduler_codegen_errors():
                    kernel = FlyDSLKernel(
                        {
                            "x": kernel_features.numel,
                            "r0_": kernel_features.reduction_numel,
                        },
                        features=kernel_features,
                        mix_order_reduction=True,
                    )
                    kernel.rsplit_size = split_size
                    kernel.mix_node_schedule = kernel_features.node_schedule
                    self.codegen_node_schedule_with_kernel(
                        kernel_features.node_schedule, kernel
                    )
                    partials = (
                        kernel_features.numel + split_size - 1
                    ) // split_size
                    _, ws_name, _ = kernel.args.workspace(
                        len(kernel.saved_partial_accumulate) * partials * kernel.rnumel,
                        False,
                        dtype=torch.float,
                    )
                    with V.set_kernel_handler(kernel):
                        src_code = kernel.codegen_kernel()
                    if not _stage_enabled(kernel.stage):
                        raise FlyDSLUnsupported(f"stage {kernel.stage} disabled")
                flydsl_simd_counters()["flydsl_kernels"] += 1
                flydsl_simd_counters()["mix_order"] += 1
                return kernel, ws_name, src_code
            except FlyDSLUnsupported as e:
                reason = str(e)
            except (NotImplementedError, TypeError, KeyError, AssertionError) as e:
                reason = f"{type(e).__name__}: {e}"
            metrics.generated_kernel_count = count_before
        if not for_benchmark:
            flydsl_simd_counters()[f"fallback: mix-order: {reason}"] += 1
            flydsl_simd_counters()["triton_kernels"] += 1
            if config.flydsl.simd_strict:
                raise FlyDSLUnsupported(f"FlyDSL SIMD kernel refused ({reason})")
        return super()._generate_kernel_code_for_mix_order_reduction(
            kernel_features, split_size, for_benchmark
        )

    def _codegen_mix_order_finalize(self, kernel, ws_name, nsplit, rnumel):
        n, r = _int_or_none(nsplit), _int_or_none(rnumel)
        bufs = [
            V.graph.get_buffer(pa.buffer_name) for pa in kernel.saved_partial_accumulate
        ]
        if not (
            isinstance(kernel, FlyDSLKernel)
            and n
            and r
            and all(
                b is not None
                and b.get_layout().is_contiguous()
                and _int_or_none(b.get_numel()) == r
                and b.get_dtype() in (torch.float32, torch.bfloat16, torch.float16)
                for b in bufs
            )
        ):
            return super()._codegen_mix_order_finalize(kernel, ws_name, nsplit, rnumel)
        src = _mix_order_finalize_source(
            n,
            r,
            [
                (pa.reduction_type, b.get_dtype())
                for pa, b in zip(kernel.saved_partial_accumulate, bufs)
            ],
        )
        name = self._define_flydsl_kernel(src, kernel.mix_node_schedule)
        wrapper = V.graph.wrapper_code
        names = [pa.buffer_name for pa in kernel.saved_partial_accumulate]
        for b in bufs:
            wrapper.writeline(wrapper.make_buffer_allocation(b))
        wrapper.generate_kernel_call(
            name,
            [ws_name, *names],
            triton=True,
            arg_types=[torch.float32, *(b.get_dtype() for b in bufs)],
        )
        wrapper.allocated.update(names)
        flydsl_simd_counters()["flydsl_kernels"] += 1

    def define_kernel(self, src_code, node_schedule, kernel):
        if isinstance(kernel, FlyDSLKernel):
            return self._define_flydsl_kernel(src_code, node_schedule)
        return super().define_kernel(src_code, node_schedule, kernel)

    def _combo_groups(self, subkernel_nodes) -> list[list[Any]]:
        """Split the sub-kernels so no launch takes more buffers than Triton's combo
        kernels may (config.combo_kernel_max_num_args)."""
        groups: list[list[Any]] = []
        names: OrderedSet[str] = OrderedSet()
        for pn in subkernel_nodes:
            rw = pn.read_writes
            mine = OrderedSet(d.name for d in (*rw.reads, *rw.writes))
            if groups and len(names | mine) <= config.combo_kernel_max_num_args:
                groups[-1].append(pn)
                names |= mine
            else:
                groups.append([pn])
                names = mine
        return groups

    def _codegen_flydsl_combo(self, combo_kernel_node) -> None:
        from ..common import KernelArgs
        from ..simd_kernel_features import SIMDKernelFeatures

        if not _stage_enabled("combo"):
            raise FlyDSLUnsupported("stage combo disabled")
        count_before = metrics.generated_kernel_count
        built = []
        try:
            for group in self._combo_groups(combo_kernel_node.get_subkernel_nodes()):
                args = KernelArgs()
                kernels, schedules = [], []
                for pn in group:
                    nodes = pn.get_nodes()
                    _, (numel, rnumel) = max(
                        nodes, key=lambda n: int(n.is_reduction())
                    ).group
                    node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
                    features = SIMDKernelFeatures(node_schedule, numel, rnumel)
                    kernel = FlyDSLKernel(
                        self._choose_tiling(features), features=features
                    )
                    kernel.args = args
                    kernel.in_combo = True  # one block size for the whole launch
                    self.codegen_node_schedule_with_kernel(node_schedule, kernel)
                    kernels.append(kernel)
                    schedules.append(node_schedule)
                built.append((kernels, schedules, _combo_source(kernels)))
        except FlyDSLUnsupported:
            metrics.generated_kernel_count = count_before
            raise
        except (NotImplementedError, TypeError, KeyError, AssertionError) as e:
            metrics.generated_kernel_count = count_before
            raise FlyDSLUnsupported(f"{type(e).__name__}: {e}") from e

        # One kernel per launch, as Triton's ComboKernel counts it.
        # pyrefly: ignore [bad-assignment]
        metrics.generated_kernel_count = count_before + len(built)
        for kernels, schedules, src_code in built:
            node_schedule = [n for s in schedules for n in s]
            kernel_name = self._define_flydsl_kernel(src_code, node_schedule)
            lead = kernels[0]
            lead.kernel_name = kernel_name
            flydsl_simd_counters()["flydsl_kernels"] += 1
            flydsl_simd_counters()["combo"] += 1
            for k in kernels:
                with V.set_kernel_handler(k):
                    for node in k.features.scheduler_nodes():
                        node.mark_run()
                V.graph.removed_buffers |= k.removed_buffers
                V.graph.inplaced_to_remove |= k.inplaced_to_remove
            self.codegen_comment(combo_kernel_node.snodes, kernel_name)
            lead.call_kernel(kernel_name)
        self.free_buffers_in_scheduler()

    def _choose_tiling(self, kernel_features) -> dict[str, sympy.Expr]:
        flat = self._flat_tiling(kernel_features)
        if _tune("force_flat"):
            return flat
        # Triton's tiling analysis adds a y (or z) dimension exactly when some access
        # is transposed relative to the others. A flat kernel then coalesces the
        # stores and strides the loads (or the reverse), which measured 8x slower on
        # a 4096^2 transpose; a pointwise kernel takes the {y, x} tiling instead and
        # stages the transposed loads through LDS.
        preferred, _ = self.get_tiling_and_scores(
            kernel_features.node_schedule,
            kernel_features.numel,
            kernel_features.reduction_numel,
            kernel_features.coalesce_analysis,
        )
        if not OrderedSet(preferred) & OrderedSet(["y", "z"]):
            return flat
        if "z" in preferred:
            raise FlyDSLUnsupported("transposed access needs a 2D tile")
        if kernel_features.is_reduction():
            # Flat, if its rows interleave across lanes (finalize_indexing refuses it
            # otherwise): a LayerNorm over an NCHW tensor's tokens.
            kernel_features.flydsl_transposed = True
            return flat
        return {"y": preferred["y"], "x": preferred["x"]}

    @staticmethod
    def _precheck(kernel_features) -> str | None:
        if V.graph.cpp_wrapper:
            return "cpp_wrapper"
        if V.graph.aot_mode:
            return "AOT mode"
        if not torch.version.hip:
            return "not ROCm"
        return None

    def _flat_tiling(self, kernel_features) -> dict[str, sympy.Expr]:
        tiling: dict[str, sympy.Expr] = {"x": kernel_features.numel}
        if kernel_features.is_reduction():
            tiling["r0_"] = kernel_features.reduction_numel
        return tiling

    def _codegen_flydsl(self, kernel_features: SIMDKernelFeatures):
        node_schedule = kernel_features.node_schedule
        tiling = self._choose_tiling(kernel_features)
        count_before = metrics.generated_kernel_count
        with _quiet_scheduler_codegen_errors():
            return self._codegen_flydsl_inner(
                kernel_features, node_schedule, tiling, count_before
            )

    def _codegen_flydsl_inner(
        self, kernel_features, node_schedule, tiling, count_before
    ):
        try:
            kernel = FlyDSLKernel(tiling, features=kernel_features)
            self.codegen_node_schedule_with_kernel(node_schedule, kernel)
            with V.set_kernel_handler(kernel):
                src_code = kernel.codegen_kernel()
            if not _stage_enabled(kernel.stage):
                raise FlyDSLUnsupported(f"stage {kernel.stage} disabled")
            if kernel.is_dynamic:
                src_code = self._with_aligned_variant(
                    kernel, src_code, kernel_features, node_schedule, tiling
                )
            else:
                src_code = self._with_candidates(
                    kernel, src_code, kernel_features, node_schedule, tiling
                )
        except FlyDSLUnsupported:
            metrics.generated_kernel_count = count_before
            raise
        except (NotImplementedError, TypeError, KeyError, AssertionError) as e:
            metrics.generated_kernel_count = count_before
            raise FlyDSLUnsupported(f"{type(e).__name__}: {e}") from e

        kernel_name = self._define_flydsl_kernel(src_code, node_schedule)
        kernel.kernel_name = kernel_name
        kernel.code_hash = code_hash(src_code)
        flydsl_simd_counters()["flydsl_kernels"] += 1
        flydsl_simd_counters()[
            "reduction" if kernel.is_reduction_kernel else "pointwise"
        ] += 1

        with V.set_kernel_handler(kernel):
            for node in kernel_features.scheduler_nodes():
                node.mark_run()
        base_nodes = [
            n
            for n in node_schedule
            if not isinstance(n, type) and hasattr(n, "get_name")
        ]
        self._launch_kernel_and_cleanup(kernel, base_nodes, free_buffers=False)
        self.free_buffers_in_scheduler()

    @staticmethod
    def _candidates(kernel: FlyDSLKernel) -> list[dict[str, int]]:
        """Launch parameters to time against the heuristic's for a large pointwise,
        column or short-row kernel. Which is fastest depends on what the kernel's
        inputs were last touched by, so it is measured, as Triton autotunes these: in
        ResNet's training step an in-place conv bias add takes 106us at 2048 elements a
        block, 84us at 256 (Triton 95us), while its BatchNorm backward takes 96us at
        2048 and 84us at 8192 (Triton 76us) -- and 1024-element blocks of either take
        150us. Each candidate costs a compile, so there are two a kernel."""
        stage = kernel.stage
        numel = kernel.xnumel * kernel.ynumel * kernel.rnumel
        tuned = _tune_table()
        if (
            kernel.in_combo
            or numel
            < (AUTOTUNE_MIN_TILE_NUMEL if stage == "tile" else AUTOTUNE_MIN_NUMEL)
            or kernel._has_atomic()
            or not _tune("autotune", 1)
            or "threads" in tuned
        ):
            return []
        if stage == "pointwise":
            if "ept" in tuned or "unroll" in tuned:
                return []
            w = kernel.ept
            # 4096-element blocks won a few ResNet kernels by 3-9% over these, never
            # by more.
            cands = [
                {"ept": max(1, w // 2), "unroll": 1, "threads": 128},
                {"ept": w, "unroll": 8, "threads": THREADS},
            ]
            base = (kernel.ept, kernel.unroll, kernel.threads)
            return [c for c in cands if (c["ept"], c["unroll"], c["threads"]) != base]
        if stage == "tile":
            if tuned.keys() & OrderedSet(["tile_f", "tile_s"]) or kernel.tile_f != TILE:
                return []
            # Twice the fast edge, at twice the threads: GPT-2's QKV gradient (three
            # bf16 heads layouts to rows) 39.2us -> 33.4us in the model (Triton 34.3us),
            # where a 4096^2 f32 transpose or an NCHW -> NHWC copy lose 1-10%.
            return [{"tile_f": 2 * TILE, "threads": 2 * kernel.threads}]
        # Not tuned, as no candidate won across ten models' autotuned kernels: a row's
        # rows per lane group.
        if stage == "column":
            # Twice the lanes on each column: Qwen3's per-head norm weight gradient
            # (bf16 rows 128 wide) 37us -> 33us (Triton 32us), but an 8192^2 f32
            # column sum 82us -> 103us.
            return [{"threads": 2 * kernel.threads}] if kernel.threads < 512 else []
        if (
            stage in ("row", "norm")
            and kernel.persistent
            and not kernel.one_step
            and "row_tpr" not in tuned
            and "xunroll" not in tuned
        ):
            # Fewer lanes a row: Qwen3's q-norm + RoPE (128 wide) 134us -> 85us at 8.
            # Not when that loops over the row instead: it lost all 55 such kernels
            # across ten models, by 13-76%.
            half = kernel.tpr // 2
            cands = (
                [{"row_tpr": half}]
                if half
                and -(-kernel.rnumel // (half * kernel.ept)) * kernel.ept
                <= PERSISTENT_MAX_UNROLL
                else []
            )
            if kernel.tpr < WAVE:
                # Or more, up to a wave: GPT-2's masked softmax backward (512 bf16)
                # 62.5us -> 58.4us in the model (Triton 61us).
                cands.append({"row_tpr": 2 * kernel.tpr})
            # Or more: an ALU-bound row (Qwen3's k-norm + RoPE, angles computed in the
            # kernel) 70us -> 61us at 4 elements a lane rather than 8 (Triton 101us).
            if kernel.ept > 2 and "ept" not in tuned:
                cands.append({"ept": kernel.ept // 2})
            return cands
        return []

    def _with_candidates(
        self, kernel, src_code, kernel_features, node_schedule, tiling
    ) -> str:
        cands = self._candidates(kernel)
        if not cands:
            return src_code
        # A model's repeated layers lower to the same source: lowering each candidate
        # again took a third of a Qwen3 prefill's codegen.
        memo = self.__dict__.setdefault("_candidate_sources", {})
        if src_code in memo:
            return memo[src_code]
        memo[src_code] = self._lower_candidates(
            cands, kernel, src_code, kernel_features, node_schedule, tiling
        )
        return memo[src_code]

    def _with_aligned_variant(
        self, kernel, src_code, kernel_features, node_schedule, tiling
    ) -> str:
        """A dynamic kernel, and beside it the variant for sizes that are multiples of
        SIZE_ALIGN when that one differs (wider runs, fewer masks): a row cumsum over a
        dynamic 8192 99us -> 17us. Sizes that are expressions (``ps0``) keep one."""
        sizes = kernel.args.sizevars
        if not all(
            isinstance(e, sympy.Symbol) and symbol_is_type(e, SymT.SIZE) for e in sizes
        ):
            return src_code
        memo = self.__dict__.setdefault("_aligned_sources", {})
        if src_code in memo:
            return memo[src_code]
        count = metrics.generated_kernel_count
        result = src_code
        try:
            with _assume_aligned():
                k = FlyDSLKernel(tiling, features=kernel_features)
                self.codegen_node_schedule_with_kernel(node_schedule, k)
                with V.set_kernel_handler(k):
                    aligned = k.codegen_kernel()
            if k.arg_names() == kernel.arg_names() and aligned != src_code:
                result = _dispatch_source(
                    src_code, aligned, kernel.arg_names(), list(sizes.values())
                )
        except FlyDSLUnsupported:
            pass
        finally:
            # pyrefly: ignore [bad-assignment]
            metrics.generated_kernel_count = count
        memo[src_code] = result
        return result

    def _recent_buffers(self, kernel_features) -> OrderedSet[str]:
        """The buffers the nodes just before this kernel's wrote or read, back through
        WARM_BYTES of them: what is likely still in the last-level cache when it runs."""
        nodes = self.scheduler.nodes
        cache = self.__dict__.get("_node_pos")
        if cache is None or cache[0] is not nodes or len(cache[1]) < len(nodes):
            pos = {}
            for i, n in enumerate(nodes):
                for sub in n.get_nodes():
                    pos[sub.get_name()] = i
            cache = self.__dict__["_node_pos"] = (nodes, pos)
        pos = cache[1]
        mine = [
            pos[n.get_name()]
            for n in kernel_features.scheduler_nodes()
            if n.get_name() in pos
        ]
        recent: OrderedSet[str] = OrderedSet()
        if not mine:
            return recent
        total = 0
        for j in range(min(mine) - 1, -1, -1):
            names = [b.get_name() for b in nodes[j].get_outputs()]
            names += [d.name for d in nodes[j].read_writes.reads]
            for name in names:
                if name in recent:
                    continue
                try:
                    buf = V.graph.get_buffer(name)
                    nbytes = int(buf.get_numel()) * buf.get_dtype().itemsize
                except (AttributeError, KeyError, NotImplementedError, TypeError):
                    continue
                recent.add(name)
                total += nbytes
            if total >= WARM_BYTES:
                break
        return recent

    def _warm_args(self, kernel, kernel_features, arg_names) -> tuple[int, ...]:
        """Positions in ``arg_names`` of the buffers ``pick_fastest`` should time with
        the last-level cache holding."""
        recent = self._recent_buffers(kernel_features)
        inner = dict(kernel.args.input_buffers)
        for name, buf in kernel.args.inplace_buffers.items():
            if not isinstance(buf, str):
                inner[name] = buf.inner_name
        real = getattr(self.scheduler, "mutation_real_name", {})
        warm = OrderedSet(
            inner[n] for n in inner if n in recent or real.get(n, n) in recent
        )
        return tuple(i for i, a in enumerate(arg_names) if a in warm)

    def _lower_candidates(
        self, cands, kernel, src_code, kernel_features, node_schedule, tiling
    ) -> str:
        arg_names = kernel.arg_names()
        warm = self._warm_args(kernel, kernel_features, arg_names)
        srcs = [src_code]
        count = metrics.generated_kernel_count
        try:
            for cand in cands:
                with _tune_override(cand):
                    k = FlyDSLKernel(tiling, features=kernel_features)
                    self.codegen_node_schedule_with_kernel(node_schedule, k)
                    with V.set_kernel_handler(k):
                        srcs.append(k.codegen_kernel())
                if k.arg_names() != arg_names:
                    return src_code
        finally:
            # pyrefly: ignore [bad-assignment]
            metrics.generated_kernel_count = count
        return _autotune_source(srcs, arg_names, warm)

    def _define_flydsl_kernel(self, src_code: str, node_schedule) -> str:
        _warm_compile_workers()
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]
        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )
        h = hashlib.sha256(src_code.encode("utf-8")).hexdigest()[:8]
        # FlyDSL names its cache directories after the kernel ("<name>_c0_launch_<md5>"),
        # and a file name has at most 255 bytes: a Qwen3 fusion's name alone took 230.
        fused_name = fused_name[:120].rstrip("_")
        kernel_name = (
            f"flydsl_{fused_name}_{h}"
            if fused_name and fused_name != "fused"
            else f"flydsl_{h}"
        )
        wrapper.src_to_kernel[src_code] = kernel_name
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)
        _, _, kernel_path = get_path(code_hash(src_code), "py")
        # An autotuned module's variants compile as separate jobs, one per worker: in
        # one job they compiled one after another, 2-3 s a kernel.
        variants = 1
        while f"{kernel_name}_c{variants}_kernel" in src_code:
            variants += 1
        source = f"{kernel_name}_source"
        literal_buf = IndentedBuffer()
        literal_buf.splice(src_code, strip=True)
        literal = literal_buf.getvalue()
        if V.graph.cpp_wrapper:
            binaries = None
        else:
            # The module's path, as async_compile writes it, names its code objects.
            stem = get_path(code_hash(literal.strip()), "py")[2].rsplit(".", 1)[0]
            binaries = [
                f"{stem}.{kernel_name}{f'_c{i}' if variants > 1 else ''}.hsaco"
                for i in range(variants)
            ]
            wrapper.add_import_once(
                "from torch._inductor.codegen.flydsl import flydsl_simd_runtime as fdp_rt"
            )
            from .flydsl_simd_runtime import compile_early

            compile_early(kernel_name, literal, binaries)

        def compile_call(src: str, meta: str, i: int) -> str:
            if binaries is None:
                return f"async_compile.flydsl({kernel_name!r}, {src}, precompile_metadata={meta})"
            return (
                f"fdp_rt.compiled(async_compile, {kernel_name!r}, {src}, "
                f"{binaries[i]!r}, precompile_metadata={meta})"
            )

        if variants > 1:
            head, tail = f"({source} := r'''", "''')"
            meta = "{'only': 0}"
        else:
            head, tail, meta = "r'''", "'''", "{}"
        call = compile_call("\0", meta, 0)
        cut = call.index("\0")
        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(call[:cut] + head)
        compile_wrapper.splice(literal)
        compile_wrapper.writeline(tail + call[cut + 1 :])
        origins, detailed = get_kernel_metadata(node_schedule, wrapper)
        metadata = f"# kernel path: {kernel_path}\n{origins}\n{detailed}"
        body = compile_wrapper.getvalue()
        if not getattr(wrapper, "flydsl_waits_for_pool", False):
            # async_compile hands kernels to the compile workers only once the pool is
            # up, and those defined before are compiled lazily on their first call, one
            # at a time: 39 flyc.compile calls, 11 s of Llama inference's 17 s first
            # step. The pool has been starting since compilation began; the module runs
            # this line on load, after all of codegen. Not when the first kernel's code
            # object is already on disk (a warm start compiles nothing): 7.6 s vs 4 s.
            wrapper.flydsl_waits_for_pool = True
            if binaries is not None:
                metadata = (
                    f"if not os.path.exists({binaries[0]!r}):\n"
                    "    async_compile.wait_process_pool_ready()\n"
                    f"{metadata}"
                )
        wrapper.define_kernel(kernel_name, body, metadata)
        for i in range(1, variants):
            wrapper.define_kernel(
                f"{kernel_name}_v{i}", compile_call(source, f"{{'only': {i}}}", i)
            )
        return kernel_name


@functools.cache
def _warm_compile_workers() -> None:
    """Once a process: start every compile worker importing and setting up FlyDSL
    (~2 s each) while the kernels are still being generated."""
    from torch._inductor import async_compile

    threads = async_compile.get_compile_threads()
    if (
        threads <= 1
        or not async_compile._process_pool_allowed()
        or config.triton.proton_profiling
    ):
        return
    from .flydsl_simd_runtime import warm_worker

    pool = async_compile.AsyncCompile.process_pool()
    preloads = getattr(pool, "preload", ())
    if any("flydsl_simd_runtime:preload_worker" in p for p in preloads):
        return  # the workers forked with it done
    for _ in range(threads):
        pool.submit(warm_worker, _gpu_arch())


@functools.cache
def flydsl_simd_available() -> bool:
    try:
        # pyrefly: ignore [missing-import]
        import flydsl.compiler  # noqa: F401
    except ImportError:
        return False
    return bool(torch.version.hip)


del BackendFeature, OpsWrapper

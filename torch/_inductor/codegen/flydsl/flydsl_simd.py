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
``flydsl_simd_runtime``), so buffers of 2 GiB or more are refused, as are dynamic shapes:
a buffer descriptor needs an exact byte extent, and with symbolic sizes it would have to
be computed on the device with no way to refuse an oversized one.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import math
import re
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, Identity, ModularIndexing
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT

from ... import config, metrics
from ...codecache import code_hash, get_path
from ...utils import get_fused_kernel_name, get_kernel_metadata, Placeholder
from ...virtualized import ops, OpsWrapper, V
from ..common import (
    BackendFeature,
    DeferredLine,
    IndentedBuffer,
    OpOverrides,
    PythonPrinter,
)
from ..simd import IterationRangesEntry, SIMDKernel
from ..triton import TritonScheduling


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...ops_handler import ReductionType, StoreMode
    from ..simd_kernel_features import SIMDKernelFeatures


log = logging.getLogger(__name__)

THREADS = 256
WAVE = 64
# A trace-time-unrolled ("persistent") reduction body is at most this many copies of the
# loop body per loop; beyond it the code size and register pressure outgrow the reuse.
PERSISTENT_MAX_UNROLL = 16
# Buffers one combo (foreach) launch may take; larger groups are split.
COMBO_MAX_ARGS = 128
# Edge of the square tile a block covers in a {y, x} (transposing) pointwise kernel.
TILE = 64
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
}
_COMPUTE = {
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
            raise FlyDSLUnsupported("index constant outside int32")
        return str(v)

    def _print_FloorDiv(self, expr):
        x, div = expr.args
        # pyrefly: ignore [missing-attribute]
        return f"fdp.idiv({self._print(x)}, {self._print(div)})"

    _print_CleanDiv = _print_FloorDiv

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
        rendered = V.kernel.kexpr(V.kernel.prepare_indexing(expr))
        return f"fdp.to({rendered}, {compute_kind(dtype)!r})"

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
        if k in ("f16", "bf16") and not use_compute_types:
            return f"fdp.to(fdp.to({x}, {k!r}), 'f32')"
        if k in ("f16", "bf16"):
            # Inductor marks low-precision results that must round; do it.
            return f"fdp.to(fdp.to({x}, {k!r}), 'f32')"
        return f"fdp.to({x}, {_COMPUTE[k]!r})"

    @staticmethod
    def to_dtype_bitcast(x, dtype, src_dtype):
        raise FlyDSLUnsupported("bitcast")

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


def _tune(name: str, default: int = 0) -> int:
    return _tune_table().get(name, default)


def _num_cus() -> int:
    try:
        return torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    except Exception:
        return 304


def _int_or_none(e) -> int | None:
    try:
        return int(e)
    except (TypeError, ValueError):
        return None


def classify_access(
    expr: sympy.Expr, var: sympy.Symbol, w: int, uniform: frozenset[str] = frozenset()
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
    linear = 0
    aligned = True
    for term in sympy.Add.make_args(expr):
        coeff, factor = term.as_coeff_Mul()
        c = _int_or_none(coeff)
        if c is None:
            return "other"
        if var not in factor.free_symbols:
            if c % w:
                # A uniform term moves the run's base; it must keep it aligned.
                f = _int_or_none(factor) if factor.is_number else None
                if f is None or (c * f) % w:
                    aligned = False
            continue
        if factor == var:
            linear += c
            continue
        if isinstance(factor, ModularIndexing) and factor.args[0] == var:
            d, m = (_int_or_none(a) for a in factor.args[1:])
            if d is None or m is None:
                return "other"
            if d % w == 0:
                if c % w:
                    aligned = False
                continue
            if d == 1 and m % w == 0:
                linear += c
                continue
            return "other"
        if isinstance(factor, FloorDiv) and factor.args[0] == var:
            d = _int_or_none(factor.args[1])
            if d is not None and d % w == 0:
                if c % w:
                    aligned = False
                continue
            return "other"
        return "other"
    if linear == 0:
        return "uniform"
    if linear == 1 and aligned:
        return "contig"
    return "other"


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
        self.xnumel = self._static(self.numels["x"], "xnumel")
        self.is_reduction_kernel = "r0_" in self.numels
        self.rnumel: int = (
            self._static(self.numels["r0_"], "rnumel")
            if self.is_reduction_kernel
            else 1
        )
        # A {y, x} pointwise kernel runs TILE x TILE tiles: see _finalize_tile2d.
        self.tile2d = "y" in self.numels
        self.ynumel = self._static(self.numels["y"], "ynumel") if self.tile2d else 1
        self.fast = "x"  # the dimension a thread's vector runs follow
        self.staged: list[dict[str, Any]] = []
        if self.xnumel * self.rnumel * self.ynumel > _I32_LIMIT:
            raise FlyDSLUnsupported("more than 2**31 elements")
        self.outside_loop_vars: OrderedSet[Any] = OrderedSet()
        self.post_loop_combine = IndentedBuffer()
        # Per loop step, around the run of a thread's elements: chunked combines.
        self.chunk_pre = IndentedBuffer()
        self.chunk_post = IndentedBuffer()
        # Pointwise only: the indexing and loads alone, replayed ahead of the body.
        self.prefetch = IndentedBuffer()
        self._node_index_vars: list[list[list[sympy.Expr]]] = []
        self.acc_counter = 0
        self.uses_smem = False
        self.max_components = 1
        self.reduction_cache: dict[Any, Any] = {}
        self.loop_stored_args: dict[str, str] = {}
        # Decided in finalize_indexing. A thread owns `unroll` runs of `ept` contiguous
        # elements per step; in a reduction, `tpr` threads `stride` apart share a row,
        # and a block covers `rpb` rows `xunroll` times over.
        self.ept = 1
        self.unroll = 1
        self.tpr = 1
        self.stride = 1
        self.rpb = THREADS
        self.xunroll = 1
        self.persistent = False
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

    def _describe(self) -> str:
        if self.tile2d:
            return (
                f"tiled pointwise ynumel={self.ynumel} xnumel={self.xnumel} fast={self.fast} "
                f"ept={self.ept} staged={[(s['arg'], s['w']) for s in self.staged]}"
            )
        if not self.is_reduction_kernel:
            return f"pointwise xnumel={self.xnumel} ept={self.ept} unroll={self.unroll}"
        mode = "column" if self.stride > 1 else "row"
        return (
            f"{mode} reduction xnumel={self.xnumel} rnumel={self.rnumel} ept={self.ept} "
            f"unroll={self.unroll} tpr={self.tpr} stride={self.stride} xunroll={self.xunroll} "
            f"persistent={self.persistent}"
        )

    def _finalize_tile2d(self, widest_vec: int) -> None:
        """Layout of a {y, x} pointwise kernel: each block covers a TILE x TILE tile.

        A thread's vector runs follow the ``fast`` dimension, chosen so the stores are
        contiguous along it. Loads contiguous along the other dimension are staged:
        read coalesced along that dimension into an LDS tile, then read back
        transposed (see ``load`` and ``emit_inner``).
        """
        stores = self._memory_roots(loads=False) or []
        x, y = self._x_sym(), self._tree_sym("y")
        x_st = sum(classify_access(e, x, 2) == "contig" for e in stores)
        y_st = sum(classify_access(e, y, 2) == "contig" for e in stores)
        self.fast = "y" if y_st > x_st else "x"
        f = self._tree_sym(self.fast)
        fast_roots = [
            e
            for e in (self._memory_roots() or [])
            if classify_access(e, f, 2) == "contig"
        ]
        ept = widest_vec
        while ept > 1 and not all(
            classify_access(e, f, ept) == "contig" for e in fast_roots
        ):
            ept //= 2
        self.ept = _tune("ept", ept)
        # Runs of a pass: TILE // ept threads across the fast dimension, the block's
        # remaining threads down the slow one; `unroll` passes cover the tile.
        self.unroll = TILE * TILE // (THREADS * self.ept)
        self.rpb = THREADS * self.ept // TILE

    def _tile_counts(self) -> tuple[int, int]:
        """Tiles along (fast, slow)."""
        nf = self.xnumel if self.fast == "x" else self.ynumel
        ns = self.ynumel if self.fast == "x" else self.xnumel
        return -(-nf // TILE), -(-ns // TILE)

    def _pure_reduction(self) -> bool:
        """Every node is a reduction: the kernel stores only row results."""
        return all(n.is_reduction() for n in self.features.scheduler_nodes())

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

    def finalize_indexing(self, indices: Sequence[sympy.Expr]) -> None:
        """Choose the thread layout -- ``ept``, ``unroll``, ``tpr``, ``stride``,
        ``xunroll``, persistence -- from the kernel's shape and access patterns."""
        widest_vec = max(1, min(8, 16 // min(self._buffer_itemsizes())))
        roots = [self._root_expr(i) for i in indices]
        if self.tile2d:
            self._finalize_tile2d(widest_vec)
            return
        # Enough blocks to fill the GPU a few times over before a thread takes more work.
        enough_blocks = 4 * _num_cus()
        if not self.is_reduction_kernel:
            self.ept = _largest_pow2_dividing(self.xnumel, widest_vec)
            # Two runs per thread keep two vector loads in flight per input, as Triton's
            # default pointwise config does, once the grid is big enough to afford it.
            per_block = THREADS * self.ept
            self.unroll = 2 if self.xnumel >= 2 * per_block * enough_blocks else 1
            return
        x, r = self._x_sym(), self._r_sym()
        r_contig = sum(
            classify_access(e, r, 2) == "contig" for e in roots if r in e.free_symbols
        )
        x_contig = sum(
            classify_access(e, x, 2) == "contig" for e in roots if r in e.free_symbols
        )
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
            self.tpr = THREADS // self.stride
            self.rpb = self.stride
        else:
            self.ept = _largest_pow2_dividing(self.rnumel, widest_vec)
            # Narrower vectors when they tile a one-wave row exactly where the widest
            # leave a partly masked last run: 768 = 64 lanes x 3 runs of 4.
            ept = self.ept
            while (
                ept > 1
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
            tpr = min(THREADS, _pow2_ceil(runs))
            if (
                tpr > WAVE
                and -(-self.rnumel // (WAVE * self.ept)) * self.ept
                <= PERSISTENT_MAX_UNROLL
            ):
                # One wave still holds the row in registers: shuffles only, no LDS
                # round trip and barriers per reduction. 31us vs 39us on a 768-wide
                # add+layernorm, where 128 threads also left a quarter of them idle.
                tpr = WAVE
            self.tpr = _tune("row_tpr", tpr)
            self.rpb = THREADS // self.tpr
        iters = -(-self.rnumel // (self.tpr * self.ept))
        self.persistent = iters * self.ept <= PERSISTENT_MAX_UNROLL
        if self.persistent:
            # Short rows: the whole row is unrolled already, so let each thread group take
            # several rows instead, while the grid stays large.
            # Normalization kernels (several passes, a full-size store) measured best
            # with one row per group: 89us vs 97us at two rows on a 64-wide softmax. A
            # bare reduction wanted two: 43us vs 49us on a 32-wide sum.
            self.xunroll = 2 if self._pure_reduction() else 1
            while (
                self.xunroll > 1
                and -(-self.xnumel // (self.rpb * self.xunroll)) < enough_blocks
            ):
                self.xunroll //= 2
            self.xunroll = _tune("xunroll", self.xunroll)
        else:
            # Long rows: several runs per loop iteration, for loads in flight.
            self.unroll = max(
                1,
                min(8 if gather else 4, PERSISTENT_MAX_UNROLL // self.ept, iters // 2),
            )
            self.unroll = _tune(
                "col_unroll" if self.stride > 1 else "row_unroll", self.unroll
            )
        self.rstep = self.tpr * self.ept * self.unroll

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
            return None if self.xnumel % TILE == 0 else "xmask"
        if self.is_reduction_kernel:
            block = self.rpb * self.xunroll
        else:
            block = THREADS * self.ept * self.unroll
        return None if self.xnumel % block == 0 else "xmask"

    def _rmask(self) -> str | None:
        if not self.is_reduction_kernel or not self.inside_reduction:
            return None
        return None if self.rnumel % self.rstep == 0 else "r0_mask"

    def _masks(self, *, include_r: bool = True) -> list[str]:
        ms = []
        if (m := self._xmask()) is not None:
            ms.append(m)
        if self.tile2d and self.ynumel % TILE:
            ms.append("ymask")
        if include_r and (m := self._rmask()) is not None:
            ms.append(m)
        if self._load_mask is not None:
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
        arg = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        st = storage_kind(dtype)
        index = self.prepare_indexing(index)
        index_str = self.kexpr(self.rename_indexing(index))
        root = self._root_expr(index)
        dependent = free_symbol_is_type(index, SymT.TMP) or self._load_mask is not None
        uniform = frozenset(self.uniform_vars)
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
        ms = self._masks(include_r=not hoist)
        mask = self._mask_arg(ms)
        var = None if hoist else self._vector_var()
        kind = "other"
        w = 1
        if var is not None and self._load_mask is None:
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
        if kind == "contig":
            line = f"fdp.vload({self._cache()}, {key}, _j, {w}, _r_{arg}, {index_str}, {st!r}{mask})"
        elif kind == "uniform":
            line = f"fdp.uload({self._cache()}, {key}, _j // {self.ept}, _r_{arg}, {index_str}, {st!r}{mask})"
        else:
            line = f"fdp.load(_r_{arg}, {index_str}, {st!r}{mask})"
            if self._persistent_region() and not hoist and not dependent:
                line = f"fdp.memo(_lc, ({key}, _j), lambda: {line})"
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
        if w == 1:
            return None
        ck = compute_kind(dtype)
        words = 2 if ck in ("i64", "f64") else 1
        for s in self.staged:
            if (s["arg"], s["index"]) == (arg, index_str):
                break
        else:
            base = sum(s["words"] * TILE * TILE_PITCH for s in self.staged)
            if base + words * TILE * TILE_PITCH > MAX_TILE_WORDS:
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
        plane = TILE * TILE_PITCH
        return f"fdp.tile_get(_smem, {s['base']}, {plane}, _fl * {TILE_PITCH} + _sl, {ck!r})"

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
        if mode not in (None, "atomic_add"):
            raise FlyDSLUnsupported(f"store mode {mode}")
        arg = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        st = storage_kind(dtype)
        if mode == "atomic_add" and st != "f32":
            raise FlyDSLUnsupported(f"atomic_add of {dtype}")
        index = self.prepare_indexing(index)
        index_str = self.kexpr(self.rename_indexing(index))
        in_loop = self._in_loop_region()
        if mode == "atomic_add":
            ms = self._masks(include_r=in_loop)
            line = f"fdp.atomic_add(_r_{arg}, {index_str}, {value}, 'f32'{self._mask_arg(ms)})"
            self.stores.writeline(DeferredLine(name, line))
            self.store_buffer_names.add(name)
            return
        ms = self._masks(include_r=in_loop)
        if self.is_reduction_kernel and not in_loop and self.tpr > 1:
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
        if contig:
            line = (
                f"fdp.vstore(_sc, {repr(f'st{self.load_site}_{name}')}, _j, {w}, _r_{arg}, "
                f"{index_str}, {value}, {st!r}{mask})"
            )
            self.load_site += 1
        else:
            line = f"fdp.store(_r_{arg}, {index_str}, {value}, {st!r}{mask})"
        # Deferred: a buffer that turns out to be kernel-local is removed after the body
        # is written, and its store has to disappear with it.
        buf.writeline(DeferredLine(name, line))
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

    def scan(self, *args, **kwargs):
        raise FlyDSLUnsupported("scan")

    def sort(self, *args, **kwargs):
        raise FlyDSLUnsupported("sort")

    def bucketize(self, *args, **kwargs):
        raise FlyDSLUnsupported("bucketize")

    def partial_accumulate(self, *args, **kwargs):
        raise FlyDSLUnsupported("partial_accumulate")

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
        geo = f"{tpr}, {smem}, {THREADS}, {self.stride}"

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
        elif reduction_type in ("argmax", "argmin"):
            logical_index = None
            if isinstance(value, tuple):
                value, logical_index = value
            op = "max" if reduction_type == "argmax" else "min"
            k = compute_kind(src_dtype)
            acc = self._new_acc()
            self.body.writeline(f"{acc}_v, {acc}_i = fdp.arg_default({op!r}, {k!r})")
            index_var = str(logical_index) if logical_index is not None else "r0_index"
            self.compute.writeline(
                f"{acc}_v, {acc}_i = fdp.arg_combine({op!r}, {acc}_v, {acc}_i, {value}, {index_var}{mask})"
            )
            _, result = self._post(
                f"fdp.group_arg_reduce({op!r}, {acc}_v, {acc}_i, {geo})", 2, dtype
            )
            self.max_components = max(self.max_components, 2)
        elif reduction_type == "welford_reduce" and self.persistent:
            acc = self._new_acc()
            self.body.writeline(f"{acc}_items = []")
            self.compute.writeline(f"{acc}_items.append({item})")
            result = tuple(
                self._post(
                    f"fdp.group_welford_twopass({acc}_items, {geo})", 3, torch.float32
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
                self.body.writeline(
                    f"for r0_offset in range_constexpr(0, {self.rnumel}, {self.rstep}):"
                )
            else:
                self.body.writeline(
                    f"for r0_offset in range(fx.Int32(0), fx.Int32({self.rnumel}), fx.Int32({self.rstep})):"
                )
            with self.body.indent():
                self.body.splice(self.chunk_pre)
                self.body.writeline(
                    f"for _j in range_constexpr({self.ept * self.unroll}):"
                )
                with self.body.indent():
                    # Run `_j // ept` of this step: runs are tpr*ept apart, so each vector
                    # instruction covers one contiguous span across the row's threads.
                    self.body.writeline(
                        f"r0_index = r0_offset + (_j // {self.ept}) * {self.tpr * self.ept} "
                        f"+ _rl * {self.ept} + _j % {self.ept}"
                    )
                    if self._rmask():
                        self.body.writeline(f"r0_mask = r0_index < {self.rnumel}")
                    self.body.splice(self.indexing_code)
                    self.body.splice(self.loads)
                    self.body.splice(self.compute)
                    self.body.splice(self.stores)
                self.body.splice(self.chunk_post)
            for arg, name in self.loop_stored_args.items():
                self.body.writeline(DeferredLine(name, f"fdp.forget(_lc, {arg!r})"))
            self.loop_stored_args.clear()
            self.cse.invalidate(self.outside_loop_vars)
            self.range_trees[-1].cache_clear()
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

    def _arg_extents(self, arg_defs) -> dict[str, int]:
        """Exact byte extent of each tensor argument, for its buffer descriptor."""
        name_of = {inner: outer for outer, inner in self.args.input_buffers.items()}
        for outer, inner in self.args.output_buffers.items():
            if not isinstance(inner, str) or inner.startswith("REMOVED"):
                continue
            name_of[inner] = outer
        for outer, inner in self.args.inplace_buffers.items():
            if hasattr(inner, "inner_name"):
                name_of[inner.inner_name] = outer
        extents = {}
        for a in arg_defs:
            buf = name_of.get(a)
            if buf is None:
                raise FlyDSLUnsupported(f"argument {a} without a buffer")
            node = V.graph.try_get_buffer(buf)
            if node is None:
                raise FlyDSLUnsupported(f"no layout for {buf}")
            layout = node.get_layout()
            n = _int_or_none(V.graph.sizevars.simplify(layout.storage_size()))
            if n is None:
                raise FlyDSLUnsupported("dynamic buffer extent")
            nbytes = max(1, n) * node.get_dtype().itemsize
            if nbytes >= _I32_LIMIT:
                raise FlyDSLUnsupported("buffer of 2 GiB or more")
            extents[a] = nbytes
        return extents

    @property
    def smem_words(self) -> int:
        if self.staged:
            return sum(s["words"] * TILE * TILE_PITCH for s in self.staged)
        # Mirrors fdp.smem_words: a slot per thread per 32-bit word of each component.
        return THREADS * 2 * self.max_components if self.uses_smem else 0

    @property
    def num_blocks(self) -> int:
        if self.tile2d:
            nf, ns = self._tile_counts()
            return nf * ns
        if not self.is_reduction_kernel:
            return -(-self.xnumel // (THREADS * self.ept * self.unroll))
        return -(-self.xnumel // (self.rpb * self.xunroll))

    def arg_names(self) -> list[str]:
        arg_defs, _, _, _ = self.args.python_argdefs()
        names = [a.full_name() for a in arg_defs]
        if any(a in self.args.sizevars.values() for a in names):
            raise FlyDSLUnsupported("dynamic shapes")
        return names

    def codegen_kernel(self, name=None):
        """The module source: the ``@flyc.kernel``, its ``@flyc.jit`` launcher, and the
        ``_main`` / ``_precompile`` entry points the wrapper calls."""
        self.codegen_body()
        arg_names = self.arg_names()
        extents = self._arg_extents(arg_names)
        code = _module_prologue(arg_names, extents, self.smem_words, [self._describe()])
        with code.indent():
            code.writeline("_b = fdp.bid()")
            self.emit_inner(code)
        code.splice(_module_epilogue(arg_names, self.num_blocks))
        return code.getvalue()

    def emit_inner(self, code: IndentedBuffer) -> None:
        """The thread layout and body, given ``_t``, ``_b``, the descriptors and ``_smem``."""
        if self.tile2d:
            self._emit_tile2d(code)
        elif not self.is_reduction_kernel:
            code.writeline("_vc = {}")
            code.writeline("_sc = {}")
            ept, span = self.ept, THREADS * self.ept
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
                code.writeline(f"for _j in range_constexpr({ept * self.unroll}):")
                with code.indent():
                    code.writeline(
                        f"xindex = _b * {span * self.unroll} + (_j // {ept}) * {span} + _t * {ept} + _j % {ept}"
                    )
                    if self._xmask():
                        code.writeline(f"xmask = xindex < {self.xnumel}")
                    code.splice(part)
        else:
            if self.stride == 1:
                code.writeline(f"_rl = _t % {self.tpr}")
                lane_x = f"_t // {self.tpr}"
            else:
                code.writeline(f"_rl = _t // {self.stride}")
                lane_x = f"_t % {self.stride}"
            code.writeline("_lane0 = _rl == 0")
            code.writeline(f"for _xr in range_constexpr({self.xunroll}):")
            with code.indent():
                # Trace-time load caches are per row.
                code.writeline("_vc = {}")
                code.writeline("_sc = {}")
                code.writeline("_lc = {}")
                code.writeline(
                    f"xindex = _b * {self.rpb * self.xunroll} + _xr * {self.rpb} + {lane_x}"
                )
                if self._xmask():
                    code.writeline(f"xmask = xindex < {self.xnumel}")
                code.splice(self.body)

    def _emit_tile2d(self, code: IndentedBuffer) -> None:
        f = self.fast
        s = "y" if f == "x" else "x"
        nf, _ = self._tile_counts()
        code.writeline(f"_bf = _b % {nf}")
        code.writeline(f"_bs = _b // {nf}")
        masks = []
        if self.xnumel % TILE:
            masks.append(f"xmask = xindex < {self.xnumel}")
        if self.ynumel % TILE:
            masks.append(f"ymask = yindex < {self.ynumel}")
        entries = [
            f"{e.name} = {self.kexpr(self.rename_indexing(e.expr))}"
            for e in self.range_tree_nodes.values()
        ]
        mask_arg = self._mask_arg([m.split(" ")[0] for m in masks])

        def coords(fl: str, sl: str) -> None:
            code.writeline(f"_fl = {fl}")
            code.writeline(f"_sl = {sl}")
            code.writeline(f"{f}index = _bf * {TILE} + _fl")
            code.writeline(f"{s}index = _bs * {TILE} + _sl")
            for m in masks:
                code.writeline(m)

        # Staging: threads run along the slow dimension, so each staged load is a
        # coalesced vector read; the tile keeps (fast, slow) coordinates.
        for i, st in enumerate(self.staged):
            w = st["w"]
            tps = TILE // w
            rps = THREADS // tps
            code.writeline(f"_vs{i} = {{}}")
            code.writeline(f"for _j in range_constexpr({w * (TILE // rps)}):")
            with code.indent():
                coords(
                    f"_t // {tps} + (_j // {w}) * {rps}",
                    f"(_t % {tps}) * {w} + _j % {w}",
                )
                for e in entries:
                    code.writeline(e)
                code.writeline(
                    f"_v = fdp.vload(_vs{i}, 'stage', _j, {w}, _r_{st['arg']}, {st['index']}, "
                    f"{st['st']!r}{mask_arg})"
                )
                code.writeline(
                    f"fdp.tile_put(_smem, {st['base']}, {TILE * TILE_PITCH}, "
                    f"_fl * {TILE_PITCH} + _sl, _v)"
                )
        if self.staged:
            code.writeline("fdp.barrier()")

        code.writeline("_vc = {}")
        code.writeline("_sc = {}")
        ept, tpf = self.ept, TILE // self.ept
        passes = [self.prefetch, self.body] if self.prefetch else [self.body]
        for part in passes:
            code.writeline(f"for _j in range_constexpr({ept * self.unroll}):")
            with code.indent():
                coords(
                    f"(_t % {tpf}) * {ept} + _j % {ept}",
                    f"_t // {tpf} + (_j // {ept}) * {self.rpb}",
                )
                code.splice(part)

    def call_kernel(self, name: str, node=None, deallocate_ws: bool = True):
        wrapper = V.graph.wrapper_code
        _, call_args, _, arg_types = self.args.python_argdefs()
        wrapper.generate_kernel_call(name, call_args, triton=True, arg_types=arg_types)


# ---------------------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------------------


def _module_prologue(
    arg_names: list[str], extents: dict[str, int], smem_words: int, notes: list[str]
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
    sig = ", ".join(f"{a}: fx.Int64" for a in arg_names)
    code.writeline("")
    code.writeline("@flyc.kernel")
    code.writeline(f"def {Placeholder.KERNEL_NAME}_kernel({sig}):")
    with code.indent():
        # FlyDSL's binary cache keys on this function's source and on helpers in the
        # same directory -- not on the runtime modules it calls -- so stamp them in.
        code.writeline(f"# runtime {_runtime_fingerprint()}")
        for note in notes:
            code.writeline(f"# {note}")
        for a in arg_names:
            code.writeline(f"_r_{a} = fdp.rsrc({a}, {extents[a]})")
        if smem_words:
            code.writeline(
                f"_smem = fx.SharedAllocator().allocate(_Smem).peek().words.view(fx.make_layout({smem_words}, 1))"
            )
        code.writeline("_t = fdp.tid()")
    return code


def _module_epilogue(arg_names: list[str], grid: int) -> str:
    kname = str(Placeholder.KERNEL_NAME)
    launch_sig = ", ".join([f"{a}: fx.Int64" for a in arg_names] + ["_grid: fx.Int32"])
    return f"""

        @flyc.jit
        def {kname}_launch({launch_sig}, stream: fx.Stream = fx.Stream(None)):
            {kname}_kernel({", ".join(arg_names)}).launch(grid=(_grid, 1, 1), block=({THREADS}, 1, 1), stream=stream)


        _compiled = None


        def {kname}_main({", ".join(arg_names)}, stream=None):
            global _compiled
            args = ({"".join(f"{a}.data_ptr(), " for a in arg_names)}{grid}, 0 if stream is None else stream)
            if _compiled is None:
                # flyc.compile launches once while compiling, so this call is done.
                _compiled = flyc.compile({kname}_launch, *args)
                return
            _compiled(*args)


        def {kname}_precompile(**_):
            # All-zero addresses with a zero grid: compiles into FlyDSL's disk cache and
            # launches nothing.
            {kname}_launch({"0, " * len(arg_names)}0, 0)
        """


def _combo_source(kernels: list[FlyDSLKernel]) -> str:
    """One launch for several kernels sharing a KernelArgs: each owns a contiguous range
    of block ids and runs, with ``_b`` rebased, under a branch on the block id."""
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
    code = _module_prologue(arg_names, extents, smem, [k._describe() for k in kernels])
    start = 0
    with code.indent():
        code.writeline("_bid = fdp.bid()")
        if len(kernels) == 1:
            code.writeline("_b = _bid")
            code.splice(inners[0])
            start = kernels[0].num_blocks
        for i, (k, inner) in enumerate(
            zip(kernels, inners) if len(kernels) > 1 else []
        ):
            end = start + k.num_blocks
            if i == len(kernels) - 1:
                code.writeline("else:")
            else:
                code.writeline(f"{'if' if i == 0 else 'elif'} _bid < {end}:")
            with code.indent():
                code.writeline(f"_b = _bid - {start}" if start else "_b = _bid")
                code.splice(inner)
            start = end
    code.splice(_module_epilogue(arg_names, start))
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

    def _combo_groups(self, subkernel_nodes) -> list[list[Any]]:
        """Split the sub-kernels so no launch takes more than COMBO_MAX_ARGS buffers."""
        groups: list[list[Any]] = []
        names: OrderedSet[str] = OrderedSet()
        for pn in subkernel_nodes:
            rw = pn.read_writes
            mine = OrderedSet(d.name for d in (*rw.reads, *rw.writes))
            if groups and len(names | mine) <= COMBO_MAX_ARGS:
                groups[-1].append(pn)
                names |= mine
            else:
                groups.append([pn])
                names = mine
        return groups

    def _codegen_flydsl_combo(self, combo_kernel_node) -> None:
        from ..common import KernelArgs
        from ..simd_kernel_features import SIMDKernelFeatures

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
        if "z" in preferred or kernel_features.is_reduction():
            raise FlyDSLUnsupported("transposed access needs a 2D tile")
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

    def _define_flydsl_kernel(self, src_code: str, node_schedule) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]
        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )
        h = hashlib.sha256(src_code.encode("utf-8")).hexdigest()[:8]
        kernel_name = (
            f"flydsl_{fused_name}_{h}"
            if fused_name and fused_name != "fused"
            else f"flydsl_{h}"
        )
        wrapper.src_to_kernel[src_code] = kernel_name
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)
        _, _, kernel_path = get_path(code_hash(src_code), "py")
        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"async_compile.flydsl({kernel_name!r}, r'''")
        compile_wrapper.splice(src_code, strip=True)
        compile_wrapper.writeline("''', precompile_metadata={})")
        origins, detailed = get_kernel_metadata(node_schedule, wrapper)
        wrapper.define_kernel(
            kernel_name,
            compile_wrapper.getvalue(),
            f"# kernel path: {kernel_path}\n{origins}\n{detailed}",
        )
        return kernel_name


@functools.cache
def flydsl_simd_available() -> bool:
    try:
        # pyrefly: ignore [missing-import]
        import flydsl.compiler  # noqa: F401
    except ImportError:
        return False
    return bool(torch.version.hip)


del BackendFeature, OpsWrapper, Identity

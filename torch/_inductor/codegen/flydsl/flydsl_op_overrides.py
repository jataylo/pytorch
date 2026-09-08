# mypy: allow-untyped-defs
"""Map Inductor pointwise ops onto FlyDSL source text.

The FlyDSL analog of ``cutedsl_op_overrides.py``, and much thinner: FlyDSL mods are scalar
valued -- one score at a time, with the vectorized ABI handled by lifting the whole mod
rather than by widening each op -- so every op here is a plain call on
``flydsl_mod_runtime`` and the type promotion lives there.

Nothing in this file may import ``flydsl``: it runs during lowering whether or not FlyDSL
is installed, and only produces strings. Hence :data:`UNSUPPORTED_OPS` living here rather
than beside the shim it describes.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from ..common import CSEVariable, OpOverrides


if TYPE_CHECKING:
    from collections.abc import Sequence

    import sympy


# Module alias the generated kernel binds for the runtime shim.
RUNTIME_ALIAS = "fdu"
RUNTIME_MODULE = "torch._inductor.codegen.flydsl.flydsl_mod_runtime"

# Ops that do not lower to AMDGPU through FlyDSL and have no expansion in the shim.
# Codegen raises on these so the failure names the op, instead of surfacing as an LLVM
# "no libcall available" error.
#
# The rest of this family *is* expanded in the shim, over the exp2/log2/sqrt primitives
# that do lower: `tanh` and `sigmoid` first, and since then `sinh`, `cosh`, `asinh`,
# `acosh`, `atanh`, `atan`, `atan2`, `asin`, `acos`, `erf` and `erfc`. What is left needs
# a long rational approximation rather than a few terms, and none of the three is
# plausible in an attention score_mod:
#
#   erfinv   inverse error function; two-branch rational fit
#   lgamma   log-gamma; Lanczos series plus a reflection for x < 0.5
#   digamma  gamma's logarithmic derivative; asymptotic series plus recurrence
#
# Adding any of them is a matter of transcribing coefficients into the shim next to
# `erf`, not of kernel work, if a real mod ever wants one.
UNSUPPORTED_OPS = frozenset(
    {
        "erfinv",
        "lgamma",
        "digamma",
    }
)


class FlyDSLCSEVariable(CSEVariable):
    """A named FlyDSL device value.

    ``index_expr`` records the semantic index a coordinate came from, so the vectorized
    aux read can tell that ``kv_idx`` for lane i is ``kv_base + i`` and collapse per-lane
    gathers into one wide load. A value of unknown provenance is gathered per lane.
    """

    def __init__(self, name, bounds, dtype=None, shape=None) -> None:
        super().__init__(name, bounds, dtype)
        self.index_expr: sympy.Expr | None = None
        self.shape = shape

    def __repr__(self) -> str:
        return str(self.name)


def _arg(x: Any) -> str:
    """Render an operand as FlyDSL source."""
    if isinstance(x, CSEVariable):
        return str(x.name)
    if isinstance(x, bool):
        return f"{RUNTIME_ALIAS}.const_bool({x})"
    if isinstance(x, int):
        return f"{RUNTIME_ALIAS}.const_i32({x})"
    if isinstance(x, float):
        return f"{RUNTIME_ALIAS}.const_f32({x!r})"
    return str(x)


# Ops whose result is an i1 regardless of operand types.
_BOOL_RESULT = frozenset(
    {
        "lt",
        "le",
        "gt",
        "ge",
        "eq",
        "ne",
        "logical_and",
        "logical_or",
        "logical_xor",
        "logical_not",
    }
)

# Ops that are float-valued whatever they are given.
_FLOAT_RESULT = frozenset(
    {
        "truediv",
        "reciprocal",
        "sqrt",
        "rsqrt",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log2",
        "log10",
        "log1p",
        "sin",
        "cos",
        "tanh",
        "sigmoid",
        "pow",
        "floor",
        "ceil",
        "trunc",
        "round",
    }
)


def _result_dtype(fn: str, args: Sequence[Any]) -> torch.dtype:
    if fn in _BOOL_RESULT:
        return torch.bool
    if fn in _FLOAT_RESULT:
        return torch.float32
    for a in args:
        if isinstance(a, CSEVariable):
            if a.dtype is not None and a.dtype.is_floating_point:
                return torch.float32
        elif isinstance(a, float):
            return torch.float32
    return torch.int32


def _call(fn: str, *args: Any) -> Any:
    """Emit a shim call, bound to a temporary.

    Without the binding every op inlines into its operands, so a mod of any size renders as
    one unreadable expression and a repeated subexpression is traced once per use.
    """
    from ...virtualized import V

    expr = f"{RUNTIME_ALIAS}.{fn}({', '.join(_arg(a) for a in args)})"
    kernel = getattr(V, "kernel", None)
    if kernel is None or not hasattr(kernel, "cse"):
        return expr
    from ..common import ValueRanges

    return kernel.cse.generate(
        kernel.body,
        expr,
        dtype=_result_dtype(fn, args),
        bounds=ValueRanges.unknown(),
    )


class FlyDSLOpOverrides(OpOverrides):
    """Emit FlyDSL source for the pointwise ops a score_mod/mask_mod can contain."""

    TORCH_TO_FLYDSL_DTYPE = {
        torch.float32: "fx.Float32",
        torch.float16: "fx.Float16",
        torch.bfloat16: "fx.BFloat16",
        torch.int64: "fx.Int64",
        torch.int32: "fx.Int32",
        torch.int16: "fx.Int16",
        torch.int8: "fx.Int8",
        torch.bool: "fx.Boolean",
    }

    # Ops whose only argument list is (x) and which map 1:1 onto a shim function.
    _UNARY = (
        "neg",
        "abs",
        "reciprocal",
        "sqrt",
        "rsqrt",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log2",
        "log10",
        "log1p",
        "floor",
        "ceil",
        "trunc",
        "round",
        "sin",
        "cos",
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
        "relu",
        "logical_not",
        "bitwise_not",
    )

    # Ops whose argument list is (a, b) and which map 1:1 onto a shim function.
    _BINARY = (
        "add",
        "sub",
        "mul",
        "truediv",
        "floordiv",
        "mod",
        "remainder",
        "maximum",
        "minimum",
        "pow",
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
        "atan2",
    )

    @staticmethod
    def constant(value: bool | float | int, dtype: torch.dtype) -> str:
        # Emitted directly rather than through _call, which would wrap the literal in a
        # second constructor.
        if dtype == torch.bool:
            return f"{RUNTIME_ALIAS}.const_bool({bool(value)})"
        if dtype.is_floating_point:
            # A score_mod masks by returning -inf, so the literal has to survive as a real
            # float rather than a name Python cannot evaluate.
            return f"{RUNTIME_ALIAS}.const_f32(float({str(float(value))!r}))"
        return f"{RUNTIME_ALIAS}.const_i32({int(value)})"

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> str:
        from ...virtualized import V

        rendered = V.kernel.kexpr(V.kernel.rename_indexing(expr))
        return _call("to_i32", rendered)

    @staticmethod
    def to_dtype(
        x: Any,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> str:
        if dtype == torch.bool:
            return _call("to_bool", x)
        if dtype.is_floating_point:
            return _call("to_f32", x)
        return _call("to_i32", x)

    @staticmethod
    def to_dtype_bitcast(x: Any, dtype: torch.dtype, src_dtype: torch.dtype) -> str:
        raise NotImplementedError(
            "bitcast is not supported in FlyDSL flex mods; it has no use in a "
            "score_mod and would need an explicit ArithValue.bitcast"
        )

    @staticmethod
    def where(cond: Any, a: Any, b: Any) -> str:
        return _call("where", cond, a, b)

    @staticmethod
    def masked(mask: Any, body: Any, other: Any) -> str:
        raise NotImplementedError("masked loads are not supported in FlyDSL flex mods")

    @staticmethod
    def indirect_indexing(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "indirect indexing is not supported in FlyDSL flex mods"
        )


def _make_unary(name: str):
    @staticmethod  # type: ignore[misc]
    def fn(x):
        return _call(name, x)

    fn.__name__ = name  # type: ignore[attr-defined]
    return fn


def _make_binary(name: str):
    @staticmethod  # type: ignore[misc]
    def fn(a, b):
        return _call(name, a, b)

    fn.__name__ = name  # type: ignore[attr-defined]
    return fn


def _make_unsupported(name: str):
    @staticmethod  # type: ignore[misc]
    def fn(*args, **kwargs):
        raise NotImplementedError(
            f"`{name}` does not lower to AMDGPU through FlyDSL and has no expansion in "
            f"{RUNTIME_MODULE}. Rewrite the mod without it, or add an expansion there."
        )

    fn.__name__ = name  # type: ignore[attr-defined]
    return fn


def _install_ops() -> None:
    for _name in FlyDSLOpOverrides._UNARY:
        setattr(FlyDSLOpOverrides, _name, _make_unary(_name))
    for _name in FlyDSLOpOverrides._BINARY:
        setattr(FlyDSLOpOverrides, _name, _make_binary(_name))
    for _name in UNSUPPORTED_OPS:
        setattr(FlyDSLOpOverrides, _name, _make_unsupported(_name))


_install_ops()

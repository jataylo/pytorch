# mypy: allow-untyped-defs
"""Map Inductor pointwise ops onto FlyDSL source text.

This is the FlyDSL analog of ``cutedsl_op_overrides.py``, and it is deliberately much
thinner. CuteDSL's overrides carry a lot of TensorSSA machinery because CuTe ops are
whole-fragment operations that need shape and lane bookkeeping. FlyDSL mods are scalar
valued -- one score at a time, with the vectorized ABI handled by lifting the whole mod
rather than by widening each op -- so every op here is a plain function call on
``flydsl_mod_runtime`` and all the type-promotion trouble lives there.

Nothing in this file may import ``flydsl``: it runs during Inductor lowering, on any
machine, whether or not FlyDSL is installed. It only produces strings. That is also why
:data:`UNSUPPORTED_OPS` is declared here rather than beside the shim it describes --
codegen has to know what the shim cannot do without importing it.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import sympy

import torch

from ..common import CSEVariable, OpOverrides


if TYPE_CHECKING:
    from collections.abc import Sequence


# Module alias the generated kernel binds for the runtime shim.
RUNTIME_ALIAS = "fdu"
RUNTIME_MODULE = "torch._inductor.codegen.flydsl.flydsl_mod_runtime"

# Ops a score_mod or mask_mod could contain that do not lower to AMDGPU through FlyDSL
# and have no expansion in the shim yet. Codegen raises on these so the failure names the
# op, instead of surfacing as an LLVM "no libcall available" error with no context.
#
# `tanh` is deliberately absent: it is in the same category but soft-cap is too common to
# reject, so the shim expands it over exp2.
UNSUPPORTED_OPS = frozenset(
    {
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "asinh",
        "acosh",
        "atanh",
        "erf",
        "erfc",
        "erfinv",
        "lgamma",
        "digamma",
    }
)


class FlyDSLCSEVariable(CSEVariable):
    """A named FlyDSL device value.

    ``index_expr`` records the semantic index a value came from when it is a coordinate
    rather than an opaque number. The vectorized aux-tensor read needs to know that
    ``kv_idx`` for lane i is ``kv_base + i`` in order to collapse per-lane gathers into
    one wide load; a value whose provenance is unknown has to be gathered per lane.
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
    """Emit a shim call, bound to a temporary so the generated body stays readable.

    Without this every op inlines into its operands and a mod of any size renders as one
    unreadable expression. Binding each step also means a repeated subexpression is
    traced once rather than once per use.
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
    )

    @staticmethod
    def constant(value: bool | float | int, dtype: torch.dtype) -> str:
        # Emitted directly rather than through _call: the value is already a Python
        # literal, and _call would render it as a constructor and wrap that again.
        if dtype == torch.bool:
            return f"{RUNTIME_ALIAS}.const_bool({bool(value)})"
        if dtype.is_floating_point:
            # A score_mod masks by returning -inf, so the literal has to survive as a
            # real float rather than a name Python cannot evaluate.
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


def rendered_ops() -> Sequence[str]:
    """Op names this backend can lower. Used by tests and by the coverage audit."""
    return tuple(
        sorted(
            set(FlyDSLOpOverrides._UNARY)
            | set(FlyDSLOpOverrides._BINARY)
            | {"constant", "index_expr", "to_dtype", "where"}
        )
    )

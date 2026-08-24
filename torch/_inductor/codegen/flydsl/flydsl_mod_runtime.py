# mypy: allow-untyped-defs
"""Runtime helpers that FlyDSL-generated FlexAttention mods call into.

Generated kernels import this module (conventionally as ``fdu``), so it is the *only*
file in the FlyDSL codegen package that may import ``flydsl``. Everything else emits
source text and must stay importable without it.

Why a shim instead of emitting FlyDSL expressions inline:

- **Type promotion.** A lowered ``score_mod`` freely mixes the f32 score with the i32
  coordinates. Inductor inserts explicit casts in most places but not all, so operands
  are promoted here rather than trusting the FX graph to have been tidy.
- **Ops that do not lower.** ``math.tanh`` has no AMDGPU libcall in FlyDSL's pipeline
  and fails at LLVM time with "no libcall available for ftanh". Soft-cap is one of the
  most common score_mods there is, so the expansion has to live somewhere; putting it in
  a normal Python module means it can be unit-tested instead of being a string template.
- **Debuggability.** Generated code reads as ``fdu.mul(tmp0, tmp1)`` rather than three
  nested constructor calls, which matters when inspecting a failing kernel.
"""

from __future__ import annotations

import math as _math

import flydsl.expr as fx
from flydsl.expr import math as fmath
from flydsl.expr.numeric import Float as _Float, Integer as _Integer
from flydsl.expr.utils.arith import ArithValue

from .flydsl_op_overrides import UNSUPPORTED_OPS as UNSUPPORTED_OPS


_LOG2E = _math.log2(_math.e)
_LN2 = 1.0 / _LOG2E
_LN10 = _math.log(10.0)


# ── type coercion ─────────────────────────────────────────────────────────────


def const_f32(v):
    return fx.Float32(float(v))


def const_i32(v):
    return fx.Int32(int(v))


def const_bool(v):
    # There is no i1 literal constructor that round-trips a Python bool, so make one
    # from a comparison the folder can collapse.
    one = fx.Int32(1)
    return ArithValue(one == fx.Int32(1 if v else 0))


def to_f32(x):
    """Convert anything a mod can hold to f32.

    An integer needs a *typed wrapper* on the way, not just signedness on the
    `ArithValue`: `fx.Float32` rejects a bare MLIR integer outright ("bare signless
    integer cannot be promoted to float") because it cannot tell sitofp from uitofp, and
    it reads signedness off `fx.Int32`'s class rather than off the value. Every operator
    result comes back bare, so without the round trip through `fx.Int32` any arithmetic
    on coordinates fails the moment it meets a float.
    """
    if isinstance(x, (bool, int, float)):
        return fx.Float32(float(x))
    if isinstance(x, (_Integer, _Float)):
        return fx.Float32(x)
    av = ArithValue(x)
    if _is_bool(av):
        return fx.Float32(av.select(const_f32(1.0), const_f32(0.0)))
    if not av.is_float:
        return fx.Float32(fx.Int32(av))
    return fx.Float32(av)


def to_i32(x):
    if isinstance(x, (bool, int, float)):
        return fx.Int32(int(x))
    if isinstance(x, (_Integer, _Float)):
        return fx.Int32(x)
    av = ArithValue(x)
    if _is_bool(av):
        # Widening an i1 by select avoids constructing an MLIR type here.
        return fx.Int32(av.select(const_i32(1), const_i32(0)))
    return fx.Int32(av)


def to_bool(x):
    """Non-zero test. Already-i1 values pass through."""
    if isinstance(x, (bool, int, float)):
        return const_bool(bool(x))
    av = ArithValue(x)
    if _is_bool(av):
        return av
    zero = const_f32(0.0) if av.is_float else const_i32(0)
    return ArithValue(av != zero)


def _is_bool(av) -> bool:
    try:
        return not av.is_float and av.type.width == 1
    except AttributeError:
        return False


def _is_float(x) -> bool:
    if isinstance(x, float):
        return True
    if isinstance(x, (bool, int)):
        return False
    try:
        return ArithValue(x).is_float
    except (TypeError, ValueError):
        return False


def _both_bool(a, b) -> bool:
    try:
        return _is_bool(ArithValue(a)) and _is_bool(ArithValue(b))
    except (TypeError, ValueError):
        return False


def _promote2(a, b):
    """Bring two operands to a common arithmetic type (f32 if either is float).

    Two i1 values are left alone. A mask_mod combines its clauses with ``&`` and ``|``,
    which Inductor lowers to bitwise ops; widening those to i32 would make the mod return
    an integer, and the kernel's ``select`` then rejects it as not bool-like.
    """
    if _both_bool(a, b):
        return ArithValue(a), ArithValue(b)
    if _is_float(a) or _is_float(b):
        return ArithValue(to_f32(a)), ArithValue(to_f32(b))
    return ArithValue(to_i32(a)), ArithValue(to_i32(b))


# ── binary arithmetic ─────────────────────────────────────────────────────────


def add(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x + y)


def sub(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x - y)


def mul(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x * y)


def truediv(a, b):
    # Division is float even when both operands are integral, matching Python/ATen.
    x, y = ArithValue(to_f32(a)), ArithValue(to_f32(b))
    return ArithValue(x / y)


def floordiv(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x // y)


def mod(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x % y)


def remainder(a, b):
    """Python/ATen remainder: sign follows the divisor, unlike C's."""
    x, y = _promote2(a, b)
    r = ArithValue(x % y)
    zero = const_f32(0.0) if r.is_float else const_i32(0)
    wrong_sign = ArithValue(ArithValue(r != zero) & ArithValue((r < zero) != (y < zero)))
    return ArithValue(wrong_sign.select(ArithValue(r + y), r))


def maximum(a, b):
    x, y = _promote2(a, b)
    if x.is_float:
        return ArithValue(x.maximumf(y))
    return ArithValue(ArithValue(x > y).select(x, y))


def minimum(a, b):
    x, y = _promote2(a, b)
    return ArithValue(ArithValue(x < y).select(x, y))


def pow(a, b):  # noqa: A001 - mirrors the ops-handler name
    x, y = ArithValue(to_f32(a)), ArithValue(to_f32(b))
    return ArithValue(fmath.powf(x, y))


# ── comparison and logic ──────────────────────────────────────────────────────


def lt(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x < y)


def le(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x <= y)


def gt(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x > y)


def ge(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x >= y)


def eq(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x == y)


def ne(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x != y)


def logical_and(a, b):
    return ArithValue(to_bool(a) & to_bool(b))


def logical_or(a, b):
    return ArithValue(to_bool(a) | to_bool(b))


def logical_xor(a, b):
    return ArithValue(to_bool(a) ^ to_bool(b))


def logical_not(a):
    return ArithValue(to_bool(a) == const_bool(False))


def bitwise_and(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x & y)


def bitwise_or(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x | y)


def bitwise_xor(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x ^ y)


def bitwise_not(a):
    return ArithValue(~ArithValue(to_i32(a)))


def bitwise_left_shift(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x << y)


def bitwise_right_shift(a, b):
    x, y = _promote2(a, b)
    return ArithValue(x >> y)


def where(cond, a, b):
    x, y = _promote2(a, b)
    return ArithValue(to_bool(cond).select(x, y))


# ── unary ─────────────────────────────────────────────────────────────────────


def neg(x):
    av = ArithValue(x)
    return ArithValue(-av)


def abs(x):  # noqa: A001 - mirrors the ops-handler name
    av = ArithValue(x)
    if av.is_float:
        return ArithValue(fmath.absf(av))
    return ArithValue(fmath.absi(av))


def reciprocal(x):
    return ArithValue(const_f32(1.0) / ArithValue(to_f32(x)))


def sqrt(x):
    return ArithValue(fmath.sqrt(to_f32(x)))


def rsqrt(x):
    return ArithValue(fmath.rsqrt(to_f32(x)))


def exp2(x):
    return ArithValue(ArithValue(to_f32(x)).exp2())


def exp(x):
    # exp2 is the native instruction; exp would go through a libcall.
    return exp2(mul(x, const_f32(_LOG2E)))


def expm1(x):
    return sub(exp(x), const_f32(1.0))


def log2(x):
    return ArithValue(fmath.log2(to_f32(x)))


def log(x):
    return mul(log2(x), const_f32(_LN2))


def log10(x):
    return mul(log2(x), const_f32(_LN2 / _LN10))


def log1p(x):
    return log(add(x, const_f32(1.0)))


def floor(x):
    return ArithValue(fmath.floor(to_f32(x)))


def ceil(x):
    return ArithValue(fmath.ceil(to_f32(x)))


def trunc(x):
    return ArithValue(fmath.trunc(to_f32(x)))


def round(x):  # noqa: A001 - mirrors the ops-handler name
    return ArithValue(fmath.roundeven(to_f32(x)))


def sin(x):
    return ArithValue(fmath.sin(to_f32(x)))


def cos(x):
    return ArithValue(fmath.cos(to_f32(x)))


def tanh(x):
    """tanh expanded over exp2, because ftanh has no AMDGPU libcall here.

    Evaluated on |x| so the exponential cannot overflow, then the sign is restored:
        tanh(x) = sign(x) * (1 - 2 / (exp2(2*|x|*log2e) + 1))
    """
    av = ArithValue(to_f32(x))
    is_neg = ArithValue(av < const_f32(0.0))
    ax = ArithValue(is_neg.select(ArithValue(-av), av))
    e = ArithValue(ArithValue(ax * const_f32(2.0 * _LOG2E)).exp2())
    t = ArithValue(
        const_f32(1.0) - ArithValue(const_f32(2.0) / ArithValue(e + const_f32(1.0)))
    )
    return ArithValue(is_neg.select(ArithValue(-t), t))


def sigmoid(x):
    """1 / (1 + exp(-x)), via exp2 for the same reason as tanh."""
    e = exp2(mul(neg(x), const_f32(_LOG2E)))
    return ArithValue(const_f32(1.0) / ArithValue(ArithValue(e) + const_f32(1.0)))


def relu(x):
    return maximum(x, const_f32(0.0))


def identity(x):
    return x

# mypy: allow-untyped-defs
"""Runtime helpers that FlyDSL-generated FlexAttention mods call into.

Generated kernels import this module (conventionally as ``fdu``), so it is the *only* file
in the FlyDSL codegen package that may import ``flydsl``; everything else emits source text
and must stay importable without it.

A shim rather than inline FlyDSL expressions because a lowered score_mod freely mixes the
f32 score with i32 coordinates and Inductor does not cast every one of them, because some
ops need expanding (``tanh`` has no AMDGPU libcall here), and because ``fdu.mul(tmp0,
tmp1)`` is easier to read in a failing kernel than nested constructors.
"""

from __future__ import annotations

import math as _math

import flydsl.expr as fx
from flydsl.expr import math as fmath
from flydsl.expr.numeric import Float as _Float, Integer as _Integer
from flydsl.expr.utils.arith import ArithValue


_LOG2E = _math.log2(_math.e)
_LN2 = 1.0 / _LOG2E
_LN10 = _math.log(10.0)
_PI = _math.pi
_PI_2 = _math.pi / 2.0


# Type coercion


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

    An integer needs the round trip through ``fx.Int32``, not just signedness on the
    ``ArithValue``: ``fx.Float32`` rejects a bare MLIR integer ("bare signless integer
    cannot be promoted to float") because it reads signedness off the class rather than the
    value, and every operator result comes back bare.
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

    Two i1 values are left alone: a mask_mod combines its clauses with ``&`` and ``|``, and
    widening those to i32 would make the mod return an integer that the kernel's ``select``
    then rejects as not bool-like.
    """
    if _both_bool(a, b):
        return ArithValue(a), ArithValue(b)
    if _is_float(a) or _is_float(b):
        return ArithValue(to_f32(a)), ArithValue(to_f32(b))
    return ArithValue(to_i32(a)), ArithValue(to_i32(b))


# Binary arithmetic


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
    wrong_sign = ArithValue(
        ArithValue(r != zero) & ArithValue((r < zero) != (y < zero))
    )
    return ArithValue(wrong_sign.select(ArithValue(r + y), r))


def maximum(a, b):
    x, y = _promote2(a, b)
    if x.is_float:
        return ArithValue(x.maximumf(y))
    return ArithValue(ArithValue(x > y).select(x, y))


def minimum(a, b):
    x, y = _promote2(a, b)
    return ArithValue(ArithValue(x < y).select(x, y))


def pow(a, b):  # shadows the builtin: the name is fixed by the ops handler
    x, y = ArithValue(to_f32(a)), ArithValue(to_f32(b))
    return ArithValue(fmath.powf(x, y))


# Comparison and logic


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


# Unary


def neg(x):
    av = ArithValue(x)
    return ArithValue(-av)


def abs(x):  # shadows the builtin: the name is fixed by the ops handler
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


def round(x):  # shadows the builtin: the name is fixed by the ops handler
    return ArithValue(fmath.roundeven(to_f32(x)))


def sin(x):
    return ArithValue(fmath.sin(to_f32(x)))


def cos(x):
    return ArithValue(fmath.cos(to_f32(x)))


def tanh(x):
    """tanh expanded over exp2, because ftanh has no AMDGPU libcall here.

    Evaluated on |x| so the exponential cannot overflow, then the sign is restored:
    ``tanh(x) = sign(x) * (1 - 2 / (exp2(2*|x|*log2e) + 1))``.
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


def _horner(z, coeffs):
    """Evaluate a polynomial in ``z`` from the highest-order coefficient down."""
    acc = const_f32(coeffs[0])
    for c in coeffs[1:]:
        acc = add(mul(acc, z), const_f32(c))
    return acc


def sinh(x):
    """(e^|x| - e^-|x|) / 2, with the sign restored afterwards.

    Folding to |x| first keeps the growing exponential the one that is evaluated, the
    same reason ``tanh`` does it. sinh overflows f32 near |x| = 89 either way; that is
    inherent to the function, not to this expansion.
    """
    e = exp(abs(x))
    half = mul(sub(e, reciprocal(e)), const_f32(0.5))
    return where(lt(x, const_f32(0.0)), neg(half), half)


def cosh(x):
    """(e^|x| + e^-|x|) / 2. Even, so no sign to restore."""
    e = exp(abs(x))
    return mul(add(e, reciprocal(e)), const_f32(0.5))


def asinh(x):
    """sign(x) * log(|x| + sqrt(x^2 + 1)).

    Taken on |x| so the sum never cancels: for large negative x the direct form
    ``log(x + sqrt(x*x + 1))`` subtracts two nearly equal numbers.
    """
    ax = abs(x)
    r = log(add(ax, sqrt(add(mul(ax, ax), const_f32(1.0)))))
    return where(lt(x, const_f32(0.0)), neg(r), r)


def acosh(x):
    """log(x + sqrt(x^2 - 1)), defined for x >= 1."""
    return log(add(x, sqrt(sub(mul(x, x), const_f32(1.0)))))


def atanh(x):
    """0.5 * log((1 + x) / (1 - x)), defined for |x| < 1."""
    return mul(
        log(truediv(add(const_f32(1.0), x), sub(const_f32(1.0), x))), const_f32(0.5)
    )


# Hastings' odd minimax polynomial for atan on |x| <= 1, max abs error ~1e-5. Far tighter
# than needed: a score_mod's result is consumed by an f32 softmax after a bf16 round trip,
# so the function's own error is nowhere near the limiting term.
_ATAN_COEFFS = (0.0208351, -0.0851330, 0.1801410, -0.3302995, 0.9998660)


def _atan_unit(x):
    """atan(x) for |x| <= 1."""
    return mul(x, _horner(mul(x, x), _ATAN_COEFFS))


def atan(x):
    """atan over the full range, by reflecting |x| > 1 through atan(x) = pi/2 - atan(1/x)."""
    ax = abs(x)
    big = gt(ax, const_f32(1.0))
    # Both branches are evaluated, so keep the reciprocal finite at x = 0 where it is
    # not selected: the guarded argument is 1/max(|x|, 1) rather than 1/|x|.
    inner = where(big, reciprocal(maximum(ax, const_f32(1.0))), ax)
    r = _atan_unit(inner)
    r = where(big, sub(const_f32(_PI_2), r), r)
    return where(lt(x, const_f32(0.0)), neg(r), r)


def atan2(y, x):
    """Quadrant-correct atan(y/x).

    ``atan(y/x)`` alone collapses the second and third quadrants onto the first and
    fourth, so x < 0 needs +/-pi added back with the sign taken from y. x == 0 is
    resolved to +/-pi/2 rather than left to the division.
    """
    ratio = truediv(y, where(eq(x, const_f32(0.0)), const_f32(1.0), x))
    base = atan(ratio)
    y_neg = lt(y, const_f32(0.0))
    shifted = add(base, where(y_neg, const_f32(-_PI), const_f32(_PI)))
    on_axis = where(y_neg, const_f32(-_PI_2), const_f32(_PI_2))
    return where(
        eq(x, const_f32(0.0)),
        on_axis,
        where(lt(x, const_f32(0.0)), shifted, base),
    )


def asin(x):
    """atan2(x, sqrt(1 - x^2)), which stays finite at |x| = 1 where atan(x/0) would not."""
    return atan2(x, sqrt(maximum(sub(const_f32(1.0), mul(x, x)), const_f32(0.0))))


def acos(x):
    """pi/2 - asin(x)."""
    return sub(const_f32(_PI_2), asin(x))


# Abramowitz & Stegun 7.1.26, max abs error 1.5e-7 on x >= 0.
_ERF_P = 0.3275911
_ERF_COEFFS = (1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592)


def erf(x):
    """A&S 7.1.26 on |x|, with erf(-x) = -erf(x).

    ``1 - poly(t) * exp(-x^2)`` where ``t = 1 / (1 + p|x|)``. Evaluated on |x| because
    the approximation is only valid for non-negative argument.
    """
    ax = abs(x)
    t = reciprocal(add(const_f32(1.0), mul(const_f32(_ERF_P), ax)))
    poly = mul(t, _horner(t, _ERF_COEFFS))
    r = sub(const_f32(1.0), mul(poly, exp(neg(mul(ax, ax)))))
    return where(lt(x, const_f32(0.0)), neg(r), r)


def erfc(x):
    """1 - erf(x).

    Cancels in the right tail, where erf approaches 1 and erfc is the small quantity, so
    this carries absolute rather than relative accuracy out there. Adequate for a
    score_mod, which feeds a softmax that is itself shift-invariant; not adequate as a
    general erfc.
    """
    return sub(const_f32(1.0), erf(x))


def relu(x):
    return maximum(x, const_f32(0.0))


def identity(x):
    return x

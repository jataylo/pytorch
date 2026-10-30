# mypy: ignore-errors
"""Device-side helpers for FlyDSL kernels generated from Inductor pointwise/reduction IR.

Generated SIMD kernels import this module as ``fdp``. A general Inductor kernel carries
every dtype, so values here live in one of five *compute kinds*:

    b1   bool                      (i1)
    i32  int8/int16/uint8/int32    (sign- or zero-extended at load, truncated at store)
    i64  int64
    f32  float16/bfloat16/float32  (widened at load, rounded at store)
    f64  float64

Storage kinds are the tensor dtypes themselves and only appear at loads and stores.
Narrow integers compute in i32: add/sub/mul/bitwise/shift-left agree with the narrow type
once truncated at the store, and compares/division agree because the operands were
extended with the right signedness.

Everything that is not plain arithmetic is branch-free on purpose. Helpers called from a
``@flyc.kernel`` body are not AST-rewritten, so a dynamic ``if`` here would be a Python
``bool()`` of a device value. Masking therefore goes through the buffer instructions
(a false mask swaps the offset for one past ``num_records`` and the hardware drops the
access), and cross-lane reductions are written so every lane computes the same answer.
"""

from __future__ import annotations

import contextlib
import math as _math
import os as _os
import re as _re
import time as _time

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import (
    arith as _arith,
    rocdl as _rocdl,
    scf as _scf,
    vector as _vector,
)
from flydsl._mlir.extras import types as T
from flydsl.expr import gpu, math as fmath
from flydsl.expr.utils.arith import ArithValue

from torch.utils._ordered_set import OrderedSet


WAVE = 64
_LOG2E = 1.4426950408889634
_LN2 = 1.0 / _LOG2E
_LN10 = _math.log(10.0)

# Abramowitz & Stegun 7.1.26, max abs error 1.5e-7 on x >= 0.
_ERF_P = 0.3275911
_ERF_COEFFS = (1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592)


def _cf32(v):
    return fx.Float32(float(v))


# ---------------------------------------------------------------------------------------
# Kinds and conversion
# ---------------------------------------------------------------------------------------

_STORAGE_IR = {
    "f32": T.f32,
    "f16": T.f16,
    "bf16": T.bf16,
    "f64": T.f64,
    "i64": T.i64,
    "i32": T.i32,
    "i16": T.i16,
    "i8": T.i8,
    "u8": T.i8,
    "u16": T.i16,
    "u32": T.i32,
    "b1": T.i8,  # torch.bool is a byte in memory
    # FP8 (gfx942's FNUZ formats) is a byte in memory, converted by the hardware's
    # v_cvt instructions at the load or store; values in between are f32.
    "f8e4": T.i8,
    "f8e5": T.i8,
}
_FP8 = {
    "f8e4": (_rocdl.CvtF32Fp8Op, _rocdl.CvtPkFp8F32Op),
    "f8e5": (_rocdl.CvtF32Bf8Op, _rocdl.CvtPkBf8F32Op),
}

# How a storage kind is laid out in memory, where that differs from the kind itself:
# torch.bool is a byte, so a b1 value is widened to u8 on store and compared on load.
_MEM_KIND = {"b1": "u8"}

_COMPUTE_OF = {
    "f32": "f32",
    "f16": "f32",
    "bf16": "f32",
    "f64": "f64",
    "i64": "i64",
    "i32": "i32",
    "i16": "i32",
    "i8": "i32",
    "u8": "i32",
    "u16": "i32",
    # Past i32's range, so computed in i64; uint64 has no compute kind.
    "u32": "i64",
    "b1": "b1",
    "f8e4": "f32",
    "f8e5": "f32",
}

_INT_KINDS = ("b1", "i8", "u8", "i16", "u16", "i32", "u32", "i64")
_UNSIGNED_KINDS = ("b1", "u8", "u16", "u32")

_COMPUTE_IR = {"b1": T.bool, "i32": T.i32, "i64": T.i64, "f32": T.f32, "f64": T.f64}

_RANK = {"b1": 0, "i32": 1, "i64": 2, "f32": 3, "f64": 4}


def _val(x):
    """Bare ``ir.Value`` behind anything the generated code can hold."""
    if isinstance(x, ir.Value):
        return x
    if isinstance(x, Aff):
        return _val(x.value())
    if hasattr(x, "ir_value"):
        return x.ir_value()
    return ArithValue(x)


def _isa(t, cls) -> bool:
    # Newer MLIR bindings downcast types and dropped `Type.isinstance`; older ones need it.
    check = getattr(cls, "isinstance", None)
    return check(t) if check is not None else isinstance(t, cls)


def _int_width(t) -> int:
    return t.width if hasattr(t, "width") else ir.IntegerType(t).width


def kind_of_type(t) -> str:
    if _isa(t, ir.IntegerType):
        return {1: "b1", 8: "i8", 16: "i16", 32: "i32", 64: "i64"}[_int_width(t)]
    if _isa(t, ir.F32Type):
        return "f32"
    if _isa(t, ir.F64Type):
        return "f64"
    if _isa(t, ir.F16Type):
        return "f16"
    if _isa(t, ir.BF16Type):
        return "bf16"
    if _isa(t, ir.IndexType):
        return "index"
    raise TypeError(f"FlyDSL SIMD runtime: unsupported value type {t}")


def kind(x) -> str:
    if isinstance(x, bool):
        return "b1"
    if isinstance(x, int):
        return "i64" if not (-(2**31) <= x < 2**31) else "i32"
    if isinstance(x, float):
        return "f32"
    return kind_of_type(_val(x).type)


def const(v, k: str):
    """A scalar constant of compute kind ``k``."""
    if k == "b1":
        return ArithValue(
            _arith.ConstantOp(
                T.bool(), ir.IntegerAttr.get(T.bool(), int(bool(v)))
            ).result
        )
    if k in ("i32", "i64"):
        t = _COMPUTE_IR[k]()
        return ArithValue(_arith.ConstantOp(t, ir.IntegerAttr.get(t, int(v))).result)
    t = _COMPUTE_IR[k]()
    return ArithValue(_arith.ConstantOp(t, ir.FloatAttr.get(t, float(v))).result)


def _f32_to_bf16(v):
    """Round to nearest even, NaN to the canonical NaN, as Triton converts: LLVM's
    ``fptrunc`` keeps the NaN's payload instead, an extra ``v_or`` an element -- a tenth
    of a bf16 softmax backward's vector-ALU work."""

    def i32(c):
        return _arith.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), c)).result

    bits = _arith.BitcastOp(T.i32(), v).result
    lsb = _arith.AndIOp(_arith.ShRUIOp(bits, i32(16)).result, i32(1)).result
    rounded = _arith.AddIOp(_arith.AddIOp(bits, i32(0x7FFF)).result, lsb).result
    nan = _arith.CmpFOp(_arith.CmpFPredicate.UNO, v, v).result
    bits = _arith.SelectOp(nan, i32(0x7FC00000), rounded).result
    half = _arith.TruncIOp(T.i16(), _arith.ShRUIOp(bits, i32(16)).result).result
    return _arith.BitcastOp(T.bf16(), half).result


def _cvt(v, src: str, dst: str, unsigned: bool = False):
    """Convert a bare value between kinds (storage or compute)."""
    if src == "index":
        v = _arith.IndexCastOp(T.i64(), v).result
        src = "i64"
    if src == dst:
        return v
    if src in _FP8:
        word = _arith.ExtUIOp(T.i32(), v).result
        v = _FP8[src][0](T.f32(), word, 0).result
        return _cvt(v, "f32", dst, unsigned)
    if dst in _FP8:
        f = _cvt(v, src, "f32", unsigned)
        zero = _arith.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), 0)).result
        word = _FP8[dst][1](T.i32(), f, f, zero, False).result
        return _arith.TruncIOp(T.i8(), word).result
    src_int = src in _INT_KINDS
    dst_int = dst in _INT_KINDS
    if dst == "b1":
        if src_int:
            zero = _arith.ConstantOp(v.type, ir.IntegerAttr.get(v.type, 0)).result
            return _arith.CmpIOp(_arith.CmpIPredicate.ne, v, zero).result
        zero = _arith.ConstantOp(v.type, ir.FloatAttr.get(v.type, 0.0)).result
        return _arith.CmpFOp(_arith.CmpFPredicate.UNE, v, zero).result
    dst_t = _STORAGE_IR[dst]() if dst in _STORAGE_IR else _COMPUTE_IR[dst]()
    if src_int and dst_int:
        sw, dw = _int_width(v.type), _int_width(dst_t)
        if sw == dw:
            return v
        if sw < dw:
            if src in _UNSIGNED_KINDS or unsigned:
                return _arith.ExtUIOp(dst_t, v).result
            return _arith.ExtSIOp(dst_t, v).result
        return _arith.TruncIOp(dst_t, v).result
    if src_int and not dst_int:
        if src in _UNSIGNED_KINDS or unsigned:
            return _arith.UIToFPOp(dst_t, v).result
        return _arith.SIToFPOp(dst_t, v).result
    if not src_int and dst_int:
        if dst in _UNSIGNED_KINDS:
            return _arith.FPToUIOp(dst_t, v).result
        return _arith.FPToSIOp(dst_t, v).result
    sw = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}
    if sw[src] < sw[dst]:
        return _arith.ExtFOp(dst_t, v).result
    if src == "f32" and dst == "bf16":
        # A widened bf16 -- Inductor rounds through bf16 before a bf16 store -- comes
        # back exactly; LLVM folds that for ``fptrunc`` but not for the bit math.
        owner = getattr(v, "owner", None)
        if (
            isinstance(owner, ir.Operation | ir.OpView)
            and owner.name == "arith.extf"
            and owner.operands[0].type == T.bf16()
        ):
            return owner.operands[0]
        return _f32_to_bf16(v)
    if sw[src] > sw[dst]:
        return _arith.TruncFOp(dst_t, v).result
    # f16 <-> bf16
    return _arith.TruncFOp(dst_t, _arith.ExtFOp(T.f32(), v).result).result


def to(x, k: str):
    """Convert to compute kind ``k``. Python scalars become constants."""
    if isinstance(x, (bool, int, float)):
        x = const(x, "f32" if k in _FP8 else k)
    v = _val(x)
    if k in _FP8:
        # Rounded to FP8 and back: in compute an FP8 value is the f32 it stands for.
        f = _cvt(v, kind_of_type(v.type), "f32")
        r = ArithValue(_cvt(_cvt(f, "f32", k), k, "f32"))
        return _derive(r, "fp8", (k, f))
    return ArithValue(_cvt(v, kind_of_type(v.type), k))


def _fp8_source(v, st: str):
    """For an FP8 store, the f32 that ``v`` rounded to ``st`` from, if it did: rounding
    again gives the same bits, so the store converts that and the round trip goes
    unused -- three conversions an element to one on a rowwise quantization."""
    d = _derived(v)
    if d is not None and d[0] == "fp8" and d[1][0] == st:
        return d[1][1]
    return v


def bitcast(x, src: str, dst: str):
    """``x.view(dst)`` of a value of storage kind ``src`` held in its compute kind: back
    to storage (exact), the bits reinterpreted, out to ``dst``'s compute kind."""
    v = _val(to(x, _COMPUTE_OF[src]))
    v = _cvt(v, kind_of_type(v.type), src)
    dst_t = _STORAGE_IR[dst]()
    if v.type != dst_t:
        v = _arith.BitcastOp(dst_t, v).result
    return ArithValue(_cvt(v, dst, _COMPUTE_OF[dst]))


def _canon(x):
    """A value already in a compute kind (Python scalars left alone)."""
    if isinstance(x, (bool, int, float)):
        return x
    v = _val(x)
    k = kind_of_type(v.type)
    ck = _COMPUTE_OF.get(k, k)
    if ck == "index":
        ck = "i64"
    return ArithValue(_cvt(v, k, ck)) if ck != k else ArithValue(v)


def _promote(*xs):
    """Bring operands to one compute kind. Python scalars adopt the others' kind."""
    xs = [_canon(x) for x in xs]
    kinds = [kind(x) for x in xs if not isinstance(x, (bool, int, float))]
    if not kinds:
        if any(isinstance(x, float) for x in xs):
            k = "f32"
        elif all(isinstance(x, bool) for x in xs):
            k = "b1"
        else:
            k = "i32" if all(-(2**31) <= int(x) < 2**31 for x in xs) else "i64"
    else:
        k = max(kinds, key=_RANK.__getitem__)
        if k in ("b1", "i32") and any(isinstance(x, float) for x in xs):
            k = "f32"
        if k == "b1" and any(
            isinstance(x, int) and not isinstance(x, bool) for x in xs
        ):
            k = "i32"
    return k, [to(x, k) for x in xs]


def _is_f(k: str) -> bool:
    return k in ("f32", "f64")


def _float_kind(*xs) -> str:
    return (
        "f64"
        if any(
            not isinstance(x, (bool, int, float)) and kind(_canon(x)) == "f64"
            for x in xs
        )
        else "f32"
    )


# ---------------------------------------------------------------------------------------
# Thread geometry and memory
# ---------------------------------------------------------------------------------------


class Aff:
    """An i32 index ``base + c`` whose constant stays a Python int at trace time.

    The generated code computes a thread's offsets from ``tid()``/``bid()`` and the
    unrolled loop counters, so every run, row and chunk of a thread shares one ``base``
    and differs only in ``c``. Added last, ``c`` is folded by the backend into the
    buffer instruction's immediate offset; added where the expression happened to
    put it, it was buried under the row stride's multiply -- a 32-wide sum's 16 loads
    took 53 address instructions, 7 now. Anything not affine in the operands
    (division, comparison, ...) folds the constant in and goes on as an ordinary
    value. No ``ir_value``, so ``ArithValue`` operators defer to these.
    """

    __slots__ = ("base", "c")

    def __init__(self, base, c: int = 0):
        self.base = base
        self.c = c

    def value(self):
        if self.base is None:
            return const(self.c, _INDEX[0])
        return self.base + self.c if self.c else self.base

    @staticmethod
    def _of(o):
        """``o`` as an Aff, or None when it is not an index of the kernel's kind."""
        if isinstance(o, Aff):
            return o
        if isinstance(o, bool) or not isinstance(o, int):
            v = _val(o) if not isinstance(o, float) else None
            if v is None or kind_of_type(v.type) != _INDEX[0]:
                return None
            return Aff(ArithValue(v))
        return Aff(None, o)

    def _join(self, o, sign: int):
        a = Aff._of(o)
        if a is None:
            return add(self.value(), o) if sign > 0 else sub(self.value(), o)
        if a.base is None:
            base = self.base
        elif self.base is None:
            base = a.base if sign > 0 else const(0, _INDEX[0]) - a.base
        else:
            base = self.base + a.base if sign > 0 else self.base - a.base
        return Aff(base, self.c + sign * a.c)

    def __add__(self, o):
        return self._join(o, 1)

    __radd__ = __add__

    def __sub__(self, o):
        return self._join(o, -1)

    def __rsub__(self, o):
        return Aff(None, 0)._join(o, 1)._join(self, -1)

    def __neg__(self):
        return Aff(None, 0)._join(self, -1)

    def __mul__(self, o):
        if isinstance(o, int) and not isinstance(o, bool):
            return Aff(None if self.base is None else self.base * o, self.c * o)
        a = Aff._of(o)
        if a is not None and a.base is None:
            return self * a.c
        if self.base is None and a is not None:
            return a * self.c
        return Aff(ArithValue(_val(mul(self.value(), o))))

    __rmul__ = __mul__

    def _opaque(self, o, op):
        o = o.value() if isinstance(o, Aff) else o
        r = op(self.value(), o)
        return Aff(r) if kind_of_type(_val(r).type) == _INDEX[0] else r

    def __floordiv__(self, o):
        return self._opaque(o, lambda a, b: a // b)

    def __mod__(self, o):
        return self._opaque(o, lambda a, b: a % b)

    def __lt__(self, o):
        return lt(self.value(), o)

    def __le__(self, o):
        return le(self.value(), o)

    def __gt__(self, o):
        return gt(self.value(), o)

    def __ge__(self, o):
        return ge(self.value(), o)

    def __eq__(self, o):
        return eq(self.value(), o)

    def __ne__(self, o):
        return ne(self.value(), o)

    __hash__ = None


# The kind of the kernel being traced's index values: i64 for one over 2**31 elements.
_INDEX = ["i32"]


def kernel_begin(wide: bool = False) -> None:
    _INDEX[0] = "i64" if wide else "i32"


def _widen(v):
    if _INDEX[0] == "i32":
        return ArithValue(v)
    return ArithValue(_arith.ExtUIOp(T.i64(), v).result)


def tid():
    return Aff(_widen(_val(_tid())))


def bid():
    return Aff(_widen(_val(fx.Int32(gpu.block_idx.x))))


def _tid():
    return fx.Int32(gpu.thread_idx.x)


class _Flat:
    """A buffer of 2 GiB or more, beyond a descriptor: there a masked lane's offset
    (0x7FFFFFFF) would land inside it. Addressed through 64-bit global pointers
    instead, masked by selects and branches -- the logits a Qwen3 cross-entropy
    reads (8192 x 151936 bf16) are 2.5 GB."""

    __slots__ = ("addr",)

    def __init__(self, addr):
        self.addr = addr  # i64


# Word 3 of a CDNA buffer descriptor: data format 7 and num format 4, both of which
# must be nonzero and neither of which the raw buffer instructions read.
_RSRC_FLAGS = (7 << 12) | (4 << 15)
_MASKED_OFFSET = 0x7FFFFFFF


def _int_const(t, v):
    return _arith.ConstantOp(t, ir.IntegerAttr.get(t, v)).result


def _i32_offset(off):
    if isinstance(off, int):
        return _int_const(T.i32(), off)
    v = _val(off)
    if not (isinstance(v.type, ir.IntegerType) and v.type.width == 32):
        v = _arith.IndexCastOp(T.i32(), v).result
    return v


def _byte_offset(off, unit: int, mask):
    """Element offset ``off`` in bytes; a masked lane's is past every descriptor's end,
    so the hardware drops it (a load returns 0)."""
    o = _arith.MulIOp(_i32_offset(off), _int_const(T.i32(), unit)).result
    if mask is not None:
        o = _arith.SelectOp(_val(mask), o, _int_const(T.i32(), _MASKED_OFFSET)).result
    return o


def _rsrc_from_addr(addr, num_records):
    from flydsl._mlir.dialects import llvm as _llvm

    base = _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), addr).result
    if isinstance(num_records, int):
        n = _int_const(T.i64(), num_records)
    else:
        n = _arith.ExtUIOp(T.i64(), _val(num_records)).result
    return _rocdl.MakeBufferRsrcOp(
        ir.Type.parse("!llvm.ptr<8>"),
        base,
        _int_const(T.i16(), 0),
        n,
        _int_const(T.i32(), _RSRC_FLAGS),
    ).result


def rsrc(addr, nbytes: int):
    """Buffer resource over ``nbytes`` from a raw device address (or a ``_Flat`` for
    one too large for it).

    ``nbytes`` must be exact: it is what makes a masked access (offset 0x7FFFFFFF) fall
    out of range.
    """
    if not isinstance(nbytes, int):
        # Computed from the size arguments; guarded below 2**31 at compile time.
        return _rsrc_from_addr(_val(addr), idx32(nbytes))
    if nbytes >= 2**31 - 1:
        return _Flat(_val(addr))
    return _rsrc_from_addr(_val(addr), int(nbytes))


def size_arg(v):
    """A size argument (passed as 64 bits, like the addresses) as an index value."""
    if _INDEX[0] == "i64":
        return ArithValue(_val(v))
    return ArithValue(_arith.TruncIOp(T.i32(), _val(v)).result)


def _type_bytes(t) -> int:
    if isinstance(t, ir.VectorType):
        return _type_bytes(t.element_type) * t.shape[0]
    return t.width // 8


def _flat_ptr(r: _Flat, off, unit: int, mask=None):
    """``r + off * unit`` bytes as a global pointer, ``off`` an unsigned element
    offset; a masked lane points at the buffer's start instead."""
    from flydsl._mlir.dialects import llvm as _llvm

    o = _val(idx(off))
    wide = kind_of_type(o.type) == "i64"
    if mask is not None:
        o = _arith.SelectOp(mask, o, _val(const(0, "i64" if wide else "i32"))).result
    scaled = _arith.MulIOp(
        o if wide else _arith.ExtUIOp(T.i64(), o).result,
        _arith.ConstantOp(T.i64(), ir.IntegerAttr.get(T.i64(), unit)).result,
    ).result
    a = _arith.AddIOp(r.addr, scaled).result
    return _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), a).result


def _bload(r, off, vec_width: int, dtype, mask=None, is_scalar: bool = False):
    """A buffer load of ``vec_width`` ``dtype`` elements at element offset ``off``, or
    its 64-bit global form for a ``_Flat``. ``is_scalar`` loads one wave-uniform i32
    through the scalar cache into an SGPR."""
    from flydsl._mlir.dialects import llvm as _llvm

    t = dtype if vec_width == 1 else ir.VectorType.get([vec_width], dtype)
    unit = _type_bytes(dtype)
    if not isinstance(r, _Flat):
        o = _byte_offset(_narrow(off), unit, mask)
        if is_scalar:
            rsrc_v4 = _llvm.bitcast(
                ir.VectorType.get([4], T.i32()),
                _llvm.ptrtoint(ir.IntegerType.get_signless(128), r),
            )
            return _llvm.call_intrinsic(
                T.i32(),
                "llvm.amdgcn.s.buffer.load.i32",
                [rsrc_v4, o, _int_const(T.i32(), 0)],
                [],
                [],
            )
        return _rocdl.RawPtrBufferLoadOp(
            t, r, o, _int_const(T.i32(), 0), aux=None
        ).result
    v = _llvm.LoadOp(t, _flat_ptr(r, off, unit, mask), alignment=unit).result
    if mask is None:
        return v
    if isinstance(t, ir.VectorType):
        zero = _arith.ConstantOp(
            t, ir.DenseElementsAttr.get_splat(t, _zero_attr(dtype))
        ).result
    else:
        zero = _arith.ConstantOp(t, _zero_attr(dtype)).result
    return _arith.SelectOp(mask, v, zero).result


def _narrow(off):
    """A descriptor's offset: its buffer is under 2 GiB, so an in-bounds offset fits
    i32 (a masked lane's is replaced)."""
    if isinstance(off, int) or _INDEX[0] == "i32":
        return off
    v = _val(off)
    if kind_of_type(v.type) != "i64":
        return off
    return ArithValue(_arith.TruncIOp(T.i32(), v).result)


def _zero_attr(t):
    if isinstance(t, ir.IntegerType):
        return ir.IntegerAttr.get(t, 0)
    return ir.FloatAttr.get(t, 0.0)


def _bstore(value, r, off, mask=None):
    """A buffer store of ``value`` at element offset ``off``, or its 64-bit global
    form for a ``_Flat``."""
    from flydsl._mlir.dialects import llvm as _llvm

    value = _val(value)
    t = value.type
    unit = _type_bytes(t.element_type if isinstance(t, ir.VectorType) else t)
    if not isinstance(r, _Flat):
        o = _byte_offset(_narrow(off), unit, mask)
        _rocdl.RawPtrBufferStoreOp(value, r, o, _int_const(T.i32(), 0), aux=None)
        return
    ptr = _flat_ptr(r, off, unit)
    _when(mask, lambda: _llvm.StoreOp(value, ptr, alignment=unit))


def pick(s, *members):
    """``members[s]`` for a block-uniform ``s``: a merged combo branch choosing its
    sub-kernel's buffers (descriptors or raw addresses), one callable per sub-kernel
    returning them.

    A uniform switch building each member's descriptors in its own case, rather than
    select chains: selects keep every candidate descriptor live at once, which spills
    SGPRs once a group has a few dozen."""
    s = _arith.IndexCastOp(T.index(), _val(idx(s))).result
    wrap: list = []

    def raw(v):
        return v.addr if isinstance(v, _Flat) else _val(v)

    def emit(member):
        vals = member()
        wrap[:] = [
            _Flat
            if isinstance(v, _Flat)
            else None
            if isinstance(v, ir.Value)
            else ArithValue
            for v in vals
        ]
        _scf.YieldOp([raw(v) for v in vals])

    # The switch needs its result types up front; this trace of member 0 is left dead.
    types = [raw(v).type for v in members[0]()]
    op = _scf.IndexSwitchOp(
        types,
        s,
        list(range(1, len(members))),
        lambda op, i, _c: emit(members[i + 1]),
        lambda op: emit(members[0]),
    )
    return tuple(w(r) if w else r for w, r in zip(wrap, op.results))


def _as_int(x, k: str):
    if isinstance(x, Aff):
        x = x.value()
    if isinstance(x, int):
        return const(x, k)
    v = _val(x)
    vk = kind_of_type(v.type)
    if vk == k:
        return ArithValue(v)
    if vk == "index":
        return ArithValue(_arith.IndexCastOp(_COMPUTE_IR[k](), v).result)
    if _is_f(vk):
        return ArithValue(_arith.FPToSIOp(_COMPUTE_IR[k](), v).result)
    return ArithValue(_cvt(v, vk, k))


def idx(x):
    """An element offset, of the kernel's index kind, from a value of any integer kind."""
    return _as_int(x, _INDEX[0])


def idx32(x):
    """An i32 from an index value: LDS offsets, lanes, descriptor extents."""
    return _as_int(x, "i32")


def ic(v: int):
    """An index constant: a loop bound or step, an extent past 2**31."""
    return const(v, _INDEX[0])


def bucket(o, ends):
    """The member of a merged combo branch that owns block ``o`` (rebased to the
    branch) when members take different numbers of blocks: how many of the members'
    cumulative block ``ends`` are at or below it."""
    o = _val(idx(o))
    one, s = _val(idx(1)), _val(idx(0))
    for e in ends:
        ge = _arith.CmpIOp(_arith.CmpIPredicate.uge, o, _val(idx(e))).result
        s = _arith.AddIOp(s, _arith.SelectOp(ge, one, _val(idx(0))).result).result
    return ArithValue(s)


def idiv(a, b):
    """Index floor division. Index terms are non-negative, so unsigned is exact and cheaper."""
    a, b = idx(a), idx(b)
    return ArithValue(_arith.DivUIOp(_val(a), _val(b)).result)


def imod(a, b):
    a, b = idx(a), idx(b)
    return ArithValue(_arith.RemUIOp(_val(a), _val(b)).result)


def imin(a, b):
    a, b = idx(a), idx(b)
    return ArithValue(_arith.MinSIOp(_val(a), _val(b)).result)


def imax(a, b):
    a, b = idx(a), idx(b)
    return ArithValue(_arith.MaxSIOp(_val(a), _val(b)).result)


def _mask(m):
    if m is None or m is True:
        return None
    v = _val(m)
    if kind_of_type(v.type) != "b1":
        v = _cvt(v, kind_of_type(v.type), "b1")
    return v


def land(*ms):
    """AND of masks, ignoring the ``None``/``True`` placeholders for "always valid"."""
    out = None
    for m in ms:
        m = _mask(m)
        if m is None:
            continue
        out = m if out is None else _arith.AndIOp(out, m).result
    return out


_ITEMSIZE = {
    "f32": 4,
    "i32": 4,
    "f16": 2,
    "bf16": 2,
    "i16": 2,
    "i8": 1,
    "u8": 1,
    "u16": 2,
    "u32": 4,
    "b1": 1,
}


def load(r, off, st: str, mask=None):
    """Scalar masked load, widened to the compute kind. A masked lane reads 0."""
    raw = _bload(r, idx(off), 1, _STORAGE_IR[st](), _mask(mask))
    return ArithValue(_cvt(raw, _MEM_KIND.get(st, st), _COMPUTE_OF[st]))


def sload(r, off: int, st: str):
    """A load at a constant offset, through the scalar cache into an SGPR, as Triton's
    ``s_load`` of a broadcast operand: a vector load would hold every wave's stores on
    its round trip. The scalar cache does not see this kernel's own stores, so only
    for buffers the kernel only reads."""
    if isinstance(r, _Flat):
        return load(r, off, st)
    size = _ITEMSIZE[st]
    byte = off * size
    word = _bload(r, fx.Int32(byte // 4), 1, T.i32(), is_scalar=True)
    if byte % 4:
        shift = _arith.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), byte % 4 * 8))
        word = _arith.ShRUIOp(word, shift.result).result
    mem = _STORAGE_IR[st]()
    if size < 4:
        word = _arith.TruncIOp(ir.IntegerType.get_signless(8 * size), word).result
    raw = word if mem == word.type else _arith.BitcastOp(mem, word).result
    return ArithValue(_cvt(raw, _MEM_KIND.get(st, st), _COMPUTE_OF[st]))


def _offset(off):
    # Offsets of loads repeated across an unrolled region come as thunks: evaluated
    # for every element, most of a 64-element RoPE variant's 4400 traced ops.
    return off() if callable(off) else off


def wload(cache: dict, key, chunk: int, r, off, st: str):
    """``sload`` at an offset computed at run time but the same across the wave: lane
    0's offset, read into an SGPR. Read-only buffers only, as for ``sload``."""
    if (key, "w", chunk) in cache:
        return cache[(key, "w", chunk)]
    off = _offset(off)
    if isinstance(r, _Flat):
        cache[(key, "w", chunk)] = load(r, off, st)
        return cache[(key, "w", chunk)]
    size = _ITEMSIZE[st]
    byte = _val(idx32(off) * size)
    byte = _rocdl.readfirstlane(T.i32(), byte)
    two = _arith.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), 2)).result
    word = _bload(r, _arith.ShRUIOp(byte, two).result, 1, T.i32(), is_scalar=True)
    if size < 4:
        three = _arith.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), 3)).result
        shift = _arith.ShLIOp(_arith.AndIOp(byte, three).result, three).result
        word = _arith.ShRUIOp(word, shift).result
        word = _arith.TruncIOp(ir.IntegerType.get_signless(8 * size), word).result
    mem = _STORAGE_IR[st]()
    raw = word if mem == word.type else _arith.BitcastOp(mem, word).result
    out = ArithValue(_cvt(raw, _MEM_KIND.get(st, st), _COMPUTE_OF[st]))
    cache[(key, "w", chunk)] = out
    return out


def _packing(st: str, w: int):
    """How a ``w``-wide vector of a sub-dword kind travels through the buffer intrinsic.

    The AMDGPU backend cannot legalize every narrow vector (``<8 x i8>`` stores abort in
    instruction selection), so sub-dword vectors move as i16/i32 words and are bitcast
    on either side. Returns ``(word type, word count, elements per word)``, or ``None``
    as the type for kinds that go through unchanged.
    """
    nbytes = {
        "f16": 2,
        "bf16": 2,
        "i16": 2,
        "u16": 2,
        "i8": 1,
        "u8": 1,
        "b1": 1,
        "f8e4": 1,
        "f8e5": 1,
    }.get(st)
    if nbytes is None:
        return None, w, 1
    total = nbytes * w
    if total % 4 == 0:
        return T.i32(), total // 4, 4 // nbytes
    if total == 2:
        return T.i16(), 1, 2
    raise ValueError(f"unpackable vector of {w} x {st}")


def _as_vector(words, n: int, word_t):
    if n == 1:
        return _vector.FromElementsOp(
            ir.VectorType.get([1], word_t), [_val(words)]
        ).result
    return words


def vload(cache: dict, key, j: int, w: int, r, off, st: str, mask=None):
    """Element ``j`` of a ``w``-wide contiguous load.

    Called once per unrolled element. The first element of each aligned chunk issues the
    vector load and caches it (trace time, so the cache is a Python dict), and the rest
    extract from it. ``off`` may be a thunk, called only to issue the load.
    """
    if (key, j // w) not in cache:
        off = _offset(off)
        # Usually issued by the chunk's first element, but a store to the same buffer
        # (see forget) can drop the vector mid-chunk; the load always starts at the base.
        base = idx(off if j % w == 0 else off - j % w)
        word_t, n, ratio = _packing(st, w)
        if word_t is None:
            raw = _bload(r, base, w, _STORAGE_IR[st](), _mask(mask))
        else:
            words = _bload(r, idiv(base, ratio), n, word_t, _mask(mask))
            raw = _vector.BitCastOp(
                ir.VectorType.get([w], _STORAGE_IR[st]()), _as_vector(words, n, word_t)
            ).result
        cache[(key, j // w)] = raw
    wide = cache.get(((key, j // w), _WIDE))
    if wide is not None:
        return ArithValue(
            _vector.ExtractOp(wide, static_position=[j % w], dynamic_position=[]).result
        )
    vec = cache[(key, j // w)]
    elem = _vector.ExtractOp(vec, static_position=[j % w], dynamic_position=[]).result
    return ArithValue(_cvt(elem, _MEM_KIND.get(st, st), _COMPUTE_OF[st]))


_WIDE = "_wide"


def widen(cache: dict, keys) -> None:
    """Convert the cached f16/bf16 vectors of the loads ``keys`` names (``(arg, index)``
    pairs) to f32 here, for vload to hand out, and keep the scheduler from sinking the
    conversions back to their uses.

    Called after a row's first pass of a persistent reduction, for the loads of later
    passes issued ahead of it (a LayerNorm's weight and bias): the conversions then run
    while the wave waits on the first pass's loads, rather than after its reduction,
    where every resident wave reaches the same normalization tail at once. GPT's
    embedding + LayerNorm 14.0us -> 12.7us (Triton 12.9us)."""
    done = False
    for k, vec in list(cache.items()):
        if not (isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], tuple)):
            continue
        if tuple(k[0][:2]) not in keys or not isinstance(vec.type, ir.VectorType):
            continue
        if kind_of_type(vec.type.element_type) not in ("f16", "bf16"):
            continue
        f32 = ir.VectorType.get(vec.type.shape, T.f32())
        cache[(k, _WIDE)] = _arith.ExtFOp(f32, vec).result
        done = True
    if done:
        sched_fence()


def uload(cache: dict, key, chunk: int, r, off, st: str, mask=None):
    """A load whose address is the same across one contiguous run of elements a thread
    owns (``chunk`` numbers the runs when the thread owns several): issue it once."""
    if (key, "u", chunk) not in cache:
        cache[(key, "u", chunk)] = load(r, _offset(off), st, mask)
    return cache[(key, "u", chunk)]


def pair_load(k, r, off, st: str, mask=None, cache=None, key=None):
    """Column ``k`` (0 or 1) of a lane's two adjacent columns, from one two-element load
    issued by column 0 at its (pair-aligned) offset. The load is unmasked -- the
    descriptor reads 0 past its extent -- and each column applies its own mask, so
    column 1 never depends on column 0's. ``k`` None is a plain ``load``."""
    if k is None or isinstance(r, _Flat):
        return load(r, off, st, mask=mask)
    v = vload(cache, key, k, 2, r, off, st)
    m = _mask(mask)
    if m is None:
        return v
    return ArithValue(_arith.SelectOp(m, _val(v), _val(const(0, kind(v)))).result)


def memo(cache: dict, key, thunk):
    """Trace-time load reuse in an unrolled region: the first trace of ``key`` loads."""
    if key not in cache:
        cache[key] = thunk()
    return cache[key]


_FRESH = "__fresh__"
_PENDING = "__pending__"


def forget(cache: dict, arg: str) -> None:
    """Issue the stores to ``arg`` vstore held back, then drop every cached load of
    ``arg``, but the stored values themselves (vstore's ``seed``) entered since the
    last forget."""

    def owner(k):
        while isinstance(k, tuple) and k:
            if k[0] == arg:
                return True
            k = k[0]
        return False

    pending = cache.get(_PENDING, [])
    for p in [p for p in pending if p[0] == arg]:
        _bstore(*p[1:])
        pending.remove(p)
    fresh = cache.get(_FRESH, OrderedSet())
    for k in [k for k in cache if owner(k)]:
        if k in fresh:
            fresh.discard(k)
        else:
            del cache[k]


def store(r, off, value, st: str, mask=None):
    v = _val(_canon(value))
    if st in _FP8:
        v = _val(_fp8_source(v, st))
    raw = _cvt(v, kind_of_type(v.type), _MEM_KIND.get(st, st))
    _bstore(raw, r, idx(off), _mask(mask))


def xstore(cache: dict, key, xr: int, n: int, tpr: int, rl, r, off, value, st: str):
    """Row ``xr``'s result -- held by every lane of the row group -- of the group's
    ``n`` rows, stored once the last is in: lane ``k`` takes rows ``k``, ``k + tpr``,
    and so on, so a wave stores with every lane rather than one in ``tpr``."""
    slot = cache.setdefault(key, [])
    slot.append((idx(off), value))
    if xr != n - 1:
        return
    del cache[key]
    for s in range(0, n, tpr):
        group = slot[s : s + tpr]
        o, v = group[0]
        for k in range(1, len(group)):
            take = eq(rl, k)
            o, v = where(take, group[k][0], o), where(take, group[k][1], v)
        store(r, o, v, st, mask=lt(rl, len(group)) if len(group) < tpr else None)


def vstore(
    cache: dict, key, j: int, w: int, r, off, value, st: str, mask=None, seed=None
):
    """Element ``j`` of a ``w``-wide contiguous store; the vector goes out on the last one.

    ``seed``, a ``(load cache, load key)`` pair, also enters the stored vector there.
    """
    v = _val(_canon(value))
    # FP8 elements stay f32 until the vector goes out, to convert two to a word half.
    f8 = st in _FP8 and w % 4 == 0
    if st in _FP8:
        v = _val(_fp8_source(v, st))
    raw = _cvt(v, kind_of_type(v.type), "f32" if f8 else _MEM_KIND.get(st, st))
    if j % w == 0:
        # A fresh slot: the same site is traced again by every unrolled loop iteration.
        cache[(key, j // w)] = {"elems": [], "off": idx(off), "mask": _mask(mask)}
    slot = cache[(key, j // w)]
    slot["elems"].append(raw)
    if j % w == w - 1:
        del cache[(key, j // w)]
        vt = ir.VectorType.get([w], _STORAGE_IR[st]())
        word_t, n, ratio = _packing(st, w)
        if f8:
            e = slot["elems"]
            pk = _FP8[st][1]
            word = _arith.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), 0)).result
            ws = []
            for q in range(0, w, 4):
                lo = pk(T.i32(), e[q], e[q + 1], word, False).result
                ws.append(pk(T.i32(), e[q + 2], e[q + 3], lo, True).result)
            wv = _vector.FromElementsOp(ir.VectorType.get([n], T.i32()), ws).result
            vec = _vector.BitCastOp(vt, wv).result
        else:
            vec = _vector.FromElementsOp(vt, slot["elems"]).result
        if word_t is None:
            args = (vec, r, slot["off"], slot["mask"])
        else:
            words = _vector.BitCastOp(ir.VectorType.get([n], word_t), vec).result
            if n == 1:
                words = _vector.ExtractOp(
                    words, static_position=[0], dynamic_position=[]
                ).result
            args = (words, r, idiv(slot["off"], ratio), slot["mask"])
        if seed is None:
            _bstore(*args)
        else:
            # A later unrolled loop reading the stored elements back takes the register,
            # in exactly vload's form, instead of a round trip through memory.
            lc, lkey = seed
            if slot["mask"] is not None:
                # What the masked reload would read: 0 where the store was masked off.
                zero = _arith.ConstantOp(
                    vt,
                    ir.DenseElementsAttr.get_splat(vt, _zero_attr(_STORAGE_IR[st]())),
                ).result
                vec = _arith.SelectOp(slot["mask"], vec, zero).result
            lc[(lkey, j // w)] = vec
            lc.setdefault(_FRESH, OrderedSet()).add((lkey, j // w))
            # Held back to the loop's forget: LLVM keeps a buffer load after a store it
            # cannot prove apart, so storing as it went left each of a gather's loads
            # (GPT's embedding + LayerNorm) a full memory round trip behind the last.
            lc.setdefault(_PENDING, []).append((lkey[0], *args))


def _packed_half(o, value, st: str):
    """(two-lane vector, element offset of its 32-bit word) for a 16-bit atomic add
    at element ``o``: the value in its lane and -0.0 in the neighbour's."""
    h = _cvt(_val(to(value, "f32")), "f32", st)
    et = _STORAGE_IR[st]()
    negz = _arith.ConstantOp(et, ir.FloatAttr.get(et, -0.0)).result
    ok = kind_of_type(_val(o).type)
    odd = _val(eq(ArithValue(_arith.AndIOp(_val(o), _val(const(1, ok))).result), 1))
    lo = _arith.SelectOp(odd, negz, h).result
    hi = _arith.SelectOp(odd, h, negz).result
    v = _vector.FromElementsOp(ir.VectorType.get([2], et), [lo, hi]).result
    return v, _arith.AndIOp(_val(o), _val(const(-2, ok))).result


def _when(m, emit):
    """Run ``emit()`` under ``scf.if m`` (or unconditionally when ``m`` is None)."""
    if m is None:
        emit()
        return
    if_op = _scf.IfOp(m)
    with ir.InsertionPoint(if_op.then_block):
        emit()
        _scf.YieldOp([])


def atomic_add_global(addr, off, value, st: str, mask=None):
    """bf16 ``*(addr + off) += value``: gfx942 has a packed bf16 atomic add only on
    global pointers (``buffer_atomic_pk_add_bf16`` is gfx950), so this one goes through
    an LLVM ``atomicrmw`` on the raw address instead of the buffer descriptor. Integer
    adds (i32, i64) go the same way."""
    r = _Flat(_val(to(addr, "i64")))
    if st in ("i32", "i64"):
        _global_add(r, idx(off), 4 if st == "i32" else 8, _val(to(value, st)), mask)
        return
    v, word = _packed_half(idx(off), value, st)
    _global_fadd(r, word, 2, v, mask)


def _global_add(r: _Flat, off, unit: int, v, mask):
    from flydsl._mlir.dialects import llvm as _llvm

    ptr = _flat_ptr(r, off, unit)

    def emit():
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            ptr,
            v,
            _llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=unit,
        )

    _when(_mask(mask), emit)


def atomic_add_pair(cache, key, j: int, target, off, value, st: str, mask=None):
    """A thread's elements ``j`` (even) and ``j + 1`` of a 16-bit atomic add, at offsets
    ``off`` and ``off + 1`` with ``off`` even, as one packed add of both halves instead
    of two each with a -0.0 half. ``j`` even holds its value; ``j + 1`` issues both,
    under its own mask, which codegen guarantees is the even element's too."""
    if j % 2 == 0:
        cache[(key, "pair")] = (idx(off), value)
        return
    o, first = cache.pop((key, "pair"))
    et = _STORAGE_IR[st]()
    halves = [_cvt(_val(to(x, "f32")), "f32", st) for x in (first, value)]
    v = _vector.FromElementsOp(ir.VectorType.get([2], et), halves).result
    if st == "bf16" and not isinstance(target, _Flat):
        target = _Flat(_val(to(target, "i64")))
    _buffer_add(target, _val(o), 2, v, mask)


def _global_fadd(r: _Flat, off, unit: int, v, mask):
    """``atomicrmw fadd`` at ``r + off * unit`` bytes; a masked lane does nothing."""
    from flydsl._mlir.dialects import llvm as _llvm

    ptr = _flat_ptr(r, off, unit)

    def emit():
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.fadd,
            ptr,
            v,
            _llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=_type_bytes(v.type),
        )

    _when(_mask(mask), emit)


def atomic_add(r, off, value, st: str, mask=None):
    """``*(r + off) += value`` through the buffer atomic; a masked lane does nothing.

    f16/bf16 have only packed (two-lane) atomic adds: the element's lane gets the value
    and its neighbour -0.0, the additive identity for every float including -0.0, so
    the neighbour is left bit-for-bit unchanged.
    """
    o = idx(off)
    if st in ("f32", "f64"):
        v = _val(to(value, st))
        _buffer_add(r, _val(o), 4 if st == "f32" else 8, v, mask)
    elif st == "f16" or (st == "bf16" and isinstance(r, _Flat)):
        v, word = _packed_half(o, value, st)
        _buffer_add(r, word, 2, v, mask)
    else:
        raise NotImplementedError(f"atomic_add of {st}")


def _buffer_add(r, off, unit: int, v, mask):
    """``*(r + off * unit bytes) += v``, the buffer atomic or, for a ``_Flat``, the
    global one."""
    if isinstance(r, _Flat):
        _global_fadd(r, off, unit, v, mask)
        return
    byte_off = _val(mul(ArithValue(_val(_narrow(ArithValue(off)))), unit))

    def fadd():
        _rocdl.RawPtrBufferAtomicFaddOp(v.type, v, r, byte_off, _int_const(T.i32(), 0))

    m = _mask(mask)
    if m is None:
        fadd()
        return
    # Unlike loads and stores, a buffer atomic steered out of range (offset 0x7FFFFFFF)
    # faults on gfx942 rather than being dropped, so a masked lane needs a real branch.
    # Built by hand: helpers are not AST-rewritten, so a Python `if` would not trace.
    if_op = _scf.IfOp(m)
    with ir.InsertionPoint(if_op.then_block):
        fadd()
        _scf.YieldOp([])


# ---------------------------------------------------------------------------------------
# Pointwise ops
# ---------------------------------------------------------------------------------------


def _is_one(v) -> bool:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v == 1
    try:
        op = _val(v).owner.opview
    except (AttributeError, ValueError):
        return False
    if not isinstance(op, _arith.ConstantOp):
        return False
    attr = op.value
    return (
        isinstance(attr, ir.FloatAttr)
        and kind_of_type(attr.type) == "f32"
        and attr.value == 1.0
    )


def add(a, b):
    for one, other in ((a, b), (b, a)):
        if _is_one(one) and (alt := _one_plus_of(other)) is not None:
            return alt
    _, (x, y) = _promote(a, b)
    return ArithValue(x + y)


def sub(a, b):
    k, (x, y) = _promote(a, b)
    r = ArithValue(x - y)
    if k == "f32" and (_derived(b) or (None,))[0] == "rowmax":
        return _derive(r, "le0")
    return r


def mul(a, b):
    k, (x, y) = _promote(a, b)
    if k == "b1":
        return ArithValue(_arith.AndIOp(_val(x), _val(y)).result)
    return ArithValue(x * y)


def truediv(a, b):
    k = _float_kind(a, b)
    x, y = to(a, k), to(b, k)
    if k == "f32" and (_derived(b) or (None,))[0] == "ge1":
        # IEEE's ~10-instruction divide sequence for a denominator v_rcp gets within
        # 1 ulp of: SiLU's x / (1 + exp(-x)).
        return ArithValue(x * _rcp(y))
    return ArithValue(_arith.DivFOp(_val(x), _val(y)).result)


def div_rn(a, b):
    return truediv(a, b)


def truncdiv(a, b):
    k, (x, y) = _promote(a, b)
    if _is_f(k):
        return trunc(ArithValue(_arith.DivFOp(_val(x), _val(y)).result))
    return ArithValue(_arith.DivSIOp(_val(x), _val(y)).result)


def floordiv(a, b):
    k, (x, y) = _promote(a, b)
    if _is_f(k):
        return floor(ArithValue(_arith.DivFOp(_val(x), _val(y)).result))
    return ArithValue(_arith.FloorDivSIOp(_val(x), _val(y)).result)


def mod(a, b):
    """C-style remainder (Triton's ``%``): the sign follows the dividend."""
    k, (x, y) = _promote(a, b)
    if _is_f(k):
        return ArithValue(_arith.RemFOp(_val(x), _val(y)).result)
    return ArithValue(_arith.RemSIOp(_val(x), _val(y)).result)


def fmod(a, b):
    return mod(a, b)


def remainder(a, b):
    """Python-style remainder: the sign follows the divisor."""
    k, (x, y) = _promote(a, b)
    r = ArithValue(mod(x, y))
    zero = const(0, k)
    wrong = ArithValue(
        ArithValue(r != zero) & ArithValue(ArithValue(r < zero) != ArithValue(y < zero))
    )
    return ArithValue(wrong.select(ArithValue(r + y), r))


def maximum(a, b):
    k, (x, y) = _promote(a, b)
    if _is_f(k):
        return ArithValue(_arith.MaximumFOp(_val(x), _val(y)).result)
    if k == "b1":
        return ArithValue(_arith.OrIOp(_val(x), _val(y)).result)
    return ArithValue(_arith.MaxSIOp(_val(x), _val(y)).result)


def minimum(a, b):
    k, (x, y) = _promote(a, b)
    if _is_f(k):
        return ArithValue(_arith.MinimumFOp(_val(x), _val(y)).result)
    if k == "b1":
        return ArithValue(_arith.AndIOp(_val(x), _val(y)).result)
    return ArithValue(_arith.MinSIOp(_val(x), _val(y)).result)


_UNSIGNED_PRED = {"slt": "ult", "sle": "ule", "sgt": "ugt", "sge": "uge"}


def _cmp(a, b, ipred, fpred):
    k, (x, y) = _promote(a, b)
    if _is_f(k):
        return ArithValue(_arith.CmpFOp(fpred, _val(x), _val(y)).result)
    if k == "b1" and ipred.name in _UNSIGNED_PRED:
        # Signed, true is -1 < false: Inductor's unrolled argmax orders NaN flags so.
        ipred = getattr(_arith.CmpIPredicate, _UNSIGNED_PRED[ipred.name])
    return ArithValue(_arith.CmpIOp(ipred, _val(x), _val(y)).result)


def lt(a, b):
    return _cmp(
        a,
        b,
        _arith.CmpIPredicate.slt,
        _arith.CmpFPredicate.OLT,  # codespell:ignore
    )  # codespell:ignore


def le(a, b):
    return _cmp(a, b, _arith.CmpIPredicate.sle, _arith.CmpFPredicate.OLE)


def gt(a, b):
    return _cmp(a, b, _arith.CmpIPredicate.sgt, _arith.CmpFPredicate.OGT)


def ge(a, b):
    return _cmp(a, b, _arith.CmpIPredicate.sge, _arith.CmpFPredicate.OGE)


def eq(a, b):
    return _cmp(a, b, _arith.CmpIPredicate.eq, _arith.CmpFPredicate.OEQ)


def ne(a, b):
    # Unordered: NaN != NaN is true, as in torch.
    return _cmp(a, b, _arith.CmpIPredicate.ne, _arith.CmpFPredicate.UNE)


def _b(x):
    return _val(to(x, "b1"))


def logical_and(a, b):
    return ArithValue(_arith.AndIOp(_b(a), _b(b)).result)


def logical_or(a, b):
    return ArithValue(_arith.OrIOp(_b(a), _b(b)).result)


def logical_xor(a, b):
    return ArithValue(_arith.XOrIOp(_b(a), _b(b)).result)


def logical_not(a):
    return ArithValue(_arith.XOrIOp(_b(a), _val(const(True, "b1"))).result)


def _int_bin(a, b, op):
    k, (x, y) = _promote(a, b)
    if _is_f(k):
        raise TypeError("bitwise op on a float")
    return ArithValue(op(_val(x), _val(y)).result)


def bitwise_and(a, b):
    return _int_bin(a, b, _arith.AndIOp)


def bitwise_or(a, b):
    return _int_bin(a, b, _arith.OrIOp)


def bitwise_xor(a, b):
    return _int_bin(a, b, _arith.XOrIOp)


def bitwise_not(a):
    x = _canon(a)
    k = kind(x)
    allones = const(True, "b1") if k == "b1" else const(-1, k)
    return ArithValue(_arith.XOrIOp(_val(x), _val(allones)).result)


def bitwise_left_shift(a, b):
    return _int_bin(a, b, _arith.ShLIOp)


def bitwise_right_shift(a, b):
    return _int_bin(a, b, _arith.ShRSIOp)


def where(c, a, b):
    _, (x, y) = _promote(a, b)
    return ArithValue(_arith.SelectOp(_b(c), _val(x), _val(y)).result)


def bucketize(
    r,
    st,
    values,
    base,
    size,
    numel,
    stride,
    right,
    sr,
    sst,
    sstride,
    sbase,
    out,
    mask=None,
):
    """``triton_helpers.bucketize_binary_search``, its ``log2(size + 1)`` steps unrolled:
    boundaries ``r`` (storage kind ``st``) from element ``base`` at ``stride``, through
    the optional sorter ``sr``; the bucket in compute kind ``out``. NaN goes last."""
    v = _canon(values)
    nan = ne(v, v) if _is_f(kind(v)) else None
    lo, hi = const(0, "i32"), const(size, "i32")
    full = size + 1
    while full > 1:
        mid = idiv(add(lo, hi), 2)
        m = land(lt(add(mul(mid, stride), base), numel), lt(mid, size), mask)
        at = mid if sr is None else load(sr, add(sbase, mul(mid, sstride)), sst, mask=m)
        b = load(r, add(base, mul(at, stride)), st, mask=m)
        above = ge(v, b) if right else gt(v, b)
        if nan is not None:
            above = logical_or(above, nan)
        lo = where(land(above, m), add(mid, 1), lo)
        hi = where(above, hi, mid)
        full = (full + 1) // 2
    return to(lo, out)


def bsearch_steps(size):
    """Halvings a search over ``size`` boundaries takes: ``size``'s bit length."""
    from flydsl._mlir.dialects import math as _mlir_math

    s = _val(idx32(size))
    clz = _mlir_math.CountLeadingZerosOp(s).result
    return ArithValue(_arith.SubIOp(_val(const(32, "i32")), clz).result)


def bsearch_init(size):
    return const(0, _INDEX[0]), idx(size)


def bsearch_step(
    lo,
    hi,
    r,
    st,
    values,
    base,
    size,
    numel,
    stride,
    right,
    sr,
    sst,
    sstride,
    sbase,
    mask=None,
):
    """One step of ``bucketize`` for a size known only at run time, in a loop of
    ``bsearch_steps(size)``; a search already settled (``lo == hi``) stays put."""
    v = _canon(values)
    live = lt(lo, hi)
    mid = idiv(add(lo, hi), 2)
    m = land(lt(add(mul(mid, stride), base), numel), lt(mid, size), mask, live)
    at = mid if sr is None else load(sr, add(sbase, mul(mid, sstride)), sst, mask=m)
    b = load(r, add(base, mul(at, stride)), st, mask=m)
    above = ge(v, b) if right else gt(v, b)
    if _is_f(kind(v)):
        above = logical_or(above, ne(v, v))
    lo = where(land(above, m), add(mid, 1), lo)
    hi = where(logical_or(above, logical_not(live)), hi, mid)
    return idx(lo), idx(hi)


def bucketize_lds(smem, base, size, k, values, right, out):
    """``bucketize`` against one table of ``size`` elements (compute kind ``k``)
    staged at word ``base`` of LDS."""
    v = _canon(values)
    nan = ne(v, v) if _is_f(kind(v)) else None
    lo, hi = const(0, "i32"), const(size, "i32")
    full = size + 1
    while full > 1:
        mid = idiv(add(lo, hi), 2)
        m = lt(mid, size)
        b = tile_get(smem, base, size, imin(mid, size - 1), k)
        above = ge(v, b) if right else gt(v, b)
        if nan is not None:
            above = logical_or(above, nan)
        lo = where(land(above, m), add(mid, 1), lo)
        hi = where(above, hi, mid)
        full = (full + 1) // 2
    return to(lo, out)


def neg(x):
    x = _canon(x)
    k = kind(x)
    if _is_f(k):
        return ArithValue(_arith.NegFOp(_val(x)).result)
    return ArithValue(const(0, k) - x)


def abs(x):
    x = _canon(x)
    if _is_f(kind(x)):
        return ArithValue(fmath.absf(x))
    return ArithValue(fmath.absi(x))


def sign(x):
    x = _canon(x)
    k = kind(x)
    zero = const(0, k)
    pos = to(ArithValue(gt(x, zero)), k)
    negv = to(ArithValue(lt(x, zero)), k)
    return ArithValue(pos - negv)


def signbit(x):
    x = _canon(x)
    k = kind(x)
    if not _is_f(k):
        return ArithValue(lt(x, const(0, k)))
    it = T.i64() if k == "f64" else T.i32()
    bits = _arith.BitcastOp(it, _val(x)).result
    zero = _arith.ConstantOp(it, ir.IntegerAttr.get(it, 0)).result
    return ArithValue(_arith.CmpIOp(_arith.CmpIPredicate.slt, bits, zero).result)


def isnan(x):
    x = _canon(x)
    if not _is_f(kind(x)):
        return const(False, "b1")
    return ArithValue(_arith.CmpFOp(_arith.CmpFPredicate.UNO, _val(x), _val(x)).result)


def isinf(x):
    x = _canon(x)
    k = kind(x)
    if not _is_f(k):
        return const(False, "b1")
    a = ArithValue(fmath.absf(x))
    return ArithValue(
        _arith.CmpFOp(
            _arith.CmpFPredicate.OEQ, _val(a), _val(const(float("inf"), k))
        ).result
    )


def isfinite(x):
    return logical_not(logical_or(isnan(x), isinf(x)))


def clamp(x, lo, hi):
    return minimum(maximum(x, lo), hi)


def relu(x):
    x = _canon(x)
    return maximum(x, const(0, kind(x)))


def _f32(x):
    return to(x, "f32")


def _require_f32(name, x):
    if _float_kind(x) == "f64":
        raise NotImplementedError(
            f"{name} on float64 has no AMDGPU lowering in the FlyDSL SIMD backend"
        )
    return _f32(x)


def _unary_f32(name, impl):
    def fn(x):
        return ArithValue(impl(_require_f32(name, x)))

    fn.__name__ = name
    return fn


def sqrt(x):
    k = _float_kind(x)
    return ArithValue(fmath.sqrt(to(x, k)))


def rsqrt(x):
    """``v_rsq_f32`` (1 ulp), as Triton's ``tl.rsqrt``: ``math.rsqrt`` lowers to a
    correctly rounded sqrt and an IEEE divide, ~40 dependent instructions on a norm's
    critical path -- GPT's embedding + LayerNorm 15.5us against Triton's 12.8us."""
    return ArithValue(_rocdl.rsq(T.f32(), _val(_require_f32("rsqrt", x))))


def reciprocal(x):
    k = _float_kind(x)
    return truediv(const(1.0, k), to(x, k))


def floor(x):
    x = _canon(x)
    return ArithValue(fmath.floor(x)) if _is_f(kind(x)) else x


def ceil(x):
    x = _canon(x)
    return ArithValue(fmath.ceil(x)) if _is_f(kind(x)) else x


def trunc(x):
    x = _canon(x)
    return ArithValue(fmath.trunc(x)) if _is_f(kind(x)) else x


def round(x):
    x = _canon(x)
    return ArithValue(fmath.roundeven(x)) if _is_f(kind(x)) else x


def round_decimal(x, decimals):
    raise NotImplementedError(
        "round with decimals is not lowered by the FlyDSL SIMD backend"
    )


def libdevice_exp(x):
    return exp(x)


def sched_fence():
    """``llvm.amdgcn.sched.barrier(0)``: the machine scheduler moves nothing across it."""
    _rocdl.sched_barrier(ir.Attribute.parse("#rocdl<sched_group_mask none>"))


def _rcp(x):
    """``v_rcp_f32``: 1 ulp, against ~10 instructions for an IEEE fdiv. Only for
    denominators kept well away from zero and denormals."""
    return ArithValue(_rocdl.rcp(T.f32(), _val(x)))


def _exp2_raw(x):
    """Bare ``v_exp_f32``, without LLVM's denormal-range fix-up (~4 instructions): for
    the callers below a denormal result and zero are interchangeable."""
    return ArithValue(_rocdl.exp2(T.f32(), _val(x)))


def _copysign(mag, x):
    return ArithValue(fmath.copysign(mag, x))


def _horner(z, coeffs):
    """``coeffs[0] + coeffs[1] z + ...``, as fused multiply-adds."""
    acc = _cf32(coeffs[-1])
    for c in reversed(coeffs[:-1]):
        acc = ArithValue(fmath.fma(acc, z, _cf32(c)))
    return acc


# Odd Taylor series for small arguments, where erf's and tanh's `1 - (close to 1)` forms
# lose all relative accuracy: 2/sqrt(pi) (-1)^n / (n! (2n+1)), and tanh's from the
# Bernoulli numbers. Truncation below 3e-8 relative on |x| < 0.25.
# erf(x) / x on [0, 1] in powers of x^2: a relative-error fit, 5e-8 in exact arithmetic.
_ERF_UNIT = (
    1.1283791065216064,
    -0.37612324953079224,
    0.11280179768800735,
    -0.02671131119132042,
    0.00491755036637187,
    -0.000563141773454845,
)
_TANH_SMALL = (1.0, -1.0 / 3, 2.0 / 15, -17.0 / 315, 62.0 / 2835)
_SMALL = 0.25


def _erf_rcp(x):
    # Below 1, erf(x) / x as a polynomial in x^2 (relative error ~1e-7). Above,
    # A&S 7.1.26 (abs error 1.5e-7) with 1 / (1 + p|x|), which lies in (0, 1], taken by
    # v_rcp_f32. Pointwise kernels around erf are ALU-bound on MI300 (~35 instructions
    # an element at HBM speed), so every instruction here shows: an erf GELU went
    # 121us -> 96us without the fdiv. The two sides are a real branch, as in OCML's
    # erf: a wave whose lanes are all below 1 skips the exp path (126us vs 143us on
    # the GELU of a freshly initialized ViT MLP).
    x = _f32(x)
    ax = ArithValue(fmath.absf(x))
    if_op = _scf.IfOp(_val(ax < _cf32(1.0)), [T.f32()], has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        small = ArithValue(ax * _horner(ArithValue(ax * ax), _ERF_UNIT))
        _scf.YieldOp([_val(small)])
    with ir.InsertionPoint(if_op.else_block):
        t = _rcp(fmath.fma(_cf32(_ERF_P), ax, _cf32(1.0)))
        poly = ArithValue(t * _horner(t, tuple(reversed(_ERF_COEFFS))))
        e = _exp2_raw(ArithValue(ArithValue(ax * ax) * _cf32(-_LOG2E)))
        big = ArithValue(fmath.fma(ArithValue(-poly), e, _cf32(1.0)))
        _scf.YieldOp([_val(big)])
    return _copysign(ArithValue(if_op.results[0]), x)


def _tanh_rcp(x):
    # 1 - 2 / (exp(2|x|) + 1), the denominator >= 2 (rcp(inf) = 0); the series below.
    x = _f32(x)
    ax = ArithValue(fmath.absf(x))
    e = _exp2_raw(ArithValue(ax * _cf32(2.0 * _LOG2E)))
    big = ArithValue(fmath.fma(_cf32(-2.0), _rcp(e + _cf32(1.0)), _cf32(1.0)))
    small = ArithValue(ax * _horner(ArithValue(ax * ax), _TANH_SMALL))
    r = ArithValue(ArithValue(ax < _cf32(_SMALL)).select(small, big))
    return _copysign(r, x)


# What a value was computed as -- (op, argument) -- for the rewrites below of the tanh
# GELU's 1 + tanh(u), SiLU's x / (1 + exp(-x)) and a softmax's exp(x - max) / sum:
# "ge1" marks a denominator >= 1 (or NaN), "rowmax" a softmax's row max and "le0" an
# x - rowmax. Those kernels are ALU-bound on MI300 (24-30 instructions an element); the
# ops a rewrite replaces are left unused, dead code.
_derived_memo: dict = {"ctx": None, "map": {}}


def _derived(v):
    if _derived_memo["ctx"] is not ir.Context.current:
        return None
    try:
        return _derived_memo["map"].get(_val(v))
    except (TypeError, ValueError):
        return None


def _derive(r, op: str, arg=None):
    if _derived_memo["ctx"] is not ir.Context.current:
        _derived_memo.update(ctx=ir.Context.current, map={})
    _derived_memo["map"][_val(r)] = (op, arg)
    return r


def _one_plus_of(v):
    """``1 + v`` rewritten, for a ``v`` that ``tanh`` or ``exp`` registered, else
    None. (Not erf: a branch-free rational erf ran the GELU 8% faster at 3x eager's
    error, an f32 evaluation floor of ~3e-7.)"""
    d = _derived(v)
    if d is None:
        return None
    op, u = d
    one = _cf32(1.0)
    if op == "tanh":
        # 2 sigmoid(2u): one exp and one rcp, no cancellation as tanh nears -1.
        e = _exp2_raw(ArithValue(u * _cf32(-2.0 * _LOG2E)))
        return ArithValue(_rcp(e + one) * _cf32(2.0))
    if op == "exp":
        # >= 1, so a division by it can take v_rcp; a denormal exp is as good as 0.
        e = _exp2_raw(ArithValue(u * _cf32(_LOG2E)))
        return _derive(ArithValue(e + one), "ge1")
    return None


def _tanh_registered(x):
    x = _f32(_require_f32("tanh", x))
    return _derive(ArithValue(_tanh_rcp(x)), "tanh", x)


def _exp_registered(x):
    x = _f32(_require_f32("exp", x))
    if (_derived(x) or (None,))[0] == "le0":
        # A softmax's exp(x - max) <= 1: a result in the denormal range, a probability
        # below 1e-38, may as well be 0. LLVM's range fix-up is 5 of the ~27
        # instructions an element of GPT-2's causal softmax.
        return _derive(_exp2_raw(ArithValue(x * _cf32(_LOG2E))), "exp", x)
    return _derive(_exp_f32(x), "exp", x)


def _exp2_f32(x):
    return ArithValue(ArithValue(_f32(x)).exp2())


def _exp_f32(x):
    # exp2 is the native instruction; exp would go through a libcall.
    return _exp2_f32(ArithValue(_f32(x) * _cf32(_LOG2E)))


def _log2_f32(x):
    return ArithValue(fmath.log2(_f32(x)))


def _log_f32(x):
    return ArithValue(_log2_f32(x) * _cf32(_LN2))


def _log10_f32(x):
    return ArithValue(_log2_f32(x) * _cf32(_LN2 / _LN10))


def _sigmoid_rcp(x):
    # 1 / (1 + exp(-x)): the denominator is >= 1, and rcp(inf) = 0 for very negative x.
    e = _exp2_raw(ArithValue(_f32(x) * _cf32(-_LOG2E)))
    return _rcp(e + _cf32(1.0))


# pi/2 as three floats summing to it to ~2^-80, for the Cody-Waite reduction; minimax
# sin(r)/r - 1 and cos(r) - 1 + r^2/2 on |r| <= pi/4 in r^2 (Cephes sinf/cosf, <=1 ulp).
_PIO2 = (1.5707963705062866, -4.371138828673793e-08, -1.7151245100059315e-15)
_SIN_POLY = (-1.6666654611e-1, 8.3321608736e-3, -1.9515295891e-4)
_COS_POLY = (4.166664568298827e-2, -1.388731625493765e-3, 2.443315711809948e-5)
# Past this the three-part reduction loses bits: OCML's Payne-Hanek path takes over.
_TRIG_BIG = 1e5
_TWO_OVER_PI = 2 / _math.pi
_trig_memo: dict = {"ctx": None, "map": {}}


def _trig_reduce(x):
    """``(r, q)`` with ``x = q pi/2 + r``, ``|r| <= pi/4``: shared by a sin and a cos
    of the same value in the same block, as RoPE takes both of every angle."""
    ctx = ir.Context.current
    if _trig_memo["ctx"] is not ctx:
        _trig_memo.update(ctx=ctx, map={})
    key = (x, ir.InsertionPoint.current.block)
    hit = _trig_memo["map"].get(key)
    if hit is not None:
        return hit
    q = ArithValue(fmath.roundeven(ArithValue(x * _cf32(_TWO_OVER_PI))))
    r = x
    for c in _PIO2:
        r = ArithValue(fmath.fma(ArithValue(-q), _cf32(c), r))
    qi = ArithValue(_arith.FPToSIOp(T.i32(), _val(q)).result)
    _trig_memo["map"][key] = (r, qi)
    return r, qi


def _trig(x, cosine: bool):
    # OCML's sin/cos each reduce the angle by Payne-Hanek, ~150 instructions apiece:
    # a Qwen3 k-norm + RoPE with in-kernel angles spent most of its time there.
    x = _f32(x)
    r, q = _trig_reduce(x)
    z = ArithValue(r * r)
    s = ArithValue(fmath.fma(ArithValue(r * z), _horner(z, _SIN_POLY), r))
    c = ArithValue(
        fmath.fma(
            ArithValue(z * z),
            _horner(z, _COS_POLY),
            fmath.fma(z, _cf32(-0.5), _cf32(1.0)),
        )
    )
    if cosine:
        q = ArithValue(q + fx.Int32(1))
    odd = ArithValue((q & fx.Int32(1)) != fx.Int32(0))
    neg_ = ArithValue((q & fx.Int32(2)) != fx.Int32(0))
    v = ArithValue(odd.select(c, s))
    v = ArithValue(neg_.select(ArithValue(-v), v))
    big = ArithValue(ArithValue(fmath.absf(x)) > _cf32(_TRIG_BIG))
    if_op = _scf.IfOp(_val(big), [T.f32()], has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        _scf.YieldOp([_val((fmath.cos if cosine else fmath.sin)(_f32(x)))])
    with ir.InsertionPoint(if_op.else_block):
        _scf.YieldOp([_val(v)])
    return ArithValue(if_op.results[0])


def sin(x):
    return _trig(_require_f32("sin", x), False)


def cos(x):
    return _trig(_require_f32("cos", x), True)


exp = _exp_registered
exp2 = _unary_f32("exp2", _exp2_f32)
log = _unary_f32("log", _log_f32)
log2 = _unary_f32("log2", _log2_f32)
log10 = _unary_f32("log10", _log10_f32)
tanh = _tanh_registered
sigmoid = _unary_f32("sigmoid", _sigmoid_rcp)
erf = _unary_f32("erf", _erf_rcp)


# -- ROCm device library (ocml), as Triton's libdevice on ROCm -----------------------------

_OCML_TYPES = {"f32": "float32", "f64": "float64", "i32": "int32"}


def _device_libs() -> list:
    """ocml and the control libraries it reads, as Triton sets them: IEEE (not finite
    only, no unsafe math, denormals kept), a correctly rounded sqrt, wave64."""
    from flydsl.runtime.device import get_rocm_arch

    root = _os.path.join(_os.environ.get("ROCM_PATH", "/opt/rocm"), "amdgcn", "bitcode")
    names = [
        "ocml",
        "oclc_finite_only_off",
        "oclc_unsafe_math_off",
        "oclc_daz_opt_off",
        "oclc_correctly_rounded_sqrt_on",
        "oclc_wavefrontsize64_on",
        f"oclc_isa_version_{get_rocm_arch()[3:]}",
        "oclc_abi_version_500",
    ]
    return [_os.path.join(root, f"{n}.bc") for n in names]


def _ocml(name: str, ret: str, *args):
    """``__ocml_<name>`` on bare values of the kinds their types name (``ret`` the
    result's): a pure call into the linked device library."""
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr.extern import ExternFunction

    ctx = CompilationContext.get_current()
    for lib in _device_libs():
        ctx.add_link_lib(lib)
    types = [_OCML_TYPES[kind_of_type(a.type)] for a in args]
    fn = ExternFunction(f"__ocml_{name}", types, _OCML_TYPES[ret], is_pure=True)
    return ArithValue(fn(*args))


def _ocml_unary(name: str):
    def fn(x):
        k = _float_kind(x)
        return _ocml(f"{name}_{k}", k, _val(to(x, k)))

    fn.__name__ = name
    return fn


# The f32 formulas these replace lost the tails: log1p(x) as log(1 + x), expm1, sinh,
# asinh and atanh 8e6 ulp at small x, erfc as it nears denormal, the inverse
# trigonometric functions 1e3 ulp.
erfc = _ocml_unary("erfc")
expm1 = _ocml_unary("expm1")
log1p = _ocml_unary("log1p")
sinh = _ocml_unary("sinh")
cosh = _ocml_unary("cosh")
asin = _ocml_unary("asin")
acos = _ocml_unary("acos")
atan = _ocml_unary("atan")
asinh = _ocml_unary("asinh")
acosh = _ocml_unary("acosh")
atanh = _ocml_unary("atanh")
lgamma = _ocml_unary("lgamma")
erfinv = _ocml_unary("erfinv")
erfcx = _ocml_unary("erfcx")
bessel_j0 = _ocml_unary("j0")
bessel_j1 = _ocml_unary("j1")
bessel_y0 = _ocml_unary("y0")
bessel_y1 = _ocml_unary("y1")
i0 = modified_bessel_i0 = _ocml_unary("i0")
i1 = modified_bessel_i1 = _ocml_unary("i1")


def nextafter(a, b):
    k = _float_kind(a, b)
    return _ocml(f"nextafter_{k}", k, _val(to(a, k)), _val(to(b, k)))


def copysign(a, b):
    k = _float_kind(a, b)
    return ArithValue(fmath.copysign(to(a, k), to(b, k)))


def ldexp(x, n):
    k = _float_kind(x)
    return _ocml(f"ldexp_{k}", k, _val(to(x, k)), _val(idx32(n)))


def inline_asm(asm: str, constraints: str, out: str, pure: bool, *xs, pack: int = 1):
    """``tl.inline_asm_elementwise``: the operands in their compute kinds, the result
    in storage kind ``out`` (as Triton types them), widened back.

    With ``pack`` > 1 an instance takes ``pack`` elements: Triton's operands are the
    outputs, then each input's ``pack`` elements, results narrower than 32 bits packed
    into 32-bit registers. Elementwise asm computes each element alone, so this
    element fills every slot and keeps the first result.
    """
    from flydsl._mlir.dialects import llvm as _llvm

    vals = [
        _val(_canon(x) if not isinstance(x, (bool, int, float)) else const(x, kind(x)))
        for x in xs
    ]
    et = _STORAGE_IR[out]()
    if pack == 1:
        op = _llvm.InlineAsmOp(et, vals, asm, constraints, has_side_effects=not pure)
        res = op.result
    else:
        per_reg = max(1, min(pack, 32 // (8 * _type_bytes(et))))
        reg = et if per_reg == 1 else ir.VectorType.get([per_reg], et)
        regs = pack // per_reg
        st = _llvm.StructType.get_literal([reg] * regs)
        operands = [v for v in vals for _ in range(pack)]
        op = _llvm.InlineAsmOp(
            st, operands, asm, constraints, has_side_effects=not pure
        )
        res = _llvm.ExtractValueOp(reg, op.result, [0]).result
        if per_reg > 1:
            res = _vector.ExtractOp(
                res, static_position=[0], dynamic_position=[]
            ).result
    return ArithValue(_cvt(res, _MEM_KIND.get(out, out), _COMPUTE_OF[out]))


def frexp(x):
    """(mantissa in [0.5, 1) with x's sign, exponent): ``x`` itself and 0 for zero,
    inf and NaN. A denormal is scaled into the normal range first."""
    x = to(x, "f32")

    def i32(c):
        return const(c, "i32")

    bits = ArithValue(_arith.BitcastOp(T.i32(), _val(x)).result)
    tiny = ArithValue(fmath.absf(x)) < _cf32(1.1754943508222875e-38)
    scaled = ArithValue(_arith.BitcastOp(T.i32(), _val(x * _cf32(16777216.0))).result)
    bits = where(tiny, scaled, bits)
    biased = bitwise_and(bitwise_right_shift(bits, i32(23)), i32(0xFF))
    e = sub(biased, where(tiny, i32(126 + 24), i32(126)))
    m = bitwise_or(bitwise_and(bits, i32(-0x7F800001)), i32(126 << 23))
    mant = ArithValue(_arith.BitcastOp(T.f32(), _val(m)).result)
    special = logical_or(eq(x, _cf32(0.0)), logical_or(isnan(x), isinf(x)))
    return where(special, x, mant), where(special, i32(0), e)


def tan(x):
    return truediv(sin(x), cos(x))


def atan2(a, b):
    k = _float_kind(a, b)
    return _ocml(f"atan2_{k}", k, _val(to(a, k)), _val(to(b, k)))


def pow(a, b):
    k, (x, y) = _promote(a, b)
    if not _is_f(k):
        # Integer pow by repeated squaring is not worth a loop here; go through f32 only
        # when exact, which it is for the small exponents Inductor emits.
        return to(ArithValue(fmath.powf(_f32(x), _f32(y))), k)
    if k == "f64":
        raise NotImplementedError(
            "pow on float64 has no AMDGPU lowering in the FlyDSL SIMD backend"
        )
    return ArithValue(fmath.powf(x, y))


def hypot(a, b):
    return sqrt(add(mul(a, a), mul(b, b)))


def identity(x):
    return x


# ---------------------------------------------------------------------------------------
# Random numbers: Philox-4x32-10, the same stream as Triton's tl.rand/tl.randn/randint
# ---------------------------------------------------------------------------------------

_PHILOX_KEY_A, _PHILOX_KEY_B = 0x9E3779B9, 0xBB67AE85
_PHILOX_ROUND_A, _PHILOX_ROUND_B = 0xD2511F53, 0xCD9E8D57


def _u32c(v: int):
    return _val(const(v - 2**32 if v >= 2**31 else v, "i32"))


def _philox(seed, offset, c1=None, c2=None, c3=None):
    """Four uint32 words (held as i32) for counter (offset, c1, c2, c3) under key
    ``seed``; a counter word left None is 0."""
    s = _val(to(seed, "i64"))
    k0 = _arith.TruncIOp(T.i32(), s).result
    k1 = _arith.TruncIOp(
        T.i32(),
        _arith.ShRUIOp(s, _val(const(32, "i64"))).result,
    ).result
    c0 = _val(idx32(offset))
    c1, c2, c3 = (None if c is None else _val(idx32(c)) for c in (c1, c2, c3))
    a, b = _u32c(_PHILOX_ROUND_A), _u32c(_PHILOX_ROUND_B)
    ka, kb = _u32c(_PHILOX_KEY_A), _u32c(_PHILOX_KEY_B)

    def xor(*vs):
        vs = [v for v in vs if v is not None]
        out = vs[0]
        for v in vs[1:]:
            out = _arith.XOrIOp(out, v).result
        return out

    for _ in range(10):
        # c1..c3 start at zero; skipping them keeps the first round short.
        hb = _arith.MulUIExtendedOp(b, c2).results if c2 is not None else None
        ha = _arith.MulUIExtendedOp(a, c0).results
        n0 = xor(hb[1] if hb else None, c1, k0)
        n2 = xor(ha[1], c3, k1)
        c1 = hb[0] if hb else _u32c(0)
        c3 = ha[0]
        c0, c2 = n0, n2
        k0 = _arith.AddIOp(k0, ka).result
        k1 = _arith.AddIOp(k1, kb).result
    return c0, c1, c2, c3


def _uniform(word):
    # Triton's uint_to_uniform_float: reinterpret as int32, fold negatives, scale 2**-31.
    x = ArithValue(word)
    x = ArithValue(
        _arith.SelectOp(
            _val(lt(x, const(0, "i32"))), _val(const(-1, "i32") - x), word
        ).result
    )
    return mul(to(x, "f32"), const(4.6566127342e-10, "f32"))


def rand(seed, offset):
    return _uniform(_philox(seed, offset)[0])


def randn(seed, offset):
    w = _philox(seed, offset)
    u1 = maximum(_uniform(w[0]), const(1.0e-7, "f32"))
    th = mul(_uniform(w[1]), const(6.283185307179586, "f32"))
    return mul(sqrt(mul(const(-2.0, "f32"), log(u1))), cos(th))


def _philox_group(cache, key, j: int, seed, offset):
    """The four Philox words of offset group ``offset // 4``, computed once per group of
    four consecutive elements (``j`` counts a thread's elements; groups are aligned)."""

    def words():
        base = sub(idx(offset), j % 4) if j % 4 else idx(offset)
        return _philox(seed, idiv(base, 4))

    return memo(cache, (key, j // 4), words)


def rand4(cache, key, j: int, seed, offset):
    return _uniform(_philox_group(cache, key, j, seed, offset)[j % 4])


def randn4(cache, key, j: int, seed, offset):
    def normals():
        w = _philox_group(cache, key, j, seed, offset)
        out = []
        for a, b in ((w[0], w[1]), (w[2], w[3])):
            u1 = maximum(_uniform(a), const(1.0e-7, "f32"))
            th = mul(_uniform(b), const(6.283185307179586, "f32"))
            r = sqrt(mul(const(-2.0, "f32"), log(u1)))
            out += [mul(r, cos(th)), mul(r, sin(th))]
        return out

    return memo(cache, (key, "n", j // 4), normals)[j % 4]


def rand_eager(seed, base_offset, threads_per_round: int, tid, vec: int):
    """``triton_helpers.rand_eager_kernel``: eager's Philox stream for element ``tid``
    (counter: 64-bit offset block, 64-bit subsequence), so a compiled ``rand`` matches
    eager's under ``config.align_random_eager``."""
    denom = vec * threads_per_round
    t = _val(idx32(tid))

    def i32(c):
        return _val(const(c, "i32"))

    t_round = _arith.DivUIOp(t, i32(denom)).result
    t_trunc = _arith.RemUIOp(t, i32(denom)).result
    subseq = _arith.DivUIOp(t_trunc, i32(vec)).result
    which4 = _arith.DivUIOp(_arith.RemUIOp(t_trunc, i32(vec)).result, i32(4)).result
    lane = _arith.RemUIOp(t_trunc, i32(4)).result
    blocks = _arith.MulIOp(t_round, i32(vec // 4)).result

    def ext(v):
        return _arith.ExtUIOp(T.i64(), v).result

    off = _arith.AddIOp(
        _val(to(base_offset, "i64")), ext(_arith.AddIOp(blocks, which4).result)
    ).result
    lo = _arith.TruncIOp(T.i32(), off).result
    hi = _arith.TruncIOp(
        T.i32(), _arith.ShRUIOp(off, _val(const(32, "i64"))).result
    ).result
    u = _philox(seed, lo, hi, subseq, None)

    def pick(c, a, b):
        eq_c = _arith.CmpIOp(_arith.CmpIPredicate.eq, lane, i32(c)).result
        return _arith.SelectOp(eq_c, a, b).result

    word = pick(0, u[0], pick(1, u[1], pick(2, u[2], u[3])))
    f = _arith.UIToFPOp(T.f32(), word).result
    inv = 1.0 / 4294967296.0
    r = ArithValue(fmath.fma(f, _cf32(inv), _cf32(inv * 0.5)))
    return sub(const(1.0, "f32"), r)


def randint64(seed, offset, low, high):
    w = _philox(seed, offset)
    lo = _arith.ExtUIOp(T.i64(), w[0]).result
    hi = _arith.ExtUIOp(T.i64(), w[1]).result
    r = _arith.OrIOp(lo, _arith.ShLIOp(hi, _val(const(32, "i64"))).result).result
    low, high = to(low, "i64"), to(high, "i64")
    size = _val(sub(high, low))
    return add(ArithValue(_arith.RemUIOp(r, size).result), low)


# ---------------------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------------------


def _combine_fn(op: str):
    return {
        "sum": add,
        "prod": mul,
        "max": maximum,
        "min": minimum,
        "fmax": lambda a, b: ArithValue(_arith.MaxNumFOp(_val(a), _val(b)).result),
        "any": logical_or,
        "xor_sum": bitwise_xor,
    }[op]


def default(op: str, k: str):
    """Identity element of a reduction, in compute kind ``k``."""
    if op == "sum" or op == "xor_sum":
        return const(0, k)
    if op == "prod":
        return const(1, k)
    if op == "any":
        return const(False, "b1")
    if op in ("max", "fmax"):
        if _is_f(k):
            return const(float("-inf"), k)
        return const(-(2**63) if k == "i64" else (False if k == "b1" else -(2**31)), k)
    if op == "min":
        if _is_f(k):
            return const(float("inf"), k)
        return const(2**63 - 1 if k == "i64" else (True if k == "b1" else 2**31 - 1), k)
    raise NotImplementedError(op)


def combine(op: str, acc, value, mask=None):
    """``acc <- op(acc, value)``, leaving ``acc`` untouched where ``mask`` is false."""
    acc = _canon(acc)
    k = kind(acc)
    v = to(value, k)
    new = _canon(_combine_fn(op)(acc, v))
    m = _mask(mask)
    if m is None:
        return new
    return ArithValue(_arith.SelectOp(m, _val(new), _val(acc)).result)


def _shfl(x, off, width: int, mode: str = "xor"):
    """``gpu.shuffle`` for any compute kind: 64-bit values travel as two dwords."""
    from flydsl._mlir.dialects.gpu import ShuffleOp

    def one(word):
        return ShuffleOp(
            word, _val(idx32(off)), _val(fx.Int32(width)), mode=mode
        ).shuffleResult

    words = _to_words(x)
    return _from_words([one(w) for w in words], kind(_canon(x)))


_DPP_ROW_SHR = 0x110  # + n: lane i reads lane i - n of its row of 16
_DPP_ROW_BCAST15 = 0x142  # lane 15 of each row to the whole next row
_DPP_ROW_BCAST31 = 0x143  # lane 31 to rows 2 and 3


def _dpp(x, ctrl: int, row_mask: int = 0xF, old=None):
    """Lane movement ``ctrl``; a lane with no source (or outside ``row_mask``) gets
    ``old``, or its own value."""
    olds = _to_words(old) if old is not None else None

    def one(i, word):
        o = olds[i] if olds is not None else word
        return _rocdl.update_dpp(T.i32(), o, word, ctrl, row_mask, 0xF, False)

    return _from_words([one(i, w) for i, w in enumerate(_to_words(x))], kind(_canon(x)))


def _readlane(x, lane: int):
    c = _val(fx.Int32(lane))
    words = [_rocdl.readlane(T.i32(), w, c) for w in _to_words(x)]
    return _from_words(words, kind(_canon(x)))


def _dpp_reduce(vals, comb, tpr: int, ident=None):
    """Reduce over groups of ``tpr`` consecutive lanes (a power of two, at most a wave)
    with DPP, as LLVM's and Triton's wave reductions do: a scan within each row of 16
    by row shifts, then row broadcasts across rows, which leaves the group total in its
    last lane; that lane is then read back. Given ``comb``'s identity (one per value),
    lanes a step has no source for read it and every lane combines, so LLVM folds each
    move into its combine (``v_add_f32_dpp``); without one they keep their value
    through a select. The scan order keeps ``comb(earlier, later)``. Instruction count
    is what counts: a 768-wide LayerNorm is vector-ALU-bound, and went 141us -> 110us
    with the identity, where DPP alone measured even with ``ds_bpermute``."""
    lane = imod(_tid(), WAVE)

    def step(ctrl, row_mask, ok):
        if ident is not None:
            moved = tuple(_dpp(v, ctrl, row_mask, i) for v, i in zip(vals, ident))
            return tuple(_canon(c) for c in comb(moved, vals))
        moved = tuple(_dpp(v, ctrl, row_mask) for v in vals)
        return tuple(_canon(where(ok, c, v)) for c, v in zip(comb(moved, vals), vals))

    k = 1
    while k < min(tpr, 16):
        vals = step(_DPP_ROW_SHR + k, 0xF, ge(imod(lane, 16), k))
        k *= 2
    if tpr >= 32:
        vals = step(_DPP_ROW_BCAST15, 0xA, eq(imod(idiv(lane, 16), 2), 1))
    if tpr == WAVE:
        vals = step(_DPP_ROW_BCAST31, 0xC, ge(lane, 32))
        return tuple(_canon(_readlane(v, WAVE - 1)) for v in vals)
    if tpr == 32:
        hi = ge(lane, 32)
        return tuple(
            _canon(where(hi, _readlane(v, 63), _readlane(v, 31))) for v in vals
        )
    last = lane - imod(lane, tpr) + (tpr - 1)
    return tuple(_canon(_shfl(v, last, WAVE, "idx")) for v in vals)


def _to_words(x):
    """A compute value as a list of i32 words, for LDS."""
    x = _canon(x)
    k = kind(x)
    if k == "i32":
        return [_val(x)]
    if k == "f32":
        return [_arith.BitcastOp(T.i32(), _val(x)).result]
    if k == "b1":
        return [_cvt(_val(x), "b1", "i32")]
    bits = _val(x) if k == "i64" else _arith.BitcastOp(T.i64(), _val(x)).result
    c32 = _arith.ConstantOp(T.i64(), ir.IntegerAttr.get(T.i64(), 32)).result
    return [
        _arith.TruncIOp(T.i32(), bits).result,
        _arith.TruncIOp(T.i32(), _arith.ShRUIOp(bits, c32).result).result,
    ]


def _from_words(words, k):
    if k == "i32":
        return ArithValue(words[0])
    if k == "f32":
        return ArithValue(_arith.BitcastOp(T.f32(), words[0]).result)
    if k == "b1":
        return ArithValue(_cvt(words[0], "i32", "b1"))
    c32 = _arith.ConstantOp(T.i64(), ir.IntegerAttr.get(T.i64(), 32)).result
    joined = _arith.OrIOp(
        _arith.ExtUIOp(T.i64(), words[0]).result,
        _arith.ShLIOp(_arith.ExtUIOp(T.i64(), words[1]).result, c32).result,
    ).result
    return ArithValue(
        joined if k == "i64" else _arith.BitcastOp(T.f64(), joined).result
    )


def _group_reduce(
    vals, comb, tpr: int, smem, threads: int, stride: int = 1, ident=None
):
    """Reduce a tuple of per-thread values over groups of ``tpr`` threads ``stride`` apart.

    A group is ``{base + k * stride : k < tpr}``: consecutive threads for a row reduction
    (``stride == 1``), or one thread per wave-slice of a column for a column reduction,
    where lanes run along the contiguous dimension. Every member ends up holding the
    result. Members within a wave combine by butterfly shuffles; if the group spans
    waves, every thread then parks its value in LDS slot ``t`` and, after a barrier,
    folds one slot per wave of its group -- serial reads, no second barrier-guarded
    store, so nothing here needs a dynamic branch.
    """
    vals = tuple(_canon(v) for v in vals)
    if tpr <= 1:
        return vals
    # A group not a whole number of waves (96 lanes) starts mid-wave: within a wave,
    # combine over the power-of-two segments its groups are made of.
    seg = min(tpr & -tpr, WAVE) if stride == 1 else WAVE // stride
    if stride == 1:
        vals = _dpp_reduce(vals, comb, seg, ident)
    else:
        off = stride
        while off < WAVE and off < stride * tpr:
            peer = tuple(_shfl(v, off, WAVE) for v in vals)
            vals = comb(vals, peer)
            off *= 2
    if stride * tpr <= WAVE and (stride > 1 or seg == tpr):
        return vals
    t = _tid()
    kinds = [kind(v) for v in vals]
    words = [_to_words(v) for v in vals]
    slot = 0
    for w in words:
        for word in w:
            fx.memref_store(ArithValue(word), smem, t + slot * threads)
            slot += 1
    gpu.barrier()
    rank = (t // stride) % tpr
    base = t - rank * stride
    kstep = max(1, seg)
    acc = None
    for k in range(0, tpr, kstep):
        member = base + k * stride
        slot = 0
        parts = []
        for w, kd in zip(words, kinds):
            ws = []
            for _ in w:
                ws.append(
                    _val(ArithValue(fx.memref_load(smem, member + slot * threads)))
                )
                slot += 1
            parts.append(_from_words(ws, kd))
        parts = tuple(parts)
        acc = parts if acc is None else comb(acc, parts)
    gpu.barrier()
    return acc


def scan_first(r0_offset, j: int):
    """Whether run ``j`` of the step at ``r0_offset`` starts the row (no carry yet)."""
    if j:
        return False
    if isinstance(r0_offset, int):
        return r0_offset == 0
    return eq(r0_offset, 0)


def scan_init(*kinds):
    return tuple(const(0, k) for k in kinds)


def group_scan(
    fn, values, carry, first, tpr: int, lane, smem=None, base: int = 0, split=None
):
    """Inclusive scan of one value per thread over each group of ``tpr`` consecutive
    lanes, continuing from ``carry``, the previous runs' total. A group wider than a
    wave goes through LDS at word ``base`` of ``smem`` (``scan_smem_words``).

    ``fn(earlier, later)`` is the combine on tuples. Returns the scanned values followed
    by the carry for the next run, as one flat tuple. ``first`` (a trace-time or device
    bool) discards ``carry``.
    """
    out, nxt = _group_scan(fn, values, carry, first, tpr, lane, smem, base, split)
    return (*out, *nxt)


def scan_smem_words(threads: int, components: int) -> int:
    """LDS words ``group_scan`` needs: two per component per wave."""
    return threads // WAVE * components * 2


def group_scan_runs(
    fn,
    values,
    carry,
    first,
    tpr: int,
    lane,
    state: dict,
    key,
    j: int,
    ept: int,
    phase: int,
    smem=None,
    base: int = 0,
    split=None,
):
    """``group_scan`` where each thread owns runs of ``ept`` contiguous elements.

    The body is traced at least twice per step. Phase 0 only records element ``j``'s
    value (its results are unused, so the trace is dead code). In phase 1 the run's
    first element scans the thread's values in registers, scans the per-thread totals
    across the lanes, and then offsets every element by the lanes before it and the
    carry, so a run costs one cross-lane scan rather than ``ept``; later phases (for a
    scan of this one's results) reuse those. Codegen numbers the phases relative to
    the scan's nesting.
    """
    run, e = j // ept, j % ept
    if phase <= 0:
        # A chained scan's earlier passes see its input unfinished; the last before
        # its results records it.
        if e == 0:
            state.pop((key, run, "out"), None)
        state[(key, run, e)] = tuple(_canon(v) for v in values)
        return (*state[(key, run, e)], *carry)
    if e == 0 and (key, run, "out") not in state:
        pre = [state[(key, run, 0)]]
        for k in range(1, ept):
            pre.append(tuple(_canon(v) for v in fn(pre[-1], state[(key, run, k)])))
        inc, nxt, before = _group_scan_waves(
            fn, pre[-1], carry, first, tpr, lane, smem, base
        )
        # inc is inclusive of this lane's total; the lane before holds the exclusive,
        # unless that lane is in the wave before.
        ex = tuple(_shfl(v, 1, WAVE, "up") for v in inc)
        if before is not None:
            wave_head = eq(_tid() % WAVE, 0)
            ex = tuple(where(wave_head, b, e) for b, e in zip(before, ex))
        lead = eq(lane, 0)
        outs = []
        for p in pre:
            after = tuple(_canon(v) for v in fn(ex, p))
            if first is True:
                head = p
            else:
                c = tuple(_canon(v) for v in fn(carry, p))
                head = (
                    c
                    if first is False
                    else tuple(where(first, a, b) for a, b in zip(p, c))
                )
            outs.append(tuple(where(lead, h, a) for h, a in zip(head, after)))
        if split is not None:
            at = base + scan_smem_words(tpr, len(nxt))
            pre = _split_lookback(fn, nxt, smem, at, split)
            later = ne(split[2], 0)
            outs = [
                tuple(where(later, s, o) for s, o in zip(fn(pre, out), out))
                for out in outs
            ]
        state[(key, run, "out")] = (outs, nxt)
    outs, nxt = state[(key, run, "out")]
    return (*outs[e], *nxt)


def _group_scan(
    fn, values, carry, first, tpr: int, lane, smem=None, base: int = 0, split=None
):
    out, nxt, _ = _group_scan_waves(fn, values, carry, first, tpr, lane, smem, base)
    if split is not None:
        out = _split_offset(fn, out, nxt, smem, base, tpr, len(out), split)
    return out, nxt


# Words of a chunk's slot in a split scan's workspace: [flag, aggregate (2 words),
# inclusive prefix (2 words)], padded. Flag 0: not yet published, 1: aggregate, 2: both.
# A single 32-bit value packs (flag << 32 | value) into the slot's first 64-bit word.
SPLIT_SLOT_WORDS = 8


def _split_offset(fn, out, total, smem, base, tpr, n, split):
    """Offset the block's scanned values ``out`` by the chunks of the row before it."""
    pre = _split_lookback(fn, total, smem, base + scan_smem_words(tpr, n), split)
    later = ne(split[2], 0)
    return tuple(where(later, s, o) for s, o in zip(fn(pre, out), out))


def split_smem_words(threads: int, components: int) -> int:
    """LDS words a split scan needs: ``group_scan``'s, then the broadcast prefix."""
    return scan_smem_words(threads, components) + 2 * components


def _split_lookback(fn, total, smem, at: int, split):
    """Decoupled look-back (Merrill & Garland): thread 0 publishes this chunk's total,
    folds in the earlier chunks' -- an aggregate each, back to one whose inclusive
    prefix is out -- then publishes its own inclusive prefix. Returns the exclusive
    prefix, to every thread through LDS at word ``at``. ``split`` is (workspace raw
    address, this chunk's slot, its index in the row); slots are zero-filled, and
    a row's chunks are consecutive slots, dispatched in order.
    """
    from flydsl._mlir.dialects import llvm as _llvm

    ws, slot, chunk = split
    r = _Flat(_val(to(ws, "i64")))
    total = tuple(_canon(v) for v in total)
    kinds = [kind(t) for t in total]
    n = len(total)
    if n == 1 and len(_to_words(total[0])) == 1:
        return _split_lookback_packed(fn, total[0], smem, at, r, slot, chunk)
    i32 = T.i32()
    i1 = ir.IntegerType.get_signless(1)
    order = _llvm.AtomicOrdering

    def ptr(s, w):
        return _flat_ptr(r, ArithValue(s) * SPLIT_SLOT_WORDS + w, 4)

    def c32(v):
        return _arith.ConstantOp(i32, ir.IntegerAttr.get(i32, v)).result

    def flag_store(s, f):
        _llvm.StoreOp(
            f, ptr(s, 0), ordering=order.release, syncscope="agent", alignment=4
        )

    def emit():
        me = _val(idx32(slot))
        first = _val(eq(chunk, 0))
        words = [w for t in total for w in _to_words(t)]
        for i, w in enumerate(words):
            _llvm.StoreOp(w, ptr(me, 1 + i), alignment=4)
            _llvm.StoreOp(w, ptr(me, 3 + i), alignment=4)
        flag_store(me, _arith.SelectOp(first, c32(2), c32(1)).result)
        acc0 = [_val(t) for t in total]
        types = [i32, i1, *[a.type for a in acc0]]
        prev = _arith.SubIOp(me, c32(1)).result
        loop = _scf.WhileOp(types, [prev, first, *acc0])
        before = loop.before.blocks.append(*types)
        with ir.InsertionPoint(before):
            done = before.arguments[1]
            go = _arith.XOrIOp(done, _val(const(True, "b1"))).result
            _scf.ConditionOp(go, list(before.arguments))
        after = loop.after.blocks.append(*types)
        with ir.InsertionPoint(after):
            k, _, *acc = after.arguments
            flag = _llvm.LoadOp(
                i32, ptr(k, 0), ordering=order.acquire, syncscope="agent", alignment=4
            ).result
            ready = _val(ne(ArithValue(flag), 0))
            inclusive = _val(eq(ArithValue(flag), 2))
            vals, w = [], 0
            for kd in kinds:
                nw = 2 if kd in ("i64", "f64") else 1
                ws_ = []
                for i in range(nw):
                    a = _llvm.LoadOp(i32, ptr(k, 1 + w + i), alignment=4).result
                    b = _llvm.LoadOp(i32, ptr(k, 3 + w + i), alignment=4).result
                    ws_.append(_arith.SelectOp(inclusive, b, a).result)
                vals.append(_canon(_from_words(ws_, kd)))
                w += nw
            # The first chunk found starts the accumulation; acc holds this chunk's
            # total until then.
            started = _val(ne(ArithValue(k), ArithValue(prev)))
            comb = fn(tuple(vals), tuple(ArithValue(a) for a in acc))
            nxt = []
            for a, v, c in zip(acc, vals, comb):
                folded = _arith.SelectOp(started, _val(_canon(c)), _val(v)).result
                nxt.append(_arith.SelectOp(ready, folded, a).result)
            k2 = _arith.SelectOp(ready, _arith.SubIOp(k, c32(1)).result, k).result
            _scf.YieldOp([k2, inclusive, *nxt])
        pre = tuple(ArithValue(v) for v in loop.results[2:])
        inc = tuple(_canon(v) for v in fn(pre, total))

        # Only a later chunk publishes again; the first's inclusive prefix is out.
        def publish():
            for i, w in enumerate(x for t in inc for x in _to_words(t)):
                _llvm.StoreOp(w, ptr(me, 3 + i), alignment=4)
            flag_store(me, c32(2))

        _when(_arith.XOrIOp(first, _val(const(True, "b1"))).result, publish)
        for c, p in enumerate(pre):
            tile_put(smem, at, 1, 2 * c, p)

    _when(_val(eq(_tid(), 0)), emit)
    barrier()
    return tuple(tile_get(smem, at, 1, 2 * c, kinds[c]) for c in range(n))


def _split_lookback_packed(fn, total, smem, at: int, r, slot, chunk):
    """``_split_lookback`` of one 32-bit value: (flag << 32 | value) in a slot's first
    64-bit word, published and polled by relaxed agent-scope atomics, as Triton's
    look-back. The release store and acquire loads of the general form cost an L2
    write-back and invalidate a hop on gfx942: ~3us against ~0.5us."""
    from flydsl._mlir.dialects import llvm as _llvm

    k0 = kind(total)
    i32, i64 = T.i32(), T.i64()
    i1 = ir.IntegerType.get_signless(1)
    mono = _llvm.AtomicOrdering.monotonic

    def c(t, v):
        return _arith.ConstantOp(t, ir.IntegerAttr.get(t, v)).result

    def ptr(s):
        return _flat_ptr(r, ArithValue(s) * SPLIT_SLOT_WORDS, 4)

    def pack(flag, v):
        w = _arith.ExtUIOp(i64, _to_words(v)[0]).result
        f = _arith.ShLIOp(_arith.ExtUIOp(i64, flag).result, c(i64, 32)).result
        return _arith.OrIOp(f, w).result

    def publish(s, flag, v):
        _llvm.StoreOp(
            pack(flag, v), ptr(s), ordering=mono, syncscope="agent", alignment=8
        )

    def emit():
        me = _val(idx32(slot))
        first = _val(eq(chunk, 0))
        publish(me, _arith.SelectOp(first, c(i32, 2), c(i32, 1)).result, total)
        prev = _arith.SubIOp(me, c(i32, 1)).result
        acc0 = _val(total)
        types = [i32, i1, acc0.type]
        loop = _scf.WhileOp(types, [prev, first, acc0])
        before = loop.before.blocks.append(*types)
        with ir.InsertionPoint(before):
            go = _arith.XOrIOp(before.arguments[1], _val(const(True, "b1"))).result
            _scf.ConditionOp(go, list(before.arguments))
        after = loop.after.blocks.append(*types)
        with ir.InsertionPoint(after):
            k, _, acc = after.arguments
            p = _llvm.LoadOp(
                i64, ptr(k), ordering=mono, syncscope="agent", alignment=8
            ).result
            flag = _arith.TruncIOp(i32, _arith.ShRUIOp(p, c(i64, 32)).result).result
            v = _canon(_from_words([_arith.TruncIOp(i32, p).result], k0))
            ready = _val(ne(ArithValue(flag), 0))
            inclusive = _val(eq(ArithValue(flag), 2))
            started = _val(ne(ArithValue(k), ArithValue(prev)))
            folded = _arith.SelectOp(
                started, _val(_canon(fn((v,), (ArithValue(acc),))[0])), _val(v)
            ).result
            nxt = _arith.SelectOp(ready, folded, acc).result
            k2 = _arith.SelectOp(ready, _arith.SubIOp(k, c(i32, 1)).result, k).result
            _scf.YieldOp([k2, inclusive, nxt])
        pre = ArithValue(loop.results[2])
        inc = _canon(fn((pre,), (total,))[0])
        _when(
            _arith.XOrIOp(first, _val(const(True, "b1"))).result,
            lambda: publish(me, c(i32, 2), inc),
        )
        tile_put(smem, at, 1, 0, pre)

    _when(_val(eq(_tid(), 0)), emit)
    barrier()
    return (tile_get(smem, at, 1, 0, k0),)


def _group_scan_waves(fn, values, carry, first, tpr, lane, smem, base):
    """``_group_scan``, plus (for a group of several waves) what precedes each wave:
    the carry and the totals of the waves before it, for its first lane."""
    xs = tuple(_canon(v) for v in values)
    width = min(tpr, WAVE)
    wl = _tid() % WAVE if tpr > WAVE else lane
    off = 1
    while off < width:
        ys = tuple(_shfl(x, off, WAVE, "up") for x in xs)
        comb = fn(ys, xs)
        ok = ge(wl, off)
        xs = tuple(where(ok, c, x) for c, x in zip(comb, xs))
        off *= 2
    if width > 1:
        last = _tid() % WAVE - wl + (width - 1)
        total = tuple(_shfl(x, last, WAVE, "idx") for x in xs)
    else:
        total = xs
    before = None
    if tpr > WAVE:
        # The waves' totals through LDS (every lane of a wave writes the same words),
        # then each wave folds in those of the waves before it.
        nw, n = tpr // WAVE, len(xs)
        kinds = [kind(t) for t in total]
        wave = _tid() // WAVE
        w = wave % nw
        row0 = wave - w

        def slot(wv, c):
            return (wv * n + c) * 2

        barrier()
        for c, t in enumerate(total):
            tile_put(smem, base, 1, slot(wave, c), t)
        barrier()
        totals = [
            tuple(
                tile_get(smem, base, 1, slot(row0 + k, c), kinds[c]) for c in range(n)
            )
            for k in range(nw)
        ]
        acc, pre = totals[0], totals[0]
        for k in range(1, nw):
            pre = tuple(where(eq(w, k), a, p) for a, p in zip(acc, pre))
            acc = tuple(_canon(v) for v in fn(acc, totals[k]))
        later = ge(w, 1)
        xs = tuple(where(later, s, x) for s, x in zip(fn(pre, xs), xs))
        before, total = pre, acc
    if first is True:
        return xs, total, before
    full = tuple(_canon(v) for v in fn(carry, xs))
    nxt = tuple(_canon(v) for v in fn(carry, total))
    if before is not None:
        before_c = tuple(_canon(v) for v in fn(carry, before))
        if first is not False:
            before_c = tuple(where(first, b, c) for b, c in zip(before, before_c))
        before = before_c
    if first is False:
        return full, nxt, before
    return (
        tuple(where(first, x, f) for x, f in zip(xs, full)),
        tuple(where(first, t, n) for t, n in zip(total, nxt)),
        before,
    )


def _words_of(k: str) -> int:
    return 2 if k in ("i64", "f64") else 1


def sort_smem_words(rows: int, p: int, kinds) -> int:
    """LDS words ``row_sort`` needs: ``p`` (key, index, valid) slots for each of
    ``rows``."""
    return rows * p * (1 + sum(_words_of(k) for k in kinds))


def row_sort(
    state: dict,
    key,
    phase: int,
    values,
    pos,
    row,
    smem,
    base: int,
    n: int,
    p: int,
    rows: int,
    threads: int,
    kinds,
    stable: bool,
    descending: bool,
    run: int = 1,
):
    """(key, index) pairs of a row sorted along it, as Triton's ``sort_with_index``:
    NaN after every number, and a stable sort's ties in index order.

    Traced in phases like ``group_scan_runs``: in phase 0 each element parks its pair
    in the row's ``p`` LDS slots (``p`` the row's power-of-two ceiling, ``row`` its
    place in the block, ``pos`` its position in the row of ``n``; every position below
    ``p`` is some element's), with whether it is past the row's end, which travels with
    it; the first element of phase 1 runs a bitonic network over every row with the
    whole block, then every element reads back the pair now at its position.
    """
    lds = _SortLds(smem, base, rows * p, kinds)
    at = idx(row) * p + idx(pos)
    if phase <= 0:
        state.pop((key, "sorted"), None)
        k, i = (_canon(v) for v in values)
        lds.put(at, (k, i, lt(pos, n)))
        return (k, i)
    if (key, "sorted") not in state:
        state[(key, "sorted")] = True
        barrier()
        _bitonic(lds, p, rows, threads, kinds[0], stable, descending, run)
    return lds.get(at)[:2]


class _SortLds:
    """A sort's slots as planes -- keys, indices, valid flags -- each word of a 64-bit
    value a plane of its own, so neighbouring slots are neighbouring banks."""

    def __init__(self, smem, base: int, total: int, kinds):
        self.smem, self.total = smem, total
        self.kinds = (*kinds, "b1")
        self.bases = []
        for k in self.kinds:
            self.bases.append(base)
            base += _words_of(k) * total

    def get(self, at):
        return tuple(
            tile_get(self.smem, b, self.total, at, k)
            for b, k in zip(self.bases, self.kinds)
        )

    def put(self, at, vals):
        for b, v in zip(self.bases, vals):
            tile_put(self.smem, b, self.total, at, v)


def _sort_after(a, b, ai, bi, kind_: str, stable: bool, descending: bool):
    """Whether ``a`` (at index ``ai``) belongs after ``b`` in the sorted row."""
    out = lt(a, b) if descending else gt(a, b)
    if _is_f(kind_):
        an, bn = isnan(a), isnan(b)
        nan_first = (bn, an) if descending else (an, bn)
        out = logical_or(out, logical_and(nan_first[0], logical_not(nan_first[1])))
        tie = logical_or(eq(a, b), logical_and(an, bn))
    else:
        tie = eq(a, b)
    if stable:
        out = logical_or(out, logical_and(tie, gt(ai, bi)))
    return out


def _compare_swap(x, y, i, k: int, p: int, kind_, stable, descending):
    """Elements ``x`` (at ``i``) and ``y`` of a bitonic step ordered."""
    (a, ai, va), (b, bi, vb) = x, y
    order = _sort_after(a, b, ai, bi, kind_, stable, descending)
    # Padding past the row's end sorts after every element.
    cond = logical_or(logical_and(vb, logical_not(va)), logical_and(eq(va, vb), order))
    if k < p:
        # Alternate blocks of k run the other way.
        cond = ne(cond, ne(i & k, 0))
    return (
        tuple(where(cond, v, u) for u, v in zip(x, y)),
        tuple(where(cond, u, v) for u, v in zip(x, y)),
    )


def _bitonic(lds, p, rows, threads, kind_, stable, descending, run: int):
    """The network over every row: steps of a distance under ``run`` on a thread's
    run of that many slots in registers, longer ones a pair a thread."""
    t = _tid()
    m = min(p, run)

    def over(work: int, body):
        for it in range(-(-work // threads)):
            g = t + it * threads
            if work % threads and (it + 1) * threads > work:
                _when(_val(lt(g, work)), lambda g=g: body(g))
            else:
                body(g)
        barrier()

    lanes = p // m
    if lanes <= WAVE and threads % lanes == 0:
        over(
            rows * lanes,
            lambda g: _bitonic_wave(lds, g, p, m, kind_, stable, descending),
        )
        return

    k = 2
    while k <= p:
        j = k // 2
        while j >= m:

            def pair(g, j=j, k=k):
                q, h = g // (p // 2), g % (p // 2)
                i = (h // j) * (2 * j) + h % j
                x, y = lds.get(idx(q) * p + idx(i)), lds.get(idx(q) * p + idx(i + j))
                x, y = _compare_swap(x, y, i, k, p, kind_, stable, descending)
                lds.put(idx(q) * p + idx(i), x)
                lds.put(idx(q) * p + idx(i + j), y)

            over(rows * p // 2, pair)
            j //= 2

        def in_registers(g, j=j, k=k):
            at = idx(g // lanes) * p + idx((g % lanes) * m)
            vals = [lds.get(at + e) for e in range(m)]
            _bitonic_run(vals, (g % lanes) * m, j, k, p, kind_, stable, descending)
            for e in range(m):
                lds.put(at + e, vals[e])

        over(rows * lanes, in_registers)
        k *= 2


def _bitonic_run(vals, e0, j: int, k: int, p: int, kind_, stable, descending):
    """Steps ``j``, ``j / 2``, ... 1 of stage ``k`` on a run of slots from ``e0``, held
    in registers (``vals``, updated in place)."""
    while j >= 1:
        for e in range(len(vals)):
            if not e & j:
                vals[e], vals[e + j] = _compare_swap(
                    vals[e], vals[e + j], e0 + e, k, p, kind_, stable, descending
                )
        j //= 2


def _bitonic_wave(lds, g, p: int, m: int, kind_, stable, descending):
    """The whole network of a row held by one wave, ``m`` slots a lane: steps across
    lanes swap through shuffles, so no step waits on the block."""
    lanes = p // m
    r = g % lanes
    e0 = r * m
    at = idx(g // lanes) * p + idx(e0)
    vals = [lds.get(at + e) for e in range(m)]
    k = 2
    while k <= p:
        j = k // 2
        while j >= m:
            d = j // m
            up = ne(r & d, 0)
            for e in range(m):
                mine = vals[e]
                other = tuple(_shfl(v, d, WAVE) for v in mine)
                lo = tuple(where(up, o, v) for v, o in zip(mine, other))
                hi = tuple(where(up, v, o) for v, o in zip(mine, other))
                x, y = _compare_swap(lo, hi, e0 + e, k, p, kind_, stable, descending)
                vals[e] = tuple(where(up, b, a) for a, b in zip(x, y))
            j //= 2
        _bitonic_run(vals, e0, j, k, p, kind_, stable, descending)
        k *= 2
    for e in range(m):
        lds.put(at + e, vals[e])


def barrier():
    gpu.barrier()


def tile_put(smem, base: int, plane: int, off, value):
    """Park ``value`` at word ``off`` of a staged tile starting at ``base``; a 64-bit
    value's second word goes one ``plane`` further on."""
    for i, word in enumerate(_to_words(_canon(value))):
        fx.memref_store(ArithValue(word), smem, idx32(off) + (base + i * plane))


def tile_get(smem, base: int, plane: int, off, k: str):
    words = [
        _val(ArithValue(fx.memref_load(smem, idx32(off) + (base + i * plane))))
        for i in range(2 if k in ("i64", "f64") else 1)
    ]
    return _from_words(words, k)


def smem_words(threads: int, components: int) -> int:
    """LDS words a group reduction needs: one slot per thread for each 32-bit word of
    each component (two words per component covers 64-bit kinds)."""
    return threads * 2 * components


def group_reduce(op: str, x, tpr: int, smem, threads: int, stride: int = 1):
    f = _combine_fn(op)
    x = _canon(x)
    (out,) = _group_reduce(
        (x,),
        lambda a, b: (_canon(f(a[0], b[0])),),
        tpr,
        smem,
        threads,
        stride,
        (default(op, kind(x)),),
    )
    return out


# -- arg reductions ----------------------------------------------------------------------


def _arg_better(op: str, v1, i1, v2, i2):
    """Whether (v2, i2) should replace (v1, i1): strictly better value, NaN counting as
    the extreme (torch picks the first NaN), lower index on ties."""
    k = kind(_canon(v1))
    if op == "max":
        better = _cmp(v2, v1, _arith.CmpIPredicate.sgt, _arith.CmpFPredicate.OGT)
    else:
        better = _cmp(
            v2,
            v1,
            _arith.CmpIPredicate.slt,
            _arith.CmpFPredicate.OLT,  # codespell:ignore
        )  # codespell:ignore
    tie = eq(v2, v1)
    if _is_f(k):
        n1, n2 = isnan(v1), isnan(v2)
        better = logical_or(better, logical_and(n2, logical_not(n1)))
        tie = logical_or(tie, logical_and(n1, n2))
        better = logical_and(better, logical_not(logical_and(n1, logical_not(n2))))
    earlier = lt(i2, i1)
    return logical_or(better, logical_and(tie, earlier))


def arg_combine(op: str, val, ind, x, xi, mask=None):
    val = _canon(val)
    k = kind(val)
    x = to(x, k)
    xi = to(xi, "i64")
    take = _arg_better(op, val, ind, x, xi)
    m = _mask(mask)
    if m is not None:
        take = ArithValue(_arith.AndIOp(_val(take), m).result)
    return where(take, x, val), where(take, xi, ind)


def arg_default(op: str, k: str):
    return default("max" if op == "max" else "min", k), const(2**63 - 1, "i64")


def group_arg_reduce(op: str, val, ind, tpr: int, smem, threads: int, stride: int = 1):
    def comb(a, b):
        v, i = arg_combine(op, a[0], a[1], b[0], b[1])
        return (_canon(v), _canon(i))

    return _group_reduce((val, to(ind, "i64")), comb, tpr, smem, threads, stride)


# -- Welford -----------------------------------------------------------------------------


def welford_default():
    return const(0.0, "f32"), const(0.0, "f32"), const(0.0, "f32")


def welford_merge(a, b, mask=None):
    """Chan's parallel combine of two (mean, m2, weight) triples."""
    (m1, s1, w1), (m2_, s2, w2) = a, b
    w = add(w1, w2)
    zero = const(0.0, "f32")
    frac = where(eq(w, zero), zero, truediv(w2, w))
    delta = sub(m2_, m1)
    mean = add(m1, mul(delta, frac))
    m2 = add(add(s1, s2), mul(mul(mul(delta, delta), w1), frac))
    out = (_canon(mean), _canon(m2), _canon(w))
    mm = _mask(mask)
    if mm is None:
        return out
    return tuple(
        ArithValue(_arith.SelectOp(mm, _val(n), _val(o)).result) for n, o in zip(out, a)
    )


def welford_block(acc, items, combine: bool):
    """``acc`` merged with a chunk of samples ``(x, mask)``, or of ``((mean, m2,
    weight), mask)`` triples for a combine: the chunk's weighted mean, then its
    squared deviations from it (Chan's formula over the chunk), then one merge -- a
    divide or two a chunk rather than an IEEE divide an element. A split BatchNorm's
    statistics were ALU-bound at 770 GB/s on it."""
    zero = const(0.0, "f32")

    def sel(m, v, o):
        if m is None:
            return v
        return ArithValue(_arith.SelectOp(m, _val(v), _val(o)).result)

    ms = [_mask(m) for _, m in items]
    if combine:
        trip = [
            (_f32(v[0]), sel(m, _f32(v[1]), zero), sel(m, _f32(v[2]), zero))
            for (v, _), m in zip(items, ms)
        ]
        w_c = _tree([w for _, _, w in trip], add)
        wx = _tree([ArithValue(mu * w) for mu, _, w in trip], add)
        mean_c = where(eq(w_c, zero), zero, truediv(wx, w_c))
        devs = []
        for mu, s, w in trip:
            d = ArithValue(mu - mean_c)
            devs.append(ArithValue(fmath.fma(ArithValue(d * w), d, s)))
    else:
        xs = [_f32(v) for v, _ in items]
        if all(m is None for m in ms):
            w_c = const(float(len(xs)), "f32")
            mean_c = mul(_tree(xs, add), const(1.0 / len(xs), "f32"))
        else:
            w_c = _tree([sel(m, const(1.0, "f32"), zero) for m in ms], add)
            sx = _tree([sel(m, x, zero) for x, m in zip(xs, ms)], add)
            mean_c = where(eq(w_c, zero), zero, truediv(sx, w_c))
        devs = []
        for x, m in zip(xs, ms):
            d = ArithValue(x - mean_c)
            devs.append(sel(m, ArithValue(d * d), zero))
    m2_c = _tree(devs, add)
    return welford_merge(acc, (_canon(mean_c), _canon(m2_c), _canon(w_c)))


def group_welford(mean, m2, weight, tpr: int, smem, threads: int, stride: int = 1):
    return _group_reduce((mean, m2, weight), welford_merge, tpr, smem, threads, stride)


# -- online softmax ----------------------------------------------------------------------


def _safe_exp_diff(a, m):
    """exp(a - m) that is 0 when both are -inf rather than NaN."""
    ninf = const(float("-inf"), "f32")
    d = sub(a, m)
    return where(eq(m, ninf), const(0.0, "f32"), exp(d))


def online_softmax_default():
    return const(float("-inf"), "f32"), const(0.0, "f32")


def online_softmax_merge(a, b):
    (m1, s1), (m2, s2) = a, b
    m = maximum(m1, m2)
    s = add(mul(s1, _safe_exp_diff(m1, m)), mul(s2, _safe_exp_diff(m2, m)))
    return _canon(m), _canon(s)


def group_online_softmax(mx, s, tpr: int, smem, threads: int, stride: int = 1):
    # The sum holds exp(max - max) = 1, as group_softmax_twopass's.
    m, s = _group_reduce((mx, s), online_softmax_merge, tpr, smem, threads, stride)
    return _derive(m, "rowmax"), _derive(s, "ge1")


# -- chunked forms -----------------------------------------------------------------------
#
# The per-element combines above are what Inductor's ops ask for, but a thread holds a
# whole chunk of elements per loop step (and a whole row in an unrolled reduction), all
# known at trace time. Combining a chunk at once is cheaper: one rescale exp per chunk
# instead of per element for softmax, and a balanced tree instead of a serial chain for
# sums. Each item is ``(value, mask)``, with ``mask`` None for an unmasked element.


def _tree(vals, f):
    vals = list(vals)
    while len(vals) > 1:
        nxt = [_canon(f(vals[i], vals[i + 1])) for i in range(0, len(vals) - 1, 2)]
        if len(vals) % 2:
            nxt.append(vals[-1])
        vals = nxt
    return vals[0]


def _masked_items(items, k, fill):
    out = []
    for x, m in items:
        v = to(x, k)
        m = _mask(m)
        out.append(
            v
            if m is None
            else ArithValue(_arith.SelectOp(m, _val(v), _val(const(fill, k))).result)
        )
    return out


def sum_block(acc, items):
    acc = _canon(acc)
    k = kind(acc)
    return _canon(add(acc, _tree(_masked_items(items, k, 0), add)))


def _softmax_base(m):
    """``m`` with -inf replaced by 0, so exp(x - base) is 0 rather than NaN for a row
    that is all -inf (or all masked) so far."""
    return where(eq(m, const(float("-inf"), "f32")), const(0.0, "f32"), m)


def online_softmax_block(mx, s, items):
    xs = _masked_items(items, "f32", float("-inf"))
    m_new = _canon(maximum(mx, _tree(xs, maximum)))
    base = _softmax_base(m_new)
    chunk = _tree([exp(sub(x, base)) for x in xs], add)
    return m_new, _canon(add(mul(s, exp(sub(mx, base))), chunk))


def group_softmax_twopass(items, tpr: int, smem, threads: int, stride: int = 1):
    """(max, sum of exp(x - max)) over a row held entirely in registers: a max pass and an
    exp-sum pass, as Triton does for a persistent reduction. ``exp(x - max)`` exactly as
    the kernel's normalizing pass writes it -- no -inf guard; an all -inf row is NaN
    either way, as in eager and Triton -- so LLVM shares each exp between the two
    passes: 268us vs 317us on a causal softmax over 512.

    The sum holds exp(max - max) = 1, so the normalizing divide takes the row's v_rcp
    (or is NaN either way). The max needn't carry a NaN, a cmp and a select an element:
    exp(NaN - max) puts it in the sum, and so in every output, as eager's."""
    xs = _masked_items(items, "f32", float("-inf"))

    def maxnum(a, b):
        return ArithValue(_arith.MaxNumFOp(_val(a), _val(b)).result)

    gm = _derive(
        group_reduce("max", _tree(xs, maxnum), tpr, smem, threads, stride), "rowmax"
    )
    ls = _tree([exp(sub(x, gm)) for x in xs], add)
    return gm, _derive(group_reduce("sum", ls, tpr, smem, threads, stride), "ge1")


def group_welford_twopass(
    items, tpr: int, smem, threads: int, stride: int = 1, count=None
):
    """(mean, m2, weight) over a row held entirely in registers: a (sum, count) pass, then
    a pass summing squared deviations from the mean -- no per-element divides, as
    Triton's persistent reductions compute a variance. ``count``, the row length when
    every element counts, drops the count reduction and the divide: the kernel is
    vector-ALU-bound at LayerNorm widths."""
    xs = _masked_items(items, "f32", 0.0)
    zero = const(0.0, "f32")
    if count is not None:
        s = group_reduce("sum", _tree(xs, add), tpr, smem, threads, stride)
        n = const(float(count), "f32")
        mean = _canon(mul(s, const(1.0 / count, "f32")))
    else:
        ones = _masked_items([(const(1.0, "f32"), m) for _, m in items], "f32", 0.0)

        def add2(a, b):
            return _canon(add(a[0], b[0])), _canon(add(a[1], b[1]))

        s, n = _group_reduce(
            (_tree(xs, add), _tree(ones, add)),
            add2,
            tpr,
            smem,
            threads,
            stride,
            (zero, zero),
        )
        mean = _canon(where(eq(n, zero), zero, truediv(s, n)))
    devs = []
    for (x, m), v in zip(items, xs):
        d = sub(v, mean)
        devs.append((mul(d, d), m))
    m2 = group_reduce(
        "sum", _tree(_masked_items(devs, "f32", 0.0), add), tpr, smem, threads, stride
    )
    return mean, m2, n


# ---------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------


def pick_fastest(mains, args, mutated, stream=None, warm=None):
    """An autotuned kernel's first call: time every candidate's ``_main`` on copies of
    the buffers it updates in place, run the fastest on the real arguments, and return
    it. Under graph capture nothing can be timed: the first (heuristic) one runs, and
    None leaves the choice to a later call.
    """
    import torch

    if torch.cuda.is_current_stream_capturing():
        mains[0](*args, stream=stream)
        return None
    path = _pick_path(mains)
    try:
        with open(path) as f:
            best = next(m for m in mains if m.__name__ == f.read().strip())
        best(*args, stream=stream)
        return best
    except (OSError, TypeError, StopIteration):
        pass
    scratch = list(args)
    for i in mutated:
        scratch[i] = args[i].clone()
    for fn in mains:
        fn(*scratch, stream=stream)  # compiles (and runs once)
    # Back-to-back calls averaged, the candidates interleaved round by round and the
    # best round kept: the fastest single call (the benchmarker's "min") favours a
    # candidate with a high variance, a 16x64 transpose tile 10% slower on average.
    # Each round is queued behind a GPU sleep: a launch from Python takes longer than
    # a 20us kernel, and unqueued the GPU would idle between them, timing the launches.
    # Every call starts from the cache state the kernel meets in the model, as far as
    # the scheduler can tell: the 256 MB last-level cache flushed, then the inputs that
    # the kernels just before it wrote or read (`warm`) rewritten in place, leaving
    # them there dirty, as their producer would. Timed back to back instead, every
    # input is cached: BERT's softmax-dropout backward, whose matmul-written gradient
    # alone is, picked a variant 115us in the model against 100us. Flushed with nothing
    # rewritten, ResNet's BatchNorm backward, whose saved activations the convolution
    # backward had just read, picked variants 0.1 ms a step slower; rewarmed by reading
    # (clean lines), its in-place conv bias add 95us against 84us. Without `warm` (an
    # older module), only L2 is flushed.
    device = next(a.device for a in args if isinstance(a, torch.Tensor))
    flush = torch.empty(
        (256 if warm is not None else 16) << 20, dtype=torch.int8, device=device
    )
    warm_bufs = [
        torch.empty(0, dtype=torch.uint8, device=device).set_(
            scratch[i].untyped_storage()
        )
        for i in (warm or ())
        if isinstance(scratch[i], torch.Tensor)
    ]

    def before():
        flush.zero_()
        for t in warm_bufs:
            t.add_(0)

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    mains[0](*scratch, stream=stream)
    ev[1].record()
    ev[1].synchronize()
    reps = max(1, min(10, int(0.3 / max(ev[0].elapsed_time(ev[1]), 1e-3))))

    def measure(fns, before):
        times = [float("inf")] * len(fns)
        # ~60us a launch (four with the flush and events); over-sleeping is cheap.
        sleep_us = len(fns) * reps * 4 * 60
        rounds = tries = 0
        while rounds < 3:
            tries += 1
            start = _time.perf_counter()
            torch.cuda._sleep(int(sleep_us * 2.1e3))  # cycles: gfx942 at 2.1 GHz
            events = []
            for fn in fns:
                pairs = []
                for _ in range(reps):
                    if before is not None:
                        before()
                    pair = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    pair[0].record()
                    fn(*scratch, stream=stream)
                    pair[1].record()
                    pairs.append(pair)
                events.append(pairs)
            queued_us = (_time.perf_counter() - start) * 1e6
            events[-1][-1][1].synchronize()
            if queued_us > sleep_us and tries < 8:
                # The GPU outran the launches (a host busy compiling other kernels),
                # so this round timed them too: a [32, 512, 768] heads permute picked
                # a variant 2.5x slower next to a pytest run.
                sleep_us *= 2
                continue
            rounds += 1
            for k, pairs in enumerate(events):
                times[k] = min(times[k], sum(s.elapsed_time(e) for s, e in pairs))
        return times

    times = measure(mains, before)
    flush = None  # 256 MB, not needed for the warm timing below
    # Candidates within 3% of each other cold are told apart warm, where their cache
    # behaviour differs: a bf16 embedding's two launches timed within 3% cold, then
    # 10% apart in the steady state, and which one won the cold timing varied by run.
    close = [m for m, t in zip(mains, times) if t <= min(times) * 1.03]
    if len(close) > 1:
        warm_times = measure(close, None)
        best = close[warm_times.index(min(warm_times))]
    else:
        best = mains[times.index(min(times))]
    if path is not None:
        tmp = f"{path}.{_os.getpid()}"
        try:
            _os.makedirs(_os.path.dirname(path), exist_ok=True)
            with open(tmp, "w") as f:
                f.write(best.__name__)
            _os.replace(tmp, path)
        except OSError:
            pass
    best(*args, stream=stream)
    return best


def _pick_path(mains):
    """Where ``pick_fastest`` keeps its choice among ``mains`` for later processes, as
    Triton keeps its autotuning results: a warm start otherwise re-timed every
    autotuned kernel (0.17 s of a GPT-2 prefill's). None with caches disabled. The
    names carry a hash of the kernel source, which fixes the shapes."""
    import hashlib

    import torch
    from torch._inductor import config
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    if config.force_disable_caches:
        return None
    key = "\n".join(
        [torch.cuda.get_device_properties().gcnArchName, *(m.__name__ for m in mains)]
    )
    h = hashlib.sha256(key.encode()).hexdigest()[:32]
    return _os.path.join(cache_dir(), "flydsl_picks", h)


# The compiled artifact's GPU binary: `#gpu.object<..., bin = "...">`, an MLIR string
# with non-printable bytes, `"` and `\` escaped as `\XX` / `\\`.
_BIN_RE = _re.compile(r'bin\s*=\s*"((?:[^"\\]|\\.)*)"')
_ESC_RE = _re.compile(rb"\\(\\|[0-9A-Fa-f]{2})")


def save_binary(launch, path: str, kernel: str) -> None:
    """After ``launch`` has compiled (in the compile worker), write its kernel's code
    object to ``path``, for ``load_binary``. Skipped, leaving the FlyDSL path, unless
    the artifact holds exactly one ELF binary whose kernel is ``kernel``."""
    last = launch._last_compiled
    art = last[1] if last else next(iter(launch._mem_cache.values()), None)
    ir_text = getattr(art, "_ir_text", "")
    found = _BIN_RE.findall(ir_text)
    if len(found) != 1 or f'"{kernel}"' not in ir_text:
        return
    image = _ESC_RE.sub(
        lambda m: b"\\" if m.group(1) == b"\\" else bytes([int(m.group(1), 16)]),
        found[0].encode("latin-1"),
    )
    if not image.startswith(b"\x7fELF"):
        return
    tmp = f"{path}.{_os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(image)
    _os.replace(tmp, path)


def _set_arch(arch: str | None) -> None:
    # Otherwise FlyDSL runs rocm_agent_enumerator in every compile worker: 0.2 s
    # alone, over a second with the pool's workers all querying the driver at once.
    if arch:
        _os.environ.setdefault("FLYDSL_GPU_ARCH", arch)


def compile_only(launch, *args, arch: str | None = None) -> None:
    """Compile ``launch`` for ``args`` into its caches without building the host
    launcher (~18 ms) or launching."""
    import io

    from flydsl.utils import env

    _set_arch(arch)
    var = type(env.compile).__dict__["compile_only"].env_var  # "COMPILE_ONLY"
    old = _os.environ.get(var)
    _os.environ[var] = "1"
    try:
        # FlyDSL prints a line for every compile-only call.
        with contextlib.redirect_stdout(io.StringIO()), _no_source_locations():
            launch(*args)
    finally:
        if old is None:
            del _os.environ[var]
        else:
            _os.environ[var] = old


@contextlib.contextmanager
def _no_source_locations():
    """FlyDSL tags every op with its Python call stack (``inspect`` per op), a third
    of tracing a generated kernel and ~15% of its compile; these kernels' line
    numbers point into generated code nobody steps through."""
    from flydsl.compiler import ast_rewriter
    from flydsl.expr import meta

    def unknown():
        return ir.Location.unknown()

    saved = meta.capture_user_location, ast_rewriter.capture_user_location
    meta.capture_user_location = ast_rewriter.capture_user_location = unknown
    try:
        yield
    finally:
        meta.capture_user_location, ast_rewriter.capture_user_location = saved


def warm_worker(arch: str | None = None) -> None:
    """A compile worker's first FlyDSL kernel paid ~1.4 s of imports and ~0.5 s of
    one-time compiler setup; run ahead of the kernels, while Inductor is still
    generating them."""
    import flydsl.compiler as flyc

    _set_arch(arch)

    @flyc.kernel
    def _warm_kernel(p: fx.Int64):
        store(rsrc(p, 4), tid(), const(0, "i32"), "i32", mask=tid() < 1)

    @flyc.jit
    def _warm_launch(p: fx.Int64, stream: fx.Stream = fx.Stream(None)):
        _warm_kernel(p).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    with contextlib.suppress(Exception):  # only a head start; the kernels report errors
        compile_only(_warm_launch, 0)


def preload_worker(arch: str | None = None) -> None:
    """Run in the compile pool's sidecar before it forks its workers, so each inherits
    FlyDSL's imports and the gfx target rather than paying for them itself."""
    _set_arch(arch)
    if "FLYDSL_GPU_ARCH" not in _os.environ:
        # Queries the driver, 0.3 s of the pool's startup, so only without ``arch``.
        from flydsl.runtime.device import get_rocm_arch

        _os.environ["FLYDSL_GPU_ARCH"] = get_rocm_arch()
    warm_worker()


_EARLY: dict = {}


def compile_early(kernel_name: str, source: str, binaries: list[str]) -> None:
    """At codegen, hand each of a module's variants to an idle compile worker, rather
    than when the finished graph module loads: a graph's first call otherwise waited
    on its slowest compile after all of codegen (Llama training's AdamW kernel, 1.4 s).
    Whenever there may be a pool, ready or not: ``use_process_pool`` only says whether
    its workers are up yet, which on a short graph they are not until after codegen,
    leaving all of its compiles to start at load. A job queued now starts as soon as
    they are up.
    ``compiled`` then waits for the job and loads its code object here."""
    from torch._inductor import async_compile
    from torch._inductor.codegen.flydsl.flydsl_kernel import MAIN_SUFFIX
    from torch._inductor.runtime.compile_tasks import _worker_compile_pycodecache_kernel

    pool_cls = async_compile.AsyncCompile
    # Starts the workers if this is the first ask; None when there is to be no pool.
    pool_cls.use_process_pool()
    if pool_cls._ready_future is None:
        return
    env = async_compile._pycodecache_kernel_compile_env()
    if "FLYDSL_RUNTIME_CACHE_DIR" in _os.environ:
        env["FLYDSL_RUNTIME_CACHE_DIR"] = _os.environ["FLYDSL_RUNTIME_CACHE_DIR"]
    pool = async_compile.AsyncCompile.process_pool()
    for i, binary in enumerate(binaries):
        if binary in _EARLY or _os.path.exists(binary):
            continue
        meta = {"only": i} if len(binaries) > 1 else {}
        _EARLY[binary] = pool.submit(
            _worker_compile_pycodecache_kernel,
            kernel_name,
            source,
            MAIN_SUFFIX,
            env,
            meta,
        )


def compiled(async_compile, kernel_name: str, source: str, binary: str, **kw):
    """``async_compile.flydsl``, except that a kernel whose code object ``binary`` is
    already on disk (a warm start) loads in this process: through the pool it waited
    on a worker to find nothing to do, 0.35 s of GPT training's warm start. So does
    one that ``compile_early`` compiled, once its job is done; if it failed, the
    regular compile reports the error."""
    early = _EARLY.pop(binary, None)
    if early is not None:
        with contextlib.suppress(Exception):
            early.result()
    if not _os.path.exists(binary):
        return async_compile.flydsl(kernel_name, source, **kw)
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.codegen.flydsl.flydsl_kernel import (
        FlyDSLKernelWrapper,
        MAIN_SUFFIX,
    )

    key, path = PyCodeCache.write(source)
    return async_compile._load_kernel_wrapper(
        kernel_name, MAIN_SUFFIX, FlyDSLKernelWrapper, key, path
    )


def have_binary(path: str) -> bool:
    """Whether ``save_binary`` already wrote ``path`` (its name holds the module's source
    hash, so it is this source's code object): a warm start's compile is then nothing."""
    return _os.path.exists(path)


def load_binary(path: str, kernel: str, threads: int, tys: str):
    """A launcher for the code object ``save_binary`` wrote, or None if there is none,
    called as ``(grid, 1, 1, stream, *args)``: an argument of type ``O`` a tensor, ``l``
    an int. FlyDSL's own first call in a process re-JITs the host launch code from the
    cached module, about 18 ms a kernel: 10 s of a Qwen3 prefill's first step. The
    vectorcall launcher Triton's kernels use reads the tensors' addresses in C++: 3 us
    a launch under the Python path's."""
    if not _os.path.exists(path):
        return None
    import torch

    device = torch.cuda.current_device()
    # On ROCm the launcher's block is num_warps times the device's warp size.
    warp = torch.cuda.get_device_properties(device).warp_size
    if threads % warp:
        return None
    launcher = torch._C._StaticCudaLauncher
    module, function, _, _ = launcher._load_kernel(path, kernel, 0, device)
    # The launchers hold the raw function: its code object stays loaded.
    _LOADED.append(module)
    warps = threads // warp
    fast = getattr(torch._C, "_FastCudaLauncher", None)
    if fast is not None:
        # It takes at most MAX_ARGS arguments: a wide combo kernel has more.
        with contextlib.suppress(ValueError):
            return fast(function, warps, 0, tys, 0)
    launch_kernel = launcher._launch_kernel
    addr_tys = tys.replace("O", "l")

    def slow(gx, gy, gz, stream, *args):
        args = tuple(a if t == "l" else a.data_ptr() for a, t in zip(args, tys))
        launch_kernel(function, gx, gy, gz, warps, 0, addr_tys, args, stream)

    return slow


_LOADED: list = []


del _math

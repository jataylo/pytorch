# mypy: ignore-errors
"""Device-side helpers for FlyDSL kernels generated from Inductor pointwise/reduction IR.

Generated SIMD kernels import this module as ``fdp``. It is the SIMD counterpart of
``flydsl_mod_runtime`` (the flex score_mod shim, imported as ``fdu``), and leans on it for
the transcendental expansions that have no AMDGPU libcall. Where the flex shim only ever
sees an f32 score and i32 coordinates, a general Inductor kernel carries every dtype, so
values here live in one of five *compute kinds*:

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

import math as _math

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

from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels import buffer_ops
from torch.utils._ordered_set import OrderedSet

from . import flydsl_mod_runtime as fdu


WAVE = 64
_LOG2E = 1.4426950408889634


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
    "b1": T.i8,  # torch.bool is a byte in memory
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
    "b1": "b1",
}

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
    src_int = src in ("b1", "i8", "u8", "i16", "i32", "i64")
    dst_int = dst in ("b1", "i8", "u8", "i16", "i32", "i64")
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
            if src == "b1" or src == "u8" or unsigned:
                return _arith.ExtUIOp(dst_t, v).result
            return _arith.ExtSIOp(dst_t, v).result
        return _arith.TruncIOp(dst_t, v).result
    if src_int and not dst_int:
        if src == "b1" or src == "u8" or unsigned:
            return _arith.UIToFPOp(dst_t, v).result
        return _arith.SIToFPOp(dst_t, v).result
    if not src_int and dst_int:
        if dst == "u8":
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
        return const(x, k)
    v = _val(x)
    return ArithValue(_cvt(v, kind_of_type(v.type), k))


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
            return const(self.c, "i32")
        return self.base + self.c if self.c else self.base

    @staticmethod
    def _of(o):
        """``o`` as an Aff, or None when it is not an i32 index."""
        if isinstance(o, Aff):
            return o
        if isinstance(o, bool) or not isinstance(o, int):
            v = _val(o) if not isinstance(o, float) else None
            if v is None or kind_of_type(v.type) != "i32":
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
            base = a.base if sign > 0 else ArithValue(fx.Int32(0)) - a.base
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
        return Aff(r) if kind_of_type(_val(r).type) == "i32" else r

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


def tid():
    return Aff(ArithValue(_val(_tid())))


def bid():
    return Aff(ArithValue(_val(fx.Int32(gpu.block_idx.x))))


def _tid():
    return fx.Int32(gpu.thread_idx.x)


def rsrc(addr, nbytes: int):
    """Buffer resource over ``nbytes`` from a raw device address.

    ``nbytes`` must be exact: it is what makes a masked access (offset 0x7FFFFFFF) fall
    out of range. The codegen refuses buffers of 2 GiB or more for the same reason.
    """
    return buffer_ops.create_buffer_resource_from_addr(
        _val(addr), num_records_bytes=int(nbytes)
    )


def idx(x):
    """An i32 element offset from an index value of any integer kind."""
    if isinstance(x, int):
        return fx.Int32(x)
    v = _val(x)
    k = kind_of_type(v.type)
    if k == "i32":
        return ArithValue(v)
    if k == "index":
        return ArithValue(_arith.IndexCastOp(T.i32(), v).result)
    if _is_f(k):
        return ArithValue(_arith.FPToSIOp(T.i32(), v).result)
    return ArithValue(_cvt(v, k, "i32"))


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
    "b1": 1,
}


def load(r, off, st: str, mask=None):
    """Scalar masked load, widened to the compute kind. A masked lane reads 0."""
    raw = buffer_ops.buffer_load(
        r, idx(off), vec_width=1, dtype=_STORAGE_IR[st](), mask=_mask(mask)
    )
    return ArithValue(_cvt(raw, _MEM_KIND.get(st, st), _COMPUTE_OF[st]))


def sload(r, off: int, st: str):
    """A load at a constant offset, through the scalar cache into an SGPR, as Triton's
    ``s_load`` of a broadcast operand: a vector load would hold every wave's stores on
    its round trip. The scalar cache does not see this kernel's own stores, so only
    for buffers the kernel only reads."""
    size = _ITEMSIZE[st]
    byte = off * size
    word = buffer_ops.buffer_load(r, fx.Int32(byte // 4), vec_width=1, is_scalar=True)
    if byte % 4:
        shift = _arith.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), byte % 4 * 8))
        word = _arith.ShRUIOp(word, shift.result).result
    mem = _STORAGE_IR[st]()
    if size < 4:
        word = _arith.TruncIOp(ir.IntegerType.get_signless(8 * size), word).result
    raw = word if mem == word.type else _arith.BitcastOp(mem, word).result
    return ArithValue(_cvt(raw, _MEM_KIND.get(st, st), _COMPUTE_OF[st]))


def wload(cache: dict, key, chunk: int, r, off, st: str):
    """``sload`` at an offset computed at run time but the same across the wave: lane
    0's offset, read into an SGPR. Read-only buffers only, as for ``sload``."""
    if (key, "w", chunk) in cache:
        return cache[(key, "w", chunk)]
    size = _ITEMSIZE[st]
    byte = _val(idx(off) * size)
    byte = _rocdl.readfirstlane(T.i32(), byte)
    two = _arith.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), 2)).result
    word = buffer_ops.buffer_load(
        r, _arith.ShRUIOp(byte, two).result, vec_width=1, is_scalar=True
    )
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
    nbytes = {"f16": 2, "bf16": 2, "i16": 2, "i8": 1, "u8": 1, "b1": 1}.get(st)
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
    extract from it; their own ``off`` is computed and left for DCE.
    """
    if (key, j // w) not in cache:
        # Usually issued by the chunk's first element, but a store to the same buffer
        # (see forget) can drop the vector mid-chunk; the load always starts at the base.
        base = idx(off if j % w == 0 else off - j % w)
        word_t, n, ratio = _packing(st, w)
        if word_t is None:
            raw = buffer_ops.buffer_load(
                r, base, vec_width=w, dtype=_STORAGE_IR[st](), mask=_mask(mask)
            )
        else:
            words = buffer_ops.buffer_load(
                r, idiv(base, ratio), vec_width=n, dtype=word_t, mask=_mask(mask)
            )
            raw = _vector.BitCastOp(
                ir.VectorType.get([w], _STORAGE_IR[st]()), _as_vector(words, n, word_t)
            ).result
        cache[(key, j // w)] = raw
    vec = cache[(key, j // w)]
    elem = _vector.ExtractOp(vec, static_position=[j % w], dynamic_position=[]).result
    return ArithValue(_cvt(elem, _MEM_KIND.get(st, st), _COMPUTE_OF[st]))


def uload(cache: dict, key, chunk: int, r, off, st: str, mask=None):
    """A load whose address is the same across one contiguous run of elements a thread
    owns (``chunk`` numbers the runs when the thread owns several): issue it once."""
    if (key, "u", chunk) not in cache:
        cache[(key, "u", chunk)] = load(r, off, st, mask)
    return cache[(key, "u", chunk)]


def memo(cache: dict, key, thunk):
    """Trace-time load reuse in an unrolled region: the first trace of ``key`` loads."""
    if key not in cache:
        cache[key] = thunk()
    return cache[key]


_FRESH = "__fresh__"


def forget(cache: dict, arg: str) -> None:
    """Drop every cached load of ``arg`` after a store to it, but the stored values
    themselves (vstore's ``seed``) entered since the last forget."""

    def owner(k):
        while isinstance(k, tuple) and k:
            if k[0] == arg:
                return True
            k = k[0]
        return False

    fresh = cache.get(_FRESH, OrderedSet())
    for k in [k for k in cache if owner(k)]:
        if k in fresh:
            fresh.discard(k)
        else:
            del cache[k]


def store(r, off, value, st: str, mask=None):
    v = _val(_canon(value))
    raw = _cvt(v, kind_of_type(v.type), _MEM_KIND.get(st, st))
    buffer_ops.buffer_store(raw, r, idx(off), mask=_mask(mask))


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
    raw = _cvt(v, kind_of_type(v.type), _MEM_KIND.get(st, st))
    if j % w == 0:
        # A fresh slot: the same site is traced again by every unrolled loop iteration.
        cache[(key, j // w)] = {"elems": [], "off": idx(off), "mask": _mask(mask)}
    slot = cache[(key, j // w)]
    slot["elems"].append(raw)
    if j % w == w - 1:
        del cache[(key, j // w)]
        vt = ir.VectorType.get([w], _STORAGE_IR[st]())
        vec = _vector.FromElementsOp(vt, slot["elems"]).result
        word_t, n, ratio = _packing(st, w)
        if word_t is None:
            buffer_ops.buffer_store(vec, r, slot["off"], mask=slot["mask"])
        else:
            words = _vector.BitCastOp(ir.VectorType.get([n], word_t), vec).result
            if n == 1:
                words = _vector.ExtractOp(
                    words, static_position=[0], dynamic_position=[]
                ).result
            buffer_ops.buffer_store(
                words, r, idiv(slot["off"], ratio), mask=slot["mask"]
            )
        if seed is not None:
            # A later unrolled loop reading the stored elements back takes the register,
            # in exactly vload's form, instead of a round trip through memory.
            lc, lkey = seed
            lc[(lkey, j // w)] = vec
            lc.setdefault(_FRESH, OrderedSet()).add((lkey, j // w))


def _packed_half(o, value, st: str):
    """(two-lane vector, byte offset of its 32-bit word) for a 16-bit atomic add at
    element ``o``: the value in its lane and -0.0 in the neighbour's."""
    h = _cvt(_val(to(value, "f32")), "f32", st)
    et = _STORAGE_IR[st]()
    negz = _arith.ConstantOp(et, ir.FloatAttr.get(et, -0.0)).result
    odd = _val(eq(ArithValue(_arith.AndIOp(_val(o), _val(fx.Int32(1))).result), 1))
    lo = _arith.SelectOp(odd, negz, h).result
    hi = _arith.SelectOp(odd, h, negz).result
    v = _vector.FromElementsOp(ir.VectorType.get([2], et), [lo, hi]).result
    word = _arith.AndIOp(_val(o), _val(fx.Int32(-2))).result
    return v, _val(mul(ArithValue(word), 2))


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
    an LLVM ``atomicrmw`` on the raw address instead of the buffer descriptor."""
    v, byte_off = _packed_half(idx(off), value, st)
    _packed_add_global(addr, byte_off, v, mask)


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
    byte_off = _val(mul(o, 2))
    if st == "bf16":
        _packed_add_global(target, byte_off, v, mask)
    else:
        _buffer_add(target, byte_off, v, mask)


def _packed_add_global(addr, byte_off, v, mask):
    from flydsl._mlir.dialects import llvm as _llvm

    a = _arith.AddIOp(
        _val(to(addr, "i64")), _arith.ExtUIOp(T.i64(), byte_off).result
    ).result

    def emit():
        ptr = _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), a).result
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.fadd,
            ptr,
            v,
            _llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=4,
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
        byte_off = _val(mul(o, 4 if st == "f32" else 8))
    elif st == "f16":
        v, byte_off = _packed_half(o, value, st)
    else:
        raise NotImplementedError(f"atomic_add of {st}")
    _buffer_add(r, byte_off, v, mask)


def _buffer_add(r, byte_off, v, mask):
    m = _mask(mask)
    if m is None:
        zero = _arith.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), 0)).result
        _rocdl.RawPtrBufferAtomicFaddOp(v, r, byte_off, zero, zero)
        return
    # Unlike loads and stores, a buffer atomic steered out of range (offset 0x7FFFFFFF)
    # faults on gfx942 rather than being dropped, so a masked lane needs a real branch.
    # Built by hand: helpers are not AST-rewritten, so a Python `if` would not trace.
    if_op = _scf.IfOp(m)
    with ir.InsertionPoint(if_op.then_block):
        zero = _arith.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), 0)).result
        _rocdl.RawPtrBufferAtomicFaddOp(v, r, byte_off, zero, zero)
        _scf.YieldOp([])


# ---------------------------------------------------------------------------------------
# Pointwise ops
# ---------------------------------------------------------------------------------------


def add(a, b):
    _, (x, y) = _promote(a, b)
    return ArithValue(x + y)


def sub(a, b):
    _, (x, y) = _promote(a, b)
    return ArithValue(x - y)


def mul(a, b):
    k, (x, y) = _promote(a, b)
    if k == "b1":
        return ArithValue(_arith.AndIOp(_val(x), _val(y)).result)
    return ArithValue(x * y)


def truediv(a, b):
    k = _float_kind(a, b)
    x, y = to(a, k), to(b, k)
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


def _cmp(a, b, ipred, fpred):
    k, (x, y) = _promote(a, b)
    if _is_f(k):
        return ArithValue(_arith.CmpFOp(fpred, _val(x), _val(y)).result)
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


def copysign(a, b):
    k = _float_kind(a, b)
    return (
        ArithValue(_arith.CopySignOp(_val(to(a, k)), _val(to(b, k))).result)
        if hasattr(_arith, "CopySignOp")
        else ArithValue(fmath.copysign(to(a, k), to(b, k)))
    )


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
    return ArithValue(fmath.rsqrt(_require_f32("rsqrt", x)))


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
    acc = fdu.const_f32(coeffs[-1])
    for c in reversed(coeffs[:-1]):
        acc = ArithValue(fmath.fma(acc, z, fdu.const_f32(c)))
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
    # Below 1, erf(x) / x as a polynomial in x^2 (relative error ~1e-7). Above, fdu.erf's
    # A&S 7.1.26 (abs error 1.5e-7) with 1 / (1 + p|x|), which lies in (0, 1], taken by
    # v_rcp_f32. Pointwise kernels around erf are ALU-bound on MI300 (~35 instructions
    # an element at HBM speed), so every instruction here shows: an erf GELU went
    # 121us -> 96us without the fdiv. The two sides are a real branch, as in OCML's
    # erf: a wave whose lanes are all below 1 skips the exp path (126us vs 143us on
    # the GELU of a freshly initialized ViT MLP).
    x = _f32(x)
    ax = ArithValue(fmath.absf(x))
    if_op = _scf.IfOp(_val(ax < fdu.const_f32(1.0)), [T.f32()], has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        small = ArithValue(ax * _horner(ArithValue(ax * ax), _ERF_UNIT))
        _scf.YieldOp([_val(small)])
    with ir.InsertionPoint(if_op.else_block):
        t = _rcp(fmath.fma(fdu.const_f32(fdu._ERF_P), ax, fdu.const_f32(1.0)))
        poly = ArithValue(t * _horner(t, tuple(reversed(fdu._ERF_COEFFS))))
        e = _exp2_raw(ArithValue(ArithValue(ax * ax) * fdu.const_f32(-_LOG2E)))
        big = ArithValue(fmath.fma(ArithValue(-poly), e, fdu.const_f32(1.0)))
        _scf.YieldOp([_val(big)])
    return _copysign(ArithValue(if_op.results[0]), x)


def _tanh_rcp(x):
    # 1 - 2 / (exp(2|x|) + 1), the denominator >= 2 (rcp(inf) = 0); the series below.
    x = _f32(x)
    ax = ArithValue(fmath.absf(x))
    e = _exp2_raw(ArithValue(ax * fdu.const_f32(2.0 * _LOG2E)))
    big = ArithValue(
        fmath.fma(fdu.const_f32(-2.0), _rcp(e + fdu.const_f32(1.0)), fdu.const_f32(1.0))
    )
    small = ArithValue(ax * _horner(ArithValue(ax * ax), _TANH_SMALL))
    r = ArithValue(ArithValue(ax < fdu.const_f32(_SMALL)).select(small, big))
    return _copysign(r, x)


def _sigmoid_rcp(x):
    # 1 / (1 + exp(-x)): the denominator is >= 1, and rcp(inf) = 0 for very negative x.
    e = _exp2_raw(ArithValue(_f32(x) * fdu.const_f32(-_LOG2E)))
    return _rcp(e + fdu.const_f32(1.0))


exp = _unary_f32("exp", fdu.exp)
exp2 = _unary_f32("exp2", fdu.exp2)
expm1 = _unary_f32("expm1", fdu.expm1)
log = _unary_f32("log", fdu.log)
log2 = _unary_f32("log2", fdu.log2)
log10 = _unary_f32("log10", fdu.log10)
log1p = _unary_f32("log1p", fdu.log1p)
sin = _unary_f32("sin", fdu.sin)
cos = _unary_f32("cos", fdu.cos)
tanh = _unary_f32("tanh", _tanh_rcp)
sigmoid = _unary_f32("sigmoid", _sigmoid_rcp)
sinh = _unary_f32("sinh", fdu.sinh)
cosh = _unary_f32("cosh", fdu.cosh)
asin = _unary_f32("asin", fdu.asin)
acos = _unary_f32("acos", fdu.acos)
atan = _unary_f32("atan", fdu.atan)
asinh = _unary_f32("asinh", fdu.asinh)
acosh = _unary_f32("acosh", fdu.acosh)
atanh = _unary_f32("atanh", fdu.atanh)
erf = _unary_f32("erf", _erf_rcp)
erfc = _unary_f32("erfc", fdu.erfc)


def tan(x):
    return truediv(sin(x), cos(x))


def atan2(a, b):
    return ArithValue(fdu.atan2(_require_f32("atan2", a), _require_f32("atan2", b)))


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


def _philox(seed, offset):
    """Four uint32 words (held as i32) for counter (offset, 0, 0, 0) under key ``seed``."""
    s = _val(to(seed, "i64"))
    k0 = _arith.TruncIOp(T.i32(), s).result
    k1 = _arith.TruncIOp(
        T.i32(),
        _arith.ShRUIOp(s, _val(const(32, "i64"))).result,
    ).result
    c0, c1, c2, c3 = _val(idx(offset)), None, None, None
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
            word, _val(idx(off)), _val(fx.Int32(width)), mode=mode
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
    if stride == 1:
        vals = _dpp_reduce(vals, comb, min(tpr, WAVE), ident)
    else:
        off = stride
        while off < WAVE and off < stride * tpr:
            peer = tuple(_shfl(v, off, WAVE) for v in vals)
            vals = comb(vals, peer)
            off *= 2
    if stride * tpr <= WAVE:
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
    kstep = max(1, WAVE // stride)
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


def group_scan(fn, values, carry, first, tpr: int, lane):
    """Inclusive scan of one value per thread over each group of ``tpr`` consecutive
    lanes (at most a wave), continuing from ``carry``, the previous runs' total.

    ``fn(earlier, later)`` is the combine on tuples. Returns the scanned values followed
    by the carry for the next run, as one flat tuple. ``first`` (a trace-time or device
    bool) discards ``carry``.
    """
    out, nxt = _group_scan(fn, values, carry, first, tpr, lane)
    return (*out, *nxt)


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
):
    """``group_scan`` where each thread owns runs of ``ept`` contiguous elements.

    The body is traced twice per step. Phase 0 only records element ``j``'s value (its
    results are unused, so the trace is dead code). In phase 1 the run's first element
    scans the thread's values in registers, scans the per-thread totals across the
    lanes, and then offsets every element by the lanes before it and the carry, so a
    run costs one cross-lane scan rather than ``ept``.
    """
    run, e = j // ept, j % ept
    if phase == 0:
        state[(key, run, e)] = tuple(_canon(v) for v in values)
        return (*state[(key, run, e)], *carry)
    if e == 0:
        pre = [state[(key, run, 0)]]
        for k in range(1, ept):
            pre.append(tuple(_canon(v) for v in fn(pre[-1], state[(key, run, k)])))
        inc, nxt = _group_scan(fn, pre[-1], carry, first, tpr, lane)
        # inc is inclusive of this lane's total; the lane before holds the exclusive.
        ex = tuple(_shfl(v, 1, WAVE, "up") for v in inc)
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
        state[(key, run, "out")] = (outs, nxt)
    outs, nxt = state[(key, run, "out")]
    return (*outs[e], *nxt)


def _group_scan(fn, values, carry, first, tpr: int, lane):
    xs = tuple(_canon(v) for v in values)
    off = 1
    while off < tpr:
        ys = tuple(_shfl(x, off, WAVE, "up") for x in xs)
        comb = fn(ys, xs)
        ok = ge(lane, off)
        xs = tuple(where(ok, c, x) for c, x in zip(comb, xs))
        off *= 2
    if tpr > 1:
        last = _tid() % WAVE - lane + (tpr - 1)
        total = tuple(_shfl(x, last, WAVE, "idx") for x in xs)
    else:
        total = xs
    if first is True:
        return xs, total
    full = tuple(_canon(v) for v in fn(carry, xs))
    nxt = tuple(_canon(v) for v in fn(carry, total))
    if first is False:
        return full, nxt
    return (
        tuple(where(first, x, f) for x, f in zip(xs, full)),
        tuple(where(first, t, n) for t, n in zip(total, nxt)),
    )


def barrier():
    gpu.barrier()


def tile_put(smem, base: int, plane: int, off, value):
    """Park ``value`` at word ``off`` of a staged tile starting at ``base``; a 64-bit
    value's second word goes one ``plane`` further on."""
    for i, word in enumerate(_to_words(_canon(value))):
        fx.memref_store(ArithValue(word), smem, idx(off) + (base + i * plane))


def tile_get(smem, base: int, plane: int, off, k: str):
    words = [
        _val(ArithValue(fx.memref_load(smem, idx(off) + (base + i * plane))))
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


def welford_update(mean, m2, weight, x, mask=None):
    """One sample into a running (mean, m2, weight); masked lanes leave it unchanged."""
    x = _f32(x)
    m = _mask(mask)
    w1 = add(weight, 1.0)
    delta = sub(x, mean)
    mean1 = add(mean, truediv(delta, w1))
    m21 = add(m2, mul(delta, sub(x, mean1)))
    if m is None:
        return mean1, m21, w1
    sel = lambda new, old: ArithValue(_arith.SelectOp(m, _val(new), _val(old)).result)  # noqa: E731
    return sel(mean1, mean), sel(m21, m2), sel(w1, weight)


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


def online_softmax_update(mx, s, x, mask=None):
    x = _f32(x)
    m_new = maximum(mx, x)
    s_new = add(mul(s, _safe_exp_diff(mx, m_new)), _safe_exp_diff(x, m_new))
    m = _mask(mask)
    if m is None:
        return m_new, s_new
    return (
        ArithValue(_arith.SelectOp(m, _val(m_new), _val(mx)).result),
        ArithValue(_arith.SelectOp(m, _val(s_new), _val(s)).result),
    )


def online_softmax_merge(a, b):
    (m1, s1), (m2, s2) = a, b
    m = maximum(m1, m2)
    s = add(mul(s1, _safe_exp_diff(m1, m)), mul(s2, _safe_exp_diff(m2, m)))
    return _canon(m), _canon(s)


def group_online_softmax(mx, s, tpr: int, smem, threads: int, stride: int = 1):
    return _group_reduce((mx, s), online_softmax_merge, tpr, smem, threads, stride)


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
    passes: 268us vs 317us on a causal softmax over 512."""
    xs = _masked_items(items, "f32", float("-inf"))
    gm = group_reduce("max", _tree(xs, maximum), tpr, smem, threads, stride)
    ls = _tree([exp(sub(x, gm)) for x in xs], add)
    return gm, group_reduce("sum", ls, tpr, smem, threads, stride)


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
        add2 = lambda a, b: (_canon(add(a[0], b[0])), _canon(add(a[1], b[1])))  # noqa: E731
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


def pick_fastest(mains, args, mutated, stream=None):
    """An autotuned kernel's first call: time every candidate's ``_main`` on copies of
    the buffers it updates in place, run the fastest on the real arguments, and return
    it. Under graph capture nothing can be timed: the first (heuristic) one runs, and
    None leaves the choice to a later call.
    """
    import torch
    from torch._inductor.runtime.benchmarking import benchmarker

    if torch.cuda.is_current_stream_capturing():
        mains[0](*args, stream=stream)
        return None
    scratch = list(args)
    for i in mutated:
        scratch[i] = args[i].clone()
    times = []
    for fn in mains:
        fn(*scratch, stream=stream)  # compiles (and runs once)
        # The scheduler's settings for timing fusion choices: the defaults (100 cache
        # flushes and up to 25 ms a candidate) took 9 s of a Qwen3 prefill's compile.
        times.append(
            benchmarker.benchmark_gpu(
                lambda fn=fn: fn(*scratch, stream=stream),
                memory_warmup_iters=5,
                benchmark_iters=10,
                max_benchmark_duration=10,
                is_vetted_benchmarking=True,
            )
        )
    best = mains[times.index(min(times))]
    best(*args, stream=stream)
    return best


del _math

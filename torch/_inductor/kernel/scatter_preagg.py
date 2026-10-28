# mypy: allow-untyped-defs
"""
Pre-aggregation for accumulating scatters.

An accumulating scatter (index_put with accumulate=True, index_add, scatter_add,
scatter_reduce(sum), embedding_dense_backward, ...) normally lowers to one
atomic_add per written element, and atomics on a shared destination serialize.
Here each program takes a block of BLOCK_T "tokens" (the iteration positions the
destination depends on) and

  1. computes each token's destination key from the fused index expression,
  2. estimates its distinct destinations, and issues plain atomics in token
     order, like regular Inductor, when pre-aggregating would save few,
  3. otherwise sorts (key, position) pairs so equal destinations become
     contiguous, evaluates the fused value expression at the sorted token
     positions and sums runs of equal keys with a segmented associative scan
     (fp16 on ROCm: a 0/1 matrix product over the block's tokens instead;
     blocks whose keys already come in runs scan in token order unsorted), and
  4. issues one atomic per (distinct key, inner element).

Iteration dims that the destination does not depend on and that map unchanged
onto an output dim ("inner" dims, e.g. the embedding dim) form a second,
vectorized tile axis, so a block sorts one key per token rather than per element.
Values are loaded after the sort, at the permuted tokens: permuting a computed
[BLOCK_T, BLOCK_D] tile in registers is far slower.

Where atomics are not wanted (deterministic algorithms) or not available (8- and
16-bit integers), a sorted lowering sorts all tokens by destination first and
sums every destination's run in a fixed order without atomics. 16-bit outputs
may accumulate into an fp32 copy, so each destination is rounded once, as in
eager (see _stage_in_fp32).
"""

import dataclasses
import functools
import logging
import math
import textwrap
from collections.abc import Callable
from typing import Any

import sympy

import torch
from torch._dynamo.utils import counters
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from .. import config, ir
from ..codegen.triton import TritonSymbols
from ..heuristics.template.triton import ScatterPreaggTarget
from ..ops_handler import DefaultHandler, WrapperHandler
from ..select_algorithm import (
    AlgorithmSelectorCache,
    autotune_select_algorithm,
    ModificationWrapper,
    SymbolicGridFn,
    TritonTemplate,
    TritonTemplateKernel,
)
from ..utils import (
    needs_fallback_due_to_atomic_add_limitations,
    sympy_index_symbol,
    sympy_product,
    triton_type,
)
from ..virtualized import ops, V


log = logging.getLogger(__name__)

# Output dtype -> dtype the segment sums accumulate in.
_ACC_DTYPE: dict[torch.dtype, torch.dtype] = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
    torch.int64: torch.int64,
    torch.int32: torch.int32,
    torch.int16: torch.int32,
    torch.int8: torch.int32,
    torch.uint8: torch.int32,
}
# Triton has no atomics for these.
_NO_ATOMICS = (torch.int16, torch.int8, torch.uint8)

# Ops the template's subgraph codegen (ModificationWrapper) cannot express
# faithfully: its loads are unmasked and it has no iteration ranges.
_UNSUPPORTED_OPS = frozenset(
    {
        "masked",
        "load_seed",
        "rand",
        "randn",
        "randint64",
        "bucketize",
        "inline_asm_elementwise",
        "store",
        "store_reduction",
        "reduction",
        "scan",
        "sort",
    }
)

IndexFn = Callable[[list[sympy.Expr]], list[sympy.Expr]]
ValueFn = Callable[[list[sympy.Expr]], Any]


def _record_skip(reason: str) -> None:
    counters["inductor"][f"scatter_preagg_skipped_{reason}"] += 1
    log.debug("scatter_preagg: skip reason=%s", reason)


def _deterministic() -> bool:
    return config.deterministic or torch.are_deterministic_algorithms_enabled()


def _ineligible(device: torch.device | None, dtype: torch.dtype) -> str | None:
    if device is None or device.type not in ("cuda", "xpu"):
        return "device"
    if dtype not in _ACC_DTYPE:
        return "dtype"
    return None


def _use_sorted(dtype: torch.dtype) -> bool:
    return _deterministic() or dtype in _NO_ATOMICS


def can_bypass_atomic_fallback(dtype: torch.dtype, device: torch.device | None) -> bool:
    """
    True when an accumulating scatter that Inductor would send to the eager
    fallback, for atomic_add dtype support or deterministic algorithms, may be
    offered to this template instead.
    """
    return _ineligible(device, dtype) is None and (
        needs_fallback_due_to_atomic_add_limitations(dtype) or _deterministic()
    )


class _RecordingHandler(DefaultHandler):
    """
    Evaluates an index or value function symbolically to find which iteration
    variables it reads through, which buffers it loads, and whether it uses ops
    the template cannot generate.
    """

    def __init__(self) -> None:
        self.symbols: OrderedSet[sympy.Symbol] = OrderedSet()
        self.read_names: OrderedSet[str] = OrderedSet()
        self.unsupported: OrderedSet[str] = OrderedSet()
        self._num_indirect = 0

    def _note(self, value: Any) -> None:
        if isinstance(value, sympy.Basic):
            self.symbols |= OrderedSet(value.free_symbols)
        elif isinstance(value, (list, tuple)):
            for v in value:
                self._note(v)

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if name in _UNSUPPORTED_OPS:
            self.unsupported.add(name)
        self._note(args)
        self._note(tuple(kwargs.values()))
        return None

    def load(self, name: str, index: sympy.Expr) -> Any:
        self.read_names.add(name)
        self._note(index)
        return None

    def indirect_indexing(self, x, size, check=True, wrap_neg=True) -> sympy.Symbol:
        self._num_indirect += 1
        return sympy.Symbol(
            f"preagg_indirect{self._num_indirect}", integer=True, nonnegative=True
        )


class _ValidityHandler(WrapperHandler):
    """
    Applies the indirect-index semantics of the regular Scatter codegen itself:
    negative-index wrapping, the bounds assert when requested, and (with
    record_validity) a mask of in-range indices, so unchecked out-of-range
    writes are dropped instead of landing out of bounds.
    """

    def __init__(self, inner: Any, record_validity: bool) -> None:
        super().__init__(inner)
        self.record_validity = record_validity
        self.valid: list[Any] = []

    def indirect_indexing(self, x, size, check=True, wrap_neg=True):
        size = sympy.sympify(size)
        if size.is_number:
            size_v = self.constant(int(size), torch.int64)
        else:
            size_v = self.index_expr(size, torch.int64)
        zero = self.constant(0, torch.int64)
        if wrap_neg:
            x = self.where(self.lt(x, zero), self.add(x, size_v), x)
        if check or self.record_validity:
            in_bounds = self.and_(self.ge(x, zero), self.lt(x, size_v))
            if check:
                self.device_assert_async(
                    in_bounds, f"index out of bounds: 0 <= index < {size}"
                )
            if self.record_validity:
                self.valid.append(in_bounds)
        return self._inner.indirect_indexing(x, size, False, False)


class _TemplateIndexHandler(_ValidityHandler):
    """_ValidityHandler for template subgraphs, which have no range trees."""

    def index_expr(self, expr, dtype):
        # The only free symbols are template-local tensors (tokens, inner ids,
        # indices), so render the expression directly.
        kernel = V.kernel
        expr = sympy.sympify(expr)
        text = kernel.kexpr(kernel.rename_indexing(expr))
        if expr.free_symbols:
            line = f"({text}).to({triton_type(dtype)})"
        else:
            line = f"tl.full([], {text}, {triton_type(dtype)})"
        return kernel.cse.generate(
            kernel.compute,
            line,
            dtype=dtype,
            shape=TritonSymbols.get_block_shape(expr),
        )


class _TileValueHandler(_TemplateIndexHandler):
    """
    _TemplateIndexHandler for the value tile. A float load indexed by the token
    alone (a per-token weight) comes out [rows, 1], and Triton then gives the
    whole [rows, cols] tile and its scan that load's layout, converting the
    tile both ways on every inner block (~2x on gfx942), so it loads through a
    pointer broadcast to the tile. Integer ones (indices, masks) stay narrow,
    where the arithmetic on them is cheap.
    """

    def __init__(self, inner: Any, inner_sym: sympy.Symbol, shape: tuple[str, str]):
        super().__init__(inner, record_validity=False)
        self.inner_sym = inner_sym
        self.shape = shape

    def load(self, name: str, index: sympy.Expr) -> Any:
        wrapper = self._inner
        if (
            name in wrapper.fixed_inputs
            or self.inner_sym in index.free_symbols
            or not (dtype := V.graph.get_buffer(name).dtype).is_floating_point
        ):
            return wrapper.load(name, index)
        ptr = f"{wrapper._add_kernel_input(name)} + {wrapper._process_indexing(index)}"
        line = f"tl.load(tl.broadcast_to({ptr}, [{', '.join(self.shape)}]))"
        if (
            dtype in (torch.float16, torch.bfloat16)
            and config.triton.codegen_upcast_to_fp32
        ):
            line += ".to(tl.float32)"
            dtype = torch.float32
        return wrapper.kernel.cse.generate(
            wrapper.kernel.compute, line, dtype=dtype, shape=self.shape
        )


def _unravel(linear: sympy.Expr, sizes: list[sympy.Expr]) -> list[sympy.Expr]:
    """Row-major coordinates of ``linear`` over ``sizes``."""
    coords: list[sympy.Expr] = []
    stride = 1
    for i in reversed(range(len(sizes))):
        if sizes[i] == 1:
            coords.append(sympy.S.Zero)
        elif i == 0:
            coords.append(FloorDiv(linear, stride))
        else:
            coords.append(ModularIndexing(linear, stride, sizes[i]))
        stride *= sizes[i]
    return coords[::-1]


@dataclasses.dataclass
class _ScatterPlan:
    iter_sizes: list[sympy.Expr]
    tok_dims: list[int]
    inner_dims: list[int]
    # Output dim fed by each inner iteration dim.
    inner_out_dims: list[int]
    key_out_dims: list[int]
    out_sizes: list[sympy.Expr]
    out_indexer: Callable[[list[sympy.Expr]], sympy.Expr]
    output_indexer: IndexFn
    value_fn: ValueFn
    acc_dtype: torch.dtype
    # Buffers only the destination keys read.
    index_reads: OrderedSet[str] = dataclasses.field(default_factory=OrderedSet)
    # Whether the last inner dim is contiguous in the output.
    inner_contiguous: bool = True
    # Size symbols guarded to be multiples of 16 (see _align_strides).
    aligned_symbols: OrderedSet[sympy.Symbol] = dataclasses.field(
        default_factory=OrderedSet
    )

    @property
    def key_sizes(self) -> list[sympy.Expr]:
        return [self.out_sizes[j] for j in self.key_out_dims]

    @property
    def num_tokens(self) -> sympy.Expr:
        return sympy_product([self.iter_sizes[d] for d in self.tok_dims])

    @property
    def num_inner(self) -> sympy.Expr:
        return sympy_product([self.iter_sizes[d] for d in self.inner_dims])

    @property
    def num_keys(self) -> sympy.Expr:
        return sympy_product(self.key_sizes)

    def static_sizes(self) -> dict[str, int]:
        """Template constexprs for the sizes that are not symbolic."""
        sizes = {
            "NUM_TOKENS": self.num_tokens,
            "NUM_INNER": self.num_inner,
            "NUM_KEYS": self.num_keys,
        }
        return {name: int(s) for name, s in sizes.items() if s.is_number}

    def iteration_coords(
        self, token: sympy.Expr, inner: sympy.Expr
    ) -> list[sympy.Expr]:
        coords: list[sympy.Expr] = [sympy.S.Zero] * len(self.iter_sizes)
        for dims, linear in ((self.tok_dims, token), (self.inner_dims, inner)):
            sizes = [self.iter_sizes[d] for d in dims]
            for d, c in zip(dims, _unravel(linear, sizes)):
                coords[d] = c
        return coords

    def key_expr(self, out_coords: list[sympy.Expr]) -> sympy.Expr:
        key: sympy.Expr = sympy.S.Zero
        for j in self.key_out_dims:
            key = key * self.out_sizes[j] + out_coords[j]
        return key

    def address(self, key: sympy.Expr, inner: sympy.Expr) -> sympy.Expr:
        coords: list[sympy.Expr] = [sympy.S.Zero] * len(self.out_sizes)
        for j, c in zip(self.key_out_dims, _unravel(key, self.key_sizes)):
            coords[j] = c
        inner_sizes = [self.iter_sizes[d] for d in self.inner_dims]
        for j, c in zip(self.inner_out_dims, _unravel(inner, inner_sizes)):
            coords[j] = c
        return self.out_indexer(coords)


def _hint(expr: sympy.Expr) -> int:
    return V.graph.sizevars.optimization_hint(expr)


def _guarded(cond: sympy.Basic) -> bool:
    """
    Whether ``cond`` holds, guarding symbolic sizes on it only when the hints
    satisfy it: a guard on the false side would recompile every other size.
    """
    hinted = cond.xreplace({s: _hint(s) for s in cond.free_symbols})
    return bool(hinted) and V.graph.sizevars.guard_or_false(cond)


def _guard_within_2gb(nodes: list[ir.IRNode]) -> None:
    """
    On HIP, pointers are marked tt.pointer_range=32 (buffer ops) only for
    storage provably under 2GB; without it the template runs ~2x slower on
    gfx942, so keep it for symbolic sizes too.
    """
    if torch.version.hip is None or not config.triton.emit_pointer_range_32:
        return
    for node in nodes:
        size = sympy.sympify(
            node.get_layout().storage_size() * node.get_dtype().itemsize
        )
        if not size.is_number:
            _guarded(sympy.Le(size, 2**31 - 1))


def _inner_mode(num_inner: sympy.Expr, block_d: int, d_iters: int) -> int:
    """
    INNER_MODE: 0 when inner blocks tile NUM_INNER exactly (no clamp), else 1.
    A symbolic size takes 0 under a guard only if its hint is also a multiple
    of 16, where _align_strides makes it pay off; other sizes share one
    unguarded kernel rather than recompiling per alignment.
    """
    tile = block_d * d_iters
    if num_inner.is_number:
        return 0 if int(num_inner) % tile == 0 else 1
    aligned = _hint(num_inner) % 16 == 0 and _guarded(
        sympy.Eq(sympy.Mod(num_inner, math.lcm(16, tile)), 0)
    )
    return 0 if aligned else 1


def _align_strides(plan: _ScatterPlan, nodes: list[ir.IRNode]) -> None:
    """
    Guard the symbolic row size and strides to multiples of 16 so the kernel is
    marked tt.divisibility for them; a symbolic row size otherwise hides the
    alignment vectorized loads and atomics need (~1.8x on 16-bit rows, gfx942).
    Token and key counts are left unguarded, or each new count would recompile.
    """
    symbols = OrderedSet(plan.num_inner.free_symbols)
    for node in nodes:
        for stride in node.get_layout().stride:
            symbols |= OrderedSet(sympy.sympify(stride).free_symbols)
    plan.aligned_symbols = OrderedSet(
        s for s in symbols if _guarded(sympy.Eq(sympy.Mod(s, 16), 0))
    )


class ScatterPreaggTemplateKernel(TritonTemplateKernel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_extra_template_env_fns(
            self.preagg_def_kernel,
            self.preagg_sizes,
            self.preagg_key,
            self.preagg_value,
            self.preagg_addr,
        )

    @property
    def _plan(self) -> _ScatterPlan:
        if self.subgraphs is None:
            raise AssertionError("scatter_preagg kernel has no value subgraph")
        return self.subgraphs[0].preagg_plan  # type: ignore[attr-defined]

    def preagg_def_kernel(self) -> str:
        names = ["OUT"] + [f"IN{i}" for i in range(len(self.input_nodes) - 1)]
        code = self.def_kernel(*names)
        # Results go through OUT (a mutated input); the template's own output is
        # a placeholder that autotune benchmarking still passes, so take it as
        # an unused argument unless the scheduler has removed it.
        out_name = self.output_node.get_name()
        if out_name not in V.graph.removed_buffers:
            self.args.output(out_name)
        return code

    def jit_lines(self) -> str:
        # Size args get tt.divisibility=16 only when statically known to be
        # multiples of 16; count the symbols guarded to be (see _align_strides).
        aligned = self._plan.aligned_symbols if self.subgraphs else None
        if not aligned:
            return super().jit_lines()
        sizevars = V.graph.sizevars
        known = sizevars.statically_known_multiple_of
        scaled = {s: 16 * s for s in aligned}

        def known_or_guarded(expr, multiple) -> bool:
            if known(expr, multiple):
                return True
            return isinstance(expr, sympy.Expr) and known(
                expr.xreplace(scaled), multiple
            )

        sizevars.statically_known_multiple_of = known_or_guarded  # type: ignore[method-assign]
        try:
            return super().jit_lines()
        finally:
            del sizevars.statically_known_multiple_of

    def preagg_sizes(self) -> str:
        """Definitions of the sizes that are kernel arguments, not constexprs."""
        plan = self._plan
        sizes = {
            "NUM_TOKENS": plan.num_tokens,
            "NUM_INNER": plan.num_inner,
            "NUM_KEYS": plan.num_keys,
        }
        return "\n".join(
            f"{name} = {self.kexpr(self.rename_indexing(size))}"
            for name, size in sizes.items()
            if not size.is_number
        )

    def _codegen_subgraph(
        self,
        body_name: str,
        fixed_inputs: dict[str, str],
        input_shapes: dict[str, tuple[str, ...]],
        emit: Callable[[], list[str]],
    ) -> str:
        with self.create_subgraph_body(body_name):
            handler = ModificationWrapper(
                self,
                0,
                fixed_inputs,
                None,
                input_shapes,
                dict.fromkeys(fixed_inputs, "index"),
            )
            with V.set_ops_handler(handler):
                lines = emit()
            self.codegen_body()
            for line in lines:
                self.body.writeline(line)
            body = self.body.getvalue()
            self.cse.invalidate(OrderedSet())
            return body

    @staticmethod
    def _symbol_of(value: Any) -> sympy.Symbol:
        return sympy_index_symbol(str(getattr(value, "value", value)))

    def preagg_key(
        self,
        tok: str,
        key_out: str,
        valid_out: str,
        body: str = "preagg_key",
        rows: str = "BLOCK_T",
    ) -> str:
        """Destination key and validity of each token in ``tok`` (shape [rows])."""
        plan = self._plan

        def emit() -> list[str]:
            handler = V.get_ops_handler()
            tok_sym = self._symbol_of(handler.load("tok", sympy.S.Zero))
            coords = plan.iteration_coords(tok_sym, sympy.S.Zero)
            indirect = _TemplateIndexHandler(handler, record_validity=True)
            with V.set_ops_handler(indirect):
                out_coords = list(plan.output_indexer(coords))
            key = self.kexpr(self.rename_indexing(plan.key_expr(out_coords)))
            valid = " & ".join(
                [f"tl.full([{rows}], 1, tl.int1)"]
                + [f"({getattr(v, 'value', v)})" for v in indirect.valid]
            )
            return [
                f"{key_out} = tl.full([{rows}], 0, tl.int64) + ({key})",
                f"{valid_out} = {valid}",
            ]

        return self._codegen_subgraph(body, {"tok": tok}, {"tok": (rows,)}, emit)

    def preagg_value(
        self,
        ptok: str,
        inner: str,
        out: str,
        body: str = "preagg_value",
        rows: str = "BLOCK_T",
        cols: str = "BLOCK_D",
    ) -> str:
        """Value at tokens ``ptok`` ([rows, 1]) and inner ids ([1, cols])."""
        plan = self._plan

        def emit() -> list[str]:
            handler = V.get_ops_handler()
            tok_sym = self._symbol_of(handler.load("ptok", sympy.S.Zero))
            inner_sym = self._symbol_of(handler.load("inner", sympy.S.Zero))
            coords = plan.iteration_coords(tok_sym, inner_sym)
            with V.set_ops_handler(_TileValueHandler(handler, inner_sym, (rows, cols))):
                value = plan.value_fn(coords)
            value = getattr(value, "value", value)
            acc = triton_type(plan.acc_dtype)
            cast = f"({value}).to({acc})"
            if len(getattr(value, "shape", None) or ()) == 2:
                return [f"{out} = tl.broadcast_to({cast}, [{rows}, {cols}])"]
            # Constants come out rank 1, which broadcast_to rejects.
            return [f"{out} = tl.full([{rows}, {cols}], 0, {acc}) + {cast}"]

        return self._codegen_subgraph(
            body,
            {"ptok": ptok, "inner": inner},
            {"ptok": (rows, "1"), "inner": ("1", cols)},
            emit,
        )

    def preagg_addr(self, key: str, inner: str) -> str:
        """Offset into OUT of destination ``key`` and inner element ``inner``."""
        key_sym = sympy.Symbol(key, integer=True, nonnegative=True)
        inner_sym = sympy.Symbol(inner, integer=True, nonnegative=True)
        return self.kexpr(self.rename_indexing(self._plan.address(key_sym, inner_sym)))


@SymbolicGridFn
def scatter_preagg_grid(num_tokens, num_inner, meta, *, cdiv):
    inner_per_prog = meta["BLOCK_D"] * meta["D_ITERS"]
    return (cdiv(num_tokens, meta["BLOCK_T"]), cdiv(num_inner, inner_per_prog), 1)


class ScatterPreaggTemplate(TritonTemplate):
    kernel_type = ScatterPreaggTemplateKernel

    def __init__(self, name: str, source: str) -> None:
        super().__init__(
            name=name,
            grid=scatter_preagg_grid,
            source=source,
            always_freeze_layout=True,
        )


_SEGMENTED_SUM = r"""

@triton.jit
def _preagg_segmented_sum(key_a, val_a, key_b, val_b):
    return key_b, tl.where(key_a == key_b, val_a + val_b, val_b)
"""

_TEMPLATE_SOURCE = (
    r"""
{%- macro inner_block(iters, width, vec) -%}
inner_raw = (pid_inner * {{iters}} + it) * {{width}} + tl.arange(0, {{width}})
inner_mask = inner_raw < NUM_INNER
# Clamping hides contiguity, which disables vectorized loads.
if INNER_MODE == 0:
    inner = inner_raw[None, :]
    if {{vec}}:
        atomic_inner = tl.max_contiguous(
            tl.multiple_of(inner_raw, {{vec}}), {{vec}}
        )[None, :]
    else:
        atomic_inner = inner
else:
    inner = tl.minimum(inner_raw, NUM_INNER - 1)[None, :]
    atomic_inner = inner
{%- endmacro %}
{%- macro scan_atomics(value_body) -%}
# Sums runs of equal skey (next_key is its successor) segmented by seg, with
# values at tokens ptok, and adds each run's sum at its end.
seg_end = (skey < NUM_KEYS) & ((skey != next_key) | (pos == BLOCK_T - 1))
out_key = tl.where(seg_end, skey, 0).to(INDEX_DTYPE)[:, None]
# Keys are below the output's storage size, so INDEX_DTYPE holds them.
seg_key = tl.broadcast_to(seg.to(INDEX_DTYPE)[:, None], (BLOCK_T, BLOCK_D))
if SKIP_UNIQUE_SCAN:
    # Every valid token that does not end a segment has a duplicate.
    has_dup = tl.max(((skey < NUM_KEYS) & ~seg_end).to(tl.int32), 0) > 0
else:
    has_dup = True

# The sort is amortized over D_ITERS inner blocks so BLOCK_T can stay
# large (more pre-aggregation) while the register tile stays small.
for it in tl.range(0, D_ITERS):
    {{inner_block("D_ITERS", "BLOCK_D", "ATOMIC_VEC") | indent_except_first(1)}}

    {{preagg_value("ptok", "inner", "val", value_body) | indent_except_first(1)}}

    if has_dup:
        _, acc = tl.associative_scan((seg_key, val), 0, _preagg_segmented_sum)
    else:
        acc = val
    # The address drops out_key when there is a single destination.
    tl.atomic_add(
        tl.broadcast_to(
            OUT + ({{preagg_addr("out_key", "atomic_inner")}}), [BLOCK_T, BLOCK_D]
        ),
        acc.to(OUT_DTYPE),
        mask=seg_end[:, None] & inner_mask[None, :],
        sem="relaxed",
    )
{%- endmacro %}
{{preagg_def_kernel()}}
    {{preagg_sizes() | indent_except_first(1)}}
    pid_tok = tl.program_id(0).to(INDEX_DTYPE)
    pid_inner = tl.program_id(1).to(INDEX_DTYPE)
    pos = tl.arange(0, BLOCK_T).to(INDEX_DTYPE)
    # Subgraph loads are unmasked, so every lane must address a real token.
    # Given one full block, the last block is shifted back to end at NUM_TOKENS
    # (lanes it shares with the previous block are masked off) rather than
    # clamped, which keeps the token range contiguous for the compiler.
    tok_block = pid_tok * BLOCK_T
    if TOK_SHIFT:
        tok_start = tl.minimum(tok_block, NUM_TOKENS - BLOCK_T)
        tok = tok_start + pos
        tok_mask = tok >= tok_block
    else:
        tok_start = tok_block
        tok = tl.minimum(tok_block + pos, NUM_TOKENS - 1)
        tok_mask = tok_block + pos < NUM_TOKENS

    {{preagg_key("tok", "key", "valid") | indent_except_first(1)}}

    dest = tl.where(tok_mask & valid, key, NUM_KEYS).to(KEY_DTYPE)
{% if ADAPTIVE %}
{% if DOT %}
    # Count the block's destinations exactly from its sorted keys (reducing
    # the product's matrix instead costs a second copy of it), and skip the
    # product when it saves few atomics. Masked lanes count as one
    # destination, which only biases partial blocks toward the product.
    sorted_key = tl.sort(dest)
    next_key = tl.gather(sorted_key, tl.minimum(pos + 1, BLOCK_T - 1), 0)
    heads = (sorted_key != next_key) | (pos == BLOCK_T - 1)
    distinct = tl.sum(heads.to(tl.int32), 0)
{% else %}
    # Runs of equal keys in token order: a segmented scan sums them without
    # the sort, which pays when sorting would save few more atomics.
    next_dest = tl.gather(dest, tl.minimum(pos + 1, BLOCK_T - 1), 0)
    runs = tl.sum(((dest != next_dest) | (pos == BLOCK_T - 1)).to(tl.int32), 0)
    distinct = runs.to(tl.float32)
    # Blocks of so few runs take the scan whatever the estimate below, so
    # narrow rows, where the histogram is a good part of the block, skip it.
    # Wider rows lose more to the branch's registers than they save.
    if BLOCK_D > 8 or runs - 1 > BLOCK_T * MIN_SAVED:
        # Estimate the block's distinct destinations from a histogram of
        # hashed keys (linear counting), and skip the sort when it saves few
        # atomics. Counting masked lanes as tokens only biases partial blocks
        # toward the sort, and saves a block-wide reduction.
        live = tok_mask & valid
        hashed = ((dest.to(tl.int64) * 2654435761) >> (32 - LOG_HIST_BINS)) & (
            HIST_BINS - 1
        )
        counts = tl.histogram(hashed.to(tl.int32), HIST_BINS, mask=live)
        # A destination taking an eighth of the block serializes its plain
        # atomics, so such blocks sort whatever they save: a hot bin counts
        # as HIST_BINS more empty ones, which takes the estimate below zero.
        empty = (counts == 0).to(tl.float32) + (counts > BLOCK_T // 8).to(
            tl.float32
        ) * HIST_BINS
        empty = tl.maximum(tl.sum(empty, 0), 1.0)
        distinct = -HIST_BINS * tl.log(empty / HIST_BINS)
    in_order = runs - tl.maximum(distinct, 1.0) <= BLOCK_T * MIN_SAVED
{% endif %}
    if BLOCK_T - distinct < BLOCK_T * MIN_SAVED:
        # Plain atomics in token order, like regular Inductor. Rows of PLAIN_D
        # put fewer tokens in each atomic instruction than BLOCK_D-wide tiles,
        # which matters when hot keys collide within one.
        for chunk in tl.range(0, BLOCK_T // PLAIN_T):
            chunk_block = tok_block + chunk * PLAIN_T
            chunk_pos = tl.arange(0, PLAIN_T).to(INDEX_DTYPE)
            if TOK_SHIFT:
                ctok = tl.minimum(chunk_block, NUM_TOKENS - PLAIN_T) + chunk_pos
                ctok_mask = ctok >= chunk_block
            else:
                ctok = tl.minimum(chunk_block + chunk_pos, NUM_TOKENS - 1)
                ctok_mask = chunk_block + chunk_pos < NUM_TOKENS

            {{preagg_key("ctok", "ckey", "cvalid", "preagg_plain_key", "PLAIN_T") | indent_except_first(3)}}

            clive = ctok_mask & cvalid
            ckey = tl.where(clive, ckey, 0).to(INDEX_DTYPE)[:, None]
            ctok = ctok[:, None]
            for it in tl.range(0, PLAIN_ITERS):
                {{inner_block("PLAIN_ITERS", "PLAIN_D", "PLAIN_VEC") | indent_except_first(4)}}

                {{preagg_value("ctok", "inner", "val", "preagg_plain_value", "PLAIN_T", "PLAIN_D") | indent_except_first(4)}}

                tl.atomic_add(
                    tl.broadcast_to(
                        OUT + ({{preagg_addr("ckey", "atomic_inner")}}),
                        [PLAIN_T, PLAIN_D],
                    ),
                    val.to(OUT_DTYPE),
                    mask=clive[:, None] & inner_mask[None, :],
                    sem="relaxed",
                )
        return
{% endif %}
{% if DOT %}
    # Pre-aggregate on the matrix cores: row i of SAME @ values sums the
    # values of the tokens sharing token i's destination, which the first of
    # them adds. Products with a 0/1 matrix are exact in 16 bits. The matrix
    # compares raw keys, since masking them first costs a layout conversion;
    # dead tokens instead contribute zero and start no segment.
    tok_live = tok_mask & valid
    raw_key = key.to(KEY_DTYPE)
    same = raw_key[:, None] == raw_key[None, :]
    earlier = same & tok_live[None, :] & (pos[None, :] < pos[:, None])
    seg_end = (tl.sum(earlier.to(tl.int32), 1) == 0) & tok_live
    out_key = tl.where(seg_end, raw_key, 0).to(INDEX_DTYPE)[:, None]
    tok_key = tl.where(tok_live, raw_key, 0).to(INDEX_DTYPE)[:, None]
    onehot = same.to(OUT_DTYPE)
    ptok = tok[:, None]
    for it in tl.range(0, D_ITERS):
        {{inner_block("D_ITERS", "BLOCK_D", "ATOMIC_VEC") | indent_except_first(2)}}

        {{preagg_value("ptok", "inner", "val") | indent_except_first(2)}}

        val = tl.where(tok_live[:, None], val.to(OUT_DTYPE), 0.0)
        # 0 * inf is nan, so the product would carry a non-finite value to
        # every destination in the block.
        if tl.max(tl.where(tl.abs(val) < float("inf"), 0, 1)) > 0:
            tl.atomic_add(
                tl.broadcast_to(
                    OUT + ({{preagg_addr("tok_key", "atomic_inner")}}),
                    [BLOCK_T, BLOCK_D],
                ),
                val,
                mask=tok_live[:, None] & inner_mask[None, :],
                sem="relaxed",
            )
        else:
            acc = tl.dot(onehot, val, out_dtype=tl.float32)
            tl.atomic_add(
                tl.broadcast_to(
                    OUT + ({{preagg_addr("out_key", "atomic_inner")}}),
                    [BLOCK_T, BLOCK_D],
                ),
                acc.to(OUT_DTYPE),
                mask=seg_end[:, None] & inner_mask[None, :],
                sem="relaxed",
            )
{% else %}
{% if ADAPTIVE %}
    if in_order:
        skey = dest
        sorted_pos = pos
        # A key can recur after others, so the scan segments by run number.
        run_end = (dest != next_dest).to(KEY_DTYPE)
        seg = tl.cumsum(run_end, 0) - run_end
{% if BLOCK_D == 1 %}
        # Single columns scan in a copy of their own, as one shared with the
        # sort hides that the tokens are contiguous (~1.3x on gfx942). Wider
        # tiles lose more to its registers.
        ptok = tok[:, None]
        next_key = next_dest
        {{scan_atomics("preagg_run_value") | indent_except_first(2)}}
        return
{% endif %}
    else:
        packed = tl.sort((dest << LOG_BLOCK_T) | pos.to(KEY_DTYPE))
        skey = packed >> LOG_BLOCK_T
        sorted_pos = (packed & (BLOCK_T - 1)).to(INDEX_DTYPE)
        seg = skey
{% else %}
    packed = tl.sort((dest << LOG_BLOCK_T) | pos.to(KEY_DTYPE))
    skey = packed >> LOG_BLOCK_T
    sorted_pos = (packed & (BLOCK_T - 1)).to(INDEX_DTYPE)
    seg = skey
{% endif %}
    if TOK_SHIFT:
        ptok = (tok_start + sorted_pos)[:, None]
    else:
        ptok = tl.minimum(tok_start + sorted_pos, NUM_TOKENS - 1)[:, None]
    next_key = tl.gather(skey, tl.minimum(pos + 1, BLOCK_T - 1), 0)
    {{scan_atomics("preagg_value") | indent_except_first(1)}}
{% endif %}
"""
    + _SEGMENTED_SUM
)

scatter_preagg_template = ScatterPreaggTemplate("scatter_preagg", _TEMPLATE_SOURCE)

# Widest row of the adaptive plain path, about one wavefront's worth of a
# regular Inductor scatter's elements.
_PLAIN_MAX_D = 256

# Up to this many destinations, every block of the default configs (at least
# 128 tokens) sorts.
_ALWAYS_SORTED_KEYS = 64

# The sorted lowering. Tokens arrive sorted by destination (SKEYS, with PERM
# their original positions), so each destination is one contiguous segment.
# Each program sums the segments in its chunk of BLOCK_T sorted tokens in a
# fixed order. A segment that starts and ends in the chunk has no other writer
# and is added to OUT directly; the parts of segments that cross a chunk
# boundary go to PART (slot 0: the chunk's first segment, continued from the
# previous chunk; slot 1: its last segment, continued into the next) for the
# fixup kernel to add in chunk order.
_SORTED_PRELUDE = r"""
{{preagg_def_kernel()}}
    {{preagg_sizes() | indent_except_first(1)}}
    NUM_CHUNKS = (NUM_TOKENS + BLOCK_T - 1) // BLOCK_T
    PART = IN0
    SKEYS = IN1
    PERM = IN2
    pid_tok = tl.program_id(0).to(INDEX_DTYPE)
    pid_inner = tl.program_id(1).to(INDEX_DTYPE)
    pos = tl.arange(0, BLOCK_T).to(INDEX_DTYPE)
    first = pid_tok * BLOCK_T
    last = tl.minimum(first + BLOCK_T, NUM_TOKENS) - 1
    first_key = tl.load(SKEYS + first)
    last_key = tl.load(SKEYS + last)
    prev_key = tl.where(pid_tok > 0, tl.load(SKEYS + tl.maximum(first - 1, 0)), -1)
    next_key = tl.where(
        last + 1 < NUM_TOKENS, tl.load(SKEYS + tl.minimum(last + 1, NUM_TOKENS - 1)), -1
    )
"""


_SORTED_INNER_LOOP = r"""
    for it in tl.range(0, D_ITERS):
        inner_raw = (pid_inner * D_ITERS + it) * BLOCK_D + tl.arange(0, BLOCK_D)
        inner_mask = inner_raw < NUM_INNER
        if INNER_MODE == 0:
            inner = inner_raw[None, :]
        else:
            inner = tl.minimum(inner_raw, NUM_INNER - 1)[None, :]
"""

_SORTED_CHUNK_SOURCE = (
    _SORTED_PRELUDE
    + r"""
    spos = first + pos
    sp = tl.minimum(spos, NUM_TOKENS - 1)
    skey = tl.where(spos < NUM_TOKENS, tl.load(SKEYS + sp), NUM_KEYS)
    ptok = tl.load(PERM + sp).to(INDEX_DTYPE)[:, None]
    next_skey = tl.gather(skey, tl.minimum(pos + 1, BLOCK_T - 1), 0)
    seg_end = (skey < NUM_KEYS) & ((skey != next_skey) | (pos == BLOCK_T - 1))
    head = (skey == first_key) & (first_key == prev_key)
    tail = (skey == last_key) & (last_key == next_key) & ~head
    direct = (seg_end & ~head & ~tail)[:, None]
    out_key = tl.where(seg_end, skey, 0).to(INDEX_DTYPE)[:, None]
    seg_key = tl.broadcast_to(skey.to(INDEX_DTYPE)[:, None], (BLOCK_T, BLOCK_D))
"""
    + _SORTED_INNER_LOOP
    + r"""
        {{preagg_value("ptok", "inner", "val") | indent_except_first(2)}}

        _, acc = tl.associative_scan((seg_key, val), 0, _preagg_segmented_sum)
        mask = direct & inner_mask[None, :]
        addr = tl.broadcast_to(
            OUT + ({{preagg_addr("out_key", "inner")}}), [BLOCK_T, BLOCK_D]
        )
        old = tl.load(addr, mask=mask, other=0).to(acc.dtype)
        tl.store(addr, (old + acc).to(OUT_DTYPE), mask=mask)
        part = PART + pid_tok * NUM_INNER + inner_raw[None, :]
        tl.store(
            part + tl.where(head, 0, NUM_CHUNKS * NUM_INNER)[:, None],
            acc,
            mask=(seg_end & (head | tail))[:, None] & inner_mask[None, :],
        )
"""
    + _SEGMENTED_SUM
)

# The chunk where a crossing segment starts owns it: it adds the head parts of
# the following chunks, in order, to its tail part and writes the destination.
_SORTED_FIXUP_SOURCE = (
    _SORTED_PRELUDE
    + r"""
    owns = (
        (last_key == next_key)
        & ((first_key != last_key) | (first_key != prev_key))
        & (last_key < NUM_KEYS)
    )
    if owns:
        out_key = last_key.to(INDEX_DTYPE)
"""
    + textwrap.indent(_SORTED_INNER_LOOP, "    ")
    + r"""
            acc = tl.load(
                PART + NUM_CHUNKS * NUM_INNER + pid_tok * NUM_INNER + inner_raw,
                mask=inner_mask,
            )[None, :]
            # Keys are sorted, so a later chunk continues the segment iff its
            # first key is still last_key.
            chunk = pid_tok + 1 + tl.arange(0, 32).to(INDEX_DTYPE)
            more = True
            while more:
                in_seg = (chunk < NUM_CHUNKS) & (
                    tl.load(SKEYS + tl.minimum(chunk, NUM_CHUNKS - 1) * BLOCK_T)
                    == last_key
                )
                heads = tl.load(
                    PART + chunk[:, None] * NUM_INNER + inner_raw[None, :],
                    mask=in_seg[:, None] & inner_mask[None, :],
                    other=0,
                )
                acc += tl.sum(heads, 0)[None, :]
                more = (tl.max(chunk) + 1 < NUM_CHUNKS) & (
                    tl.min(in_seg.to(tl.int32)) == 1
                )
                chunk += 32
            addr = tl.broadcast_to(
                OUT + ({{preagg_addr("out_key", "inner")}}), [1, BLOCK_D]
            )
            mask = inner_mask[None, :]
            old = tl.load(addr, mask=mask, other=0).to(acc.dtype)
            tl.store(addr, (old + acc).to(OUT_DTYPE), mask=mask)
"""
)

scatter_sorted_chunk_template = ScatterPreaggTemplate(
    "scatter_sorted_chunk", _SORTED_CHUNK_SOURCE
)
scatter_sorted_fixup_template = ScatterPreaggTemplate(
    "scatter_sorted_fixup", _SORTED_FIXUP_SOURCE
)


def _analyze(
    out_layout: ir.Layout,
    iter_sizes: list[sympy.Expr],
    value_fn: ValueFn,
    output_indexer: IndexFn,
    acc_dtype: torch.dtype,
) -> tuple[_ScatterPlan, _RecordingHandler, _RecordingHandler] | str:
    """
    Split the iteration dims into token and inner dims. Returns the plan and the
    reads and ops recorded from the key and value functions, or a skip reason.
    """
    iter_syms = [
        sympy.Symbol(f"preagg_i{d}", integer=True, nonnegative=True)
        for d in range(len(iter_sizes))
    ]
    key_rec = _RecordingHandler()
    with V.set_ops_handler(key_rec):
        out_coords = [sympy.sympify(c) for c in output_indexer(iter_syms)]
    value_rec = _RecordingHandler()
    with V.set_ops_handler(value_rec):
        value_fn(iter_syms)
    out_sizes = [sympy.sympify(s) for s in out_layout.size]
    if len(out_coords) != len(out_sizes):
        return "rank_mismatch"

    tok_dims: list[int] = []
    inner_dims: list[int] = []
    inner_out_dims: list[int] = []
    for d, sym in enumerate(iter_syms):
        uses = [j for j, c in enumerate(out_coords) if sym in c.free_symbols]
        if sym not in key_rec.symbols and len(uses) == 1 and out_coords[uses[0]] == sym:
            inner_dims.append(d)
            inner_out_dims.append(uses[0])
        else:
            tok_dims.append(d)

    plan = _ScatterPlan(
        iter_sizes=iter_sizes,
        tok_dims=tok_dims,
        inner_dims=inner_dims,
        inner_out_dims=inner_out_dims,
        key_out_dims=[j for j in range(len(out_sizes)) if j not in inner_out_dims],
        out_sizes=out_sizes,
        out_indexer=out_layout.make_indexer(),
        output_indexer=output_indexer,
        value_fn=value_fn,
        acc_dtype=acc_dtype,
    )
    return plan, key_rec, value_rec


def _realize(node: ir.TensorBox) -> ir.TensorBox:
    node.realize()
    ir.as_storage_and_layout(node, freeze=True)
    return node


def _materialize_values(
    value_fn: ValueFn, sizes: list[sympy.Expr], device: torch.device, dtype: torch.dtype
) -> ValueFn:
    """
    Compute the values into their own buffer, before the scatter mutates the
    output, and return a loader of it.
    """
    values = ir.Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=lambda idx: ops.to_dtype(value_fn(idx), dtype),
        ranges=sizes,
    )
    return _realize(values).make_loader()


def _key_value(plan: _ScatterPlan, token: sympy.Expr) -> Any:
    """
    int64 destination key of ``token`` in a regular kernel, NUM_KEYS for a
    dropped write (unchecked and out of range).
    """
    coords = plan.iteration_coords(token, sympy.S.Zero)
    handler = _ValidityHandler(V.get_ops_handler(), record_validity=True)
    with V.set_ops_handler(handler):
        out_coords = plan.output_indexer(coords)
    key = ops.index_expr(plan.key_expr(out_coords), torch.int64)
    if handler.valid:
        valid = functools.reduce(ops.and_, handler.valid)
        key = ops.where(valid, key, ops.index_expr(plan.num_keys, torch.int64))
    return key


def _keys(plan: _ScatterPlan, device: torch.device) -> ir.TensorBox:
    return ir.Pointwise.create(
        device=device,
        dtype=torch.int64,
        inner_fn=lambda idx: _key_value(plan, idx[0]),
        ranges=[plan.num_tokens],
    )


def _materialize_keys(plan: _ScatterPlan, device: torch.device) -> IndexFn:
    """
    Compute the destination keys into their own buffer, before the scatter
    mutates the output, and return an output indexer that reads them.
    """
    load_key = _realize(_keys(plan, device)).make_loader()

    def output_indexer(coords: list[sympy.Expr]) -> list[sympy.Expr]:
        token: sympy.Expr = sympy.S.Zero
        for d in plan.tok_dims:
            token = token * plan.iter_sizes[d] + coords[d]
        key = ops.indirect_indexing(
            load_key([token]), plan.num_keys, check=False, wrap_neg=False
        )
        out: list[sympy.Expr] = [sympy.S.Zero] * len(plan.out_sizes)
        for j, c in zip(plan.key_out_dims, _unravel(key, plan.key_sizes)):
            out[j] = c
        for d, j in zip(plan.inner_dims, plan.inner_out_dims):
            out[j] = coords[d]
        return out

    return output_indexer


def _plan(
    out_layout: ir.Layout,
    out_name: str,
    sizes: list[sympy.Expr],
    value_fn: ValueFn,
    output_indexer: IndexFn,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[_ScatterPlan, OrderedSet[str]] | str:
    """
    The scatter's plan and the buffers the template reads. Index and value
    functions the template cannot inline (unsupported ops, or reads of the
    output it mutates) are computed into buffers first.
    """

    def analyze():
        return _analyze(out_layout, sizes, value_fn, output_indexer, _ACC_DTYPE[dtype])

    analyzed = analyze()
    if isinstance(analyzed, str):
        return analyzed
    plan, key_rec, value_rec = analyzed
    if value_rec.unsupported or out_name in value_rec.read_names:
        value_fn = _materialize_values(value_fn, sizes, device, dtype)
        counters["inductor"]["scatter_preagg_materialized_values"] += 1
    if key_rec.unsupported or out_name in key_rec.read_names:
        output_indexer = _materialize_keys(plan, device)
        counters["inductor"]["scatter_preagg_materialized_keys"] += 1
    if value_fn is not plan.value_fn or output_indexer is not plan.output_indexer:
        analyzed = analyze()
        if isinstance(analyzed, str):
            return analyzed
        plan, key_rec, value_rec = analyzed
    if key_rec.unsupported or value_rec.unsupported:
        return "unsupported_op"
    reads = key_rec.read_names | value_rec.read_names
    if out_name in reads:
        return "reads_output"
    plan.index_reads = key_rec.read_names - value_rec.read_names
    return plan, reads


def maybe_preaggregated_scatter(
    out: ir.TensorBox,
    value_fn: ValueFn,
    iter_sizes: Any,
    output_indexer: IndexFn,
    inputs: list[Any],
    atomic_fallback: Callable[[], Any],
) -> bool:
    """
    Lower ``out[output_indexer(i)] += value_fn(i)`` over ``iter_sizes`` with the
    pre-aggregating template. ``inputs`` are the IR nodes value_fn and
    output_indexer read. If the template declines a scatter that reached the
    caller only through can_bypass_atomic_fallback, ``atomic_fallback()`` is
    emitted instead. Returns False, emitting nothing, when the caller should
    lower a regular atomic Scatter.
    """
    read_names: OrderedSet[str] = OrderedSet()
    for x in inputs:
        if isinstance(x, ir.IRNode):
            read_names |= x.get_read_names()
            # A realized input reports the reads of its producer, not itself.
            if (name := ir.try_get_name(x)) is not None:
                read_names.add(name)
    sizes = [sympy.sympify(s) for s in iter_sizes]
    if _try_template(out, value_fn, sizes, output_indexer, read_names):
        return True
    if (
        needs_fallback_due_to_atomic_add_limitations(out.get_dtype())
        or _deterministic()
    ):
        atomic_fallback()
        return True
    return False


def _output_storage(out: ir.TensorBox) -> tuple[ir.StorageBox, ir.Layout] | None:
    """Storage and layout a (possibly view) output is scattered through."""
    try:
        view = out.data
        if isinstance(view, ir.BaseView):
            view = ir.ExternKernel.convert_to_reinterpret_view(view)
        storage, layout = ir.as_storage_and_layout(view, freeze=True)
    except NotImplementedError:
        return None
    if not isinstance(storage.data, ir.Buffer):
        return None
    return storage, layout


def _scatter_into_copy(
    reason: str,
    out: ir.TensorBox,
    dtype: torch.dtype,
    value_fn: ValueFn,
    sizes: list[sympy.Expr],
    output_indexer: IndexFn,
    read_names: OrderedSet[str],
) -> bool:
    """Scatter into a contiguous ``dtype`` copy of ``out``, then copy it back."""
    from ..lowering import lowerings, to_dtype

    tmp = _realize(to_dtype(out, dtype, copy=True))
    if _output_storage(tmp) is None or not _try_template(
        tmp, value_fn, sizes, output_indexer, read_names
    ):
        return False
    lowerings[torch.ops.aten.copy_.default](out, tmp)
    counters["inductor"][f"scatter_preagg_{reason}"] += 1
    return True


def _stage_in_fp32(
    dtype: torch.dtype, num_keys: sympy.Expr, matrix_preagg: bool
) -> bool:
    """
    Whether to accumulate a 16-bit output in an fp32 copy rather than with
    16-bit atomics, which round every block's partial sum into the output.
    """
    if dtype == torch.bfloat16:
        # Slower and less accurate than the copy on gfx942 at any contention.
        return True
    # Blocks that issue plain atomics would take twice as many in fp32 as
    # packed fp16 ones, so fp16 is staged only when every block sorts, where
    # the copy is small and its rounding saving comes free. The matrix
    # product's blocks lose their speed with fp32 atomics.
    return (
        dtype == torch.float16
        and not matrix_preagg
        and isinstance(num_keys, (int, sympy.Integer))
        and num_keys <= _ALWAYS_SORTED_KEYS
    )


def _try_template(
    out: ir.TensorBox,
    value_fn: ValueFn,
    sizes: list[sympy.Expr],
    output_indexer: IndexFn,
    read_names: OrderedSet[str],
) -> bool:
    device, dtype = out.get_device(), out.get_dtype()
    if (reason := _ineligible(device, dtype)) is not None:
        _record_skip(reason)
        return False
    storage = _output_storage(out)
    if storage is None:
        return _scatter_into_copy(
            "output_copy", out, dtype, value_fn, sizes, output_indexer, read_names
        )
    out_storage, out_layout = storage
    out_buf = out_storage.data
    if not out_layout.size or 0 in [_hint(s) for s in (*out_layout.size, *sizes)]:
        _record_skip("empty")
        return False

    # Tracing the loaders indexes any realized input directly, which needs a
    # fixed layout.
    for name in read_names:
        buf = V.graph.try_get_buffer(name)
        if buf is not None and isinstance(buf.get_layout(), ir.FlexibleLayout):
            buf.freeze_layout()
    planned = _plan(
        out_layout, out_buf.get_name(), sizes, value_fn, output_indexer, device, dtype
    )
    if isinstance(planned, str):
        _record_skip(planned)
        return False
    plan, reads = planned

    use_sorted = _use_sorted(dtype)
    num_tokens, num_inner, num_keys = (
        _hint(plan.num_tokens),
        _hint(plan.num_inner),
        _hint(plan.num_keys),
    )
    # The sorted lowering replaces the eager fallback rather than atomics, so
    # the size gate does not apply to it.
    if not use_sorted and num_tokens * num_inner < config.scatter_preagg_min_index_size:
        _record_skip("too_small")
        return False
    plan.inner_contiguous = not plan.inner_out_dims or (
        V.graph.sizevars.statically_known_equals(
            out_layout.stride[plan.inner_out_dims[-1]], 1
        )
    )
    matrix_preagg = ScatterPreaggTarget.of(device).matrix_preagg(
        dtype, num_inner, plan.inner_contiguous
    )
    if _stage_in_fp32(dtype, plan.num_keys, matrix_preagg) and not use_sorted:
        return _scatter_into_copy(
            "fp32_copy",
            out,
            torch.float32,
            plan.value_fn,
            sizes,
            plan.output_indexer,
            read_names,
        )

    inputs: list[ir.IRNode] = []
    for name in reads:
        buf = V.graph.try_get_buffer(name)
        if buf is None:
            _record_skip("unknown_input")
            return False
        if isinstance(buf.get_layout(), ir.FlexibleLayout):
            buf.freeze_layout()
        inputs.append(buf)

    # Deterministic results need the same config on every compile.
    with config.patch(max_autotune=config.max_autotune and not use_sorted):
        configs = V.choices.get_scatter_preagg_configs(
            num_tokens,
            num_inner,
            dtype,
            device_type=device.type,
            inner_contiguous=plan.inner_contiguous,
            num_keys=num_keys,
            device=device,
        )
    if not use_sorted:
        # The atomic template packs (key, position) into one integer; the
        # sorted lowering takes destinations too many for that.
        key_bits = (num_keys + 1).bit_length()
        packable = [c for c in configs if key_bits + c.block_t.bit_length() - 1 <= 62]
        if not packable:
            use_sorted = True
            counters["inductor"]["scatter_preagg_sorted_key_space"] += 1
        configs = packable or configs

    # The template inlines value_fn and output_indexer through the plan; the
    # buffer wraps the value function so index-width selection sees its reads.
    value_subgraph = ir.ComputedBuffer(
        name=None,
        layout=ir.FlexibleLayout(device=device, dtype=dtype, size=sizes),
        data=ir.Pointwise(
            device=device, dtype=dtype, inner_fn=plan.value_fn, ranges=sizes
        ),
    )
    value_subgraph.preagg_plan = plan  # type: ignore[attr-defined]
    out = ir.TensorBox(out_storage)
    _guard_within_2gb([out, *inputs])
    if not plan.num_inner.is_number and any(
        _inner_mode(plan.num_inner, c.block_d, c.d_iters) == 0 for c in configs
    ):
        _align_strides(plan, [out, *inputs])

    lower = _lower_sorted if use_sorted else _lower_atomic
    if not lower(out, plan, value_subgraph, inputs, configs):
        _record_skip("codegen_failed")
        return False
    counters["inductor"]["scatter_preagg_applied"] += 1
    if use_sorted:
        counters["inductor"]["scatter_preagg_sorted"] += 1
    log.debug(
        "scatter_preagg: applied to %s tokens=%d inner=%d keys=%d sorted=%s",
        out_buf.get_name(),
        num_tokens,
        num_inner,
        num_keys,
        use_sorted,
    )
    return True


def _placeholder_layout(out: ir.TensorBox) -> ir.FixedLayout:
    return ir.FixedLayout(
        device=out.get_device(), dtype=out.get_dtype(), size=[1], stride=[1]
    )


def _index_gen_fns(
    plan: _ScatterPlan, input_nodes: list[ir.IRNode]
) -> dict[int, Callable[[ir.Buffer], torch.Tensor]]:
    """
    Autotuning benchmarks integer inputs as zeros, which sends every write to
    one destination. Configs that win there lose on spread indices and vice
    versa, so give index tensors both: half spread uniformly, half on one
    destination.
    """
    bound = min((_hint(s) for s in plan.key_sizes), default=1)

    def mixed(node: ir.Buffer) -> torch.Tensor:
        t = AlgorithmSelectorCache.benchmark_example_value(node)
        flat = t.view(-1) if t.is_contiguous() else t
        flat[: flat.shape[0] // 2].random_(0, bound)
        return t

    return {
        i: mixed
        for i, node in enumerate(input_nodes)
        if node.get_name() in plan.index_reads
        and not node.get_dtype().is_floating_point
        and node.get_dtype() != torch.bool
    }


def _lower_atomic(
    out: ir.TensorBox,
    plan: _ScatterPlan,
    value_subgraph: ir.ComputedBuffer,
    inputs: list[ir.IRNode],
    configs: list[Any],
) -> bool:
    dtype = out.get_dtype()
    target = ScatterPreaggTarget.of(out.get_device())
    few_keys = _hint(plan.num_keys) <= 64
    # On CDNA, 32-bit atomics run ~1.7x faster at two elements per thread than
    # at the four the compiler derives from load contiguity, except for wide
    # rows into a handful of destinations.
    atomic_vec = 2 if target.cdna3_or_later else 0
    # Plain atomics run up to 2x faster with one 32-bit word per lane, so a
    # wavefront updates whole cache lines (fewer L2 atomic transactions, which
    # bound hot destinations). The sort path keeps two: one word per lane costs
    # it registers for a layout conversion after the scan. So do outputs of
    # 2GB or more, whose 64-bit addresses already halve occupancy. The word
    # also sets the width of value loads, which 16-bit inputs keep at 32 bits
    # once the output outgrows L2, and atomics wait on memory rather than
    # issue.
    plain_vec = atomic_vec
    out_bytes = _hint(out.get_layout().storage_size()) * dtype.itemsize
    if target.cdna3_or_later and out_bytes < 2**31:
        elem_bytes = dtype.itemsize
        if out_bytes > target.l2_bytes:
            elem_bytes = min(
                [elem_bytes]
                + [
                    i.get_dtype().itemsize
                    for i in inputs
                    if i.get_dtype().is_floating_point
                ]
            )
        plain_vec = max(1, 4 // elem_bytes)

    # Sorting pays less on wide rows, whose scans cost more than the atomics
    # they save, and with at least as many destinations as tokens, whose
    # duplicates are cold. Each doubles the saving required (gfx942).
    min_saved = config.scatter_preagg_min_saved
    if _hint(plan.num_inner) > 128:
        min_saved *= 2
    if _hint(plan.num_tokens) <= _hint(plan.num_keys):
        min_saved *= 2

    def adaptive_kwargs(cfg, inner_mode, dot) -> dict[str, Any]:
        block_min_saved = min_saved
        if dot:
            # The product costs little more than plain atomics without
            # duplicates: it pays at half the sort's saving, and is taken
            # without counting whenever uniformly spread keys would save
            # enough (skew only adds duplicates) (gfx942).
            block_min_saved /= 2
            keys = _hint(plan.num_keys)
            distinct = keys * (1 - (1 - 1 / keys) ** cfg.block_t)
        elif isinstance(plan.num_keys, (int, sympy.Integer)):
            # With fewer keys than tokens per block, full blocks always save
            # enough atomics to sort.
            distinct = int(plan.num_keys)
        else:
            distinct = cfg.block_t
        if distinct <= cfg.block_t * (1 - block_min_saved):
            return {"ADAPTIVE": False}
        hist_bins = min(2 * cfg.block_t, 2048)
        adaptive = {
            "ADAPTIVE": True,
            "HIST_BINS": hist_bins,
            "LOG_HIST_BINS": hist_bins.bit_length() - 1,
            "MIN_SAVED": block_min_saved,
        }
        if not plan.inner_contiguous:
            # Strided inner elements take a transaction each, so the plain
            # path runs one column at a time with lanes over tokens, like a
            # regular Inductor scatter. Lanes over a power-of-two row stride
            # would also put a whole atomic on one memory channel (up to 7x
            # on gfx942).
            return adaptive | {
                "PLAIN_VEC": 0,
                "PLAIN_T": cfg.block_t,
                "PLAIN_D": 1,
                "PLAIN_ITERS": cfg.d_iters * cfg.block_d,
            }
        # Widen the plain path's rows by a power of two that divides D_ITERS,
        # so each program still covers exactly its own inner range, up to
        # _PLAIN_MAX_D. Chunks of fewer than 128 tokens pay only in rows at
        # least 128 wide (gfx942); the matrix product's short blocks keep
        # 32-token chunks, which cost less per atomic.
        widen = 1
        if inner_mode == 0:
            widen = cfg.d_iters & -cfg.d_iters
            min_chunk = 32 if dot else 1

            def too_wide(widen: int) -> bool:
                rows, chunk = cfg.block_d * widen, cfg.block_t // widen
                return (
                    rows > _PLAIN_MAX_D
                    or chunk < min_chunk
                    or (not dot and rows < 128 and chunk < 128)
                )

            while widen > 1 and too_wide(widen):
                widen //= 2
        return adaptive | {
            "PLAIN_VEC": plain_vec,
            "PLAIN_T": cfg.block_t // widen,
            "PLAIN_D": cfg.block_d * widen,
            "PLAIN_ITERS": cfg.d_iters // widen,
        }

    layout = _placeholder_layout(out)
    choices: list[Any] = []
    for cfg in configs:
        log_block_t = cfg.block_t.bit_length() - 1
        int32_keys = _guarded(sympy.Lt(plan.num_keys + 1, 2 ** (31 - log_block_t)))
        # The [BLOCK_T, BLOCK_T] matrix outgrows registers past 128 tokens.
        dot = (
            target.matrix_preagg(dtype, _hint(plan.num_inner), plan.inner_contiguous)
            and cfg.block_d >= 32
            and 16 <= cfg.block_t <= 128
        )
        wide_few_keys = cfg.block_d >= 64 and few_keys and not dot
        inner_mode = _inner_mode(plan.num_inner, cfg.block_d, cfg.d_iters)
        adaptive = adaptive_kwargs(cfg, inner_mode, dot)
        num_warps = cfg.num_warps
        if dot and not adaptive["ADAPTIVE"]:
            # Without plain atomics to serve, the product runs faster on 128
            # threads.
            num_warps = max(num_warps, target.warps(128))
        scatter_preagg_template.maybe_append_choice(
            choices,
            input_nodes=(out, *inputs),
            layout=layout,
            num_stages=cfg.num_stages,
            num_warps=num_warps,
            subgraphs=[value_subgraph],
            mutated_inputs=[out],
            call_sizes=[plan.num_tokens, plan.num_inner],
            BLOCK_T=cfg.block_t,
            BLOCK_D=cfg.block_d,
            D_ITERS=cfg.d_iters,
            LOG_BLOCK_T=log_block_t,
            KEY_DTYPE="tl.int32" if int32_keys else "tl.int64",
            INNER_MODE=inner_mode,
            OUT_DTYPE=triton_type(dtype),
            DOT=dot,
            ATOMIC_VEC=(0 if cfg.block_d < atomic_vec or wide_few_keys else atomic_vec),
            # Skipping the scan for blocks without a repeated destination pays
            # for its check only on wide slices (gfx942).
            SKIP_UNIQUE_SCAN=cfg.block_d >= 64,
            # The shifted token block needs one full block; the clamped one
            # handles any size.
            TOK_SHIFT=_guarded(sympy.Ge(plan.num_tokens, cfg.block_t)),
            **adaptive,
            **plan.static_sizes(),
        )
    if not choices:
        return False
    # MultiTemplateBuffer (deferred selection) does not track mutated inputs, so
    # the in-place result would be dropped.
    input_nodes = [out, *inputs]
    autotune_select_algorithm(
        "scatter_preagg",
        choices,
        input_nodes,
        layout,
        input_gen_fns=_index_gen_fns(plan, input_nodes),
        return_multi_template=False,
    )
    return True


def _lower_sorted(
    out: ir.TensorBox,
    plan: _ScatterPlan,
    value_subgraph: ir.ComputedBuffer,
    inputs: list[ir.IRNode],
    configs: list[Any],
) -> bool:
    """
    Atomic-free lowering: sort the tokens by destination, then sum each
    destination's run in a fixed order (see _SORTED_CHUNK_SOURCE).
    """
    from ..lowering import empty_strided, lowerings

    cfg = configs[-1]
    device = out.get_device()
    sorted_keys, perm = lowerings[torch.ops.aten.sort.stable](
        _keys(plan, device), stable=True
    )
    num_chunks = FloorDiv(plan.num_tokens + cfg.block_t - 1, cfg.block_t)
    part = empty_strided(
        [2 * num_chunks * plan.num_inner], [1], dtype=plan.acc_dtype, device=device
    )
    sort_nodes = [_realize(x) for x in (part, sorted_keys, perm)]

    layout = _placeholder_layout(out)
    stages = [
        ("scatter_sorted_chunk", scatter_sorted_chunk_template, [out, part], inputs),
        ("scatter_sorted_fixup", scatter_sorted_fixup_template, [out], []),
    ]
    # Generate both kernels before emitting either.
    selections = []
    for name, template, mutated, extra in stages:
        choices: list[Any] = []
        input_nodes = [out, *sort_nodes, *extra]
        template.maybe_append_choice(
            choices,
            input_nodes=input_nodes,
            layout=layout,
            num_stages=cfg.num_stages,
            num_warps=cfg.num_warps,
            subgraphs=[value_subgraph],
            mutated_inputs=mutated,
            call_sizes=[plan.num_tokens, plan.num_inner],
            BLOCK_T=cfg.block_t,
            BLOCK_D=cfg.block_d,
            D_ITERS=cfg.d_iters,
            INNER_MODE=_inner_mode(plan.num_inner, cfg.block_d, cfg.d_iters),
            OUT_DTYPE=triton_type(out.get_dtype()),
            **plan.static_sizes(),
        )
        if not choices:
            return False
        selections.append((name, choices, input_nodes))
    for name, choices, input_nodes in selections:
        autotune_select_algorithm(
            name, choices, input_nodes, layout, return_multi_template=False
        )
    return True

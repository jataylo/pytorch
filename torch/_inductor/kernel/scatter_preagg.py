# mypy: allow-untyped-defs
"""
Sort-based pre-aggregation for atomic-add scatters.

An accumulating scatter (index_put with accumulate=True, index_add, scatter_add,
scatter_reduce(sum), embedding_dense_backward, ...) normally lowers to one
atomic_add per written element. When many elements target the same
destination, those atomics serialize. This template instead has each program
take a block of BLOCK_T "tokens" (the iteration positions the destination depends
on) and:

  1. computes each token's destination key from the fused index expression,
  2. sorts (key, position) pairs so equal destinations become contiguous,
  3. evaluates the fused value expression at the sorted token positions,
  4. sums runs of equal keys with a segmented associative scan,
  5. issues one atomic per (distinct key, inner element) instead of one per element.

Iteration dims that the destination does not depend on and that map unchanged
onto an output dim ("inner" dims, e.g. the embedding dim) are processed as a
second, vectorized tile axis, so a chunk sorts one key per token, not per element.

Values are evaluated after the sort, at permuted token indices, because permuting
an already-computed [BLOCK_T, BLOCK_D] tile in registers is far slower than loading
it in sorted order.
"""

import logging
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import sympy

import torch
from torch._dynamo.utils import counters
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from .. import config, ir
from ..codegen.triton import TritonSymbols
from ..ops_handler import DefaultHandler, WrapperHandler
from ..select_algorithm import (
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
from ..virtualized import V


log = logging.getLogger(__name__)

# Output dtype -> dtype the per-chunk segment sums accumulate in.
_ACC_DTYPE: dict[torch.dtype, torch.dtype] = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
    torch.int32: torch.int32,
}

# Ops that the template's subgraph codegen (ModificationWrapper) cannot express
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


def _record_skip(reason: str) -> None:
    counters["inductor"][f"scatter_preagg_skipped_{reason}"] += 1
    log.debug("scatter_preagg: skip reason=%s", reason)


def _has_native_bf16_atomics(device: torch.device) -> bool:
    if torch.version.hip:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        return arch in ("gfx942", "gfx950")
    return torch.cuda.get_device_capability(device) >= (9, 0)


def _ineligible(device: torch.device | None, dtype: torch.dtype) -> str | None:
    """Skip reason from the device, dtype and determinism alone, or None."""
    if device is None or device.type != "cuda":
        return "device"
    if dtype not in _ACC_DTYPE:
        return "dtype"
    if dtype == torch.bfloat16 and not _has_native_bf16_atomics(device):
        return "bf16_atomics"
    if config.deterministic or torch.are_deterministic_algorithms_enabled():
        return "deterministic"
    return None


def can_bypass_atomic_fallback(dtype: torch.dtype, device: torch.device | None) -> bool:
    """
    True when an accumulating scatter that Inductor would otherwise send to the
    eager fallback because of atomic_add dtype limitations may be offered to this
    template instead. The template sums each chunk in fp32 and issues one
    output-dtype atomic per destination per chunk, so it only needs the
    hardware to have native atomics for the output dtype (plain Triton bf16
    atomics are a CAS loop and serialize badly under contention).
    """
    return (
        config.scatter_preagg_enabled
        and dtype == torch.bfloat16
        and _ineligible(device, dtype) is None
    )


def _passes_size_gates(num_tokens: int, num_inner: int, num_keys: int) -> str | None:
    if config.scatter_preagg_force:
        return None
    if num_tokens * num_inner < config.scatter_preagg_min_index_size:
        return "too_small"
    if num_tokens / num_keys < config.scatter_preagg_min_contention_ratio:
        return "low_contention"
    return None


def claims_index_put(
    input_meta: dict[str, Any],
    index_meta: dict[str, Any],
    index_size: int,
    output_size: int,
    scatter_dim_size: int,
) -> bool:
    """
    Static approximation of the lowering-time gates for an FX index_put, so the
    partitioned scatter pass can leave the scatters this template will take.
    """
    if not config.scatter_preagg_enabled or scatter_dim_size <= 0:
        return False
    if any(
        isinstance(s, torch.SymInt)
        for s in (*input_meta["shape"], *index_meta["shape"])
    ):
        return False
    if _ineligible(input_meta["device"], input_meta["dtype"]) is not None:
        return False
    return (
        _passes_size_gates(
            index_size, output_size // scatter_dim_size, scatter_dim_size
        )
        is None
    )


class _Placeholder:
    """Value returned by the recording handler for every op."""


_PLACEHOLDER = _Placeholder()


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
        return _PLACEHOLDER

    def load(self, name: str, index: sympy.Expr) -> Any:
        self.read_names.add(name)
        self._note(index)
        return _PLACEHOLDER

    def indirect_indexing(self, x, size, check=True, wrap_neg=True) -> sympy.Symbol:
        self._num_indirect += 1
        return sympy.Symbol(
            f"preagg_indirect{self._num_indirect}", integer=True, nonnegative=True
        )


class _IndirectIndexHandler(WrapperHandler):
    """
    ModificationWrapper resolves indirect indices to the raw loaded value. This
    restores the semantics the regular Scatter codegen provides: negative-index
    wrapping, the bounds assert when requested, and (for destination keys) a
    validity mask so unchecked out-of-range writes are dropped instead of
    landing out of bounds.
    """

    def __init__(self, inner: Any, record_validity: bool) -> None:
        super().__init__(inner)
        self.record_validity = record_validity
        self.valid: list[Any] = []

    def indirect_indexing(self, x, size, check=True, wrap_neg=True):
        size_v = self.constant(int(size), torch.int64)
        zero = self.constant(0, torch.int64)
        if wrap_neg:
            x = self.where(self.lt(x, zero), self.add(x, size_v), x)
        if check or self.record_validity:
            in_bounds = self.and_(self.ge(x, zero), self.lt(x, size_v))
            if check:
                self.device_assert_async(
                    in_bounds, f"index out of bounds: 0 <= index < {int(size)}"
                )
            if self.record_validity:
                self.valid.append(in_bounds)
        return self._inner.indirect_indexing(x, size, False, False)

    def index_expr(self, expr, dtype):
        # The template has no range trees, so render the expression directly; its
        # only free symbols are template-local tensors (tokens, inner ids, indices).
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


def _unravel(linear: sympy.Expr, sizes: list[int]) -> list[sympy.Expr]:
    """Row-major coordinates of ``linear`` over ``sizes``."""
    coords: list[sympy.Expr] = []
    stride = 1
    for i in reversed(range(len(sizes))):
        size = sizes[i]
        if size == 1:
            coords.append(sympy.S.Zero)
        elif i == 0:
            coords.append(FloorDiv(linear, stride))
        else:
            coords.append(ModularIndexing(linear, stride, size))
        stride *= size
    return list(reversed(coords))


@dataclass
class _ScatterPlan:
    iter_sizes: list[int]
    tok_dims: list[int]
    inner_dims: list[int]
    # output dim fed by each inner iteration dim
    inner_out_dims: list[int]
    key_out_dims: list[int]
    key_sizes: list[int]
    out_sizes: list[int]
    out_indexer: Callable[[list[sympy.Expr]], sympy.Expr]
    output_indexer: Callable[[list[sympy.Expr]], list[sympy.Expr]]
    value_fn: Callable[[list[sympy.Expr]], Any]
    acc_dtype: torch.dtype

    @property
    def num_tokens(self) -> int:
        return int(sympy_product([self.iter_sizes[d] for d in self.tok_dims]))

    @property
    def num_inner(self) -> int:
        return int(sympy_product([self.iter_sizes[d] for d in self.inner_dims]))

    @property
    def num_keys(self) -> int:
        return int(sympy_product(self.key_sizes))

    def iteration_coords(
        self, token: sympy.Expr, inner: sympy.Expr
    ) -> list[sympy.Expr]:
        coords: list[sympy.Expr] = [sympy.S.Zero] * len(self.iter_sizes)
        tok_coords = _unravel(token, [self.iter_sizes[d] for d in self.tok_dims])
        inner_coords = _unravel(inner, [self.iter_sizes[d] for d in self.inner_dims])
        for d, c in zip(self.tok_dims, tok_coords):
            coords[d] = c
        for d, c in zip(self.inner_dims, inner_coords):
            coords[d] = c
        return coords

    def key_expr(self, out_coords: list[sympy.Expr]) -> sympy.Expr:
        key: sympy.Expr = sympy.S.Zero
        mult = 1
        for j, size in zip(reversed(self.key_out_dims), reversed(self.key_sizes)):
            key = key + out_coords[j] * mult
            mult *= size
        return key

    def address(self, key: sympy.Expr, inner: sympy.Expr) -> sympy.Expr:
        coords: list[sympy.Expr] = [sympy.S.Zero] * len(self.out_sizes)
        for j, c in zip(self.key_out_dims, _unravel(key, self.key_sizes)):
            coords[j] = c
        inner_coords = _unravel(inner, [self.iter_sizes[d] for d in self.inner_dims])
        for j, c in zip(self.inner_out_dims, inner_coords):
            coords[j] = c
        return self.out_indexer(coords)


def _static_ints(sizes: Any) -> list[int] | None:
    out = []
    for s in sizes:
        if isinstance(s, int):
            out.append(s)
        elif isinstance(s, sympy.Integer):
            out.append(int(s))
        else:
            return None
    return out


class ScatterPreaggTemplateKernel(TritonTemplateKernel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_extra_template_env_fns(
            self.preagg_def_kernel,
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
        # a placeholder. Autotune benchmarking still passes it, so take it as an
        # unused argument unless the scheduler has already removed it.
        out_name = self.output_node.get_name()
        if out_name not in V.graph.removed_buffers:
            self.args.output(out_name)
        return code

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

    def preagg_key(self, tok: str, key_out: str, valid_out: str) -> str:
        """Destination key and validity of each token in ``tok`` (shape [BLOCK_T])."""
        plan = self._plan

        def emit() -> list[str]:
            handler = V.get_ops_handler()
            tok_sym = self._symbol_of(handler.load("tok", sympy.S.Zero))
            coords_in = plan.iteration_coords(tok_sym, sympy.S.Zero)
            indirect = _IndirectIndexHandler(handler, record_validity=True)
            with V.set_ops_handler(indirect):
                out_coords = list(plan.output_indexer(coords_in))
            key = self.kexpr(self.rename_indexing(plan.key_expr(out_coords)))
            valid = " & ".join(
                ["tl.full([BLOCK_T], 1, tl.int1)"]
                + [f"({getattr(v, 'value', v)})" for v in indirect.valid]
            )
            return [
                f"{key_out} = tl.full([BLOCK_T], 0, tl.int64) + ({key})",
                f"{valid_out} = {valid}",
            ]

        return self._codegen_subgraph(
            "preagg_key", {"tok": tok}, {"tok": ("BLOCK_T",)}, emit
        )

    def preagg_value(self, ptok: str, inner: str, out: str) -> str:
        """Value at sorted tokens ``ptok`` ([BLOCK_T, 1]) and inner ids ([1, BLOCK_D])."""
        plan = self._plan

        def emit() -> list[str]:
            handler = V.get_ops_handler()
            tok_sym = self._symbol_of(handler.load("ptok", sympy.S.Zero))
            inner_sym = self._symbol_of(handler.load("inner", sympy.S.Zero))
            coords_in = plan.iteration_coords(tok_sym, inner_sym)
            indirect = _IndirectIndexHandler(handler, record_validity=False)
            with V.set_ops_handler(indirect):
                value = plan.value_fn(coords_in)
            value = getattr(value, "value", value)
            acc = triton_type(plan.acc_dtype)
            return [
                f"{out} = tl.broadcast_to(({value}).to({acc}), [BLOCK_T, BLOCK_D])",
            ]

        return self._codegen_subgraph(
            "preagg_value",
            {"ptok": ptok, "inner": inner},
            {"ptok": ("BLOCK_T", "1"), "inner": ("1", "BLOCK_D")},
            emit,
        )

    def preagg_addr(self, key: str, inner: str) -> str:
        """Offset into OUT of destination ``key`` and inner element ``inner``."""
        plan = self._plan
        key_sym = sympy.Symbol(key, integer=True, nonnegative=True)
        inner_sym = sympy.Symbol(inner, integer=True, nonnegative=True)
        return self.kexpr(self.rename_indexing(plan.address(key_sym, inner_sym)))


@SymbolicGridFn
def scatter_preagg_grid(num_tokens, num_inner, meta, *, cdiv):
    inner_per_prog = meta["BLOCK_D"] * meta["D_ITERS"]
    return (cdiv(num_tokens, meta["BLOCK_T"]), cdiv(num_inner, inner_per_prog), 1)


class ScatterPreaggTemplate(TritonTemplate):
    kernel_type = ScatterPreaggTemplateKernel


_TEMPLATE_SOURCE = r"""
{{preagg_def_kernel()}}
    pid_tok = tl.program_id(0).to(INDEX_DTYPE)
    pid_inner = tl.program_id(1).to(INDEX_DTYPE)
    pos = tl.arange(0, BLOCK_T).to(INDEX_DTYPE)
    # Subgraph loads are unmasked, so every lane must address a real token. The
    # last chunk is shifted back to end at NUM_TOKENS (lanes it shares with the
    # previous chunk are masked off) rather than clamped, which keeps the token
    # range contiguous for the compiler.
    if NUM_TOKENS >= BLOCK_T:
        tok_start = tl.minimum(pid_tok * BLOCK_T, NUM_TOKENS - BLOCK_T)
        tok = tok_start + pos
        tok_mask = tok >= pid_tok * BLOCK_T
    else:
        tok_start = 0
        tok = tl.minimum(pos, NUM_TOKENS - 1)
        tok_mask = pos < NUM_TOKENS

    {{preagg_key("tok", "key", "valid") | indent_except_first(1)}}

    dest = tl.where(tok_mask & valid, key, NUM_KEYS).to(KEY_DTYPE)
    packed = tl.sort((dest << LOG_BLOCK_T) | pos.to(KEY_DTYPE))
    skey = packed >> LOG_BLOCK_T
    sorted_pos = (packed & (BLOCK_T - 1)).to(INDEX_DTYPE)
    if NUM_TOKENS >= BLOCK_T:
        ptok = (tok_start + sorted_pos)[:, None]
    else:
        ptok = tl.minimum(sorted_pos, NUM_TOKENS - 1)[:, None]

    next_key = tl.gather(skey, tl.minimum(pos + 1, BLOCK_T - 1), 0)
    seg_end = (skey < NUM_KEYS) & ((skey != next_key) | (pos == BLOCK_T - 1))
    out_key = tl.where(seg_end, skey, 0).to(INDEX_DTYPE)[:, None]
    seg_key = tl.broadcast_to(skey[:, None], (BLOCK_T, BLOCK_D))
    if SKIP_UNIQUE_SCAN:
        # Every valid token that does not end a segment has a duplicate.
        has_dup = tl.max(((skey < NUM_KEYS) & ~seg_end).to(tl.int32), 0) > 0
    else:
        has_dup = True

    # The sort is amortized over D_ITERS inner blocks so BLOCK_T can stay
    # large (more pre-aggregation) while the register tile stays small.
    for it in tl.range(0, D_ITERS):
        inner_raw = (pid_inner * D_ITERS + it) * BLOCK_D + tl.arange(0, BLOCK_D)
        inner_mask = inner_raw < NUM_INNER
        # Clamping also hides contiguity, which disables vectorized loads.
        if NUM_INNER % (BLOCK_D * D_ITERS) == 0:
            inner = inner_raw[None, :]
            if ATOMIC_VEC:
                atomic_inner = tl.max_contiguous(
                    tl.multiple_of(inner_raw, ATOMIC_VEC), ATOMIC_VEC
                )[None, :]
            else:
                atomic_inner = inner
        else:
            inner = tl.minimum(inner_raw, NUM_INNER - 1)[None, :]
            atomic_inner = inner

        {{preagg_value("ptok", "inner", "val") | indent_except_first(2)}}

        if has_dup:
            _, acc = tl.associative_scan((seg_key, val), 0, _preagg_segmented_sum)
        else:
            acc = val
        tl.atomic_add(
            OUT + ({{preagg_addr("out_key", "atomic_inner")}}),
            acc.to(OUT_DTYPE),
            mask=seg_end[:, None] & inner_mask[None, :],
            sem="relaxed",
        )


@triton.jit
def _preagg_segmented_sum(key_a, val_a, key_b, val_b):
    return key_b, tl.where(key_a == key_b, val_a + val_b, val_b)
"""

scatter_preagg_template = ScatterPreaggTemplate(
    name="scatter_preagg",
    grid=scatter_preagg_grid,
    always_freeze_layout=True,
    source=_TEMPLATE_SOURCE,
)


def _analyze(
    out_buf: ir.Buffer,
    out_sizes: list[int],
    iter_sizes: list[int],
    value_fn: Callable[[list[sympy.Expr]], Any],
    output_indexer: Callable[[list[sympy.Expr]], list[sympy.Expr]],
    acc_dtype: torch.dtype,
) -> tuple[_ScatterPlan, OrderedSet[str]] | str:
    """Classify iteration dims into token/inner dims. Returns a skip reason on failure."""
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

    if key_rec.unsupported or value_rec.unsupported:
        log.debug(
            "scatter_preagg: unsupported ops key=%s value=%s",
            list(key_rec.unsupported),
            list(value_rec.unsupported),
        )
        return "unsupported_op"
    if len(out_coords) != len(out_sizes):
        return "rank_mismatch"

    key_deps = key_rec.symbols & OrderedSet(iter_syms)
    tok_dims: list[int] = []
    inner_dims: list[int] = []
    inner_out_dims: list[int] = []
    for d, sym in enumerate(iter_syms):
        if sym not in key_deps:
            uses = [j for j, c in enumerate(out_coords) if sym in c.free_symbols]
            if len(uses) == 1 and out_coords[uses[0]] == sym:
                inner_dims.append(d)
                inner_out_dims.append(uses[0])
                continue
        tok_dims.append(d)

    key_out_dims = [j for j in range(len(out_sizes)) if j not in inner_out_dims]
    plan = _ScatterPlan(
        iter_sizes=iter_sizes,
        tok_dims=tok_dims,
        inner_dims=inner_dims,
        inner_out_dims=inner_out_dims,
        key_out_dims=key_out_dims,
        key_sizes=[out_sizes[j] for j in key_out_dims],
        out_sizes=out_sizes,
        out_indexer=out_buf.get_layout().make_indexer(),
        output_indexer=output_indexer,
        value_fn=value_fn,
        acc_dtype=acc_dtype,
    )
    return plan, key_rec.read_names | value_rec.read_names


def maybe_preaggregated_scatter(
    out: ir.TensorBox,
    value_fn: Callable[[list[sympy.Expr]], Any],
    iter_sizes: Any,
    output_indexer: Callable[[list[sympy.Expr]], list[sympy.Expr]],
    inputs: list[Any],
    atomic_fallback: Callable[[], Any],
) -> bool:
    """
    Lower ``out[output_indexer(i)] += value_fn(i)`` over ``iter_sizes`` with the
    pre-aggregating template. ``inputs`` are the IR nodes value_fn and
    output_indexer read. If the template declines a scatter whose dtype reached
    the caller only through can_bypass_atomic_fallback, ``atomic_fallback()`` is
    emitted instead. Returns False (emitting nothing) when the caller should
    lower a regular atomic Scatter.
    """
    if not config.scatter_preagg_enabled:
        return False
    read_names: OrderedSet[str] = OrderedSet()
    for x in inputs:
        if isinstance(x, ir.IRNode):
            read_names |= x.get_read_names()
    if _try_template(out, value_fn, iter_sizes, output_indexer, read_names):
        return True
    if needs_fallback_due_to_atomic_add_limitations(out.get_dtype()):
        atomic_fallback()
        return True
    return False


def _try_template(
    out: ir.TensorBox,
    value_fn: Callable[[list[sympy.Expr]], Any],
    iter_sizes: Any,
    output_indexer: Callable[[list[sympy.Expr]], list[sympy.Expr]],
    read_names: OrderedSet[str],
) -> bool:
    device = out.get_device()
    dtype = out.get_dtype()
    reason = _ineligible(device, dtype)
    if reason is not None:
        _record_skip(reason)
        return False

    if not (
        isinstance(out, ir.TensorBox)
        and isinstance(out.data, ir.StorageBox)
        and isinstance(out.data.data, ir.Buffer)
    ):
        _record_skip("output_view")
        return False
    out_buf = out.data.data

    out_sizes = _static_ints(out.get_size())
    sizes = _static_ints(iter_sizes)
    if out_sizes is None or sizes is None:
        _record_skip("dynamic_shape")
        return False
    if 0 in out_sizes or 0 in sizes or len(out_sizes) == 0:
        _record_skip("empty")
        return False

    if out_buf.get_name() in read_names:
        _record_skip("reads_output")
        return False
    # Tracing the loaders below indexes any realized input directly, which
    # needs a fixed layout.
    for name in read_names:
        buf = V.graph.try_get_buffer(name)
        if buf is not None and isinstance(buf.get_layout(), ir.FlexibleLayout):
            buf.freeze_layout()
    if isinstance(out_buf.layout, ir.FlexibleLayout):
        out_buf.freeze_layout()

    analyzed = _analyze(
        out_buf, out_sizes, sizes, value_fn, output_indexer, _ACC_DTYPE[dtype]
    )
    if isinstance(analyzed, str):
        _record_skip(analyzed)
        return False
    plan, traced_reads = analyzed

    if out_buf.get_name() in traced_reads:
        _record_skip("reads_output")
        return False

    num_tokens, num_inner, num_keys = plan.num_tokens, plan.num_inner, plan.num_keys
    if num_keys >= 2**31:
        _record_skip("key_space")
        return False
    reason = _passes_size_gates(num_tokens, num_inner, num_keys)
    if reason is not None:
        _record_skip(reason)
        return False

    inputs: list[ir.IRNode] = []
    for name in traced_reads:
        buf = V.graph.try_get_buffer(name)
        if buf is None:
            _record_skip("unknown_input")
            return False
        layout = buf.get_layout() if hasattr(buf, "get_layout") else None
        if isinstance(layout, ir.FlexibleLayout):
            buf.freeze_layout()
        inputs.append(buf)

    inner_contiguous = (
        not plan.inner_out_dims
        or V.graph.sizevars.statically_known_equals(
            out_buf.get_layout().stride[plan.inner_out_dims[-1]], 1
        )
    )
    configs = V.choices.get_scatter_preagg_configs(
        num_tokens,
        num_inner,
        dtype,
        device_type=device.type,
        inner_contiguous=inner_contiguous,
    )
    # The sort packs (key, position) into one integer.
    key_bits = (num_keys + 1).bit_length()
    configs = [c for c in configs if key_bits + c.block_t.bit_length() - 1 <= 62]
    if not configs:
        _record_skip("key_space")
        return False

    # The template inlines value_fn and output_indexer through this plan; the
    # buffer wraps the value function so template index-width selection sees
    # its reads.
    value_subgraph = ir.ComputedBuffer(
        name=None,
        layout=ir.FlexibleLayout(device=device, dtype=dtype, size=list(sizes)),
        data=ir.Pointwise(
            device=device,
            dtype=dtype,
            inner_fn=value_fn,
            ranges=list(sizes),
        ),
    )
    value_subgraph.preagg_plan = plan  # type: ignore[attr-defined]

    layout = ir.FixedLayout(device=device, dtype=dtype, size=[1], stride=[1])
    # On CDNA, fp32 atomics run ~1.7x faster at two elements per thread than at
    # the four the compiler derives from load contiguity, except for wide fp32
    # rows into a handful of destinations.
    atomic_vec = 2 if torch.version.hip else 0
    # Skipping the scan for blocks with no repeated destination pays for its
    # check only on wide slices or 16-bit outputs (measured on gfx942).
    choices: list[Any] = []
    for cfg in configs:
        log_block_t = cfg.block_t.bit_length() - 1
        scatter_preagg_template.maybe_append_choice(
            choices,
            input_nodes=(out, *inputs),
            layout=layout,
            num_stages=cfg.num_stages,
            num_warps=cfg.num_warps,
            subgraphs=[value_subgraph],
            mutated_inputs=[out],
            call_sizes=[num_tokens, num_inner],
            BLOCK_T=cfg.block_t,
            BLOCK_D=cfg.block_d,
            D_ITERS=cfg.d_iters,
            LOG_BLOCK_T=log_block_t,
            NUM_TOKENS=num_tokens,
            NUM_INNER=num_inner,
            NUM_KEYS=num_keys,
            KEY_DTYPE="tl.int32" if key_bits + log_block_t <= 31 else "tl.int64",
            OUT_DTYPE=triton_type(dtype),
            ATOMIC_VEC=(
                0
                if cfg.block_d < atomic_vec
                or (cfg.block_d >= 64 and num_keys <= 64 and dtype.itemsize >= 4)
                else atomic_vec
            ),
            SKIP_UNIQUE_SCAN=dtype.itemsize < 4 or cfg.block_d >= 64,
        )
    if not choices:
        _record_skip("codegen_failed")
        return False
    # MultiTemplateBuffer (deferred selection) does not track mutated inputs, so
    # the in-place result would be dropped.
    autotune_select_algorithm(
        "scatter_preagg",
        choices,
        [out, *inputs],
        layout,
        return_multi_template=False,
    )

    counters["inductor"]["scatter_preagg_applied"] += 1
    log.debug(
        "scatter_preagg: APPLY out=%s tokens=%d inner=%d keys=%d configs=%s",
        out_buf.get_name(),
        num_tokens,
        num_inner,
        num_keys,
        configs,
    )
    return True

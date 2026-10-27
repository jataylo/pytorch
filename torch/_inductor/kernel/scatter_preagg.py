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
    """Skip reason from the device and dtype alone, or None."""
    if device is None or device.type != "cuda":
        return "device"
    if dtype not in _ACC_DTYPE:
        return "dtype"
    return None


def _deterministic() -> bool:
    return config.deterministic or torch.are_deterministic_algorithms_enabled()


def _needs_sorted(device: torch.device, dtype: torch.dtype) -> bool:
    """
    True when the scatter must use the atomic-free sorted lowering: under
    deterministic algorithms, and for bf16 without native atomics (plain Triton
    bf16 atomics are a CAS loop and serialize badly under contention).
    """
    return _deterministic() or (
        dtype == torch.bfloat16 and not _has_native_bf16_atomics(device)
    )


def can_bypass_atomic_fallback(dtype: torch.dtype, device: torch.device | None) -> bool:
    """
    True when an accumulating scatter that Inductor would otherwise send to the
    eager fallback, because of atomic_add dtype limitations or deterministic
    algorithms, may be offered to this template instead.
    """
    return (
        config.scatter_preagg_enabled
        and _ineligible(device, dtype) is None
        and (dtype == torch.bfloat16 or _deterministic())
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
    scatter_dim: int,
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
        for j, s in enumerate(input_meta["shape"])
        if j != scatter_dim
    ):
        return False
    device, dtype = input_meta["device"], input_meta["dtype"]
    if _ineligible(device, dtype) is not None:
        return False
    if _needs_sorted(device, dtype):
        # The sorted lowering has no contention gate but needs static sizes.
        return not any(
            isinstance(s, torch.SymInt)
            for s in (*input_meta["shape"], *index_meta["shape"])
        )
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


def _unravel(linear: sympy.Expr, sizes: list[sympy.Expr]) -> list[sympy.Expr]:
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
    iter_sizes: list[sympy.Expr]
    tok_dims: list[int]
    inner_dims: list[int]
    # output dim fed by each inner iteration dim
    inner_out_dims: list[int]
    key_out_dims: list[int]
    key_sizes: list[sympy.Expr]
    out_sizes: list[sympy.Expr]
    out_indexer: Callable[[list[sympy.Expr]], sympy.Expr]
    output_indexer: Callable[[list[sympy.Expr]], list[sympy.Expr]]
    value_fn: Callable[[list[sympy.Expr]], Any]
    acc_dtype: torch.dtype

    @property
    def num_tokens(self) -> sympy.Expr:
        return sympy_product([self.iter_sizes[d] for d in self.tok_dims])

    @property
    def num_inner(self) -> sympy.Expr:
        return sympy_product([self.iter_sizes[d] for d in self.inner_dims])

    @property
    def num_keys(self) -> sympy.Expr:
        return sympy_product(self.key_sizes)

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


def _hint(expr: sympy.Expr) -> int:
    return V.graph.sizevars.optimization_hint(expr)


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
        # a placeholder. Autotune benchmarking still passes it, so take it as an
        # unused argument unless the scheduler has already removed it.
        out_name = self.output_node.get_name()
        if out_name not in V.graph.removed_buffers:
            self.args.output(out_name)
        return code

    def preagg_sizes(self) -> str:
        """Kernel-argument definitions of the sizes that are not constexprs."""
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
    packed = tl.sort((dest << LOG_BLOCK_T) | pos.to(KEY_DTYPE))
    skey = packed >> LOG_BLOCK_T
    sorted_pos = (packed & (BLOCK_T - 1)).to(INDEX_DTYPE)
    if TOK_SHIFT:
        ptok = (tok_start + sorted_pos)[:, None]
    else:
        ptok = tl.minimum(tok_start + sorted_pos, NUM_TOKENS - 1)[:, None]

    next_key = tl.gather(skey, tl.minimum(pos + 1, BLOCK_T - 1), 0)
    seg_end = (skey < NUM_KEYS) & ((skey != next_key) | (pos == BLOCK_T - 1))
    out_key = tl.where(seg_end, skey, 0).to(INDEX_DTYPE)[:, None]
    # Keys are below the output's storage size, so INDEX_DTYPE holds them.
    seg_key = tl.broadcast_to(skey.to(INDEX_DTYPE)[:, None], (BLOCK_T, BLOCK_D))
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
        # The address drops out_key when there is a single destination.
        tl.atomic_add(
            tl.broadcast_to(
                OUT + ({{preagg_addr("out_key", "atomic_inner")}}), [BLOCK_T, BLOCK_D]
            ),
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

# The sorted (atomic-free, deterministic) lowering. Tokens arrive sorted by
# destination key (SKEYS, with PERM their original positions), so every
# destination is one contiguous segment. Each program sums the segments in its
# chunk of BLOCK_T sorted tokens in a fixed order. A segment that starts and
# ends in the chunk has no other writer and is added to OUT directly; the parts
# of segments that cross a chunk boundary go to PART (slot 0: the chunk's first
# segment, continued from the previous chunk; slot 1: its last segment,
# continued into the next) for the fixup kernel to add in chunk order.
_SORTED_PRELUDE = r"""
{{preagg_def_kernel()}}
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
        if NUM_INNER % (BLOCK_D * D_ITERS) == 0:
            inner = inner_raw[None, :]
        else:
            inner = tl.minimum(inner_raw, NUM_INNER - 1)[None, :]
"""

_SORTED_CHUNK_SOURCE = (
    _SORTED_PRELUDE
    + r"""
    spos = first + pos
    in_range = spos < NUM_TOKENS
    sp = tl.minimum(spos, NUM_TOKENS - 1)
    skey = tl.where(in_range, tl.load(SKEYS + sp), NUM_KEYS)
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


@triton.jit
def _preagg_segmented_sum(key_a, val_a, key_b, val_b):
    return key_b, tl.where(key_a == key_b, val_a + val_b, val_b)
"""
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
    + _SORTED_INNER_LOOP.replace("\n    ", "\n        ")
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
    name="scatter_sorted_chunk",
    grid=scatter_preagg_grid,
    always_freeze_layout=True,
    source=_SORTED_CHUNK_SOURCE,
)
scatter_sorted_fixup_template = ScatterPreaggTemplate(
    name="scatter_sorted_fixup",
    grid=scatter_preagg_grid,
    always_freeze_layout=True,
    source=_SORTED_FIXUP_SOURCE,
)


def _analyze(
    out_layout: ir.Layout,
    out_sizes: list[sympy.Expr],
    iter_sizes: list[sympy.Expr],
    value_fn: Callable[[list[sympy.Expr]], Any],
    output_indexer: Callable[[list[sympy.Expr]], list[sympy.Expr]],
    acc_dtype: torch.dtype,
) -> tuple[_ScatterPlan, OrderedSet[str], OrderedSet[str]] | str:
    """
    Classify iteration dims into token/inner dims. Returns the plan and the
    buffers the key and value functions read, or a skip reason on failure.
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

    if key_rec.unsupported or value_rec.unsupported:
        log.debug(
            "scatter_preagg: unsupported ops key=%s value=%s",
            list(key_rec.unsupported),
            list(value_rec.unsupported),
        )
        return "unsupported_key_op" if key_rec.unsupported else "unsupported_value_op"
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
        out_indexer=out_layout.make_indexer(),
        output_indexer=output_indexer,
        value_fn=value_fn,
        acc_dtype=acc_dtype,
    )
    return plan, key_rec.read_names, value_rec.read_names


def _materialize_values(
    value_fn: Callable[[list[sympy.Expr]], Any],
    sizes: list[Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Callable[[list[sympy.Expr]], Any]:
    """
    Compute the values into their own buffer (before the scatter mutates the
    output) and return a plain loader of it, for value functions the template
    cannot inline: ones that read the output or use unsupported ops.
    """
    from ..virtualized import ops

    values = ir.Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=lambda idx: ops.to_dtype(value_fn(idx), dtype),
        ranges=list(sizes),
    )
    values.realize()
    values.data.data.freeze_layout()  # type: ignore[union-attr]
    return values.make_loader()


def _lower_sorted(
    out: ir.TensorBox,
    plan: _ScatterPlan,
    value_subgraph: ir.ComputedBuffer,
    inputs: list[ir.IRNode],
    cfg: Any,
) -> bool:
    """
    Atomic-free lowering: sort the tokens by destination, then sum each
    destination's contiguous run in a fixed order (see _SORTED_CHUNK_SOURCE).
    """
    from ..lowering import empty_strided, lowerings
    from ..virtualized import ops

    if not (plan.num_tokens.is_number and plan.num_inner.is_number):
        _record_skip("sorted_dynamic")
        return False
    device, dtype = out.get_device(), out.get_dtype()
    num_tokens, num_inner = int(plan.num_tokens), int(plan.num_inner)

    def key_fn(idx: list[sympy.Expr]) -> Any:
        coords = plan.iteration_coords(idx[0], sympy.S.Zero)
        return ops.index_expr(plan.key_expr(plan.output_indexer(coords)), torch.int64)

    keys = ir.Pointwise.create(
        device=device, dtype=torch.int64, inner_fn=key_fn, ranges=[num_tokens]
    )
    sorted_keys, perm = lowerings[torch.ops.aten.sort.stable](keys, stable=True)
    sorted_keys.realize()
    perm.realize()
    num_chunks = -(-num_tokens // cfg.block_t)
    part = empty_strided(
        [2 * num_chunks * num_inner], [1], dtype=plan.acc_dtype, device=device
    )
    part.realize()
    nodes: list[ir.IRNode] = []
    for x in (part, sorted_keys, perm):
        ir.as_storage_and_layout(x, freeze=True)
        nodes.append(x)

    layout = ir.FixedLayout(device=device, dtype=dtype, size=[1], stride=[1])
    kwargs = {
        "BLOCK_T": cfg.block_t,
        "BLOCK_D": cfg.block_d,
        "D_ITERS": cfg.d_iters,
        "NUM_TOKENS": num_tokens,
        "NUM_INNER": num_inner,
        "NUM_KEYS": int(plan.num_keys),
        "NUM_CHUNKS": num_chunks,
        "OUT_DTYPE": triton_type(dtype),
    }
    stages = [
        ("scatter_sorted_chunk", scatter_sorted_chunk_template, [out, part], inputs),
        ("scatter_sorted_fixup", scatter_sorted_fixup_template, [out], []),
    ]
    selections = []
    for name, template, mutated, extra in stages:
        choices: list[Any] = []
        input_nodes = [out, *nodes, *extra]
        template.maybe_append_choice(
            choices,
            input_nodes=input_nodes,
            layout=layout,
            num_stages=cfg.num_stages,
            num_warps=cfg.num_warps,
            subgraphs=[value_subgraph],
            mutated_inputs=mutated,
            call_sizes=[plan.num_tokens, plan.num_inner],
            **kwargs,
        )
        if not choices:
            _record_skip("codegen_failed")
            return False
        selections.append((name, choices, input_nodes))
    for name, choices, input_nodes in selections:
        autotune_select_algorithm(
            name, choices, input_nodes, layout, return_multi_template=False
        )
    counters["inductor"]["scatter_preagg_sorted"] += 1
    return True


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
            # A realized input reports the reads of its producer, not itself.
            if (name := ir.try_get_name(x)) is not None:
                read_names.add(name)
    if _try_template(out, value_fn, iter_sizes, output_indexer, read_names):
        return True
    if needs_fallback_due_to_atomic_add_limitations(out.get_dtype()) or _deterministic():
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

    # A view output is scattered through its layout into the base buffer.
    try:
        view = out.data
        if isinstance(view, ir.BaseView):
            view = ir.ExternKernel.convert_to_reinterpret_view(view)
        out_storage, out_layout = ir.as_storage_and_layout(view, freeze=True)
    except NotImplementedError:
        out_storage = None
    if out_storage is None or not isinstance(out_storage.data, ir.Buffer):
        _record_skip("output_view")
        return False
    out_buf = out_storage.data
    out = ir.TensorBox(out_storage)

    out_sizes = [sympy.sympify(s) for s in out_layout.size]
    sizes = [sympy.sympify(s) for s in iter_sizes]
    if len(out_sizes) == 0 or 0 in [_hint(s) for s in (*out_sizes, *sizes)]:
        _record_skip("empty")
        return False

    # Tracing the loaders below indexes any realized input directly, which
    # needs a fixed layout.
    for name in read_names:
        buf = V.graph.try_get_buffer(name)
        if buf is not None and isinstance(buf.get_layout(), ir.FlexibleLayout):
            buf.freeze_layout()
    if isinstance(out_buf.layout, ir.FlexibleLayout):
        out_buf.freeze_layout()

    def analyze():
        return _analyze(
            out_layout, out_sizes, sizes, value_fn, output_indexer, _ACC_DTYPE[dtype]
        )

    analyzed = analyze()
    if analyzed == "unsupported_value_op" or (
        not isinstance(analyzed, str) and out_buf.get_name() in analyzed[2]
    ):
        value_fn = _materialize_values(value_fn, sizes, device, dtype)
        counters["inductor"]["scatter_preagg_materialized_values"] += 1
        analyzed = analyze()
    if isinstance(analyzed, str):
        _record_skip(analyzed)
        return False
    plan, key_reads, value_reads = analyzed
    traced_reads = key_reads | value_reads

    if out_buf.get_name() in traced_reads:
        _record_skip("reads_output")
        return False
    # A symbolic row size hides the alignment that vectorized loads and atomics
    # need; measured on gfx942 this loses to partitioned scatter and atomics.
    if not plan.num_inner.is_number:
        _record_skip("dynamic_inner")
        return False

    num_tokens, num_inner, num_keys = (
        _hint(plan.num_tokens),
        _hint(plan.num_inner),
        _hint(plan.num_keys),
    )
    # The sorted lowering replaces eager's fallback, not atomics, so the
    # contention gates do not apply to it.
    sorted_path = _needs_sorted(device, dtype)
    reason = None if sorted_path else _passes_size_gates(num_tokens, num_inner, num_keys)
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
            out_layout.stride[plan.inner_out_dims[-1]], 1
        )
    )
    # Deterministic results need the same config on every compile.
    with config.patch(max_autotune=config.max_autotune and not sorted_path):
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
    # Symbolic sizes are runtime arguments; the key width cannot follow the hint.
    static_sizes = {
        name: int(size)
        for name, size in (
            ("NUM_TOKENS", plan.num_tokens),
            ("NUM_INNER", plan.num_inner),
            ("NUM_KEYS", plan.num_keys),
        )
        if size.is_number
    }
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
    if sorted_path:
        if not _lower_sorted(out, plan, value_subgraph, inputs, configs[-1]):
            return False
        counters["inductor"]["scatter_preagg_applied"] += 1
        return True

    layout = ir.FixedLayout(device=device, dtype=dtype, size=[1], stride=[1])
    # On CDNA, fp32 atomics run ~1.7x faster at two elements per thread than at
    # the four the compiler derives from load contiguity, except for wide fp32
    # rows into a handful of destinations.
    atomic_vec = 2 if torch.version.hip else 0
    # Skipping the scan for blocks with no repeated destination pays for its
    # check only on wide slices or 16-bit outputs (measured on gfx942).
    sizevars = V.graph.sizevars
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
            call_sizes=[plan.num_tokens, plan.num_inner],
            BLOCK_T=cfg.block_t,
            BLOCK_D=cfg.block_d,
            D_ITERS=cfg.d_iters,
            LOG_BLOCK_T=log_block_t,
            KEY_DTYPE=(
                "tl.int32"
                if "NUM_KEYS" in static_sizes and key_bits + log_block_t <= 31
                else "tl.int64"
            ),
            OUT_DTYPE=triton_type(dtype),
            ATOMIC_VEC=(
                0
                if cfg.block_d < atomic_vec
                or (cfg.block_d >= 64 and num_keys <= 64 and dtype.itemsize >= 4)
                else atomic_vec
            ),
            SKIP_UNIQUE_SCAN=dtype.itemsize < 4 or cfg.block_d >= 64,
            **static_sizes,
            # A symbolic token count takes the shifted path under a guard when
            # its hint does; the clamped path handles any size.
            TOK_SHIFT=num_tokens >= cfg.block_t
            and sizevars.guard_or_false(sympy.Ge(plan.num_tokens, cfg.block_t)),
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

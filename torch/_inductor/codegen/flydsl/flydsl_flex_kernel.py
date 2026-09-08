# mypy: allow-untyped-defs
"""FlexAttention additions to the FlyDSL template kernel.

The FlyDSL analog of ``cutedsl_kernel.py``. The generic FlyDSL backend
(``flydsl_kernel.py``, ``flydsl_template.py``) cannot inline a user subgraph into a kernel
body, which is the one thing FlexAttention needs from it: a score_mod or mask_mod traced by
Dynamo and lowered by Inductor has to become FlyDSL source at the point in the attention
inner loop where the score sits in a register.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import hashlib
import math
from typing import Any, TYPE_CHECKING

import sympy

import torch

from ...ir import ComputedBuffer, InputBuffer
from ...virtualized import V
from ..common import CSE, IndentedBuffer, PythonPrinter, ValueRanges
from .flydsl_kernel import FlyDSLTemplateKernel
from .flydsl_op_overrides import (
    FlyDSLCSEVariable,
    FlyDSLOpOverrides,
    RUNTIME_ALIAS,
    RUNTIME_MODULE,
)
from .flydsl_template import FlyDSLTemplate


if TYPE_CHECKING:
    from ...ir import Buffer


class FlyDSLPrinter(PythonPrinter):
    """Prints sympy index expressions as FlyDSL-safe Python."""

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("ToFloat expects exactly one argument")
        return f"{RUNTIME_ALIAS}.to_f32({self.doprint(expr.args[0])})"

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        x, div = expr.args
        return f"({self.doprint(x)} // {self.doprint(div)})"


flydsl_pexpr = FlyDSLPrinter().doprint


@dataclasses.dataclass
class FlyDSLSubgraphInfo:
    """Kernel state swapped out while one subgraph is being rendered."""

    body: IndentedBuffer
    template_mask: str | None = None
    template_out: str | None = None
    cse: CSE[Any, str] | None = None

    def __post_init__(self):
        self.only_copy_if_non_none_fields = ("cse",)

    def to_dict(self):
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }


class FlyDSLFlexTemplateKernel(FlyDSLTemplateKernel):
    """A FlyDSL template kernel that can inline lowered score_mod / mask_mod subgraphs."""

    overrides = FlyDSLOpOverrides  # type: ignore[assignment]

    def __init__(
        self,
        kernel_name: str,
        input_nodes: list[Buffer],
        output_node: Buffer,
        subgraphs: list[Buffer] | None = None,
    ) -> None:
        super().__init__(
            kernel_name=kernel_name,
            input_nodes=input_nodes,
            output_node=output_node,
        )
        # Keyword-optional so the generic FlyDSLTemplate.generate(), which knows nothing
        # about subgraphs, can still construct this class.
        self.subgraphs = subgraphs
        self.subgraph_bodies: dict[str, FlyDSLSubgraphInfo] = {}

        self.body: IndentedBuffer = IndentedBuffer()
        self.template_mask: str | None = None
        self.template_out: str | None = None
        self.cse = CSE(name_prefix="tmp")

        self.named_input_nodes: dict[str, Buffer] = {}
        for i, input_node in enumerate(input_nodes):
            self.named_input_nodes[getattr(input_node, "name", f"input_{i}")] = (
                input_node
            )

        # See Note [FlyDSL aux slots]. Slots are handed out as the mods turn out to read
        # them, so a capture no mod touches never becomes a slot.
        self._aux_slots: dict[tuple[str, tuple[int, ...]], int] = {}
        self._aux_arg_names: list[str] = []
        self._aux_specs: list[list[int]] = []
        self._aux_numels: list[int] = []

    def resolve_extra_input(self, name: str) -> Buffer:
        """Look captures up in the graph's capture table before its buffers.

        ``realize_captures_for_cutedsl`` files a captured *view* under a synthetic input
        name, so for those names there is no graph buffer to find.
        """
        node = self._capture_node(name)
        return node if node is not None else V.graph.get_buffer(name)

    @staticmethod
    def _capture_node(name: str) -> Buffer | None:
        return getattr(V.graph, "_cutedsl_capture_nodes", {}).get(name)

    def get_tensor_buffers(self) -> list[str]:
        """Kernel argument names of the captures the mods read, in aux-slot order."""
        return list(self._aux_arg_names)

    def aux_slot(self, name: str, arg_name: str, spec: list[int], numel: int) -> int:
        """Index of one captured-tensor *reading* in the ``aux`` list the mods are handed.

        Note [FlyDSL aux slots]:
        score_mod and mask_mod receive the same ``aux`` list, so slots are numbered across
        the whole kernel rather than per subgraph: the same read from both mods resolves to
        one slot, not two. The order is an ABI -- slot i of ``aux``, of ``aux_specs()`` and
        of ``get_tensor_buffers()`` are the same tensor -- but every side of it is derived
        from this one render, so they cannot disagree.

        A slot is a *(tensor, stride spec)* pair rather than a tensor, so one capture read
        with two index patterns takes two slots. Document masking is the case that needs
        this: it reads its document-id table once by query row and once by key column, and
        a reader carries one stride per coordinate, so those two readings cannot share a
        spec. They can share the tensor -- both slots name the same kernel argument, and
        the template converts it once.
        """
        key = (name, tuple(spec))
        slot = self._aux_slots.get(key)
        if slot is None:
            slot = len(self._aux_arg_names)
            self._aux_slots[key] = slot
            self._aux_arg_names.append(arg_name)
            self._aux_specs.append(list(spec))
            self._aux_numels.append(numel)
        return slot

    def aux_spec_for_read(
        self, name: str, dim_indices, dim_sizes, coord_positions
    ) -> tuple[list[int], int, dict[int, str]]:
        """Resolve one read into a stride spec, a numel, and any computed indices.

        The spec is ``(stride_b, stride_h, stride_q, stride_kv)``: each axis of the capture
        is matched to the coordinate that indexes it, and that axis's element stride is
        filed under that coordinate. ``coord_positions`` maps the coordinate names *as they
        appear in the generated body* to their spec position; the caller supplies it
        because the subgraph's placeholders (``m``, ``n``) and the template's variables
        (``q_idx``, ``kv_idx``) differ.

        An axis may also be indexed by a *value* rather than a coordinate, which is what
        ``offsets[document_id[q_idx]]`` is. The reader has no separate gather entry point,
        but it does not need one: it sums ``stride * argument`` over four argument
        positions, so a value passed in a position whose stride is the axis stride is
        exactly that gather. Such an axis therefore claims a spare position, and the
        returned dict says which position carries which expression. Positions are claimed
        lowest-first, which keeps ``stride_kv`` free -- the vectorised reader treats
        ``stride_kv == 1`` as a promise that ``kv`` is contiguous, and a gather is not.

        Anything the reader cannot express is rejected here rather than silently reading
        the wrong element, which is this addressing scheme's failure mode.
        """
        sizes = [V.graph.sizevars.guard_int(s) for s in dim_sizes]
        # Strides of the contiguous f32 copy the template passes, which is what the kernel
        # will actually read, not the buffer's own layout.
        strides = [1] * len(sizes)
        for i in range(len(sizes) - 2, -1, -1):
            strides[i] = strides[i + 1] * sizes[i + 1]

        spec = [0, 0, 0, 0]
        claimed: set[int] = set()
        computed: list[tuple[int, sympy.Expr]] = []
        for axis_index, index_expr in enumerate(dim_indices):
            expr = self.rename_indexing(index_expr)
            if expr == sympy.Integer(0):
                continue
            position = (
                coord_positions.get(str(expr))
                if isinstance(expr, sympy.Symbol)
                else None
            )
            if position is None:
                # Not a coordinate, so it is a value the body computed. Held back until
                # the coordinates have claimed their positions, since it can take any
                # position they leave.
                computed.append((axis_index, expr))
                continue
            if position in claimed:
                raise NotImplementedError(
                    f"captured tensor {name} indexes two axes with {expr!r}; the aux "
                    "reader has one stride per coordinate"
                )
            claimed.add(position)
            spec[position] = strides[axis_index]

        overrides: dict[int, str] = {}
        for axis_index, expr in computed:
            position = next((p for p in range(4) if p not in claimed), None)
            if position is None:
                raise NotImplementedError(
                    f"captured tensor {name} is indexed with {expr!r} on axis "
                    f"{axis_index}, but its four reader positions are already taken by "
                    f"coordinates ({', '.join(coord_positions)}); a computed index needs "
                    "one of its own"
                )
            claimed.add(position)
            spec[position] = strides[axis_index]
            # Through the index printer, not str(): the expression reaches the body as
            # source, and only the printer is guaranteed to render one that parses.
            overrides[position] = self.kexpr(expr)

        # The kernel bounds its buffer descriptor with this, so the padding lanes of the
        # score tile read zero instead of off the end of the tensor. See Note [aux reads
        # run past the logical extent] in flex_flash_generic.py.
        return spec, math.prod(sizes), overrides

    def aux_specs(self) -> list[list[int]]:
        """Stride spec per aux slot, in slot order. Call after the mods are rendered."""
        return [list(spec) for spec in self._aux_specs]

    def aux_numels(self) -> list[int]:
        """Element count per aux slot, in slot order, for the kernel's buffer bounds."""
        return list(self._aux_numels)

    def create_cse_var(self, *args, **kwargs):
        return FlyDSLCSEVariable(*args, **kwargs)

    def kexpr(self, expr: sympy.Expr) -> str:
        return flydsl_pexpr(expr)

    def gen_imports(self) -> str:
        """The generic imports plus what a rendered mod body needs."""
        imports = IndentedBuffer()
        imports.splice(super().gen_imports())
        imports.splice(
            f"""
            from flydsl.expr import math as fmath
            from flydsl.expr.utils.arith import ArithValue
            import {RUNTIME_MODULE} as {RUNTIME_ALIAS}
            """
        )
        return imports.getvalue()

    def mod_key(self, *extra: Any) -> str:
        """Cache-discriminating key over every mod body rendered so far, plus ``extra``.

        FlyDSL's JIT cache keys on traced source plus recursively collected *scalar closure
        values*. A constant a mod reads from anywhere else -- a module global, one of the
        constexprs ``gen_defines()`` emits -- is invisible to that key, so two generated
        mods differing only in such a constant collide and the second silently reuses the
        first one's binary. The rendered bodies are the right thing to hash because they
        are what FlyDSL traces; ``extra`` covers build parameters that change the kernel
        without changing those bodies, the vectorization width being the one that does.

        Call from the template *after* the mods are defined.
        """
        bodies = [
            f"{name}\x00{info.body.getvalue()}"
            for name, info in sorted(self.subgraph_bodies.items())
        ]
        if not bodies:
            raise AssertionError(
                "mod_key() was called before any modification() was rendered, so it "
                "would not discriminate between mods"
            )
        payload = "\x00".join(bodies + [str(e) for e in extra])
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def render(self, template, **kwargs):
        """Render with the flex hooks added.

        Duplicates the shape of ``FlyDSLTemplateKernel.render`` because the base class
        builds its ``template_env`` inline; there is no seam to extend.
        """
        from ...select_algorithm import PartialRender

        template_env = {
            "def_kernel": self.def_kernel,
            "gen_defines": lambda: self.gen_defines(**kwargs),
            "get_output": self.get_output,
            "modification": self.modification,
            "mod_key": self.mod_key,
            "aux_specs": self.aux_specs,
            "aux_numels": self.aux_numels,
            "get_tensor_buffers": self.get_tensor_buffers,
        }
        rendered_code = template.render(
            kernel_name=self.kernel_name,
            input_nodes=self.input_nodes,
            output_node=self.output_node,
            **template_env,
            **kwargs,
        )
        return PartialRender(self.gen_imports() + rendered_code, self.render_hooks)

    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str):
        if not all(
            hasattr(self, field.name)
            for field in dataclasses.fields(FlyDSLSubgraphInfo)
        ):
            raise AssertionError(
                "expected all FlyDSLSubgraphInfo fields to be set on self"
            )
        old_state = {
            key.name: getattr(self, key.name)
            for key in dataclasses.fields(FlyDSLSubgraphInfo)
        }

        if body_name not in self.subgraph_bodies:
            self.subgraph_bodies[body_name] = FlyDSLSubgraphInfo(body=IndentedBuffer())

        subgraph = self.subgraph_bodies[body_name]
        for key, value in subgraph.to_dict().items():
            if value is None and key in getattr(
                subgraph, "only_copy_if_non_none_fields", ()
            ):
                continue
            setattr(self, key, value)

        try:
            yield
        finally:
            self.subgraph_bodies[body_name] = FlyDSLSubgraphInfo(
                **{
                    key.name: getattr(self, key.name)
                    for key in dataclasses.fields(FlyDSLSubgraphInfo)
                }
            )
            for key, value in old_state.items():
                setattr(self, key, value)

    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str, *, clear_cse: bool = False):
        if body_name in self.subgraph_bodies:
            raise AssertionError(f"Subgraph body '{body_name}' already exists")
        self.subgraph_bodies[body_name] = FlyDSLSubgraphInfo(
            body=IndentedBuffer(),
            cse=self.cse.clone() if clear_cse else None,
        )
        with self.set_subgraph_body(body_name):
            yield

    def _get_subgraph(self, subgraph_number: int):
        if self.subgraphs is None:
            raise AssertionError("Expected subgraphs to be set")
        if not 0 <= subgraph_number < len(self.subgraphs):
            raise AssertionError(
                f"Invalid subgraph number {subgraph_number}, must be < "
                f"{len(self.subgraphs)}"
            )
        if self.body.getvalue() != "":
            raise AssertionError("Body should be clear before adding a modification")
        return self.subgraphs[subgraph_number]

    def modification(
        self,
        subgraph_number: int,
        output_name: str | None,
        mask: str | None = None,
        **fixed_inputs,
    ) -> str:
        """Generate FlyDSL code for a subgraph modification.

        ``fixed_inputs`` maps subgraph placeholder names to the template's own variable
        names (``m`` -> ``q_idx``, and so on), so the emitted body reads the surrounding
        kernel's values directly instead of taking arguments.
        """
        # Find a unique name to avoid collisions between multiple modifications of the
        # same subgraph.
        num = 0
        while f"mod_{subgraph_number}_{num}" in self.subgraph_bodies:
            num += 1

        with self.create_subgraph_body(f"mod_{subgraph_number}_{num}", clear_cse=True):
            subgraph = self._get_subgraph(subgraph_number)
            handler = ModificationWrapperFlyDSL(
                self, subgraph_number, fixed_inputs, mask
            )
            with V.set_kernel_handler(self), V.set_ops_handler(handler):
                if isinstance(subgraph, list):
                    raise NotImplementedError(
                        "Scatter graphs are not supported for FlyDSL (backward only)"
                    )
                if not isinstance(subgraph, ComputedBuffer):
                    raise AssertionError(
                        f"Expected ComputedBuffer, got {type(subgraph)}"
                    )
                if isinstance(subgraph.data, InputBuffer):
                    # grad_score_mod can be an InputBuffer
                    out = subgraph.data.make_loader()(())
                else:
                    # Inline a pointwise lowering into the template
                    out = subgraph.data.inner_fn(())

            if output_name is None:
                raise NotImplementedError(
                    "Side-effect only modifications are not supported for FlyDSL"
                )
            if out is None:
                raise AssertionError(
                    f"Expected a computation result for named output {output_name}"
                )
            self.body.writeline(f"{output_name} = {out}")

            return self.body.getvalue()


class ModificationWrapperFlyDSL(V.WrapperHandler):  # type: ignore[name-defined]
    """Resolves subgraph placeholders and captured-tensor reads during lowering.

    Sits between Inductor's IR and FlyDSL codegen: ops go through
    :class:`FlyDSLOpOverrides`, placeholder loads resolve to the template's variable
    names, and everything else is a read from a captured tensor.
    """

    def __init__(
        self,
        kernel: FlyDSLFlexTemplateKernel,
        subgraph_number: int,
        fixed_inputs: dict[str, Any],
        mask: str | None,
    ):
        super().__init__(FlyDSLOpOverrides())
        self.name = f"FlyDSLPlaceholderSubstitution_{subgraph_number}"
        self.kernel = kernel
        self.fixed_inputs = dict(fixed_inputs)
        self.mask = mask

    def _get_input_dtype(self, name: str) -> torch.dtype:
        if name in self.kernel.named_input_nodes:
            return self.kernel.named_input_nodes[name].dtype
        return torch.int32 if name in ("b", "h", "m", "n") else torch.float32

    def _add_kernel_input(self, name: str, spec: list[int], numel: int) -> str:
        """Register a captured tensor as a kernel input and resolve its aux reader.

        The mod body only ever names a slot, never the kernel argument, so this can run
        part way through the render even though the mods are emitted above the kernel.
        """
        arg_name = self.kernel.args.input(name)
        return f"aux[{self.kernel.aux_slot(name, arg_name, spec, numel)}]"

    def load(self, name: str, index: sympy.Expr):
        """Read a subgraph input: either a template value or a captured tensor."""
        if name in self.fixed_inputs:
            value = self.fixed_inputs[name]
            dtype = self._get_input_dtype(name)
            result = self.kernel.cse.generate(
                self.kernel.body, value, bounds=ValueRanges.unknown(), dtype=dtype
            )
            if (
                isinstance(result, FlyDSLCSEVariable)
                and isinstance(value, str)
                and dtype in (torch.int32, torch.int64)
            ):
                from ...utils import sympy_index_symbol

                result.index_expr = sympy_index_symbol(value)
            return result

        return self._load_captured_tensor(name, index)

    def _load_captured_tensor(self, name: str, index: sympy.Expr):
        """Emit a read from a captured tensor, and record its stride spec.

        An aux slot is a reader callable as ``reader(b, h, q_idx, kv_idx)`` that resolves
        one element offset from four per-axis strides, which is why the index has to arrive
        as a coordinate tuple -- what ``HierarchicalIndex`` carries.

        The reader is always handed the true four coordinates; which of them matter is
        expressed in the stride spec, where 0 means "broadcast over this axis". So the spec
        is not derivable from the capture's shape alone -- an ``[H]`` table is
        ``(0, 1, 0, 0)`` and an ``[S]`` table of the same length is ``(0, 0, 1, 0)`` -- it
        depends on which coordinate the mod indexed each axis with, which is known only
        here, so the spec is recorded now and read back through :meth:`aux_specs`.
        """
        from ...kernel.flex.flex_flash_attention import HierarchicalIndex

        buffer = self.kernel.named_input_nodes.get(
            name
        ) or self.kernel.resolve_extra_input(name)
        var_dtype = buffer.get_dtype()

        dim_indices = index.args if isinstance(index, HierarchicalIndex) else (index,)
        dim_sizes = buffer.get_size()
        if len(dim_sizes) == 0:
            dim_indices = ()
        if len(dim_indices) != len(dim_sizes):
            raise AssertionError(
                f"captured tensor {name} has rank {len(dim_sizes)} but was indexed with "
                f"{len(dim_indices)} coordinates"
            )

        # The subgraph names the coordinates b/h/m/n; the template binds them to its own
        # variables. Both the spec and the emitted call have to speak the latter.
        coord_names = [self.fixed_inputs[axis] for axis in ("b", "h", "m", "n")]
        # Resolved before the slot is claimed, because the spec is half of a slot's
        # identity: the same tensor read two ways needs two of them.
        spec, numel, overrides = self.kernel.aux_spec_for_read(
            name,
            dim_indices,
            dim_sizes,
            {coord: position for position, coord in enumerate(coord_names)},
        )
        var = self._add_kernel_input(name, spec, numel)

        reader_args = list(coord_names)
        for position, index_expr in overrides.items():
            # aux is f32, so a value read out of one capture arrives as a float even when
            # it is a document id; the reader multiplies it by a stride and expects an int.
            reader_args[position] = f"{RUNTIME_ALIAS}.to_i32({index_expr})"
        expr = f"{var}({', '.join(reader_args)})"

        # Aux is f32 on the kernel side; anything narrower is widened on read so the mod
        # body only ever sees f32.
        if var_dtype in (torch.float16, torch.bfloat16):
            expr = f"{RUNTIME_ALIAS}.to_f32({expr})"
            var_dtype = torch.float32

        return self.kernel.cse.generate(
            self.kernel.body,
            expr,
            dtype=var_dtype,
            bounds=ValueRanges.unknown(),
            shape=(1,),
        )

    def store(self, name, index, value, mode=None):
        raise NotImplementedError("Stores are not supported in FlyDSL flex mods")

    def indirect_indexing(self, index_var, size, check=True, wrap_neg=True):
        """Turn a device value back into something usable as a tensor index.

        This is what makes ``bias[b, h, q_idx, kv_idx]`` or ``slopes[h]`` work: Inductor
        sees those coordinates as *loaded values*, so using one as an index round-trips
        through here into a sympy symbol named after the variable holding it, which the
        index printer then renders as that same name.
        """
        from ...utils import sympy_index_symbol

        if isinstance(index_var, (int, sympy.Integer)):
            if wrap_neg and index_var < 0:
                return V.graph.sizevars.simplify(sympy.Integer(index_var) + size)
            return sympy.Integer(index_var)

        index_expr = getattr(index_var, "index_expr", None)
        if (
            wrap_neg
            and index_expr is not None
            and not V.graph.sizevars.statically_known_geq(index_expr, 0)
        ):
            # Python indexing admits negatives and nothing here proves this one is not,
            # so fold the wrap into the value before it becomes an index.
            wrapped = self.kernel.cse.newvar(dtype=index_var.dtype)
            size_expr = f"{RUNTIME_ALIAS}.const_i32({self.kernel.kexpr(size)})"
            self.kernel.body.writeline(
                f"{wrapped} = {RUNTIME_ALIAS}.where("
                f"{RUNTIME_ALIAS}.lt({index_var}, {RUNTIME_ALIAS}.const_i32(0)), "
                f"{RUNTIME_ALIAS}.add({index_var}, {size_expr}), {index_var})"
            )
            return sympy_index_symbol(str(wrapped))

        # Prefer the name the value was loaded from ("h", "q_idx"): it reads better in the
        # generated body than the temporary that holds it, and both are in scope. Only when
        # that name is a bare identifier, though -- a CSE key is the *expression* that
        # produced the value, and for a captured-tensor read that is a whole reader call,
        # which an index may be used in several places. Returning it there would re-emit
        # the load at each use instead of reading the temporary once.
        for expr, var in self.kernel.cse._cache.items():
            if var is index_var and isinstance(expr, str) and expr.isidentifier():
                return sympy_index_symbol(expr)
        return sympy_index_symbol(str(index_var))


class FlyDSLFlexTemplate(FlyDSLTemplate):
    """A FlyDSL template whose kernel can inline score_mod / mask_mod subgraphs."""

    kernel_type: type[Any] = FlyDSLFlexTemplateKernel

    def generate(self, **kwargs: Any) -> Any:
        """Same as the base, but routes ``subgraphs`` to the kernel constructor.

        The base ``generate()`` predates subgraphs and builds its kernel in two places:
        once now, to benchmark, and once inside ``make_kernel_render`` for the deferred
        render at scheduling time. Both resolve ``self.kernel_type`` when they run, so
        patching this instance would miss the deferred one; binding the subgraphs onto a
        shallow copy covers both without mutating the template registered under this name,
        which is shared across lowerings.

        Keeping ``subgraphs`` out of ``kwargs`` also keeps it out of ``gen_defines()``,
        which would otherwise emit the buffer list into the generated source as a
        constexpr, and out of the autotune choice description.
        """
        subgraphs = kwargs.pop("subgraphs", None)
        bound = copy.copy(self)
        bound.kernel_type = functools.partial(self.kernel_type, subgraphs=subgraphs)
        return FlyDSLTemplate.generate(bound, **kwargs)

    def render_kernel(
        self,
        kernel_name: str,
        input_nodes: list[Any],
        output_node: Any,
        subgraphs: list[Any] | None = None,
        **kwargs: Any,
    ) -> tuple[FlyDSLFlexTemplateKernel, str]:
        """Render this template to FlyDSL source, outside the autotune path.

        Returns the kernel alongside the source so callers can read back what the lowering
        collected, notably ``kernel.get_tensor_buffers()``.
        """
        kernel = self.kernel_type(
            kernel_name=kernel_name,
            input_nodes=input_nodes,
            output_node=output_node,
            subgraphs=subgraphs,
        )
        partial = kernel.render(self.template, **kwargs)
        return kernel, partial.finalize_all()

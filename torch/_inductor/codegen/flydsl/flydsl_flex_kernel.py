# mypy: allow-untyped-defs
"""FlexAttention additions to the FlyDSL template kernel.

The generic FlyDSL backend (``flydsl_kernel.py`` / ``flydsl_template.py``) knows how to
define a kernel, benchmark it, schedule it and compile it. What it does not have -- and
does not need for a GEMM -- is a way to inline a *user subgraph* into the kernel body.
That is what FlexAttention requires: a `score_mod` or `mask_mod` written in Python, traced
by Dynamo and lowered by Inductor, has to end up as FlyDSL device-value source at the
exact point in the attention inner loop where each score sits in a register.

So everything here is the flex-specific delta on top of the generic backend:

- :class:`FlyDSLFlexTemplateKernel` adds subgraph bodies, a CSE scope and
  :meth:`~FlyDSLFlexTemplateKernel.modification`, which renders one lowered subgraph.
- :class:`ModificationWrapperFlyDSL` resolves subgraph placeholders against the
  template's own variables and turns captured-tensor reads into aux-slot reads.
- :class:`FlyDSLFlexTemplate` binds the two together.

The emitted source targets the vocabulary the hand-written mods in
``flex_kernels/flex_mods.py`` use, because those were written as the contract this
lowering has to hit.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import hashlib
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
        # Keyword-optional so the generic FlyDSLTemplate.generate(), which does not know
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

        # Aux slots are handed out as the mods turn out to read them, so a capture no mod
        # touches never becomes a slot and never gets converted at call time. The order is
        # still an ABI -- slot i of the `aux` list, entry i of `aux_specs`, and entry i of
        # `get_tensor_buffers()` are the same tensor -- but every side of it is derived
        # from this one render, so they cannot disagree.
        self._aux_slots: dict[str, int] = {}
        self._aux_arg_names: list[str] = []
        self._aux_specs: list[list[int]] = []

    # ── captured-tensor bookkeeping ──────────────────────────────────────────

    def resolve_extra_input(self, name: str) -> Buffer:
        """Look captures up in the graph's capture table before its buffers.

        ``realize_captures_for_cutedsl`` gives a captured *view* a synthetic input name
        and files the real ``ReinterpretView`` under it, so for those names there is no
        graph buffer to find.
        """
        node = self._capture_node(name)
        return node if node is not None else V.graph.get_buffer(name)

    @staticmethod
    def _capture_node(name: str) -> Buffer | None:
        return getattr(V.graph, "_cutedsl_capture_nodes", {}).get(name)

    def get_tensor_buffers(self) -> list[str]:
        """Kernel argument names of the captures the mods read, in aux-slot order."""
        return list(self._aux_arg_names)

    def aux_slot(self, name: str, arg_name: str) -> int:
        """Index of a captured tensor in the ``aux`` list the mods are handed.

        score_mod and mask_mod receive the same ``aux`` list, so slots are numbered across
        the whole kernel rather than per subgraph: a tensor read by both mods resolves to
        one slot, not two.
        """
        slot = self._aux_slots.get(name)
        if slot is None:
            slot = len(self._aux_arg_names)
            self._aux_slots[name] = slot
            self._aux_arg_names.append(arg_name)
            self._aux_specs.append([0, 0, 0, 0])
        return slot

    def record_aux_spec(self, name: str, dim_indices, dim_sizes, coord_positions) -> None:
        """Record ``(stride_b, stride_h, stride_q, stride_kv)`` for one capture.

        Each axis of the capture is matched to the coordinate that indexes it, and that
        axis's element stride is filed under that coordinate. ``coord_positions`` maps the
        coordinate *variable names as they appear in the generated body* to their position
        in the spec; the caller supplies it because the subgraph's placeholder names
        (``m``, ``n``) and the template's variable names (``q_idx``, ``kv_idx``) differ.

        Anything the reader cannot express -- an offset index, an axis indexed by
        arithmetic on a coordinate, two axes indexed by the same coordinate -- is rejected
        here rather than silently reading the wrong element, which is the failure mode
        this addressing scheme has.
        """
        slot = self._aux_slots[name]
        sizes = [V.graph.sizevars.guard_int(s) for s in dim_sizes]
        # Strides of the contiguous f32 copy the template passes, which is what the
        # kernel will actually read, not the buffer's own layout.
        strides = [1] * len(sizes)
        for i in range(len(sizes) - 2, -1, -1):
            strides[i] = strides[i + 1] * sizes[i + 1]

        spec = [0, 0, 0, 0]
        for axis_index, index_expr in enumerate(dim_indices):
            expr = self.rename_indexing(index_expr)
            if expr == sympy.Integer(0):
                # A constant-0 index contributes nothing to the offset.
                continue
            position = (
                coord_positions.get(str(expr))
                if isinstance(expr, sympy.Symbol)
                else None
            )
            if position is None:
                raise NotImplementedError(
                    f"captured tensor {name} is indexed with {expr!r} on axis "
                    f"{axis_index}; the aux reader resolves one element offset from "
                    "per-axis strides, so each axis must be indexed by a bare "
                    f"coordinate ({', '.join(coord_positions)})"
                )
            if spec[position] != 0:
                raise NotImplementedError(
                    f"captured tensor {name} indexes two axes with {expr!r}; the aux "
                    "reader has one stride per coordinate"
                )
            spec[position] = strides[axis_index]

        existing = self._aux_specs[slot]
        if any(existing) and existing != spec:
            raise NotImplementedError(
                f"captured tensor {name} is read with two different index patterns "
                f"({existing} and {spec}); it occupies one aux slot with one stride spec"
            )
        self._aux_specs[slot] = spec

    def aux_specs(self) -> list[list[int]]:
        """Stride spec per aux slot, in slot order.

        Call from the template after the mods are rendered; before that there are no slots.
        """
        return [list(spec) for spec in self._aux_specs]

    # ── codegen plumbing ─────────────────────────────────────────────────────

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

        FlyDSL's JIT cache keys on traced source plus recursively collected *scalar
        closure values*. A constant a mod reads from anywhere else -- a module global, one
        of the constexprs ``gen_defines()`` emits -- is invisible to that key, so two
        generated mods differing only in such a constant collide and the second silently
        reuses the first one's binary. Wrong numbers, no error.

        The rendered subgraph bodies are the right thing to hash because they are what
        FlyDSL will trace. ``extra`` is for build parameters that change the kernel
        without changing those bodies, the vectorization width being the one that does.

        Call this from the template *after* the mods are defined; it sees only what has
        been rendered by then.
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

        This duplicates the shape of ``FlyDSLTemplateKernel.render`` because the base
        class builds its ``template_env`` inline; there is no seam to extend. Worth
        factoring out upstream if the flex template lands.
        """
        from ...select_algorithm import PartialRender

        template_env = {
            "def_kernel": self.def_kernel,
            "gen_defines": lambda: self.gen_defines(**kwargs),
            "get_output": self.get_output,
            "modification": self.modification,
            "mod_key": self.mod_key,
            "aux_specs": self.aux_specs,
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

    # ── subgraph bodies ──────────────────────────────────────────────────────

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
        """Inline a lowered subgraph as FlyDSL source.

        ``fixed_inputs`` maps subgraph placeholder names to the template's own variable
        names (``score`` -> ``score``, ``m`` -> ``q_idx``, and so on), so the emitted
        body reads the surrounding kernel's values directly instead of taking arguments.
        Returns the body text, which the template splices in at the call site.
        """
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
                    raise AssertionError(f"Expected ComputedBuffer, got {type(subgraph)}")
                if isinstance(subgraph.data, InputBuffer):
                    out = subgraph.data.make_loader()(())
                else:
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

    def _add_kernel_input(self, name: str) -> str:
        """Register a captured tensor as a kernel input and resolve its aux reader.

        The mod body only ever names a slot, never the kernel argument, so this can run
        part way through the render even though the mods are emitted above the kernel.
        """
        arg_name = self.kernel.args.input(name)
        return f"aux[{self.kernel.aux_slot(name, arg_name)}]"

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
        one i32 element offset from four per-axis strides -- which is why it wants
        coordinates rather than a flattened offset, and why the index has to arrive as a
        coordinate tuple. That is what ``HierarchicalIndex`` carries.

        The reader is always handed the true four coordinates; which of them matter is
        expressed entirely in the stride spec, where 0 means "broadcast over this axis".
        So the spec is not derivable from the capture's shape alone -- an ``[H]`` table is
        ``(0, 1, 0, 0)`` and an ``[S]`` table of the same length would be ``(0, 0, 1, 0)``
        -- it depends on which coordinate the mod indexed each axis with. That is known
        only here, while the subgraph is being lowered, so the spec is recorded now and
        the template reads it back through :meth:`aux_specs`.
        """
        from ...kernel.flex.flex_flash_attention import HierarchicalIndex

        var = self._add_kernel_input(name)
        buffer = self.kernel.named_input_nodes.get(name) or self.kernel.resolve_extra_input(
            name
        )
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
        self.kernel.record_aux_spec(
            name,
            dim_indices,
            dim_sizes,
            {coord: position for position, coord in enumerate(coord_names)},
        )

        expr = f"{var}({', '.join(coord_names)})"

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

        This is what makes a capture indexed by the mod's own coordinates work --
        ``bias[b, h, q_idx, kv_idx]`` or ``slopes[h]``. Inductor sees those coordinates as
        *loaded values*, so using one as an index has to round-trip through here: we hand
        back a sympy symbol named after the variable holding it, which the index printer
        then renders as that same name in the generated body.
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
        # generated body than the temporary that holds it, and both are in scope.
        for expr, var in self.kernel.cse._cache.items():
            if var is index_var and isinstance(expr, str):
                return sympy_index_symbol(expr)
        return sympy_index_symbol(str(index_var))


class FlyDSLFlexTemplate(FlyDSLTemplate):
    """A FlyDSL template whose kernel can inline score_mod / mask_mod subgraphs."""

    kernel_type: type[Any] = FlyDSLFlexTemplateKernel

    def generate(self, **kwargs: Any) -> Any:
        """Same as the base, but routes ``subgraphs`` to the kernel constructor.

        The base ``generate()`` predates subgraphs and builds its kernel with
        ``(kernel_name, input_nodes, output_node)``, in two places: once now, to
        benchmark, and once inside ``make_kernel_render`` for the deferred render at
        scheduling time. Both resolve ``self.kernel_type`` when they run, so temporarily
        patching this instance would miss the deferred one. Binding the subgraphs onto a
        shallow copy covers both without mutating the template registered under this
        name, which is shared across lowerings.

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

        Returns the kernel alongside the source so callers can read back what the
        lowering collected -- notably ``kernel.get_tensor_buffers()``, which fixes the
        order of the ``aux`` tuple the kernel has to be called with.
        """
        kernel = self.kernel_type(
            kernel_name=kernel_name,
            input_nodes=input_nodes,
            output_node=output_node,
            subgraphs=subgraphs,
        )
        partial = kernel.render(self.template, **kwargs)
        return kernel, partial.finalize_all()


__all__ = [
    "FlyDSLFlexTemplate",
    "FlyDSLFlexTemplateKernel",
    "FlyDSLSubgraphInfo",
    "ModificationWrapperFlyDSL",
    "flydsl_pexpr",
]

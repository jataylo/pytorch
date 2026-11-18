from __future__ import annotations

import ast
import dataclasses

from torch.utils._ordered_set import OrderedSet

from ..runtime.hints import AMDConfigLimits, DeviceProperties


@dataclasses.dataclass
class AMDCodegenSource:
    """Share one parsed source across guarded Inductor codegen passes."""

    source: str
    tree: ast.Module = dataclasses.field(init=False)
    changed: bool = False

    def __post_init__(self) -> None:
        self.tree = ast.parse(self.source)

    def emit(self) -> str:
        """emit() -> str

        Keep unmatched source byte-for-byte unchanged.
        """
        if not self.changed:
            return self.source
        ast.fix_missing_locations(self.tree)
        return ast.unparse(self.tree) + "\n"


def _finish_source(
    tree: ast.Module, source: str, context: AMDCodegenSource | None
) -> str:
    if context is not None:
        context.tree = tree
        context.changed = True
        return source
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and ast.unparse(node.func) == name


def supports_amd_kernel_optimizations(device: DeviceProperties) -> bool:
    return (
        device.type == "hip"
        and isinstance(device.cc, str)
        and device.cc.startswith("gfx")
        and isinstance(device.warp_size, int)
        and (device.warp_size in (32, 64))
        and isinstance(device.max_threads_per_block, int)
        and (device.max_threads_per_block >= device.warp_size)
        and (device.max_threads_per_block % device.warp_size == 0)
    )


def _source_limits(meta: ast.Dict) -> AMDConfigLimits | None:
    if not isinstance(meta, ast.Dict):
        return None
    values = {
        ast.literal_eval(key): value for key, value in zip(meta.keys, meta.values)
    }
    device = values.get("device")
    if device is None:
        return AMDConfigLimits(64, 16, 32768)
    if (
        not isinstance(device, ast.Call)
        or ast.unparse(device.func) != "DeviceProperties"
        or device.args
    ):
        return None
    try:
        properties = {
            keyword.arg: ast.literal_eval(keyword.value) for keyword in device.keywords
        }
        target = DeviceProperties(**properties)
    except (TypeError, ValueError):
        return None
    return (
        AMDConfigLimits.from_device(target)
        if supports_amd_kernel_optimizations(target)
        else None
    )


def _native_metadata(meta: ast.Dict, family: str, **values) -> None:
    values = {"amd_native_family": family, **values}
    for key, value in values.items():
        index = next(
            (i for i, name in enumerate(meta.keys) if ast.literal_eval(name) == key),
            None,
        )
        if index is None:
            meta.keys.append(ast.Constant(key))
            meta.values.append(ast.Constant(value))
        else:
            meta.values[index] = ast.Constant(value)


def _row_tail_safe(fn: ast.FunctionDef, assignments: dict[str, ast.AST]) -> bool:
    def protected(node, seen=()):
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Lt)
        ):
            return (
                ast.unparse(node.left) == "xindex"
                and ast.unparse(node.comparators[0]) == "xnumel"
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            return protected(node.left, seen) or protected(node.right, seen)
        if (
            isinstance(node, ast.Name)
            and node.id in assignments
            and (node.id not in seen)
        ):
            return protected(assignments[node.id], (*seen, node.id))
        return False

    assigned = OrderedSet()
    for statement in fn.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
            return False
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
            if isinstance(statement, ast.AnnAssign)
            else []
        )
        for target in targets:
            for node in ast.walk(target):
                if isinstance(node, ast.Name):
                    if node.id in assigned:
                        return False
                    assigned.add(node.id)
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if (
                    node.func.id not in ("float", "int", "bool")
                    or not all(isinstance(arg, ast.Constant) for arg in node.args)
                    or node.keywords
                ):
                    return False
            if isinstance(node, ast.Attribute):
                root = node.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                module = isinstance(root, ast.Name) and root.id in (
                    "tl",
                    "tl_math",
                    "libdevice",
                    "triton_helpers",
                )
                if not module and node.attr not in ("to", "dtype"):
                    return False
            if isinstance(node, ast.Subscript):
                slices = (
                    node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else [node.slice]
                )
                if any(
                    not (
                        isinstance(item, ast.Slice)
                        and item.lower is None
                        and (item.upper is None)
                        and (item.step is None)
                    )
                    and (not (isinstance(item, ast.Constant) and item.value is None))
                    for item in slices
                ):
                    return False
    xindex = assignments.get("xindex")
    if xindex is None:
        return False
    expression = ast.unparse(xindex)
    offset = assignments.get("xoffset")
    if offset is not None:
        expression = expression.replace("xoffset", ast.unparse(offset))
    if expression != "tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]":
        return False
    memory = [
        node
        for node in ast.walk(fn)
        if _call(node, "tl.load") or _call(node, "tl.store")
    ]
    if not memory:
        return False
    definitions = dict(assignments)
    definitions.update(
        {
            node.target.id: node.value
            for node in fn.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and (node.value is not None)
        }
    )
    arguments = OrderedSet([arg.arg for arg in fn.args.args])

    def row_dependent(node, seen=()):
        if isinstance(node, ast.Name):
            if node.id in ("xindex", "xoffset", "XBLOCK") or node.id in seen:
                return True
            if node.id in definitions:
                return row_dependent(definitions[node.id], (*seen, node.id))
            return node.id not in arguments and node.id not in (
                "tl",
                "tl_math",
                "libdevice",
                "triton_helpers",
            )
        if _call(node, "tl.program_id"):
            return True
        return any(row_dependent(child, seen) for child in ast.iter_child_nodes(node))

    for call in memory:
        if _call(call, "tl.load") and (not row_dependent(call.args[0])):
            continue
        position = 1 if _call(call, "tl.load") else 2
        mask = (
            call.args[position]
            if len(call.args) > position
            else next((kw.value for kw in call.keywords if kw.arg == "mask"), None)
        )
        if mask is None or not protected(mask):
            return False
    elementwise = OrderedSet(
        [
            "tl.load",
            "tl.store",
            "tl.arange",
            "tl.program_id",
            "tl.full",
            "tl.broadcast_to",
            "tl.where",
            "tl.abs",
            "tl.exp",
            "tl.exp2",
            "tl.log",
            "tl.log2",
            "tl.sqrt",
            "tl.rsqrt",
            "tl.maximum",
            "tl.minimum",
            "tl.div_rn",
            "tl.sum",
            "tl.max",
            "tl.min",
            "tl.reduce",
            "tl.associative_scan",
        ]
    )
    for statement in fn.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and (node.func.value.id == "tl")
                    and (name not in elementwise)
                ):
                    return False
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and ast.unparse(node.func) in (
            "tl.sum",
            "tl.max",
            "tl.min",
            "tl.reduce",
            "tl.associative_scan",
            "triton_helpers.fmax2",
            "triton_helpers.max2",
            "triton_helpers.any",
        ):
            axis = (
                node.args[1]
                if len(node.args) > 1
                else next((kw.value for kw in node.keywords if kw.arg == "axis"), None)
            )
            if not isinstance(axis, ast.Constant) or axis.value != 1:
                return False
    return True


def optimize_row_source(
    source: str, *, _context: AMDCodegenSource | None = None
) -> str:
    """Keep short row reductions and their statistics in one wave layout."""
    tree = _context.tree if _context is not None else ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    fn = next(
        (
            node
            for node in functions
            if any(
                (
                    _call(decorator, "triton_heuristics.persistent_reduction")
                    for decorator in node.decorator_list
                )
            )
        ),
        None,
    )
    helpers = [node for node in functions if node is not fn]
    if fn is None:
        return source
    for helper in helpers:
        if len(helper.args.args) != 2 or len(helper.body) != 2:
            return source
        assignment, result = helper.body
        if not isinstance(assignment, ast.Assign) or not isinstance(result, ast.Return):
            return source
        if len(assignment.targets) != 1 or not isinstance(
            assignment.targets[0], ast.Name
        ):
            return source
        left, right = helper.args.args
        expected = ast.parse(f"{left.arg} + {right.arg}", mode="eval").body
        if ast.dump(assignment.value) != ast.dump(expected):
            return source
        if (
            not isinstance(result.value, ast.Name)
            or result.value.id != assignment.targets[0].id
        ):
            return source
        if (
            helper.args.posonlyargs
            or helper.args.kwonlyargs
            or helper.args.vararg
            or helper.args.kwarg
            or helper.args.defaults
        ):
            return source
    decorators = [
        node
        for node in fn.decorator_list
        if _call(node, "triton_heuristics.persistent_reduction")
    ]
    if len(decorators) != 1:
        return source
    keywords = {keyword.arg: keyword.value for keyword in decorators[0].keywords}
    hints = ast.literal_eval(keywords["size_hints"])
    if OrderedSet(hints) != OrderedSet(["x", "r0_"]) or not 32 <= hints["r0_"] <= 32768:
        return source
    triton_meta = keywords["triton_meta"]
    limits = _source_limits(triton_meta)
    if limits is None or hints["r0_"] > limits.max_tile:
        return source
    metadata = {
        ast.literal_eval(key): value
        for key, value in zip(triton_meta.keys, triton_meta.values)
    }
    signature = ast.literal_eval(metadata["signature"])
    if list(signature)[-3:] != ["xnumel", "r0_numel", "XBLOCK"]:
        return source
    if any(
        dtype not in ("*bf16", "*fp16", "*fp32", "*i1", "*i32", "*i64")
        for dtype in signature.values()
        if dtype.startswith("*")
    ):
        return source
    calls = OrderedSet(
        [
            ast.unparse(node.func)
            for statement in fn.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ]
    )
    supported = OrderedSet(
        ["triton_helpers.fmax2", "triton_helpers.max2", "triton_helpers.any"]
    )
    if any(
        name.startswith("triton_helpers.") and name not in supported for name in calls
    ):
        return source
    if any(
        name.startswith("tl.")
        and name.split(".")[1]
        in (
            "atomic_add",
            "atomic_max",
            "atomic_min",
            "dot",
            "sort",
            "inline_asm_elementwise",
            "load_tensor_descriptor",
        )
        for name in calls
    ):
        return source
    scan = "tl.associative_scan" in calls
    if scan:
        input_dtype = next(
            (dtype for name, dtype in signature.items() if name.startswith("in_ptr")),
            None,
        )
        if not helpers or hints["r0_"] != 32768 or input_dtype != "*bf16":
            return source
        helper_names = OrderedSet([helper.name for helper in helpers])
        for node in ast.walk(fn):
            if _call(node, "tl.associative_scan"):
                if (
                    len(node.args) != 3
                    or not isinstance(node.args[2], ast.Name)
                    or node.args[2].id not in helper_names
                ):
                    return source
    if any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(fn)):
        return source
    assignments = {
        node.targets[0].id: node.value
        for node in fn.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    names = [
        node.targets[0].id
        for node in fn.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ]
    if len(names) != len(OrderedSet(names)):
        return source
    xnumel = assignments.get("xnumel")
    dynamic_rows = xnumel is None and any(arg.arg == "xnumel" for arg in fn.args.args)
    if not dynamic_rows and (
        not isinstance(xnumel, ast.Constant)
        or not isinstance(xnumel.value, int)
        or xnumel.value <= 0
    ):
        return source
    row_tail_safe = _row_tail_safe(fn, assignments)
    if dynamic_rows and (not row_tail_safe):
        return source
    import triton.language as target_language
    from triton.language.extra import libdevice as target_libdevice

    if any(
        name.startswith("tl.")
        and name[3:] != "broadcast_to"
        and (not hasattr(target_language, name[3:]))
        for name in calls
    ):
        return source
    if any(
        name.startswith("libdevice.")
        and (not hasattr(target_libdevice, name.split(".")[1]))
        for name in calls
    ):
        return source
    for node in ast.walk(fn):
        if _call(node, "tl.full"):
            shape = ast.unparse(node.args[0])
            if shape not in (
                "[XBLOCK]",
                "[R0_BLOCK]",
                "[XBLOCK, R0_BLOCK]",
                "[1, 1]",
                "[1]",
            ):
                return source
        if _call(node, "tl.arange") and ast.unparse(node.args[1]) not in (
            "XBLOCK",
            "R0_BLOCK",
        ):
            return source
    _native_metadata(
        keywords["inductor_meta"],
        "row_reduction",
        amd_reduction_extent=hints["r0_"],
        amd_row_count=None if dynamic_rows else xnumel.value,
        amd_row_tail_safe=row_tail_safe,
        amd_cache_policy="balanced",
    )
    if scan:
        _native_metadata(
            keywords["inductor_meta"],
            "row_reduction",
            amd_graph_tuning=True,
            amd_confirm_tuning=True,
        )
    return _finish_source(tree, source, _context)

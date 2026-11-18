from __future__ import annotations

import ast
import copy
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


def optimize_tiled_transpose_source(
    source: str, *, _context: AMDCodegenSource | None = None
) -> str:
    """Add native tiled alternatives to proven static transpose copies.

    Zero tile dimensions retain the flat copy with normalized tail masks. Positive
    tile dimensions traverse a batched input matrix while preserving trailing
    contiguous elements in the output permutation and the existing flat grid.
    """
    tree = _context.tree if _context is not None else ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        return source
    fn = functions[0]
    decorators = [
        node for node in fn.decorator_list if _call(node, "triton_heuristics.pointwise")
    ]
    if len(decorators) != 1 or ast.unparse(fn.decorator_list[-1]) != "triton.jit":
        return source
    keywords = {keyword.arg: keyword.value for keyword in decorators[0].keywords}
    meta = keywords.get("triton_meta")
    if not isinstance(meta, ast.Dict):
        return source
    limits = _source_limits(meta)
    if limits is None:
        return source
    values = {
        ast.literal_eval(key): value for key, value in zip(meta.keys, meta.values)
    }
    signature_node = values.get("signature")
    if not isinstance(signature_node, ast.Dict):
        return source
    signature = ast.literal_eval(signature_node)
    two_dimensional = list(signature) == [
        "in_ptr0",
        "out_ptr0",
        "ynumel",
        "xnumel",
        "YBLOCK",
        "XBLOCK",
    ]
    if not two_dimensional and list(signature) != [
        "in_ptr0",
        "out_ptr0",
        "xnumel",
        "XBLOCK",
    ]:
        return source
    if (
        signature["in_ptr0"] not in ("*bf16", "*fp16", "*fp32")
        or signature["out_ptr0"] != signature["in_ptr0"]
    ):
        return source
    inductor_meta = keywords.get("inductor_meta")
    if not isinstance(inductor_meta, ast.Dict):
        return source
    metadata = {
        ast.literal_eval(key): value
        for key, value in zip(inductor_meta.keys, inductor_meta.values)
    }
    if ast.literal_eval(
        metadata.get("mutated_arg_names", ast.List(elts=[], ctx=ast.Load()))
    ):
        return source
    assignments = {}
    stores = []
    loads = []
    for statement in fn.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            name = statement.targets[0].id
            if name in assignments or name in (
                "in_ptr0",
                "out_ptr0",
                "XBLOCK",
                "YBLOCK",
            ):
                return source
            assignments[name] = statement.value
        elif isinstance(statement, ast.Expr) and _call(statement.value, "tl.store"):
            stores.append(statement.value)
        else:
            return source
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            cast = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "to"
                and (len(node.args) == 1)
                and (not node.keywords)
                and (ast.unparse(node.args[0]) == "tl.float32")
            )
            if ast.unparse(node.func) not in (
                "tl.load",
                "tl.store",
                "tl.arange",
                "tl.program_id",
                "tl.full",
            ) and (not cast):
                return source
            if _call(node, "tl.load"):
                loads.append(node)
            if _call(node, "tl.store") and (
                not (isinstance(statement, ast.Expr) and statement.value is node)
            ):
                return source
    if len(loads) != 1 or len(stores) != 1:
        return source
    load, store = (loads[0], stores[0])
    n = assignments.get("xnumel")
    y = assignments.get("ynumel") if two_dimensional else ast.Constant(1)
    if (
        not isinstance(n, ast.Constant)
        or type(n.value) is not int
        or (not isinstance(y, ast.Constant))
        or (type(y.value) is not int)
        or (n.value <= 0)
        or (y.value <= 0)
        or (n.value * y.value >= 2**31 - limits.max_tile)
    ):
        return source

    def matches(node, expression):
        return ast.dump(node) == ast.dump(ast.parse(expression, mode="eval").body)

    required = {
        "xoffset": "tl.program_id(0) * XBLOCK",
        "xindex": "xoffset + tl.arange(0, XBLOCK)[:]",
    }
    if two_dimensional:
        required = {
            "xoffset": "tl.program_id(0) * XBLOCK",
            "xindex": "xoffset + tl.arange(0, XBLOCK)[None, :]",
            "yoffset": "tl.program_id(1) * YBLOCK",
            "yindex": "yoffset + tl.arange(0, YBLOCK)[:, None]",
            "xmask": "xindex < xnumel",
            "ymask": "yindex < ynumel",
            "x1": "xindex",
            "y0": "yindex",
        }
    if any(
        (
            not matches(assignments.get(name, ast.Constant(None)), expression)
            for name, expression in required.items()
        )
    ):
        return source
    if ast.unparse(assignments.get("xmask", ast.Constant(None))) not in (
        "xindex < xnumel",
        "tl.full([XBLOCK], True, tl.int1)[:]",
    ):
        return source
    if (
        len(store.args) < 2
        or not isinstance(store.args[0], ast.BinOp)
        or (not isinstance(store.args[0].op, ast.Add))
    ):
        return source
    address = store.args[0]
    if not load.args:
        return source
    if two_dimensional:
        if not matches(load.args[0], f"in_ptr0 + (y0 + {y.value} * x1)") or not matches(
            address, f"out_ptr0 + (x1 + {n.value} * y0)"
        ):
            return source
    else:
        if ast.unparse(address.left) != "out_ptr0" or not isinstance(
            address.right, ast.Name
        ):
            return source
        if address.right.id != "xindex" and (
            not matches(assignments.get(address.right.id, ast.Constant(None)), "xindex")
        ):
            return source
    masks = (
        ("xmask & ymask", "ymask & xmask", "None")
        if two_dimensional
        else ("xmask", "None")
    )
    for call, position in ((load, 1), (store, 2)):
        if len(call.args) > position and ast.unparse(call.args[position]) not in masks:
            return source
        if any(
            keyword.arg == "mask" and ast.unparse(keyword.value) not in masks
            for keyword in call.keywords
        ):
            return source

    def constant_mod(name):
        node = assignments.get(name)
        if (
            not isinstance(node, ast.BinOp)
            or not isinstance(node.op, ast.Mod)
            or (not isinstance(node.right, ast.Constant))
        ):
            return None
        return (
            node.right.value
            if type(node.right.value) is int and node.right.value > 0
            else None
        )

    shape = (1, n.value, y.value, 1) if two_dimensional else None
    c = None if two_dimensional else constant_mod("x0")
    if c is not None and matches(assignments["x0"].left, "xindex"):
        if matches(assignments.get("x1", ast.Constant(None)), f"xindex // {c}"):
            columns = n.value // c
            if n.value % c == 0 and matches(
                load.args[0], f"in_ptr0 + (x1 + {columns} * x0)"
            ):
                shape = (1, c, columns, 1)
            mods = {
                node.left.id: node.right.value
                for node in ast.walk(load.args[0])
                if isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Mod)
                and isinstance(node.left, ast.Name)
                and isinstance(node.right, ast.Constant)
            }
            if OrderedSet(mods) == OrderedSet(["x0", "x1"]):
                d, columns = (mods["x0"], mods["x1"])
                if (
                    all(type(value) is int and value > 0 for value in (d, columns))
                    and c % d == 0
                    and (n.value % (c * columns) == 0)
                ):
                    offset = f"{d} * (x1 % {columns}) + {columns * d} * (x0 // {d}) + {columns * c} * (x1 // {columns}) + x0 % {d}"
                    if matches(load.args[0], f"in_ptr0 + ({offset})"):
                        shape = (n.value // (c * columns), c // d, columns, d)
        columns = constant_mod("x1")
        if columns is not None and n.value % (c * columns) == 0:
            indices = {
                "x1": f"(xindex // {c}) % {columns}",
                "x2": f"xindex // {c * columns}",
            }
            offset = f"x1 + {columns} * x0 + {c * columns} * x2"
            if all(
                (
                    matches(assignments.get(name, ast.Constant(None)), value)
                    for name, value in indices.items()
                )
            ) and matches(load.args[0], f"in_ptr0 + ({offset})"):
                shape = (n.value // (c * columns), c, columns, 1)
    d, rows, columns = (
        (None, None, None)
        if two_dimensional
        else (constant_mod(name) for name in ("x0", "x1", "x2"))
    )
    if all(value is not None for value in (d, rows, columns)):
        c = d * rows
        indices = {
            "x0": f"xindex % {d}",
            "x1": f"(xindex // {d}) % {rows}",
            "x2": f"(xindex // {c}) % {columns}",
            "x3": f"xindex // {c * columns}",
        }
        offset = f"x0 + {d} * x2 + {columns * d} * x1 + {columns * c} * x3"
        if (
            n.value % (c * columns) == 0
            and all(
                (
                    matches(assignments.get(name, ast.Constant(None)), value)
                    for name, value in indices.items()
                )
            )
            and matches(load.args[0], f"in_ptr0 + ({offset})")
        ):
            shape = (n.value // (c * columns), rows, columns, d)
    if shape is None:
        return source

    def copy_value(node, active):
        if node is load:
            return ast.Name(id="_amd_value", ctx=ast.Load())
        if (
            isinstance(node, ast.Name)
            and node.id in assignments
            and (node.id not in active)
        ):
            return copy_value(assignments[node.id], active | OrderedSet([node.id]))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and (node.func.attr == "to")
        ):
            value = copy_value(node.func.value, active)
            if value is not None:
                result = copy.deepcopy(node)
                result.func.value = value
                return result
        return None

    value = copy_value(store.args[1], OrderedSet())
    if value is None:
        return source

    def normalize_mask(call, position, mask):
        if len(call.args) > position:
            call.args[position] = ast.parse(mask, mode="eval").body
        else:
            keyword = next(
                (keyword for keyword in call.keywords if keyword.arg == "mask"), None
            )
            if keyword is None:
                call.keywords.append(
                    ast.keyword(arg="mask", value=ast.parse(mask, mode="eval").body)
                )
            else:
                keyword.value = ast.parse(mask, mode="eval").body

    flat_body = copy.deepcopy(fn.body)
    for node in ast.walk(ast.Module(body=flat_body, type_ignores=[])):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and (node.targets[0].id == "xmask")
        ):
            node.value = ast.parse(f"xindex < {n.value}", mode="eval").body
        elif (
            two_dimensional
            and isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and (node.targets[0].id == "ymask")
        ):
            node.value = ast.parse(f"yindex < {y.value}", mode="eval").body
        elif _call(node, "tl.load"):
            normalize_mask(node, 1, "xmask & ymask" if two_dimensional else "xmask")
        elif _call(node, "tl.store"):
            normalize_mask(node, 2, "xmask & ymask" if two_dimensional else "xmask")
    batch, rows, columns, inner = shape
    first = (
        "tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)"
        if two_dimensional
        else "tl.program_id(0)"
    )
    step = (
        "tl.num_programs(0) * tl.num_programs(1)"
        if two_dimensional
        else "tl.num_programs(0)"
    )
    tiled = ast.parse(
        f"\n_amd_tiles_m: tl.constexpr = triton.cdiv({rows}, AMD_TILE_M)\n_amd_tiles_n: tl.constexpr = triton.cdiv({columns * inner}, AMD_TILE_N)\nfor _amd_tile in tl.range({first}, {batch} * _amd_tiles_m * _amd_tiles_n, {step}):\n    _amd_batch = _amd_tile // (_amd_tiles_m * _amd_tiles_n)\n    _amd_row = (_amd_tile // _amd_tiles_n % _amd_tiles_m) * AMD_TILE_M + tl.arange(0, AMD_TILE_M)[:, None]\n    _amd_col = (_amd_tile % _amd_tiles_n) * AMD_TILE_N + tl.arange(0, AMD_TILE_N)[None, :]\n    _amd_mask = (_amd_row < {rows}) & (_amd_col < {columns * inner})\n    pass\n"
    ).body
    tiled_load = copy.deepcopy(load)
    tiled_load.args[0] = ast.parse(
        f"in_ptr0 + _amd_batch * {rows * columns * inner} + _amd_row * {columns * inner} + _amd_col",
        mode="eval",
    ).body
    normalize_mask(tiled_load, 1, "_amd_mask")
    tiled_store = copy.deepcopy(store)
    tiled_store.args[0] = ast.parse(
        f"out_ptr0 + _amd_batch * {rows * columns * inner} + (_amd_col // {inner}) * {rows * inner} + _amd_row * {inner} + _amd_col % {inner}",
        mode="eval",
    ).body
    tiled_store.args[1] = value
    normalize_mask(tiled_store, 2, "_amd_mask")
    tiled[-1].body[-1:] = [
        ast.Assign(
            targets=[ast.Name(id="_amd_value", ctx=ast.Store())], value=tiled_load
        ),
        ast.Expr(value=tiled_store),
    ]
    block = "XBLOCK * YBLOCK" if two_dimensional else "XBLOCK"
    input_linear = ast.parse(
        f"\n_amd_linear = ({first}) * ({block}) + tl.arange(0, {block})\n_amd_linear_mask = _amd_linear < {batch * rows * columns * inner}\n_amd_output = (_amd_linear // {rows * columns * inner}) * {rows * columns * inner} + ((_amd_linear // {inner}) % {columns}) * {rows * inner} + ((_amd_linear // {columns * inner}) % {rows}) * {inner} + _amd_linear % {inner}\n"
    ).body
    linear_load = copy.deepcopy(load)
    linear_load.args[0] = ast.parse("in_ptr0 + _amd_linear", mode="eval").body
    normalize_mask(linear_load, 1, "_amd_linear_mask")
    linear_store = copy.deepcopy(store)
    linear_store.args[0] = ast.parse("out_ptr0 + _amd_output", mode="eval").body
    linear_store.args[1] = copy.deepcopy(value)
    normalize_mask(linear_store, 2, "_amd_linear_mask")
    input_linear.extend(
        [
            ast.Assign(
                targets=[ast.Name(id="_amd_value", ctx=ast.Store())], value=linear_load
            ),
            ast.Expr(value=linear_store),
        ]
    )
    fn.body = [
        ast.If(
            test=ast.parse("AMD_TILE_M == 0", mode="eval").body,
            body=[
                ast.If(
                    test=ast.parse("AMD_READ_ORDER == 0", mode="eval").body,
                    body=flat_body,
                    orelse=input_linear,
                )
            ],
            orelse=tiled,
        )
    ]
    for name in ("AMD_TILE_M", "AMD_TILE_N", "AMD_READ_ORDER"):
        signature_node.keys.append(ast.Constant(name))
        signature_node.values.append(ast.Constant("constexpr"))
        fn.args.args.append(
            ast.arg(arg=name, annotation=ast.parse("tl.constexpr", mode="eval").body)
        )
    native_values = {"amd_transpose_shape": shape}
    if two_dimensional:
        native_values["amd_kernel_autotune"] = True
    native_values["amd_cache_policy"] = "balanced"
    _native_metadata(inductor_meta, "tiled_transpose", **native_values)
    return _finish_source(tree, source, _context)

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


def optimize_pointwise_source(
    source: str, *, _context: AMDCodegenSource | None = None
) -> str:
    """Subtile static copy and arithmetic kernels on AMD GPUs."""
    tree = _context.tree if _context is not None else ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1:
        return source
    fn = functions[0]
    decorators = [
        n for n in fn.decorator_list if _call(n, "triton_heuristics.pointwise")
    ]
    if len(decorators) != 1:
        return source
    decorator = decorators[0]
    keywords = {k.arg: k.value for k in decorator.keywords}
    meta = keywords["triton_meta"]
    if not isinstance(meta, ast.Dict):
        return source
    limits = _source_limits(meta)
    if limits is None:
        return source
    values = {ast.literal_eval(k): v for k, v in zip(meta.keys, meta.values)}
    signature = ast.literal_eval(values["signature"])
    pointers = [name for name, dtype in signature.items() if dtype.startswith("*")]
    if list(signature) != [*pointers, "xnumel", "XBLOCK"]:
        return source
    if any(
        (
            dtype not in ("*bf16", "*fp16", "*fp32")
            for name, dtype in signature.items()
            if name in pointers
        )
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
            if statement.targets[0].id in assignments:
                return source
            assignments[statement.targets[0].id] = statement.value
        elif isinstance(statement, ast.Expr) and _call(statement.value, "tl.store"):
            stores.append(statement.value)
        else:
            return source
        for node in ast.walk(statement):
            if _call(node, "tl.load"):
                loads.append(node)
            if isinstance(node, ast.Call) and (
                not (
                    ast.unparse(node.func)
                    in (
                        "tl.load",
                        "tl.store",
                        "tl.full",
                        "tl.broadcast_to",
                        "tl.arange",
                        "tl.program_id",
                    )
                    or (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "to"
                        and (len(node.args) == 1)
                        and (ast.unparse(node.args[0]) == "tl.float32")
                    )
                )
            ):
                return source
    if len(stores) != 1:
        return source
    numel_node = assignments.get("xnumel")
    dynamic = numel_node is None and any(arg.arg == "xnumel" for arg in fn.args.args)
    n = numel_node.value if isinstance(numel_node, ast.Constant) else None
    if not dynamic and (not isinstance(n, int) or n <= 0 or n >= 2**31):
        return source
    extent = "xnumel" if dynamic else str(n)
    store = stores[0]
    address = store.args[0]
    if (
        not isinstance(address, ast.BinOp)
        or not isinstance(address.op, ast.Add)
        or ast.unparse(address.left) != "out_ptr0"
        or (not isinstance(address.right, ast.Name))
        or (
            ast.unparse(assignments.get(address.right.id, ast.Constant(None)))
            != "xindex"
        )
    ):
        return source
    for name, expression in (
        ("xoffset", "tl.program_id(0) * XBLOCK"),
        ("xindex", "xoffset + tl.arange(0, XBLOCK)[:]"),
    ):
        if name not in assignments or ast.dump(assignments[name]) != ast.dump(
            ast.parse(expression, mode="eval").body
        ):
            return source

    def expand(node: ast.AST) -> ast.AST:
        if (
            isinstance(node, ast.Name)
            and node.id.startswith("tmp")
            and (node.id in assignments)
        ):
            return expand(assignments[node.id])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and (node.func.attr == "to")
        ):
            return expand(node.func.value)
        if _call(node, "tl.broadcast_to"):
            return expand(node.args[0])
        if _call(node, "tl.full"):
            return copy.deepcopy(node.args[1])
        if _call(node, "tl.load"):
            return ast.Name(
                id=f"load{next((i for i, value in enumerate(loads) if value is node))}",
                ctx=ast.Load(),
            )
        if isinstance(node, ast.BinOp):
            return ast.BinOp(
                left=expand(node.left), op=node.op, right=expand(node.right)
            )
        return copy.deepcopy(node)

    valid_masks = OrderedSet(["None", "xmask"])
    xmask = ast.unparse(assignments.get("xmask", ast.Constant(None)))
    if xmask not in ("xindex < xnumel", "tl.full([XBLOCK], True, tl.int1)[:]"):
        return source
    if len(store.args) > 2 and ast.unparse(store.args[2]) not in valid_masks:
        return source
    if any(
        len(load.args) > 1 and ast.unparse(load.args[1]) not in valid_masks
        for load in loads
    ):
        return source
    if any(
        keyword.arg == "mask" and ast.unparse(keyword.value) not in valid_masks
        for call in [store, *loads]
        for keyword in call.keywords
    ):
        return source
    expression = ast.dump(expand(store.args[1]))
    family = None
    body = ""
    if (
        len(loads) == 2
        and pointers == ["in_ptr0", "in_ptr1", "out_ptr0"]
        and (ast.unparse(assignments.get("x0", ast.Constant(None))) == "xindex")
    ):
        offsets = [ast.unparse(load.args[0]) for load in loads]
        broadcast = ast.dump(
            ast.parse(f"(load0 * {1.0 / n!r}) * (load1 * 2.0)", mode="eval").body
            if n is not None
            else ast.Constant(None)
        )
        add = ast.dump(ast.parse("load0 + load1", mode="eval").body)
        if offsets == ["in_ptr0 + 0", "in_ptr1 + x0"] and expression == broadcast:
            family = "broadcast"
            body = f"\nscale = tl.load(in_ptr0) * {1.0 / n!r}\nfor u in tl.static_range(AMD_UNROLL):\n    i = tl.program_id(0) * XBLOCK + u * (XBLOCK // AMD_UNROLL) + lane\n    value = tl.load(in_ptr1 + i, i < {extent}, other=0).to(tl.float32)\n    tl.store(out_ptr0 + i, scale * (value * 2.0), i < {extent})\n"
        elif offsets == ["in_ptr0 + x0", "in_ptr1 + x0"] and expression == add:
            family = "add"
            body = f"\nfor u in tl.static_range(AMD_UNROLL):\n    i = tl.program_id(0) * XBLOCK + u * (XBLOCK // AMD_UNROLL) + lane\n    a = tl.load(in_ptr0 + i, i < {extent}, other=0).to(tl.float32)\n    b = tl.load(in_ptr1 + i, i < {extent}, other=0).to(tl.float32)\n    tl.store(out_ptr0 + i, a + b, i < {extent})\n"
    elif (
        n is not None
        and len(loads) == 1
        and (pointers == ["in_ptr0", "out_ptr0"])
        and (signature[pointers[0]] == signature[pointers[1]])
        and (expression == ast.dump(ast.Name(id="load0", ctx=ast.Load())))
    ):
        x0 = assignments.get("x0", ast.Constant(None))
        if (
            isinstance(x0, ast.BinOp)
            and isinstance(x0.op, ast.Mod)
            and isinstance(x0.right, ast.Constant)
            and (ast.unparse(x0.left) == "xindex")
            and (
                ast.unparse(assignments.get("x1", ast.Constant(None)))
                == f"xindex // {x0.right.value}"
            )
        ):
            c = x0.right.value
            mods = [
                node
                for node in ast.walk(loads[0].args[0])
                if isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Mod)
                and isinstance(node.left, ast.Name)
                and isinstance(node.right, ast.Constant)
            ]
            dims = {node.left.id: node.right.value for node in mods}
            if OrderedSet(dims) == OrderedSet(["x0", "x1"]):
                d, s = (dims["x0"], dims["x1"])
                offset = f"{d} * (x1 % {s}) + {s * d} * (x0 // {d}) + {s * c} * (x1 // {s}) + x0 % {d}"
                expected = ast.parse(f"in_ptr0 + ({offset})", mode="eval").body
                if (
                    all(isinstance(v, int) and v > 0 for v in (c, s, d))
                    and c % d == 0
                    and (n % (c * s) == 0)
                    and (ast.dump(loads[0].args[0]) == ast.dump(expected))
                ):
                    family = "transpose"
                    body = f"\ni = tl.program_id(0) * XBLOCK + lane\noutput = (i // {s * c}) * {s * c} + ((i // {d}) % {s}) * {c} + ((i // {s * d}) % {c // d}) * {d} + (i % {d})\nvalue = tl.load(in_ptr0 + i, i < {extent}, other=0)\ntl.store(out_ptr0 + output, value, i < {extent})\n"
    if (
        family is None
        and n is not None
        and (len(loads) == 1)
        and (pointers == ["in_ptr0", "out_ptr0"])
        and (signature[pointers[0]] == signature[pointers[1]])
        and (expression == ast.dump(ast.Name(id="load0", ctx=ast.Load())))
    ):
        dims = []
        for name in ("x0", "x1", "x2"):
            node = assignments.get(name)
            if (
                not isinstance(node, ast.BinOp)
                or not isinstance(node.op, ast.Mod)
                or (not isinstance(node.right, ast.Constant))
                or (not isinstance(node.right.value, int))
                or (node.right.value <= 0)
            ):
                break
            dims.append(node.right.value)
        if len(dims) == 3:
            d, h, s = dims
            c = d * h
            expected_indices = {
                "x0": f"xindex % {d}",
                "x1": f"(xindex // {d}) % {h}",
                "x2": f"(xindex // {c}) % {s}",
                "x3": f"xindex // {s * c}",
            }
            indices_match = all(
                (
                    ast.dump(assignments.get(key, ast.Constant(None)))
                    == ast.dump(ast.parse(value, mode="eval").body)
                    for key, value in expected_indices.items()
                )
            )
            expected_offset = ast.parse(
                f"in_ptr0 + (x0 + {d}*x2 + {s * d}*x1 + {s * c}*x3)", mode="eval"
            ).body
            if (
                indices_match
                and n % (s * c) == 0
                and (ast.dump(loads[0].args[0]) == ast.dump(expected_offset))
            ):
                family = "transpose"
                body = f"\ni = tl.program_id(0) * XBLOCK + lane\noutput = (i // {s * c}) * {s * c} + ((i // {d}) % {s}) * {c} + ((i // {s * d}) % {h}) * {d} + (i % {d})\nvalue = tl.load(in_ptr0 + i, i < {extent}, other=0)\ntl.store(out_ptr0 + output, value, i < {extent})\n"
    if family is None:
        return source
    if family == "transpose":
        input_linear = "\n".join(
            "    " + line for line in body.strip().splitlines()[1:]
        )
        body = f"\ni = tl.program_id(0) * XBLOCK + lane\nif AMD_READ_ORDER == 0:\n    x0 = i % {c}\n    x1 = i // {c}\n    offset = {d} * (x1 % {s}) + {s * d} * (x0 // {d}) + {s * c} * (x1 // {s}) + x0 % {d}\n    value = tl.load(in_ptr0 + offset, i < {extent}, other=0)\n    tl.store(out_ptr0 + i, value, i < {extent})\nelse:\n{input_linear}\n"
    extra_args = ["AMD_UNROLL"] + (["AMD_READ_ORDER"] if family == "transpose" else [])
    for name in extra_args:
        values["signature"].keys.append(ast.Constant(name))
        values["signature"].values.append(ast.Constant("constexpr"))
        fn.args.args.append(
            ast.arg(arg=name, annotation=ast.parse("tl.constexpr", mode="eval").body)
        )
    _native_metadata(keywords["inductor_meta"], family, amd_cache_policy="balanced")
    fn.body = ast.parse("lane = tl.arange(0, XBLOCK // AMD_UNROLL)\n" + body).body
    return _finish_source(tree, source, _context)


def optimize_flat_pointwise_source(
    source: str, *, _context: AMDCodegenSource | None = None
) -> str:
    """Tune layouts of static flat elementwise fusions without rewriting math."""
    tree = _context.tree if _context is not None else ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        return source
    fn = functions[0]
    decorators = [
        node for node in fn.decorator_list if _call(node, "triton_heuristics.pointwise")
    ]
    if len(decorators) != 1:
        return source
    keywords = {keyword.arg: keyword.value for keyword in decorators[0].keywords}
    meta = keywords["triton_meta"]
    limits = _source_limits(meta)
    if limits is None:
        return source
    signature_node = next(
        (
            value
            for key, value in zip(meta.keys, meta.values)
            if ast.literal_eval(key) == "signature"
        )
    )
    signature = ast.literal_eval(signature_node)
    pointers = [name for name, dtype in signature.items() if dtype.startswith("*")]
    if list(signature) != [*pointers, "xnumel", "XBLOCK"] or not pointers:
        return source
    if any(
        signature[name] not in ("*bf16", "*fp16", "*fp32", "*i1", "*i32", "*i64")
        for name in pointers
    ):
        return source
    assignments = {}
    stores = []
    loads = []
    elementwise = OrderedSet(
        [
            "tl.load",
            "tl.store",
            "tl.full",
            "tl.broadcast_to",
            "tl.arange",
            "tl.program_id",
            "tl.where",
            "tl.abs",
            "tl.exp",
            "tl.exp2",
            "tl.log",
            "tl.log2",
            "tl.sqrt",
            "tl.rsqrt",
            "tl.minimum",
            "tl.maximum",
            "tl.sigmoid",
            "tl.div_rn",
        ]
    )
    from triton.language.extra import libdevice as target_libdevice

    for statement in fn.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            if statement.targets[0].id in assignments:
                return source
            assignments[statement.targets[0].id] = statement.value
        elif isinstance(statement, ast.Expr) and _call(statement.value, "tl.store"):
            stores.append(statement.value)
        else:
            return source
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            cast = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "to"
                and (len(node.args) == 1)
                and (not node.keywords)
                and (
                    ast.unparse(node.args[0])
                    in (
                        "tl.float32",
                        "tl.float16",
                        "tl.bfloat16",
                        "tl.int32",
                        "tl.int64",
                        "tl.int1",
                    )
                )
            )
            if name == "libdevice.erf":
                return source
            device_math = name.startswith("libdevice.") and hasattr(
                target_libdevice, name.split(".")[1]
            )
            if name not in elementwise and (not cast) and (not device_math):
                return source
            if _call(node, "tl.load"):
                loads.append(node)
            if _call(node, "tl.full") and ast.unparse(node.args[0]) not in (
                "[1]",
                "[XBLOCK]",
            ):
                return source
            if _call(node, "tl.arange") and ast.unparse(node) != "tl.arange(0, XBLOCK)":
                return source
            if _call(node, "tl.program_id") and ast.unparse(node) != "tl.program_id(0)":
                return source
    n = assignments.get("xnumel")
    dynamic = n is None and any(arg.arg == "xnumel" for arg in fn.args.args)
    if (
        not dynamic
        and (
            not isinstance(n, ast.Constant)
            or not isinstance(n.value, int)
            or (not 0 < n.value < 2**31)
        )
        or not stores
        or (not loads)
    ):
        return source
    extent = "xnumel" if dynamic else str(n.value)
    expected = {
        "xoffset": "tl.program_id(0) * XBLOCK",
        "xindex": "xoffset + tl.arange(0, XBLOCK)[:]",
        "x0": "xindex",
    }
    if any(
        (
            ast.dump(assignments.get(name, ast.Constant(None)))
            != ast.dump(ast.parse(value, mode="eval").body)
            for name, value in expected.items()
        )
    ):
        return source
    if ast.unparse(assignments.get("xmask", ast.Constant(None))) not in (
        "xindex < xnumel",
        "tl.full([XBLOCK], True, tl.int1)[:]",
    ):
        return source
    addresses = OrderedSet([f"{pointer} + x0" for pointer in pointers])
    scalar_addresses = OrderedSet([f"{pointer} + 0" for pointer in pointers])
    for call in [*loads, *stores]:
        is_store = call in stores
        if ast.unparse(call.args[0]) not in addresses | (
            OrderedSet() if is_store else scalar_addresses
        ):
            return source
        mask_index = 2 if is_store else 1
        if len(call.args) > mask_index and ast.unparse(call.args[mask_index]) not in (
            "xmask",
            "None",
        ):
            return source
        if any(
            kw.arg == "mask" and ast.unparse(kw.value) not in ("xmask", "None")
            for kw in call.keywords
        ):
            return source
        if (
            len(call.args) <= mask_index
            and (not any(kw.arg == "mask" for kw in call.keywords))
            and (ast.unparse(call.args[0]) in addresses)
        ):
            return source
    for call in [*loads, *stores]:
        if ast.unparse(call.args[0]) in addresses:
            mask_index = 2 if call in stores else 1
            if len(call.args) > mask_index:
                call.args[mask_index] = ast.Name(id="xmask", ctx=ast.Load())
            else:
                for keyword in call.keywords:
                    if keyword.arg == "mask":
                        keyword.value = ast.Name(id="xmask", ctx=ast.Load())
    body = []
    for statement in fn.body:
        name = statement.targets[0].id if isinstance(statement, ast.Assign) else None
        if name in ("xnumel", "xoffset"):
            continue
        if name == "xindex":
            statement = ast.parse(
                "xindex = tl.program_id(0) * XBLOCK + _amd_u * (XBLOCK // AMD_UNROLL) + tl.arange(0, XBLOCK // AMD_UNROLL)"
            ).body[0]
        elif name == "xmask":
            statement = ast.parse(f"xmask = xindex < {extent}").body[0]
        for node in ast.walk(statement):
            if isinstance(node, ast.List):
                node.elts = [
                    ast.parse("XBLOCK // AMD_UNROLL", mode="eval").body
                    if isinstance(item, ast.Name) and item.id == "XBLOCK"
                    else item
                    for item in node.elts
                ]
        body.append(statement)
    loop = ast.parse("for _amd_u in tl.static_range(AMD_UNROLL):\n    pass").body[0]
    loop.body = body
    fn.body = ([] if dynamic else ast.parse(f"xnumel = {n.value}").body) + [loop]
    signature_node.keys.append(ast.Constant("AMD_UNROLL"))
    signature_node.values.append(ast.Constant("constexpr"))
    fn.args.args.append(
        ast.arg(
            arg="AMD_UNROLL", annotation=ast.parse("tl.constexpr", mode="eval").body
        )
    )
    _native_metadata(
        keywords["inductor_meta"], "flat_pointwise", amd_cache_policy="balanced"
    )
    return _finish_source(tree, source, _context)

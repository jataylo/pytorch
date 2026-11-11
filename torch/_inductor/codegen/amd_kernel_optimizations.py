from __future__ import annotations

import ast
import copy

from torch.utils._ordered_set import OrderedSet

from ..runtime.hints import AMDConfigLimits, DeviceProperties


def _call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and ast.unparse(node.func) == name


def supports_amd_kernel_optimizations(device: DeviceProperties) -> bool:
    return (
        device.type == "hip"
        and isinstance(device.cc, str)
        and device.cc.startswith("gfx")
        and isinstance(device.warp_size, int)
        and device.warp_size in (32, 64)
        and isinstance(device.max_threads_per_block, int)
        and device.max_threads_per_block >= device.warp_size
        and device.max_threads_per_block % device.warp_size == 0
    )


def _source_limits(meta: ast.Dict) -> AMDConfigLimits | None:
    if not isinstance(meta, ast.Dict):
        return None
    values = {
        ast.literal_eval(key): value for key, value in zip(meta.keys, meta.values)
    }
    device = values.get("device")
    if device is None:
        # Legacy source-only callers omit device metadata; codegen always supplies it.
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


def optimize_pointwise_source(source: str, *, extended: bool = False) -> str:
    """Use explicit layouts for static copy and arithmetic kernels on AMD GPUs."""
    tree = ast.parse(source)
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
    warp_size = limits.warp_size
    values = {ast.literal_eval(k): v for k, v in zip(meta.keys, meta.values)}
    signature = ast.literal_eval(values["signature"])
    pointers = [name for name, dtype in signature.items() if dtype.startswith("*")]
    if list(signature) != [*pointers, "xnumel", "XBLOCK"]:
        return source
    if any(
        dtype not in ("*bf16", "*fp16", "*fp32")
        for name, dtype in signature.items()
        if name in pointers
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
            assignments[statement.targets[0].id] = statement.value
        elif isinstance(statement, ast.Expr) and _call(statement.value, "tl.store"):
            stores.append(statement.value)
        else:
            return source
        for node in ast.walk(statement):
            if _call(node, "tl.load"):
                loads.append(node)
            if isinstance(node, ast.Call) and not (
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
                    and len(node.args) == 1
                    and ast.unparse(node.args[0]) == "tl.float32"
                )
            ):
                return source
    if len(stores) != 1 or not isinstance(assignments.get("xnumel"), ast.Constant):
        return source
    n = assignments["xnumel"].value
    if not isinstance(n, int) or n <= 0 or n >= 2**31:
        return source
    store = stores[0]
    address = store.args[0]
    if (
        not isinstance(address, ast.BinOp)
        or not isinstance(address.op, ast.Add)
        or ast.unparse(address.left) != "out_ptr0"
        or not isinstance(address.right, ast.Name)
        or ast.unparse(assignments.get(address.right.id, ast.Constant(None)))
        != "xindex"
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
            and node.id in assignments
        ):
            return expand(assignments[node.id])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to"
        ):
            return expand(node.func.value)
        if _call(node, "tl.broadcast_to"):
            return expand(node.args[0])
        if _call(node, "tl.full"):
            return copy.deepcopy(node.args[1])
        if _call(node, "tl.load"):
            return ast.Name(
                id=f"load{next(i for i, value in enumerate(loads) if value is node)}",
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
        and ast.unparse(assignments.get("x0", ast.Constant(None))) == "xindex"
    ):
        offsets = [ast.unparse(load.args[0]) for load in loads]
        broadcast = ast.dump(
            ast.parse(f"(load0 * {1.0 / n!r}) * (load1 * 2.0)", mode="eval").body
        )
        add = ast.dump(ast.parse("load0 + load1", mode="eval").body)
        if offsets == ["in_ptr0 + 0", "in_ptr1 + x0"] and expression == broadcast:
            family = "broadcast"
            body = f"""
scale = gl.load(in_ptr0) * {1.0 / n!r}
for u in gl.static_range(AMD_UNROLL):
    i = gl.program_id(0) * XBLOCK + u * (XBLOCK // AMD_UNROLL) + lane
    value = gl.load(in_ptr1 + i, i < {n}, other=0).to(gl.float32)
    gl.store(out_ptr0 + i, scale * (value * 2.0), i < {n})
"""
        elif offsets == ["in_ptr0 + x0", "in_ptr1 + x0"] and expression == add:
            family = "add"
            body = f"""
for u in gl.static_range(AMD_UNROLL):
    i = gl.program_id(0) * XBLOCK + u * (XBLOCK // AMD_UNROLL) + lane
    a = gl.load(in_ptr0 + i, i < {n}, other=0).to(gl.float32)
    b = gl.load(in_ptr1 + i, i < {n}, other=0).to(gl.float32)
    gl.store(out_ptr0 + i, a + b, i < {n})
"""
    elif (
        len(loads) == 1
        and pointers == ["in_ptr0", "out_ptr0"]
        and signature[pointers[0]] == signature[pointers[1]]
        and expression == ast.dump(ast.Name(id="load0", ctx=ast.Load()))
    ):
        x0 = assignments.get("x0", ast.Constant(None))
        if (
            isinstance(x0, ast.BinOp)
            and isinstance(x0.op, ast.Mod)
            and isinstance(x0.right, ast.Constant)
            and ast.unparse(x0.left) == "xindex"
            and ast.unparse(assignments.get("x1", ast.Constant(None)))
            == f"xindex // {x0.right.value}"
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
                d, s = dims["x0"], dims["x1"]
                offset = f"{d} * (x1 % {s}) + {s * d} * (x0 // {d}) + {s * c} * (x1 // {s}) + x0 % {d}"
                expected = ast.parse(f"in_ptr0 + ({offset})", mode="eval").body
                if (
                    all(isinstance(v, int) and v > 0 for v in (c, s, d))
                    and c % d == 0
                    and n % (c * s) == 0
                    and ast.dump(loads[0].args[0]) == ast.dump(expected)
                ):
                    family = "transpose"
                    body = f"""
i = gl.program_id(0) * XBLOCK + lane
output = (i // {s * c}) * {s * c} + ((i // {d}) % {s}) * {c} + ((i // {s * d}) % {c // d}) * {d} + (i % {d})
value = gl.load(in_ptr0 + i, i < {n}, other=0)
gl.store(out_ptr0 + output, value, i < {n})
"""
    if (
        family is None
        and len(loads) == 1
        and pointers == ["in_ptr0", "out_ptr0"]
        and signature[pointers[0]] == signature[pointers[1]]
        and expression == ast.dump(ast.Name(id="load0", ctx=ast.Load()))
    ):
        dims = []
        for name in ("x0", "x1", "x2"):
            node = assignments.get(name)
            if (
                not isinstance(node, ast.BinOp)
                or not isinstance(node.op, ast.Mod)
                or not isinstance(node.right, ast.Constant)
                or not isinstance(node.right.value, int)
                or node.right.value <= 0
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
                ast.dump(assignments.get(key, ast.Constant(None)))
                == ast.dump(ast.parse(value, mode="eval").body)
                for key, value in expected_indices.items()
            )
            expected_offset = ast.parse(
                f"in_ptr0 + (x0 + {d}*x2 + {s * d}*x1 + {s * c}*x3)", mode="eval"
            ).body
            if (
                indices_match
                and n % (s * c) == 0
                and ast.dump(loads[0].args[0]) == ast.dump(expected_offset)
            ):
                family = "transpose"
                body = f"""
i = gl.program_id(0) * XBLOCK + lane
output = (i // {s * c}) * {s * c} + ((i // {d}) % {s}) * {c} + ((i // {s * d}) % {h}) * {d} + (i % {d})
value = gl.load(in_ptr0 + i, i < {n}, other=0)
gl.store(out_ptr0 + output, value, i < {n})
"""
    if family is None:
        return source
    extra_args = ["AMD_VEC", "AMD_UNROLL", "AMD_WARPS"]
    if family == "transpose":
        input_linear = "\n".join(
            "    " + line for line in body.strip().splitlines()[1:]
        )
        body = f"""
i = gl.program_id(0) * XBLOCK + lane
if AMD_READ_ORDER == 0:
    x0 = i % {c}
    x1 = i // {c}
    offset = {d} * (x1 % {s}) + {s * d} * (x0 // {d}) + {s * c} * (x1 // {s}) + x0 % {d}
    value = gl.load(in_ptr0 + offset, i < {n}, other=0)
    gl.store(out_ptr0 + i, value, i < {n})
else:
{input_linear}
"""
        extra_args.append("AMD_READ_ORDER")
    for name in extra_args:
        values["signature"].keys.append(ast.Constant(name))
        values["signature"].values.append(ast.Constant("constexpr"))
        fn.args.args.append(
            ast.arg(arg=name, annotation=ast.parse("gl.constexpr", mode="eval").body)
        )
    inductor = keywords["inductor_meta"]
    inductor.keys.extend([ast.Constant("amd_gluon_family"), ast.Constant("use_gluon")])
    inductor.values.extend([ast.Constant(family), ast.Constant(True)])
    if extended:
        inductor.keys.append(ast.Constant("amd_cache_policy"))
        inductor.values.append(ast.Constant("balanced"))
    fn.decorator_list[-1] = ast.parse("gluon.jit", mode="eval").body
    fn.body = ast.parse(
        f"lane = gl.arange(0, XBLOCK // AMD_UNROLL, layout=gl.BlockedLayout([AMD_VEC], [{warp_size}], [AMD_WARPS], [0]))\n"
        + body
    ).body
    tree.body.insert(
        0,
        ast.ImportFrom(
            module="triton.experimental.gluon",
            names=[ast.alias(name="language", asname="gl")],
            level=0,
        ),
    )
    tree.body.insert(
        0,
        ast.ImportFrom(
            module="triton.experimental", names=[ast.alias(name="gluon")], level=0
        ),
    )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def simplify_masked_modulo(source: str) -> str:
    """Remove a row-reduction modulo when the load mask proves its range."""
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        return source
    fn = functions[0]
    if not any(
        _call(node, "triton_heuristics.reduction") for node in fn.decorator_list
    ):
        return source
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    x = assignments.get("xnumel")
    r = assignments.get("r0_numel")
    if (
        not isinstance(x, ast.Constant)
        or not isinstance(r, ast.Constant)
        or not isinstance(x.value, int)
        or not isinstance(r.value, int)
        or x.value <= 0
        or r.value <= 0
        or x.value * r.value >= 2**31
    ):
        return source
    required = {
        "xoffset": "tl.program_id(0) * XBLOCK",
        "xindex": "xoffset + tl.arange(0, XBLOCK)[:, None]",
        "xmask": "xindex < xnumel",
        "r0_base": "tl.arange(0, R0_BLOCK)[None, :]",
        "r0_index": "r0_offset + r0_base",
        "r0_mask": "r0_index < r0_numel",
        "x0": "xindex",
        "r0_1": "r0_index",
    }
    if any(
        ast.dump(assignments.get(name, ast.Constant(None)))
        != ast.dump(ast.parse(value, mode="eval").body)
        for name, value in required.items()
    ):
        return source
    loops = [node for node in ast.walk(fn) if isinstance(node, ast.For)]
    if not loops or any(
        not _call(loop.iter, "tl.range")
        or [ast.unparse(arg) for arg in loop.iter.args] != ["0", "r0_numel", "R0_BLOCK"]
        for loop in loops
    ):
        return source
    index = ast.parse(f"r0_1 + {r.value} * x0", mode="eval").body

    def resolve(node):
        if (
            isinstance(node, ast.Name)
            and node.id.startswith("tmp")
            and node.id in assignments
        ):
            return resolve(assignments[node.id])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to"
        ):
            return resolve(node.func.value)
        if _call(node, "tl.full"):
            return node.args[1]
        return node

    def conjunction(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            return conjunction(node.left) + conjunction(node.right)
        return [node]

    changed = False
    for load in ast.walk(fn):
        if not _call(load, "tl.load") or len(load.args) < 2:
            continue
        address = load.args[0]
        if (
            not isinstance(address, ast.BinOp)
            or not isinstance(address.op, ast.Add)
            or not isinstance(address.right, ast.BinOp)
            or not isinstance(address.right.op, ast.Mod)
        ):
            continue
        mod = address.right
        if (
            ast.dump(mod.left) != ast.dump(index)
            or not isinstance(mod.right, ast.Constant)
            or not isinstance(mod.right.value, int)
            or mod.right.value <= 0
        ):
            continue
        leaves = conjunction(load.args[1])
        if not OrderedSet(["xmask", "r0_mask"]).issubset(
            OrderedSet([leaf.id for leaf in leaves if isinstance(leaf, ast.Name)])
        ):
            continue
        for leaf in leaves:
            predicate = resolve(leaf)
            if (
                isinstance(predicate, ast.Compare)
                and len(predicate.ops) == 1
                and isinstance(predicate.ops[0], ast.Lt)
                and ast.dump(resolve(predicate.left)) == ast.dump(index)
                and isinstance(resolve(predicate.comparators[0]), ast.Constant)
                and resolve(predicate.comparators[0]).value == mod.right.value
            ):
                address.right = mod.left
                changed = True
                break
    if not changed:
        return source
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def optimize_row_source(
    source: str,
    *,
    approximate: bool = False,
    extended: bool = False,
    further: bool = False,
) -> str:
    """Keep short row reductions and their statistics in one wave layout."""
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    fn = next(
        (
            node
            for node in functions
            if any(
                _call(decorator, "triton_heuristics.persistent_reduction")
                for decorator in node.decorator_list
            )
        ),
        None,
    )
    helpers = [node for node in functions if node is not fn]
    if fn is None or (helpers and not further):
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
    if OrderedSet(hints) != OrderedSet(["x", "r0_"]) or not 32 <= hints["r0_"] <= (
        32768 if extended else 8192
    ):
        return source
    outer = further and ast.unparse(
        keywords.get("reduction_hint", ast.Constant(None))
    ) in ("ReductionHint.OUTER", "ReductionHint.OUTER_TINY")
    triton_meta = keywords["triton_meta"]
    limits = _source_limits(triton_meta)
    if limits is None or hints["r0_"] > limits.max_tile:
        return source
    warp_size = limits.warp_size
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
        if (
            not further
            or not helpers
            or hints["r0_"] != 32768
            or input_dtype != "*bf16"
        ):
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
    xnumel = assignments.get("xnumel")
    if (
        not isinstance(xnumel, ast.Constant)
        or not isinstance(xnumel.value, int)
        or xnumel.value <= 0
    ):
        return source
    softmax = "triton_helpers.fmax2" in calls and "libdevice.exp" in calls
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.extra import libdevice as gdev

    if any(
        name.startswith("tl.")
        and name[3:] != "broadcast_to"
        and not hasattr(gl, name[3:])
        for name in calls
    ):
        return source
    if any(
        name.startswith("libdevice.") and not hasattr(gdev, name.split(".")[1])
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

    class Convert(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in ("tl", "tl_math"):
                return ast.copy_location(ast.Name(id="gl", ctx=node.ctx), node)
            return node

        def visit_Call(self, node):
            original = ast.unparse(node.func)
            self.generic_visit(node)
            if original == "tl.broadcast_to":
                zero = ast.Call(
                    func=ast.parse("gl.full", mode="eval").body,
                    args=[
                        node.args[1],
                        ast.Constant(0),
                        ast.Attribute(value=node.args[0], attr="dtype", ctx=ast.Load()),
                    ],
                    keywords=[
                        ast.keyword(
                            arg="layout",
                            value=ast.Name(id="_amd_layout", ctx=ast.Load()),
                        )
                    ],
                )
                return ast.Subscript(
                    value=ast.Call(
                        func=ast.parse("gl.broadcast", mode="eval").body,
                        args=[node.args[0], zero],
                        keywords=[],
                    ),
                    slice=ast.Constant(0),
                    ctx=ast.Load(),
                )
            if original == "tl.arange":
                axis = 1 if ast.unparse(node.args[1]) == "XBLOCK" else 0
                node.keywords.append(
                    ast.keyword(
                        arg="layout",
                        value=ast.parse(
                            f"gl.SliceLayout({axis}, _amd_layout)", mode="eval"
                        ).body,
                    )
                )
            elif original == "tl.full":
                shape = ast.unparse(node.args[0])
                layout = (
                    "_amd_layout"
                    if shape in ("[XBLOCK, R0_BLOCK]", "[1, 1]")
                    else "gl.SliceLayout(1, _amd_layout)"
                    if shape == "[XBLOCK]"
                    else "gl.SliceLayout(0, _amd_layout)"
                )
                node.keywords.append(
                    ast.keyword(arg="layout", value=ast.parse(layout, mode="eval").body)
                )
            elif original == "triton_helpers.fmax2":
                node.func = ast.parse("gl.max", mode="eval").body
            elif original == "triton_helpers.max2":
                node.func = ast.Name(id="_amd_max2", ctx=ast.Load())
            elif original == "triton_helpers.any":
                node.func = ast.Name(id="_amd_any", ctx=ast.Load())
            elif approximate and softmax and original == "libdevice.exp":
                node.func = ast.parse("gl.exp", mode="eval").body
            return node

        def visit_BinOp(self, node):
            self.generic_visit(node)
            if approximate and softmax and isinstance(node.op, ast.Div):
                return ast.copy_location(
                    ast.BinOp(
                        left=node.left,
                        op=ast.Mult(),
                        right=ast.Call(
                            func=ast.Name(id="_amd_reciprocal", ctx=ast.Load()),
                            args=[node.right],
                            keywords=[],
                        ),
                    ),
                    node,
                )
            return node

    fn.body = [Convert().visit(statement) for statement in fn.body]
    for helper in helpers:
        helper.decorator_list = [ast.parse("gluon.jit", mode="eval").body]
    extra_args = ["AMD_LANES", "AMD_VEC", "AMD_WARPS", "AMD_SPLIT_WARPS"]
    if outer:
        extra_args.append("AMD_ROW_ORDER")
    for name in extra_args:
        metadata["signature"].keys.append(ast.Constant(name))
        metadata["signature"].values.append(ast.Constant("constexpr"))
        fn.args.args.append(
            ast.arg(arg=name, annotation=ast.parse("gl.constexpr", mode="eval").body)
        )
    inductor_meta = keywords["inductor_meta"]
    inductor_meta.keys.extend(
        [
            ast.Constant("use_gluon"),
            ast.Constant("amd_gluon_family"),
            ast.Constant("amd_reduction_extent"),
            ast.Constant("amd_row_count"),
            ast.Constant("amd_approximate_math"),
        ]
    )
    inductor_meta.values.extend(
        [
            ast.Constant(True),
            ast.Constant("row_reduction"),
            ast.Constant(hints["r0_"]),
            ast.Constant(xnumel.value),
            ast.Constant(approximate and softmax),
        ]
    )
    if extended:
        inductor_meta.keys.append(ast.Constant("amd_cache_policy"))
        inductor_meta.values.append(ast.Constant("balanced"))
    if scan:
        for name in ("amd_graph_tuning", "amd_confirm_tuning"):
            index = next(
                (
                    index
                    for index, key in enumerate(inductor_meta.keys)
                    if ast.literal_eval(key) == name
                ),
                None,
            )
            if index is None:
                inductor_meta.keys.append(ast.Constant(name))
                inductor_meta.values.append(ast.Constant(True))
            else:
                inductor_meta.values[index] = ast.Constant(True)
    if outer:
        inductor_meta.keys.append(ast.Constant("amd_outer_row"))
        inductor_meta.values.append(ast.Constant(True))
    fn.decorator_list[-1] = ast.parse("gluon.jit", mode="eval").body
    layout = (
        f"gl.BlockedLayout([1, AMD_VEC], [{warp_size} // AMD_LANES, AMD_LANES], "
        "([1, AMD_WARPS] if AMD_SPLIT_WARPS else [AMD_WARPS, 1]), [1, 0])"
    )
    if outer:
        layout = (
            f"gl.BlockedLayout([AMD_VEC, 1], [AMD_LANES, {warp_size} // AMD_LANES], "
            "([1, AMD_WARPS] if AMD_SPLIT_WARPS else [AMD_WARPS, 1]), [0, 1]) "
            f"if AMD_ROW_ORDER else {layout}"
        )
    fn.body.insert(0, ast.parse("_amd_layout: gl.constexpr = " + layout).body[0])
    helpers = """
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice

@gluon.jit
def _amd_maximum(a, b):
    return gl.where((a > b) | (a != a), a, b)

@gluon.jit
def _amd_max2(value, axis: gl.constexpr):
    return gl.reduce(value, axis, _amd_maximum)

@gluon.jit
def _amd_any(value, axis: gl.constexpr):
    return gl.sum((value != 0).to(gl.int32), axis) != 0

@gluon.jit
def _amd_reciprocal(value):
    return gl.inline_asm_elementwise("v_rcp_f32 $0, $1", constraints="=v,v", args=[value], dtype=gl.float32, is_pure=True, pack=1)
"""
    index = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.FunctionDef)
    )
    tree.body[index:index] = ast.parse(helpers).body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def optimize_gelu_source(source: str, *, extended: bool = False) -> str:
    """Share the Gaussian exponential in opt-in approximate GELU backward."""
    tree = ast.parse(source)
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
    signature_node = next(
        value
        for key, value in zip(meta.keys, meta.values)
        if ast.literal_eval(key) == "signature"
    )
    signature = ast.literal_eval(signature_node)
    pointers = [name for name, dtype in signature.items() if dtype.startswith("*")]
    if (
        len(pointers) not in (2, 3)
        or list(signature)[-2:] != ["xnumel", "XBLOCK"]
        or any(signature[name] not in ("*bf16", "*fp16", "*fp32") for name in pointers)
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
            assignments[statement.targets[0].id] = statement.value
        elif isinstance(statement, ast.Expr) and _call(statement.value, "tl.store"):
            stores.append(statement.value)
        else:
            return source
        for node in ast.walk(statement):
            if _call(node, "tl.load"):
                loads.append(node)
            if isinstance(node, ast.Call) and not (
                ast.unparse(node.func)
                in (
                    "tl.load",
                    "tl.store",
                    "tl.full",
                    "tl.broadcast_to",
                    "tl.arange",
                    "tl.program_id",
                    "libdevice.erf",
                    "libdevice.exp",
                )
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "to"
                    and [ast.unparse(arg) for arg in node.args] == ["tl.float32"]
                )
            ):
                return source
    if len(loads) != 2 or len(stores) != 1:
        return source
    n = assignments.get("xnumel")
    if (
        not isinstance(n, ast.Constant)
        or not isinstance(n.value, int)
        or not 0 < n.value < 2**31
    ):
        return source
    required = {
        "xoffset": "tl.program_id(0) * XBLOCK",
        "xindex": "xoffset + tl.arange(0, XBLOCK)[:]",
        "x0": "xindex",
    }
    if any(
        ast.dump(assignments.get(name, ast.Constant(None)))
        != ast.dump(ast.parse(value, mode="eval").body)
        for name, value in required.items()
    ):
        return source
    grad, x = pointers[:2]
    output = (
        pointers[0]
        if len(pointers) == 2 and pointers[0].startswith("in_out_ptr")
        else pointers[-1]
        if len(pointers) == 3
        else None
    )
    if (
        output is None
        or [ast.unparse(load.args[0]) for load in loads]
        != [f"{grad} + x0", f"{x} + x0"]
        or ast.unparse(stores[0].args[0]) != f"{output} + x0"
    ):
        return source
    if ast.unparse(assignments.get("xmask", ast.Constant(None))) not in (
        "xindex < xnumel",
        "tl.full([XBLOCK], True, tl.int1)[:]",
    ):
        return source
    if any(
        len(call.args) > (2 if call is stores[0] else 1)
        and ast.unparse(call.args[2 if call is stores[0] else 1])
        not in ("None", "xmask")
        for call in [*loads, *stores]
    ):
        return source

    def expand(node):
        if (
            isinstance(node, ast.Name)
            and node.id.startswith("tmp")
            and node.id in assignments
        ):
            return expand(assignments[node.id])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to"
        ):
            return expand(node.func.value)
        if _call(node, "tl.full"):
            return node.args[1]
        if _call(node, "tl.broadcast_to"):
            return expand(node.args[0])
        if _call(node, "tl.load"):
            return ast.Name(id="g" if node is loads[0] else "x", ctx=ast.Load())
        if isinstance(node, ast.Call) and ast.unparse(node.func) in (
            "libdevice.erf",
            "libdevice.exp",
        ):
            return ast.Call(
                func=ast.Name(id=node.func.attr, ctx=ast.Load()),
                args=[expand(arg) for arg in node.args],
                keywords=[],
            )
        if isinstance(node, ast.BinOp):
            return ast.BinOp(
                left=expand(node.left), op=node.op, right=expand(node.right)
            )
        return node

    expected = ast.parse(
        "g * (((erf(x * 0.7071067811865476) + 1.0) * 0.5) + x * (exp((x * x) * -0.5) * 0.3989422804014327))",
        mode="eval",
    ).body
    if ast.dump(expand(stores[0].args[1])) != ast.dump(expected):
        return source
    signature_node.keys.append(ast.Constant("AMD_UNROLL"))
    signature_node.values.append(ast.Constant("constexpr"))
    fn.args.args.append(
        ast.arg(
            arg="AMD_UNROLL", annotation=ast.parse("tl.constexpr", mode="eval").body
        )
    )
    inductor_meta = keywords["inductor_meta"]
    inductor_meta.keys.extend(
        [ast.Constant("amd_pointwise_family"), ast.Constant("amd_approximate_math")]
    )
    inductor_meta.values.extend([ast.Constant("gelu_cdf"), ast.Constant(True)])
    if extended:
        inductor_meta.keys.append(ast.Constant("amd_cache_policy"))
        inductor_meta.values.append(ast.Constant("balanced"))
    fn.body = ast.parse(f"""
for u in tl.static_range(AMD_UNROLL):
    i=tl.program_id(0)*XBLOCK+u*(XBLOCK//AMD_UNROLL)+tl.arange(0,XBLOCK//AMD_UNROLL)
    a=tl.load({grad}+i,i<{n.value},other=0).to(tl.float32)
    b=tl.load({x}+i,i<{n.value},other=0).to(tl.float32)
    ex=tl.exp(-0.5*b*b)
    t=tl.inline_asm_elementwise("v_rcp_f32 $0, $1",constraints="=v,v",args=[1.0+0.3275911*tl.abs(b)*0.7071067811865476],dtype=tl.float32,is_pure=True,pack=1)
    poly=t*(0.254829592+t*(-0.284496736+t*(1.421413741+t*(-1.453152027+t*1.061405429))))
    tail=0.5*poly*ex
    cdf=tl.where(b<0.0,tail,1.0-tail)
    value=a*(cdf+b*(0.3989422804014327*ex))
    tl.store({output}+i,value,i<{n.value})
""").body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def optimize_flat_pointwise_source(source: str) -> str:
    """Tune layouts of static flat elementwise fusions without rewriting math."""
    tree = ast.parse(source)
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
    warp_size = limits.warp_size
    signature_node = next(
        value
        for key, value in zip(meta.keys, meta.values)
        if ast.literal_eval(key) == "signature"
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
    from triton.experimental.gluon.language.extra import libdevice as gdev

    for statement in fn.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
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
                and len(node.args) == 1
                and not node.keywords
                and ast.unparse(node.args[0])
                in (
                    "tl.float32",
                    "tl.float16",
                    "tl.bfloat16",
                    "tl.int32",
                    "tl.int64",
                    "tl.int1",
                )
            )
            if name == "libdevice.erf":
                return source
            device_math = name.startswith("libdevice.") and hasattr(
                gdev, name.split(".")[1]
            )
            if name not in elementwise and not cast and not device_math:
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
    if (
        not isinstance(n, ast.Constant)
        or not isinstance(n.value, int)
        or not 0 < n.value < 2**31
        or not stores
        or not loads
    ):
        return source
    expected = {
        "xoffset": "tl.program_id(0) * XBLOCK",
        "xindex": "xoffset + tl.arange(0, XBLOCK)[:]",
        "x0": "xindex",
    }
    if any(
        ast.dump(assignments.get(name, ast.Constant(None)))
        != ast.dump(ast.parse(value, mode="eval").body)
        for name, value in expected.items()
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
            and not any(kw.arg == "mask" for kw in call.keywords)
            and ast.unparse(call.args[0]) in addresses
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

    class Convert(ast.NodeTransformer):
        def visit_Name(self, node):
            return (
                ast.copy_location(ast.Name(id="gl", ctx=node.ctx), node)
                if node.id in ("tl", "tl_math")
                else node
            )

        def visit_Call(self, node):
            original = ast.unparse(node.func)
            self.generic_visit(node)
            if original == "tl.sigmoid":
                return ast.BinOp(
                    left=ast.Constant(1),
                    op=ast.Div(),
                    right=ast.BinOp(
                        left=ast.Constant(1),
                        op=ast.Add(),
                        right=ast.Call(
                            func=ast.parse("gl.exp", mode="eval").body,
                            args=[ast.UnaryOp(op=ast.USub(), operand=node.args[0])],
                            keywords=[],
                        ),
                    ),
                )
            if original in ("tl.arange", "tl.full"):
                node.keywords.append(
                    ast.keyword(
                        arg="layout", value=ast.Name(id="_amd_layout", ctx=ast.Load())
                    )
                )
            elif original == "tl.broadcast_to":
                zero = ast.Call(
                    func=ast.parse("gl.full", mode="eval").body,
                    args=[
                        node.args[1],
                        ast.Constant(0),
                        ast.Attribute(value=node.args[0], attr="dtype", ctx=ast.Load()),
                    ],
                    keywords=[
                        ast.keyword(
                            arg="layout",
                            value=ast.Name(id="_amd_layout", ctx=ast.Load()),
                        )
                    ],
                )
                return ast.Subscript(
                    value=ast.Call(
                        func=ast.parse("gl.broadcast", mode="eval").body,
                        args=[node.args[0], zero],
                        keywords=[],
                    ),
                    slice=ast.Constant(0),
                    ctx=ast.Load(),
                )
            return node

    body = []
    for statement in fn.body:
        name = statement.targets[0].id if isinstance(statement, ast.Assign) else None
        if name == "xnumel":
            continue
        if name == "xindex":
            body.extend(
                ast.parse(
                    "xindex = gl.program_id(0) * XBLOCK + _amd_u * (XBLOCK // AMD_UNROLL) "
                    "+ gl.arange(0, XBLOCK // AMD_UNROLL, layout=_amd_layout)"
                ).body
            )
        elif name == "xmask":
            body.extend(ast.parse(f"xmask = xindex < {n.value}").body)
        elif name != "xoffset":
            body.append(Convert().visit(statement))
    # Full vectors in the original body follow the unrolled tile extent.
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.List):
                node.elts = [
                    ast.parse("XBLOCK // AMD_UNROLL", mode="eval").body
                    if isinstance(item, ast.Name) and item.id == "XBLOCK"
                    else item
                    for item in node.elts
                ]
    prefix = ast.parse(
        f"\nxnumel = {n.value}\n"
        f"_amd_layout: gl.constexpr = gl.BlockedLayout([AMD_VEC], [{warp_size}], [AMD_WARPS], [0])\n"
        "for _amd_u in gl.static_range(AMD_UNROLL):\n    pass\n"
    ).body
    prefix[-1].body = body
    fn.body = prefix
    for name in ("AMD_VEC", "AMD_UNROLL", "AMD_WARPS"):
        signature_node.keys.append(ast.Constant(name))
        signature_node.values.append(ast.Constant("constexpr"))
        fn.args.args.append(
            ast.arg(arg=name, annotation=ast.parse("gl.constexpr", mode="eval").body)
        )
    inductor_meta = keywords["inductor_meta"]
    inductor_meta.keys.extend(
        [
            ast.Constant("use_gluon"),
            ast.Constant("amd_gluon_family"),
            ast.Constant("amd_cache_policy"),
        ]
    )
    inductor_meta.values.extend(
        [ast.Constant(True), ast.Constant("flat_pointwise"), ast.Constant("balanced")]
    )
    fn.decorator_list[-1] = ast.parse("gluon.jit", mode="eval").body
    tree.body[tree.body.index(fn) : tree.body.index(fn)] = ast.parse(
        "from triton.experimental import gluon\n"
        "from triton.experimental.gluon import language as gl\n"
        "from triton.experimental.gluon.language.extra import libdevice"
    ).body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"

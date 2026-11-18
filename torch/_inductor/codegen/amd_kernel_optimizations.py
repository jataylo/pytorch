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


def optimize_scan_source(
    source: str, *, _context: AMDCodegenSource | None = None
) -> str:
    r"""optimize_scan_source(source, *, _context=None) -> str

    Expose flat and hierarchical additive scans to the native AMD autotuner.
    """
    tree = _context.tree if _context is not None else ast.parse(source)
    kernels = [
        (node, decorator)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if _call(decorator, "triton_heuristics.persistent_reduction")
        or _call(decorator, "triton_heuristics.reduction")
    ]
    if len(kernels) != 1:
        return source
    fn, decorator = kernels[0]
    keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
    hints = ast.literal_eval(keywords["size_hints"])
    if OrderedSet(hints) != OrderedSet(["x", "r0_"]):
        return source
    meta = keywords["triton_meta"]
    limits = _source_limits(meta)
    if limits is None or hints["r0_"] < limits.warp_size * 64:
        return source
    fields = {
        ast.literal_eval(key): value for key, value in zip(meta.keys, meta.values)
    }
    signature = fields["signature"]
    names = list(ast.literal_eval(signature))
    if names[-3:] != ["xnumel", "r0_numel", "XBLOCK"] and names[-4:] != [
        "xnumel",
        "r0_numel",
        "XBLOCK",
        "R0_BLOCK",
    ]:
        return source
    helpers = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node is not fn
    }
    scans = [node for node in ast.walk(fn) if _call(node, "tl.associative_scan")]
    if not scans:
        return source
    for node in scans:
        if (
            len(node.args) != 3
            or node.keywords
            or not isinstance(node.args[0], ast.Tuple)
            or len(node.args[0].elts) != 1
            or not isinstance(node.args[1], ast.Constant)
            or node.args[1].value != 1
            or not isinstance(node.args[2], ast.Name)
        ):
            return source
        helper = helpers.get(node.args[2].id)
        if helper is None or len(helper.args.args) != 2 or len(helper.body) != 2:
            return source
        assignment, result = helper.body
        if (
            not isinstance(assignment, ast.Assign)
            or len(assignment.targets) != 1
            or not isinstance(assignment.targets[0], ast.Name)
            or not isinstance(result, ast.Return)
        ):
            return source
        left, right = helper.args.args
        expected = ast.parse(f"{left.arg} + {right.arg}", mode="eval").body
        if (
            ast.dump(assignment.value) != ast.dump(expected)
            or not isinstance(result.value, ast.Name)
            or result.value.id != assignment.targets[0].id
        ):
            return source
    for node in scans:
        node.func = ast.parse(
            "triton_helpers.associative_scan_in_chunks", mode="eval"
        ).body
        node.args = [*node.args[:2], ast.Name(id="AMD_SCAN_TILE", ctx=ast.Load())]
    fn.args.args.append(
        ast.arg(
            arg="AMD_SCAN_TILE", annotation=ast.parse("tl.constexpr", mode="eval").body
        )
    )
    signature.keys.append(ast.Constant("AMD_SCAN_TILE"))
    signature.values.append(ast.Constant("constexpr"))
    inductor_meta = keywords["inductor_meta"]
    family = next(
        (
            ast.literal_eval(value)
            for key, value in zip(inductor_meta.keys, inductor_meta.values)
            if ast.literal_eval(key) == "amd_native_family"
        ),
        "scan",
    )
    _native_metadata(
        inductor_meta,
        family,
        amd_hierarchical_scan=True,
        amd_graph_tuning=True,
        amd_confirm_tuning=True,
        amd_cache_policy="balanced",
    )
    return _finish_source(tree, source, _context)

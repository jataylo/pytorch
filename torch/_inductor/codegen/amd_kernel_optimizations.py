from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from ..runtime.hints import DeviceProperties


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


def simplify_masked_modulo(
    source: str, *, _context: AMDCodegenSource | None = None
) -> str:
    """Remove a row-reduction modulo when the load mask proves its range."""
    tree = _context.tree if _context is not None else ast.parse(source)
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
        or (not isinstance(x.value, int))
        or (not isinstance(r.value, int))
        or (x.value <= 0)
        or (r.value <= 0)
        or (x.value * r.value >= 2**31)
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
        (
            ast.dump(assignments.get(name, ast.Constant(None)))
            != ast.dump(ast.parse(value, mode="eval").body)
            for name, value in required.items()
        )
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
            and (node.id in assignments)
        ):
            return resolve(assignments[node.id])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and (node.func.attr == "to")
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
            or (not isinstance(address.right, ast.BinOp))
            or (not isinstance(address.right.op, ast.Mod))
        ):
            continue
        mod = address.right
        if (
            ast.dump(mod.left) != ast.dump(index)
            or not isinstance(mod.right, ast.Constant)
            or (not isinstance(mod.right.value, int))
            or (mod.right.value <= 0)
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
                and (ast.dump(resolve(predicate.left)) == ast.dump(index))
                and isinstance(resolve(predicate.comparators[0]), ast.Constant)
                and (resolve(predicate.comparators[0]).value == mod.right.value)
            ):
                address.right = mod.left
                changed = True
                break
    if not changed:
        return source
    return _finish_source(tree, source, _context)

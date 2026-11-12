# Owner(s): ["module: inductor"]

import ast
import textwrap

import torch
from torch._inductor.codegen.amd_kernel_optimizations import (
    optimize_tiled_transpose_source,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def transpose_source(
    batch=1, rows=37, columns=19, inner=1, pattern="matrix", dtype="fp32"
):
    n = batch * rows * columns * inner
    c = rows * inner
    if pattern == "matrix":
        indices = f"x0 = xindex % {rows}\nx1 = xindex // {rows}"
        offset = f"x1 + {columns} * x0"
    elif pattern == "batch":
        indices = f"x0 = xindex % {rows}\nx1 = (xindex // {rows}) % {columns}\nx2 = xindex // {rows * columns}"
        offset = f"x1 + {columns} * x0 + {rows * columns} * x2"
    elif pattern == "collapsed":
        indices = f"x0 = xindex % {c}\nx1 = xindex // {c}"
        offset = f"{inner} * (x1 % {columns}) + {columns * inner} * (x0 // {inner}) + {columns * c} * (x1 // {columns}) + x0 % {inner}"
    else:
        indices = f"x0 = xindex % {inner}\nx1 = (xindex // {inner}) % {rows}\nx2 = (xindex // {c}) % {columns}\nx3 = xindex // {columns * c}"
        offset = f"x0 + {inner} * x2 + {columns * inner} * x1 + {columns * c} * x3"
    prefix = textwrap.dedent(
        f"\n        @triton_heuristics.pointwise(size_hints={{'x': 1024}}, triton_meta={{'signature': {{'in_ptr0': '*{dtype}', 'out_ptr0': '*{dtype}', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}}}}, inductor_meta={{'mutated_arg_names': []}})\n        @triton.jit\n        def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):\n            xnumel = {n}\n            xoffset = tl.program_id(0) * XBLOCK\n            xindex = xoffset + tl.arange(0, XBLOCK)[:]\n            xmask = xindex < xnumel\n    "
    )
    return (
        prefix
        + textwrap.indent(indices + "\n", "    ")
        + textwrap.indent(
            f"output_index = xindex\ntmp0 = tl.load(in_ptr0 + ({offset}), xmask).to(tl.float32)\ntl.store(out_ptr0 + output_index, tmp0, xmask)\n",
            "    ",
        )
    )


def grid2d_source(rows=37, columns=19):
    return textwrap.dedent(
        f"\n        @triton_heuristics.pointwise(size_hints={{'y': 32, 'x': 64}}, triton_meta={{'signature': {{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}}}}, inductor_meta={{'mutated_arg_names': [], 'amd_kernel_autotune': False}})\n        @triton.jit\n        def kernel(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):\n            ynumel = {columns}\n            xnumel = {rows}\n            yoffset = tl.program_id(1) * YBLOCK\n            yindex = yoffset + tl.arange(0, YBLOCK)[:, None]\n            ymask = yindex < ynumel\n            xoffset = tl.program_id(0) * XBLOCK\n            xindex = xoffset + tl.arange(0, XBLOCK)[None, :]\n            xmask = xindex < xnumel\n            x1 = xindex\n            y0 = yindex\n            tmp0 = tl.load(in_ptr0 + (y0 + {columns} * x1), xmask & ymask)\n            tl.store(out_ptr0 + (x1 + {rows} * y0), tmp0, xmask & ymask)\n    "
    )


@instantiate_parametrized_tests
class TestAMDTiledTransposeSource(TestCase):
    def test_grid2d_signature_and_masks_are_proven(self):
        source = grid2d_source()
        result = optimize_tiled_transpose_source(source)
        self.assertNotEqual(result, source)
        self.assertIn("'amd_transpose_shape': (1, 37, 19, 1)", result)
        self.assertIn("'amd_kernel_autotune': True", result)
        self.assertIn("tl.num_programs(0) * tl.num_programs(1)", result)
        fn = next(
            node for node in ast.parse(result).body if isinstance(node, ast.FunctionDef)
        )
        self.assertEqual(
            [arg.arg for arg in fn.args.args],
            [
                "in_ptr0",
                "out_ptr0",
                "ynumel",
                "xnumel",
                "YBLOCK",
                "XBLOCK",
                "AMD_TILE_M",
                "AMD_TILE_N",
                "AMD_READ_ORDER",
            ],
        )

    @parametrize("change", ["xmask", "ymask", "input", "output", "yindex", "count"])
    def test_unproven_grid2d_keeps_original(self, change):
        source = grid2d_source()
        if change == "xmask":
            source = source.replace("xmask & ymask", "ymask")
        elif change == "ymask":
            source = source.replace("xmask & ymask", "xmask")
        elif change == "input":
            source = source.replace("y0 + 19 * x1", "y0 + 20 * x1")
        elif change == "output":
            source = source.replace("x1 + 37 * y0", "x1 + 38 * y0")
        elif change == "yindex":
            source = source.replace("tl.program_id(1)", "tl.program_id(0)")
        else:
            source = source.replace("ynumel = 19", "ynumel = symbolic")
        self.assertEqual(optimize_tiled_transpose_source(source), source)

    @parametrize(
        "pattern,batch,inner",
        [("matrix", 1, 1), ("batch", 3, 1), ("collapsed", 2, 7), ("expanded", 2, 7)],
    )
    def test_static_permutation_is_proven(self, pattern, batch, inner):
        source = transpose_source(batch=batch, inner=inner, pattern=pattern)
        result = optimize_tiled_transpose_source(source)
        self.assertNotEqual(result, source)
        self.assertIn("'amd_native_family': 'tiled_transpose'", result)
        self.assertIn(f"'amd_transpose_shape': ({batch}, 37, 19, {inner})", result)
        self.assertIn("if AMD_TILE_M == 0:", result)
        self.assertIn("tl.num_programs(0)", result)
        self.assertNotIn("gluon", result.replace("'use_gluon': False", ""))
        tree = ast.parse(result)
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        self.assertEqual(
            [arg.arg for arg in fn.args.args][-3:],
            ["AMD_TILE_M", "AMD_TILE_N", "AMD_READ_ORDER"],
        )

    @parametrize(
        "change",
        [
            "dtype",
            "output",
            "input",
            "mask",
            "mutation",
            "count",
            "dynamic",
            "value",
            "duplicate",
            "overflow",
            "extra_load",
            "hidden_store",
            "index",
        ],
    )
    def test_unproven_source_keeps_original(self, change):
        source = transpose_source()
        if change == "dtype":
            source = source.replace("'out_ptr0': '*fp32'", "'out_ptr0': '*fp16'")
        elif change == "output":
            source = source.replace(
                "out_ptr0 + output_index", "out_ptr0 + 2 * output_index"
            )
        elif change == "input":
            source = source.replace("x1 + 19 * x0", "x1 + 20 * x0")
        elif change == "mask":
            source = source.replace("tmp0, xmask)", "tmp0, xindex < 17)")
        elif change == "mutation":
            source = source.replace(
                "'mutated_arg_names': []", "'mutated_arg_names': ['in_ptr0']"
            )
        elif change == "count":
            source = source.replace("xnumel = 703", "xnumel = 704")
        elif change == "dynamic":
            source = source.replace("xnumel = 703", "xnumel = dynamic_numel")
        elif change == "value":
            source = source.replace("tmp0, xmask)", "tmp0 + 1, xmask)")
        elif change == "duplicate":
            source = source.replace(
                "    output_index = xindex",
                "    x0 = xindex\n    output_index = xindex",
            )
        elif change == "overflow":
            source = source.replace("xnumel = 703", "xnumel = 2147483646")
        elif change == "extra_load":
            source += "    extra = tl.load(in_ptr0 + xindex, xmask)\n"
        elif change == "hidden_store":
            source += "    extra = tl.store(out_ptr0 + xindex, tmp0, xmask)\n"
        else:
            source = source.replace("x0 = xindex % 37", "x0 = (xindex + 1) % 37")
        self.assertEqual(optimize_tiled_transpose_source(source), source)

    def test_flat_fallback_normalizes_masks(self):
        source = transpose_source().replace(
            "xmask = xindex < xnumel", "xmask = tl.full([XBLOCK], True, tl.int1)[:]"
        )
        source = source.replace("), xmask)", "), None)").replace(
            "tmp0, xmask)", "tmp0, None)"
        )
        result = optimize_tiled_transpose_source(source)
        self.assertNotEqual(result, source)
        self.assertIn("xmask = xindex < 703", result)
        fn = next(
            node for node in ast.parse(result).body if isinstance(node, ast.FunctionDef)
        )
        for node in ast.walk(ast.Module(body=fn.body[0].body[0].body, type_ignores=[])):
            if isinstance(node, ast.Call) and ast.unparse(node.func) in (
                "tl.load",
                "tl.store",
            ):
                index = 1 if ast.unparse(node.func) == "tl.load" else 2
                self.assertEqual(ast.unparse(node.args[index]), "xmask")

    def test_cache_policy_is_preserved(self):
        source = transpose_source()
        result = optimize_tiled_transpose_source(source)
        self.assertIn("'amd_cache_policy': 'balanced'", result)

    @parametrize("shape", [(1, 37, 19, 1), (2, 5, 37, 7), (3, 1, 19, 1)])
    @parametrize("tile", [(16, 16), (16, 32), (32, 16)])
    @parametrize("programs", [1, 7, 100])
    def test_grid_stride_coordinates_cover_exact_permutation(
        self, shape, tile, programs
    ):
        batch, rows, columns, inner = shape
        m, n = tile
        tiles_m = (rows + m - 1) // m
        tiles_n = (columns * inner + n - 1) // n
        pairs = {}
        for program in range(programs):
            for tile_index in range(program, batch * tiles_m * tiles_n, programs):
                b = tile_index // (tiles_m * tiles_n)
                for i in range(m):
                    row = tile_index // tiles_n % tiles_m * m + i
                    for j in range(n):
                        col = tile_index % tiles_n * n + j
                        if row < rows and col < columns * inner:
                            source = (
                                b * rows * columns * inner + row * columns * inner + col
                            )
                            output = (
                                b * rows * columns * inner
                                + col // inner * rows * inner
                                + row * inner
                                + col % inner
                            )
                            self.assertNotIn(output, pairs)
                            pairs[output] = source
        expected = torch.arange(batch * rows * columns * inner).reshape(
            batch, rows, columns, inner
        )
        expected = expected.permute(0, 2, 1, 3).reshape(-1).tolist()
        self.assertEqual([pairs[index] for index in range(len(expected))], expected)


class TestAMDTiledTransposeDevice(TestCase):
    @onlyCUDA
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    @parametrize("shape", [(37, 19), (2, 5, 37, 7)])
    def test_compiled_transpose_uses_tiled_family(self, device, dtype, shape):
        if torch.version.hip is None:
            self.skipTest("AMD Triton specialization")
        from torch._inductor import config
        from torch._inductor.utils import run_and_get_code

        def transpose(x):
            if x.ndim == 2:
                return x.t().contiguous()
            return x.permute(0, 2, 1, 3).contiguous()

        x = torch.randn(shape, device=device, dtype=dtype)
        x.flatten()[:4] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), -0.0],
            device=device,
            dtype=dtype,
        )
        with config.patch(
            {"triton.amd_native_kernel_optimizations": True, "fx_graph_cache": False}
        ):
            actual, sources = run_and_get_code(
                torch.compile(transpose, fullgraph=True), x
            )
        self.assertEqual(actual, transpose(x), equal_nan=True)
        self.assertIn("'amd_native_family': 'tiled_transpose'", "\n".join(sources))

    @onlyCUDA
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    @parametrize(
        "pattern,batch,inner",
        [("matrix", 1, 1), ("grid2d", 1, 1), ("collapsed", 2, 7), ("expanded", 2, 64)],
    )
    @parametrize(
        "tile,read_order", [((0, 0), 0), ((0, 0), 1), ((16, 32), 0), ((32, 16), 0)]
    )
    def test_tails_nonfinite_and_flat_fallback(
        self, device, dtype, pattern, batch, inner, tile, read_order
    ):
        if torch.version.hip is None:
            self.skipTest("AMD Triton specialization")
        import triton

        from torch._inductor.codecache import PyCodeCache

        rows, columns = (37, 19) if pattern == "grid2d" else (5, 37)
        if pattern == "grid2d":
            source = grid2d_source(rows, columns)
            if dtype == torch.bfloat16:
                source = source.replace("*fp32", "*bf16")
        else:
            source = transpose_source(
                batch,
                rows,
                columns,
                inner,
                pattern,
                "bf16" if dtype == torch.bfloat16 else "fp32",
            )
        result = optimize_tiled_transpose_source(source)
        self.assertNotEqual(source, result)
        tree = ast.parse(result)
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        fn.decorator_list = [ast.parse("triton.jit", mode="eval").body]
        result = "import triton\nimport triton.language as tl\n" + ast.unparse(tree)
        kernel = PyCodeCache.load(result).kernel
        x = torch.randn(batch, rows, columns, inner, device=device, dtype=dtype)
        x.flatten()[:4] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), -0.0],
            device=device,
            dtype=dtype,
        )
        output = torch.full_like(x, -12345).reshape(-1)
        m, n = tile
        if pattern == "grid2d":
            kernel[triton.cdiv(rows, 16), triton.cdiv(columns, 16)](
                x,
                output,
                columns,
                rows,
                YBLOCK=16,
                XBLOCK=16,
                AMD_TILE_M=m,
                AMD_TILE_N=n,
                AMD_READ_ORDER=read_order,
                num_warps=4,
            )
        else:
            block = 256
            kernel[triton.cdiv(x.numel(), block),](
                x,
                output,
                x.numel(),
                XBLOCK=block,
                AMD_TILE_M=m,
                AMD_TILE_N=n,
                AMD_READ_ORDER=read_order,
                num_warps=4,
            )
        expected = x.permute(0, 2, 1, 3).contiguous().reshape(-1)
        self.assertEqual(output, expected, exact_dtype=True, equal_nan=True)
        self.assertEqual(
            torch.signbit(output[output == 0]), torch.signbit(expected[expected == 0])
        )


instantiate_device_type_tests(TestAMDTiledTransposeDevice, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()

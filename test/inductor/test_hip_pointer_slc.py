# Owner(s): ["module: inductor"]
import os
import unittest
from unittest import mock

from torch._inductor.runtime.hip_pointer_attrs import (
    clear_pointer_slc_overrides,
    cpp_distribution_tile_decl,
    DISTRIBUTION_TILE,
    first_tile,
    hip_pointer_get_attributes,
    IS_SLC_BIT,
    mark_pointer_slc,
    pointer_is_slc,
    prepare_distribution_tile_launcher,
    tile_cpp_expr,
)


SLC, PLAIN = 1, 2
PLACEHOLDER = 7  # caller-mode wrapper value; must never reach the kernel
TILE = 64


class _Tensor:
    def __init__(self, ptr):
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr


def _run_launcher(mode, ptr):
    """Build the launcher the way _gen_launcher_code does, call it, return the tile.

    Caller mode: the wrapper passes PLACEHOLDER, which the launcher overwrites.
    Launcher mode: the wrapper does not pass distribution_tile at all.
    """
    scope = {}
    def_args = ["x", DISTRIBUTION_TILE] if mode == "caller" else ["x"]
    def_args, lines = prepare_distribution_tile_launcher(
        scope,
        def_args,
        ["x", DISTRIBUTION_TILE],
        {DISTRIBUTION_TILE: mode},
        {"x": "*fp16", "XBLOCK": "constexpr"},
        TILE,
    )
    body = [*(f"    {ln}" for ln in lines), "    return distribution_tile"]
    exec("\n".join([f"def launcher({', '.join(def_args)}):", *body]), scope)
    extra = (PLACEHOLDER,) if mode == "caller" else ()
    return scope["launcher"](_Tensor(ptr), *extra)


class HipPointerSlcTest(unittest.TestCase):
    def setUp(self):
        mark_pointer_slc(SLC, True)
        mark_pointer_slc(PLAIN, False)

    def tearDown(self):
        clear_pointer_slc_overrides()

    def test_attribute(self):
        self.assertFalse(hip_pointer_get_attributes(0).is_slc)
        attrs = hip_pointer_get_attributes(SLC)
        self.assertTrue(attrs.is_slc)
        self.assertTrue(attrs.allocation_flags & IS_SLC_BIT)
        self.assertFalse(hip_pointer_get_attributes(PLAIN).is_slc)
        self.assertFalse(pointer_is_slc(None))
        self.assertTrue(pointer_is_slc(_Tensor(SLC)))

    def test_failed_lookup_is_not_slc(self):
        # No HIP runtime, or an address HIP rejects, must not raise at launch.
        env = {k: v for k, v in os.environ.items() if k != "TORCHINDUCTOR_FAKE_SLC"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(pointer_is_slc(0x1234))

    def test_launcher_modes(self):
        for mode in ("launcher", "caller"):
            with self.subTest(mode=mode):
                self.assertEqual(_run_launcher(mode, SLC), TILE)
                self.assertEqual(_run_launcher(mode, PLAIN), 0)

    def test_first_tile(self):
        pointwise = {"x": "*fp16", "XBLOCK": "constexpr", "R0_BLOCK": "constexpr"}
        self.assertEqual(first_tile(pointwise, {"XBLOCK": 128, "R0_BLOCK": 32}), 128)
        # Templates: no tiling name in the signature, BLOCK_M comes from config_args.
        template = {"a": "*fp16", "b": "*fp16"}
        meta = {"EVEN_K": True, "BLOCK_M": 32, "BLOCK_N": 64}
        self.assertEqual(first_tile(template, meta), 32)
        self.assertEqual(first_tile({"x": "*fp16"}, {"num_warps": 4}), 0)

    def test_cpp_snippets(self):
        decl = cpp_distribution_tile_decl(["var_0"], "i64", "64")
        self.assertIn("static_cast<int64_t>(64)", decl)
        self.assertIn(f"{IS_SLC_BIT}u", decl)
        self.assertIn("inductor_ptr_is_slc(var_0)", decl)
        self.assertIn("#else", decl)

        signature = {"x": "*fp16", "XBLOCK": "constexpr"}
        self.assertEqual(
            tile_cpp_expr(signature, config={"XBLOCK": 128, "R0_BLOCK": 16}), "128"
        )
        self.assertEqual(
            tile_cpp_expr(signature, lazy_kernel="kernel"), "kernel_result.xblocks[0]"
        )
        self.assertEqual(
            tile_cpp_expr(
                {"a": "*fp16"},
                lazy_kernel="kernel",
                inductor_meta={"config_args": {"BLOCK_M": 128}},
            ),
            "128",
        )

        def grids(*blocks):
            return {
                "grid_type": "PrecomputedGrid",
                "precomputed_grids": [{"config": {"XBLOCK": b}} for b in blocks],
            }

        mixed = tile_cpp_expr(signature, lazy_kernel="k", inductor_meta=grids(32, 64))
        self.assertIn("case 0: return 32;", mixed)
        self.assertIn("case 1: return 64;", mixed)
        same = tile_cpp_expr(signature, lazy_kernel="k", inductor_meta=grids(32, 32))
        self.assertEqual(same, "32")


if __name__ == "__main__":
    unittest.main()

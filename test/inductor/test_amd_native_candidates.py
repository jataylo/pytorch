# Owner(s): ["module: inductor"]

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._inductor.runtime.amd_autotune import _neighbors, tune
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.runtime.triton_compat import Config
from torch._inductor.runtime.triton_heuristics import (
    _amd_native_seed_configs,
    _amd_native_union_configs,
    _amd_reduction_configs,
    _persistent_reduction_configs,
    persistent_reduction,
    pointwise,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def config_key(candidate):
    return (
        *sorted(candidate.kwargs.items()),
        candidate.num_warps,
        candidate.num_stages,
    )


@instantiate_parametrized_tests
class TestAMDNativeCandidates(TestCase):
    def setUp(self):
        super().setUp()
        self.device = DeviceProperties(
            type="hip",
            index=0,
            multi_processor_count=8,
            cc="gfx950",
            warp_size=64,
            max_threads_per_block=1024,
            regs_per_multiprocessor=131072,
        )

    @parametrize("family", ["add", "broadcast", "flat_pointwise", "transpose"])
    def test_stock_pointwise_candidates_are_retained(self, family):
        signature = {
            "in_ptr0": "*fp32",
            "out_ptr0": "*fp32",
            "xnumel": "i32",
            "XBLOCK": "constexpr",
            "AMD_UNROLL": "constexpr",
        }
        if family == "transpose":
            signature["AMD_READ_ORDER"] = "constexpr"
        triton_meta = {"device": self.device, "signature": signature}
        options = {"max_autotune_pointwise": True}
        stock = pointwise(
            {"x": 8192}, triton_meta, inductor_meta=options, return_configs=True
        )
        with patch(
            "torch._inductor.runtime.triton_heuristics._amd_layout_configs",
            side_effect=AssertionError("native seeds used Gluon layouts"),
        ):
            configs = pointwise(
                {"x": 8192},
                triton_meta,
                inductor_meta=dict(options, amd_native_family=family),
                return_configs=True,
            )
        keys = {config_key(candidate) for candidate in configs}
        self.assertEqual(len(keys), len(configs))
        for candidate in stock:
            keywords = dict(candidate.kwargs, AMD_UNROLL=1)
            if family == "transpose":
                keywords["AMD_READ_ORDER"] = 0
            expected = Config(
                keywords, num_warps=candidate.num_warps, num_stages=candidate.num_stages
            )
            self.assertIn(config_key(expected), keys)
        self.assertTrue(all("AMD_UNROLL" in candidate.kwargs for candidate in configs))

    def test_stock_rows_are_retained(self):
        hints = {"x": 128, "r0_": 128}
        triton_meta = {
            "device": self.device,
            "signature": {"in_ptr0": "*fp32", "XBLOCK": "constexpr"},
        }
        meta = {
            "amd_native_family": "row_reduction",
            "amd_row_count": 128,
            "amd_reduction_extent": 128,
            "max_autotune_pointwise": True,
        }
        stock = _persistent_reduction_configs(
            hints, inductor_meta=meta, triton_meta=triton_meta
        )
        with patch(
            "torch._inductor.runtime.triton_heuristics._amd_layout_configs",
            side_effect=AssertionError("native seeds used Gluon layouts"),
        ):
            configs = persistent_reduction(
                hints, triton_meta=triton_meta, inductor_meta=meta, return_configs=True
            )
        keys = {config_key(candidate) for candidate in configs}
        for candidate in stock:
            self.assertIn(config_key(candidate), keys)

    @parametrize("rows", [37, None])
    @parametrize("tail_safe", [False, True])
    def test_row_tail_candidates_require_proof(self, rows, tail_safe):
        meta = {
            "amd_native_family": "row_reduction",
            "amd_row_count": rows,
            "amd_reduction_extent": 128,
            "amd_row_tail_safe": tail_safe,
        }
        configs = persistent_reduction(
            {"x": 64, "r0_": 128},
            triton_meta={"device": self.device},
            inductor_meta=meta,
            return_configs=True,
        )
        blocks = {candidate.kwargs["XBLOCK"] for candidate in configs}
        self.assertTrue(blocks)
        if tail_safe:
            self.assertIn(8, blocks)
        else:
            self.assertEqual(blocks, {1})
        tuner = SimpleNamespace(
            fn=SimpleNamespace(arg_names=["xnumel", "r0_numel"]),
            device_props=self.device,
            inductor_meta=meta,
        )
        neighbors = list(
            _neighbors(
                tuner, Config({"XBLOCK": 1}, num_warps=2, num_stages=1), (37, 128), {}
            )
        )
        self.assertEqual(
            any(candidate.kwargs["XBLOCK"] == 2 for candidate in neighbors), tail_safe
        )

    def test_seed_geometry_and_dtype_are_independent(self):
        meta = {"amd_native_family": "flat_pointwise"}
        blocks = {}
        for dtype in ("*fp32", "*bf16"):
            triton_meta = {"device": self.device, "signature": {"in_ptr0": dtype}}
            configs = _amd_native_seed_configs({"x": 8192}, triton_meta, meta)
            blocks[dtype] = max(candidate.kwargs["XBLOCK"] for candidate in configs)
        self.assertGreater(blocks["*bf16"], blocks["*fp32"])
        tiny = _amd_native_seed_configs({"x": 128}, {"device": self.device}, meta)
        self.assertTrue(tiny)
        self.assertTrue(all(candidate.kwargs["XBLOCK"] <= 128 for candidate in tiny))

    @parametrize("wave", [32, 64])
    def test_seeds_obey_device_limits(self, wave):
        device = self.device._replace(
            warp_size=wave, max_threads_per_block=128, regs_per_multiprocessor=32768
        )
        for family, hints in (
            ("flat_pointwise", {"x": 8192}),
            ("row_reduction", {"x": 128, "r0_": 128}),
        ):
            configs = _amd_native_seed_configs(
                hints, {"device": device}, {"amd_native_family": family}
            )
            self.assertTrue(configs)
            for candidate in configs:
                self.assertLessEqual(candidate.num_warps * wave, 128)
                self.assertLessEqual(
                    candidate.kwargs["XBLOCK"] * hints.get("r0_", 1), 8192
                )

    def test_long_scan_keeps_sixteen_warp_seed(self):
        configs = _amd_native_seed_configs(
            {"x": 64, "r0_": 32768},
            {"device": self.device},
            {"amd_native_family": "row_reduction", "amd_confirm_tuning": True},
        )
        self.assertIn(16, [candidate.num_warps for candidate in configs])

    @parametrize(
        "inputs, outputs, expected",
        [
            ((torch.float32, torch.int64), (torch.float32,), 4),
            ((torch.float32,), (torch.int64,), 8),
            ((torch.int64,), (torch.int64,), 8),
            ((torch.float64, torch.int64), (torch.float64,), 8),
        ],
    )
    def test_reduction_value_width_ignores_auxiliary_indices(
        self, inputs, outputs, expected
    ):
        from torch._inductor.codegen.amd_kernel_optimizations import (
            amd_reduction_value_bytes,
        )

        self.assertEqual(amd_reduction_value_bytes(inputs, outputs), expected)

    @parametrize("value_bytes", [4, 8])
    def test_mixed_dtype_row_pressure_uses_reduction_values(self, value_bytes):
        hints = {"x": 1, "r0_": 32768}
        triton_meta = {
            "device": self.device,
            "signature": {
                "in_ptr0": "*fp32",
                "in_ptr1": "*i64",
                "out_ptr0": "*fp32",
                "XBLOCK": "constexpr",
            },
        }
        meta = {
            "amd_native_family": "row_reduction",
            "amd_row_count": 1,
            "amd_confirm_tuning": True,
            "num_load": 2,
            "num_reduction": 1,
            "amd_reduction_value_bytes": value_bytes,
        }
        seeds = _amd_native_seed_configs(hints, triton_meta, meta)
        self.assertEqual(
            any(candidate.num_warps == 16 for candidate in seeds), value_bytes == 4
        )
        stock = Config({"XBLOCK": 1}, num_warps=1, num_stages=1)
        configs = _amd_native_union_configs([stock], seeds, hints, triton_meta, meta)
        self.assertIn(
            config_key(stock), {config_key(candidate) for candidate in configs}
        )

    def test_reduction_value_width_does_not_change_flat_seeds(self):
        triton_meta = {
            "device": self.device,
            "signature": {"in_ptr0": "*fp32", "in_ptr1": "*i64"},
        }
        meta = {"amd_native_family": "flat_pointwise", "amd_reduction_value_bytes": 4}
        narrow = _amd_native_seed_configs({"x": 8192}, triton_meta, meta)
        wide = _amd_native_seed_configs(
            {"x": 8192}, triton_meta, dict(meta, amd_reduction_value_bytes=8)
        )
        self.assertEqual(
            [config_key(candidate) for candidate in narrow],
            [config_key(candidate) for candidate in wide],
        )

    def test_row_seed_pressure_does_not_prune_stock(self):
        hints = {"x": 256, "r0_": 1024}
        triton_meta = {
            "device": self.device,
            "signature": {"in_ptr0": "*fp32", "XBLOCK": "constexpr"},
        }
        meta = {
            "amd_native_family": "row_reduction",
            "amd_row_count": 256,
            "num_load": 3,
            "num_reduction": 2,
        }
        seeds = _amd_native_seed_configs(hints, triton_meta, meta)
        keys = {config_key(candidate) for candidate in seeds}
        expensive = Config({"XBLOCK": 32}, num_warps=1, num_stages=1)
        self.assertNotIn(config_key(expensive), keys)
        for rows, warps in ((1, 1), (2, 4)):
            self.assertIn(
                config_key(Config({"XBLOCK": rows}, num_warps=warps, num_stages=1)),
                keys,
            )
        configs = _amd_native_union_configs(
            [expensive], seeds, hints, triton_meta, meta
        )
        self.assertIn(
            config_key(expensive), {config_key(candidate) for candidate in configs}
        )

    @parametrize("resource_aware", [False, True])
    def test_generic_persistent_extra_pressure_keeps_stock(self, resource_aware):
        hints = {"x": 256, "r0_": 2048}
        meta = {
            "amd_reduction_tuning": True,
            "amd_resource_aware_tuning": resource_aware,
            "num_load": 7,
            "num_reduction": 3,
        }
        expensive = Config({"XBLOCK": 16}, num_warps=1, num_stages=1)
        extras = _amd_reduction_configs([], hints, meta, self.device, persistent=True)
        self.assertEqual(
            config_key(expensive) in {config_key(candidate) for candidate in extras},
            not resource_aware,
        )
        stock = [expensive, Config({"XBLOCK": 1}, num_warps=8, num_stages=1)]
        configs = _amd_reduction_configs(
            stock, hints, meta, self.device, persistent=True
        )
        keys = {config_key(candidate) for candidate in configs}
        for candidate in stock:
            self.assertIn(config_key(candidate), keys)

    @parametrize("grid2d", [False, True])
    def test_tiled_transpose_retains_true_flat_baseline(self, grid2d):
        triton_meta = {
            "device": self.device,
            "signature": {
                "in_ptr0": "*fp32",
                "XBLOCK": "constexpr",
                "AMD_TILE_M": "constexpr",
                "AMD_TILE_N": "constexpr",
                "AMD_READ_ORDER": "constexpr",
            },
        }
        hints = {"y": 128, "x": 64} if grid2d else {"x": 16384}
        if grid2d:
            triton_meta["signature"].update(YBLOCK="constexpr", ynumel="i32")
        stock = pointwise(hints, triton_meta, return_configs=True)
        meta = {
            "amd_native_family": "tiled_transpose",
            "amd_transpose_shape": (2, 37, 127, 1),
        }
        configs = pointwise(hints, triton_meta, inductor_meta=meta, return_configs=True)
        keys = {config_key(candidate) for candidate in configs}
        for candidate in stock:
            expected = Config(
                dict(candidate.kwargs, AMD_TILE_M=0, AMD_TILE_N=0, AMD_READ_ORDER=0),
                num_warps=candidate.num_warps,
                num_stages=candidate.num_stages,
            )
            self.assertIn(config_key(expected), keys)
            alternate = Config(
                dict(expected.kwargs, AMD_READ_ORDER=1),
                num_warps=expected.num_warps,
                num_stages=expected.num_stages,
            )
            self.assertIn(config_key(alternate), keys)
        tiled = [candidate for candidate in configs if candidate.kwargs["AMD_TILE_M"]]
        self.assertTrue(tiled)
        if not grid2d:
            self.assertTrue(
                any(
                    candidate.kwargs["XBLOCK"] == 1024
                    and candidate.num_warps == 2
                    and candidate.kwargs["AMD_READ_ORDER"] == 1
                    and not candidate.kwargs["AMD_TILE_M"]
                    for candidate in configs
                )
            )
        for candidate in tiled:
            self.assertEqual(candidate.kwargs["AMD_READ_ORDER"], 0)
            self.assertEqual(
                candidate.kwargs["XBLOCK"] * candidate.kwargs.get("YBLOCK", 1),
                candidate.kwargs["AMD_TILE_M"] * candidate.kwargs["AMD_TILE_N"],
            )
            if grid2d:
                self.assertEqual(
                    candidate.kwargs["YBLOCK"], candidate.kwargs["AMD_TILE_N"]
                )
        tuner = SimpleNamespace(
            fn=SimpleNamespace(
                arg_names=["ynumel", "xnumel"] if grid2d else ["xnumel"]
            ),
            device_props=self.device,
            inductor_meta=meta,
        )
        flat_input_order = Config(
            dict(configs[0].kwargs, AMD_READ_ORDER=1),
            num_warps=configs[0].num_warps,
            num_stages=configs[0].num_stages,
        )
        for config in (configs[0], flat_input_order, tiled[0]):
            args = (127, 37) if grid2d else (2 * 37 * 127,)
            neighbors = list(_neighbors(tuner, config, args, {}))
            self.assertTrue(neighbors)
            if not config.kwargs["AMD_TILE_M"]:
                self.assertTrue(
                    any(
                        not candidate.kwargs["AMD_TILE_M"]
                        and candidate.kwargs["AMD_READ_ORDER"]
                        != config.kwargs["AMD_READ_ORDER"]
                        for candidate in neighbors
                    )
                )
            for candidate in neighbors:
                self.assertEqual(
                    bool(candidate.kwargs["AMD_TILE_M"]),
                    bool(candidate.kwargs["AMD_TILE_N"]),
                )
                if candidate.kwargs["AMD_TILE_M"]:
                    self.assertEqual(candidate.kwargs["AMD_READ_ORDER"], 0)
                if grid2d and candidate.kwargs["AMD_TILE_M"]:
                    self.assertEqual(
                        candidate.kwargs["YBLOCK"], candidate.kwargs["AMD_TILE_N"]
                    )
                    self.assertEqual(
                        candidate.kwargs["XBLOCK"] * candidate.kwargs["YBLOCK"],
                        candidate.kwargs["AMD_TILE_M"] * candidate.kwargs["AMD_TILE_N"],
                    )

    def test_spill_pressure_reorders_without_pruning(self):
        tuner = SimpleNamespace(
            fn=SimpleNamespace(arg_names=["xnumel"]),
            device_props=self.device,
            inductor_meta={"amd_native_family": "flat_pointwise"},
        )
        config = Config({"XBLOCK": 1024}, num_warps=2, num_stages=1)
        regular = list(_neighbors(tuner, config, (8192,), {}))
        pressure = list(_neighbors(tuner, config, (8192,), {}, n_regs=256, n_spills=4))
        self.assertEqual(
            {config_key(candidate) for candidate in regular},
            {config_key(candidate) for candidate in pressure},
        )
        self.assertLess(pressure[0].kwargs["XBLOCK"], config.kwargs["XBLOCK"])
        bigger_warp = config_key(Config({"XBLOCK": 1024}, num_warps=4, num_stages=1))
        self.assertLess(
            [config_key(candidate) for candidate in pressure].index(bigger_warp),
            [config_key(candidate) for candidate in regular].index(bigger_warp),
        )

    def test_resource_and_timing_diagnostics(self):
        launcher = SimpleNamespace(
            config=Config({"XBLOCK": 128}, num_warps=1, num_stages=1),
            n_regs=32,
            n_spills=2,
            shared=1024,
        )

        class Launcher:
            config = launcher.config
            n_regs = launcher.n_regs
            n_spills = launcher.n_spills
            shared = launcher.shared

        choice = Launcher()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.jsonl"
            tuner = SimpleNamespace(
                fn=SimpleNamespace(arg_names=[]),
                device_props=self.device,
                precompile_time_taken_ns=1234,
                launchers=[choice],
                inductor_meta={"amd_autotune_diagnostics_path": str(path)},
                coordesc_tuner=SimpleNamespace(
                    cache_benchmark_result=lambda *args: None
                ),
            )
            with (
                patch("torch.cuda.mem_get_info", return_value=(1 << 40, 1 << 40)),
                patch(
                    "torch._inductor.runtime.amd_autotune._neighbors", return_value=()
                ),
                patch(
                    "torch._inductor.runtime.amd_autotune.benchmark_configs",
                    return_value=({choice: 0.5}, {choice: [0.5]}),
                ),
            ):
                tune(tuner, (), {})
            report = json.loads(path.read_text())
        self.assertEqual(report["initial_candidates"], 1)
        self.assertEqual(report["measured_candidates"], 1)
        self.assertEqual(report["reported_precompile_time_ns"], 1234)
        self.assertEqual(report["neighbor_compile_time_ns"], 0)
        self.assertEqual(report["accounted_compile_time_ns"], 1234)
        self.assertGreaterEqual(report["tuning_time_ns"], 0)
        candidate = report["candidates"][0]
        self.assertEqual(
            (candidate["n_regs"], candidate["n_spills"], candidate["shared"]),
            (32, 2, 1024),
        )


if __name__ == "__main__":
    run_tests()

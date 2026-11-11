# Owner(s): ["module: inductor"]

from types import SimpleNamespace

from torch._inductor.runtime.amd_autotune import _neighbors
from torch._inductor.runtime.hints import AMDConfigLimits, DeviceProperties
from torch._inductor.runtime.triton_compat import Config
from torch._inductor.runtime.triton_heuristics import (
    _amd_layout_configs,
    _amd_reduction_configs,
    persistent_reduction,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestAMDHardwareConfigs(TestCase):
    @parametrize("wave", [32, 64])
    @parametrize("max_threads", [128, 1024])
    @parametrize("family", ["gelu_cdf", "transpose", "flat_pointwise"])
    def test_pointwise_block_limits(self, wave, max_threads, family):
        device = DeviceProperties(
            type="hip",
            index=0,
            multi_processor_count=1,
            cc="gfx950",
            warp_size=wave,
            max_threads_per_block=max_threads,
            regs_per_multiprocessor=65536,
        )
        configs = _amd_layout_configs(family, device)
        self.assertTrue(configs)
        for config in configs:
            self.assertLessEqual(config.num_warps * wave, max_threads)
            self.assertLessEqual(config.kwargs["XBLOCK"], 16384)

    @parametrize("wave", [32, 64])
    @parametrize("max_threads", [128, 1024])
    def test_row_layout_and_register_limits(self, wave, max_threads):
        device = DeviceProperties(
            type="hip",
            index=0,
            multi_processor_count=1,
            cc="gfx950",
            warp_size=wave,
            max_threads_per_block=max_threads,
            regs_per_multiprocessor=65536,
        )
        meta = {
            "amd_gluon_family": "row_reduction",
            "amd_reduction_extent": 8192,
            "amd_row_count": 128,
            "amd_confirm_tuning": True,
            "amd_outer_row": True,
        }
        configs = persistent_reduction(
            {"x": 128, "r0_": 8192},
            triton_meta={"device": device},
            inductor_meta=meta,
            return_configs=True,
        )
        self.assertTrue(configs)
        for config in configs:
            self.assertLessEqual(config.num_warps * wave, max_threads)
            self.assertLessEqual(config.kwargs["AMD_LANES"], wave)
            self.assertEqual(wave % config.kwargs["AMD_LANES"], 0)
            self.assertLessEqual(config.kwargs["XBLOCK"] * 8192, 16384)

    @parametrize("persistent", [False, True])
    def test_reduction_additions_respect_limits(self, persistent):
        device = DeviceProperties(
            type="hip",
            index=0,
            multi_processor_count=1,
            cc="gfx1100",
            warp_size=32,
            max_threads_per_block=128,
            regs_per_multiprocessor=8192,
        )
        configs = _amd_reduction_configs(
            [],
            {"x": 128, "r0_": 1024},
            {"amd_reduction_tuning": True, "amd_confirm_tuning": True},
            device,
            persistent=persistent,
        )
        self.assertTrue(configs)
        for config in configs:
            self.assertLessEqual(config.num_warps * 32, 128)
            rblock = config.kwargs.get("R0_BLOCK", 1024)
            self.assertLessEqual(config.kwargs["XBLOCK"] * rblock, 2048)

    @parametrize("wave", [32, 64])
    def test_neighbors_check_warp_and_tile_changes(self, wave):
        device = DeviceProperties(
            type="hip",
            index=0,
            multi_processor_count=1,
            cc="gfx950",
            warp_size=wave,
            max_threads_per_block=128,
            regs_per_multiprocessor=32768,
        )
        tuner = SimpleNamespace(
            fn=SimpleNamespace(arg_names=["xnumel", "r0_numel"]),
            device_props=device,
            inductor_meta={
                "use_gluon": True,
                "amd_gluon_family": "row_reduction",
                "amd_confirm_tuning": True,
            },
        )
        config = Config(
            {"XBLOCK": 1, "AMD_LANES": wave, "AMD_VEC": 1, "AMD_WARPS": 2},
            num_warps=2,
            num_stages=1,
        )
        neighbors = list(_neighbors(tuner, config, (128, 8192), {}))
        self.assertTrue(neighbors)
        for candidate in neighbors:
            self.assertLessEqual(candidate.num_warps * wave, 128)
            self.assertEqual(candidate.kwargs["AMD_WARPS"], candidate.num_warps)
            self.assertLessEqual(candidate.kwargs["AMD_LANES"], wave)
            self.assertLessEqual(candidate.kwargs["XBLOCK"] * 8192, 8192)

    def test_cdna_baseline_candidates_are_preserved(self):
        device = DeviceProperties(
            type="hip",
            index=0,
            multi_processor_count=256,
            cc="gfx950",
            warp_size=64,
            max_threads_per_block=1024,
            regs_per_multiprocessor=131072,
        )
        self.assertEqual(AMDConfigLimits.from_device(device), (64, 16, 32768))
        for family, count in (
            ("gelu_cdf", 6),
            ("transpose", 10),
            ("flat_pointwise", 18),
        ):
            self.assertEqual(len(_amd_layout_configs(family, device)), count)


if __name__ == "__main__":
    run_tests()

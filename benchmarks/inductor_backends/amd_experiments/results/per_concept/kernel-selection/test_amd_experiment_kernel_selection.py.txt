# Owner(s): ["module: inductor"]

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from torch._inductor.runtime.amd_autotune import _confirmed_choice, _neighbors, tune
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.runtime.triton_compat import Config
from torch._inductor.runtime.triton_heuristics import _amd_reduction_configs
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
    @parametrize(
        "candidate, expected_reason",
        [
            ([0.9] * 9, "confirmed_paired_gain"),
            ([0.995] * 9, "baseline_within_variation"),
            (
                [0.8, 1.1, 0.99, 0.8, 1.1, 0.99, 0.8, 1.1, 0.99],
                "baseline_within_variation",
            ),
        ],
    )
    def test_confirmation_requires_repeatable_gain(self, candidate, expected_reason):
        baseline, challenger = (object(), object())
        times = {baseline: 1.0, challenger: sorted(candidate)[len(candidate) // 2]}
        samples = {baseline: [1.0] * 9, challenger: candidate}
        selected, reason, _ = _confirmed_choice(times, samples, baseline)
        self.assertEqual(reason, expected_reason)
        self.assertIs(
            selected, challenger if reason == "confirmed_paired_gain" else baseline
        )

    def test_baseline_survives_noisy_initial_shortlist(self):
        class Launcher:
            def __init__(self, block):
                self.config = Config({"XBLOCK": block}, num_warps=1, num_stages=1)

        launchers = [Launcher(2**i) for i in range(8)]
        baseline = launchers[-1]
        measured = []

        def benchmark(tuner, choices, *args, **kwargs):
            measured.append(list(choices))
            if len(measured) == 1:
                return (
                    {c: float(i + 1) for i, c in enumerate(choices)},
                    {c: [float(i + 1)] * 9 for i, c in enumerate(choices)},
                )
            times = {
                c: 1.0 if c is baseline else 1.02 if c is launchers[-2] else 0.995
                for c in choices
            }
            return (times, {c: [value] * 9 for c, value in times.items()})

        tuner = SimpleNamespace(
            fn=SimpleNamespace(arg_names=[]),
            device_props=SimpleNamespace(index=0),
            launchers=launchers,
            inductor_meta={
                "amd_baseline_configs": [
                    {"XBLOCK": 64, "num_warps": 1, "num_stages": 1},
                    {"XBLOCK": 128, "num_warps": 1, "num_stages": 1},
                ]
            },
            coordesc_tuner=SimpleNamespace(cache_benchmark_result=lambda *args: None),
        )
        with (
            patch("torch.cuda.mem_get_info", return_value=(1 << 40, 1 << 40)),
            patch("torch._inductor.runtime.amd_autotune._neighbors", return_value=()),
            patch(
                "torch._inductor.runtime.amd_autotune.benchmark_configs",
                side_effect=benchmark,
            ),
        ):
            result = tune(tuner, (), {})
        self.assertIn(baseline, measured[-1])
        self.assertIn(launchers[-2], measured[-1])
        self.assertEqual(list(result), [baseline])

    @parametrize("confirm", [False, True])
    def test_generic_reduction_confirmation_budget(self, confirm):
        class Launcher:
            def __init__(self, warps):
                self.config = Config(
                    {"XBLOCK": 1, "R0_BLOCK": 128}, num_warps=warps, num_stages=1
                )

        launchers = [Launcher(4), Launcher(8)]
        meta = {"amd_reduction_tuning": True, "amd_confirm_tuning": confirm}
        _amd_reduction_configs(
            [c.config for c in launchers],
            {"x": 128, "r0_": 1024},
            meta,
            self.device,
            persistent=False,
        )
        self.assertEqual(len(meta["amd_baseline_configs"]), len(launchers))
        tuner = SimpleNamespace(
            fn=SimpleNamespace(arg_names=[]),
            device_props=self.device,
            launchers=launchers,
            inductor_meta=meta,
            coordesc_tuner=SimpleNamespace(cache_benchmark_result=lambda *args: None),
        )
        times = {c: float(i + 1) for i, c in enumerate(launchers)}
        with (
            patch("torch.cuda.mem_get_info", return_value=(1 << 40, 1 << 40)),
            patch("torch._inductor.runtime.amd_autotune._neighbors", return_value=()),
            patch(
                "torch._inductor.runtime.amd_autotune.benchmark_configs",
                return_value=(times, {c: [v] * 9 for c, v in times.items()}),
            ) as benchmark,
        ):
            tune(tuner, (), {})
        self.assertEqual(benchmark.call_count, 2)

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

# Owner(s): ["module: inductor"]

import gc
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._inductor.runtime.amd_autotune import benchmark_configs
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


@instantiate_parametrized_tests
class TestAMDAutotuneCaptureState(TestCase):
    @parametrize("gc_enabled", [False, True])
    @parametrize("failure", [None, "enter", "launcher", "exit"])
    def test_gc_state_restored_after_capture(self, gc_enabled, failure):
        original = gc.isenabled()
        capturing = False
        capture_states = []

        class Capture:
            def __enter__(self):
                nonlocal capturing
                capture_states.append(gc.isenabled())
                if failure == "enter":
                    raise RuntimeError("capture enter failure")
                capturing = True

            def __exit__(self, *args):
                nonlocal capturing
                capturing = False
                if failure == "exit":
                    raise RuntimeError("capture exit failure")

        def launcher(stream=None):
            if capturing and failure == "launcher":
                raise RuntimeError("capture launcher failure")

        tuner = SimpleNamespace(
            fn=SimpleNamespace(arg_names=[]),
            mutated_arg_names=(),
            inductor_meta={"amd_graph_tuning": True},
            device_props=SimpleNamespace(index=0),
            get_device_interface=lambda: SimpleNamespace(
                get_raw_stream=lambda index: 0
            ),
            reset_to_zero_args=lambda: None,
        )
        event = SimpleNamespace(record=lambda: None, elapsed_time=lambda end: 1.0)
        try:
            gc.enable() if gc_enabled else gc.disable()
            with ExitStack() as stack:
                stack.enter_context(
                    patch(
                        "torch.empty", return_value=SimpleNamespace(zero_=lambda: None)
                    )
                )
                stack.enter_context(patch("torch.cuda.Event", return_value=event))
                stack.enter_context(patch("torch.cuda.synchronize"))
                stack.enter_context(patch("torch.cuda._sleep"))
                stack.enter_context(
                    patch(
                        "torch.cuda.CUDAGraph",
                        return_value=SimpleNamespace(replay=lambda: None),
                    )
                )
                stack.enter_context(
                    patch("torch.cuda.graph", side_effect=lambda graph: Capture())
                )
                if failure is None:
                    benchmark_configs(tuner, [launcher], (), {}, rounds=1, repeats=1)
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, f"capture {failure} failure"
                    ):
                        benchmark_configs(
                            tuner, [launcher], (), {}, rounds=1, repeats=1
                        )
            self.assertEqual(capture_states, [False])
            self.assertEqual(gc.isenabled(), gc_enabled)
        finally:
            gc.enable() if original else gc.disable()


class TestAMDDeferredCaptureFinalizers(TestCase):
    @onlyCUDA
    def test_cycles_finalize_outside_capture(self, device):
        if torch.version.hip is None:
            self.skipTest("AMD capture lifetime regression")
        finalized = []
        capture_calls = []

        class Cycle:
            def __init__(self):
                self.cycle = self
                self.created_during_capture = torch.cuda.is_current_stream_capturing()

            def __del__(self):
                finalized.append(
                    (
                        self.created_during_capture,
                        torch.cuda.is_current_stream_capturing(),
                    )
                )

        def launcher(value, output, stream=None):
            output.copy_(value + 1)
            capture_calls.append(torch.cuda.is_current_stream_capturing())
            for _ in range(64):
                Cycle()

        value = torch.arange(127, device=device, dtype=torch.float32)
        output = torch.empty_like(value)
        index = value.device.index
        tuner = SimpleNamespace(
            fn=SimpleNamespace(arg_names=["in_ptr0", "out_ptr0"]),
            mutated_arg_names=(),
            inductor_meta={"amd_graph_tuning": True},
            device_props=SimpleNamespace(index=index),
            get_device_interface=lambda: SimpleNamespace(
                get_raw_stream=lambda index: torch.cuda.current_stream(
                    index
                ).cuda_stream
            ),
            reset_to_zero_args=lambda *args: None,
        )
        original_enabled = gc.isenabled()
        original_thresholds = gc.get_threshold()
        gc.collect()
        try:
            gc.enable()
            gc.set_threshold(20, 20, 20)
            benchmark_configs(
                tuner, [launcher], (value, output), {}, rounds=1, repeats=3
            )
            self.assertTrue(gc.isenabled())
            gc.collect()
            self.assertTrue(any(capture_calls))
            self.assertTrue(any((created for created, captured in finalized)))
            self.assertTrue(all((not captured for created, captured in finalized)))
            self.assertEqual(output, value + 1)
        finally:
            gc.set_threshold(*original_thresholds)
            gc.enable() if original_enabled else gc.disable()


instantiate_device_type_tests(
    TestAMDDeferredCaptureFinalizers, globals(), only_for="cuda"
)
if __name__ == "__main__":
    run_tests()

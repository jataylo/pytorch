# Owner(s): ["module: inductor"]

import importlib.util
from pathlib import Path

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/inductor_backends/amd_benchmark_protocol.py"
)
SPEC = importlib.util.spec_from_file_location("amd_benchmark_protocol", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load GPU benchmark protocol from {SOURCE}")
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


class TestAMDBenchmarkProtocol(TestCase):
    hw_classification = HardwareClassification.CUDA

    @parametrize("invocation", ("direct", "graph_replay"))
    @parametrize("cache", ("cold", "warm"))
    def test_mutation_restored_and_executed(self, device, invocation, cache):
        value = torch.zeros(257, device=device)
        original = value.clone()

        def increment(value):
            return value.add_(1)

        with torch.cuda.device(device):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                increment(value)
            value.zero_()
            samples = protocol.measure(
                [dict(fn=increment, graph=graph)],
                (value,),
                rounds=2,
                repeats=3,
                cache=cache,
                invocation=invocation,
                originals={0: original},
            )
        self.assertEqual(value, torch.ones_like(value))
        self.assertEqual(original, torch.zeros_like(original))
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(samples[0]), 2)
        self.assertTrue(all(sample > 0 for sample in samples[0]))
        self.assertEqual(
            protocol.HOST_OVERHEAD_INCLUDED[invocation], invocation == "graph_replay"
        )

    @parametrize("invocation", ("direct", "graph_replay"))
    @parametrize("cache", ("cold", "warm"))
    def test_readonly_input_multiple_outputs(self, device, invocation, cache):
        value = torch.arange(257, device=device, dtype=torch.float32)
        original = value.clone()
        first, second = torch.empty_like(value), torch.empty_like(value)

        def outputs(value):
            torch.add(value, 3, out=first)
            torch.mul(value, 2, out=second)
            return first, second

        with torch.cuda.device(device):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs(value)
            first.fill_(-1)
            second.fill_(-1)
            protocol.measure(
                [dict(fn=outputs, graph=graph)],
                (value,),
                rounds=2,
                repeats=3,
                cache=cache,
                invocation=invocation,
                originals={},
            )
        self.assertEqual(value, original)
        self.assertEqual(first, original + 3)
        self.assertEqual(second, original * 2)


instantiate_device_type_tests(TestAMDBenchmarkProtocol, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()

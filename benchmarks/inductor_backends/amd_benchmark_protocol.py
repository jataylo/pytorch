"""Measure captured GPU kernel sequences or inline CUDA graph replay.

``direct`` captures the frozen callable's GPU work and external timing events.
``graph_replay`` times an actual host-issued replay with inline events; host launch
latency can contribute to that interval. Callers validate outputs before timing.
"""

import statistics

import torch


INVOCATION_PROTOCOL = {
    "direct": "captured_gpu_kernel_sequence",
    "graph_replay": "inline_graph_replay_events",
}
HOST_OVERHEAD_INCLUDED = {"direct": False, "graph_replay": True}
MEASUREMENT_PROTOCOL = {
    "version": 2,
    "invocations": INVOCATION_PROTOCOL,
    "host_overhead_included": HOST_OVERHEAD_INCLUDED,
    "event_type": {
        "direct": "captured_external_cuda_event",
        "graph_replay": "inline_cuda_event",
    },
    "preparation": "256MiB_flush_then_mutable_input_restore_then_optional_storage_warm",
    "preparation_submission": {"direct": "captured", "graph_replay": "inline"},
    "preparation_in_timed_interval": False,
    "round_order": "rotating_entries",
    "round_statistic": "mean_of_repeat_event_intervals",
}


def measure(entries, inputs, rounds, repeats, cache, invocation, originals):
    if cache not in ("cold", "warm"):
        raise ValueError(f"Unknown cache protocol: {cache}")
    if invocation not in INVOCATION_PROTOCOL:
        raise ValueError(f"Unknown invocation protocol: {invocation}")
    if rounds < 1 or repeats < 1:
        raise ValueError("rounds and repeats must be positive")
    if not entries:
        return []
    if not inputs or not inputs[0].is_cuda:
        raise ValueError("GPU measurement requires CUDA tensor inputs")
    device = inputs[0].device
    if any(value.device != device for value in (*inputs, *originals.values())):
        raise ValueError("All inputs and restoration buffers must use the same device")

    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device=device)
        flush = torch.empty(256 << 20, device=device, dtype=torch.uint8)
        warm = [
            torch.empty(0, device=device, dtype=torch.uint8).set_(x.untyped_storage())
            for x in inputs
        ]
        captured = invocation == "direct"
        events = [
            [
                (
                    torch.cuda.Event(enable_timing=True, external=captured),
                    torch.cuda.Event(enable_timing=True, external=captured),
                )
                for _ in range(repeats)
            ]
            for _ in entries
        ]
        graphs = []
        samples = [[] for _ in entries]

        def prepare():
            flush.zero_()
            for index, original in originals.items():
                inputs[index].copy_(original)
            if cache == "warm":
                for value in warm:
                    value.add_(0)

        try:
            torch.cuda.synchronize(device)
            if captured:
                for entry, pairs in zip(entries, events):
                    graph = torch.cuda.CUDAGraph()
                    graphs.append(graph)
                    with torch.cuda.graph(graph, stream=stream):
                        for start, end in pairs:
                            prepare()
                            start.record()
                            # The graph owns its pool after these output references are dropped.
                            entry["fn"](*inputs)
                            end.record()
                for graph in graphs:
                    graph.replay()
            else:
                for entry in entries:
                    prepare()
                    entry["graph"].replay()
            torch.cuda.synchronize(device)
            for index in range(rounds):
                torch.cuda._sleep(40_000_000)
                for offset in range(len(entries)):
                    current = (index + offset) % len(entries)
                    if captured:
                        graphs[current].replay()
                    else:
                        for start, end in events[current]:
                            prepare()
                            start.record()
                            entries[current]["graph"].replay()
                            end.record()
                torch.cuda.synchronize(device)
                for values, pairs in zip(samples, events):
                    values.append(
                        statistics.mean(
                            start.elapsed_time(end) * 1000 for start, end in pairs
                        )
                    )
        finally:
            # Keep caller-stream inputs and scratch alive until side-stream capture has finished.
            torch.cuda.synchronize(device)
        return samples

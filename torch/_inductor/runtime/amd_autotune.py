from __future__ import annotations

import json
import math
import statistics

import torch
from torch.utils._ordered_set import OrderedSet


def benchmark_configs(
    tuner, launchers, args, kwargs, *, rounds=5, repeats=3, cache_samples=None
):
    """Time GPU execution after restoring aliases and modeled cache residency."""
    names = tuner.fn.arg_names[: len(args)]
    values = dict(zip(names, args)) | kwargs
    mutations = OrderedSet(
        [
            value.untyped_storage()._cdata
            for name, value in values.items()
            if name in tuner.mutated_arg_names and isinstance(value, torch.Tensor)
        ]
    )
    storages = {}
    for value in values.values():
        if (
            isinstance(value, torch.Tensor)
            and value.untyped_storage()._cdata in mutations
        ):
            key = value.untyped_storage()._cdata
            if key not in storages:
                original = torch.empty(0, device=value.device, dtype=torch.uint8)
                original = original.set_(value.untyped_storage())
                storages[key] = (original, original.clone())

    def scratch(value):
        if (
            isinstance(value, torch.Tensor)
            and value.untyped_storage()._cdata in storages
        ):
            copy = storages[value.untyped_storage()._cdata][1]
            view = torch.empty(0, device=value.device, dtype=value.dtype)
            return view.set_(
                copy.untyped_storage(),
                value.storage_offset(),
                value.size(),
                value.stride(),
            )
        return value

    scratch_args = [scratch(value) for value in args]
    scratch_kwargs = {name: scratch(value) for name, value in kwargs.items()}
    scratch_values = dict(zip(names, scratch_args)) | scratch_kwargs
    warm_names = tuner.inductor_meta.get("amd_warm_arg_names", ())
    warm = [
        torch.empty(0, device=value.device, dtype=torch.uint8).set_(
            value.untyped_storage()
        )
        for name, value in scratch_values.items()
        if (name in warm_names or name in tuner.mutated_arg_names)
        and isinstance(value, torch.Tensor)
    ]
    flush = torch.empty(256 << 20, device=tuner.device_props.index, dtype=torch.uint8)
    graph_timing = tuner.inductor_meta.get("amd_graph_tuning", False)
    balanced = tuner.inductor_meta.get("amd_cache_policy") == "balanced"
    input_warm = []
    if balanced:
        seen_warm = OrderedSet()
        for name, value in scratch_values.items():
            if not name.startswith(("in_ptr", "in_out_ptr")) or not isinstance(
                value, torch.Tensor
            ):
                continue
            key = value.untyped_storage()._cdata
            if key not in seen_warm:
                seen_warm.add(key)
                view = torch.empty(0, device=value.device, dtype=torch.uint8)
                input_warm.append(view.set_(value.untyped_storage()))

    def before(all_inputs_warm=False):
        flush.zero_()
        for original, copy in storages.values():
            copy.copy_(original)
        tuner.reset_to_zero_args(*scratch_args, **scratch_kwargs)
        warm_buffers = input_warm if all_inputs_warm else warm
        for value in warm_buffers:
            value.add_(0)

    def call(launcher):
        stream = tuner.get_device_interface().get_raw_stream(tuner.device_props.index)
        launcher(*scratch_args, **scratch_kwargs, stream=stream)

    samples = {launcher: [] for launcher in launchers}
    scratch_mutations = OrderedSet(
        [copy.untyped_storage()._cdata for original, copy in storages.values()]
    )
    output_items = [
        (name, value)
        for name, value in scratch_values.items()
        if isinstance(value, torch.Tensor)
        and (
            name.startswith(("out_ptr", "in_out_ptr"))
            or name in tuner.mutated_arg_names
        )
    ]
    outputs = [value for name, value in output_items]
    before()
    call(launchers[0])
    torch.cuda.synchronize()
    reference = [value.clone() for value in outputs]
    valid = []
    for launcher in launchers:
        before()
        for (name, actual), expected in zip(output_items, reference):
            if actual.untyped_storage()._cdata not in scratch_mutations:
                if actual.is_floating_point():
                    actual.copy_(
                        torch.where(torch.isnan(expected), float("inf"), float("nan"))
                    )
                else:
                    actual.copy_(expected)
                    raw = torch.empty(0, device=actual.device, dtype=torch.uint8)
                    raw = raw.set_(actual.untyped_storage())
                    raw.bitwise_not_()
        call(launcher)
        torch.cuda.synchronize()
        try:
            for actual, expected in zip(outputs, reference):
                low_precision = actual.dtype in (torch.float16, torch.bfloat16)
                tolerance = (
                    0.02 if low_precision else 1e-4 if actual.is_floating_point() else 0
                )
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance, equal_nan=True
                )
        except AssertionError:
            continue
        valid.append(launcher)
    del reference
    if not valid:
        raise RuntimeError("No numerically valid AMD kernel configuration")
    launchers = valid
    events = [
        [
            (
                torch.cuda.Event(enable_timing=True, external=graph_timing),
                torch.cuda.Event(enable_timing=True, external=graph_timing),
            )
            for _ in range(repeats)
        ]
        for _ in launchers
    ]
    contexts = ("producer", "inputs_warm") if balanced else ("producer",)
    context_samples = {
        launcher: {context: [] for context in contexts} for launcher in launchers
    }
    graphs = {}
    if graph_timing:
        for context in contexts:
            before(context == "inputs_warm")
            call(launchers[0])
            torch.cuda.synchronize()
            graphs[context] = []
            for index, launcher in enumerate(launchers):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for start, end in events[index]:
                        before(context == "inputs_warm")
                        start.record()
                        call(launcher)
                        end.record()
                graphs[context].append(graph)
    for iteration in range(rounds):
        torch.cuda._sleep(40_000_000)
        first_context = iteration % len(contexts)
        for context in contexts[first_context:] + contexts[:first_context]:
            for offset in range(len(launchers)):
                index = (iteration + offset) % len(launchers)
                launcher = launchers[index]
                if graph_timing:
                    graphs[context][index].replay()
                else:
                    for start, end in events[index]:
                        before(context == "inputs_warm")
                        start.record()
                        call(launcher)
                        end.record()
            torch.cuda.synchronize()
            for launcher, pairs in zip(launchers, events):
                elapsed = statistics.mean(
                    start.elapsed_time(end) for start, end in pairs
                )
                context_samples[launcher][context].append(elapsed)
    for launcher in launchers:
        samples[launcher] = []
        for index in range(rounds):
            log_times = (
                math.log(max(context_samples[launcher][context][index], 1e-9))
                for context in contexts
            )
            samples[launcher].append(math.exp(statistics.mean(log_times)))
    if cache_samples is not None:
        cache_samples.update(context_samples)

    timings = {
        launcher: statistics.median(values) if values else float("inf")
        for launcher, values in samples.items()
    }
    return timings, samples


def _key(cfg):
    return (*sorted(cfg.kwargs.items()), cfg.num_warps, cfg.num_stages)


def _neighbors(tuner, config, args, kwargs):
    from .hints import AMDConfigLimits
    from .triton_compat import Config

    limits = AMDConfigLimits.from_device(tuner.device_props)
    values = dict(zip(tuner.fn.arg_names, args)) | kwargs
    xnumel = values.get("xnumel")
    rnumel = values.get("r0_numel")
    family = tuner.inductor_meta.get("amd_gluon_family")
    gluon = tuner.inductor_meta.get("use_gluon", False)
    options = {
        "XBLOCK": [max(1, config.kwargs["XBLOCK"] // 2), config.kwargs["XBLOCK"] * 2]
    }
    for field in ("R0_BLOCK", "AMD_VEC", "AMD_UNROLL", "AMD_LANES"):
        if field in config.kwargs:
            options[field] = [
                max(1, config.kwargs[field] // 2),
                config.kwargs[field] * 2,
            ]
    options["waves_per_eu"] = [0, 1, 2]
    for field in ("AMD_ROW_ORDER", "AMD_READ_ORDER", "AMD_SPLIT_WARPS"):
        if field in config.kwargs:
            options[field] = [0, 1]
    candidates = []
    for field, choices in options.items():
        for value in choices:
            kw = dict(config.kwargs)
            kw[field] = value
            if kw != config.kwargs:
                candidates.append(
                    Config(kw, num_warps=config.num_warps, num_stages=config.num_stages)
                )
    wide_scan = (
        tuner.inductor_meta.get("amd_confirm_tuning")
        and family == "row_reduction"
        and isinstance(rnumel, int)
        and rnumel >= 8192
    )
    max_warps = limits.max_warps if wide_scan else min(8, limits.max_warps)
    for warps in (config.num_warps // 2, config.num_warps * 2):
        if warps < 1 or warps > max_warps:
            continue
        kw = dict(config.kwargs)
        if "AMD_WARPS" in kw:
            kw["AMD_WARPS"] = warps
        candidates.append(Config(kw, num_warps=warps, num_stages=config.num_stages))
    for candidate in candidates:
        kw = candidate.kwargs
        xblock = kw["XBLOCK"]
        rblock = kw.get("R0_BLOCK", rnumel if isinstance(rnumel, int) else 1)
        lanes = kw.get("AMD_LANES", limits.warp_size)
        if (
            candidate.num_warps > limits.max_warps
            or not limits.warp_size // 8 <= lanes <= limits.warp_size
        ):
            continue
        if kw.get("AMD_VEC", 1) > 32 or kw.get("AMD_UNROLL", 1) > 8:
            continue
        if (
            xblock > min(16384, limits.max_tile)
            or kw.get("R0_BLOCK", 1) > 8192
            or xblock * rblock > limits.max_tile
        ):
            continue
        if family == "transpose" and kw.get("AMD_UNROLL", 1) != 1:
            continue
        if xblock % kw.get("AMD_UNROLL", 1):
            continue
        if (
            isinstance(xnumel, int)
            and (not gluon or family == "row_reduction")
            and xnumel % xblock
        ):
            continue
        yield candidate


def tune(tuner, args, kwargs):
    r"""tune(tuner, args, kwargs) -> dict | None

    Measure launchers and bounded neighbors with restored mutations and cache contexts.
    Return confirmed finalist timings when requested, or ``None`` if scratch memory is insufficient.
    """
    values = dict(zip(tuner.fn.arg_names, args)) | kwargs
    tensors = {
        value.untyped_storage()._cdata: value
        for value in values.values()
        if isinstance(value, torch.Tensor)
    }
    needed = sum(value.untyped_storage().nbytes() for value in tensors.values())
    free, _ = torch.cuda.mem_get_info(tuner.device_props.index)
    if 3 * needed + (256 << 20) > free // 2:
        return None
    cache_samples = {}
    timings, samples = benchmark_configs(
        tuner, tuner.launchers, args, kwargs, cache_samples=cache_samples
    )
    seen = OrderedSet([_key(launcher.config) for launcher in tuner.launchers])
    for _ in range(3):
        best = min(timings, key=timings.get)
        neighbors = []
        for config in _neighbors(tuner, best.config, args, kwargs):
            key = _key(config)
            if key in seen:
                continue
            seen.add(key)
            if len(neighbors) >= 10:
                break
            try:
                with tuner.lock:
                    result = tuner._precompile_config(config)
                    launcher = result.make_launcher()
                tuner.compile_results.append(result)
                tuner.launchers.append(launcher)
                neighbors.append(launcher)
            except Exception as error:
                from triton.runtime.errors import OutOfResources

                if not isinstance(error, OutOfResources):
                    raise
        if not neighbors:
            break
        measured, raw = benchmark_configs(
            tuner, [best, *neighbors], args, kwargs, cache_samples=cache_samples
        )
        timings.update(measured)
        samples.update(raw)
        if min(measured, key=measured.get) is best:
            break
    confirmed = []
    if tuner.inductor_meta.get("amd_confirm_tuning"):
        finite = (launcher for launcher in timings if math.isfinite(timings[launcher]))
        confirmed = sorted(finite, key=timings.get)[:6]
        measured, raw = benchmark_configs(
            tuner,
            confirmed,
            args,
            kwargs,
            rounds=9,
            repeats=5,
            cache_samples=cache_samples,
        )
        timings.update(measured)
        samples.update(raw)
    for launcher, timing in timings.items():
        tuner.coordesc_tuner.cache_benchmark_result(launcher.config, timing)
    selected = min(confirmed or timings, key=timings.get)
    meta = tuner.inductor_meta
    report = {
        "kernel_name": meta.get("kernel_name"),
        "family": meta.get(
            "amd_gluon_family", meta.get("amd_pointwise_family", "native")
        ),
        "approximate_math": meta.get("amd_approximate_math", False),
        "cache_policy": meta.get("amd_cache_policy", "producer"),
        "timing_protocol": "captured_gpu_protocol"
        if meta.get("amd_graph_tuning")
        else "inline_gpu_events",
        "score_statistic": "median_round_geomean_ms",
        "confirmed_candidates": len(confirmed),
        "inputs": [
            {
                "name": name,
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
            }
            for name, value in values.items()
            if isinstance(value, torch.Tensor)
        ],
        "warm_arg_names": meta.get("amd_warm_arg_names", ()),
        "selected_config": dict(
            selected.config.kwargs,
            num_warps=selected.config.num_warps,
            num_stages=selected.config.num_stages,
        ),
        "candidates": [
            {
                "config": dict(
                    launcher.config.kwargs, num_warps=launcher.config.num_warps
                ),
                "median_ms": value if math.isfinite(value) else None,
                "confirmed": launcher in confirmed,
                "samples_ms": samples[launcher],
                "cache_samples_ms": cache_samples.get(launcher, {}),
            }
            for launcher, value in timings.items()
        ],
    }
    from torch._logging import trace_structured

    trace_structured(
        "artifact",
        metadata_fn=lambda: {"name": "amd_kernel_autotune", "encoding": "json"},
        payload_fn=lambda: json.dumps(report),
    )
    path = meta.get("amd_autotune_diagnostics_path")
    if path:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as output:
            output.write(json.dumps(report) + "\n")
    if confirmed:
        timings = {
            launcher: value
            for launcher, value in timings.items()
            if launcher in confirmed
        }
    return timings

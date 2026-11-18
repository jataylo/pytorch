from __future__ import annotations

import gc
import json
import math
import statistics
import time

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
    banks = [(scratch_args, scratch_kwargs)]
    bank_storages = []
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

    def before(all_inputs_warm=False, steady=False):
        flush.zero_()
        for original, copy in storages.values():
            copy.copy_(original)
        tuner.reset_to_zero_args(*scratch_args, **scratch_kwargs)
        warm_buffers = input_warm if all_inputs_warm else warm
        for value in warm_buffers:
            value.add_(0)
        if steady:
            for storage_map, (bank_args, bank_kwargs) in zip(bank_storages, banks[1:]):
                for original, copy in storages.values():
                    storage_map[copy.untyped_storage()._cdata].copy_(original)
                tuner.reset_to_zero_args(*bank_args, **bank_kwargs)

    def call(launcher, bank=0):
        stream = tuner.get_device_interface().get_raw_stream(tuner.device_props.index)
        bank_args, bank_kwargs = banks[bank]
        launcher(*bank_args, **bank_kwargs, stream=stream)

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
    if graph_timing and tuner.inductor_meta.get("amd_confirm_tuning"):
        # Replicate read/write storage; overwrite-only outputs can be reused.
        written = {}
        reset_names = getattr(tuner, "reset_to_zero_arg_names", ())
        for name, value in scratch_values.items():
            if isinstance(value, torch.Tensor) and (
                name in tuner.mutated_arg_names or name in reset_names
            ):
                key = value.untyped_storage()._cdata
                if key not in written:
                    raw = torch.empty(0, device=value.device, dtype=torch.uint8)
                    written[key] = raw.set_(value.untyped_storage())
        footprint = sum(value.numel() for value in written.values())
        free, _ = torch.cuda.mem_get_info(tuner.device_props.index)
        device = torch.cuda.get_device_properties(tuner.device_props.index)
        cache = getattr(device, "L2_cache_size", 0)
        budget = min(free // 8, max(cache, footprint))
        burst = next((n for n in (32, 16, 8, 4, 2) if (n - 1) * footprint <= budget), 1)
        tuner.inductor_meta["amd_steady_bank_bytes"] = (burst - 1) * footprint
        for _ in range(burst - 1):
            storage_map = {key: value.clone() for key, value in written.items()}

            def bank_value(value):
                if isinstance(value, torch.Tensor):
                    copy = storage_map.get(value.untyped_storage()._cdata)
                    if copy is not None:
                        view = torch.empty(0, device=value.device, dtype=value.dtype)
                        return view.set_(
                            copy.untyped_storage(),
                            value.storage_offset(),
                            value.size(),
                            value.stride(),
                        )
                return value

            banks.append(
                (
                    [bank_value(v) for v in scratch_args],
                    {k: bank_value(v) for k, v in scratch_kwargs.items()},
                )
            )
            bank_storages.append(storage_map)
    tuner.inductor_meta["amd_steady_batch_size"] = len(banks)
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
    if len(banks) > 1:
        contexts += ("steady",)
    context_samples = {
        launcher: {context: [] for context in contexts} for launcher in launchers
    }
    graphs = {}
    if graph_timing:
        for context in contexts:
            before(context != "producer", steady=context == "steady")
            call(launchers[0])
            torch.cuda.synchronize()
            graphs[context] = []
            for index, launcher in enumerate(launchers):
                graph = torch.cuda.CUDAGraph()
                # Collecting discarded launcher cycles can unload Triton modules.
                gc_enabled = gc.isenabled()
                gc.disable()
                try:
                    with torch.cuda.graph(graph):
                        if context == "steady":
                            # Keep event nodes outside the short-kernel batch.
                            for bank in range(len(banks)):
                                call(launcher, bank)
                        else:
                            for start, end in events[index]:
                                before(context != "producer")
                                start.record()
                                call(launcher)
                                end.record()
                finally:
                    if gc_enabled:
                        gc.enable()
                graphs[context].append(graph)
    for iteration in range(rounds):
        torch.cuda._sleep(40_000_000)
        first_context = iteration % len(contexts)
        for context in contexts[first_context:] + contexts[:first_context]:
            order = [
                (iteration // 2 + offset) % len(launchers)
                for offset in range(len(launchers))
            ]
            if iteration % 2:
                order.reverse()
            for index in order:
                launcher = launchers[index]
                if graph_timing and context != "steady":
                    graphs[context][index].replay()
                else:
                    for start, end in events[index]:
                        before(context != "producer", steady=context == "steady")
                        start.record()
                        if context == "steady":
                            graphs[context][index].replay()
                        else:
                            call(launcher)
                        end.record()
            torch.cuda.synchronize()
            for launcher, pairs in zip(launchers, events):
                elapsed = statistics.mean(
                    start.elapsed_time(end) for start, end in pairs
                )
                if context == "steady":
                    elapsed /= len(banks)
                context_samples[launcher][context].append(elapsed)
    for launcher in launchers:
        if "steady" in contexts:
            samples[launcher] = list(context_samples[launcher]["steady"])
            continue
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


def _neighbors(tuner, config, args, kwargs, *, n_regs=None, n_spills=None):
    from .hints import AMDConfigLimits
    from .triton_compat import Config

    limits = AMDConfigLimits.from_device(tuner.device_props)
    values = dict(zip(tuner.fn.arg_names, args)) | kwargs
    xnumel = values.get("xnumel")
    rnumel = values.get("r0_numel")
    family = tuner.inductor_meta.get("amd_native_family")
    tail_safe = family in (
        "add",
        "broadcast",
        "transpose",
        "flat_pointwise",
        "tiled_transpose",
    ) or (
        family == "row_reduction"
        and tuner.inductor_meta.get("amd_row_tail_safe", False)
    )
    options = {
        "XBLOCK": [max(1, config.kwargs["XBLOCK"] // 2), config.kwargs["XBLOCK"] * 2]
    }
    for field in ("YBLOCK", "R0_BLOCK", "AMD_UNROLL"):
        if field in config.kwargs:
            options[field] = [
                max(1, config.kwargs[field] // 2),
                config.kwargs[field] * 2,
            ]
    options["waves_per_eu"] = [0, 1, 2]
    for field in ("AMD_READ_ORDER",):
        if field in config.kwargs:
            if (
                field == "AMD_READ_ORDER"
                and family == "tiled_transpose"
                and config.kwargs["AMD_TILE_M"]
            ):
                continue
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
    if (
        family in ("add", "broadcast", "flat_pointwise")
        and "AMD_UNROLL" in config.kwargs
    ):
        joint = []
        base_block, base_unroll = config.kwargs["XBLOCK"], config.kwargs["AMD_UNROLL"]
        for block, unroll in (
            (base_block // 2, base_unroll // 2),
            (base_block * 2, base_unroll * 2),
        ):
            if block and unroll:
                joint.append(
                    Config(
                        dict(config.kwargs, XBLOCK=block, AMD_UNROLL=unroll),
                        num_warps=config.num_warps,
                        num_stages=config.num_stages,
                    )
                )
        candidates[:0] = joint
    if tuner.inductor_meta.get("amd_reduction_tuning") and "R0_BLOCK" in config.kwargs:
        block, reduction = config.kwargs["XBLOCK"], config.kwargs["R0_BLOCK"]
        joint = []
        for x, r in (
            (block // 2, reduction // 2),
            (block // 2, reduction // 4),
            (block * 2, reduction * 2),
        ):
            if x and r:
                joint.append(
                    Config(
                        dict(config.kwargs, XBLOCK=x, R0_BLOCK=r),
                        num_warps=config.num_warps,
                        num_stages=config.num_stages,
                    )
                )
        candidates[:0] = joint
    if family == "tiled_transpose":
        m, n = config.kwargs["AMD_TILE_M"], config.kwargs["AMD_TILE_N"]
        if m and n:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.kwargs["XBLOCK"] == config.kwargs["XBLOCK"]
            ]
            tiles = [(m // 2, n), (m * 2, n), (m, n // 2), (m, n * 2)]
        else:
            tiles = [(16, 16), (32, 32)]
        for m, n in tiles:
            if 8 <= m <= 128 and 8 <= n <= 128:
                kw = dict(config.kwargs, XBLOCK=m * n, AMD_TILE_M=m, AMD_TILE_N=n)
                if "YBLOCK" in kw:
                    kw["XBLOCK"] = m
                    kw["YBLOCK"] = n
                if "AMD_READ_ORDER" in kw:
                    kw["AMD_READ_ORDER"] = 0
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
        candidates.append(Config(kw, num_warps=warps, num_stages=config.num_stages))
    registers = tuner.device_props.regs_per_multiprocessor
    register_pressure = (
        n_regs is not None
        and registers is not None
        and n_regs * config.num_warps * limits.warp_size > registers // 2
    )
    if n_spills or register_pressure:
        candidates.sort(
            key=lambda candidate: (
                candidate.kwargs["XBLOCK"] >= config.kwargs["XBLOCK"],
                candidate.num_warps <= config.num_warps,
            )
        )
    for candidate in candidates:
        kw = candidate.kwargs
        xblock = kw["XBLOCK"]
        rblock = kw.get("R0_BLOCK", rnumel if isinstance(rnumel, int) else 1)
        tile = xblock * rblock
        if family == "tiled_transpose":
            m, n = kw["AMD_TILE_M"], kw["AMD_TILE_N"]
            tile = m * n if m else xblock * kw.get("YBLOCK", 1)
        if candidate.num_warps > limits.max_warps:
            continue
        if kw.get("AMD_UNROLL", 1) > 8:
            continue
        if (
            xblock > min(16384, limits.max_tile)
            or kw.get("R0_BLOCK", 1) > 8192
            or tile > limits.max_tile
        ):
            continue
        if family == "transpose" and kw.get("AMD_UNROLL", 1) != 1:
            continue
        if xblock % kw.get("AMD_UNROLL", 1):
            continue
        if family == "tiled_transpose":
            m, n = kw["AMD_TILE_M"], kw["AMD_TILE_N"]
            tile_xblock = m if "YBLOCK" in kw else m * n
            if bool(m) != bool(n) or m and xblock != tile_xblock:
                continue
            if m and "YBLOCK" in kw and kw["YBLOCK"] != n:
                continue
        if isinstance(xnumel, int) and not tail_safe and xnumel % xblock:
            continue
        if family == "row_reduction":
            count = tuner.inductor_meta.get("amd_row_count", xnumel)
            if not isinstance(count, int):
                count = xnumel
            if (
                isinstance(count, int)
                and count > 0
                and xblock > 1 << (count - 1).bit_length()
            ):
                continue
        yield candidate


def _confirmed_choice(timings, samples, baseline):
    """Require a repeatable paired gain before replacing the baseline contender."""
    best = min(timings, key=timings.get)
    if baseline is None or baseline not in timings or best is baseline:
        return best, "fastest_confirmed", 0.0
    gains = [
        math.log(reference / candidate)
        for reference, candidate in zip(samples[baseline], samples[best])
        if reference > 0 and candidate > 0
    ]
    if len(gains) < 3:
        return baseline, "insufficient_paired_samples", 0.0
    gain = statistics.median(gains)
    deviation = statistics.median(abs(value - gain) for value in gains)
    threshold = max(math.log(1.01), 2 * 1.4826 * deviation / math.sqrt(len(gains)))
    if gain <= threshold:
        return baseline, "baseline_within_variation", threshold
    return best, "confirmed_paired_gain", threshold


def tune(tuner, args, kwargs):
    r"""tune(tuner, args, kwargs) -> dict | None

    Measure launchers and bounded neighbors with restored mutations and cache contexts.
    Return confirmed finalist timings when requested, or ``None`` if scratch memory is insufficient.
    """
    started = time.perf_counter_ns()
    neighbor_compile_ns = 0
    initial_count = len(tuner.launchers)
    compile_attempts = 0
    compile_failures = 0
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
    baseline_configs = tuner.inductor_meta.get("amd_baseline_configs", ())
    baseline_choices = [
        launcher
        for launcher in timings
        if math.isfinite(timings[launcher])
        and dict(
            launcher.config.kwargs,
            num_warps=launcher.config.num_warps,
            num_stages=launcher.config.num_stages,
        )
        in baseline_configs
    ]
    baseline = min(baseline_choices, key=timings.get) if baseline_choices else None
    seen = OrderedSet([_key(launcher.config) for launcher in tuner.launchers])
    for _ in range(3):
        best = min(timings, key=timings.get)
        neighbors = []
        for config in _neighbors(
            tuner,
            best.config,
            args,
            kwargs,
            n_regs=getattr(best, "n_regs", None),
            n_spills=getattr(best, "n_spills", None),
        ):
            key = _key(config)
            if key in seen:
                continue
            if len(neighbors) >= 10:
                break
            seen.add(key)
            compile_attempts += 1
            compile_started = time.perf_counter_ns()
            try:
                from torch._dynamo.device_interface import DeviceGuard

                with (
                    DeviceGuard(tuner.get_device_interface(), tuner.device_props.index),
                    tuner.lock,
                ):
                    result = tuner._precompile_config(config)
                    launcher = result.make_launcher()
                tuner.compile_results.append(result)
                tuner.launchers.append(launcher)
                neighbors.append(launcher)
            except Exception as error:
                from triton.runtime.errors import OutOfResources

                if not isinstance(error, OutOfResources):
                    raise
                compile_failures += 1
            finally:
                neighbor_compile_ns += time.perf_counter_ns() - compile_started
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
    if tuner.inductor_meta.get("amd_confirm_tuning") or baseline is not None:
        finite = (launcher for launcher in timings if math.isfinite(timings[launcher]))
        confirmed = sorted(finite, key=timings.get)[:6]
        confirmed.extend(
            launcher for launcher in baseline_choices if launcher not in confirmed
        )
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
        baseline = min(baseline_choices, key=timings.get) if baseline_choices else None
    for launcher, timing in timings.items():
        tuner.coordesc_tuner.cache_benchmark_result(launcher.config, timing)
    final_timings = (
        {launcher: timings[launcher] for launcher in confirmed}
        if confirmed
        else timings
    )
    selected, reason, threshold = _confirmed_choice(final_timings, samples, baseline)
    meta = tuner.inductor_meta
    report = {
        "kernel_name": meta.get("kernel_name"),
        "family": meta.get("amd_native_family", "native"),
        "cache_policy": meta.get("amd_cache_policy", "producer"),
        "timing_protocol": "captured_gpu_protocol"
        if meta.get("amd_graph_tuning")
        else "inline_gpu_events",
        "score_statistic": "median_steady_batch_ms"
        if "steady" in cache_samples.get(selected, {})
        else "median_round_geomean_ms",
        "selection_reason": reason,
        "steady_batch_size": meta.get("amd_steady_batch_size", 1),
        "steady_bank_bytes": meta.get("amd_steady_bank_bytes", 0),
        "steady_timing_protocol": "inline_graph_replay_events",
        "minimum_paired_log_gain": threshold,
        "baseline_config": dict(
            baseline.config.kwargs,
            num_warps=baseline.config.num_warps,
            num_stages=baseline.config.num_stages,
        )
        if baseline is not None
        else None,
        "confirmed_candidates": len(confirmed),
        "initial_candidates": initial_count,
        "measured_candidates": len(timings),
        "neighbor_compile_attempts": compile_attempts,
        "neighbor_compile_failures": compile_failures,
        "reported_precompile_time_ns": getattr(tuner, "precompile_time_taken_ns", 0),
        "neighbor_compile_time_ns": neighbor_compile_ns,
        "accounted_compile_time_ns": getattr(tuner, "precompile_time_taken_ns", 0)
        + neighbor_compile_ns,
        "compile_time_accounting": "precompile counter may exclude async initial compilation",
        "tuning_time_ns": time.perf_counter_ns() - started - neighbor_compile_ns,
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
                    launcher.config.kwargs,
                    num_warps=launcher.config.num_warps,
                    num_stages=launcher.config.num_stages,
                ),
                "median_ms": value if math.isfinite(value) else None,
                "confirmed": launcher in confirmed,
                "samples_ms": samples[launcher],
                "cache_samples_ms": cache_samples.get(launcher, {}),
                "n_regs": getattr(launcher, "n_regs", None),
                "n_spills": getattr(launcher, "n_spills", None),
                "shared": getattr(launcher, "shared", None),
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
    return {selected: timings[selected]}

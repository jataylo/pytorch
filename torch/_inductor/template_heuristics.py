from __future__ import annotations

import itertools
from typing import Any, cast, Dict, List
from functools import partial

class BaseConfigHeuristic:
    """
    Base class for mm_configs, device specific triton kernels config inherit from here
    """

    # List of dictionaries to store the kernel configs. Configs that evaluate to true
    # will be utilised on the target platform. The configs are as follows:
    # (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
    mm_configs = [
        # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
        {"config": (32, 32, 16, 1, 2), "cond": True},
        {"config": (32, 32, 128, 2, 4), "cond": True},
        {"config": (32, 64, 32, 5, 8), "cond": True},
        {"config": (64, 32, 32, 5, 8), "cond": True},
        {"config": (64, 32, 128, 5, 4), "cond": True},
        {"config": (64, 64, 16, 2, 4), "cond": True},
        {"config": (64, 64, 32, 2, 4), "cond": True},
        {"config": (64, 64, 64, 3, 8), "cond": True},
        {"config": (64, 64, 128, 5, 4), "cond": True},
        {"config": (64, 128, 32, 3, 4), "cond": True},
        {"config": (64, 128, 32, 4, 8), "cond": True},
        {"config": (64, 128, 64, 3, 4), "cond": True},
        {"config": (64, 128, 128, 4, 4), "cond": True},
        {"config": (128, 64, 32, 3, 4), "cond": True},
        {"config": (128, 64, 32, 4, 8), "cond": True},
        {"config": (128, 128, 32, 2, 8), "cond": True},
        {"config": (128, 128, 32, 3, 4), "cond": True},
        {"config": (128, 128, 64, 3, 4), "cond": True},
        {"config": (128, 128, 64, 5, 8), "cond": True},
    ]

    # Exhaustive search for mm configs
    exhaustive_configs = [
        {"config": (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps), "cond": True}
        for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
            [16, 32, 64, 128, 256], repeat=3
        )
        for num_stages in [1, 2, 3, 4, 5]
        for num_warps in [2, 4, 8]
    ]

    # these are only used in tuned_mm when AutoHeuristic is enabled
    # the idea is that when AutoHeuristic collects data to learn a heuristic, more configs are autotuned
    # when the learned heuristic is used, the learned heuristic reduces the number of configs down to 10
    # which saves compilation time (since less configs are autotuned) and potentially increase performance
    # because the learned heuristic might predict a config that is not part mm_configs
    extra_mm_configs = [
        # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
        {"config": (16, 32, 16, 3, 2), "cond": True},
        {"config": (16, 32, 32, 4, 2), "cond": True},
        {"config": (16, 32, 32, 5, 2), "cond": True},
        {"config": (64, 64, 128, 3, 4), "cond": True},
        {"config": (128, 64, 32, 2, 2), "cond": True},
        {"config": (128, 64, 64, 3, 8), "cond": True},
        {"config": (128, 64, 128, 4, 8), "cond": True},
        {"config": (128, 128, 32, 4, 4), "cond": True},
        {"config": (128, 128, 64, 3, 8), "cond": True},
        {"config": (128, 128, 64, 5, 4), "cond": True},
    ]

    int8_mm_configs = [
        # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
        {"config": (64, 64, 32, 2, 4), "cond": True},
        {"config": (64, 128, 32, 3, 4), "cond": True},
        {"config": (128, 64, 32, 3, 4), "cond": True},
        {"config": (64, 128, 32, 4, 8), "cond": True},
        {"config": (128, 64, 32, 4, 8), "cond": True},
        {"config": (64, 32, 32, 5, 8), "cond": True},
        {"config": (32, 64, 32, 5, 8), "cond": True},
        {"config": (128, 128, 32, 2, 8), "cond": True},
        {"config": (64, 64, 64, 3, 8), "cond": True},
        {"config": (128, 256, 128, 3, 8), "cond": True},
        {"config": (256, 128, 128, 3, 8), "cond": True},
    ]

    mixed_mm_configs = [
        # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
        {"config": (16, 128, 256, 3, 4), "cond": True},
        {"config": (16, 128, 256, 5, 8), "cond": True},
    ]

    persistent_mm_configs = [
        # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
        {"config": (128, 256, 64, 3, 8), "cond": True},
        {"config": (128, 128, 64, 3, 8), "cond": True},
        {"config": (128, 128, 128, 3, 8), "cond": True},
        {"config": (128, 128, 128, 3, 4), "cond": True},
        {"config": (128, 128, 64, 4, 8), "cond": True},
    ]

    scaled_mm_configs = [
        # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
        {"config": (128, 256, 32, 3, 8), "cond": True},
        {"config": (256, 128, 32, 3, 8), "cond": True},
        {"config": (256, 64, 32, 4, 4), "cond": True},
        {"config": (64, 256, 32, 4, 4), "cond": True},
        {"config": (128, 128, 32, 4, 4), "cond": True},
        {"config": (128, 64, 32, 4, 4), "cond": True},
        {"config": (64, 128, 32, 4, 4), "cond": True},
        {"config": (128, 32, 32, 4, 4), "cond": True},
        {"config": (64, 32, 32, 5, 2), "cond": True},
        {"config": (256, 128, 128, 3, 8), "cond": True},
        {"config": (256, 64, 128, 4, 4), "cond": True},
        {"config": (64, 256, 128, 4, 4), "cond": True},
        {"config": (128, 128, 128, 4, 4), "cond": True},
        {"config": (128, 64, 64, 4, 4), "cond": True},
        {"config": (64, 128, 64, 4, 4), "cond": True},
        {"config": (128, 32, 64, 4, 4), "cond": True},
        {"config": (64, 32, 64, 5, 2), "cond": True},
        {"config": (16, 32, 32, 2, 2), "cond": True},
        {"config": (16, 64, 32, 2, 2), "cond": True},
        {"config": (16, 128, 32, 2, 4), "cond": True},
        {"config": (16, 256, 32, 2, 4), "cond": True},
        {"config": (16, 32, 64, 2, 2), "cond": True},
        {"config": (16, 64, 64, 2, 2), "cond": True},
        {"config": (16, 128, 64, 2, 4), "cond": True},
        {"config": (16, 256, 64, 2, 4), "cond": True},
        {"config": (32, 32, 32, 2, 2), "cond": True},
        {"config": (32, 64, 32, 2, 2), "cond": True},
        {"config": (32, 128, 32, 2, 4), "cond": True},
        {"config": (32, 256, 32, 2, 4), "cond": True},
        {"config": (32, 32, 64, 2, 2), "cond": True},
        {"config": (32, 64, 64, 2, 2), "cond": True},
        {"config": (32, 128, 64, 2, 4), "cond": True},
        {"config": (32, 256, 64, 2, 4), "cond": True},
        {"config": (16, 32, 32, 3, 2), "cond": True},
        {"config": (16, 64, 32, 3, 2), "cond": True},
        {"config": (16, 128, 32, 3, 4), "cond": True},
        {"config": (16, 256, 32, 3, 4), "cond": True},
        {"config": (16, 32, 64, 3, 2), "cond": True},
        {"config": (16, 64, 64, 3, 2), "cond": True},
        {"config": (16, 128, 64, 3, 4), "cond": True},
        {"config": (16, 256, 64, 3, 4), "cond": True},
        {"config": (32, 32, 32, 3, 2), "cond": True},
        {"config": (32, 64, 32, 3, 2), "cond": True},
        {"config": (32, 128, 32, 3, 4), "cond": True},
        {"config": (32, 256, 32, 3, 4), "cond": True},
        {"config": (32, 32, 64, 3, 2), "cond": True},
        {"config": (32, 64, 64, 3, 2), "cond": True},
        {"config": (32, 128, 64, 3, 4), "cond": True},
        {"config": (32, 256, 64, 3, 4), "cond": True},
        {"config": (16, 32, 32, 4, 2), "cond": True},
        {"config": (16, 64, 32, 4, 2), "cond": True},
        {"config": (16, 128, 32, 4, 4), "cond": True},
        {"config": (16, 256, 32, 4, 4), "cond": True},
        {"config": (16, 32, 64, 4, 2), "cond": True},
        {"config": (16, 64, 64, 4, 2), "cond": True},
        {"config": (16, 128, 64, 4, 4), "cond": True},
        {"config": (16, 256, 64, 4, 4), "cond": True},
        {"config": (32, 32, 32, 4, 2), "cond": True},
        {"config": (32, 64, 32, 4, 2), "cond": True},
        {"config": (32, 128, 32, 4, 4), "cond": True},
        {"config": (32, 256, 32, 4, 4), "cond": True},
        {"config": (32, 32, 64, 4, 2), "cond": True},
        {"config": (32, 64, 64, 4, 2), "cond": True},
        {"config": (32, 128, 64, 4, 4), "cond": True},
        {"config": (32, 256, 64, 4, 4), "cond": True},
        {"config": (16, 32, 32, 5, 2), "cond": True},
        {"config": (16, 64, 32, 5, 2), "cond": True},
        {"config": (16, 128, 32, 5, 4), "cond": True},
        {"config": (16, 256, 32, 5, 4), "cond": True},
        {"config": (16, 32, 64, 5, 2), "cond": True},
        {"config": (16, 64, 64, 5, 2), "cond": True},
        {"config": (16, 128, 64, 5, 4), "cond": True},
        {"config": (16, 256, 64, 5, 4), "cond": True},
        {"config": (32, 32, 32, 5, 2), "cond": True},
        {"config": (32, 64, 32, 5, 2), "cond": True},
        {"config": (32, 128, 32, 5, 4), "cond": True},
        {"config": (32, 256, 32, 5, 4), "cond": True},
        {"config": (32, 32, 64, 5, 2), "cond": True},
        {"config": (32, 64, 64, 5, 2), "cond": True},
        {"config": (32, 128, 64, 5, 4), "cond": True},
        {"config": (32, 256, 64, 5, 4), "cond": True},
        {"config": (16, 32, 32, 6, 2), "cond": True},
        {"config": (16, 64, 32, 6, 2), "cond": True},
        {"config": (16, 128, 32, 6, 4), "cond": True},
        {"config": (16, 256, 32, 6, 4), "cond": True},
        {"config": (16, 32, 64, 6, 2), "cond": True},
        {"config": (16, 64, 64, 6, 2), "cond": True},
        {"config": (16, 128, 64, 6, 4), "cond": True},
        {"config": (16, 256, 64, 6, 4), "cond": True},
        {"config": (32, 32, 32, 6, 2), "cond": True},
        {"config": (32, 64, 32, 6, 2), "cond": True},
        {"config": (32, 128, 32, 6, 4), "cond": True},
        {"config": (32, 256, 32, 6, 4), "cond": True},
        {"config": (32, 32, 64, 6, 2), "cond": True},
        {"config": (32, 64, 64, 6, 2), "cond": True},
        {"config": (32, 128, 64, 6, 4), "cond": True},
        {"config": (32, 256, 64, 6, 4), "cond": True},
    ]

    scaled_persistent_mm_configs = [
        # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
        {"config": (128, 128, 64, 3, 8), "cond": True},
        {"config": (128, 128, 128, 3, 8), "cond": True},
        {"config": (128, 128, 128, 4, 8), "cond": True},
        {"config": (128, 128, 128, 4, 4), "cond": True},
        {"config": (128, 128, 128, 3, 4), "cond": True},
        {"config": (128, 128, 128, 5, 4), "cond": True},
        {"config": (128, 128, 128, 5, 8), "cond": True},
        {"config": (128, 128, 128, 6, 8), "cond": True},
        {"config": (128, 128, 64, 4, 8), "cond": True},
    ]

    # TODO: Unify with other gemm patterns, mm_plus_mm currently follows
    # slightly different pattern than rest
    mm_plus_mm_configs = [
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 2,
            "num_warps": 4,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 3,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 4,
            "num_warps": 16,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            "num_stages": 4,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 4,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            "num_stages": 1,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            "num_stages": 1,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128},
            "num_stages": 1,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16},
            "num_stages": 2,
            "num_warps": 4,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16},
            "num_stages": 1,
            "num_warps": 2,
            "cond": True,
        },
    ]

    conv_configs = [
        # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
        {"config": (64, 256, 16, 2, 4), "cond": True},
        {"config": (256, 64, 16, 2, 4), "cond": True},
        {"config": (1024, 16, 16, 1, 8), "cond": True},
        {"config": (128, 128, 32, 2, 8), "cond": True},
        {"config": (64, 64, 32, 2, 4), "cond": True},
        {"config": (64, 256, 32, 2, 8), "cond": True},
        {"config": (256, 64, 32, 2, 8), "cond": True},
    ]

    def _filter_configs(self, configs):
        return tuple(
            cast(tuple[int, int, int, int, int], config["config"])
            for config in configs
            if config["cond"]
        )

    def _preprocess_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: Sequence[tuple[int, int, int, int, int]],
        has_int8_tensor=False,
        scale=1,
        exclude=lambda m, n, k: False,
    ):
        from .runtime.runtime_utils import next_power_of_2
        from torch.utils._ordered_set import OrderedSet
        from torch._inductor.virtualized import V
        """
        Heuristic to preprocess configs e.g. shrink when they are bigger than the input size

        :param scale: scale factor applied to the config values
        :param exclude: whether a given config should be excluded
        """
        from torch._inductor import config

        max_mm_configs = config.test_configs.max_mm_configs

        min_block_size = 16
        # block_k=16 seems to be causing issues
        # see: https://github.com/triton-lang/triton/issues/2156#issuecomment-1695897424
        min_block_size_k = 32 if has_int8_tensor else 16
        m = max(
            next_power_of_2(
                V.graph.sizevars.size_hint(
                    m, fallback=config.unbacked_symint_fallback  # type: ignore[arg-type]
                )
            ),
            min_block_size,
        )
        n = max(
            next_power_of_2(
                V.graph.sizevars.size_hint(
                    n, fallback=config.unbacked_symint_fallback  # type: ignore[arg-type]
                )
            ),
            min_block_size,
        )
        k = max(
            next_power_of_2(
                V.graph.sizevars.size_hint(
                    k, fallback=config.unbacked_symint_fallback  # type: ignore[arg-type]
                )
            ),
            min_block_size_k,
        )
        used = OrderedSet[tuple[int, int, int, int, int, int]]()
        for block_m, block_n, block_k, num_stages, num_warps in configs:
            # shrink configs for small sizes
            block_m = max(min(int(block_m * scale), m), min_block_size)
            block_n = max(min(int(block_n * scale), n), min_block_size)
            block_k = max(min(int(block_k * scale), k), min_block_size_k)

            if exclude(block_m, block_n, block_k):
                continue

            # each warp computes 16x16 tile = 256
            num_warps = min(num_warps, block_m * block_n // 256)
            
            if (block_m, block_n, block_k, num_stages, num_warps, 0) not in used and (
                max_mm_configs is None or len(used) < max_mm_configs
            ):
                used.add((block_m, block_n, block_k, num_stages, num_warps, 0))
                yield self.triton_config(
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    num_stages=num_stages,
                    num_warps=num_warps,
                )
 
    def triton_config(self, num_stages, num_warps, **kwargs):
        from triton import Config  # type: ignore[attr-defined]
        return Config(kwargs, num_stages=num_stages, num_warps=num_warps)
 
    def get_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.mm_configs)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_exhaustive_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.exhaustive_configs)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_extra_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.extra_mm_configs)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_int8_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.int8_mm_configs)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_mixed_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.mixed_mm_configs)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_persistent_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.persistent_mm_configs)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_scaled_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.scaled_mm_configs)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_scaled_persistent_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.scaled_persistent_mm_configs)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_mm_plus_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self.mm_plus_mm_configs
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_conv_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.conv_configs)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)


class CPUConfigHeuristic(BaseConfigHeuristic):
    pass


class CUDAConfigHeuristic(BaseConfigHeuristic):
    pass


class ROCmConfigHeuristic(BaseConfigHeuristic):
    """
    Abstract interface for device specific matmul config heuristics
    """

    from .utils import get_backend_num_stages

    default_num_stages = get_backend_num_stages()

    def _filter_configs(self, configs, num_stages):
        return tuple(
            cast(tuple[int, int, int, int, int], config["config"])
            for config in configs
            if config["cond"]
        )

        
    def _preprocess_mm_configs(
        m: int,
        n: int,
        k: int,
        configs: Sequence[tuple[int, int, int, int, int]],
        has_int8_tensor=False,
        scale=1,
        exclude=lambda m, n, k: False,
    ):
        """
        Heuristic to preprocess configs e.g. shrink when they are bigger than the input size

        :param scale: scale factor applied to the config values
        :param exclude: whether a given config should be excluded
        """
        from torch._inductor import config

        max_mm_configs = config.test_configs.max_mm_configs

        min_block_size = 16
        # block_k=16 seems to be causing issues
        # see: https://github.com/triton-lang/triton/issues/2156#issuecomment-1695897424
        min_block_size_k = 32 if has_int8_tensor else 16
        m = max(
            next_power_of_2(
                V.graph.sizevars.size_hint(
                    m, fallback=torch._inductor.config.unbacked_symint_fallback  # type: ignore[arg-type]
                )
            ),
            min_block_size,
        )
        n = max(
            next_power_of_2(
                V.graph.sizevars.size_hint(
                    n, fallback=torch._inductor.config.unbacked_symint_fallback  # type: ignore[arg-type]
                )
            ),
            min_block_size,
        )
        k = max(
            next_power_of_2(
                V.graph.sizevars.size_hint(
                    k, fallback=torch._inductor.config.unbacked_symint_fallback  # type: ignore[arg-type]
                )
            ),
            min_block_size_k,
        )
        used = OrderedSet[tuple[int, int, int, int, int, int]]()
        for block_m, block_n, block_k, num_stages, num_warps in configs:
            # shrink configs for small sizes
            block_m = max(min(int(block_m * scale), m), min_block_size)
            block_n = max(min(int(block_n * scale), n), min_block_size)
            block_k = max(min(int(block_k * scale), k), min_block_size_k)

            if exclude(block_m, block_n, block_k):
                continue

            # each warp computes 16x16 tile = 256
            num_warps = min(num_warps, block_m * block_n // 256)
            
            for matrix_instr_nonkdim in [0, 16]:
                if matrix_instr_nonkdim != 0 and (
                    block_m % matrix_instr_nonkdim != 0
                    or block_n % matrix_instr_nonkdim != 0
                ):
                    #  block_m and block_n must be a multiple of matrix_instr_nonkdim
                    continue
                if (
                    block_m,
                    block_n,
                    block_k,
                    num_stages,
                    num_warps,
                    matrix_instr_nonkdim,
                ) not in used and (
                    max_mm_configs is None or len(used) < max_mm_configs
                ):
                    used.add(
                        (
                            block_m,
                            block_n,
                            block_k,
                            num_stages,
                            num_warps,
                            matrix_instr_nonkdim,
                        )
                    )
                    yield triton_config(
                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                        BLOCK_K=block_k,
                        num_stages=num_stages,
                        num_warps=num_warps,
                        matrix_instr_nonkdim=matrix_instr_nonkdim,
                    )

    def get_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.mm_configs, num_stages=self.default_num_stages)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_exhaustive_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.exhaustive_configs, num_stages=self.default_num_stages)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_extra_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.extra_mm_configs, num_stages=self.default_num_stages)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_int8_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.int8_mm_configs, num_stages=self.default_num_stages)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_mixed_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.mixed_mm_configs, num_stages=self.default_num_stages)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_persistent_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.persistent_mm_configs, num_stages=self.default_num_stages)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_scaled_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.scaled_mm_configs, num_stages=self.default_num_stages)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_scaled_persistent_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self._filter_configs(self.scaled_persistent_mm_configs, num_stages=self.default_num_stages)
        return partial(self._preprocess_mm_configs, configs=filtered_configs)

    def get_mm_plus_mm_configs(self) -> List[Dict[str, Any]]:
        filtered_configs = self.mm_plus_mm_configs
        return partial(self._preprocess_mm_configs, configs=filtered_configs)


class XPUConfigHeuristic(BaseConfigHeuristic):
    pass

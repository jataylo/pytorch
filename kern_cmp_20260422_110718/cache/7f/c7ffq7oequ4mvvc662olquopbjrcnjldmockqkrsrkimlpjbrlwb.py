
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 4, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_std_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 3, 'num_reduction': 3, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.00013112, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_std_3(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp14_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp14_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp14_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((3 + ks0*ks1*ks2) // 4)
        tmp1 = ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tmp2.to(tl.int1)
        tmp4 = tl.load(in_ptr0 + (((r0_1 + x0*((3 + ks0*ks1*ks2) // 4)) % (ks0*ks1*ks2))), r0_mask & tmp3 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp3, tmp5, tmp6)
        tmp8 = 1.0
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp3, tmp8, tmp9)
        tmp11 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp12 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp13 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp14_mean_next, tmp14_m2_next, tmp14_weight_next = triton_helpers.welford_combine(
            tmp14_mean, tmp14_m2, tmp14_weight,
            tmp11, tmp12, tmp13
        )
        tmp14_mean = tl.where(r0_mask & xmask, tmp14_mean_next, tmp14_mean)
        tmp14_m2 = tl.where(r0_mask & xmask, tmp14_m2_next, tmp14_m2)
        tmp14_weight = tl.where(r0_mask & xmask, tmp14_weight_next, tmp14_weight)
    tmp15, tmp16, tmp17 = triton_helpers.welford(tmp14_mean, tmp14_m2, tmp14_weight, 1)
    tmp14 = tmp15[:, None]
    tmp18 = tmp16[:, None]
    tmp19 = tmp17[:, None]
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp18, xmask)
    tl.store(out_ptr2 + (x0), tmp19, xmask)


def get_args():
    arg_0 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((4,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((4,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = 32
    arg_5 = 32
    arg_6 = 32
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 4, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_std_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_std_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.00013112
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.000524304, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp38 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
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
        tmp5 = tl.full([1, 1], 0, tl.int32)
        tmp6 = tl.maximum(tmp5, tmp4, tl.PropagateNan.ALL)
        tmp7 = tl.load(in_ptr1 + (((r0_1 + x0*((3 + ks0*ks1*ks2) // 4)) % (ks0*ks1*ks2))), r0_mask & tmp3 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.sigmoid(tmp7)
        tmp9 = tmp6 * tmp8
        tmp10 = tl.load(in_ptr2 + (((r0_1 + x0*((3 + ks0*ks1*ks2) // 4)) % (ks0*ks1*ks2))), r0_mask & tmp3 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = libdevice.tanh(tmp10)
        tmp12 = tl.load(in_ptr3 + (((r0_1 + x0*((3 + ks0*ks1*ks2) // 4)) % (ks0*ks1*ks2))), r0_mask & tmp3 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = 0.5
        tmp14 = tmp12 * tmp13
        tmp15 = 0.7071067811865476
        tmp16 = tmp12 * tmp15
        tmp17 = libdevice.erf(tmp16)
        tmp18 = 1.0
        tmp19 = tmp17 + tmp18
        tmp20 = tmp14 * tmp19
        tmp21 = tmp11 * tmp20
        tmp22 = tmp9 + tmp21
        tmp23 = tl_math.abs(tmp22)
        tmp24 = 1e-06
        tmp25 = tmp23 + tmp24
        tmp26 = tl.sqrt_rn(tmp25)
        tmp27 = tmp23 + tmp26
        tmp28 = tl_math.sin(tmp27)
        tmp29 = tl_math.cos(tmp22)
        tmp30 = tmp28 * tmp29
        tmp31 = tl_math.abs(tmp30)
        tmp32 = -tmp31
        tmp33 = libdevice.exp(tmp32)
        tmp34 = tmp30 + tmp33
        tmp35 = tl.full(tmp34.shape, 0, tmp34.dtype)
        tmp36 = tl.where(tmp3, tmp34, tmp35)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(r0_mask & xmask, tmp39, _tmp38)
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp38, xmask)


def get_args():
    arg_0 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((4,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = 32
    arg_6 = 32
    arg_7 = 32
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 4, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.000524304
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")

# AOT ID: ['46_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/om/comtoyq25v3xeulumpgwui54p2fkgimkizsbrtayh5orx73wtrct.py
# Topologically Sorted Source Nodes: [mul, h1, h1_act, mul_1, h2, residual], Original ATen: [aten.mul, aten.add, aten.gelu]
# Source node to ATen node mapping:
#   h1 => add_3
#   h1_act => add_7, erf, mul_5, mul_6, mul_7
#   h2 => add_14
#   mul => mul
#   mul_1 => mul_10
#   residual => add_18
# Graph fragment:
#   %arg2_1 : Tensor "f32[s23, s12][s12, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[s23, s12][s12, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[s23, s12][s12, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[s23, s12][s12, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[s23, s12][s12, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %mul : Tensor "f32[s23, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, %arg3_1), kwargs = {})
#   %add_3 : Tensor "f32[s23, s12][s12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %arg4_1), kwargs = {})
#   %mul_5 : Tensor "f32[s23, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 0.5), kwargs = {})
#   %mul_6 : Tensor "f32[s23, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[s23, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_7 : Tensor "f32[s23, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_7 : Tensor "f32[s23, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_7), kwargs = {})
#   %mul_10 : Tensor "f32[s23, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %arg5_1), kwargs = {})
#   %add_14 : Tensor "f32[s23, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %arg6_1), kwargs = {})
#   %add_18 : Tensor "f32[s23, s12][s12, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %add_14), kwargs = {})
#   return %add_18
triton_poi_fused_add_gelu_mul_0 = async_compile.triton('triton_poi_fused_add_gelu_mul_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.100663296, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_gelu_mul_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    tmp13 = tl.load(in_ptr3 + (x0), xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x0), tmp17, xmask)


def get_args():
    arg_0 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 4194304,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_gelu_mul_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_gelu_mul_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.100663296
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/z3/cz3yqmgynx3f447w47e2ed445trhrryr5wjhbncb7fobgrk6zijj.py
# Topologically Sorted Source Nodes: [mean, std], Original ATen: [aten.mean, aten.std]
# Source node to ATen node mapping:
#   mean => mean
#   std => var
# Graph fragment:
#   %add_18 : Tensor "f32[s23, s12][s12, 1]cuda:0" = PlaceHolder[target=add_18]
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_18,), kwargs = {})
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%add_18,), kwargs = {correction: 1.0})
#   return %buf1,%buf3,%buf4,%buf5
triton_red_fused_mean_std_1 = async_compile.triton('triton_red_fused_mean_std_1', '''
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
    size_hints={'x': 512, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_std_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 4, 'num_reduction': 4, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.016785408, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_mean_std_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp16_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp16_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp16_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((511 + ks0*ks1) // 512)
        tmp1 = ks0*ks1
        tmp2 = tmp0 < tmp1
        tmp3 = tmp2.to(tl.int1)
        tmp4 = tl.load(in_ptr0 + (((r0_1 + x0*((511 + ks0*ks1) // 512)) % (ks0*ks1))), r0_mask & tmp3 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tmp8 = 0.0
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp3, tmp8, tmp9)
        tmp11 = 1.0
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp3, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp15 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16_mean_next, tmp16_m2_next, tmp16_weight_next = triton_helpers.welford_combine(
            tmp16_mean, tmp16_m2, tmp16_weight,
            tmp5, tmp14, tmp15
        )
        tmp16_mean = tl.where(r0_mask & xmask, tmp16_mean_next, tmp16_mean)
        tmp16_m2 = tl.where(r0_mask & xmask, tmp16_m2_next, tmp16_m2)
        tmp16_weight = tl.where(r0_mask & xmask, tmp16_weight_next, tmp16_weight)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp17, tmp18, tmp19 = triton_helpers.welford(tmp16_mean, tmp16_m2, tmp16_weight, 1)
    tmp16 = tmp17[:, None]
    tmp20 = tmp18[:, None]
    tmp21 = tmp19[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr3 + (x0), tmp21, xmask)


def get_args():
    arg_0 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = 2048
    arg_6 = 2048
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 512, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_std_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_mean_std_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.016785408
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/65/c65f7eotyy4bwhqoxujl5tckqpulh6vehnonr2utrcuivhf6ydha.py
# Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean
# Graph fragment:
#   %buf1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=buf1]
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_18,), kwargs = {})
#   return %buf2
triton_per_fused_mean_2 = async_compile.triton('triton_per_fused_mean_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 2048}, 'kernel_num_gb': 2.052e-06, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_mean_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp3, None)


def get_args():
    arg_0 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1, 512,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_mean_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 2.052e-06
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/zr/czrggnmu4oau47s6fjdmlglr3pdtzjqfmlfgphjttwkj4yirzsop.py
# Topologically Sorted Source Nodes: [std], Original ATen: [aten.std]
# Source node to ATen node mapping:
#   std => var
# Graph fragment:
#   %buf3 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=buf3]
#   %buf4 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=buf4]
#   %buf5 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=buf5]
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%add_18,), kwargs = {correction: 1.0})
#   return %buf7
triton_per_fused_std_3 = async_compile.triton('triton_per_fused_std_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_std_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 6144}, 'kernel_num_gb': 6.148e-06, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_std_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.load(in_ptr1 + (r0_0), None)
    tmp2 = tl.load(in_ptr2 + (r0_0), None)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp3, tmp4, tmp5, 1)
    tmp10 = tmp7[:, None]
    tmp11 = tmp8[:, None]
    tmp12 = tmp9[:, None]
    tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp11, None)


def get_args():
    arg_0 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 1, 512,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_std_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_std_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 6.148e-06
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/lu/clurh5j3rvyzd5fd77bkcf7ithxrdiwb3u74udcvph5ggekjfhrl.py
# Topologically Sorted Source Nodes: [mean, sub, std, add_3, normalized], Original ATen: [aten.mean, aten.sub, aten.std, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_3 => add_25
#   mean => mean
#   normalized => div
#   std => sqrt, var
#   sub => sub_12
# Graph fragment:
#   %add_18 : Tensor "f32[s23, s12][s12, 1]cuda:0" = PlaceHolder[target=add_18]
#   %buf2 : Tensor "f32[][]cuda:0" = PlaceHolder[target=buf2]
#   %buf7 : Tensor "f32[][]cuda:0" = PlaceHolder[target=buf7]
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_18,), kwargs = {})
#   %sub_12 : Tensor "f32[s23, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_18, %mean), kwargs = {})
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%add_18,), kwargs = {correction: 1.0})
#   %sqrt : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%var,), kwargs = {})
#   %add_25 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt, 1e-06), kwargs = {})
#   %div : Tensor "f32[s23, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_12, %add_25), kwargs = {})
#   return %div
triton_poi_fused_add_div_mean_std_sub_4 = async_compile.triton('triton_poi_fused_add_div_mean_std_sub_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_std_sub_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.03355444, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_std_sub_4(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp3 = ks0*ks1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = (tmp2 / tmp4)
    tmp6 = tmp0 - tmp5
    tmp9 = 1.0
    tmp10 = tmp4 - tmp9
    tmp11 = 0.0
    tmp12 = tl.maximum(tmp11, tmp10, tl.PropagateNan.ALL)
    tmp13 = (tmp8 / tmp12)
    tmp14 = tl.sqrt_rn(tmp13)
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = (tmp6 / tmp16)
    tl.store(in_out_ptr0 + (x0), tmp17, xmask)


def get_args():
    arg_0 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_3 = 2048
    arg_4 = 2048
    return arg_0, arg_1, arg_2, arg_3, arg_4, 4194304,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_std_sub_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_div_mean_std_sub_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.03355444
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
        args.clear()
        s77 = arg0_1
        s27 = arg1_1
        arg2_1_size = arg2_1.size()
        s23 = arg2_1_size[0]
        s12 = arg2_1_size[1]
        assert_size_stride(arg2_1, (s23, s12), (s12, 1))
        assert_size_stride(arg3_1, (s23, s12), (s12, 1))
        assert_size_stride(arg4_1, (s23, s12), (s12, 1))
        assert_size_stride(arg5_1, (s23, s12), (s12, 1))
        assert_size_stride(arg6_1, (s23, s12), (s12, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s23, s12), (s12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul, h1, h1_act, mul_1, h2, residual], Original ATen: [aten.mul, aten.add, aten.gelu]
            triton_poi_fused_add_gelu_mul_0_xnumel = s12*s23
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_gelu_mul_0.run(arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, buf0, triton_poi_fused_add_gelu_mul_0_xnumel, stream=stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            buf1 = empty_strided_cuda((512, ), (1, ), torch.float32)
            buf3 = empty_strided_cuda((512, ), (1, ), torch.float32)
            buf4 = empty_strided_cuda((512, ), (1, ), torch.float32)
            buf5 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [mean, std], Original ATen: [aten.mean, aten.std]
            triton_red_fused_mean_std_1_r0_numel = (511 + s12*s23) // 512
            stream0 = get_raw_stream(0)
            triton_red_fused_mean_std_1.run(buf0, buf1, buf3, buf4, buf5, s12, s23, 512, triton_red_fused_mean_std_1_r0_numel, stream=stream0)
            buf2 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_mean_2.run(buf1, buf2, 1, 512, stream=stream0)
            del buf1
            buf7 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [std], Original ATen: [aten.std]
            stream0 = get_raw_stream(0)
            triton_per_fused_std_3.run(buf3, buf4, buf5, buf7, 1, 512, stream=stream0)
            del buf3
            del buf4
            del buf5
            buf9 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [mean, sub, std, add_3, normalized], Original ATen: [aten.mean, aten.sub, aten.std, aten.add, aten.div]
            triton_poi_fused_add_div_mean_std_sub_4_xnumel = s12*s23
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_div_mean_std_sub_4.run(buf9, buf2, buf7, s12, s23, triton_poi_fused_add_div_mean_std_sub_4_xnumel, stream=stream0)
            del buf2
            del buf7
        return (buf9, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 2048
    arg1_1 = 2048
    arg2_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

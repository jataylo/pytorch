# AOT ID: ['50_inference']
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


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/or/corlbepnhjbonxpokdwio2avcstn2lujdm4zwn6futht4tzw24wp.py
# Topologically Sorted Source Nodes: [mul, mul_1, scores_masked, max_1], Original ATen: [aten.mul, aten.add, aten.max]
# Source node to ATen node mapping:
#   max_1 => max_1
#   mul => mul
#   mul_1 => mul_5
#   scores_masked => add_9
# Graph fragment:
#   %arg2_1 : Tensor "f32[s31, s25][s25, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg4_1 : Tensor "f32[s31, s25][s25, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[s31, s25][s25, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %mul : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, %arg4_1), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg1_1,), kwargs = {})
#   %convert_element_type_default : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%scalar_tensor_default, torch.float64), kwargs = {})
#   %full_default : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.5), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %pow_tensor_tensor : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%convert_element_type_default, %full_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_tensor_tensor, torch.float32), kwargs = {})
#   %div_tensor : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, %convert_element_type_default_1), kwargs = {})
#   %mul_5 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg5_1, -1000000000.0), kwargs = {})
#   %add_9 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_tensor, %mul_5), kwargs = {})
#   %max_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%add_9,), kwargs = {})
#   return %buf1
triton_red_fused_add_max_mul_0 = async_compile.triton('triton_red_fused_add_max_mul_0', '''
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
    size_hints={'x': 32, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_max_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.003145856, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_add_max_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp20 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((31 + ks0*ks1) // 32)
        tmp1 = ks0*ks1
        tmp2 = tmp0 < tmp1
        tmp3 = tmp2.to(tl.int1)
        tmp4 = tl.load(in_ptr0 + (((r0_1 + x0*((31 + ks0*ks1) // 32)) % (ks0*ks1))), r0_mask & tmp3 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + (((r0_1 + x0*((31 + ks0*ks1) // 32)) % (ks0*ks1))), r0_mask & tmp3 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(ks0, [XBLOCK, R0_BLOCK])
        tmp8 = tmp7.to(tl.float64)
        tmp9 = tl.full([1, 1], 0.5, tl.float64)
        tmp10 = libdevice.pow(tmp8, tmp9)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = (tmp6 / tmp11)
        tmp13 = tl.load(in_ptr2 + (((r0_1 + x0*((31 + ks0*ks1) // 32)) % (ks0*ks1))), r0_mask & tmp3 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = -1000000000.0
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 + tmp15
        tmp17 = tl.full(tmp16.shape, float("-inf"), tmp16.dtype)
        tmp18 = tl.where(tmp3, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp21 = tl.maximum(_tmp20, tmp19, tl.PropagateNan.ALL)
        _tmp20 = tl.where(r0_mask & xmask, tmp21, _tmp20)
    tmp20 = triton_helpers.max2(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp20, xmask)


def get_args():
    arg_0 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = 512
    arg_5 = 512
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_max_mul_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_add_max_mul_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.003145856
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/m4/cm4xuo6k7nbyqikaxayyl4ahykoayghtym4efhjdkfhdsytrmofy.py
# Topologically Sorted Source Nodes: [mul, mul_1, scores_masked, max_1], Original ATen: [aten.mul, aten.add, aten.max]
# Source node to ATen node mapping:
#   max_1 => max_1
#   mul => mul
#   mul_1 => mul_5
#   scores_masked => add_9
# Graph fragment:
#   %buf1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=buf1]
#   %mul : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, %arg4_1), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg1_1,), kwargs = {})
#   %convert_element_type_default : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%scalar_tensor_default, torch.float64), kwargs = {})
#   %full_default : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.5), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %pow_tensor_tensor : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%convert_element_type_default, %full_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_tensor_tensor, torch.float32), kwargs = {})
#   %div_tensor : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, %convert_element_type_default_1), kwargs = {})
#   %mul_5 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg5_1, -1000000000.0), kwargs = {})
#   %add_9 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_tensor, %mul_5), kwargs = {})
#   %max_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%add_9,), kwargs = {})
#   return %max_1
triton_per_fused_add_max_mul_1 = async_compile.triton('triton_per_fused_add_max_mul_1', '''
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
    size_hints={'x': 1, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_max_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 128}, 'kernel_num_gb': 1.32e-07, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_add_max_mul_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
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
    tmp3 = triton_helpers.max2(tmp1, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp3, None)


def get_args():
    arg_0 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1, 32,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_add_max_mul_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_add_max_mul_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 1.32e-07
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/rx/crxgm7hemz5gn2xbimjw6rrzst5ckcfbrg6c66eynt7bg53tvglw.py
# Topologically Sorted Source Nodes: [mul, mul_1, scores_masked, sub, scores_exp, scores_sum, add_1, attn_weights, output], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten.sum, aten.div]
# Source node to ATen node mapping:
#   add_1 => add_22
#   attn_weights => div_1
#   mul => mul
#   mul_1 => mul_5
#   output => mul_18
#   scores_exp => exp
#   scores_masked => add_9
#   scores_sum => sum_1
#   sub => sub_8
# Graph fragment:
#   %arg2_1 : Tensor "f32[s31, s25][s25, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg4_1 : Tensor "f32[s31, s25][s25, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[s31, s25][s25, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %add_9 : Tensor "f32[s31, s25][s25, 1]cuda:0" = PlaceHolder[target=add_9]
#   %max_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=max_1]
#   %sum_1 : Tensor "f32[s31, 1][1, s31]cuda:0" = PlaceHolder[target=sum_1]
#   %arg6_1 : Tensor "f32[s31, s25][s25, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %mul : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, %arg4_1), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg1_1,), kwargs = {})
#   %convert_element_type_default : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%scalar_tensor_default, torch.float64), kwargs = {})
#   %full_default : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.5), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %pow_tensor_tensor : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%convert_element_type_default, %full_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_tensor_tensor, torch.float32), kwargs = {})
#   %div_tensor : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, %convert_element_type_default_1), kwargs = {})
#   %mul_5 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg5_1, -1000000000.0), kwargs = {})
#   %add_9 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_tensor, %mul_5), kwargs = {})
#   %sub_8 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %max_1), kwargs = {})
#   %exp : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_8,), kwargs = {})
#   %sum_1 : Tensor "f32[s31, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %add_22 : Tensor "f32[s31, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-09), kwargs = {})
#   %div_1 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %add_22), kwargs = {})
#   %mul_18 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %arg6_1), kwargs = {})
#   return %add_9,%sum_1,%mul_18
triton_red_fused_add_div_exp_mul_sub_sum_2 = async_compile.triton('triton_red_fused_add_div_exp_mul_sub_sum_2', '''
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
    size_hints={'x': 512, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_exp_mul_sub_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 3, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.006293508, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_add_div_exp_mul_sub_sum_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp13 = tl.load(in_ptr3 + (0))
    tmp14 = tl.broadcast_to(tmp13, [1, 1])
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = ks0
        tmp4 = tmp3.to(tl.float64)
        tmp5 = tl.full([1, 1], 0.5, tl.float64)
        tmp6 = libdevice.pow(tmp4, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = (tmp2 / tmp7)
        tmp10 = -1000000000.0
        tmp11 = tmp9 * tmp10
        tmp12 = tmp8 + tmp11
        tmp15 = tmp12 - tmp14
        tmp16 = libdevice.exp(tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask & xmask, tmp19, _tmp18)
        tl.store(out_ptr0 + (r0_1 + ks0*x0), tmp12, r0_mask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp18, xmask)
    tmp21 = tl.load(in_ptr3 + (0))
    tmp22 = tl.broadcast_to(tmp21, [1, 1])
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp20 = tl.load(out_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr4 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tmp20 - tmp22
        tmp24 = libdevice.exp(tmp23)
        tmp25 = 1e-09
        tmp26 = tmp18 + tmp25
        tmp27 = (tmp24 / tmp26)
        tmp29 = tmp27 * tmp28
        tl.store(out_ptr2 + (r0_1 + ks0*x0), tmp29, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((512, 1), (1, 512), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_8 = 512
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 512, 512,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_exp_mul_sub_sum_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_add_div_exp_mul_sub_sum_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.006293508
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/ym/cymz5log7gqbxcammlpigeumtxcxb5x2rgs6pepac36c7cc4zlf6.py
# Topologically Sorted Source Nodes: [sub, scores_exp, add_1, attn_weights, output, mean], Original ATen: [aten.sub, aten.exp, aten.add, aten.div, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   add_1 => add_22
#   attn_weights => div_1
#   mean => mean
#   output => mul_18
#   scores_exp => exp
#   sub => sub_8
# Graph fragment:
#   %add_9 : Tensor "f32[s31, s25][s25, 1]cuda:0" = PlaceHolder[target=add_9]
#   %max_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=max_1]
#   %sum_1 : Tensor "f32[s31, 1][1, s31]cuda:0" = PlaceHolder[target=sum_1]
#   %arg6_1 : Tensor "f32[s31, s25][s25, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %sub_8 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %max_1), kwargs = {})
#   %exp : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_8,), kwargs = {})
#   %add_22 : Tensor "f32[s31, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-09), kwargs = {})
#   %div_1 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %add_22), kwargs = {})
#   %mul_18 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %arg6_1), kwargs = {})
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul_18,), kwargs = {})
#   return %buf5
triton_red_fused_add_div_exp_mean_mul_sub_3 = async_compile.triton('triton_red_fused_add_div_exp_mean_mul_sub_3', '''
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
    size_hints={'x': 32, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_exp_mean_mul_sub_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.002099332, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_add_div_exp_mean_mul_sub_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((31 + ks0*ks1) // 32)
        tmp1 = ks0*ks1
        tmp2 = tmp0 < tmp1
        tmp3 = tmp2.to(tl.int1)
        tmp4 = tl.load(in_ptr0 + (((r0_1 + x0*((31 + ks0*ks1) // 32)) % (ks0*ks1))), r0_mask & tmp3 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + (0))
        tmp6 = tl.broadcast_to(tmp5, [1, 1])
        tmp7 = tl.where(tmp3, tmp6, 0.0)
        tmp8 = tmp4 - tmp7
        tmp9 = libdevice.exp(tmp8)
        tmp10 = tl.load(in_ptr2 + ((((r0_1 + x0*((31 + ks0*ks1) // 32)) // ks0) % ks1)), r0_mask & tmp3 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = 1e-09
        tmp12 = tmp10 + tmp11
        tmp13 = (tmp9 / tmp12)
        tmp14 = tl.load(in_ptr3 + (((r0_1 + x0*((31 + ks0*ks1) // 32)) % (ks0*ks1))), r0_mask & tmp3 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 * tmp14
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp3, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp19, xmask)


def get_args():
    arg_0 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512, 1), (1, 512), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = 512
    arg_6 = 512
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 32, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_exp_mean_mul_sub_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_add_div_exp_mean_mul_sub_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.002099332
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/q7/cq7zjxbqk3sfpouw2wovavvomq35tuyjhompdya6byaegr3ra7ps.py
# Topologically Sorted Source Nodes: [sub, scores_exp, add_1, attn_weights, output, mean], Original ATen: [aten.sub, aten.exp, aten.add, aten.div, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   add_1 => add_22
#   attn_weights => div_1
#   mean => mean
#   output => mul_18
#   scores_exp => exp
#   sub => sub_8
# Graph fragment:
#   %buf5 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=buf5]
#   %sub_8 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %max_1), kwargs = {})
#   %exp : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_8,), kwargs = {})
#   %add_22 : Tensor "f32[s31, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-09), kwargs = {})
#   %div_1 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %add_22), kwargs = {})
#   %mul_18 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %arg6_1), kwargs = {})
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul_18,), kwargs = {})
#   return %buf6
triton_per_fused_add_div_exp_mean_mul_sub_4 = async_compile.triton('triton_per_fused_add_div_exp_mean_mul_sub_4', '''
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
    size_hints={'x': 1, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_exp_mean_mul_sub_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 128}, 'kernel_num_gb': 1.32e-07, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_add_div_exp_mean_mul_sub_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
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
    arg_0 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1, 32,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_exp_mean_mul_sub_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_add_div_exp_mean_mul_sub_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 1.32e-07
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/v3/cv3jzzrpga5wtjwsgsppljhmolban7cuq62d6xmrgla5pltwcbvi.py
# Topologically Sorted Source Nodes: [var], Original ATen: [aten.var]
# Source node to ATen node mapping:
#   var => var
# Graph fragment:
#   %mul_18 : Tensor "f32[s31, s25][s25, 1]cuda:0" = PlaceHolder[target=mul_18]
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%mul_18,), kwargs = {})
#   return %buf7,%buf8,%buf9
triton_red_fused_var_5 = async_compile.triton('triton_red_fused_var_5', '''
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
    size_hints={'x': 32, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_var_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 3, 'num_reduction': 3, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.00104896, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_var_5(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32
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
        tmp0 = r0_1 + x0*((31 + ks0*ks1) // 32)
        tmp1 = ks0*ks1
        tmp2 = tmp0 < tmp1
        tmp3 = tmp2.to(tl.int1)
        tmp4 = tl.load(in_ptr0 + (((r0_1 + x0*((31 + ks0*ks1) // 32)) % (ks0*ks1))), r0_mask & tmp3 & xmask, eviction_policy='evict_last', other=0.0)
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
    arg_0 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = 512
    arg_5 = 512
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_var_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_var_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.00104896
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/oh/coh6bjbujlhznv5hp3lkjgl76btltabpzxqkckuv47cqn6isyx5l.py
# Topologically Sorted Source Nodes: [var], Original ATen: [aten.var]
# Source node to ATen node mapping:
#   var => var
# Graph fragment:
#   %buf7 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=buf7]
#   %buf8 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=buf8]
#   %buf9 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=buf9]
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%mul_18,), kwargs = {})
#   return %buf11
triton_per_fused_var_6 = async_compile.triton('triton_per_fused_var_6', '''
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
    size_hints={'x': 1, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_var_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 384}, 'kernel_num_gb': 3.88e-07, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_var_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
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
    arg_0 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 1, 32,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_var_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_var_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 3.88e-07
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/oe/coej7ak7vigsp2uxfswlyxahuypul3hzszxbkdlki26bcmkspr3q.py
# Topologically Sorted Source Nodes: [sub, scores_exp, add_1, attn_weights, output, mean, sub_1, var, add_2, sqrt, normalized], Original ATen: [aten.sub, aten.exp, aten.add, aten.div, aten.mul, aten.mean, aten.var, aten.sqrt]
# Source node to ATen node mapping:
#   add_1 => add_22
#   add_2 => add_35
#   attn_weights => div_1
#   mean => mean
#   normalized => div_2
#   output => mul_18
#   scores_exp => exp
#   sqrt => sqrt
#   sub => sub_8
#   sub_1 => sub_19
#   var => var
# Graph fragment:
#   %mul_18 : Tensor "f32[s31, s25][s25, 1]cuda:0" = PlaceHolder[target=mul_18]
#   %buf6 : Tensor "f32[][]cuda:0" = PlaceHolder[target=buf6]
#   %buf11 : Tensor "f32[][]cuda:0" = PlaceHolder[target=buf11]
#   %sub_8 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %max_1), kwargs = {})
#   %exp : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_8,), kwargs = {})
#   %add_22 : Tensor "f32[s31, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-09), kwargs = {})
#   %div_1 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %add_22), kwargs = {})
#   %mul_18 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %arg6_1), kwargs = {})
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul_18,), kwargs = {})
#   %sub_19 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_18, %mean), kwargs = {})
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%mul_18,), kwargs = {})
#   %add_35 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%var, 1e-06), kwargs = {})
#   %sqrt : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_35,), kwargs = {})
#   %div_2 : Tensor "f32[s31, s25][s25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_19, %sqrt), kwargs = {})
#   return %div_2
triton_poi_fused_add_div_exp_mean_mul_sqrt_sub_var_7 = async_compile.triton('triton_poi_fused_add_div_exp_mean_mul_sqrt_sub_var_7', '''
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
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_exp_mean_mul_sqrt_sub_var_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.00209716, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_exp_mean_mul_sqrt_sub_var_7(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sqrt_rn(tmp15)
    tmp17 = (tmp6 / tmp16)
    tl.store(in_out_ptr0 + (x0), tmp17, xmask)


def get_args():
    arg_0 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_3 = 512
    arg_4 = 512
    return arg_0, arg_1, arg_2, arg_3, arg_4, 262144,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_exp_mean_mul_sqrt_sub_var_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_div_exp_mean_mul_sqrt_sub_var_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.00209716
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
        s85 = arg0_1
        s34 = arg1_1
        s25 = arg3_1
        arg2_1_size = arg2_1.size()
        s31 = arg2_1_size[0]
        assert_size_stride(arg2_1, (s31, s25), (s25, 1))
        assert_size_stride(arg4_1, (s31, s25), (s25, 1))
        assert_size_stride(arg5_1, (s31, s25), (s25, 1))
        assert_size_stride(arg6_1, (s31, s25), (s25, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((32, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [mul, mul_1, scores_masked, max_1], Original ATen: [aten.mul, aten.add, aten.max]
            triton_red_fused_add_max_mul_0_r0_numel = (31 + s25*s31) // 32
            stream0 = get_raw_stream(0)
            triton_red_fused_add_max_mul_0.run(arg2_1, arg4_1, arg5_1, buf1, s25, s31, 32, triton_red_fused_add_max_mul_0_r0_numel, stream=stream0)
            buf2 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [mul, mul_1, scores_masked, max_1], Original ATen: [aten.mul, aten.add, aten.max]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_max_mul_1.run(buf1, buf2, 1, 32, stream=stream0)
            buf0 = empty_strided_cuda((s31, s25), (s25, 1), torch.float32)
            buf3 = empty_strided_cuda((s31, 1), (1, s31), torch.float32)
            buf4 = empty_strided_cuda((s31, s25), (s25, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul, mul_1, scores_masked, sub, scores_exp, scores_sum, add_1, attn_weights, output], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten.sum, aten.div]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_div_exp_mul_sub_sum_2.run(arg2_1, arg4_1, arg5_1, buf2, arg6_1, buf0, buf3, buf4, s25, s31, s25, stream=stream0)
            del arg2_1
            del arg4_1
            del arg5_1
            buf5 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [sub, scores_exp, add_1, attn_weights, output, mean], Original ATen: [aten.sub, aten.exp, aten.add, aten.div, aten.mul, aten.mean]
            triton_red_fused_add_div_exp_mean_mul_sub_3_r0_numel = (31 + s25*s31) // 32
            stream0 = get_raw_stream(0)
            triton_red_fused_add_div_exp_mean_mul_sub_3.run(buf0, buf2, buf3, arg6_1, buf5, s25, s31, 32, triton_red_fused_add_div_exp_mean_mul_sub_3_r0_numel, stream=stream0)
            del arg6_1
            del buf0
            del buf3
            buf6 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [sub, scores_exp, add_1, attn_weights, output, mean], Original ATen: [aten.sub, aten.exp, aten.add, aten.div, aten.mul, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_div_exp_mean_mul_sub_4.run(buf5, buf6, 1, 32, stream=stream0)
            buf7 = buf5; del buf5  # reuse
            buf8 = empty_strided_cuda((32, ), (1, ), torch.float32)
            buf9 = empty_strided_cuda((32, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [var], Original ATen: [aten.var]
            triton_red_fused_var_5_r0_numel = (31 + s25*s31) // 32
            stream0 = get_raw_stream(0)
            triton_red_fused_var_5.run(buf4, buf7, buf8, buf9, s25, s31, 32, triton_red_fused_var_5_r0_numel, stream=stream0)
            buf11 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [var], Original ATen: [aten.var]
            stream0 = get_raw_stream(0)
            triton_per_fused_var_6.run(buf7, buf8, buf9, buf11, 1, 32, stream=stream0)
            del buf7
            del buf8
            del buf9
            buf13 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [sub, scores_exp, add_1, attn_weights, output, mean, sub_1, var, add_2, sqrt, normalized], Original ATen: [aten.sub, aten.exp, aten.add, aten.div, aten.mul, aten.mean, aten.var, aten.sqrt]
            triton_poi_fused_add_div_exp_mean_mul_sqrt_sub_var_7_xnumel = s25*s31
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_div_exp_mean_mul_sqrt_sub_var_7.run(buf13, buf6, buf11, s25, s31, triton_poi_fused_add_div_exp_mean_mul_sqrt_sub_var_7_xnumel, stream=stream0)
            del buf11
            del buf6
        return (buf13, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 512
    arg1_1 = 512
    arg2_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = 512
    arg4_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

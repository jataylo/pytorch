# AOT ID: ['49_inference']
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


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/2j/c2jmdnkj7ugi5ubl4r7r7z5wqwbdrevsgg7mfnsdsgvdzht6kwpe.py
# Topologically Sorted Source Nodes: [mul, mul_1, scores_masked, max_1], Original ATen: [aten.mul, aten.add, aten.max]
# Source node to ATen node mapping:
#   max_1 => max_1
#   mul => mul
#   mul_1 => mul_3
#   scores_masked => add_6
# Graph fragment:
#   %arg1_1 : Tensor "f32[s31][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[s31][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg4_1 : Tensor "f32[s31][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %mul : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg0_1,), kwargs = {})
#   %convert_element_type_default : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%scalar_tensor_default, torch.float64), kwargs = {})
#   %full_default : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.5), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %pow_tensor_tensor : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%convert_element_type_default, %full_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_tensor_tensor, torch.float32), kwargs = {})
#   %div_tensor : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, %convert_element_type_default_1), kwargs = {})
#   %mul_3 : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, -1000000000.0), kwargs = {})
#   %add_6 : Tensor "f32[s31][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_tensor, %mul_3), kwargs = {})
#   %max_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%add_6,), kwargs = {})
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
    size_hints={'x': 128, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_max_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.012583424, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_add_max_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = r0_1 + x0*((127 + ks0) // 128)
        tmp1 = ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tmp2.to(tl.int1)
        tmp4 = tl.load(in_ptr0 + (r0_1 + x0*((127 + ks0) // 128)), r0_mask & tmp3 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + (r0_1 + x0*((127 + ks0) // 128)), r0_mask & tmp3 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(ks0, [XBLOCK, R0_BLOCK])
        tmp8 = tmp7.to(tl.float64)
        tmp9 = tl.full([1, 1], 0.5, tl.float64)
        tmp10 = libdevice.pow(tmp8, tmp9)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = (tmp6 / tmp11)
        tmp13 = tl.load(in_ptr2 + (r0_1 + x0*((127 + ks0) // 128)), r0_mask & tmp3 & xmask, eviction_policy='evict_first', other=0.0)
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
    arg_0 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = 1048576
    return arg_0, arg_1, arg_2, arg_3, arg_4, 128, 8192,


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
    num_gb = 0.012583424
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/zw/czwu7gyx5y74jzfybj4ekzmer5scka4tu3o3wrbe2gojqnlgnyei.py
# Topologically Sorted Source Nodes: [mul, mul_1, scores_masked, max_1], Original ATen: [aten.mul, aten.add, aten.max]
# Source node to ATen node mapping:
#   max_1 => max_1
#   mul => mul
#   mul_1 => mul_3
#   scores_masked => add_6
# Graph fragment:
#   %buf1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=buf1]
#   %mul : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg0_1,), kwargs = {})
#   %convert_element_type_default : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%scalar_tensor_default, torch.float64), kwargs = {})
#   %full_default : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.5), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %pow_tensor_tensor : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%convert_element_type_default, %full_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_tensor_tensor, torch.float32), kwargs = {})
#   %div_tensor : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, %convert_element_type_default_1), kwargs = {})
#   %mul_3 : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, -1000000000.0), kwargs = {})
#   %add_6 : Tensor "f32[s31][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_tensor, %mul_3), kwargs = {})
#   %max_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%add_6,), kwargs = {})
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
    size_hints={'x': 1, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_max_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 512}, 'kernel_num_gb': 5.16e-07, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_add_max_mul_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
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
    arg_0 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1, 128,


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
    num_gb = 5.16e-07
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/mz/cmzmcctlpkrilupsao2emevvtb3zfkizzn7arjevgfqdctectxzy.py
# Topologically Sorted Source Nodes: [mul, mul_1, scores_masked, sub, scores_exp], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp]
# Source node to ATen node mapping:
#   mul => mul
#   mul_1 => mul_3
#   scores_exp => exp
#   scores_masked => add_6
#   sub => sub_4
# Graph fragment:
#   %arg1_1 : Tensor "f32[s31][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[s31][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg4_1 : Tensor "f32[s31][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %add_6 : Tensor "f32[s31][1]cuda:0" = PlaceHolder[target=add_6]
#   %max_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=max_1]
#   %mul : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg0_1,), kwargs = {})
#   %convert_element_type_default : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%scalar_tensor_default, torch.float64), kwargs = {})
#   %full_default : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.5), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %pow_tensor_tensor : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%convert_element_type_default, %full_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_tensor_tensor, torch.float32), kwargs = {})
#   %div_tensor : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, %convert_element_type_default_1), kwargs = {})
#   %mul_3 : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, -1000000000.0), kwargs = {})
#   %add_6 : Tensor "f32[s31][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_tensor, %mul_3), kwargs = {})
#   %sub_4 : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %max_1), kwargs = {})
#   %exp : Tensor "f32[s31][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_4,), kwargs = {})
#   return %add_6,%exp
triton_poi_fused_add_exp_mul_sub_2 = async_compile.triton('triton_poi_fused_add_exp_mul_sub_2', '''
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
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_mul_sub_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.020971524, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_exp_mul_sub_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp13 = tl.load(in_ptr3 + (0))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp2 = tmp0 * tmp1
    tmp3 = ks0
    tmp4 = tmp3.to(tl.float64)
    tmp5 = tl.full([1], 0.5, tl.float64)
    tmp6 = libdevice.pow(tmp4, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = (tmp2 / tmp7)
    tmp10 = -1000000000.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp15 = tmp12 - tmp14
    tmp16 = libdevice.exp(tmp15)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)


def get_args():
    arg_0 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = 1048576
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 1048576,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_exp_mul_sub_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_exp_mul_sub_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.020971524
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/4d/c4daiazeqkhahfscdiqmzb4tzk65q6ugtt6qwfijcmkudiv7wcyq.py
# Topologically Sorted Source Nodes: [sub, scores_exp, scores_sum], Original ATen: [aten.sub, aten.exp, aten.sum]
# Source node to ATen node mapping:
#   scores_exp => exp
#   scores_sum => sum_1
#   sub => sub_4
# Graph fragment:
#   %add_6 : Tensor "f32[s31][1]cuda:0" = PlaceHolder[target=add_6]
#   %max_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=max_1]
#   %sub_4 : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %max_1), kwargs = {})
#   %exp : Tensor "f32[s31][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_4,), kwargs = {})
#   %sum_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   return %buf4
triton_red_fused_exp_sub_sum_3 = async_compile.triton('triton_red_fused_exp_sub_sum_3', '''
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
    size_hints={'x': 128, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_exp_sub_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.00419482, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_exp_sub_sum_3(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((127 + ks0) // 128)
        tmp1 = ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tmp2.to(tl.int1)
        tmp4 = tl.load(in_ptr0 + (r0_1 + x0*((127 + ks0) // 128)), r0_mask & tmp3 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + (0))
        tmp6 = tl.broadcast_to(tmp5, [1, 1])
        tmp7 = tl.where(tmp3, tmp6, 0.0)
        tmp8 = tmp4 - tmp7
        tmp9 = libdevice.exp(tmp8)
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp3, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)


def get_args():
    arg_0 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = 1048576
    return arg_0, arg_1, arg_2, arg_3, 128, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_exp_sub_sum_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_exp_sub_sum_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.00419482
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/ln/clniqoblbeu6iddby36hxfojdsotjdc63vaif5ftcangfswpwhnt.py
# Topologically Sorted Source Nodes: [sub, scores_exp, scores_sum], Original ATen: [aten.sub, aten.exp, aten.sum]
# Source node to ATen node mapping:
#   scores_exp => exp
#   scores_sum => sum_1
#   sub => sub_4
# Graph fragment:
#   %buf4 : Tensor "f32[1, 128][128, 1]cuda:0" = PlaceHolder[target=buf4]
#   %sub_4 : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %max_1), kwargs = {})
#   %exp : Tensor "f32[s31][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_4,), kwargs = {})
#   %sum_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   return %sum_1
triton_per_fused_exp_sub_sum_4 = async_compile.triton('triton_per_fused_exp_sub_sum_4', '''
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
    size_hints={'x': 1, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_exp_sub_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 512}, 'kernel_num_gb': 5.16e-07, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_exp_sub_sum_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
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
    arg_0 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1, 128,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_exp_sub_sum_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_exp_sub_sum_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 5.16e-07
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/an/cannnngqrtp33qout6wpu2yzh6vbevpn3wyttfe4ntwht4mi7p4k.py
# Topologically Sorted Source Nodes: [add_1, attn_weights, output], Original ATen: [aten.add, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add_1 => add_13
#   attn_weights => div_1
#   output => mul_9
# Graph fragment:
#   %exp : Tensor "f32[s31][1]cuda:0" = PlaceHolder[target=exp]
#   %sum_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sum_1]
#   %arg5_1 : Tensor "f32[s31][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %add_13 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-09), kwargs = {})
#   %div_1 : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %add_13), kwargs = {})
#   %mul_9 : Tensor "f32[s31][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %arg5_1), kwargs = {})
#   return %mul_9
triton_poi_fused_add_div_mul_5 = async_compile.triton('triton_poi_fused_add_div_mul_5', '''
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
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.012582916, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp3 = 1e-09
    tmp4 = tmp2 + tmp3
    tmp5 = (tmp0 / tmp4)
    tmp7 = tmp5 * tmp6
    tl.store(out_ptr0 + (x0), tmp7, xmask)


def get_args():
    arg_0 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 1048576,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_div_mul_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.012582916
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/y3/cy3te3xpwmfebqgm3kuiq5kfalkxn3zdossgyqlwpg6lh3umycno.py
# Topologically Sorted Source Nodes: [add_1, attn_weights, output, mean], Original ATen: [aten.add, aten.div, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   add_1 => add_13
#   attn_weights => div_1
#   mean => mean
#   output => mul_9
# Graph fragment:
#   %exp : Tensor "f32[s31][1]cuda:0" = PlaceHolder[target=exp]
#   %sum_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sum_1]
#   %arg5_1 : Tensor "f32[s31][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %add_13 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-09), kwargs = {})
#   %div_1 : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %add_13), kwargs = {})
#   %mul_9 : Tensor "f32[s31][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %arg5_1), kwargs = {})
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul_9,), kwargs = {})
#   return %buf7
triton_red_fused_add_div_mean_mul_6 = async_compile.triton('triton_red_fused_add_div_mean_mul_6', '''
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
    size_hints={'x': 128, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mean_mul_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.008389124, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_add_div_mean_mul_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((127 + ks0) // 128)
        tmp1 = ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tmp2.to(tl.int1)
        tmp4 = tl.load(in_ptr0 + (r0_1 + x0*((127 + ks0) // 128)), r0_mask & tmp3 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + (0))
        tmp6 = tl.broadcast_to(tmp5, [1, 1])
        tmp7 = tl.where(tmp3, tmp6, 0.0)
        tmp8 = 1e-09
        tmp9 = tmp7 + tmp8
        tmp10 = (tmp4 / tmp9)
        tmp11 = tl.load(in_ptr2 + (r0_1 + x0*((127 + ks0) // 128)), r0_mask & tmp3 & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tmp10 * tmp11
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp3, tmp12, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp16, xmask)


def get_args():
    arg_0 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = 1048576
    return arg_0, arg_1, arg_2, arg_3, arg_4, 128, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_mean_mul_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_add_div_mean_mul_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.008389124
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/ge/cgeiaagrlvjw523fceurn3e6ucmtmsfndpzphth7l7djeg4x3ws2.py
# Topologically Sorted Source Nodes: [add_1, attn_weights, output, mean], Original ATen: [aten.add, aten.div, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   add_1 => add_13
#   attn_weights => div_1
#   mean => mean
#   output => mul_9
# Graph fragment:
#   %buf7 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=buf7]
#   %add_13 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-09), kwargs = {})
#   %div_1 : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %add_13), kwargs = {})
#   %mul_9 : Tensor "f32[s31][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %arg5_1), kwargs = {})
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul_9,), kwargs = {})
#   return %buf8
triton_per_fused_add_div_mean_mul_7 = async_compile.triton('triton_per_fused_add_div_mean_mul_7', '''
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
    size_hints={'x': 1, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 512}, 'kernel_num_gb': 5.16e-07, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_add_div_mean_mul_7(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
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
    arg_0 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1, 128,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mean_mul_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_add_div_mean_mul_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 5.16e-07
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/gu/cgutizvjwoytvzyw27v4cmts3nymkhja5x2e5nucp5yfoli44fvi.py
# Topologically Sorted Source Nodes: [var], Original ATen: [aten.var]
# Source node to ATen node mapping:
#   var => var
# Graph fragment:
#   %mul_9 : Tensor "f32[s31][1]cuda:0" = PlaceHolder[target=mul_9]
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%mul_9,), kwargs = {})
#   return %buf9,%buf10,%buf11
triton_red_fused_var_8 = async_compile.triton('triton_red_fused_var_8', '''
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
    size_hints={'x': 128, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_var_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 3, 'num_reduction': 3, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.00419584, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_var_8(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = r0_1 + x0*((127 + ks0) // 128)
        tmp1 = ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tmp2.to(tl.int1)
        tmp4 = tl.load(in_ptr0 + (r0_1 + x0*((127 + ks0) // 128)), r0_mask & tmp3 & xmask, eviction_policy='evict_first', other=0.0)
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
    arg_0 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = 1048576
    return arg_0, arg_1, arg_2, arg_3, arg_4, 128, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_var_8.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_var_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.00419584
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/7n/c7n2to6ta3hbitnyejds2ikdemeo2qhechiohrkb3l2zaypsow7b.py
# Topologically Sorted Source Nodes: [var], Original ATen: [aten.var]
# Source node to ATen node mapping:
#   var => var
# Graph fragment:
#   %buf9 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=buf9]
#   %buf10 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=buf10]
#   %buf11 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=buf11]
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%mul_9,), kwargs = {})
#   return %buf13
triton_per_fused_var_9 = async_compile.triton('triton_per_fused_var_9', '''
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
    size_hints={'x': 1, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_var_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 1536}, 'kernel_num_gb': 1.54e-06, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_var_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
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
    arg_0 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 1, 128,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_var_9.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_var_9.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 1.54e-06
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/5h/c5hasrxn7f33kepz5papdg22sfr27yjrffhly6ndyhofnd3e2kb2.py
# Topologically Sorted Source Nodes: [add_1, attn_weights, output, mean, sub_1, var, add_2, sqrt, normalized], Original ATen: [aten.add, aten.div, aten.mul, aten.mean, aten.sub, aten.var, aten.sqrt]
# Source node to ATen node mapping:
#   add_1 => add_13
#   add_2 => add_20
#   attn_weights => div_1
#   mean => mean
#   normalized => div_2
#   output => mul_9
#   sqrt => sqrt
#   sub_1 => sub_9
#   var => var
# Graph fragment:
#   %mul_9 : Tensor "f32[s31][1]cuda:0" = PlaceHolder[target=mul_9]
#   %buf8 : Tensor "f32[][]cuda:0" = PlaceHolder[target=buf8]
#   %buf13 : Tensor "f32[][]cuda:0" = PlaceHolder[target=buf13]
#   %add_13 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-09), kwargs = {})
#   %div_1 : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %add_13), kwargs = {})
#   %mul_9 : Tensor "f32[s31][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %arg5_1), kwargs = {})
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul_9,), kwargs = {})
#   %sub_9 : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_9, %mean), kwargs = {})
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%mul_9,), kwargs = {})
#   %add_20 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%var, 1e-06), kwargs = {})
#   %sqrt : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_20,), kwargs = {})
#   %div_2 : Tensor "f32[s31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_9, %sqrt), kwargs = {})
#   return %div_2
triton_poi_fused_add_div_mean_mul_sqrt_sub_var_10 = async_compile.triton('triton_poi_fused_add_div_mean_mul_sqrt_sub_var_10', '''
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
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_mul_sqrt_sub_var_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.008388616, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_mul_sqrt_sub_var_10(in_out_ptr0, in_ptr0, in_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp3 = ks0
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
    arg_0 = rand_strided((1048576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_3 = 1048576
    return arg_0, arg_1, arg_2, arg_3, 1048576,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_mul_sqrt_sub_var_10.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_div_mean_mul_sqrt_sub_var_10.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.008388616
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
        args.clear()
        s85 = arg0_1
        s31 = arg3_1
        assert_size_stride(arg1_1, (s31, ), (1, ))
        assert_size_stride(arg2_1, (s31, ), (1, ))
        assert_size_stride(arg4_1, (s31, ), (1, ))
        assert_size_stride(arg5_1, (s31, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [mul, mul_1, scores_masked, max_1], Original ATen: [aten.mul, aten.add, aten.max]
            triton_red_fused_add_max_mul_0_r0_numel = (127 + s31) // 128
            stream0 = get_raw_stream(0)
            triton_red_fused_add_max_mul_0.run(arg1_1, arg2_1, arg4_1, buf1, s31, 128, triton_red_fused_add_max_mul_0_r0_numel, stream=stream0)
            buf2 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [mul, mul_1, scores_masked, max_1], Original ATen: [aten.mul, aten.add, aten.max]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_max_mul_1.run(buf1, buf2, 1, 128, stream=stream0)
            buf0 = empty_strided_cuda((s31, ), (1, ), torch.float32)
            buf3 = empty_strided_cuda((s31, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [mul, mul_1, scores_masked, sub, scores_exp], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_exp_mul_sub_2.run(arg1_1, arg2_1, arg4_1, buf2, buf0, buf3, s31, s31, stream=stream0)
            del arg1_1
            del arg2_1
            del arg4_1
            buf4 = reinterpret_tensor(buf1, (1, 128), (128, 1), 0); del buf1  # reuse
            # Topologically Sorted Source Nodes: [sub, scores_exp, scores_sum], Original ATen: [aten.sub, aten.exp, aten.sum]
            triton_red_fused_exp_sub_sum_3_r0_numel = (127 + s31) // 128
            stream0 = get_raw_stream(0)
            triton_red_fused_exp_sub_sum_3.run(buf0, buf2, buf4, s31, 128, triton_red_fused_exp_sub_sum_3_r0_numel, stream=stream0)
            buf5 = reinterpret_tensor(buf2, (1, ), (1, ), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [sub, scores_exp, scores_sum], Original ATen: [aten.sub, aten.exp, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused_exp_sub_sum_4.run(buf4, buf5, 1, 128, stream=stream0)
            buf6 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [add_1, attn_weights, output], Original ATen: [aten.add, aten.div, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_div_mul_5.run(buf3, buf5, arg5_1, buf6, s31, stream=stream0)
            buf7 = reinterpret_tensor(buf4, (128, ), (1, ), 0); del buf4  # reuse
            # Topologically Sorted Source Nodes: [add_1, attn_weights, output, mean], Original ATen: [aten.add, aten.div, aten.mul, aten.mean]
            triton_red_fused_add_div_mean_mul_6_r0_numel = (127 + s31) // 128
            stream0 = get_raw_stream(0)
            triton_red_fused_add_div_mean_mul_6.run(buf3, buf5, arg5_1, buf7, s31, 128, triton_red_fused_add_div_mean_mul_6_r0_numel, stream=stream0)
            del arg5_1
            del buf3
            buf8 = reinterpret_tensor(buf5, (), (), 0); del buf5  # reuse
            # Topologically Sorted Source Nodes: [add_1, attn_weights, output, mean], Original ATen: [aten.add, aten.div, aten.mul, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_div_mean_mul_7.run(buf7, buf8, 1, 128, stream=stream0)
            buf9 = buf7; del buf7  # reuse
            buf10 = empty_strided_cuda((128, ), (1, ), torch.float32)
            buf11 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [var], Original ATen: [aten.var]
            triton_red_fused_var_8_r0_numel = (127 + s31) // 128
            stream0 = get_raw_stream(0)
            triton_red_fused_var_8.run(buf6, buf9, buf10, buf11, s31, 128, triton_red_fused_var_8_r0_numel, stream=stream0)
            buf13 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [var], Original ATen: [aten.var]
            stream0 = get_raw_stream(0)
            triton_per_fused_var_9.run(buf9, buf10, buf11, buf13, 1, 128, stream=stream0)
            del buf10
            del buf11
            del buf9
            buf15 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [add_1, attn_weights, output, mean, sub_1, var, add_2, sqrt, normalized], Original ATen: [aten.add, aten.div, aten.mul, aten.mean, aten.sub, aten.var, aten.sqrt]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_div_mean_mul_sqrt_sub_var_10.run(buf15, buf8, buf13, s31, s31, stream=stream0)
            del buf13
            del buf8
        return (buf15, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 1048576
    arg1_1 = rand_strided((1048576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1048576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = 1048576
    arg4_1 = rand_strided((1048576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1048576, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

# AOT ID: ['48_inference']
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


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/wk/cwkvfjnvqsd7eq26abuetiowlgboro73wjq4itvzsoooapsalfmd.py
# Topologically Sorted Source Nodes: [mul, scores, mul_1, scores_masked, max_1], Original ATen: [aten.mul, aten.div, aten.add, aten.max]
# Source node to ATen node mapping:
#   max_1 => max_1
#   mul => mul
#   mul_1 => mul_1
#   scores => div
#   scores_masked => add
# Graph fragment:
#   %arg0_1 : Tensor "f32[65536][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[65536][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[65536][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %mul : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %div : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 256.0), kwargs = {})
#   %mul_1 : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, -1000000000.0), kwargs = {})
#   %add : Tensor "f32[65536][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %mul_1), kwargs = {})
#   %max_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%add,), kwargs = {})
#   return %buf1
triton_red_fused_add_div_max_mul_0 = async_compile.triton('triton_red_fused_add_div_max_mul_0', '''
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
    size_hints={'x': 8, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_max_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 64, 'r0_': 786432}, 'kernel_num_gb': 0.000786464, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_add_div_max_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = 0.00390625
        tmp4 = tmp2 * tmp3
        tmp6 = -1000000000.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = tl.maximum(_tmp10, tmp9, tl.PropagateNan.ALL)
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = triton_helpers.max2(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)


def get_args():
    arg_0 = rand_strided((65536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((65536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((65536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 8, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_max_mul_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_add_div_max_mul_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.000786464
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/gc/cgctmxg6yemswth46rlwkmg4ni7ff7txhqanfm2hh6dou2oc5oto.py
# Topologically Sorted Source Nodes: [mul, scores, mul_1, scores_masked, max_1], Original ATen: [aten.mul, aten.div, aten.add, aten.max]
# Source node to ATen node mapping:
#   max_1 => max_1
#   mul => mul
#   mul_1 => mul_1
#   scores => div
#   scores_masked => add
# Graph fragment:
#   %buf1 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=buf1]
#   %mul : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %div : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 256.0), kwargs = {})
#   %mul_1 : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, -1000000000.0), kwargs = {})
#   %add : Tensor "f32[65536][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %mul_1), kwargs = {})
#   %max_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%add,), kwargs = {})
#   return %max_1
triton_per_fused_add_div_max_mul_1 = async_compile.triton('triton_per_fused_add_div_max_mul_1', '''
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
    size_hints={'x': 1, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_max_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 32}, 'kernel_num_gb': 3.6e-08, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_add_div_max_mul_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
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
    arg_0 = rand_strided((8,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1, 8,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_max_mul_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_add_div_max_mul_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 3.6e-08
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/6m/c6mjnjak63sfzrhseasg2lgdybatgrcyldhnj2fw5ncp2kt2it7x.py
# Topologically Sorted Source Nodes: [mul, scores, mul_1, scores_masked, sub, scores_exp, scores_sum], Original ATen: [aten.mul, aten.div, aten.add, aten.sub, aten.exp, aten.sum]
# Source node to ATen node mapping:
#   mul => mul
#   mul_1 => mul_1
#   scores => div
#   scores_exp => exp
#   scores_masked => add
#   scores_sum => sum_1
#   sub => sub
# Graph fragment:
#   %arg0_1 : Tensor "f32[65536][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[65536][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[65536][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %add : Tensor "f32[65536][1]cuda:0" = PlaceHolder[target=add]
#   %max_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=max_1]
#   %mul : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %div : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 256.0), kwargs = {})
#   %mul_1 : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, -1000000000.0), kwargs = {})
#   %add : Tensor "f32[65536][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %mul_1), kwargs = {})
#   %sub : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %max_1), kwargs = {})
#   %exp : Tensor "f32[65536][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   return %add,%exp,%buf4
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
    size_hints={'x': 8, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_exp_mul_sub_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 64, 'r0_': 1310720}, 'kernel_num_gb': 0.001048612, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_add_div_exp_mul_sub_sum_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [1, 1])
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = 0.00390625
        tmp4 = tmp2 * tmp3
        tmp6 = -1000000000.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp4 + tmp7
        tmp11 = tmp8 - tmp10
        tmp12 = libdevice.exp(tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
        tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp12, r0_mask & xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp14, xmask)


def get_args():
    arg_0 = rand_strided((65536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((65536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((65536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((65536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((1, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 8, 8192,


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
    num_gb = 0.001048612
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/5s/c5svr2ejkdcsm3pkyhox55niztgqfx4owqkeoza2k4nskeifwyni.py
# Topologically Sorted Source Nodes: [sub, scores_exp, scores_sum], Original ATen: [aten.sub, aten.exp, aten.sum]
# Source node to ATen node mapping:
#   scores_exp => exp
#   scores_sum => sum_1
#   sub => sub
# Graph fragment:
#   %buf4 : Tensor "f32[1, 8][8, 1]cuda:0" = PlaceHolder[target=buf4]
#   %sub : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %max_1), kwargs = {})
#   %exp : Tensor "f32[65536][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   return %sum_1
triton_per_fused_exp_sub_sum_3 = async_compile.triton('triton_per_fused_exp_sub_sum_3', '''
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
    size_hints={'x': 1, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_exp_sub_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 32}, 'kernel_num_gb': 3.6e-08, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_exp_sub_sum_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
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
    arg_0 = rand_strided((1, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1, 8,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_exp_sub_sum_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_exp_sub_sum_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 3.6e-08
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/3n/c3nbe6luf7zrq575jkrbgnlx3hbf6o4xy7pdgnqny2q6m47kwszc.py
# Topologically Sorted Source Nodes: [add_1, attn_weights, output, mean, var], Original ATen: [aten.add, aten.div, aten.mul, aten.mean, aten.var]
# Source node to ATen node mapping:
#   add_1 => add_1
#   attn_weights => div_1
#   mean => mean
#   output => mul_2
#   var => var
# Graph fragment:
#   %exp : Tensor "f32[65536][1]cuda:0" = PlaceHolder[target=exp]
#   %sum_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sum_1]
#   %arg3_1 : Tensor "f32[65536][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mul_2 : Tensor "f32[65536][1]cuda:0" = PlaceHolder[target=mul_2]
#   %add_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-09), kwargs = {})
#   %div_1 : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %add_1), kwargs = {})
#   %mul_2 : Tensor "f32[65536][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %arg3_1), kwargs = {})
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul_2,), kwargs = {})
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%mul_2,), kwargs = {})
#   return %mul_2,%buf7,%buf9,%buf10,%buf11
triton_red_fused_add_div_mean_mul_var_4 = async_compile.triton('triton_red_fused_add_div_mean_mul_var_4', '''
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
    size_hints={'x': 8, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mean_mul_var_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 5, 'num_reduction': 4, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 256, 'r0_': 1310720}, 'kernel_num_gb': 0.000786564, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_add_div_mean_mul_var_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [1, 1])
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp11_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = 1e-09
        tmp4 = tmp2 + tmp3
        tmp5 = (tmp0 / tmp4)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp11_mean, tmp11_m2, tmp11_weight, roffset == 0
        )
        tmp11_mean = tl.where(r0_mask & xmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(r0_mask & xmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(r0_mask & xmask, tmp11_weight_next, tmp11_weight)
        tl.store(out_ptr0 + (r0_1 + 8192*x0), tmp7, r0_mask & xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp12, tmp13, tmp14 = triton_helpers.welford(tmp11_mean, tmp11_m2, tmp11_weight, 1)
    tmp11 = tmp12[:, None]
    tmp15 = tmp13[:, None]
    tmp16 = tmp14[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tl.store(out_ptr2 + (x0), tmp11, xmask)
    tl.store(out_ptr3 + (x0), tmp15, xmask)
    tl.store(out_ptr4 + (x0), tmp16, xmask)


def get_args():
    arg_0 = rand_strided((65536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((65536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((65536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((8,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((8,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((8,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((8,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 8, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_mean_mul_var_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_add_div_mean_mul_var_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.000786564
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/jj/cjjpmr2fb7uoodczjd7slbd4qsqurgtqe65guejq4burx32nedlb.py
# Topologically Sorted Source Nodes: [add_1, attn_weights, output, mean], Original ATen: [aten.add, aten.div, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   add_1 => add_1
#   attn_weights => div_1
#   mean => mean
#   output => mul_2
# Graph fragment:
#   %buf7 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=buf7]
#   %add_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-09), kwargs = {})
#   %div_1 : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %add_1), kwargs = {})
#   %mul_2 : Tensor "f32[65536][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %arg3_1), kwargs = {})
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul_2,), kwargs = {})
#   return %buf8
triton_per_fused_add_div_mean_mul_5 = async_compile.triton('triton_per_fused_add_div_mean_mul_5', '''
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
    size_hints={'x': 1, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 32}, 'kernel_num_gb': 3.6e-08, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_add_div_mean_mul_5(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
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
    arg_0 = rand_strided((8,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1, 8,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mean_mul_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_add_div_mean_mul_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 3.6e-08
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/uw/cuw22dp4cqllncsmt3egaf3teieyrv5cbjbu6lua4gfhv47mw7ye.py
# Topologically Sorted Source Nodes: [var], Original ATen: [aten.var]
# Source node to ATen node mapping:
#   var => var
# Graph fragment:
#   %buf9 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=buf9]
#   %buf10 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=buf10]
#   %buf11 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=buf11]
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%mul_2,), kwargs = {})
#   return %buf13
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
    size_hints={'x': 1, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_var_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 96}, 'kernel_num_gb': 1e-07, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_var_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
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
    arg_0 = rand_strided((8,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 1, 8,


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
    num_gb = 1e-07
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/b5/cb5cs4lltgoafv5pn2u5zwhblrskelm3qkkv5mbjxj7lvk2p2e6x.py
# Topologically Sorted Source Nodes: [add_1, attn_weights, output, mean, sub_1, var, add_2, sqrt, normalized], Original ATen: [aten.add, aten.div, aten.mul, aten.mean, aten.sub, aten.var, aten.sqrt]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_2 => add_2
#   attn_weights => div_1
#   mean => mean
#   normalized => div_2
#   output => mul_2
#   sqrt => sqrt
#   sub_1 => sub_1
#   var => var
# Graph fragment:
#   %mul_2 : Tensor "f32[65536][1]cuda:0" = PlaceHolder[target=mul_2]
#   %buf8 : Tensor "f32[][]cuda:0" = PlaceHolder[target=buf8]
#   %buf13 : Tensor "f32[][]cuda:0" = PlaceHolder[target=buf13]
#   %add_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-09), kwargs = {})
#   %div_1 : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %add_1), kwargs = {})
#   %mul_2 : Tensor "f32[65536][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %arg3_1), kwargs = {})
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul_2,), kwargs = {})
#   %sub_1 : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2, %mean), kwargs = {})
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%mul_2,), kwargs = {})
#   %add_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%var, 1e-06), kwargs = {})
#   %sqrt : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %div_2 : Tensor "f32[65536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %sqrt), kwargs = {})
#   return %div_2
triton_poi_fused_add_div_mean_mul_sqrt_sub_var_7 = async_compile.triton('triton_poi_fused_add_div_mean_mul_sqrt_sub_var_7', '''
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
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_mul_sqrt_sub_var_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 786432}, 'kernel_num_gb': 0.000524296, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_mul_sqrt_sub_var_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp6 = tl.load(in_ptr1 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp3 = 65536.0
    tmp4 = (tmp2 / tmp3)
    tmp5 = tmp0 - tmp4
    tmp8 = 65535.0
    tmp9 = (tmp7 / tmp8)
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.sqrt_rn(tmp11)
    tmp13 = (tmp5 / tmp12)
    tl.store(in_out_ptr0 + (x0), tmp13, None)


def get_args():
    arg_0 = rand_strided((65536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 65536,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_mul_sqrt_sub_var_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_div_mean_mul_sqrt_sub_var_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.000524296
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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        assert_size_stride(arg0_1, (65536, ), (1, ))
        assert_size_stride(arg1_1, (65536, ), (1, ))
        assert_size_stride(arg2_1, (65536, ), (1, ))
        assert_size_stride(arg3_1, (65536, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((8, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [mul, scores, mul_1, scores_masked, max_1], Original ATen: [aten.mul, aten.div, aten.add, aten.max]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_div_max_mul_0.run(arg0_1, arg1_1, arg2_1, buf1, 8, 8192, stream=stream0)
            buf2 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [mul, scores, mul_1, scores_masked, max_1], Original ATen: [aten.mul, aten.div, aten.add, aten.max]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_div_max_mul_1.run(buf1, buf2, 1, 8, stream=stream0)
            buf3 = empty_strided_cuda((65536, ), (1, ), torch.float32)
            buf4 = reinterpret_tensor(buf1, (1, 8), (8, 1), 0); del buf1  # reuse
            # Topologically Sorted Source Nodes: [mul, scores, mul_1, scores_masked, sub, scores_exp, scores_sum], Original ATen: [aten.mul, aten.div, aten.add, aten.sub, aten.exp, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_div_exp_mul_sub_sum_2.run(arg0_1, arg1_1, arg2_1, buf2, buf3, buf4, 8, 8192, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            buf5 = reinterpret_tensor(buf2, (1, ), (1, ), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [sub, scores_exp, scores_sum], Original ATen: [aten.sub, aten.exp, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused_exp_sub_sum_3.run(buf4, buf5, 1, 8, stream=stream0)
            buf6 = empty_strided_cuda((65536, ), (1, ), torch.float32)
            buf7 = reinterpret_tensor(buf4, (8, ), (1, ), 0); del buf4  # reuse
            buf9 = empty_strided_cuda((8, ), (1, ), torch.float32)
            buf10 = empty_strided_cuda((8, ), (1, ), torch.float32)
            buf11 = empty_strided_cuda((8, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [add_1, attn_weights, output, mean, var], Original ATen: [aten.add, aten.div, aten.mul, aten.mean, aten.var]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_div_mean_mul_var_4.run(buf3, buf5, arg3_1, buf6, buf7, buf9, buf10, buf11, 8, 8192, stream=stream0)
            del arg3_1
            del buf3
            buf8 = reinterpret_tensor(buf5, (), (), 0); del buf5  # reuse
            # Topologically Sorted Source Nodes: [add_1, attn_weights, output, mean], Original ATen: [aten.add, aten.div, aten.mul, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_div_mean_mul_5.run(buf7, buf8, 1, 8, stream=stream0)
            del buf7
            buf13 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [var], Original ATen: [aten.var]
            stream0 = get_raw_stream(0)
            triton_per_fused_var_6.run(buf9, buf10, buf11, buf13, 1, 8, stream=stream0)
            del buf10
            del buf11
            del buf9
            buf15 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [add_1, attn_weights, output, mean, sub_1, var, add_2, sqrt, normalized], Original ATen: [aten.add, aten.div, aten.mul, aten.mean, aten.sub, aten.var, aten.sqrt]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_div_mean_mul_sqrt_sub_var_7.run(buf15, buf8, buf13, 65536, stream=stream0)
            del buf13
            del buf8
        return (buf15, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((65536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((65536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((65536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((65536, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

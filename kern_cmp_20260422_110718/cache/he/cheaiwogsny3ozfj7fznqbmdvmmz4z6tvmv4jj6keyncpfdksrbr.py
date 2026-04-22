# AOT ID: ['59_inference']
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


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/kw/ckw62ie7padwa6ynym4easjfodp3s5sfanpiq2uqrdxvnwqzw5ba.py
# Topologically Sorted Source Nodes: [a, b, mul, c, d, mul_1, e, f, abs_2, add_1, sqrt, g, h, cos, i, abs_3, neg, exp, j], Original ATen: [aten.relu, aten.sigmoid, aten.mul, aten.tanh, aten.gelu, aten.add, aten.abs, aten.sqrt, aten.sin, aten.cos, aten.neg, aten.exp]
# Source node to ATen node mapping:
#   a => relu
#   abs_2 => abs_2
#   abs_3 => abs_3
#   add_1 => add_38
#   b => sigmoid
#   c => tanh
#   cos => cos
#   d => add_12, erf, mul_10, mul_11, mul_9
#   e => add_25
#   exp => exp
#   f => abs_1
#   g => add_47
#   h => sin
#   i => mul_47
#   j => add_76
#   mul => mul_15
#   mul_1 => mul_19
#   neg => neg
#   sqrt => sqrt
# Graph fragment:
#   %arg3_1 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %relu : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%arg3_1,), kwargs = {})
#   %sigmoid : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg4_1,), kwargs = {})
#   %mul_15 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu, %sigmoid), kwargs = {})
#   %tanh : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%arg5_1,), kwargs = {})
#   %mul_9 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg6_1, 0.5), kwargs = {})
#   %mul_10 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg6_1, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_10,), kwargs = {})
#   %add_12 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_11 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %add_12), kwargs = {})
#   %mul_19 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %mul_11), kwargs = {})
#   %add_25 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %mul_19), kwargs = {})
#   %abs_1 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_25,), kwargs = {})
#   %abs_2 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_25,), kwargs = {})
#   %add_38 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_2, 1e-06), kwargs = {})
#   %sqrt : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_38,), kwargs = {})
#   %add_47 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, %sqrt), kwargs = {})
#   %sin : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%add_47,), kwargs = {})
#   %cos : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%add_25,), kwargs = {})
#   %mul_47 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, %cos), kwargs = {})
#   %abs_3 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_47,), kwargs = {})
#   %neg : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_3,), kwargs = {})
#   %exp : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_76 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %exp), kwargs = {})
#   return %add_76
triton_poi_fused_abs_add_cos_exp_gelu_mul_neg_relu_sigmoid_sin_sqrt_tanh_0 = async_compile.triton('triton_poi_fused_abs_add_cos_exp_gelu_mul_neg_relu_sigmoid_sin_sqrt_tanh_0', '''
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
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_cos_exp_gelu_mul_neg_relu_sigmoid_sin_sqrt_tanh_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.00065536, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_cos_exp_gelu_mul_neg_relu_sigmoid_sin_sqrt_tanh_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.maximum(tmp1, tmp0, tl.PropagateNan.ALL)
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tmp7 = libdevice.tanh(tmp6)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = 0.7071067811865476
    tmp12 = tmp8 * tmp11
    tmp13 = libdevice.erf(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp10 * tmp15
    tmp17 = tmp7 * tmp16
    tmp18 = tmp5 + tmp17
    tmp19 = tl_math.abs(tmp18)
    tmp20 = 1e-06
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt_rn(tmp21)
    tmp23 = tmp19 + tmp22
    tmp24 = tl_math.sin(tmp23)
    tmp25 = tl_math.cos(tmp18)
    tmp26 = tmp24 * tmp25
    tmp27 = tl_math.abs(tmp26)
    tmp28 = -tmp27
    tmp29 = libdevice.exp(tmp28)
    tmp30 = tmp26 + tmp29
    tl.store(out_ptr0 + (x0), tmp30, xmask)


def get_args():
    arg_0 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 32768,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_cos_exp_gelu_mul_neg_relu_sigmoid_sin_sqrt_tanh_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_abs_add_cos_exp_gelu_mul_neg_relu_sigmoid_sin_sqrt_tanh_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.00065536
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/57/c57v3qwodjqbmynonspryfn5xjxk5mxsiervhapggp6cych2ea7a.py
# Topologically Sorted Source Nodes: [a, b, mul, c, d, mul_1, e, f, abs_2, add_1, sqrt, g, h, cos, i, abs_3, neg, exp, j, mean], Original ATen: [aten.relu, aten.sigmoid, aten.mul, aten.tanh, aten.gelu, aten.add, aten.abs, aten.sqrt, aten.sin, aten.cos, aten.neg, aten.exp, aten.mean]
# Source node to ATen node mapping:
#   a => relu
#   abs_2 => abs_2
#   abs_3 => abs_3
#   add_1 => add_38
#   b => sigmoid
#   c => tanh
#   cos => cos
#   d => add_12, erf, mul_10, mul_11, mul_9
#   e => add_25
#   exp => exp
#   f => abs_1
#   g => add_47
#   h => sin
#   i => mul_47
#   j => add_76
#   mean => mean
#   mul => mul_15
#   mul_1 => mul_19
#   neg => neg
#   sqrt => sqrt
# Graph fragment:
#   %arg3_1 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %relu : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%arg3_1,), kwargs = {})
#   %sigmoid : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg4_1,), kwargs = {})
#   %mul_15 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu, %sigmoid), kwargs = {})
#   %tanh : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%arg5_1,), kwargs = {})
#   %mul_9 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg6_1, 0.5), kwargs = {})
#   %mul_10 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg6_1, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_10,), kwargs = {})
#   %add_12 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_11 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %add_12), kwargs = {})
#   %mul_19 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %mul_11), kwargs = {})
#   %add_25 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %mul_19), kwargs = {})
#   %abs_1 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_25,), kwargs = {})
#   %abs_2 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_25,), kwargs = {})
#   %add_38 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_2, 1e-06), kwargs = {})
#   %sqrt : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_38,), kwargs = {})
#   %add_47 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, %sqrt), kwargs = {})
#   %sin : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%add_47,), kwargs = {})
#   %cos : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%add_25,), kwargs = {})
#   %mul_47 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, %cos), kwargs = {})
#   %abs_3 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_47,), kwargs = {})
#   %neg : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_3,), kwargs = {})
#   %exp : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_76 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %exp), kwargs = {})
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_76,), kwargs = {})
#   return %buf1
triton_red_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_1 = async_compile.triton('triton_red_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_1', '''
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
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/x7/cx77h6mqwy3eq5m5rd6lgcov7gto2qmirlmsjfkp6p37thalkgms.py
# Topologically Sorted Source Nodes: [a, b, mul, c, d, mul_1, e, f, abs_2, add_1, sqrt, g, h, cos, i, abs_3, neg, exp, j, mean], Original ATen: [aten.relu, aten.sigmoid, aten.mul, aten.tanh, aten.gelu, aten.add, aten.abs, aten.sqrt, aten.sin, aten.cos, aten.neg, aten.exp, aten.mean]
# Source node to ATen node mapping:
#   a => relu
#   abs_2 => abs_2
#   abs_3 => abs_3
#   add_1 => add_38
#   b => sigmoid
#   c => tanh
#   cos => cos
#   d => add_12, erf, mul_10, mul_11, mul_9
#   e => add_25
#   exp => exp
#   f => abs_1
#   g => add_47
#   h => sin
#   i => mul_47
#   j => add_76
#   mean => mean
#   mul => mul_15
#   mul_1 => mul_19
#   neg => neg
#   sqrt => sqrt
# Graph fragment:
#   %buf1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=buf1]
#   %relu : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%arg3_1,), kwargs = {})
#   %sigmoid : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg4_1,), kwargs = {})
#   %mul_15 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu, %sigmoid), kwargs = {})
#   %tanh : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%arg5_1,), kwargs = {})
#   %mul_9 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg6_1, 0.5), kwargs = {})
#   %mul_10 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg6_1, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_10,), kwargs = {})
#   %add_12 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_11 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %add_12), kwargs = {})
#   %mul_19 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %mul_11), kwargs = {})
#   %add_25 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %mul_19), kwargs = {})
#   %abs_1 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_25,), kwargs = {})
#   %abs_2 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_25,), kwargs = {})
#   %add_38 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_2, 1e-06), kwargs = {})
#   %sqrt : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_38,), kwargs = {})
#   %add_47 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, %sqrt), kwargs = {})
#   %sin : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%add_47,), kwargs = {})
#   %cos : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%add_25,), kwargs = {})
#   %mul_47 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, %cos), kwargs = {})
#   %abs_3 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_47,), kwargs = {})
#   %neg : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_3,), kwargs = {})
#   %exp : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_76 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %exp), kwargs = {})
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_76,), kwargs = {})
#   return %buf2
triton_per_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_2 = async_compile.triton('triton_per_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_2', '''
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
    size_hints={'x': 1, 'r0_': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 8}, 'kernel_num_gb': 2e-08, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
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
    arg_0 = rand_strided((4,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1, 4,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 2e-08
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/7f/c7ffq7oequ4mvvc662olquopbjrcnjldmockqkrsrkimlpjbrlwb.py
# Topologically Sorted Source Nodes: [std], Original ATen: [aten.std]
# Source node to ATen node mapping:
#   std => var
# Graph fragment:
#   %add_76 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0" = PlaceHolder[target=add_76]
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%add_76,), kwargs = {correction: 1.0})
#   return %buf3,%buf4,%buf5
triton_red_fused_std_3 = async_compile.triton('triton_red_fused_std_3', '''
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
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/6v/c6vcfvmhq3tpkl6i72fdpkwlvdygv37qor7u5yctgy5viwy5df2f.py
# Topologically Sorted Source Nodes: [std], Original ATen: [aten.std]
# Source node to ATen node mapping:
#   std => var
# Graph fragment:
#   %buf3 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=buf3]
#   %buf4 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=buf4]
#   %buf5 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=buf5]
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%add_76,), kwargs = {correction: 1.0})
#   return %buf7
triton_per_fused_std_4 = async_compile.triton('triton_per_fused_std_4', '''
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
    size_hints={'x': 1, 'r0_': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_std_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 24}, 'kernel_num_gb': 5.2e-08, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_std_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
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
    arg_0 = rand_strided((4,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((4,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 1, 4,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_std_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_std_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 5.2e-08
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/sz/cszpv2hvwjcxieuydevbcd4fbik6feok2y2cs6bgbkcg5noi5gx5.py
# Topologically Sorted Source Nodes: [a, b, mul, c, d, mul_1, e, f, abs_2, add_1, sqrt, g, h, cos, i, abs_3, neg, exp, j, mean, sub, std, add_4, result], Original ATen: [aten.relu, aten.sigmoid, aten.mul, aten.tanh, aten.gelu, aten.add, aten.abs, aten.sqrt, aten.sin, aten.cos, aten.neg, aten.exp, aten.mean, aten.sub, aten.std, aten.div]
# Source node to ATen node mapping:
#   a => relu
#   abs_2 => abs_2
#   abs_3 => abs_3
#   add_1 => add_38
#   add_4 => add_85
#   b => sigmoid
#   c => tanh
#   cos => cos
#   d => add_12, erf, mul_10, mul_11, mul_9
#   e => add_25
#   exp => exp
#   f => abs_1
#   g => add_47
#   h => sin
#   i => mul_47
#   j => add_76
#   mean => mean
#   mul => mul_15
#   mul_1 => mul_19
#   neg => neg
#   result => div
#   sqrt => sqrt
#   std => sqrt_1, var
#   sub => sub_57
# Graph fragment:
#   %add_76 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0" = PlaceHolder[target=add_76]
#   %buf2 : Tensor "f32[][]cuda:0" = PlaceHolder[target=buf2]
#   %buf7 : Tensor "f32[][]cuda:0" = PlaceHolder[target=buf7]
#   %relu : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%arg3_1,), kwargs = {})
#   %sigmoid : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg4_1,), kwargs = {})
#   %mul_15 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu, %sigmoid), kwargs = {})
#   %tanh : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%arg5_1,), kwargs = {})
#   %mul_9 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg6_1, 0.5), kwargs = {})
#   %mul_10 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg6_1, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_10,), kwargs = {})
#   %add_12 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_11 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %add_12), kwargs = {})
#   %mul_19 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %mul_11), kwargs = {})
#   %add_25 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %mul_19), kwargs = {})
#   %abs_1 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_25,), kwargs = {})
#   %abs_2 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_25,), kwargs = {})
#   %add_38 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_2, 1e-06), kwargs = {})
#   %sqrt : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_38,), kwargs = {})
#   %add_47 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, %sqrt), kwargs = {})
#   %sin : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%add_47,), kwargs = {})
#   %cos : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%add_25,), kwargs = {})
#   %mul_47 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, %cos), kwargs = {})
#   %abs_3 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_47,), kwargs = {})
#   %neg : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_3,), kwargs = {})
#   %exp : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_76 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %exp), kwargs = {})
#   %mean : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_76,), kwargs = {})
#   %sub_57 : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_76, %mean), kwargs = {})
#   %var : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%add_76,), kwargs = {correction: 1.0})
#   %sqrt_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%var,), kwargs = {})
#   %add_85 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_1, 1e-06), kwargs = {})
#   %div : Tensor "f32[s24, s0, s1][s0*s1, s1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_57, %add_85), kwargs = {})
#   return %div
triton_poi_fused_abs_add_cos_div_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_std_sub_tanh_5 = async_compile.triton('triton_poi_fused_abs_add_cos_div_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_std_sub_tanh_5', '''
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
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_cos_div_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_std_sub_tanh_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.000262152, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_cos_div_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_std_sub_tanh_5(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp3 = ks0*ks1*ks2
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
    arg_0 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_3 = 32
    arg_4 = 32
    arg_5 = 32
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32768,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_cos_div_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_std_sub_tanh_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_abs_add_cos_div_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_std_sub_tanh_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.000262152
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
        s6 = arg0_1
        s0 = arg1_1
        s71 = arg2_1
        arg3_1_size = arg3_1.size()
        s24 = arg3_1_size[0]
        s1 = arg3_1_size[2]
        assert_size_stride(arg3_1, (s24, s0, s1), (s0*s1, s1, 1))
        assert_size_stride(arg4_1, (s24, s0, s1), (s0*s1, s1, 1))
        assert_size_stride(arg5_1, (s24, s0, s1), (s0*s1, s1, 1))
        assert_size_stride(arg6_1, (s24, s0, s1), (s0*s1, s1, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s24, s0, s1), (s0*s1, s1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [a, b, mul, c, d, mul_1, e, f, abs_2, add_1, sqrt, g, h, cos, i, abs_3, neg, exp, j], Original ATen: [aten.relu, aten.sigmoid, aten.mul, aten.tanh, aten.gelu, aten.add, aten.abs, aten.sqrt, aten.sin, aten.cos, aten.neg, aten.exp]
            triton_poi_fused_abs_add_cos_exp_gelu_mul_neg_relu_sigmoid_sin_sqrt_tanh_0_xnumel = s0*s1*s24
            stream0 = get_raw_stream(0)
            triton_poi_fused_abs_add_cos_exp_gelu_mul_neg_relu_sigmoid_sin_sqrt_tanh_0.run(arg3_1, arg4_1, arg5_1, arg6_1, buf0, triton_poi_fused_abs_add_cos_exp_gelu_mul_neg_relu_sigmoid_sin_sqrt_tanh_0_xnumel, stream=stream0)
            buf1 = empty_strided_cuda((4, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [a, b, mul, c, d, mul_1, e, f, abs_2, add_1, sqrt, g, h, cos, i, abs_3, neg, exp, j, mean], Original ATen: [aten.relu, aten.sigmoid, aten.mul, aten.tanh, aten.gelu, aten.add, aten.abs, aten.sqrt, aten.sin, aten.cos, aten.neg, aten.exp, aten.mean]
            triton_red_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_1_r0_numel = (3 + s0*s1*s24) // 4
            stream0 = get_raw_stream(0)
            triton_red_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_1.run(arg3_1, arg4_1, arg5_1, arg6_1, buf1, s0, s1, s24, 4, triton_red_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_1_r0_numel, stream=stream0)
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            buf2 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [a, b, mul, c, d, mul_1, e, f, abs_2, add_1, sqrt, g, h, cos, i, abs_3, neg, exp, j, mean], Original ATen: [aten.relu, aten.sigmoid, aten.mul, aten.tanh, aten.gelu, aten.add, aten.abs, aten.sqrt, aten.sin, aten.cos, aten.neg, aten.exp, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_abs_add_cos_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_tanh_2.run(buf1, buf2, 1, 4, stream=stream0)
            buf3 = buf1; del buf1  # reuse
            buf4 = empty_strided_cuda((4, ), (1, ), torch.float32)
            buf5 = empty_strided_cuda((4, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [std], Original ATen: [aten.std]
            triton_red_fused_std_3_r0_numel = (3 + s0*s1*s24) // 4
            stream0 = get_raw_stream(0)
            triton_red_fused_std_3.run(buf0, buf3, buf4, buf5, s0, s1, s24, 4, triton_red_fused_std_3_r0_numel, stream=stream0)
            buf7 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [std], Original ATen: [aten.std]
            stream0 = get_raw_stream(0)
            triton_per_fused_std_4.run(buf3, buf4, buf5, buf7, 1, 4, stream=stream0)
            del buf3
            del buf4
            del buf5
            buf9 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [a, b, mul, c, d, mul_1, e, f, abs_2, add_1, sqrt, g, h, cos, i, abs_3, neg, exp, j, mean, sub, std, add_4, result], Original ATen: [aten.relu, aten.sigmoid, aten.mul, aten.tanh, aten.gelu, aten.add, aten.abs, aten.sqrt, aten.sin, aten.cos, aten.neg, aten.exp, aten.mean, aten.sub, aten.std, aten.div]
            triton_poi_fused_abs_add_cos_div_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_std_sub_tanh_5_xnumel = s0*s1*s24
            stream0 = get_raw_stream(0)
            triton_poi_fused_abs_add_cos_div_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_std_sub_tanh_5.run(buf9, buf2, buf7, s0, s1, s24, triton_poi_fused_abs_add_cos_div_exp_gelu_mean_mul_neg_relu_sigmoid_sin_sqrt_std_sub_tanh_5_xnumel, stream=stream0)
            del buf2
            del buf7
        return (buf9, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 32
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

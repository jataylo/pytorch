# AOT ID: ['55_inference']
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


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/xr/cxr6gxlvdzuywxxlwgzjmeatou5pgifhkgzfevw7mwxhhlk25a6k.py
# Topologically Sorted Source Nodes: [sub, add, sqrt, normalized, mul, scaled, activated, with_residual, output], Original ATen: [aten.sub, aten.add, aten.sqrt, aten.div, aten.mul, aten.relu, aten.silu]
# Source node to ATen node mapping:
#   activated => relu
#   add => add_4
#   mul => mul_12
#   normalized => div
#   output => mul_25, sigmoid
#   scaled => add_21
#   sqrt => sqrt
#   sub => sub
#   with_residual => add_30
# Graph fragment:
#   %arg3_1 : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[s15, s12, s0][0, 0, 0]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg7_1 : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %add_30 : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0" = PlaceHolder[target=add_30]
#   %sub : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg3_1, %arg4_1), kwargs = {})
#   %add_4 : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg5_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %div : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %sqrt), kwargs = {})
#   %mul_12 : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %arg6_1), kwargs = {})
#   %add_21 : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %arg7_1), kwargs = {})
#   %relu : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_21,), kwargs = {})
#   %add_30 : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu, %arg8_1), kwargs = {})
#   %sigmoid : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_30,), kwargs = {})
#   %mul_25 : Tensor "f32[s15, s12, s0][s0*s12, s0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_30, %sigmoid), kwargs = {})
#   return %add_30,%mul_25
triton_poi_fused_add_div_mul_relu_silu_sqrt_sub_0 = async_compile.triton('triton_poi_fused_add_div_mul_relu_silu_sqrt_sub_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_relu_silu_sqrt_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.033554432, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_relu_silu_sqrt_sub_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (x0), xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp11 = tl.load(in_ptr4 + (x0), xmask)
    tmp15 = tl.load(in_ptr5 + (x0), xmask)
    tmp3 = tmp0 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sqrt_rn(tmp6)
    tmp8 = (tmp3 / tmp7)
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = tl.maximum(tmp13, tmp12, tl.PropagateNan.ALL)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tl.store(in_out_ptr0 + (x0), tmp18, xmask)


def get_args():
    arg_0 = rand_strided((4, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((4, 512, 512), (0, 0, 0), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((4, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((4, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((4, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((4, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 1048576,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_relu_silu_sqrt_sub_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_div_mul_relu_silu_sqrt_sub_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 0.033554432
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
        args.clear()
        s77 = arg0_1
        s27 = arg1_1
        s53 = arg2_1
        arg3_1_size = arg3_1.size()
        s15 = arg3_1_size[0]
        s12 = arg3_1_size[1]
        s0 = arg3_1_size[2]
        assert_size_stride(arg3_1, (s15, s12, s0), (s0*s12, s0, 1))
        assert_size_stride(arg4_1, (s15, s12, s0), (0, 0, 0))
        assert_size_stride(arg5_1, (s15, s12, s0), (s0*s12, s0, 1))
        assert_size_stride(arg6_1, (s15, s12, s0), (s0*s12, s0, 1))
        assert_size_stride(arg7_1, (s15, s12, s0), (s0*s12, s0, 1))
        assert_size_stride(arg8_1, (s15, s12, s0), (s0*s12, s0, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s15, s12, s0), (s0*s12, s0, 1), torch.float32)
            buf1 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [sub, add, sqrt, normalized, mul, scaled, activated, with_residual, output], Original ATen: [aten.sub, aten.add, aten.sqrt, aten.div, aten.mul, aten.relu, aten.silu]
            triton_poi_fused_add_div_mul_relu_silu_sqrt_sub_0_xnumel = s0*s12*s15
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_div_mul_relu_silu_sqrt_sub_0.run(buf1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, triton_poi_fused_add_div_mul_relu_silu_sqrt_sub_0_xnumel, stream=stream0)
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            del arg7_1
            del arg8_1
        return (buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 4
    arg1_1 = 512
    arg2_1 = 512
    arg3_1 = rand_strided((4, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((4, 512, 512), (0, 0, 0), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((4, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((4, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((4, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((4, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

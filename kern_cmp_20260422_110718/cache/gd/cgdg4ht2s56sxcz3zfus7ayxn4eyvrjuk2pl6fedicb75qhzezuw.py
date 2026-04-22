# AOT ID: ['38_inference']
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


# kernel path: /dockerx/meta-tuning/pytorch/kern_cmp_20260422_110718/cache/xx/cxx2x7eav7d7lxveqxiv7xve6mg57dtxs5xyseknrg5myjwknyfm.py
# Topologically Sorted Source Nodes: [x_perm, mul, add], Original ATen: [aten.permute, aten.mul, aten.add]
# Source node to ATen node mapping:
#   add => add
#   mul => mul
#   x_perm => permute
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 16, 16][256, 16, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[16, 16, 16][256, 16, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[16, 16, 16][256, 16, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %permute : Tensor "f32[16, 16, 16][1, 16, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg0_1, [2, 1, 0]), kwargs = {})
#   %mul : Tensor "f32[16, 16, 16][256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %add : Tensor "f32[16, 16, 16][1, 16, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %mul), kwargs = {})
#   return %add
triton_poi_fused_add_mul_permute_0 = async_compile.triton('triton_poi_fused_add_mul_permute_0', '''
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
    size_hints={'y': 16, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_permute_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'B975290662C72363321B97D54DE97D8BFA030AAA24318719843F115358856350', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 49152, 'x': 32768}, 'kernel_num_gb': 6.5536e-05, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_permute_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 16)
    x2 = xindex // 16
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16*x2 + 256*x1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + 256*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x3 + 256*y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (y0 + 16*x2 + 256*x1), tmp4, xmask & ymask)


def get_args():
    arg_0 = rand_strided((16, 16, 16), (256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 16, 16), (256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 16, 16), (256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((16, 16, 16), (1, 16, 256), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 16, 256,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_permute_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_mul_permute_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda:0', rep=40)
    num_gb = 6.5536e-05
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
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        assert_size_stride(arg0_1, (16, 16, 16), (256, 16, 1))
        assert_size_stride(arg1_1, (16, 16, 16), (256, 16, 1))
        assert_size_stride(arg2_1, (16, 16, 16), (256, 16, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((16, 16, 16), (1, 16, 256), torch.float32)
            # Topologically Sorted Source Nodes: [x_perm, mul, add], Original ATen: [aten.permute, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_permute_0.run(arg0_1, arg1_1, arg2_1, buf0, 16, 256, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 16, 16), (256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, 16, 16), (256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, 16, 16), (256, 16, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

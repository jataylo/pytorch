
import torch
from torch.utils.cpp_extension import load_inline

cpp_wrapper_src = (
'''

#include <c10/util/Exception.h>
#include <c10/cuda/CUDAGuard.h>

#define AT_CUDA_DRIVER_CHECK_OVERRIDE(EXPR)                         \
do {                                                                \
    CUresult __err = EXPR;                                          \
    if (__err != CUDA_SUCCESS) {                                    \
        AT_ERROR("CUDA driver error: ", static_cast<int>(__err));   \
    }                                                               \
} while (0)

static inline CUfunction loadKernel(const std::string &filePath,
        const std::string &funcName) {
    CUmodule mod;
    CUfunction func;
    AT_CUDA_DRIVER_CHECK_OVERRIDE(cuModuleLoad(&mod, filePath.c_str()));
    AT_CUDA_DRIVER_CHECK_OVERRIDE(cuModuleGetFunction(&func, mod, funcName.c_str()));
    return func;
}

static inline void launchKernel(
        CUfunction func,
        int gridX,
        int gridY,
        int gridZ,
        int numWraps,
        int sharedMemBytes,
        void* args[],
        int device_index) {
    AT_CUDA_DRIVER_CHECK_OVERRIDE(cuLaunchKernel(
        func, gridX, gridY, gridZ, 32*numWraps, 1, 1, sharedMemBytes,
        at::cuda::getCurrentCUDAStream(device_index), args, nullptr));
}

static CUfunction triton_poi_fused_sigmoid_0 = nullptr;

std::vector<at::Tensor> inductor_entry_cpp(const std::vector<at::Tensor>& args) {
    at::Tensor primals_1;
    primals_1 = args[0];
    at::Tensor primals_2;
    primals_2 = args[1];
    at::Tensor primals_3;
    primals_3 = args[2];

    at::cuda::CUDAGuard device_guard(0);
    auto buf0 = at::empty_strided({2L, 16L}, {16L, 1L}, at::device(at::kCUDA).dtype(at::kFloat));
    at::addmm_out(buf0, primals_2, primals_3, at::as_strided(primals_1, {8L, 16L}, {1L, 8L}, 0L), 1, 1);
    primals_1.reset();
    primals_2.reset();
    auto buf1 = buf0; buf0.reset();  // reuse
    if (triton_poi_fused_sigmoid_0 == nullptr) {
         triton_poi_fused_sigmoid_0 = loadKernel("/tmp/torchinductor_root/cubin/cl7fzuysuhhdflwdsuvkttairt27qersw46belrll6wmleg2ruzm.hsaco", "triton__0d1d");
    }
    CUdeviceptr var_0 = reinterpret_cast<CUdeviceptr>(buf1.data_ptr());
    int var_1 = 32;
    void* kernel_args_var_0[] = {&var_0, &var_1};
    launchKernel(triton_poi_fused_sigmoid_0, 1, 1, 1, 1, 0, kernel_args_var_0, 0);
    return {buf1, primals_3, buf1};
}
'''
)

module = load_inline(
    name='inline_extension_cmwu25kmywq67ekwk4whakytkkr6n374ddvovto4frbowjleeffi',
    cpp_sources=[cpp_wrapper_src],
    functions=['inductor_entry_cpp'],
    extra_cflags=['-std=c++17 -Wno-unused-variable -O3 -ffast-math -fno-finite-math-only -march=native -fopenmp -Wall -DCPU_CAPABILITY_AVX512 -D C10_USING_CUSTOM_GENERATED_MACROS'],
    extra_ldflags=['-shared -fPIC -L/var/lib/jenkins/pytorch/torch/lib -L/opt/rocm/lib -L/opt/rocm/hip/lib -L/opt/conda/envs/py_3.9/lib -lc10 -ltorch -ltorch_cpu -ltorch_python -lgomp -lc10_cuda -lcuda -ltorch_cuda'],
    extra_include_paths=['-I/var/lib/jenkins/pytorch/torch/include -I/var/lib/jenkins/pytorch/torch/include/torch/csrc/api/include -I/var/lib/jenkins/pytorch/torch/include/TH -I/var/lib/jenkins/pytorch/torch/include/THC -I/var/lib/jenkins/pytorch/torch/include/THH -I/opt/rocm/include -I/opt/conda/envs/py_3.9/include/python3.9'])

def _wrap_func(f):
    def g(args):
        args_tensor = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg) for arg in args]
        return f(args_tensor)
    return g
call = _wrap_func(module.inductor_entry_cpp)


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((2, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.utils import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

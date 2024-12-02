#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <torch/csrc/utils/pybind.h>
#if !defined(USE_ROCM)
#include <hip/hip_runtime_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif

#include <c10/hip/HIPException.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

namespace torch::cuda::shared {

#ifdef USE_ROCM
namespace {
hipError_t hipReturnSuccess() {
  return hipSuccess;
}
} // namespace
#endif

void initCudartBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cudart = m.def_submodule("_cudart", "libcudart.so bindings");

  // By splitting the names of these objects into two literals we prevent the
  // HIP rewrite rules from changing these names when building with HIP.

#if !defined(USE_ROCM) && defined(TORCH_HIP_VERSION) && TORCH_HIP_VERSION < 12000
  // cudaOutputMode_t is used in hipProfilerInitialize only. The latter is gone
  // in CUDA 12.
  py::enum_<cudaOutputMode_t>(
      cudart,
      "cuda"
      "OutputMode")
      .value("KeyValuePair", hipKeyValuePair)
      .value("CSV", hipCSV);
#endif

  py::enum_<hipError_t>(
      cudart,
      "cuda"
      "Error")
      .value("success", hipSuccess);

  cudart.def(
      "cuda"
      "GetErrorString",
      hipGetErrorString);
  cudart.def(
      "cuda"
      "ProfilerStart",
#ifdef USE_ROCM
      hipReturnSuccess
#else
      hipProfilerStart
#endif
  );
  cudart.def(
      "cuda"
      "ProfilerStop",
#ifdef USE_ROCM
      hipReturnSuccess
#else
      hipProfilerStop
#endif
  );
  cudart.def(
      "cuda"
      "HostRegister",
      [](uintptr_t ptr, size_t size, unsigned int flags) -> hipError_t {
        py::gil_scoped_release no_gil;
        return C10_HIP_ERROR_HANDLED(
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            hipHostRegister((void*)ptr, size, flags));
      });
  cudart.def(
      "cuda"
      "HostUnregister",
      [](uintptr_t ptr) -> hipError_t {
        py::gil_scoped_release no_gil;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        return C10_HIP_ERROR_HANDLED(hipHostUnregister((void*)ptr));
      });
  cudart.def(
      "cuda"
      "StreamCreate",
      [](uintptr_t ptr) -> hipError_t {
        py::gil_scoped_release no_gil;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        return C10_HIP_ERROR_HANDLED(hipStreamCreate((hipStream_t*)ptr));
      });
  cudart.def(
      "cuda"
      "StreamDestroy",
      [](uintptr_t ptr) -> hipError_t {
        py::gil_scoped_release no_gil;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        return C10_HIP_ERROR_HANDLED(hipStreamDestroy((hipStream_t)ptr));
      });
#if !defined(USE_ROCM) && defined(TORCH_HIP_VERSION) && TORCH_HIP_VERSION < 12000
  // hipProfilerInitialize is no longer needed after CUDA 12:
  // https://forums.developer.nvidia.com/t/cudaprofilerinitialize-is-deprecated-alternative/200776/3
  cudart.def(
      "cuda"
      "ProfilerInitialize",
      hipProfilerInitialize,
      py::call_guard<py::gil_scoped_release>());
#endif
  cudart.def(
      "cuda"
      "MemGetInfo",
      [](c10::DeviceIndex device) -> std::pair<size_t, size_t> {
        c10::hip::HIPGuardMasqueradingAsCUDA guard(device);
        size_t device_free = 0;
        size_t device_total = 0;
        py::gil_scoped_release no_gil;
        C10_HIP_CHECK(hipMemGetInfo(&device_free, &device_total));
        return {device_free, device_total};
      });
}

} // namespace torch::cuda::shared

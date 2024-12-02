#pragma once

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.

#ifdef USE_ROCM

// FIXME: Currently, CPU and CUDA backend are mutually exclusive.
// This is a temporary workaround. We need a better way to support
// multi devices.

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#define AOTI_RUNTIME_DEVICE_CHECK(EXPR)                    \
  do {                                                     \
    const hipError_t code = EXPR;                         \
    const char* msg = hipGetErrorString(code);            \
    if (code != hipSuccess) {                             \
      throw std::runtime_error(                            \
          std::string("CUDA error: ") + std::string(msg)); \
    }                                                      \
  } while (0)

namespace torch::aot_inductor {

using DeviceStreamType = hipStream_t;

} // namespace torch::aot_inductor

#else // !USE_ROCM

#define AOTI_RUNTIME_DEVICE_CHECK(EXPR)            \
  bool ok = EXPR;                                  \
  if (!ok) {                                       \
    throw std::runtime_error("CPU runtime error"); \
  }

namespace torch::aot_inductor {

using DeviceStreamType = void*;

} // namespace torch::aot_inductor

#endif // USE_ROCM

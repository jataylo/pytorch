#ifdef _WIN32
#include <wchar.h> // _wgetenv for nvtx
#endif
#ifdef TORCH_CUDA_USE_NVTX3
#include <roctracer/roctx.h>
#else
#include <roctracer/roctx.h>
#endif
#include <torch/csrc/utils/pybind.h>

namespace torch::cuda::shared {

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

#ifdef TORCH_CUDA_USE_NVTX3
  auto nvtx = m.def_submodule("_nvtx", "nvtx3 bindings");
#else
  auto nvtx = m.def_submodule("_nvtx", "libNvToolsExt.so bindings");
#endif
  nvtx.def("rangePushA", roctxRangePushA);
  nvtx.def("rangePop", roctxRangePop);
  nvtx.def("rangeStartA", roctxRangeStartA);
  nvtx.def("rangeEnd", roctxRangeStop);
  nvtx.def("markA", roctxMarkA);
}

} // namespace torch::cuda::shared

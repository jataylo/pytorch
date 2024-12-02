#include <torch/csrc/python_headers.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THCPMemPool_init(PyObject* module) {
  auto torch_C_m = py::handle(module).cast<py::module>();
  shared_ptr_class_<::c10::hip::MemPool>(torch_C_m, "_MemPool")
      .def(
          py::init([](c10::hip::HIPCachingAllocator::HIPAllocator* allocator,
                      bool is_user_created) {
            torch::utils::device_lazy_init(at::kCUDA);
            return std::make_shared<::c10::hip::MemPool>(
                allocator, is_user_created);
          }))
      .def_property_readonly("id", &::c10::hip::MemPool::id)
      .def_property_readonly("allocator", &::c10::hip::MemPool::allocator)
      .def("use_count", &::c10::hip::MemPool::use_count);
  shared_ptr_class_<::c10::hip::MemPoolContext>(torch_C_m, "_MemPoolContext")
      .def(py::init<c10::hip::MemPool*>())
      .def_static(
          "active_pool", &::c10::hip::MemPoolContext::getActiveMemPool);
}

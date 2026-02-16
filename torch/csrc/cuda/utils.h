#pragma once

#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/csrc/utils/python_numbers.h>

#include <vector>

std::vector<std::optional<at::hip::HIPStreamMasqueradingAsCUDA>>
THPUtils_PySequence_to_CUDAStreamList(PyObject* obj);

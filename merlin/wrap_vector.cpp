// Copyright 2024 quocdang1998
#include "merlin/vector.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "py_common.hpp"

namespace merlin {

// Convert Python list to floatvec
floatvec pylist_to_fvec(py::list & float_list) {
    floatvec result(float_list.size());
    std::uint64_t idx = 0;
    for (auto it = float_list.begin(); it != float_list.end(); ++it) {
        result[idx] = (*it).cast<double>();
        idx += 1;
    }
    return result;
}

// Convert Python list to intvec
intvec pylist_to_ivec(py::list & float_list) {
    intvec result(float_list.size());
    std::uint64_t idx = 0;
    for (auto it = float_list.begin(); it != float_list.end(); ++it) {
        result[idx] = (*it).cast<std::uint64_t>();
        idx += 1;
    }
    return result;
}

}  // namespace merlin

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

// Convert floatvec to Python list
py::list fvec_to_pylist(const floatvec & vector) {
    py::list result;
    for (const double & value : vector) {
        result.append(value);
    }
    return result;
}

// Convert Python list to intvec
intvec pylist_to_ivec(py::list & int_list) {
    intvec result(int_list.size());
    std::uint64_t idx = 0;
    for (auto it = int_list.begin(); it != int_list.end(); ++it) {
        result[idx] = (*it).cast<std::uint64_t>();
        idx += 1;
    }
    return result;
}

// Convert intvec to Python list
py::list ivec_to_pylist(const intvec & vector) {
    py::list result;
    for (const std::uint64_t & value : vector) {
        result.append(value);
    }
    return result;
}

}  // namespace merlin

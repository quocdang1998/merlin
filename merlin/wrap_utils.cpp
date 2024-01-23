// Copyright 2024 quocdang1998
#include "merlin/utils.hpp"

#include <vector>

#include "merlin/vector.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "py_common.hpp"

namespace merlin {

void wrap_utils(py::module & merlin_package) {
    // add utils submodule
    py::module utils_module = merlin_package.def_submodule("utils", "Utils.");
    // contiguous to ndim idx
    utils_module.def(
        "contiguous_to_ndim_idx",
        [](std::uint64_t index, py::list & shape) {
            intvec shape_cpp(pylist_to_ivec(shape));
            py::list ndim_idx = ivec_to_pylist(contiguous_to_ndim_idx(index, shape_cpp));
            return ndim_idx;
        },
        "Convert C-contiguous index to n-dimensional index with allocating memory for result.",
        py::arg("index"), py::arg("shape")
    );
    // get random subset
    utils_module.def(
        "get_random_subset",
        [](std::uint64_t num_points, std::uint64_t i_max, std::uint64_t i_min) {
            intvec random_subset = get_random_subset(num_points, i_max, i_min);
            std::vector subset_cpp(random_subset.begin(), random_subset.end());
            py::list subset_py = py::cast(subset_cpp);
            return subset_py;
        },
        "Get a random subset of index in a range.",
        py::arg("num_points"), py::arg("i_max"), py::arg("i_min") = 0
    );
}

}  // namespace merlin

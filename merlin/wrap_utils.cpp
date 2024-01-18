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
    // add regpl submodule
    py::module utils_module = merlin_package.def_submodule("utils", "Utils.");
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

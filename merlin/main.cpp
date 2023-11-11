// Copyright 2023 quocdang1998
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace merlin {

/** @brief Wrap ``merlin::Environment`` class.*/
void wrap_env(py::module & merlin_package);

/** @brief Wrap ``merlin::cuda`` library.*/
void wrap_cuda(py::module & merlin_package);

}  // namespace merlin

// Wrap main module
PYBIND11_MODULE(merlin, merlin_package) {
    merlin_package.doc() = "Python interface of Merlin library.";
    // wrap merlin::Environment
    merlin::wrap_env(merlin_package);
    // add cuda submodule
    merlin::wrap_cuda(merlin_package);
}

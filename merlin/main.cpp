// Copyright 2023 quocdang1998
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace merlin {

/** @brief Wrap ``merlin::ProcessorType`` class.*/
void wrap_proctype(py::module & merlin_package);

/** @brief Wrap ``merlin::Environment`` class.*/
void wrap_env(py::module & merlin_package);

/** @brief Wrap ``merlin::cuda`` library.*/
void wrap_cuda(py::module & merlin_package);

/** @brief Wrap ``merlin::array`` library.*/
void wrap_array(py::module & merlin_package);

/** @brief Wrap ``merlin::stat`` library.*/
void wrap_stat(py::module & merlin_package);

/** @brief Wrap ``merlin::grid`` library.*/
void wrap_grid(py::module & merlin_package);

/** @brief Wrap ``merlin::candy`` library.*/
void wrap_candy(py::module & merlin_package);

/** @brief Wrap ``merlin::splint`` library.*/
void wrap_splint(py::module & merlin_package);

}  // namespace merlin

// Wrap main module
PYBIND11_MODULE(merlin, merlin_package) {
    merlin_package.doc() = "Python interface of Merlin library.";
    // wrap merlin::ProcessorType
    merlin::wrap_proctype(merlin_package);
    // wrap merlin::Environment
    merlin::wrap_env(merlin_package);
    // add cuda submodule
    merlin::wrap_cuda(merlin_package);
    // add array submodule
    merlin::wrap_array(merlin_package);
    // add stat submodule
    merlin::wrap_stat(merlin_package);
    // add grid submodule
    merlin::wrap_grid(merlin_package);
    // add candy submodule
    merlin::wrap_candy(merlin_package);
    // add splint submodule
    merlin::wrap_splint(merlin_package);
}

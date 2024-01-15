// Copyright 2023 quocdang1998
#include "py_common.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

// Wrap main module
PYBIND11_MODULE(merlin, merlin_package) {
    merlin_package.doc() = "Python interface of Merlin library.";
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
    // add regpl submodule
    merlin::wrap_regpl(merlin_package);
    // add utils submodule
    merlin::wrap_utils(merlin_package);
}

// Copyright 2023 quocdang1998
#include "py_api.hpp"

// Wrap main module
PYBIND11_MODULE(merlin, merlin_package) {
    // metadata
    merlin_package.doc() = "Python interface of Merlin library.";
    merlin_package.attr("__version__") = merlin::version;
    // wrap library
    merlin::wrap_mics(merlin_package);
    merlin::wrap_cuda(merlin_package);
    merlin::wrap_array(merlin_package);
    merlin::wrap_grid(merlin_package);
    merlin::wrap_splint(merlin_package);
    merlin::wrap_regpl(merlin_package);
    merlin::wrap_candy(merlin_package);
}

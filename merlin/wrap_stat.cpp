// Copyright 2023 quocdang1998
#include "merlin/stat/moment.hpp"

#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace merlin {

// Wrap mean functions
void wrap_mean(py::module & stat_module) {
    stat_module.def(
        "mean",
        [] (const array::Array & data, std::uint64_t n_threads) { return stat::mean(data, n_threads); },
        "Calculate mean on all elements of the array.",
        py::arg("data"), py::arg("n_threads") = 1
    );
    stat_module.def(
        "mean",
        [] (const array::Parcel & data, std::uint64_t n_threads, const cuda::Stream & stream) {
            return stat::mean(data, n_threads, stream);
        },
        "Calculate mean on all elements of the array.",
        py::arg("data"), py::arg("n_threads") = 1, py::arg("stream") = cuda::Stream()
    );
}

// Wrap mean variance functions
void wrap_meanvar(py::module & stat_module) {
    stat_module.def(
        "mean_variance",
        [] (const array::Array & data, std::uint64_t n_threads) {
            auto [mean, variance] = stat::mean_variance(data, n_threads);
            return py::make_tuple(mean, variance);
        },
        "Calculate mean and variance on all elements of the array.",
        py::arg("data"), py::arg("n_threads") = 1
    );
    stat_module.def(
        "mean_variance",
        [] (const array::Parcel & data, std::uint64_t n_threads, const cuda::Stream & stream) {
            auto [mean, variance] = stat::mean_variance(data, n_threads, stream);
            return py::make_tuple(mean, variance);
        },
        "Calculate mean and variance on all elements of the array.",
        py::arg("data"), py::arg("n_threads") = 1, py::arg("stream") = cuda::Stream()
    );
}

void wrap_stat(py::module & merlin_package) {
    // add array submodule
    py::module stat_module = merlin_package.def_submodule("stat", "Calculate mean and variance.");
    // add functions
    wrap_mean(stat_module);
    wrap_meanvar(stat_module);
}

}  // namespace merlin

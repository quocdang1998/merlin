// Copyright 2023 quocdang1998
#include "py_api.hpp"

#include <cstdint>  // std::uint64_t
#include <random>   // std::seed_seq

#include "merlin/env.hpp"    // merlin::Environment
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx, merlin::get_random_subset

namespace merlin {

// Wrap merlin::Environment class
static void wrap_env(py::module & merlin_package) {
    auto env_pyclass = py::class_<Environment>(
        merlin_package,
        "Environment",
        R"(
        Execution environment of the merlin library.

        Wrapper of :cpp:class:`merlin::Environment`.
        )"
    );
    // constructors
    env_pyclass.def(
        py::init([]() { return new Environment(); }),
        "Default constructor."
    );
    // number of instances
    env_pyclass.def_readonly_static(
        "is_initialized",
        &Environment::is_initialized,
        "Check if the environment is initialized or not."
    );
    env_pyclass.def_static(
        "num_instances",
        []() { return Environment::num_instances.load(); },
        "Number of Environment instances created."
    );
    // seed setting operator
    env_pyclass.def_static(
        "set_random_seed",
        [](std::uint64_t new_seed) { Environment::random_generator.seed(new_seed); },
        "Set new random seed to the random generator.",
        py::arg("new_seed")
    );
    env_pyclass.def_static(
        "set_random_seed",
        [](py::sequence & seed_seq) {
            UIntVec cpp_seed_seq(pyseq_to_vector<std::uint64_t>(seed_seq));
            std::seed_seq s(cpp_seed_seq.begin(), cpp_seed_seq.end());
            Environment::random_generator.seed(s);
        },
        "Set a sequence of new random seed to the random generator.",
        py::arg("seed_seq")
    );
}

// Wrap utils functions
static void wrap_utils(py::module & merlin_package) {
    // contiguous to ndim idx
    merlin_package.def(
        "contiguous_to_ndim_idx",
        [](std::uint64_t index, py::sequence & shape) {
            UIntVec cpp_shape(pyseq_to_vector<std::uint64_t>(shape));
            UIntVec nd_index(cpp_shape.size());
            contiguous_to_ndim_idx(index, cpp_shape.data(), cpp_shape.size(), nd_index.data());
            return vector_to_pylist(nd_index);
        },
        R"(
        Convert C-contiguous index to n-dimensional index.

        Parameters
        ----------
        index : int
            C-contiguous index.
        shape : Sequence[int]
            Shape of n-dimensional array.)",
        py::arg("index"), py::arg("shape")
    );
    // get random subset
    merlin_package.def(
        "get_random_subset",
        [](std::uint64_t num_points, std::uint64_t i_max, std::uint64_t i_min) {
            UIntVec random_subset = get_random_subset(num_points, i_max, i_min);
            return vector_to_pylist(random_subset);
        },
        R"(
        Get a random subset of index in a range.

        Parameters
        ----------
        num_points : int
            Number of random integer to generate.
        i_max : int
            Max value of the range.
        i_min : int, default = 0
            Min value of the range.
        )",
        py::arg("num_points"), py::arg("i_max"), py::arg("i_min") = 0
    );
}

void wrap_mics(py::module & merlin_package) {
    wrap_env(merlin_package);
    wrap_utils(merlin_package);
}

}  // namespace merlin

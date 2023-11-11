// Copyright 2023 quocdang1998
#include "merlin/env.hpp"

#include <cstdint>  // std::uint64_t
#include <random>   // std::seed_seq
#include <vector>   // std::vector

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace merlin {

// Wrap merlin::Environment class
void wrap_env(py::module & merlin_package) {
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
        py::init([](){ return new Environment(); }),
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
        [](py::sequence & sed_seq) {
            std::vector<std::uint64_t> vect_seed_seq;
            vect_seed_seq.reserve(sed_seq.size());
            for (auto it = sed_seq.begin(); it != sed_seq.end(); ++it) {
                vect_seed_seq.push_back((*it).cast<std::uint64_t>());
            }
            std::seed_seq cpp_seed_seq(vect_seed_seq.begin(), vect_seed_seq.end());
            Environment::random_generator.seed(cpp_seed_seq);
        },
        "Set a sequence of new random seed to the random generator.",
        py::arg("sed_seq")
    );
}

}  // namespace merlin

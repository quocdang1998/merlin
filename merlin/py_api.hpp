// Copyright 2024 quocdang1998
#ifndef MERLIN_PY_API_HPP_
#define MERLIN_PY_API_HPP_

#include <map>     // std::map
#include <string>  // std::string

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "merlin/config.hpp"
#include "merlin/logger.hpp"
#include "merlin/synchronizer.hpp"
#include "merlin/vector.hpp"

namespace merlin {

// Conversion Python sequence <-> merlin vector
// --------------------------------------------

// Convert from Python to C++ vector
template <typename T, class PySequence>
Vector<T> pyseq_to_vector(const PySequence & py_seq) {
    Vector<T> cpp_vec(py_seq.size());
    std::uint64_t idx = 0;
    for (auto it = py_seq.begin(); it != py_seq.end(); ++it) {
        py::handle element = *it;
        cpp_vec[idx++] = element.cast<T>();
    }
    return cpp_vec;
}

// Convert from Python to C++ array
template <typename T, class PySequence>
std::array<T, max_dim> pyseq_to_array(const PySequence & py_seq) {
    std::array<T, max_dim> cpp_arr;
    cpp_arr.fill(T());
    if (py_seq.size() > max_dim) {
        Fatal<std::invalid_argument>("Exceeding maximum ndim (%" PRIu64 ").\n", max_dim);
    }
    std::uint64_t idx = 0;
    for (auto it = py_seq.begin(); it != py_seq.end(); ++it) {
        py::handle element = *it;
        cpp_arr[idx++] = element.cast<T>();
    }
    return cpp_arr;
}

// Convert from C++ vector to Python
template <typename T>
py::list vector_to_pylist(const Vector<T> & vector) {
    py::list py_list;
    for (const T & value : vector) {
        py_list.append(value);
    }
    return py_list;
}

// Convert from C++ array to Python
template <typename T>
py::list array_to_pylist(const std::array<T, max_dim> & array, std::uint64_t size) {
    Vector<T> assigned_vector;
    assigned_vector.assign(const_cast<T*>(array.data()), size);
    return vector_to_pylist(assigned_vector);
}

// Wrap common enum
// ----------------

// wrap ProcessorType
static const std::map<std::string, ProcessorType> proctype_map = {
    {"cpu", ProcessorType::Cpu},
    {"gpu", ProcessorType::Gpu}
};

// Wrap Libraries
// --------------

// Wrap miscellaneous objects
void wrap_mics(py::module & merlin_package);

// @brief Wrap ``merlin::cuda`` library
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

/** @brief Wrap ``merlin::regpl`` library.*/
void wrap_regpl(py::module & merlin_package);

}  // namespace merlin

#endif  // MERLIN_PY_API_HPP_

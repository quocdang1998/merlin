// Copyright 2024 quocdang1998
#ifndef MERLIN_PY_API_HPP_
#define MERLIN_PY_API_HPP_

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "merlin/vector.hpp"

namespace merlin {

// Conversion Python sequence <-> merlin vector
// --------------------------------------------

// Convert from Python to C++
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

// Convert from C++ to Python
template <typename T>
py::list vector_to_pylist(const Vector<T> & vector) {
    py::list py_list;
    for (const T & value : vector) {
        py_list.append(value);
    }
    return py_list;
}

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

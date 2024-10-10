// Copyright 2024 quocdang1998
#ifndef MERLIN_PY_API_HPP_
#define MERLIN_PY_API_HPP_

#include <cinttypes>  // PRIu64
#include <map>        // std::map
#include <string>     // std::string

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "merlin/config.hpp"
#include "merlin/logger.hpp"
#include "merlin/synchronizer.hpp"
#include "merlin/vector/dynamic_vector.hpp"
#include "merlin/vector/static_vector.hpp"
#include "merlin/vector/view.hpp"

namespace merlin {

// Conversion Python sequence <-> merlin vector
// --------------------------------------------

// Convert from Python to C++ vector
template <typename T, class PySequence>
vector::DynamicVector<T> pyseq_to_vector(const PySequence & py_seq) {
    vector::DynamicVector<T> cpp_vec(py_seq.size());
    std::uint64_t idx = 0;
    for (auto it = py_seq.begin(); it != py_seq.end(); ++it) {
        py::handle element = *it;
        cpp_vec[idx++] = element.cast<T>();
    }
    return cpp_vec;
}

// Convert from Python to C++ array
template <typename T, std::uint64_t Capacity = max_dim, class PySequence>
vector::StaticVector<T, Capacity> pyseq_to_array(const PySequence & py_seq) {
    vector::StaticVector<T, Capacity> cpp_arr(py_seq.size());
    if (py_seq.size() > Capacity) {
        Fatal<std::invalid_argument>("Exceeding maximum ndim (%" PRIu64 ").\n", max_dim);
    }
    std::uint64_t idx = 0;
    for (auto it = py_seq.begin(); it != py_seq.end(); ++it) {
        py::handle element = *it;
        cpp_arr[idx++] = element.cast<T>();
    }
    return cpp_arr;
}

// Convert from C++ view to Python
template <typename T>
py::list view_to_pylist(vector::View<T> view) {
    py::list py_list;
    for (const T & value : view) {
        py_list.append(value);
    }
    return py_list;
}

// Convert from C++ vector to Python
template <typename T>
py::list vector_to_pylist(const vector::DynamicVector<T> & vector) {
    return view_to_pylist<T>(vector.get_view());
}

// Convert from C++ array to Python
template <typename T, std::uint64_t Capacity>
py::list array_to_pylist(const vector::StaticVector<T, Capacity> & array) {
    return view_to_pylist(array.get_view());
}

// Wrap l-value reference by a NumPy array
// ---------------------------------------

// Create capsule
template <typename T>
py::capsule make_capsule(T * p_ref, std::uint64_t size) {
    if (size == 1) {
        return py::capsule(p_ref, [](void * data) { delete reinterpret_cast<T *>(data); });
    }
    return py::capsule(p_ref, [](void * data) { delete[] reinterpret_cast<T *>(data); });
}

// Create NumPy array wrapping the reference
template <typename T>
py::array_t<T> make_wrapper_array(T * p_ref, std::uint64_t size) {
    return py::array_t<T>({size}, {sizeof(T)}, p_ref, make_capsule<T>(p_ref, size));
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

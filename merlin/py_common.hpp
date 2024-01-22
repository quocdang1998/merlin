// Copyright 2024 quocdang1998
#ifndef MERLIN_PY_COMMON_HPP_
#define MERLIN_PY_COMMON_HPP_

#include <string>
#include <unordered_map>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "merlin/synchronizer.hpp"
#include "merlin/vector.hpp"

namespace merlin {

// Conversion
// ----------

// Convert Python list to floatvec
floatvec pylist_to_fvec(py::list & float_list);

// Convert Python list to intvec
intvec pylist_to_ivec(py::list & float_list);

// Wrap Libraries
// --------------

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

/** @brief Wrap ``merlin::regpl`` library.*/
void wrap_regpl(py::module & merlin_package);

/** @brief Wrap utils library.*/
void wrap_utils(py::module & merlin_package);

// Wrap Enums
// ----------

// wrap ProcessorType
static std::unordered_map<std::string, ProcessorType> proctype_map = {
    {"cpu", ProcessorType::Cpu},
    {"gpu", ProcessorType::Gpu}
};

}  // namespace merlin

#endif  // MERLIN_PY_COMMON_HPP_

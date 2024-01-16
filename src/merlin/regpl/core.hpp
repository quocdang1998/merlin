// Copyright 2024 quocdang1998
#ifndef MERLIN_REGPL_CORE_HPP_
#define MERLIN_REGPL_CORE_HPP_

#include "merlin/array/declaration.hpp"  // merlin::array::NdData
#include "merlin/cuda_interface.hpp"  // __cuhostdev__
#include "merlin/grid/declaration.hpp"  // merlin::grid::CartesianGrid
#include "merlin/linalg/declaration.hpp"  // merlin::linalg::Matrix
#include "merlin/regpl/declaration.hpp"  // merlin::regpl::Polynomial

namespace merlin::regpl {

// Calculate the vector to solve for regression system
// ---------------------------------------------------

/** @brief Calculate the vector to solve for regression system with Cartesian grid.*/
__cuhostdev__ void cacl_system(const grid::CartesianGrid & grid, const array::NdData & data,
                               regpl::Polynomial & polynom, std::uint64_t thread_idx, std::uint64_t n_threads) noexcept;

// Calculate the system to solve for regression system
// ---------------------------------------------------

/** @brief Calculate the matrix to solve for regression system with Cartesian grid.*/
__cuhostdev__ void cacl_system(const grid::CartesianGrid & grid, const regpl::Polynomial & polynom,
                               linalg::Matrix & matrix, std::uint64_t thread_idx, std::uint64_t n_threads) noexcept;

}  // namespace merlin::regpl

#endif  // MERLIN_REGPL_DECLARATION_HPP_

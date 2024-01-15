// Copyright 2024 quocdang1998
#ifndef MERLIN_REGPL_CORE_HPP_
#define MERLIN_REGPL_CORE_HPP_

#include "merlin/array/declaration.hpp"  // merlin::array::NdData
#include "merlin/cuda_interface.hpp"  // __cuhostdev__
#include "merlin/grid/declaration.hpp"  // merlin::grid::CartesianGrid, grid::RegularGrid
#include "merlin/linalg/declaration.hpp"  // merlin::linalg::Matrix
#include "merlin/regpl/declaration.hpp"  // merlin::regpl::Polynomial
#include "merlin/vector.hpp"             // merlin::floatvec

namespace merlin::regpl {

// Calculate the vector to solve for regression system
// ---------------------------------------------------

/** @brief Calculate the vector to solve for regression system with Cartesian grid.
 *  @param grid Cartesian grid of points.
 *  @param data Data to train the regression model (expected to have the same shape as the grid).
 *  @param polynom Polynomial coefficients.
 *  @param buffer Buffer memory for calculation, should be at least ``std::uint64_t[n_threads*ndim]``.
 *  @param thread_idx Index of the current thread.
 *  @param n_threads Number of threads to calculate.
 */
__cuhostdev__ void calc_vector(const grid::CartesianGrid & grid, const array::NdData & data,
                               regpl::Polynomial & polynom, std::uint64_t * buffer, std::uint64_t thread_idx,
                               std::uint64_t n_threads) noexcept;

/** @brief Calculate the vector to solve for regression system with regular grid.
 *  @param grid Regular grid of points.
 *  @param data Data to train the regression model (expected to have the same number of points as the grid).
 *  @param polynom Polynomial coefficients.
 *  @param buffer Buffer memory for calculation, should be at least ``std::uint64_t[n_threads*ndim]``.
 *  @param thread_idx Index of the current thread.
 *  @param n_threads Number of threads to calculate.
 */
__cuhostdev__ void calc_vector(const grid::RegularGrid & grid, const floatvec & data,
                               regpl::Polynomial & polynom, std::uint64_t * buffer, std::uint64_t thread_idx,
                               std::uint64_t n_threads) noexcept;

// Calculate the system to solve for regression system
// ---------------------------------------------------

/** @brief Calculate the matrix to solve for regression system with Cartesian grid.
 *  @param grid Cartesian grid of points.
 *  @param polynom Polynomial coefficients.
 *  @param matrix Matrix storing the result.
 *  @param buffer Buffer memory for calculation, should be at least ``std::uint64_t[2*n_threads*ndim]``.
 *  @param thread_idx Index of the current thread.
 *  @param n_threads Number of threads to calculate.
 */
__cuhostdev__ void calc_system(const grid::CartesianGrid & grid, const regpl::Polynomial & polynom,
                               linalg::Matrix & matrix, std::uint64_t * buffer, std::uint64_t thread_idx,
                               std::uint64_t n_threads) noexcept;

/** @brief Calculate the matrix to solve for regression system with regular grid.
 *  @param grid Grid of points.
 *  @param polynom Polynomial coefficients.
 *  @param matrix Matrix storing the result.
 *  @param buffer Buffer memory for calculation, should be at least ``std::uint64_t[2*n_threads*ndim]``.
 *  @param thread_idx Index of the current thread.
 *  @param n_threads Number of threads to calculate.
 */
__cuhostdev__ void calc_system(const grid::RegularGrid & grid, const regpl::Polynomial & polynom,
                               linalg::Matrix & matrix, std::uint64_t * buffer, std::uint64_t thread_idx,
                               std::uint64_t n_threads) noexcept;

}  // namespace merlin::regpl

#endif  // MERLIN_REGPL_DECLARATION_HPP_

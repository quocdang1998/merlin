// Copyright 2023 quocdang1998
#ifndef MERLIN_LINALG_TRI_SOLVE_HPP_
#define MERLIN_LINALG_TRI_SOLVE_HPP_

#include "merlin/exports.hpp"             // MERLIN_EXPORTS
#include "merlin/old_linalg/declaration.hpp"  // merlin::Matrix

namespace merlin::linalg {

// Upper triangular
// ----------------

/** @brief Solve an upper triangular matrix with ones on diagonal elemets.
 *  @details Solve upper triangular matrix having diagonal elements equal to 1. Diagonal and sub-diagonal elements are
 *  ignored, only upper-diagonal elements are read.
 *  @param triu_matrix Upper triangular matrix.
 *  @param solution Pointer to the first element of the vector of solution.
 */
MERLIN_EXPORTS void triu_one_solve(const linalg::Matrix & triu_matrix, double * solution) noexcept;

/** @brief Solve an upper triangular matrix.
 *  @details Solve upper triangular matrix. Sub-diagonal elements are ignored, only diagonal and upper-diagonal elements
 *  are read.
 *  @param triu_matrix Upper triangular matrix.
 *  @param solution Pointer to the first element of the vector of solution.
 */
MERLIN_EXPORTS void triu_solve(const linalg::Matrix & triu_matrix, double * solution) noexcept;

}  // namespace merlin::linalg

#endif  // MERLIN_LINALG_TRI_SOLVE_HPP_

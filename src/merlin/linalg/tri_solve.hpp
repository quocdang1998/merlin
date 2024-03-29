// Copyright 2023 quocdang1998
#ifndef MERLIN_LINALG_TRI_SOLVE_HPP_
#define MERLIN_LINALG_TRI_SOLVE_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin::linalg {

// Upper triangular
// ----------------

/** @brief Solve an upper triangular matrix with ones on diagonal elemets.
 *  @details Solve upper triangular matrix having diagonal elements equal to 1. Diagonal and sub-diagonal elements are
 *  ignored, only upper-diagonal elements are read.
 *  @param triu_matrix Pointer to the first element of column major triangular matrix.
 *  @param lda Leading dimension of the matrix.
 *  @param target Pointer to the first element of the vector to inverse.
 *  @param solution Pointer to the first element of the vector of solution, can be the same as the target vector.
 *  @param size Number of column/row of the linear system.
 */
MERLIN_EXPORTS void triu_one_solve(const double * triu_matrix, std::uint64_t lda, const double * target,
                                   double * solution, std::uint64_t size) noexcept;

/** @brief Solve a 4x4 block upper triangular matrix.*/
void __block_triu_no_avx(double * block_start, std::uint64_t lead_dim, double * output);


}  // namespace merlin::linalg

#endif  // MERLIN_LINALG_TRI_SOLVE_HPP_

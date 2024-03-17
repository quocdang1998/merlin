// Copyright 2023 quocdang1998
#ifndef MERLIN_LINALG_TRI_SOLVE_HPP_
#define MERLIN_LINALG_TRI_SOLVE_HPP_

#include <cstdint>  // std::uint64_t

namespace merlin::linalg {

// Upper Triangular
// ----------------

/** @brief Solve a 4x4 block upper triangular matrix.*/
void __block_triu_no_avx(double * block_start, std::uint64_t lead_dim, double * output);


}  // namespace merlin::linalg

#endif  // MERLIN_LINALG_TRI_SOLVE_HPP_

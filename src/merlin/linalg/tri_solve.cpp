// Copyright 2023 quocdang1998
#include "merlin/linalg/tri_solve.hpp"

#include <array>    // std::array
#include <cmath>    // std::fma

#ifdef __AVX__
#include <immintrin.h>
#endif  // __AVX__

#include "merlin/logger.hpp"  // WARNING

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Upper Triangular
// ---------------------------------------------------------------------------------------------------------------------

// Solve a 4x4 block upper triangular matrix
void linalg::__block_triu_no_avx(double * block_start, std::uint64_t lead_dim, double * output) {
    // copy output
    std::array<double, 4> solution;
    for (std::uint64_t i = 0; i < 4; i++) {
        solution[i] = output[i];
    }
    // divide by diagonal
    std::array<double, 4> diag = {
        block_start[0],
        block_start[lead_dim + 1],
        block_start[2 * lead_dim + 2],
        block_start[3 * lead_dim + 3]
    };
    for (std::uint64_t i = 0; i < 4; i++) {
        solution[i] /= diag[i];
    }
    // solve for each element
    std::array<double, 4> column, broadcast;
    for (std::int64_t i_col = 3; i_col > 0; i_col--) {
        // broadcast i_col-th element from the solution into 4-sized vector
        for (std::uint64_t j = 0; j < 4; j++) {
            broadcast[j] = solution[i_col];
        }
        // copy i_col-th column from the matrix
        for (std::uint64_t i_row = 0; i_row < i_col; i_row++) {
            column[i_row] = block_start[i_col * lead_dim + i_row];
        }
        // zero diagonal element on the column
        column[i_col] = 0;
        // divide column by diag
        for (std::uint64_t j = 0; j < 4; j++) {
            column[j] /= diag[j];
        }
        // multiply column by broadcast and subtract the product from the solution
        for (std::uint64_t j = 0; j < 4; j++) {
            solution[j] = solution[j] - column[j] * broadcast[j];
        }
    }
    // copy solution back to output
    for (std::uint64_t i = 0; i < 4; i++) {
        output[i] = solution[i];
    }
}

}  // namespace merlin

// Copyright 2023 quocdang1998
#include "merlin/linalg/tri_solve.hpp"

#include <array>    // std::array
#include <cmath>    // std::fma

#include "merlin/avx.hpp"     // merlin::PackedDouble, merlin::use_avx
#include "merlin/linalg/matrix.hpp"  // merlin::linalg::Matrix
#include "merlin/logger.hpp"  // WARNING

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// KernelMatrix
// ---------------------------------------------------------------------------------------------------------------------

// Load a 4-size block of matrix into AVX memory
static inline void load_triangular(const linalg::Matrix & total, std::uint64_t start_row, std::uint64_t start_col,
                                   std::uint64_t size, linalg::KernelMatrix & kernel) {
    const double * data = &(total.cget(start_row, start_col));
    for (std::uint64_t i_col = 0; i_col < size; i_col++) {
        kernel.core[i_col] = PackedDouble<use_avx>(data, i_col);
        data += total.lead_dim();
    }
}

// Solve triangular matrix
static inline void solve_triangular(const linalg::KernelMatrix & kernel, std::uint64_t size, double * result) {
    for (std::int64_t i_row = size - 1; i_row >= 0; i_row--) {
        for (std::int64_t i_col = size - 1; i_col > i_row; i_col--) {
            result[i_row] -= result[i_col] * kernel.core[i_col][i_row];
        }
        result[i_row] /= kernel.core[i_row][i_row];
    }
}

// Substract an element from a vector by chunks
static inline void block_substract(const double * column_vector, std::uint64_t nchunks, double elem, double * result) {
    PackedDouble<use_avx> chunk_vector, chunk_result;
    PackedDouble<use_avx> chunk_elem((-1.0) * elem);
    for (std::uint64_t i_chunk = 0; i_chunk < nchunks; i_chunk++) {
        chunk_vector = PackedDouble<use_avx>(column_vector);
        chunk_result = PackedDouble<use_avx>(result);
        chunk_result.fma(chunk_elem, chunk_vector);
        chunk_result.store(result);
        column_vector += 4;
        result += 4;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Upper Triangular
// ---------------------------------------------------------------------------------------------------------------------

// Solve an upper triangular matrix with ones on diagonal elemets
void linalg::triu_one_solve(const linalg::Matrix & triu_matrix, double * solution) noexcept {
    std::uint64_t size = triu_matrix.ncol();
    std::uint64_t nchunks = size / 4, remainder = size % 4, block_size = nchunks * 4;
}

}  // namespace merlin

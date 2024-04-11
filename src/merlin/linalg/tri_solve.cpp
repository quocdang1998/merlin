// Copyright 2023 quocdang1998
#include "merlin/linalg/tri_solve.hpp"

#include <cstdint>  // std::uint64_t

#include "merlin/avx.hpp"            // merlin::AvxDouble, merlin::use_avx
#include "merlin/linalg/matrix.hpp"  // merlin::linalg::Matrix
#include "merlin/logger.hpp"         // WARNING

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// AVX utility
// ---------------------------------------------------------------------------------------------------------------------

// Subtract an element from a vector by chunks
static inline void block_substract(const double * column_vector, std::uint64_t nchunks, double elem, double * result) {
    AvxDouble<use_avx> chunk_vector, chunk_result;
    AvxDouble<use_avx> chunk_elem((-1.0) * elem);
    for (std::uint64_t i_chunk = 0; i_chunk < nchunks; i_chunk++) {
        chunk_vector = AvxDouble<use_avx>(column_vector);
        chunk_result = AvxDouble<use_avx>(result);
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
    // get matrix size
    std::uint64_t size = triu_matrix.ncol();
    std::uint64_t nchunks = size / 4, remainder = size % 4, rstart = nchunks * 4;
    // solve for remainder matrix
    switch (remainder) {
        case 2 : {
            solution[rstart] -= triu_matrix.cget(rstart, rstart + 1) * solution[rstart + 1];
            break;
        }
        case 3 : {
            solution[rstart + 1] -= triu_matrix.cget(rstart + 1, rstart + 2) * solution[rstart + 2];
            solution[rstart] -= triu_matrix.cget(rstart, rstart + 2) * solution[rstart + 2];
            solution[rstart] -= triu_matrix.cget(rstart, rstart + 1) * solution[rstart + 1];
            break;
        }
        default : {
            break;
        }
    }
    for (std::uint64_t i_elem = 0; i_elem < remainder; i_elem++) {
        block_substract(&triu_matrix.cget(0, rstart + i_elem), nchunks, solution[rstart + i_elem], solution);
    }
    // solve for each chunk
    AvxDouble<use_avx> chunk_solution;
    for (std::int64_t i_chunk = static_cast<std::int64_t>(nchunks) - 1; i_chunk >= 0; i_chunk--) {
        std::uint64_t cstart = i_chunk * 4;
        chunk_solution = AvxDouble<use_avx>(solution + cstart);
        // solve triangular matrix
        chunk_solution[2] -= triu_matrix.cget(cstart + 2, cstart + 3) * chunk_solution[3];
        chunk_solution[1] -= triu_matrix.cget(cstart + 1, cstart + 3) * chunk_solution[3];
        chunk_solution[0] -= triu_matrix.cget(cstart, cstart + 3) * chunk_solution[3];
        chunk_solution[1] -= triu_matrix.cget(cstart + 1, cstart + 2) * chunk_solution[2];
        chunk_solution[0] -= triu_matrix.cget(cstart, cstart + 2) * chunk_solution[2];
        chunk_solution[0] -= triu_matrix.cget(cstart, cstart + 1) * chunk_solution[1];
        chunk_solution.store(solution + cstart);
        for (std::uint64_t i_elem = 0; i_elem < 4; i_elem++) {
            block_substract(&triu_matrix.cget(0, cstart + i_elem), i_chunk, chunk_solution[i_elem], solution);
        }
    }
}

}  // namespace merlin

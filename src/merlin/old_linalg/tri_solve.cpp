// Copyright 2023 quocdang1998
#include "merlin/linalg/tri_solve.hpp"

#include <cstdint>  // std::uint64_t

#include "merlin/avx.hpp"            // merlin::AvxDouble, merlin::use_avx
#include "merlin/linalg/matrix.hpp"  // merlin::linalg::Matrix

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

// Solve a small block of triangular matrix with ones on diagonal elements
static inline void block_triu_one_solve(const linalg::Matrix & triu_matrix, double * solution, std::uint64_t start_idx,
                                        std::uint64_t size) noexcept {
    switch (size) {
        case 2 : {
            solution[0] -= triu_matrix.cget(start_idx, start_idx + 1) * solution[1];
            break;
        }
        case 3 : {
            solution[1] -= triu_matrix.cget(start_idx + 1, start_idx + 2) * solution[2];
            solution[0] -= triu_matrix.cget(start_idx, start_idx + 2) * solution[2];
            solution[0] -= triu_matrix.cget(start_idx, start_idx + 1) * solution[1];
            break;
        }
        case 4 : {
            solution[2] -= triu_matrix.cget(start_idx + 2, start_idx + 3) * solution[3];
            solution[1] -= triu_matrix.cget(start_idx + 1, start_idx + 3) * solution[3];
            solution[0] -= triu_matrix.cget(start_idx, start_idx + 3) * solution[3];
            solution[1] -= triu_matrix.cget(start_idx + 1, start_idx + 2) * solution[2];
            solution[0] -= triu_matrix.cget(start_idx, start_idx + 2) * solution[2];
            solution[0] -= triu_matrix.cget(start_idx, start_idx + 1) * solution[1];
        }
        default : {
            break;
        }
    }
}

// Solve an upper triangular matrix with ones on diagonal elemets
void linalg::triu_one_solve(const linalg::Matrix & triu_matrix, double * solution) noexcept {
    // get matrix size
    std::uint64_t size = triu_matrix.ncol();
    std::uint64_t nchunks = size / 4, remainder = size % 4, rstart = nchunks * 4;
    // solve for remainder matrix
    block_triu_one_solve(triu_matrix, solution + rstart, rstart, remainder);
    for (std::uint64_t i_elem = 0; i_elem < remainder; i_elem++) {
        block_substract(&triu_matrix.cget(0, rstart + i_elem), nchunks, solution[rstart + i_elem], solution);
    }
    // solve for each chunk
    AvxDouble<use_avx> chunk_solution;
    for (std::int64_t i_chunk = static_cast<std::int64_t>(nchunks) - 1; i_chunk >= 0; i_chunk--) {
        std::uint64_t cstart = i_chunk * 4;
        chunk_solution = AvxDouble<use_avx>(solution + cstart);
        // solve triangular matrix
        block_triu_one_solve(triu_matrix, chunk_solution.data(), cstart, 4);
        chunk_solution.store(solution + cstart);
        for (std::uint64_t i_elem = 0; i_elem < 4; i_elem++) {
            block_substract(&triu_matrix.cget(0, cstart + i_elem), i_chunk, chunk_solution[i_elem], solution);
        }
    }
}

// Solve a small block of triangular matrix
static inline void block_triu_solve(const linalg::Matrix & triu_matrix, double * solution, std::uint64_t start_idx,
                                    std::uint64_t size) noexcept {
    switch (size) {
        case 1 : {
            solution[0] /= triu_matrix.cget(start_idx, start_idx);
        }
        case 2 : {
            solution[1] /= triu_matrix.cget(start_idx + 1, start_idx + 1);
            solution[0] -= triu_matrix.cget(start_idx, start_idx + 1) * solution[1];
            solution[0] /= triu_matrix.cget(start_idx, start_idx);
            break;
        }
        case 3 : {
            solution[2] /= triu_matrix.cget(start_idx + 2, start_idx + 2);
            solution[1] -= triu_matrix.cget(start_idx + 1, start_idx + 2) * solution[2];
            solution[0] -= triu_matrix.cget(start_idx, start_idx + 2) * solution[2];
            solution[1] /= triu_matrix.cget(start_idx + 1, start_idx + 1);
            solution[0] -= triu_matrix.cget(start_idx, start_idx + 1) * solution[1];
            solution[0] /= triu_matrix.cget(start_idx, start_idx);
            break;
        }
        case 4 : {
            solution[3] /= triu_matrix.cget(start_idx + 3, start_idx + 3);
            solution[2] -= triu_matrix.cget(start_idx + 2, start_idx + 3) * solution[3];
            solution[1] -= triu_matrix.cget(start_idx + 1, start_idx + 3) * solution[3];
            solution[0] -= triu_matrix.cget(start_idx, start_idx + 3) * solution[3];
            solution[2] /= triu_matrix.cget(start_idx + 2, start_idx + 2);
            solution[1] -= triu_matrix.cget(start_idx + 1, start_idx + 2) * solution[2];
            solution[0] -= triu_matrix.cget(start_idx, start_idx + 2) * solution[2];
            solution[1] /= triu_matrix.cget(start_idx + 1, start_idx + 1);
            solution[0] -= triu_matrix.cget(start_idx, start_idx + 1) * solution[1];
            solution[0] /= triu_matrix.cget(start_idx, start_idx);
        }
        default : {
            break;
        }
    }
}

// Solve an upper triangular matrix
void linalg::triu_solve(const linalg::Matrix & triu_matrix, double * solution) noexcept {
    // get matrix size
    std::uint64_t size = triu_matrix.ncol();
    std::uint64_t nchunks = size / 4, remainder = size % 4, rstart = nchunks * 4;
    // solve for remainder matrix
    block_triu_solve(triu_matrix, solution + rstart, rstart, remainder);
    for (std::uint64_t i_elem = 0; i_elem < remainder; i_elem++) {
        block_substract(&triu_matrix.cget(0, rstart + i_elem), nchunks, solution[rstart + i_elem], solution);
    }
    // solve for each chunk
    AvxDouble<use_avx> chunk_solution;
    for (std::int64_t i_chunk = static_cast<std::int64_t>(nchunks) - 1; i_chunk >= 0; i_chunk--) {
        std::uint64_t cstart = i_chunk * 4;
        chunk_solution = AvxDouble<use_avx>(solution + cstart);
        // solve triangular matrix
        block_triu_solve(triu_matrix, chunk_solution.data(), cstart, 4);
        chunk_solution.store(solution + cstart);
        for (std::uint64_t i_elem = 0; i_elem < 4; i_elem++) {
            block_substract(&triu_matrix.cget(0, cstart + i_elem), i_chunk, chunk_solution[i_elem], solution);
        }
    }
}

}  // namespace merlin

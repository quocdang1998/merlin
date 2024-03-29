// Copyright 2023 quocdang1998
#include "merlin/linalg/tri_solve.hpp"

#include <array>    // std::array
#include <cmath>    // std::fma

#include "merlin/avx.hpp"     // merlin::PackedDouble, merlin::use_avx
#include "merlin/logger.hpp"  // WARNING

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Upper Triangular
// ---------------------------------------------------------------------------------------------------------------------

// Inverse remainder matrix
static void triu_one_solve_remainder(const double * triu_matrix, std::uint64_t lda, const double * target,
                                     double * solution, std::uint64_t remainder) noexcept {
    PackedDouble<use_avx> clone_target(target, remainder);
    switch (remainder) {
        case 0: {
            break;
        }
        case 1: {
            break;
        }    
        case 2: {
            clone_target[0] = std::fma((-1) * clone_target[1], triu_matrix[lda], clone_target[0]);
            break;
        }
        case 3: {
            clone_target[1] = std::fma((-1) * clone_target[2], triu_matrix[2 * lda + 1], clone_target[1]);
            clone_target[0] = std::fma((-1) * clone_target[2], triu_matrix[2 * lda], clone_target[0]);
            clone_target[0] = std::fma((-1) * clone_target[1], triu_matrix[lda], clone_target[0]);
            break;
        }
    }
    clone_target.store(solution, remainder);
}

static void triu_one_solve_block(const double * triu_matrix, std::uint64_t lda, const double * target,
                                 double * solution) noexcept {
    
}

// Solve an upper triangular matrix with ones on diagonal elemets
void linalg::triu_one_solve(const double * triu_matrix, std::uint64_t lda, const double * target, double * solution,
                            std::uint64_t size) noexcept {
    std::uint64_t num_chunks = size / 4, remainder = size % 4;
    // inverse remainder
    const double * remainder_matrix = triu_matrix + (num_chunks * 4) * lda + (num_chunks * 4);
    const double * remainder_target = target + (num_chunks * 4);
    double * remainder_solution = solution + (num_chunks * 4);
    triu_one_solve_remainder(remainder_matrix, lda, remainder_target, remainder_solution, remainder);
    // subtract remainder amount to other column and copy result from target to solution
    PackedDouble<use_avx> chunk_element, chunk_column, chunk_solution;
    const double * ptr_target = target;
    double * ptr_solution = solution;
    for (std::uint64_t i_chunk = 0; i_chunk < num_chunks; i_chunk++) {
        // copy values from target
        chunk_solution = PackedDouble<use_avx>(ptr_target);
        // subtract by remainder
        for (std::uint64_t i_elem = 0; i_elem < remainder; i_elem++) {
            chunk_element = PackedDouble<use_avx>((-1) * remainder_solution[i_elem]);
            chunk_column = PackedDouble<use_avx>(triu_matrix + (num_chunks * 4 + i_elem) * lda + 4 * i_chunk);
            chunk_solution.fma(chunk_element, chunk_column);
        }
        // write values back to solution
        chunk_solution.store(ptr_solution);
        // increment pointer
        ptr_target += 4;
        ptr_solution += 4;
    }
}

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

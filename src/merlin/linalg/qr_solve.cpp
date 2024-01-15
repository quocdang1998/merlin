// Copyright 2023 quocdang1998
#include "merlin/linalg/qr_solve.hpp"

#include <cmath>  // std::copysign, std::sqrt

#include <omp.h>  // #pragma omp

#include "merlin/linalg/matrix.hpp"  // merlin::linalg::Matrix

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Matrix inversion by QR decomposition
// ---------------------------------------------------------------------------------------------------------------------

// QR decomposition by CPU parallelism
void linalg::qr_decomposition_cpu(linalg::Matrix & M, linalg::Matrix & B, double * buffer, double & norm,
                                  std::uint64_t thread_idx, std::uint64_t nthreads) noexcept {
    // initialize temporary memory for refectant vector
    floatvec reflect_vector;
    reflect_vector.assign(buffer, M.nrow());
    std::uint64_t dim_max = M.nrow() - 1;
    // recursively call for each dimension
    for (std::uint64_t i_dim = 0; i_dim < dim_max; i_dim++) {
        // get the norm of the first column vector of the sub matrix
        double thread_norm = 0.0;
        if (thread_idx == 0) {
            norm = 0.0;
        }
        #pragma omp barrier
        for (std::uint64_t i_row = i_dim + thread_idx; i_row < M.nrow(); i_row += nthreads) {
            double & element = M.get(i_row, i_dim);
            thread_norm += element * element;
        }
        // std::printf("Thread norm: %f\n", thread_norm);
        #pragma omp atomic update
        norm += thread_norm;
        #pragma omp barrier
        if (thread_idx == 0) {
            norm = std::sqrt(norm);
            // std::printf("Norm of first col: %f\n", norm);
        }
        #pragma omp barrier
        // calculate v = (matrix[:,i_dim] - norm * e_idim) / norm(v)
        for (std::uint64_t i_row = i_dim + thread_idx; i_row < M.nrow(); i_row += nthreads) {
            reflect_vector[i_row] = M.get(i_row, i_dim);
            reflect_vector[i_row] += (i_row == i_dim) ? std::copysign(norm, reflect_vector[i_dim]) : 0.0;
        }
        #pragma omp barrier
        // normalize v
        if (thread_idx == 0) {
            norm = 0.0;
        }
        #pragma omp barrier
        thread_norm = 0.0;
        for (std::uint64_t i_row = i_dim + thread_idx; i_row < M.nrow(); i_row += nthreads) {
            thread_norm += reflect_vector[i_row] * reflect_vector[i_row];
        }
        #pragma omp atomic update
        norm += thread_norm;
        #pragma omp barrier
        if (thread_idx == 0) {
            norm = std::sqrt(norm);
        }
        #pragma omp barrier
        for (std::uint64_t i_row = i_dim + thread_idx; i_row < M.nrow(); i_row += nthreads) {
            reflect_vector[i_row] /= norm;
        }
        #pragma omp barrier
        // perform Householder refection
        linalg::householder_reflect(M, reflect_vector, i_dim, thread_idx, nthreads);
        linalg::householder_reflect(B, reflect_vector, i_dim, thread_idx, nthreads);
        #pragma omp barrier
    }
}

// Solve linear system by QR decomposition
void linalg::qr_solve_cpu(linalg::Matrix & M, linalg::Matrix & B, double * buffer, double & norm,
                          std::uint64_t thread_idx, std::uint64_t nthreads) noexcept {
    linalg::qr_decomposition_cpu(M, B, buffer, norm, thread_idx, nthreads);
    linalg::upright_solver(M, B, thread_idx, nthreads);
}

}  // namespace merlin

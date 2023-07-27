// Copyright 2023 quocdang1998
#include "merlin/linalg/qr_solve.hpp"

#include <cmath>  // std::copysign, std::sqrt

#include <omp.h>  // #pragma

#include "merlin/linalg/matrix.hpp"  // merlin::linalg::Matrix

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Matrix inversion by QR decomposition
// --------------------------------------------------------------------------------------------------------------------

// Solve linear system by QR decomposition
void linalg::qr_solve_cpu(linalg::Matrix & M, floatvec & x, std::uint64_t nthreads) noexcept {
    linalg::qr_decomposition_cpu(M, x, nthreads);
    linalg::upright_solver_cpu(M, x, nthreads);
}

// Inverse matrix by QR decomposition by CPU parallelism
void linalg::qr_decomposition_cpu(linalg::Matrix & matrix, floatvec & x, std::uint64_t nthreads) noexcept {
    const std::uint64_t & size = x.size();
    floatvec v(size);  // allocate vector for temporary memory of calculation
    std::uint64_t dim_max = size - 1;
    for (std::uint64_t i_dim = 0; i_dim < dim_max; i_dim++) {
        // get the norm of the first column vector of the sub matrix
        double norm_alpha = 0.0;
        for (std::uint64_t j_row = i_dim; j_row < size; j_row++) {
            double element = matrix.get(j_row, i_dim);
            norm_alpha +=  element * element;
        }
        norm_alpha = std::sqrt(norm_alpha);
        // calculate v = (matrix[:,i_dim] - alpha * e_idim) / norm(v)
        for (std::uint64_t j = i_dim; j < size; j++) {
            v[j] = matrix.get(j, i_dim);
        }
        v[i_dim] += std::copysign(norm_alpha, v[i_dim]);
        // normalize v
        double norm_v = 0.0;
        for (std::uint64_t j = i_dim; j < size; j++) {
            norm_v += v[j] * v[j];
        }
        norm_v = std::sqrt(norm_v);
        for (std::uint64_t j = i_dim; j < size; j++) {
            v[j] /= norm_v;
        }
        // perform Householder refection
        linalg::householder_cpu(matrix, x, v, i_dim, nthreads);
    }
}

// Perform Householder reflection on a matrix and a vector
void linalg::householder_cpu(linalg::Matrix & M, floatvec & x, const floatvec & v, std::uint64_t start_dimension,
                             std::uint64_t nthreads) noexcept {
    // apply Householder transformation to each matrix column
    const std::uint64_t & matrix_size = v.size();
    #pragma omp parallel for num_threads(nthreads)
    for (std::int64_t i_col = start_dimension; i_col < matrix_size; i_col++) {
        // calculate inner product with column vector
        double column_inner_prod = 0.0;
        for (std::uint64_t i_row = start_dimension; i_row < matrix_size; i_row++) {
            column_inner_prod += M.get(i_row, i_col) * v[i_row];
        }
        column_inner_prod *= 2.f;
        // substract by 2*<col*v>*v_i
        for (std::uint64_t i_row = start_dimension; i_row < matrix_size; i_row++) {
            M.get(i_row, i_col) -= column_inner_prod * v[i_row];
        }
    }
    // apply Householder transformation to the vector
    double inner_xv = 0.f;
    for (std::uint64_t i_row = start_dimension; i_row < matrix_size; i_row++) {
        inner_xv += x[i_row] * v[i_row];
    }
    inner_xv *= 2.f;
    for (std::uint64_t i = start_dimension; i < matrix_size; i++) {
        x[i] -= inner_xv * v[i];
    }
}

// Solve upper right linear system
void linalg::upright_solver_cpu(linalg::Matrix & R, floatvec & x, std::uint64_t nthreads) noexcept {
    const std::uint64_t & size = x.size();
    for (std::int64_t i_row = size-1; i_row >=0; i_row--) {
        // solve for the value of i_row
        double & diagonal_element = R.get(i_row, i_row);
        x[i_row] /= diagonal_element;
        diagonal_element = 1.f;
        // subtract each previous rows by coeff * x[i_row]
        #pragma omp parallel for num_threads(nthreads)
        for (std::uint64_t i_previous = 0; i_previous < i_row; i_previous++) {
            double & off_diagonal = R.get(i_previous, i_row);
            x[i_previous] -= off_diagonal * x[i_row];
            off_diagonal = 0.f;
        }
    }
}

}  // namespace merlin

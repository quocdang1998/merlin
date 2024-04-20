// Copyright 2023 quocdang1998
#include "merlin/linalg/qrp_decomp.hpp"

#include <algorithm>  // std::max_element, std::swap
#include <cmath>      // std::copysign, std::sqrt
#include <iterator>   // std::distance

#include <omp.h>  // ::omp_get_thread_num

#include <iostream>

#include "merlin/linalg/dot.hpp"        // merlin::linalg::norm
#include "merlin/linalg/matrix.hpp"     // merlin::linalg::Matrix
#include "merlin/linalg/tri_solve.hpp"  // merlin::linalg::triu_one_solve
#include "merlin/logger.hpp"            // merlin::Fatal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// QR Decomposition with Column Pivoting
// ---------------------------------------------------------------------------------------------------------------------

// Constructor of an empty matrix from shape
linalg::QRPDecomp::QRPDecomp(std::uint64_t nrow, std::uint64_t ncol) : core_(nrow, ncol), diag_(ncol), permut_(ncol) {
    if (nrow < ncol) {
        Fatal<std::invalid_argument>("Expected nrow >= ncol, please transpose the matrix.\n");
    }
}

// Perform the QR decomposition with column pivoting.
void linalg::QRPDecomp::decompose(std::uint64_t nthreads) {
    // error if matrix is not initialized
    if (this->core_.data() == nullptr) {
        Fatal<std::invalid_argument>("Cannot decompose a non-initialized matrix.\n");
    }
    // check if the current object is decomposed
    if (this->is_decomposed) {
        Fatal<std::runtime_error>("The current object is already decomposed.\n");
    }
    // calculate norm of each vector and save it into the diagonal vector
    std::uint64_t nchunks = this->nrow() / 4, remainder = this->nrow() % 4;
    _Pragma("omp parallel num_threads(nthreads)") {
        std::uint64_t thread_idx = ::omp_get_thread_num();
        for (std::uint64_t i_col = thread_idx; i_col < this->ncol(); i_col += nthreads) {
            linalg::avx_norm(&(this->core_.get(0, i_col)), nchunks, remainder, this->diag_[i_col]);
            this->diag_[i_col] = std::sqrt(this->diag_[i_col]);
        }
    }
    // perform decomposition on each column
    for (std::uint64_t i_col = 0; i_col < this->ncol(); i_col++) {
        // permute current column with column of max norm
        std::uint64_t max_idx = std::distance(this->diag_.begin(),
                                              std::max_element(this->diag_.data() + i_col, this->diag_.end()));
        std::swap(this->diag_[i_col], this->diag_[max_idx]);
        linalg::avx_swap(&(this->core_.get(0, i_col)), &(this->core_.get(0, max_idx)), nchunks, remainder);
        this->permut_.transpose(i_col, max_idx);
        // transform the i_col-th vector into a reflector and save diagonal element into diag vector
        double * reflector = &(this->core_.get(i_col, i_col));
        this->diag_[i_col] = std::copysign(this->diag_[i_col], reflector[0]);
        reflector[0] += this->diag_[i_col];
        this->diag_[i_col] *= -1.0;
        std::uint64_t sub_nchunks = (this->nrow() - i_col) / 4, sub_remainder = (this->nrow() - i_col) % 4;
        std::uint64_t sub_nchunks_1 = (this->nrow() - i_col - 1) / 4, sub_remainder_1 = (this->nrow() - i_col - 1) % 4;
        linalg::avx_normalize(reflector, reflector, sub_nchunks, sub_remainder);
        // apply Householder reflection to other columns and recalculate the norm
        _Pragma("omp parallel num_threads(nthreads)") {
            std::uint64_t thread_idx = ::omp_get_thread_num();
            for (std::uint64_t j_col = i_col + thread_idx + 1; j_col < this->ncol(); j_col += nthreads) {
                // apply Householder reflection
                double * target = &(this->core_.get(i_col, j_col));
                linalg::avx_householder(reflector, target, sub_nchunks, sub_remainder);
                // recalculating the norm
                // this->diag_[j_col] = std::sqrt((this->diag_[j_col] + target[0]) * (this->diag_[j_col] - target[0]));
                linalg::avx_norm(target + 1, sub_nchunks_1, sub_remainder_1, this->diag_[j_col]);
                this->diag_[j_col] = std::sqrt(this->diag_[j_col]);
                // update i_col-th row
                target[0] /= this->diag_[i_col];
            }
        }
    }
    // switch on the flag
    this->is_decomposed = true;
}

// Solve the linear least-square problem
double linalg::QRPDecomp::solve(double * solution) const {
    // check if this instance is decomposed
    if (!(this->is_decomposed)) {
        Fatal<std::invalid_argument>("Cannot solve a vector using a non-decomposed QRPDecomp object.\n");
    }
    // apply Householder reflection for each column
    for (std::uint64_t i_col = 0; i_col < this->ncol(); i_col++) {
        const double * reflector = &(this->core_.cget(i_col, i_col));
        std::uint64_t nchunks = (this->nrow() - i_col) / 4, remainder = (this->nrow() - i_col) % 4;
        linalg::avx_householder(reflector, solution + i_col, nchunks, remainder);
    }
    // apply division by diagonal element
    linalg::vecdiv(solution, this->diag_.data(), solution, this->ncol());
    // solve right upper triangular matrix
    linalg::triu_one_solve(this->core_, solution);
    // permute solution
    Permutation perm_copy(this->permut_);
    perm_copy.inplace_permute(solution);
    // residual
    double residual;
    linalg::norm(solution + this->ncol(), this->nrow() - this->ncol(), residual);
    return residual;
}

}  // namespace merlin

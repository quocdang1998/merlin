// Copyright 2024 quocdang1998
#include "merlin/regpl/polynomial.hpp"

#include <cmath>    // std::pow
#include <sstream>  // std::ostringstream

#include <omp.h>  // #pragma omp

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/logger.hpp"  // FAILURE, merlin::cuda_compile_error
#include "merlin/utils.hpp"  // merlin::prod_elements

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Polynomial
// ---------------------------------------------------------------------------------------------------------------------

// Constructor of an empty polynomial from order per dimension
regpl::Polynomial::Polynomial(const intvec & order_per_dim) :
order_(order_per_dim), coeff_(prod_elements(order_per_dim)) { }

// Constructor of a pre-allocated array of coefficients and order per dimension
regpl::Polynomial::Polynomial(double * coeff_data, const intvec & order_per_dim) : order_(order_per_dim) {
    this->coeff_.assign(coeff_data, prod_elements(order_per_dim));
}

// Generate Vandermonde matrix
array::Array regpl::Polynomial::calc_vandermonde(const array::Array & grid_points, std::uint64_t n_threads) const {
    // check argument
    if (grid_points.ndim() != 2) {
        FAILURE(std::invalid_argument, "Expected array of 2 dimensions.\n");
    }
    if (grid_points.shape()[1] != this->ndim()) {
        FAILURE(std::invalid_argument, "Points in grid have a different number of dimension as the polynomial.\n");
    }
    // allocate matrix
    std::uint64_t num_points = grid_points.shape()[0];
    array::Array vandermonde_matrix({num_points, this->size()});
    #pragma omp parallel num_threads(n_threads)
    {
        std::uint64_t thread_idx = ::omp_get_thread_num();
        intvec point_index(2);
        // loop for each point in the grid
        for (std::uint64_t i_point = thread_idx; i_point < num_points; i_point += n_threads) {
            // get pointer to corresponding row
            double * row_data = vandermonde_matrix.data() + i_point * this->size();
            point_index[0] = i_point;
            // loop for each monomial
            for (std::uint64_t i_term = 0; i_term < this->size(); i_term++) {
                row_data[i_term] = 1;
                // calculate monomial expansion for each dimension
                std::uint64_t cum_prod = 1;
                for (std::int64_t i_dim = this->ndim() - 1; i_dim >= 0; i_dim--) {
                    std::uint64_t nd_index = (i_term / cum_prod) % this->order_[i_dim];
                    point_index[1] = i_dim;
                    row_data[i_term] *= std::pow(grid_points.get(point_index), nd_index);
                    cum_prod *= this->order_[i_dim];
                }
            }
        }
    }
    return vandermonde_matrix;
}

#ifndef __MERLIN_CUDA__

// Copy data to a pre-allocated memory
void * regpl::Polynomial::copy_to_gpu(regpl::Polynomial * gpu_ptr, void * coeff_data_ptr,
                                      std::uintptr_t stream_ptr) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

// Copy data from GPU to CPU
void * regpl::Polynomial::copy_from_gpu(double * data_from_gpu, std::uintptr_t stream_ptr) noexcept {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

#endif  // __MERLIN_CUDA__

// String representation
std::string regpl::Polynomial::str(void) const {
    std::ostringstream os;
    os << "<Polynomial(";
    os << "order=" << this->order_.str() << ", ";
    os << "coeff=" << this->coeff_.str();
    os << ")";
    return os.str();
}

}  // namespace merlin

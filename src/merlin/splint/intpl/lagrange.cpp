// Copyright 2022 quocdang1998
#include "merlin/splint/intpl/lagrange.hpp"

#include <omp.h>  // #pragma omp, omp_get_num_threads

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Lagrange interpolation (CPU)
// ---------------------------------------------------------------------------------------------------------------------

// Construct interpolation coefficients by Lagrange method on CPU
void splint::intpl::construction_lagrange_cpu(double * coeff, const double * grid_nodes, std::uint64_t shape,
                                              std::uint64_t element_size, std::uint64_t thread_idx,
                                              std::uint64_t n_threads) noexcept {
    for (std::uint64_t i_node = 0; i_node < shape; i_node++) {
        // calculate solvent = prod (x_i_node - x_j) for all j != i_node
        double solvent = 1.0;
        for (std::uint64_t j = 0; j < shape; j++) {
            solvent *= (j != i_node) ? (grid_nodes[i_node] - grid_nodes[j]) : 1.0;
        }
        // inverse the solvent
        solvent = 1.f / solvent;
        // multiply each sub-array by solvent
        for (std::int64_t i_coeff = thread_idx; i_coeff < element_size; i_coeff += n_threads) {
            coeff[i_node*element_size + i_coeff] *= solvent;
        }
    }
}

static double eval_lagrange_basis_function(const double * grid_nodes, const std::uint64_t & grid_shape,
                                           const double & point, const std::uint64_t & i_node) noexcept {
    double eval_basis_function = 1.0;
    for (std::uint64_t i = 0; i < grid_shape; i++) {
        eval_basis_function *= (i != i_node) ? (point - grid_nodes[i]) : 1.0;
    }
    return eval_basis_function;
}

// Interpolate recursively on each dimension
void splint::intpl::eval_lagrange_cpu(const double * coeff, const std::uint64_t & num_coeff,
                                      const std::uint64_t & c_index_coeff, const std::uint64_t * ndim_index_coeff,
                                      double * cache_array, const double * point, const std::int64_t & i_dim,
                                      const std::uint64_t * grid_shape, double * const * grid_vectors,
                                      const std::uint64_t & ndim) {
    // trivial case for the last dimension
    if (i_dim == ndim-1) {
        double eval_basis_function = eval_lagrange_basis_function(grid_vectors[i_dim], grid_shape[i_dim],
                                                                  point[i_dim], ndim_index_coeff[i_dim]);
        cache_array[i_dim] += coeff[c_index_coeff] * eval_basis_function;
    } else {
        // recursive save upto the current dimension
        for (std::int64_t i = ndim-2; i >= i_dim; i--) {
            std::uint64_t previous_index = (ndim_index_coeff[i] != 0) ? ndim_index_coeff[i] - 1 : grid_shape[i] - 1;
            double eval_basis_function = eval_lagrange_basis_function(grid_vectors[i], grid_shape[i],
                                                                      point[i], previous_index);
            cache_array[i] += cache_array[i+1] * eval_basis_function;
            cache_array[i+1] = 0.0;
        }
        // calculate for the current index on the last dimension
        double last_dim_eval = eval_lagrange_basis_function(grid_vectors[ndim-1], grid_shape[ndim-1], point[ndim-1], 0);
        cache_array[ndim-1] = coeff[c_index_coeff] * last_dim_eval;
    }
}

}  // namespace merlin

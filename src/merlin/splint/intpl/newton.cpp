// Copyright 2022 quocdang1998
#include "merlin/splint/intpl/newton.hpp"

#include <omp.h>  // #pragma omp, omp_get_num_threads

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Lagrange interpolation (CPU)
// ---------------------------------------------------------------------------------------------------------------------

// Construct interpolation coefficients by Newton method on CPU
void splint::intpl::construction_newton_cpu(double * coeff, double * grid_nodes, std::uint64_t shape,
                                            std::uint64_t element_size, std::uint64_t thread_idx,
                                            std::uint64_t n_threads) noexcept {
    for (std::uint64_t i_node = 1; i_node < shape; i_node++) {
        for (std::uint64_t j_node = shape-1; j_node >= i_node; j_node--) {
            // calculate 1 / (x_j - x_{j-i})
            double solvent = grid_nodes[j_node] - grid_nodes[j_node-i_node];
            solvent = 1.f / solvent;
            // calculate divide difference: a[j] <- (a[j] - a[j-1]) /  (x[j] - x[j-i])
            for (std::int64_t i_coeff = thread_idx; i_coeff < element_size; i_coeff += n_threads) {
                coeff[j_node*element_size + i_coeff] -= coeff[(j_node-1)*element_size + i_coeff];
                coeff[j_node*element_size + i_coeff] *= solvent;
            }
        }
    }
}

}  // namespace merlin

// Copyright 2022 quocdang1998
#include "merlin/splint/intpl/lagrange.hpp"

#include <omp.h>  // #pragma omp, omp_get_num_threads

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Lagrange interpolation (CPU)
// ---------------------------------------------------------------------------------------------------------------------

// Construct interpolation coefficients by Lagrange method on CPU
void splint::intpl::construction_lagrange_cpu(double * coeff, double * grid_nodes, std::uint64_t shape,
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

}  // namespace merlin

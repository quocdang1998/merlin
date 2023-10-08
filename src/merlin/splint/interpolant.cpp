// Copyright 2022 quocdang1998
#include "merlin/splint/interpolant.hpp"

#include "merlin/array/array.hpp"            // merlin::array::Array
#include "merlin/logger.hpp"                 // FAILURE
#include "merlin/splint/cartesian_grid.hpp"  // merlin::splint::CartesianGrid
#include "merlin/splint/intpl/map.hpp"       // merlin::splint::intpl::construction_func_cpu
#include "merlin/utils.hpp"                  // merlin::prod_elements

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Construct coefficients by CPU
// ---------------------------------------------------------------------------------------------------------------------

// Construct interpolation coefficients
void splint::construct_coeff_cpu(double * coeff, const splint::CartesianGrid & grid,
                                 const Vector<splint::Method> & method, std::uint64_t n_threads) noexcept {
    // initialization
    const intvec & shape = grid.shape();
    std::uint64_t num_subsystem = 1, element_size = prod_elements(shape);
    // solve matrix for each dimension
    for (std::uint64_t i_dim = 0; i_dim < grid.ndim(); i_dim++) {
        std::uint64_t subsystem_size = element_size;
        element_size /= shape[i_dim];
        unsigned int i_method = static_cast<unsigned int>(method[i_dim]);
        std::uint64_t thread_per_subsystem = ((num_subsystem >= n_threads) ? 1 : n_threads);
        // parallel at subsystem level when num_subsystem >= n_threads
        #pragma omp parallel for num_threads(n_threads) if (num_subsystem >= n_threads)
        for (std::uint64_t i_subsystem = 0; i_subsystem < num_subsystem; i_subsystem++) {
            double * subsystem_start = coeff + i_subsystem * subsystem_size;
            splint::intpl::construction_func_cpu[i_method](subsystem_start, grid.grid_vectors()[i_dim], shape[i_dim],
                                                           element_size, thread_per_subsystem);
        }
        num_subsystem *= shape[i_dim];
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Interpolant
// ---------------------------------------------------------------------------------------------------------------------

// Construct from a CPU array
splint::Interpolant::Interpolant(const splint::CartesianGrid & grid, const array::Array & data,
                                 const Vector<splint::Method> & method) :
p_grid_(&grid), method_(method) {
    // check shape
    if (grid.shape() != data.shape()) {
        FAILURE(std::invalid_argument, "Grid and data have different shape.\n");
    }
    // copy data
    this->p_coeff_ = new array::Array(data);
}

// Destructor
splint::Interpolant::~Interpolant(void) {
    if (this->p_coeff_ != nullptr) {
        delete this->p_coeff_;
    }
}

}  // namespace merlin

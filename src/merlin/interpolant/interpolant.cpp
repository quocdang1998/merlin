// Copyright 2022 quocdang1998
#include "merlin/interpolant/interpolant.hpp"

#include <omp.h>  // pragma omp
#include <utility>  // std::pair

#include "merlin/array/copy.hpp"  // merlin::array::array_copy
#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CatesianGrid
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Calculate coefficients
// --------------------------------------------------------------------------------------------------------------------

// Calculate lagrange coefficient with CPU parallelism
array::Array calc_lagrange_coeffs_cpu(const interpolant::CartesianGrid * pgrid, const array::Array * pvalue,
                                      const Vector<array::Slice> & slices) {
    // get information
    std::uint64_t value_size = pvalue->size();
    std::uint64_t ndim = pvalue->ndim();
    array::Array result(pvalue->shape());
    // loop over each point in the grid
    #pragma omp parallel for
    for (int i = 0; i < value_size; i++) {
        intvec index_in_value_array = contiguous_to_ndim_idx(i, pvalue->shape());
        intvec index_in_grid(ndim);
        for (int i_dim = 0; i_dim < ndim; i_dim++) {
            index_in_grid[i_dim] = slices[i_dim].get_index_in_whole_array(index_in_value_array[i_dim]);
        }
        float denominator = 1.0;
        for (int i_dim = 0; i_dim < ndim; i_dim++) {
            for (int i_node = 0; i_node < pgrid->grid_shape()[i_dim]; i_node++) {
                if (i_node == index_in_grid[i_dim]) {
                    continue;
                }
                denominator *= (pgrid->grid_vectors()[i_dim][index_in_grid[i_dim]]
                                - pgrid->grid_vectors()[i_dim][i_node]);
            }
        }
        result.set(i, pvalue->get(index_in_value_array) / static_cast<float>(denominator));
    }
    return result;
}

// --------------------------------------------------------------------------------------------------------------------
// CartesianInterpolant
// --------------------------------------------------------------------------------------------------------------------

interpolant::CartesianInterpolant::CartesianInterpolant(const interpolant::CartesianGrid & grid,
                                                        const array::NdData & value, array::NdData & coeff,
                                                        interpolant::CartesianInterpolant::Method method) :
pgrid_(&grid), pvalue_(&value), pcoeff_(&coeff) {
    // check ndim and shape of grid and value
    if (grid.ndim() != value.ndim()) {
        FAILURE(std::invalid_argument, "Ndim of Grid (%d) and value tensor (%d) are inconsistent.\n",
                grid.ndim(), value.ndim());
    }
    intvec grid_shape = grid.grid_shape();
    for (int i = 0; i < grid.ndim(); i++) {
        if (grid_shape[i] != value.shape()[i]) {
            FAILURE(std::invalid_argument, "Expected shape Grid (%d) less or equal value tensor (%d) at index %d.\n",
                    grid_shape[i], value.shape()[i], i);
        }
    }
    // check ndim and shape of coeff and value
    if (coeff.ndim() != value.ndim()) {
        FAILURE(std::invalid_argument, "Ndim of Coeff (%d) and value tensor (%d) are inconsistent.\n",
                coeff.ndim(), value.ndim());
    }
    for (int i = 0; i < value.ndim(); i++) {
        if (coeff.shape()[i] != value.shape()[i]) {
            FAILURE(std::invalid_argument, "Expected shape Coeff (%d) equal value tensor (%d) at index %d.\n",
                    coeff.shape()[i], value.shape()[i], i);
        }
    }
}

}  // namespace merlin

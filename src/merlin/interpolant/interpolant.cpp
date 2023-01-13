// Copyright 2022 quocdang1998
#include "merlin/interpolant/interpolant.hpp"

#include <cinttypes>
#include <omp.h>  // pragma omp
#include <utility>  // std::pair

#include "merlin/array/copy.hpp"  // merlin::array::array_copy
#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CatesianGrid
#include "merlin/interpolant/sparse_grid.hpp"  // merlin::interpolant::SparseGrid
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx, merlin::cumulative_prod
#include "merlin/vector.hpp"  // merlin::floatvec, merlin::intvec

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Calculate coefficients
// --------------------------------------------------------------------------------------------------------------------

// Evaluate value of function at a point x (supposed the size is checked, with isn't implemented)
float eval_lagrange_cpu(const interpolant::CartesianGrid * pgrid, const array::NdData * pcoeff,
                        const merlin::floatvec & x) {
    // check if value of dimension i is value of a given point in grid
    std::uint64_t ndim = pgrid->ndim();
    Vector<std::uint64_t> grid_line_index(ndim, UINT64_MAX);
    std::uint64_t sub_coeff_ndim = 0;  // number of dimension on which evaluate point lies on grid line
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        const floatvec & nodes = pgrid->grid_vectors()[i_dim];
        for (std::uint64_t i_node = 0; i_node < nodes.size(); i_node++) {
            if (x[i_dim] == nodes[i_node]) {
                grid_line_index[i_dim] = i_node;
                sub_coeff_ndim++;
                break;
            }
        }
    }
    // calculate common factor for points on grid line
    float common_factor = 1.0;
    float product_point = 1.0;
    intvec sub_coeff_shape(ndim - sub_coeff_ndim);  // size of dimension of which point do not lie on grid line
    int i_sub_coeff = 0;
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        const floatvec & nodes = pgrid->grid_vectors()[i_dim];
        if (grid_line_index[i_dim] != UINT64_MAX) {
            // calculate common_factor if point on grid line
            for (std::uint64_t i_node = 0; i_node < nodes.size(); i_node++) {
                if (i_node == grid_line_index[i_dim]) {
                    continue;
                }
                common_factor *= (nodes[grid_line_index[i_dim]] - nodes[i_node]);
            }
        } else {
            // calculate product of point wrt. every node
            for (std::uint64_t i_node = 0; i_node < nodes.size(); i_node++) {
                product_point *= (x[i_dim] - nodes[i_node]);
            }
            sub_coeff_shape[i_sub_coeff++] = nodes.size();
        }
    }
    // calculate interpolated value
    std::uint64_t size = 1;
    for (std::uint64_t i = 0; i < sub_coeff_shape.size(); i++) {
        size *= sub_coeff_shape[i];
    }
    float result = 0.0;
    for (std::uint64_t i = 0; i < size; i++) {
        intvec loop_index = contiguous_to_ndim_idx(i, sub_coeff_shape);
        // get index vector corresponding to i
        intvec index(ndim, 0);
        int index_coeff = 0;
        float denominator = 1.0;
        for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
            if (grid_line_index[i_dim] == UINT64_MAX) {
                index[i_dim] = loop_index[index_coeff++];
                denominator *= x[i_dim] - pgrid->grid_vectors()[i_dim][index[i_dim]];
            } else {
                index[i_dim] = grid_line_index[i_dim];
            }
        }
        result += pcoeff->get(index) * common_factor * product_point / denominator;
    }
    return result;
}

// Calculate lagrange coefficient with CPU parallelism
array::Array calc_lagrange_coeffs_cpu(const interpolant::CartesianGrid * pgrid, const array::Array * pvalue,
                                      const Vector<array::Slice> & slices) {
    // get information
    std::uint64_t value_size = pvalue->size();
    std::uint64_t ndim = pvalue->ndim();
    array::Array result(pvalue->shape());
    // loop over each point in the grid
    #pragma omp parallel for
    for (std::uint64_t i = 0; i < value_size; i++) {
        intvec index_in_value_array = contiguous_to_ndim_idx(i, pvalue->shape());
        intvec index_in_grid(ndim);
        for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
            index_in_grid[i_dim] = slices[i_dim].get_index_in_whole_array(index_in_value_array[i_dim]);
        }
        float denominator = 1.0;
        for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
            for (std::uint64_t i_node = 0; i_node < pgrid->grid_shape()[i_dim]; i_node++) {
                if (i_node == index_in_grid[i_dim]) {
                    continue;
                }
                denominator *= pgrid->grid_vectors()[i_dim][index_in_grid[i_dim]]
                               - pgrid->grid_vectors()[i_dim][i_node];
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
    for (std::uint64_t i = 0; i < grid.ndim(); i++) {
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
    for (std::uint64_t i = 0; i < value.ndim(); i++) {
        if (coeff.shape()[i] != value.shape()[i]) {
            FAILURE(std::invalid_argument, "Expected shape Coeff (%d) equal value tensor (%d) at index %d.\n",
                    coeff.shape()[i], value.shape()[i], i);
        }
    }
}

}  // namespace merlin

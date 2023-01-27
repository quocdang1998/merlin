// Copyright 2022 quocdang1998
#include "merlin/interpolant/interpolant.hpp"

#include <algorithm>  // std::max
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
void calc_lagrange_coeffs_cpu(const interpolant::CartesianGrid * pgrid, const array::Array * pvalue,
                              const Vector<array::Slice> & slices, array::Array * presult) {
    // get information
    std::uint64_t value_size = pvalue->size();
    std::uint64_t ndim = pvalue->ndim();
    // check shape of result is identical to shape of p value
    if (presult->ndim() != pvalue->ndim()) {
        FAILURE(std::invalid_argument, "Expected result array has the same n-dim as value array.\n");
    }
    for (std::uint64_t i_dim = 0; i_dim < ndim; ++i_dim) {
        if (presult->shape()[i_dim] != pvalue->shape()[i_dim]) {
            FAILURE(std::invalid_argument, "Expected result array has the same shape as value array.\n");
        }
    }
    // loop over each point in the grid (i is signed since OpenMP requires)
    #pragma omp parallel for
    for (std::int64_t i = 0; i < value_size; i++) {
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
        presult->set(i, pvalue->get(index_in_value_array) / static_cast<float>(denominator));
    }
}

static Vector<array::Slice> get_slice_from_level(const intvec & level, const intvec & max_level) {
    Vector<array::Slice> result(level.size());
    for (std::uint64_t i_dim = 0; i_dim < level.size(); i_dim++) {
        // calculate the size of max_level
        const std::uint64_t & max = max_level[i_dim];
        std::uint64_t size = (max == 0) ? 1 : (1 << max) + 1;
        // calculate start point and step
        const std::uint64_t & lv = level[i_dim];
        if (lv == 0) {
            result[i_dim] = array::Slice({(size - 1) / 2});
        } else if (lv == 1) {
            result[i_dim].step() = size-1;
        } else {
            result[i_dim].step() = 1 << (max_level[i_dim] - level[i_dim] + 1);
            result[i_dim].start() = result[i_dim].step() >> 1;
        }
    }
    return result;
}

array::Array calc_lagrange_coeffs_cpu(const interpolant::SparseGrid * pgrid, const array::Array * pvalue) {
    // check size of p_value
    if (pvalue->ndim() != 1) {
        FAILURE(std::runtime_error, "Invalid shpae of value array.\n");
    }
    if (pvalue->size() != pgrid->size()) {
        FAILURE(std::invalid_argument, "Expected value array and grid having the same size.");
    }
    // loop over each level, updat maximum current level
    std::uint64_t num_level = pgrid->level_vectors().size() / pgrid->ndim();
    intvec max_current_level(pgrid->ndim(), 0);
    interpolant::CartesianGrid current_cart_grid(pgrid->ndim());
    array::Array result(pvalue->shape());
    for (std::uint64_t i_level = 0; i_level < num_level; ++i_level) {
        intvec level;
        level.data() = const_cast<std::uint64_t *>(pgrid->level_vectors().data()) + i_level*pvalue->ndim();
        level.size() = pvalue->ndim();
        // update max current level
        for (std::uint64_t i_dim = 0; i_dim < max_current_level.size(); i_dim++) {
            max_current_level[i_dim] = std::max(max_current_level[i_dim], level[i_dim]);
        }
        // calculate union of cartesian grid
        current_cart_grid = current_cart_grid + pgrid->get_cartesian_grid(level);
        // get values of cart grid
        array::Slice sub_slice(pgrid->sub_grid_start_index()[i_level], pgrid->sub_grid_start_index()[i_level+1], 1);
        Vector<array::Slice> vec_slice({sub_slice});
        array::Array level_value(*pvalue, Vector<array::Slice>({sub_slice}));
        // get slices
        Vector<array::Slice> level_slc = get_slice_from_level(level, max_current_level);
        // get sub array of result
        array::Array level_coeff(result, vec_slice);
        // calculate coefficient
        calc_lagrange_coeffs_cpu(&current_cart_grid, &level_value, level_slc, &level_coeff);
        // neutralize level vector
        level.data() = nullptr;
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

// Copyright 2022 quocdang1998
#include "merlin/interpolant/lagrange.hpp"

#include <algorithm>  // std::max

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/copy.hpp"  // merlin::array::contiguous_strides
#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/array/stock.hpp"  // merlin::array::Stock
#include "merlin/logger.hpp"  // CUHDERR
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/interpolant/sparse_grid.hpp"  // merlin::interpolant::SparseGrid
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx, merlin::get_level_from_valid_size,
                             // merlin::get_level_shape
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Utils
// --------------------------------------------------------------------------------------------------------------------

static Vector<array::Slice> get_slice_from_level(const intvec & level_index, const intvec & max_level) {
    Vector<array::Slice> result(level_index.size());
    for (std::uint64_t i_dim = 0; i_dim < result.size(); i_dim++) {
        std::uint64_t grid_size = (1 << max_level[i_dim]) + 1;
        switch (level_index[i_dim]) {
        case 0:
            result[i_dim] = array::Slice({(grid_size - 1) / 2});
            break;
        case 1:
            result[i_dim].step() = grid_size - 1;
            break;
        default:
            result[i_dim].start() = 1 << (max_level[i_dim] - level_index[i_dim]);
            result[i_dim].step() = 2 * result[i_dim].start();
            break;
        }
    }
    return result;
}

static void accumulate_level(intvec & old_level, const intvec & new_level) {
    for (std::uint64_t i_dim = 0; i_dim < old_level.size(); i_dim++) {
        old_level[i_dim] = std::max(old_level[i_dim], new_level[i_dim]);
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Calculate coefficient
// --------------------------------------------------------------------------------------------------------------------

// Calculate coefficients for cartesian grid
void interpolant::calc_lagrange_coeffs_cpu(const interpolant::CartesianGrid & grid, const array::Array & value,
                                           const Vector<array::Slice> & slices, array::Array & coeff) {
    std::uint64_t value_size = value.size();
    std::uint64_t ndim = grid.ndim();
    intvec grid_shape = grid.get_grid_shape();
    // parallel loop calculation
    #pragma omp parallel for
    for (std::uint64_t i = 0; i < value_size; i++) {
        intvec index_in_value_array = contiguous_to_ndim_idx(i, value.shape());
        // calculate index wrt. the full grid
        intvec index_in_grid(ndim);
        for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
            index_in_grid[i_dim] = slices[i_dim].get_index_in_whole_array(index_in_value_array[i_dim]);
        }
        // calculate the denomiantor (product of diferences of node values)
        long double denominator = 1.0;
        for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
            for (std::uint64_t i_node = 0; i_node < grid_shape[i_dim]; i_node++) {
                // skip for repeating index
                if (i_node == index_in_grid[i_dim]) {
                    continue;
                }
                denominator *= grid.grid_vectors()[i_dim][index_in_grid[i_dim]] - grid.grid_vectors()[i_dim][i_node];
            }
        }
        double result = value.get(index_in_value_array) / static_cast<double>(denominator);
        coeff.set(index_in_value_array, result);
    }
}

// // Calculate coefficients for sparse grid after copied value to coeff
void interpolant::calc_lagrange_coeffs_cpu(const interpolant::SparseGrid & grid, array::NdData & coeff) {
    // initialize common used value
    std::uint64_t num_level = grid.num_level();
    intvec max_level = grid.max_levels();
    std::uint64_t grid_size = grid.size();
    // initialize accumulated max level and cartesian grid
    interpolant::CartesianGrid accumulated_cart_grid(grid.ndim());
    intvec accumulated_max_level(grid.ndim(), 0);
    for (std::uint64_t i_level = 0; i_level < num_level; i_level++) {
        // get level index vector and accumulated max level
        intvec level_index = grid.level_index(i_level);
        accumulate_level(accumulated_max_level, level_index);
        // calculate coefficient on cartesian grid of that level
        accumulated_cart_grid += grid.get_cartesian_grid(level_index);
        // get slice of value array wrt. the accumulated grid
        Vector<array::Slice> sub_slice = get_slice_from_level(level_index, accumulated_max_level);
        // reshape the coefficient sub-array
        array::Array * sub_coeff;
        intvec coeff_shape = get_level_shape(level_index);
        intvec coeff_strides = array::contiguous_strides(coeff_shape, sizeof(double));
        if (array::Array * pcoeff = dynamic_cast<array::Array *>(&coeff); pcoeff != nullptr) {
            double * coeff_data = &(coeff.data()[grid.sub_grid_start_index()[i_level]]);
            sub_coeff = new array::Array(coeff_data, grid.ndim(), coeff_shape.data(), coeff_strides.data(), false);
        } else if (array::Parcel * pcoeff = dynamic_cast<array::Parcel *>(&coeff); pcoeff != nullptr) {
            sub_coeff = new array::Array(coeff_shape);
            intvec temporary_shape({grid.sub_grid_start_index()[i_level+1] - grid.sub_grid_start_index()[i_level]});
            intvec temporary_strides({sizeof(double)});
            array::Array temporary(sub_coeff->data(), 1, temporary_shape.data(), temporary_strides.data(), false);
            Vector<array::Slice> slice_on_coeff({array::Slice({grid.sub_grid_start_index()[i_level],
                                                               grid.sub_grid_start_index()[i_level+1]})});
            array::Parcel sliced_coeff(*pcoeff, slice_on_coeff);
            temporary.clone_data_from_gpu(sliced_coeff);
        } else if (array::Stock * pcoeff = dynamic_cast<array::Stock *>(&coeff); pcoeff != nullptr) {
            sub_coeff = new array::Array(coeff_shape);
            intvec temporary_shape({grid.sub_grid_start_index()[i_level+1] - grid.sub_grid_start_index()[i_level]});
            intvec temporary_strides({sizeof(double)});
            array::Array temporary(sub_coeff->data(), 1, temporary_shape.data(), temporary_strides.data(), false);
            Vector<array::Slice> slice_on_coeff({array::Slice({grid.sub_grid_start_index()[i_level],
                                                               grid.sub_grid_start_index()[i_level+1]})});
            array::Stock sliced_coeff(*pcoeff, slice_on_coeff);
            temporary.extract_data_from_file(sliced_coeff);
        } else {
            FAILURE(std::invalid_argument, "Cannot down cast NdData pointer to Array, Parcel or Stock.\n");
        }
        // calculate coefficient (changeble) -> possibly move to template
        interpolant::calc_lagrange_coeffs_cpu(accumulated_cart_grid, *sub_coeff, sub_slice, *sub_coeff);
        // subtract value by the amount of the evaluation
        for (std::uint64_t i_point = grid.sub_grid_start_index()[i_level+1]; i_point < grid_size; i_point++) {
            Vector<double> point_coordinate = grid.point_at_index(grid.index_from_contiguous(i_point));
            double value = eval_lagrange_cpu(accumulated_cart_grid, *sub_coeff, sub_slice, point_coordinate);
            coeff.set(i_point, coeff.get(i_point) - value);
        }
        delete sub_coeff;
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Evaluate interpolation
// --------------------------------------------------------------------------------------------------------------------

double interpolant::eval_lagrange_cpu(const interpolant::CartesianGrid & grid, const array::Array & coeff,
                                      const Vector<array::Slice> & slices, const Vector<double> & x) {
    // check if point x lies on a hyperplane passing through a node of the grid
    std::uint64_t ndim = grid.ndim();
    intvec collapsed_index(ndim, UINT64_MAX);
    std::uint64_t collapsed_dim_count = 0;
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        const Vector<double> & nodes = grid.grid_vectors()[i_dim];
        for (std::uint64_t i_node = 0; i_node < nodes.size(); i_node++) {
            if (x[i_dim] == nodes[i_node]) {
                if (slices[i_dim].in_slice(i_node)) {
                    collapsed_index[i_dim] = i_node;
                    collapsed_dim_count++;
                    break;
                } else {
                    return 0.0;
                }
            }
        }
    }
    // calculate common factor for points on grid line
    long double common_factor = 1.0;
    long double product_point = 1.0;
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        const Vector<double> & nodes = grid.grid_vectors()[i_dim];
        if (collapsed_index[i_dim] != UINT64_MAX) {
            // calculate common_factor if point on grid line
            for (std::uint64_t i_node = 0; i_node < nodes.size(); i_node++) {
                if (i_node == collapsed_index[i_dim]) {
                    continue;
                }
                common_factor *= (nodes[collapsed_index[i_dim]] - nodes[i_node]);
            }
        } else {
            // calculate product of point wrt. every node
            for (std::uint64_t i_node = 0; i_node < nodes.size(); i_node++) {
                product_point *= (x[i_dim] - nodes[i_node]);
            }
        }
    }
    // calculate shape of collapsed coefficient array
    intvec non_collapse_shape(ndim - collapsed_dim_count);
    std::uint64_t non_collapse_size = 1;
    for (std::uint64_t i_dim = 0, i_non_collapse = 0; i_dim < ndim; i_dim++) {
        if (collapsed_index[i_dim] == UINT64_MAX) {
            non_collapse_shape[i_non_collapse] = coeff.shape()[i_dim];
            non_collapse_size *= non_collapse_shape[i_non_collapse];
            i_non_collapse++;
        }
    }
    // loop over each uncollapsed point of the coeff array
    long double result = 0.0;
    for (std::uint64_t i_point = 0; i_point < non_collapse_size; i_point++) {
        intvec sub_coeff_index = contiguous_to_ndim_idx(i_point, non_collapse_shape);
        // index wrt. uncollapsed coeff array
        intvec whole_coeff_index(collapsed_index);
        for (std::uint64_t i_dim = 0, i_dim_non_collapsed = 0; i_dim < ndim; i_dim++) {
            if (collapsed_index[i_dim] == UINT64_MAX) {
                whole_coeff_index[i_dim] = sub_coeff_index[i_dim_non_collapsed++];
            }
        }
        // index wrt. grid
        intvec in_grid_index(ndim);
        for (std::uint64_t i_dim = 0; i_dim < grid.ndim(); i_dim++) {
            in_grid_index[i_dim] = slices[i_dim].get_index_in_whole_array(whole_coeff_index[i_dim]);
        }
        // calculate denominator
        long double denominator = 1.0;
        for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
            if (collapsed_index[i_dim] == UINT64_MAX) {
                denominator *= x[i_dim] - grid.grid_vectors()[i_dim][in_grid_index[i_dim]];
            }
        }
        result += static_cast<long double>(coeff.get(whole_coeff_index)) / denominator;
    }
    result *= common_factor * product_point;
    return result;
}


}  // namespace merlin

// Copyright 2022 quocdang1998
#include "merlin/interpolant/newton.hpp"

#include <cinttypes>
#include <cstring>  // std::memcpy
#include <utility>  // std::move

#include <omp.h>  // pragma omp, omp_get_num_threads

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/copy.hpp"  // merlin::array::array_copy
#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/env.hpp"  // merlin::Environment
#include "merlin/logger.hpp"  // CUHDERR
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/utils.hpp"  // merlin::prod_elements, merlin::contiguous_to_ndim_idx, merlin::decrement_index
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Utils
// --------------------------------------------------------------------------------------------------------------------

// Calculate divide diference between 2 arrays
void divide_difference_cpu_parallel(const array::Array & a1, const array::Array & a2, double x1,
                                    double x2, array::Array & result) {
    long double denominator = x1 - x2;
    std::uint64_t size = a1.size();
    #pragma omp parallel for schedule(guided, Environment::parallel_chunk)
    for (std::int64_t i = 0; i < size; i++) {
        intvec index = contiguous_to_ndim_idx(i, a1.shape());
        double div_diff = (a1.get(index) - a2.get(index)) / denominator;
        result.set(i, div_diff);
    }
}

// Calculate coefficients for cartesian grid (supposed shape value == shape of coeff)
void calc_newton_coeffs_cpu_recursive(const interpolant::CartesianGrid & grid, array::Array & coeff,
                                      std::uint64_t max_dimension, merlin::Vector<array::Array> & sub_slices,
                                      std::uint64_t start_index) {
    // get associated 1D grid to calculate on
    std::uint64_t ndim = grid.ndim();
    std::uint64_t current_dim = ndim - coeff.ndim();
    if (current_dim > max_dimension) {
        return;
    }
    const Vector<double> & grid_vector = grid.grid_vectors()[current_dim];
    // trivial case (1D)
    if (coeff.ndim() == 1) {
        for (std::uint64_t i = 1; i < coeff.shape()[0]; i++) {
            for (std::uint64_t k = coeff.shape()[0]-1; k >=i; k--) {
                long double coeff_calc = (coeff.get({k}) - coeff.get({k-1})) / (grid_vector[k] - grid_vector[k-i]);
                coeff.set({k}, coeff_calc);
            }
        }
        return;
    }
    // calculate divdiff on dim i-th
    for (std::uint64_t i = 1; i < coeff.shape()[0]; i++) {
        for (std::uint64_t k = coeff.shape()[0]-1; k >= i; k--) {
            // get NdData of sub slice
            Vector<array::Slice> slice_k(coeff.ndim()), slice_k_1(coeff.ndim());
            slice_k[0] = array::Slice({k});
            slice_k_1[0] = array::Slice({k-1});
            const array::Array array_k(coeff, slice_k);
            const array::Array array_k_1(coeff, slice_k_1);
            array::Array array_result(coeff, slice_k);
            // calculate divide difference
            divide_difference_cpu_parallel(array_k, array_k_1, grid_vector[k], grid_vector[k-i], array_result);
        }
    }
    // calculate new start index jump
    intvec shape_other_dims;
    intvec total_dim = grid.get_grid_shape();
    shape_other_dims.assign(total_dim.begin()+current_dim+1, total_dim.begin()+max_dimension+1);
    std::uint64_t start_index_jump = prod_elements(shape_other_dims);
    // recursively calculate divide difference for dimension from i-1-th
    for (std::uint64_t i = 0; i < coeff.shape()[0]; i++) {
        // calculate new start index
        std::uint64_t new_start_index = start_index + i*start_index_jump;
        // get array assigned to slice
        Vector<array::Slice> slice_i(coeff.ndim());
        slice_i[0] = array::Slice({static_cast<std::uint64_t>(i)});
        array::Array array_coeff_i(coeff, slice_i);
        array_coeff_i.remove_dim(0);
        calc_newton_coeffs_cpu_recursive(grid, array_coeff_i, max_dimension, sub_slices, new_start_index);
        // push instance to vector
        if (current_dim == max_dimension) {
            // sub_slices[new_start_index] = array::Array(coeff, slice_i);
            sub_slices[new_start_index] = std::move(array_coeff_i);
        }
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Calculate coefficient
// --------------------------------------------------------------------------------------------------------------------

// Calculate coefficients for cartesian grid
void interpolant::calc_newton_coeffs_cpu(const interpolant::CartesianGrid & grid, const array::Array & value,
                                         array::Array & coeff) {
    // get associated 1D grid to calculate on
    std::uint64_t ndim = grid.ndim();
    // copy value to coeff
    if (&coeff != &value) {
        array::array_copy(&coeff, &value, std::memcpy);
    }
    // get max recursive dimension
    static std::uint64_t parallel_limit = Environment::parallel_chunk;
    intvec total_shape = grid.get_grid_shape();
    std::uint64_t cumulative_size = 1, dim_max = 0;
    while (dim_max < ndim) {
        cumulative_size *= total_shape[dim_max];
        if (cumulative_size >= parallel_limit) {
            break;
        }
        dim_max++;
    }
    // trivial case: size too small
    if (dim_max == ndim) {
        interpolant::calc_newton_coeffs_single_core(grid, coeff);
        return;
    }
    // recursive calculation
    Vector<array::Array> sub_slices = make_vector<array::Array>(cumulative_size);
    calc_newton_coeffs_cpu_recursive(grid, coeff, dim_max, sub_slices, 0);
    // parallel calculation after that
    #pragma omp parallel for schedule(guided, Environment::parallel_chunk)
    for (std::int64_t i = 0; i < sub_slices.size(); i++) {
        interpolant::calc_newton_coeffs_single_core(grid, sub_slices[i]);
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Evaluate interpolation
// --------------------------------------------------------------------------------------------------------------------

// Evaluate Newton interpolation on a full Cartesian grid using CPU (supposed shape of grid == shape of coeff)
double interpolant::eval_newton_cpu2(const interpolant::CartesianGrid & grid, const array::Array & coeff,
                                     const Vector<double> & x) {
    long double result = 0;
    std::uint64_t ndim = grid.ndim();
    const Vector<double> & grid_vector = grid.grid_vectors()[ndim - coeff.ndim()];
    // trivial case
    if (coeff.ndim() == 1) {
        const std::uint64_t & shape = coeff.shape()[0];
        result += coeff.get({shape-1});
        for (std::int64_t i = shape-2; i >= 0; i--) {
            result *= (x[ndim - coeff.ndim()] - grid_vector[i]);
            result += coeff.get({static_cast<std::uint64_t>(i)});
        }
        return result;
    }
    // recursively calculate for non-trivial case
    const std::uint64_t & shape = coeff.shape()[0];
    Vector<array::Slice> slice_i(coeff.ndim());
    slice_i[0] = array::Slice({shape-1});
    array::Array array_coeff_i(coeff, slice_i);
    array_coeff_i.remove_dim(0);
    result += interpolant::eval_newton_cpu2(grid, array_coeff_i, x);
    for (std::int64_t i = shape-2; i >= 0; i--) {
        result *= (x[ndim - coeff.ndim()] - grid_vector[i]);
        slice_i[0] = array::Slice({static_cast<std::uint64_t>(i)});
        array_coeff_i = array::Array(coeff, slice_i);
        array_coeff_i.remove_dim(0);
        result += interpolant::eval_newton_cpu2(grid, array_coeff_i, x);
    }
    return result;
}

// Evaluate Newton interpolation without recursive
double interpolant::eval_newton_cpu(const interpolant::CartesianGrid & grid, const array::Array & coeff,
                                    const Vector<double> & x) {
    // initialize storing vector
    std::uint64_t ndim = grid.ndim(), max_dim = ndim-1;
    intvec shape = grid.get_grid_shape();
    intvec begin(ndim, 0), iterator(coeff.end().index());
    Vector<double> cum(ndim, 0.f);
    decrement_index(iterator, shape);
    cum[max_dim] = coeff.get(iterator);
    // loop over each point in coeff array
    while (iterator != begin) {
        std::uint64_t i_dim = decrement_index(iterator, shape);
        if (i_dim == max_dim) {
            cum[i_dim] *= x[i_dim] - grid.grid_vectors()[i_dim][iterator[i_dim]];
            cum[i_dim] += coeff.get(iterator);
        } else {
            cum[i_dim] *= x[i_dim] - grid.grid_vectors()[i_dim][iterator[i_dim]+1];
            for (std::uint64_t i = i_dim+1; i < max_dim; i++) {
                cum[i_dim] += (x[i] - grid.grid_vectors()[i][0]) * cum[i];
                cum[i] = 0;
            }
            cum[i_dim] += cum[max_dim];
            cum[max_dim] = coeff.get(iterator);
        }
    }
    // finalize
    double result = 0.0;
    for (std::uint64_t i = 0; i < max_dim; i++) {
        result += (x[i] - grid.grid_vectors()[i][0]) * cum[i];
    }
    result += cum[max_dim];
    return result;
}

}  // namespace merlin

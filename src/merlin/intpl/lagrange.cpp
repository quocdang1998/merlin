// Copyright 2022 quocdang1998
#include "merlin/intpl/lagrange.hpp"

#include <algorithm>  // std::find
#include <cstring>    // std::memcpy

#include <omp.h>  // #pragma omp, omp_get_num_threads

#include "merlin/array/array.hpp"           // merlin::array::Array
#include "merlin/array/parcel.hpp"          // merlin::array::Parcel
#include "merlin/intpl/cartesian_grid.hpp"  // merlin::intpl::CartesianGrid
#include "merlin/intpl/sparse_grid.hpp"     // merlin::intpl::SparseGrid
#include "merlin/logger.hpp"                // FAILURE, cuda_compile_error
#include "merlin/slice.hpp"                 // merlin::Slice
#include "merlin/utils.hpp"                 // merlin::contiguous_to_ndim_idx, merlin::get_level_shape

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Calculate coefficients
// ---------------------------------------------------------------------------------------------------------------------

// Calculate Lagrange interpolation coefficient on a full Cartesian grid using CPU
void intpl::calc_lagrange_coeffs_cpu(const intpl::CartesianGrid & grid, const array::Array & value,
                                     array::Array & coeff, std::uint64_t nthreads) {
    std::uint64_t ndim = grid.ndim();
    intvec grid_shape = grid.get_grid_shape();
    // parallel loop calculation
    #pragma omp parallel for schedule(guided, 96) num_threads(nthreads)
    for (std::int64_t i = 0; i < value.size(); i++) {
        intvec index = contiguous_to_ndim_idx(i, grid_shape);
        // calculate the denomiantor (product of diferences of node values)
        long double denominator = 1.0;
        for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
            for (std::uint64_t i_node = 0; i_node < grid_shape[i_dim]; i_node++) {
                // skip for repeating index
                if (i_node == index[i_dim]) {
                    continue;
                }
                denominator *= grid.grid_vectors()[i_dim][index[i_dim]] - grid.grid_vectors()[i_dim][i_node];
            }
        }
        double result = value.get(index) / static_cast<double>(denominator);
        coeff.set(index, result);
    }
}

#ifndef __MERLIN_CUDA__

// Calculate Lagrange interpolation coefficients on a full Cartesian grid using GPU
void intpl::calc_lagrange_coeffs_gpu(const intpl::CartesianGrid & grid, const array::Parcel & value,
                                     array::Parcel & coeff, const cuda::Stream & stream, std::uint64_t n_thread) {
    FAILURE(cuda_compile_error, "Compile the package with CUDA option enabled to access this feature.\n");
}

// Call CUDA kernel calculating coefficients on GPU
void call_lagrange_coeff_kernel(const intpl::CartesianGrid * p_grid, const array::Parcel * p_value,
                                array::Parcel * p_coeff, std::uint64_t shared_mem_size, std::uintptr_t stream_ptr,
                                std::uint64_t n_thread) {}

// Evaluate Lagrange interpolation on a full Cartesian grid using GPU
Vector<double> intpl::eval_lagrange_gpu(const intpl::CartesianGrid & grid, const array::Parcel & coeff,
                                        const array::Parcel & points, const cuda::Stream & stream,
                                        std::uint64_t n_thread) {
    FAILURE(cuda_compile_error, "Compile the package with CUDA option enabled to access this feature.\n");
    return Vector<double>();
}

#endif  // __MERLIN_CUDA__

// Calculate max_level from old max_level and new_max_level
static void accumulate_level(intvec & old_level, const intvec & new_level) {
    for (std::uint64_t i_dim = 0; i_dim < old_level.size(); i_dim++) {
        old_level[i_dim] = std::max(old_level[i_dim], new_level[i_dim]);
    }
}

// Calculate Lagrange interpolation coefficient on an added Cartesian grid to Sparse grid using CPU
static void calc_lagrange_coeffs_of_added_grid_cpu(const intpl::CartesianGrid & accumulated_grid,
                                                   const intpl::CartesianGrid & grid, const array::Array & value,
                                                   array::Array & coeff) {
    // calculate cartesian interpolation
    intpl::calc_lagrange_coeffs_cpu(grid, value, coeff);
    // update coefficient by a factor
    std::uint64_t ndim = grid.ndim();
    intvec grid_shape = grid.get_grid_shape();
    for (std::uint64_t i_point = 0; i_point < value.size(); i_point++) {
        intvec index = contiguous_to_ndim_idx(i_point, grid_shape);
        Vector<double> point = grid[index];
        coeff[index] /= intpl::exclusion_grid(accumulated_grid, grid, point);
    }
}

// Calculate Lagrange interpolation evaluation on an added Cartesian grid to Sparse grid using CPU
static double eval_lagrange_of_added_grid_cpu(const intpl::CartesianGrid & accumulated_grid,
                                              const intpl::CartesianGrid & grid, const array::Array & coeff,
                                              const Vector<double> & x) {
    double result = intpl::eval_lagrange_cpu(grid, coeff, x);
    double factor = intpl::exclusion_grid(accumulated_grid, grid, x);
    return result * factor;
}

// Calculate Lagrange interpolation coefficients on a sparse grid using CPU (function value are preprocessed)
void intpl::calc_lagrange_coeffs_cpu(const intpl::SparseGrid & grid, const array::Array & value, array::Array & coeff) {
    // copy value to coeff
    if (&value != &coeff) {
        intpl::copy_value_from_cartesian_array(coeff, value, grid);
    }
    // Initialize
    std::uint64_t num_subgrid = grid.num_level();
    intpl::CartesianGrid accumulated_cart_grid(grid.ndim());
    for (std::uint64_t i_subgrid = 0; i_subgrid < num_subgrid; i_subgrid++) {
        // get hiearchical level
        const intvec level_index = grid.level_index(i_subgrid);
        // calculate coefficient at current grid level
        intpl::CartesianGrid level_cartgrid = intpl::get_cartesian_grid(grid, i_subgrid);
        accumulated_cart_grid += level_cartgrid;
        intvec level_shape = get_level_shape(level_index);
        Slice level_slice(grid.sub_grid_start_index()[i_subgrid], grid.sub_grid_start_index()[i_subgrid + 1]);
        array::Array level_coeff(coeff, {level_slice});
        level_coeff.reshape(level_shape);
        calc_lagrange_coeffs_of_added_grid_cpu(accumulated_cart_grid, level_cartgrid, level_coeff, level_coeff);
        // subtract other points of the grid
        for (std::uint64_t j_subgrid = i_subgrid + 1; j_subgrid < num_subgrid; j_subgrid++) {
            std::uint64_t start_index = grid.sub_grid_start_index()[j_subgrid];
            intpl::CartesianGrid level_j_cartgrid = intpl::get_cartesian_grid(grid, j_subgrid);
            std::uint64_t level_j_cartgrid_size = level_j_cartgrid.size();
            for (std::uint64_t i_point = 0; i_point < level_j_cartgrid_size; i_point++) {
                Vector<double> point = level_j_cartgrid[i_point];
                std::uint64_t i_point_sparsegrid = start_index + i_point;
                coeff[{i_point_sparsegrid}] -=
                    eval_lagrange_of_added_grid_cpu(accumulated_cart_grid, level_cartgrid, level_coeff, point);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Evaluate interpolation
// ---------------------------------------------------------------------------------------------------------------------

// Evaluate Lagrange interpolation on a cartesian grid using CPU
double intpl::eval_lagrange_cpu(const intpl::CartesianGrid & grid, const array::Array & coeff,
                                const Vector<double> & x) {
    return intpl::eval_lagrange_single_core(grid, coeff, x);
}

// Evaluate Lagrange interpolation on a sparse grid using CPU (function value are preprocessed)
double intpl::eval_lagrange_cpu(const intpl::SparseGrid & grid, const array::Array & coeff, const Vector<double> & x) {
    // Initialize
    long double result = 0.0;
    std::uint64_t num_subgrid = grid.num_level();
    intpl::CartesianGrid accumulated_cart_grid(grid.ndim());
    for (std::uint64_t i_subgrid = 0; i_subgrid < num_subgrid; i_subgrid++) {
        // get hiearchical level
        const intvec level_index = grid.level_index(i_subgrid);
        // calculate coefficient at current grid level
        intpl::CartesianGrid level_cartgrid = intpl::get_cartesian_grid(grid, i_subgrid);
        accumulated_cart_grid += level_cartgrid;
        intvec level_shape = get_level_shape(level_index);
        Slice level_slice(grid.sub_grid_start_index()[i_subgrid], grid.sub_grid_start_index()[i_subgrid + 1]);
        array::Array level_coeff(coeff, {level_slice});
        level_coeff.reshape(level_shape);
        result += eval_lagrange_of_added_grid_cpu(accumulated_cart_grid, level_cartgrid, level_coeff, x);
    }
    return result;
}

}  // namespace merlin

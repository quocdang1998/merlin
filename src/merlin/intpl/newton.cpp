// Copyright 2022 quocdang1998
#include "merlin/intpl/newton.hpp"

#include <cinttypes>
#include <cstring>  // std::memcpy
#include <utility>  // std::move

#include <omp.h>  // pragma omp, omp_get_num_threads

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/copy.hpp"  // merlin::array::array_copy
#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/logger.hpp"  // CUHDERR
#include "merlin/intpl/cartesian_grid.hpp"  // merlin::intpl::CartesianGrid
#include "merlin/intpl/sparse_grid.hpp"  // merlin::intpl::SparseGrid
#include "merlin/utils.hpp"  // merlin::prod_elements, merlin::contiguous_to_ndim_idx
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Utils
// --------------------------------------------------------------------------------------------------------------------

// Calculate divide diference between 2 arrays
static void divide_difference_cpu_parallel(const array::Array & a1, const array::Array & a2, double x1, double x2,
                                           array::Array & result, std::uint64_t nthreads) {
    long double denominator = x1 - x2;
    std::uint64_t size = a1.size();
    #pragma omp parallel for schedule(guided, Environment::parallel_chunk) num_threads(nthreads)
    for (std::int64_t i = 0; i < size; i++) {
        intvec index = contiguous_to_ndim_idx(i, a1.shape());
        double div_diff = (a1.get(index) - a2.get(index)) / denominator;
        result.set(i, div_diff);
    }
}

// Calculate coefficients for cartesian grid (supposed shape value == shape of coeff)
static void calc_newton_coeffs_cpu_recursive(const intpl::CartesianGrid & grid, array::Array & coeff,
                                             std::uint64_t max_dimension, merlin::Vector<array::Array> & sub_slices,
                                             std::uint64_t start_index, std::uint64_t nthreads) {
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
            divide_difference_cpu_parallel(array_k, array_k_1, grid_vector[k], grid_vector[k-i],
                                           array_result, nthreads);
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
        calc_newton_coeffs_cpu_recursive(grid, array_coeff_i, max_dimension, sub_slices, new_start_index, nthreads);
        // push instance to vector
        if (current_dim == max_dimension) {
            // sub_slices[new_start_index] = array::Array(coeff, slice_i);
            sub_slices[new_start_index] = std::move(array_coeff_i);
        }
    }
}

// Concatenate 3 intvec
static intvec merge_3vectors(const intvec & v1, std::uint64_t v2, const intvec & v3) {
    intvec result(v1.size() + 1 + v3.size());
    for (std::uint64_t i = 0; i < v1.size(); i++) {
        result[i] = v1[i];
    }
    result[v1.size()] = v2;
    for (std::uint64_t i = 0; i < v3.size(); i++) {
        result[1+v1.size()+i] = v3[i];
    }
    return result;
}

// Calculate Newton coefficients by a signle CPU or GPU core
static void calc_newton_coeffs_single_core(const intpl::CartesianGrid & grid, array::Array & coeff) {
    // loop on each dimension
    for (std::uint64_t i_dim = 0; i_dim < coeff.ndim(); i_dim++) {
        // get grid vector at current diemnsion
        const Vector<double> & grid_vector = grid.grid_vectors()[grid.ndim() - coeff.ndim() + i_dim];
        // get shape and size of previous dimensions
        intvec shape_previous_dims;
        shape_previous_dims.assign(const_cast<std::uint64_t *>(coeff.shape().begin()), i_dim);
        std::uint64_t size_previous_dims = prod_elements(shape_previous_dims);
        // get shape and size of divdiff subspace
        intvec shape_divdiff_space;
        shape_divdiff_space.assign(const_cast<std::uint64_t *>(coeff.shape().begin())+i_dim+1,
                                   const_cast<std::uint64_t *>(coeff.shape().end()));
        std::uint64_t size_divdiff_space = prod_elements(shape_divdiff_space);
        // loop on each previous dims point
        for (std::int64_t i_previous_dims = 0; i_previous_dims < size_previous_dims; i_previous_dims++) {
            intvec index_previous_dims = contiguous_to_ndim_idx(i_previous_dims, shape_previous_dims);
            // loop on indices of current dim for divide difference
            for (std::uint64_t i = 1; i < coeff.shape()[i_dim]; i++) {
                for (std::uint64_t k = coeff.shape()[i_dim]-1; k >= i; k--) {
                    // loop on each point in divdiff space
                    for (std::uint64_t i_divdiff_space = 0; i_divdiff_space < size_divdiff_space; i_divdiff_space++) {
                        intvec index_divdiff_space = contiguous_to_ndim_idx(i_divdiff_space, shape_divdiff_space);
                        intvec point_index_k = merge_3vectors(index_previous_dims, k, index_divdiff_space);
                        intvec point_index_k_1 = merge_3vectors(index_previous_dims, k-1, index_divdiff_space);
                        double divdiff_result = (coeff[point_index_k] - coeff[point_index_k_1]);
                        divdiff_result /= grid_vector[k] - grid_vector[k-i];
                        intvec point_index_result = std::move(point_index_k);
                        coeff[point_index_result] = divdiff_result;
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Calculate coefficient
// --------------------------------------------------------------------------------------------------------------------

// Calculate coefficients for cartesian grid
void intpl::calc_newton_coeffs_cpu(const intpl::CartesianGrid & grid, const array::Array & value, array::Array & coeff,
                                   std::uint64_t nthreads) {
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
        calc_newton_coeffs_single_core(grid, coeff);
        return;
    }
    // recursive calculation
    Vector<array::Array> sub_slices = make_vector<array::Array>(cumulative_size);
    calc_newton_coeffs_cpu_recursive(grid, coeff, dim_max, sub_slices, 0, nthreads);
    // parallel calculation after that
    #pragma omp parallel for schedule(guided, Environment::parallel_chunk) num_threads(nthreads)
    for (std::int64_t i = 0; i < sub_slices.size(); i++) {
        calc_newton_coeffs_single_core(grid, sub_slices[i]);
    }
}

// Calculate Newton interpolation coefficient on an added Cartesian grid to Sparse grid using CPU
static void calc_newton_coeffs_of_added_grid_cpu(const intpl::CartesianGrid & accumulated_grid,
                                                 const intpl::CartesianGrid & grid, const array::Array & value,
                                                 array::Array & coeff) {
    // copy value to coeff
    if (&coeff != &value) {
        array::array_copy(&coeff, &value, std::memcpy);
    }
    // update coefficient by a factor
    std::uint64_t ndim = grid.ndim();
    intvec grid_shape = grid.get_grid_shape();
    for (std::uint64_t i_point = 0; i_point < value.size(); i_point++) {
        intvec index = contiguous_to_ndim_idx(i_point, grid_shape);
        Vector<double> point = grid[index];
        coeff[index] /= intpl::exclusion_grid(accumulated_grid, grid, point);
    }
    // calculate cartesian interpolation
    intpl::calc_newton_coeffs_cpu(grid, coeff, coeff);
}

// Calculate Newton interpolation evaluation on an added Cartesian grid to Sparse grid using CPU
static double eval_newton_of_added_grid_cpu(const intpl::CartesianGrid & accumulated_grid,
                                            const intpl::CartesianGrid & grid, const array::Array & coeff,
                                            const Vector<double> & x) {
    double result = intpl::eval_newton_cpu(grid, coeff, x);
    double factor = intpl::exclusion_grid(accumulated_grid, grid, x);
    return result*factor;
}

// Calculate Newton interpolation coefficients on a sparse grid using CPU (function value are preprocessed)
void intpl::calc_newton_coeffs_cpu(const intpl::SparseGrid & grid, const array::Array & value, array::Array & coeff) {
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
        array::Slice level_slice(grid.sub_grid_start_index()[i_subgrid], grid.sub_grid_start_index()[i_subgrid+1]);
        array::Array level_coeff(coeff, {level_slice});
        level_coeff.reshape(level_shape);
        calc_newton_coeffs_of_added_grid_cpu(accumulated_cart_grid, level_cartgrid, level_coeff, level_coeff);
        // subtract other points of the grid
        for (std::uint64_t j_subgrid = i_subgrid+1; j_subgrid < num_subgrid; j_subgrid++) {
            std::uint64_t start_index = grid.sub_grid_start_index()[j_subgrid];
            intpl::CartesianGrid level_j_cartgrid = intpl::get_cartesian_grid(grid, j_subgrid);
            std::uint64_t level_j_cartgrid_size = level_j_cartgrid.size();
            for (std::uint64_t i_point = 0; i_point < level_j_cartgrid_size; i_point++) {
                Vector<double> point = level_j_cartgrid[i_point];
                std::uint64_t i_point_sparsegrid = start_index + i_point;
                coeff[{i_point_sparsegrid}] -= eval_newton_of_added_grid_cpu(accumulated_cart_grid, level_cartgrid,
                                                                             level_coeff, point);
            }
        }
    }
}

#ifndef __MERLIN_CUDA__

// Call CUDA kernel calculating coefficients on GPU
void call_newton_coeff_kernel(const intpl::CartesianGrid * p_grid, const array::Parcel * p_value,
                              array::Parcel * p_coeff, std::uint64_t shared_mem_size, std::uintptr_t stream_ptr,
                              std::uint64_t n_thread) {}

// Calculate Newton interpolation coefficients on a full Cartesian grid using GPU
void intpl::calc_newton_coeffs_gpu(const intpl::CartesianGrid & grid, const array::Parcel & value,
                                   array::Parcel & coeff, const cuda::Stream & stream, std::uint64_t n_thread) {
    FAILURE(cuda_compile_error, "Compile the package with CUDA option enabled to access this feature.\n");
}

// Evaluate Newton interpolation on a full Cartesian grid using GPU
Vector<double> intpl::eval_newton_gpu(const intpl::CartesianGrid & grid, const array::Parcel & coeff,
                                      const array::Parcel & points, const cuda::Stream & stream,
                                      std::uint64_t n_thread) {
    FAILURE(cuda_compile_error, "Compile the package with CUDA option enabled to access this feature.\n");
    return Vector<double>();
}

#endif  // __MERLIN_CUDA__

// --------------------------------------------------------------------------------------------------------------------
// Evaluate interpolation
// --------------------------------------------------------------------------------------------------------------------

// Evaluate Newton interpolation on a cartesian grid using CPU
double intpl::eval_newton_cpu(const intpl::CartesianGrid & grid, const array::Array & coeff,
                              const Vector<double> & x) {
    return intpl::eval_newton_single_core(grid, coeff, x);
}

// Evaluate Newton interpolation on a sparse grid using CPU (function value are preprocessed)
double intpl::eval_newton_cpu(const intpl::SparseGrid & grid, const array::Array & coeff, const Vector<double> & x) {
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
        array::Slice level_slice(grid.sub_grid_start_index()[i_subgrid], grid.sub_grid_start_index()[i_subgrid+1]);
        array::Array level_coeff(coeff, {level_slice});
        level_coeff.reshape(level_shape);
        result += eval_newton_of_added_grid_cpu(accumulated_cart_grid, level_cartgrid, level_coeff, x);
    }
    return result;
}

}  // namespace merlin

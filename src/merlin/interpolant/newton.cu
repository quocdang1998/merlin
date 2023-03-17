// Copyright 2022 quocdang1998
#include "merlin/interpolant/newton.hpp"

#include <functional>  // std::bind, std::placeholders
#include <utility> // std::move

#include "merlin/array/copy.hpp"  // merlin::array::array_copy
#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/array/slice.hpp" // merlin::array::Slice
#include "merlin/cuda/memory.hpp"  // merlin::cuda::Memory
#include "merlin/env.hpp"  // merlin::Environment
#include "merlin/utils.hpp"  // merlin::prod_elements
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Utils
// --------------------------------------------------------------------------------------------------------------------

// Calculate parallelized divide difference on GPU
void calc_divdiff_gpu(const array::Parcel & a1, const array::Parcel & a2, double x1, double x2,
                      array::Parcel & result, const cuda::Stream & stream) {
    // check for validity
    stream.check_cuda_context();
    // copy data to GPU
    cuda::Memory mem(stream.get_stream_ptr(), a1, a2, result);
    array::Parcel * ptr_a1_on_gpu = const_cast<array::Parcel *>(mem.get<0>());
    array::Parcel * ptr_a2_on_gpu = const_cast<array::Parcel *>(mem.get<1>());
    array::Parcel * ptr_result_on_gpu = const_cast<array::Parcel *>(mem.get<2>());
    std::uint64_t total_malloc_size = mem.get_total_malloc_size();
    // call divide difference algorithm on GPU
    std::uint64_t size = a1.size();
    interpolant::call_divdiff_kernel(ptr_a1_on_gpu, ptr_a2_on_gpu, x1, x2, ptr_result_on_gpu, size,
                                     total_malloc_size, stream.get_stream_ptr());
}

// Calculate coefficients for cartesian grid (supposed shape value == shape of coeff)
void calc_newton_coeffs_gpu_recursive(const interpolant::CartesianGrid & grid, array::Parcel & coeff,
                                      std::uint64_t max_dimension, merlin::Vector<array::Parcel> & sub_slices,
                                      std::uint64_t start_index, const cuda::Stream & stream) {
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
            const array::Parcel array_k(coeff, slice_k);
            const array::Parcel array_k_1(coeff, slice_k_1);
            array::Parcel array_result(coeff, slice_k);
            // calculate divide difference
            calc_divdiff_gpu(array_k, array_k_1, grid_vector[k], grid_vector[k-i], array_result, stream);
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
        array::Parcel array_coeff_i(coeff, slice_i);
        array_coeff_i.remove_dim(0);
        calc_newton_coeffs_gpu_recursive(grid, array_coeff_i, max_dimension, sub_slices, new_start_index, stream);
        // push instance to vector
        if (current_dim == max_dimension) {
            sub_slices[new_start_index] = array::Parcel(coeff, slice_i);
        }
    }
}

void calc_newton_coeff_single_core_gpu(const interpolant::CartesianGrid & grid, array::Parcel * p_coeff,
                                       std::uint64_t size, const cuda::Stream & stream) {
    // check for validity
    stream.check_cuda_context();
    // copy grid and coeff to GPU
    void * gpu_memory;
    std::uint64_t coeff_size = p_coeff->malloc_size();
    ::cudaError_t err_ = ::cudaMalloc(&gpu_memory, grid.malloc_size() + size * coeff_size);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Alloc data faile with message \"%s\"\n", ::cudaGetErrorString(err_));
    }
    interpolant::CartesianGrid * grid_gpu = reinterpret_cast<interpolant::CartesianGrid *>(gpu_memory);
    void * coeff_gpu_data = grid.copy_to_gpu(grid_gpu, grid_gpu+1, stream.get_stream_ptr());
    std::uintptr_t coeff_i_destination_ptr = reinterpret_cast<std::uintptr_t>(coeff_gpu_data);
    for (std::uint64_t i_coeff = 0; i_coeff < size; i_coeff++) {
        array::Parcel * p_coeff_dest = reinterpret_cast<array::Parcel *>(coeff_i_destination_ptr);
        p_coeff[i_coeff].copy_to_gpu(p_coeff_dest, p_coeff_dest+1);
        coeff_i_destination_ptr += coeff_size;
    }
    // call kernel
    std::uint64_t shared_mem_size = grid.malloc_size() + Environment::default_block_size * coeff_size;
    interpolant::call_single_core_kernel(grid_gpu, reinterpret_cast<array::Parcel *>(coeff_gpu_data), size,
                                         shared_mem_size, stream.get_stream_ptr());
    // deallocate data
    ::cudaFree(gpu_memory);
}

// Calculate Lagrange interpolation coefficients on a full Cartesian grid using GPU
void interpolant::calc_newton_coeffs_gpu(const interpolant::CartesianGrid & grid, const array::Parcel & value,
                                         array::Parcel & coeff, const cuda::Stream & stream) {
    // check for validity
    stream.check_cuda_context();
    // get associated 1D grid to calculate on
    std::uint64_t ndim = grid.ndim();
    // copy value to coeff
    if (&coeff != &value) {
        ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
        if (coeff.device() != value.device()) {
            auto copy_func = std::bind(::cudaMemcpyPeerAsync, std::placeholders::_1, coeff.device().id(),
                                       std::placeholders::_2, value.device().id(), std::placeholders::_3, cuda_stream);
            array::array_copy(&coeff, &value, copy_func);
        } else {
            auto copy_func = std::bind(::cudaMemcpyAsync, std::placeholders::_1, std::placeholders::_2,
                                    std::placeholders::_3, ::cudaMemcpyDeviceToDevice, cuda_stream);
            array::array_copy(&coeff, &value, copy_func);
        }
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
        calc_newton_coeff_single_core_gpu(grid, &coeff, 1, stream);
        return;
    }
    // recursive calculation
    Vector<array::Parcel> sub_slices = make_vector<array::Parcel>(cumulative_size);
    calc_newton_coeffs_gpu_recursive(grid, coeff, dim_max, sub_slices, 0, stream);
    // parallel calculation after that
    calc_newton_coeff_single_core_gpu(grid, sub_slices.data(), sub_slices.size(), stream);
}

}  // namespace merlin

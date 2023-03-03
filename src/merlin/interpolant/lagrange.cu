// Copyright 2022 quocdang1998
#include "merlin/interpolant/lagrange.hpp"

#include <cinttypes>
#include <cstdio>

#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/env.hpp"  // merlin::Environment
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/utils.hpp"  // merlin::flatten_kernel_index, merlin::get_block_count, merlin::contiguous_to_ndim_idx

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// CUDA kernel for calculating Lagrange coefficients
// --------------------------------------------------------------------------------------------------------------------
/*
// Calculate interpolation on a given index
__cudevice__ static void calc_lagrange_coefficients(std::uint64_t c_index, const interpolant::CartesianGrid * p_grid,
                                                    const array::Parcel * p_value, array::Parcel * p_coeff) {
    // get array index from contiguous index
    std::uint64_t ndim = p_grid->ndim();
    intvec grid_shape = p_grid->get_grid_shape();
    intvec index = contiguous_to_ndim_idx(c_index, grid_shape);
     // calculate the denomiantor (product of diferences of node values)
    double denominator = 1.0;
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        for (std::uint64_t i_node = 0; i_node < grid_shape[i_dim]; i_node++) {
            // skip for repeating index
            if (i_node == index[i_dim]) {
                continue;
            }
            denominator *= p_grid->grid_vectors()[i_dim][index[i_dim]] - p_grid->grid_vectors()[i_dim][i_node];
        }
    }
    double result = p_value->operator[](index) / static_cast<double>(denominator);
    p_coeff->operator[](index) = result;
}
*/
// Main kernel
__global__ void calc_lagrange_kernel(const interpolant::CartesianGrid * p_grid, const array::Parcel * p_value,
                                            array::Parcel * p_coeff, std::uint64_t size) {
    // copy meta data to shared memory
    extern __shared__ char share_ptr[];
    std::printf("Okay.\n");
/*
    interpolant::CartesianGrid * p_grid_shared = reinterpret_cast<interpolant::CartesianGrid *>(share_ptr);
    void * current_mem_ptr = p_grid->copy_to_shared_mem(p_grid_shared, p_grid_shared+1);
    array::Parcel * p_value_shared = reinterpret_cast<array::Parcel *>(current_mem_ptr);
    current_mem_ptr = p_value->copy_to_shared_mem(p_value_shared, p_value_shared+1);
    array::Parcel * p_coeff_shared = reinterpret_cast<array::Parcel *>(current_mem_ptr);
    p_coeff->copy_to_shared_mem(p_coeff_shared, p_coeff_shared+1);
    // perform the calculation
    std::uint64_t c_index = flatten_kernel_index();
    std::printf("c_index = %" PRIu64 "\n", c_index);
    if (c_index < size) {
        calc_lagrange_coefficients(c_index, p_grid_shared, p_value_shared, p_coeff_shared);
    }
*/
}

// Calculate Lagrange interpolation coefficients on a full Cartesian grid using GPU
void interpolant::calc_lagrange_coeffs_gpu(const interpolant::CartesianGrid & grid, const array::Parcel & value,
                                           array::Parcel & coeff, const cuda::Stream & stream) {
    // check for validity
    stream.check_cuda_context();
    // copy data to GPU
    std::uint64_t grid_malloc_size = grid.malloc_size();
    std::uint64_t coeff_malloc_size = coeff.malloc_size();
    std::uint64_t total_malloc_size = grid_malloc_size + 2*coeff_malloc_size;
    void * ptr_meta_data;
    ::cudaError_t err_ = ::cudaMalloc(&ptr_meta_data, total_malloc_size);
    interpolant::CartesianGrid * ptr_grid_on_gpu = reinterpret_cast<interpolant::CartesianGrid *>(ptr_meta_data);
    void * ptr_current_meta_data = grid.copy_to_gpu(ptr_grid_on_gpu, ptr_grid_on_gpu+1);
    array::Parcel * ptr_value_on_gpu = reinterpret_cast<array::Parcel *>(ptr_current_meta_data);
    ptr_current_meta_data = value.copy_to_gpu(ptr_value_on_gpu, ptr_value_on_gpu+1);
    array::Parcel * ptr_coeff_on_gpu = reinterpret_cast<array::Parcel *>(ptr_current_meta_data);
    coeff.copy_to_gpu(ptr_coeff_on_gpu, ptr_coeff_on_gpu+1);
    // call kernel
    std::uint64_t size = value.size();
    std::uint64_t block_count = get_block_count(Environment::default_block_size, size);
    ::cudaStream_t stream_ptr = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    calc_lagrange_kernel<<<block_count, Environment::default_block_size,
                           total_malloc_size, stream_ptr>>>(ptr_grid_on_gpu, ptr_value_on_gpu, ptr_coeff_on_gpu, size);
    stream.synchronize();
    // release memory
    ::cudaFree(ptr_meta_data);
}

}  // namespace merlin

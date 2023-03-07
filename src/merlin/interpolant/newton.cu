// Copyright 2022 quocdang1998
#include "merlin/interpolant/newton.hpp"

#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Utils
// --------------------------------------------------------------------------------------------------------------------

/** @brief Calculate parallelized divide difference on GPU.*/
void interpolant::divide_difference_gpu_parallel(const array::Parcel & a1, const array::Parcel & a2, double x1,
                                                 double x2, array::Parcel & result, const cuda::Stream & stream) {
    // check for validity
    stream.check_cuda_context();
    // copy data to GPU
    std::uint64_t array_malloc_size = a1.malloc_size();
    std::uint64_t total_malloc_size = 3*array_malloc_size;
    void * ptr_meta_data;
    ::cudaError_t err_ = ::cudaMalloc(&ptr_meta_data, total_malloc_size);
    array::Parcel * ptr_a1_on_gpu = reinterpret_cast<array::Parcel *>(ptr_meta_data);
    void * ptr_current_meta_data = a1.copy_to_gpu(ptr_a1_on_gpu, ptr_a1_on_gpu+1);
    array::Parcel * ptr_a2_on_gpu = reinterpret_cast<array::Parcel *>(ptr_current_meta_data);
    ptr_current_meta_data = a2.copy_to_gpu(ptr_a2_on_gpu, ptr_a2_on_gpu+1);
    array::Parcel * ptr_result_on_gpu = reinterpret_cast<array::Parcel *>(ptr_current_meta_data);
    result.copy_to_gpu(ptr_result_on_gpu, ptr_result_on_gpu+1);
    // call divide difference algorithm on GPU
    std::uint64_t size = a1.size();
    interpolant::call_divdiff_kernel(ptr_a1_on_gpu, ptr_a2_on_gpu, x1, x2, ptr_result_on_gpu, size,
                                     total_malloc_size, stream.get_stream_ptr());
    stream.synchronize();
    // release memory
    ::cudaFree(ptr_meta_data);
}







// Calculate Lagrange interpolation coefficients on a full Cartesian grid using GPU
void interpolant::calc_newton_coeffs_gpu(const interpolant::CartesianGrid & grid, const array::Parcel & value,
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

    // release memory
    ::cudaFree(ptr_meta_data);
}

}  // namespace merlin

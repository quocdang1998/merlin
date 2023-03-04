// Copyright 2022 quocdang1998
#include "merlin/interpolant/lagrange.hpp"

#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid

namespace merlin {

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
    interpolant::call_lagrange_coeff_kernel(ptr_grid_on_gpu, ptr_value_on_gpu, ptr_coeff_on_gpu, size,
                                            total_malloc_size, stream.get_stream_ptr());
    // release memory
    ::cudaFree(ptr_meta_data);
}

}  // namespace merlin

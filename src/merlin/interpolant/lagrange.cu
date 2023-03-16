// Copyright 2022 quocdang1998
#include "merlin/interpolant/lagrange.hpp"

#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/cuda/memory.hpp"  // merlin::cuda::Memory
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid

namespace merlin {

// Calculate Lagrange interpolation coefficients on a full Cartesian grid using GPU
void interpolant::calc_lagrange_coeffs_gpu(const interpolant::CartesianGrid & grid, const array::Parcel & value,
                                           array::Parcel & coeff, const cuda::Stream & stream) {
    // check for validity
    stream.check_cuda_context();
    // copy data to GPU
    cuda::Memory mem(stream.get_stream_ptr(), grid, value, coeff);
    interpolant::CartesianGrid * ptr_grid_on_gpu = const_cast<interpolant::CartesianGrid *>(mem.get<0>());
    array::Parcel * ptr_value_on_gpu = const_cast<array::Parcel *>(mem.get<1>());
    array::Parcel * ptr_coeff_on_gpu = const_cast<array::Parcel *>(mem.get<2>());
    std::uint64_t total_malloc_size = mem.get_total_malloc_size();
    // call kernel
    std::uint64_t size = value.size();
    interpolant::call_lagrange_coeff_kernel(ptr_grid_on_gpu, ptr_value_on_gpu, ptr_coeff_on_gpu, size,
                                            total_malloc_size, stream.get_stream_ptr());
}

}  // namespace merlin

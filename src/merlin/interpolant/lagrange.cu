// Copyright 2022 quocdang1998
#include "merlin/interpolant/lagrange.hpp"

#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/cuda/memory.hpp"  // merlin::cuda::Memory
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Calculate coefficients
// --------------------------------------------------------------------------------------------------------------------

// Calculate Lagrange interpolation coefficients on a full Cartesian grid using GPU
void interpolant::calc_lagrange_coeffs_gpu(const interpolant::CartesianGrid & grid, const array::Parcel & value,
                                           array::Parcel & coeff, const cuda::Stream & stream,
                                           std::uint64_t n_thread) {
    // check for validity
    stream.check_cuda_context();
    // copy data to GPU
    cuda::Memory mem(stream.get_stream_ptr(), grid, value, coeff);
    mem.defer_allocation();
    interpolant::CartesianGrid * ptr_grid_on_gpu = mem.get<0>();
    array::Parcel * ptr_value_on_gpu = mem.get<1>();
    array::Parcel * ptr_coeff_on_gpu = mem.get<2>();
    std::uint64_t shared_mem_size = mem.get_total_malloc_size() + n_thread * grid.ndim() * sizeof(std::uint64_t);
    // call kernel
    interpolant::call_lagrange_coeff_kernel(ptr_grid_on_gpu, ptr_value_on_gpu, ptr_coeff_on_gpu, shared_mem_size,
                                            stream.get_stream_ptr(), n_thread);
}

}  // namespace merlin

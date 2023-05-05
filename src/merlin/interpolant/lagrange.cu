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
    interpolant::CartesianGrid * ptr_grid_on_gpu = mem.get<0>();
    array::Parcel * ptr_value_on_gpu = mem.get<1>();
    array::Parcel * ptr_coeff_on_gpu = mem.get<2>();
    std::uint64_t shared_mem_size = mem.get_total_malloc_size() + n_thread * grid.ndim() * sizeof(std::uint64_t);
    // call kernel
    interpolant::call_lagrange_coeff_kernel(ptr_grid_on_gpu, ptr_value_on_gpu, ptr_coeff_on_gpu, shared_mem_size,
                                            stream.get_stream_ptr(), n_thread);
}

// Evaluate Lagrange interpolation on a full Cartesian grid using GPU
Vector<double> interpolant::eval_lagrange_gpu(const interpolant::CartesianGrid & grid, const array::Parcel & coeff,
                                              const array::Parcel & points, const cuda::Stream & stream,
                                              std::uint64_t n_thread) {
    // check for validity
    stream.check_cuda_context();
    // copy data to GPU
    Vector<double> result(points.shape()[0]);
    cuda::Memory mem(stream.get_stream_ptr(), grid, coeff, points, result);
    interpolant::CartesianGrid * ptr_grid_on_gpu = mem.get<0>();
    array::Parcel * ptr_coeff_on_gpu = mem.get<1>();
    array::Parcel * ptr_points_on_gpu = mem.get<2>();
    Vector<double> * ptr_result_on_gpu = mem.get<3>();
    std::uint64_t shared_mem_size = mem.get_total_malloc_size() + n_thread * grid.ndim() * 2 * sizeof(std::uint64_t);
    // call kernel
    interpolant::call_lagrange_eval_kernel(ptr_grid_on_gpu, ptr_coeff_on_gpu, ptr_points_on_gpu, ptr_result_on_gpu,
                                           shared_mem_size, stream.get_stream_ptr(), n_thread);
    // copy result back to CPU
    result.copy_from_gpu(reinterpret_cast<double *>(ptr_result_on_gpu+1), stream.get_stream_ptr());
    return result;
}

}  // namespace merlin

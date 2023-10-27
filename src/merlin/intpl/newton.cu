// Copyright 2022 quocdang1998
#include "merlin/intpl/newton.hpp"

#include <functional>  // std::bind, std::placeholders
#include <utility>     // std::move, std::make_pair

#include "merlin/array/operation.hpp"       // merlin::array::copy
#include "merlin/array/parcel.hpp"          // merlin::array::Parcel
#include "merlin/cuda/memory.hpp"           // merlin::cuda::Memory
#include "merlin/env.hpp"                   // merlin::Environment
#include "merlin/intpl/cartesian_grid.hpp"  // merlin::intpl::CartesianGrid
#include "merlin/slice.hpp"                 // merlin::Slice
#include "merlin/utils.hpp"                 // merlin::prod_elements

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Calculate coefficients
// ---------------------------------------------------------------------------------------------------------------------

// Calculate Newton interpolation coefficients on a full Cartesian grid using GPU
void intpl::calc_newton_coeffs_gpu(const intpl::CartesianGrid & grid, const array::Parcel & value,
                                   array::Parcel & coeff, const cuda::Stream & stream, std::uint64_t n_thread) {
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
            array::copy(&coeff, &value, copy_func);
        } else {
            auto copy_func = std::bind(::cudaMemcpyAsync, std::placeholders::_1, std::placeholders::_2,
                                       std::placeholders::_3, ::cudaMemcpyDeviceToDevice, cuda_stream);
            array::copy(&coeff, &value, copy_func);
        }
    }
    // copy data to GPU
    cuda::Memory mem(stream.get_stream_ptr(), grid, coeff);
    intpl::CartesianGrid * ptr_grid_on_gpu = mem.get<0>();
    array::Parcel * ptr_coeff_on_gpu = mem.get<1>();
    std::uint64_t total_malloc_size = mem.get_total_malloc_size() + n_thread * grid.ndim() * sizeof(std::uint64_t);
    // call calculation kernel
    intpl::call_newton_coeff_kernel(ptr_grid_on_gpu, ptr_coeff_on_gpu, total_malloc_size, stream.get_stream_ptr(),
                                    n_thread);
}

// Evaluate Newton interpolation on a full Cartesian grid using GPU
Vector<double> intpl::eval_newton_gpu(const intpl::CartesianGrid & grid, const array::Parcel & coeff,
                                      const array::Parcel & points, const cuda::Stream & stream,
                                      std::uint64_t n_thread) {
    // check for validity
    stream.check_cuda_context();
    // copy data to GPU
    Vector<double> result(points.shape()[0]);
    cuda::Memory mem(stream.get_stream_ptr(), grid, coeff, points, result);
    intpl::CartesianGrid * ptr_grid_on_gpu = mem.get<0>();
    array::Parcel * ptr_coeff_on_gpu = mem.get<1>();
    array::Parcel * ptr_points_on_gpu = mem.get<2>();
    Vector<double> * ptr_result_on_gpu = mem.get<3>();
    std::uint64_t additional_shared_mem = n_thread * grid.ndim() * (sizeof(std::uint64_t) + sizeof(double));
    std::uint64_t shared_mem_size = mem.get_total_malloc_size() + additional_shared_mem;
    // call kernel
    intpl::call_newton_eval_kernel(ptr_grid_on_gpu, ptr_coeff_on_gpu, ptr_points_on_gpu, ptr_result_on_gpu,
                                   shared_mem_size, stream.get_stream_ptr(), n_thread);
    // copy result back to CPU
    result.copy_from_gpu(ptr_result_on_gpu, stream.get_stream_ptr());
    return result;
}

}  // namespace merlin

// Copyright 2022 quocdang1998
#include "merlin/interpolant/newton.hpp"

#include <functional>  // std::bind, std::placeholders
#include <utility> // std::move, std::make_pair

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

// Calculate Lagrange interpolation coefficients on a full Cartesian grid using GPU
void interpolant::calc_newton_coeffs_gpu(const interpolant::CartesianGrid & grid, const array::Parcel & value,
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
            array::array_copy(&coeff, &value, copy_func);
        } else {
            auto copy_func = std::bind(::cudaMemcpyAsync, std::placeholders::_1, std::placeholders::_2,
                                    std::placeholders::_3, ::cudaMemcpyDeviceToDevice, cuda_stream);
            array::array_copy(&coeff, &value, copy_func);
        }
    }
    // copy data to GPU
    cuda::Memory mem(stream.get_stream_ptr(), grid, coeff);
    mem.defer_allocation();
    interpolant::CartesianGrid * ptr_grid_on_gpu = mem.get<0>();
    array::Parcel * ptr_coeff_on_gpu = mem.get<1>();
    std::uint64_t total_malloc_size = mem.get_total_malloc_size();
    // call calculation kernel
    interpolant::call_newton_coeff_kernel(ptr_grid_on_gpu, ptr_coeff_on_gpu, total_malloc_size,
                                          stream.get_stream_ptr(), n_thread);
}

}  // namespace merlin

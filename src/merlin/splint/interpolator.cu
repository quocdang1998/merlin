// Copyright 2022 quocdang1998
#include "merlin/splint/interpolator.hpp"

#include <utility>  // std::move

#include "merlin/array/parcel.hpp"           // merlin::array::Parcel
#include "merlin/cuda/memory.hpp"            // merlin::cuda::Memory
#include "merlin/logger.hpp"                 // FAILURE
#include "merlin/splint/cartesian_grid.hpp"  // merlin::splint::CartesianGrid
#include "merlin/splint/tools.hpp"           // merlin::splint::construct_coeff_gpu

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Interpolator
// ---------------------------------------------------------------------------------------------------------------------

// Construct from a GPU array
splint::Interpolator::Interpolator(const splint::CartesianGrid & grid, array::Parcel & values,
                                   const Vector<splint::Method> & method, const cuda::Stream & stream,
                                   std::uint64_t num_threads) {
    // check shape
    if (grid.shape() != values.shape()) {
        FAILURE(std::invalid_argument, "Grid and data have different shape.\n");
    }
    // check if data array is C-contiguous
    if (!values.is_c_contiguous()) {
        FAILURE(std::invalid_argument, "Data must be C-contiguous.\n");
    }
    // check CUDA context of the stream
    stream.check_cuda_context();
    this->allocation_stream_ = stream.get_stream_ptr();
    // copy data onto GPU
    cuda::Memory gpu_mem(stream.get_stream_ptr(), grid, values, method);
    this->p_grid_ = gpu_mem.get<0>();
    this->p_coeff_ = gpu_mem.get<1>();
    this->p_method_ = gpu_mem.get<2>();
    gpu_mem.force_release();
    // calculate coefficients directly on the data
    std::uint64_t shared_mem_size = grid.sharedmem_size() + method.sharedmem_size();
    splint::construct_coeff_gpu(values.data(), this->p_grid_, this->p_method_, num_threads, shared_mem_size, &stream);
}

}  // namespace merlin

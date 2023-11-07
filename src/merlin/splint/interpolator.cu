// Copyright 2022 quocdang1998
#include "merlin/splint/interpolator.hpp"

#include <utility>  // std::move

#include "merlin/array/parcel.hpp"           // merlin::array::Parcel
#include "merlin/cuda_interface.hpp"         // merlin::cuda_mem_alloc, merlin::cuda_mem_cpy_device_to_host,
                                             // merlin::cuda_mem_free
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
                                   std::uint64_t n_threads) : ndim_(grid.ndim()) {
    // check shape
    if (grid.shape() != values.shape()) {
        FAILURE(std::invalid_argument, "Grid and data have different shape.\n");
    }
    if (grid.ndim() != method.size()) {
        FAILURE(std::invalid_argument, "Grid and method vector have different shape.\n");
    }
    // check if data array is C-contiguous
    if (!values.is_c_contiguous()) {
        FAILURE(std::invalid_argument, "Data must be C-contiguous.\n");
    }
    // check CUDA context of the stream
    stream.check_cuda_context();
    this->allocation_stream_ = stream.get_stream_ptr();
    this->gpu_id_ = stream.get_gpu().id();
    // copy data onto GPU
    cuda::Memory gpu_mem(stream.get_stream_ptr(), grid, values, method);
    this->p_grid_ = gpu_mem.get<0>();
    this->p_coeff_ = gpu_mem.get<1>();
    this->p_method_ = gpu_mem.get<2>();
    gpu_mem.force_release();
    // calculate coefficients directly on the data
    this->shared_mem_size_ = grid.sharedmem_size() + method.sharedmem_size();
    splint::construct_coeff_gpu(values.data(), this->p_grid_, this->p_method_, n_threads, this->shared_mem_size_,
                                &stream);
}

// Interpolate by GPU
floatvec splint::Interpolator::interpolate(const array::Parcel & points, const cuda::Stream & stream,
                                           std::uint64_t n_threads) {
    // check if interpolator is on CPU
    if (!this->on_gpu()) {
        FAILURE(std::invalid_argument, "Interpolator is initialized on CPU.\n");
    }
    // check points array
    if (points.ndim() != 2) {
        FAILURE(std::invalid_argument, "Expected array of coordinates a 2D table.\n");
    }
    if (!points.is_c_contiguous()) {
        FAILURE(std::invalid_argument, "Expected array of coordinates to be C-contiguous.\n");
    }
    if (points.shape()[1] != this->ndim_) {
        FAILURE(std::invalid_argument, "Array of coordinates and interpolator have different dimension.\n");
    }
    // evaluate interpolation
    floatvec evaluated_values(points.shape()[0]);
    std::uint64_t bytes_size = evaluated_values.size() * sizeof(double);
    void * result_gpu = cuda_mem_alloc(bytes_size, stream.get_stream_ptr());
    splint::eval_intpl_gpu(this->p_coeff_->data(), this->p_grid_, this->p_method_, points.data(),
                           evaluated_values.size(), reinterpret_cast<double *>(result_gpu), n_threads, this->ndim_,
                           this->shared_mem_size_, &stream);
    cuda_mem_cpy_device_to_host(evaluated_values.data(), result_gpu, bytes_size, stream.get_stream_ptr());
    cuda_mem_free(result_gpu, stream.get_stream_ptr());
    return evaluated_values;
}

}  // namespace merlin

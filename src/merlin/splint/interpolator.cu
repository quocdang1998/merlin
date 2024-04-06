// Copyright 2022 quocdang1998
#include "merlin/splint/interpolator.hpp"

#include <utility>  // std::move

#include "merlin/array/parcel.hpp"    // merlin::array::Parcel
#include "merlin/cuda_interface.hpp"  // merlin::cuda_mem_alloc, merlin::cuda_mem_cpy_device_to_host,
                                      // merlin::cuda_mem_free
#include "merlin/cuda/memory.hpp"     // merlin::cuda::Memory
#include "merlin/env.hpp"             // merlin::Environment
#include "merlin/logger.hpp"          // FAILURE
#include "merlin/splint/tools.hpp"    // merlin::splint::construct_coeff_gpu

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Create pointer to copied members of merlin::splint::Interpolator on GPU
void splint::create_intpl_gpuptr(const grid::CartesianGrid & cpu_grid, const Vector<splint::Method> & cpu_methods,
                                 grid::CartesianGrid *& gpu_pgrid, std::array<unsigned int, max_dim> *& gpu_pmethods,
                                 std::uintptr_t stream_ptr) {
    std::array<unsigned int, max_dim> converted_cpu_methods;
    converted_cpu_methods.fill(0);
    for (std::uint64_t i = 0; i < cpu_methods.size(); i++) {
        converted_cpu_methods[i] = static_cast<unsigned int>(cpu_methods[i]);
    }
    cuda::Memory gpu_mem(stream_ptr, cpu_grid, converted_cpu_methods);
    gpu_pgrid = gpu_mem.get<0>();
    gpu_pmethods = gpu_mem.get<1>();
    gpu_mem.disown();
}

// Interpolate by GPU
void splint::Interpolator::evaluate(const array::Parcel & points, DoubleVec & result, std::uint64_t n_threads) {
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
    if (points.shape()[0] != result.size()) {
        FAILURE(std::invalid_argument, "Size of result array must be equal to the number of points.\n");
    }
    // get CUDA Stream
    cuda::Stream & stream = std::get<cuda::Stream>(this->synchronizer_.synchronizer);
    std::uintptr_t current_ctx = stream.get_gpu().push_context();
    // evaluate interpolation
    std::uint64_t bytes_size = result.size() * sizeof(double);
    void * result_gpu = cuda_mem_alloc(bytes_size, stream.get_stream_ptr());
    splint::eval_intpl_gpu(this->p_coeff_->data(), this->p_grid_, this->p_method_, points.data(), result.size(),
                           reinterpret_cast<double *>(result_gpu), n_threads, this->ndim_, this->shared_mem_size_,
                           &stream);
    cuda_mem_cpy_device_to_host(result.data(), result_gpu, bytes_size, stream.get_stream_ptr());
    cuda_mem_free(result_gpu, stream.get_stream_ptr());
    cuda::Device::pop_context(current_ctx);
}

}  // namespace merlin

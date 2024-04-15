// Copyright 2022 quocdang1998
#include "merlin/splint/interpolator.hpp"

#include <utility>  // std::move

#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/cuda/memory.hpp"   // merlin::cuda::Memory
#include "merlin/env.hpp"           // merlin::Environment
#include "merlin/logger.hpp"        // merlin::Fatal
#include "merlin/splint/tools.hpp"  // merlin::splint::construct_coeff_gpu

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

// Deallocate memory on the global memory space of the current GPU.
void splint::cuda_mem_free(void * ptr, std::uint64_t stream_ptr) {
    ::cudaFreeAsync(ptr, reinterpret_cast<::cudaStream_t>(stream_ptr));
}

// ---------------------------------------------------------------------------------------------------------------------
// Interpolator
// ---------------------------------------------------------------------------------------------------------------------

// Interpolate by GPU
void splint::Interpolator::evaluate(const array::Parcel & points, DoubleVec & result, std::uint64_t n_threads) {
    // check if interpolator is on CPU
    if (!this->on_gpu()) {
        Fatal<std::invalid_argument>("Interpolator is initialized on CPU.\n");
    }
    // check points array
    if (points.ndim() != 2) {
        Fatal<std::invalid_argument>("Expected array of coordinates a 2D table.\n");
    }
    if (!points.is_c_contiguous()) {
        Fatal<std::invalid_argument>("Expected array of coordinates to be C-contiguous.\n");
    }
    if (points.shape()[1] != this->ndim_) {
        Fatal<std::invalid_argument>("Array of coordinates and interpolator have different dimension.\n");
    }
    if (points.shape()[0] != result.size()) {
        Fatal<std::invalid_argument>("Size of result array must be equal to the number of points.\n");
    }
    // get CUDA Stream
    cuda::Stream & stream = std::get<cuda::Stream>(this->synchronizer_.synchronizer);
    ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    std::uintptr_t current_ctx = stream.get_gpu().push_context();
    // evaluate interpolation
    std::uint64_t bytes_size = result.size() * sizeof(double);
    double * result_gpu;
    ::cudaMallocAsync(&result_gpu, bytes_size, cuda_stream);
    splint::eval_intpl_gpu(this->p_coeff_->data(), this->p_grid_, this->p_method_, points.data(), result.size(),
                           result_gpu, n_threads, this->ndim_, this->shared_mem_size_, &stream);
    ::cudaMemcpyAsync(result.data(), result_gpu, bytes_size, ::cudaMemcpyDeviceToHost, cuda_stream);
    ::cudaFreeAsync(result_gpu, cuda_stream);
    cuda::Device::pop_context(current_ctx);
}

}  // namespace merlin

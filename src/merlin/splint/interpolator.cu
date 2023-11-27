// Copyright 2022 quocdang1998
#include "merlin/splint/interpolator.hpp"

#include <utility>  // std::move

#include "merlin/array/parcel.hpp"           // merlin::array::Parcel
#include "merlin/cuda_interface.hpp"         // merlin::cuda_mem_alloc, merlin::cuda_mem_cpy_device_to_host,
                                             // merlin::cuda_mem_free
#include "merlin/cuda/memory.hpp"            // merlin::cuda::Memory
#include "merlin/env.hpp"                    // merlin::Environment
#include "merlin/logger.hpp"                 // FAILURE
#include "merlin/splint/tools.hpp"           // merlin::splint::construct_coeff_gpu

#define push_gpu(gpu)                                                                                                  \
    bool lock_success = Environment::mutex.try_lock();                                                                 \
    std::uintptr_t current_ctx = gpu.push_context()
#define pop_gpu()                                                                                                      \
    cuda::Device::pop_context(current_ctx);                                                                            \
    if (lock_success) Environment::mutex.unlock()

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Create pointer to copied members of merlin::splint::Interpolator on GPU
void splint::create_intpl_gpuptr(const grid::CartesianGrid & cpu_grid, const Vector<splint::Method> & cpu_methods,
                                 grid::CartesianGrid *& gpu_pgrid, Vector<splint::Method> *& gpu_pmethods,
                                 std::uintptr_t stream_ptr) {
    cuda::Memory gpu_mem(stream_ptr, cpu_grid, cpu_methods);
    gpu_pgrid = gpu_mem.get<0>();
    gpu_pmethods = gpu_mem.get<1>();
    gpu_mem.disown();
}

// Interpolate by GPU
floatvec splint::Interpolator::evaluate(const array::Parcel & points, std::uint64_t n_threads) {
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
    // get CUDA Stream
    cuda::Stream & stream = std::get<cuda::Stream>(this->synchronizer_.synchronizer);
    push_gpu(stream.get_gpu());
    // evaluate interpolation
    floatvec evaluated_values(points.shape()[0]);
    std::uint64_t bytes_size = evaluated_values.size() * sizeof(double);
    void * result_gpu = cuda_mem_alloc(bytes_size, stream.get_stream_ptr());
    splint::eval_intpl_gpu(this->p_coeff_->data(), this->p_grid_, this->p_method_, points.data(),
                           evaluated_values.size(), reinterpret_cast<double *>(result_gpu), n_threads, this->ndim_,
                           this->shared_mem_size_, &stream);
    cuda_mem_cpy_device_to_host(evaluated_values.data(), result_gpu, bytes_size, stream.get_stream_ptr());
    cuda_mem_free(result_gpu, stream.get_stream_ptr());
    pop_gpu();
    return evaluated_values;
}

}  // namespace merlin

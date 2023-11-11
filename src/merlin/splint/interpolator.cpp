// Copyright 2022 quocdang1998
#include "merlin/splint/interpolator.hpp"

#include <utility>  // std::move

#include "merlin/array/array.hpp"            // merlin::array::Array
#include "merlin/array/parcel.hpp"           // merlin::array::Parcel
#include "merlin/cuda_interface.hpp"         // merlin::cuda_mem_free
#include "merlin/cuda/device.hpp"            // merlin::cuda::Device
#include "merlin/env.hpp"                    // merlin::Environment
#include "merlin/logger.hpp"                 // FAILURE, merlin::cuda_compile_error
#include "merlin/splint/cartesian_grid.hpp"  // merlin::splint::CartesianGrid
#include "merlin/splint/tools.hpp"           // merlin::splint::construct_coeff_cpu

#define safety_lock() bool lock_success = Environment::mutex.try_lock()
#define safety_unlock()                                                                                                \
    if (lock_success) Environment::mutex.unlock()

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__

// Create pointer to copied members of merlin::splint::Interpolator on GPU
void splint::create_intpl_gpuptr(const splint::CartesianGrid & cpu_grid, const Vector<splint::Method> & cpu_methods,
                                 splint::CartesianGrid *& gpu_pgrid, Vector<splint::Method> *& gpu_pmethods,
                                 std::uintptr_t stream_ptr) {
    FAILURE(cuda_compile_error, "Cannot invoke GPU function since merlin is not compiled with CUDA option.\n");
}

#endif  // __MERLIN_CUDA__

// ---------------------------------------------------------------------------------------------------------------------
// Interpolator
// ---------------------------------------------------------------------------------------------------------------------

// Construct from a CPU array
splint::Interpolator::Interpolator(const splint::CartesianGrid & grid, const array::Array & values,
                                   const Vector<splint::Method> & method, ProcessorType processor) :
ndim_(grid.ndim()), shared_mem_size_(grid.sharedmem_size() + method.sharedmem_size()) {
    // check shape
    if (grid.shape() != values.shape()) {
        FAILURE(std::invalid_argument, "Grid and data have different shape.\n");
    }
    if (grid.ndim() != method.size()) {
        FAILURE(std::invalid_argument, "Grid and method vector have different shape.\n");
    }
    // initialize pointers
    if (processor == ProcessorType::Cpu) {
        // CPU
        this->synchronizer_ = Synchronizer(new std::future<void>());
        this->p_grid_ = new splint::CartesianGrid(grid);
        this->p_method_ = new Vector<splint::Method>(method);
        this->p_coeff_ = new array::Array(values);
        this->gpu_id_ = -1;
    } else if (processor == ProcessorType::Gpu) {
        // GPU
        this->synchronizer_ = Synchronizer(cuda::Stream(cuda::StreamSetting::NonBlocking));
        cuda::Stream & stream = std::get<cuda::Stream>(this->synchronizer_.synchronizer);
        splint::create_intpl_gpuptr(grid, method, this->p_grid_, this->p_method_, stream.get_stream_ptr());
        this->p_coeff_ = new array::Parcel(values.shape());
        static_cast<array::Parcel *>(this->p_coeff_)->transfer_data_to_gpu(values, stream);
        this->gpu_id_ = cuda::Device::get_current_gpu().id();
    }
}

// Calculate interpolation coefficients based on provided method
void splint::Interpolator::build_coefficients(std::uint64_t n_threads) {
    if (!(this->on_gpu())) {
        std::future<void> * current_sync = std::get<std::future<void> *>(this->synchronizer_.synchronizer);
        std::future<void> new_sync = std::async(std::launch::async, splint::construct_coeff_cpu, current_sync,
                                                this->p_coeff_->data(), this->p_grid_, this->p_method_, n_threads);
        this->synchronizer_ = Synchronizer(new std::future<void>(std::move(new_sync)));
    } else {
        safety_lock();
        cuda::Stream & stream = std::get<cuda::Stream>(this->synchronizer_.synchronizer);
        stream.get_gpu().set_as_current();
        splint::construct_coeff_gpu(this->p_coeff_->data(), this->p_grid_, this->p_method_, n_threads,
                                    this->shared_mem_size_, &stream);
        safety_unlock();
    }
}

/*
// Interpolation by CPU.
floatvec splint::Interpolator::interpolate(const array::Array & points, std::uint64_t n_threads) {
    // check if interpolator is on CPU
    if (this->on_gpu()) {
        FAILURE(std::invalid_argument, "Interpolator is initialized on GPU.\n");
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
    splint::eval_intpl_cpu(this->p_coeff_->data(), *(this->p_grid_), *(this->p_method_), points.data(),
                           evaluated_values.size(), evaluated_values.data(), n_threads);
    return evaluated_values;
}
*/
// Destructor
splint::Interpolator::~Interpolator(void) {
    if (this->p_coeff_ != nullptr) {
        delete this->p_coeff_;
    }
    if (!(this->on_gpu())) {
        // delete pointer to grid and method if the interpolator are on CPU
        if (this->p_grid_ != nullptr) {
            delete this->p_grid_;
        }
        if (this->p_method_ != nullptr) {
            delete this->p_method_;
        }
        std::future<void> * current_sync = std::get<std::future<void> *>(this->synchronizer_.synchronizer);
        if (current_sync != nullptr) {
            delete current_sync;
        }
    } else {
        // delete pointer to everything on GPU
        safety_lock();
        cuda::Device gpu(this->gpu_id_);
        gpu.set_as_current();
        if (this->p_grid_ != nullptr) {
            cuda_mem_free(this->p_grid_);
        }
        safety_unlock();
    }
}

}  // namespace merlin

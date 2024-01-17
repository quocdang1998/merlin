// Copyright 2022 quocdang1998
#include "merlin/splint/interpolator.hpp"

#include <future>   // std::async, std::shared_future
#include <sstream>  // std::ostringstream
#include <utility>  // std::move

#include "merlin/array/array.hpp"            // merlin::array::Array
#include "merlin/array/parcel.hpp"           // merlin::array::Parcel
#include "merlin/cuda/device.hpp"            // merlin::cuda::Device
#include "merlin/cuda_interface.hpp"         // merlin::cuda_mem_free
#include "merlin/env.hpp"                    // merlin::Environment
#include "merlin/logger.hpp"                 // FAILURE, merlin::cuda_compile_error
#include "merlin/splint/tools.hpp"           // merlin::splint::construct_coeff_cpu

#define push_gpu(gpu)                                                                                                  \
    std::uintptr_t current_ctx = gpu.push_context()
#define pop_gpu()                                                                                                      \
    cuda::Device::pop_context(current_ctx)

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__

// Create pointer to copied members of merlin::splint::Interpolator on GPU
void splint::create_intpl_gpuptr(const grid::CartesianGrid & cpu_grid, const Vector<splint::Method> & cpu_methods,
                                 grid::CartesianGrid *& gpu_pgrid, Vector<splint::Method> *& gpu_pmethods,
                                 std::uintptr_t stream_ptr) {
    FAILURE(cuda_compile_error, "Cannot invoke GPU function since merlin is not compiled with CUDA option.\n");
}

#endif  // __MERLIN_CUDA__

// ---------------------------------------------------------------------------------------------------------------------
// Interpolator
// ---------------------------------------------------------------------------------------------------------------------

// Construct from a CPU array
splint::Interpolator::Interpolator(const grid::CartesianGrid & grid, const array::Array & values,
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
        this->synchronizer_ = Synchronizer(std::shared_future<void>());
        this->p_grid_ = new grid::CartesianGrid(grid);
        this->p_method_ = new Vector<splint::Method>(method);
        this->p_coeff_ = new array::Array(values);
    } else if (processor == ProcessorType::Gpu) {
        // GPU
        this->synchronizer_ = Synchronizer(cuda::Stream(cuda::StreamSetting::NonBlocking));
        cuda::Stream & stream = std::get<cuda::Stream>(this->synchronizer_.synchronizer);
        push_gpu(stream.get_gpu());
        splint::create_intpl_gpuptr(grid, method, this->p_grid_, this->p_method_, stream.get_stream_ptr());
        this->p_coeff_ = new array::Parcel(values.shape());
        static_cast<array::Parcel *>(this->p_coeff_)->transfer_data_to_gpu(values, stream);
        pop_gpu();
    }
}

// Calculate interpolation coefficients based on provided method
void splint::Interpolator::build_coefficients(std::uint64_t n_threads) {
    if (!(this->on_gpu())) {
        std::shared_future<void> & current_sync = std::get<std::shared_future<void>>(this->synchronizer_.synchronizer);
        std::shared_future<void> new_sync = std::async(std::launch::async, splint::construct_coeff_cpu, current_sync,
                                                       this->p_coeff_->data(), this->p_grid_, this->p_method_,
                                                       n_threads).share();
        this->synchronizer_ = Synchronizer(std::move(new_sync));
    } else {
        cuda::Stream & stream = std::get<cuda::Stream>(this->synchronizer_.synchronizer);
        push_gpu(stream.get_gpu());
        splint::construct_coeff_gpu(this->p_coeff_->data(), this->p_grid_, this->p_method_, n_threads,
                                    this->shared_mem_size_, &stream);
        pop_gpu();
    }
}

// Interpolation by CPU.
floatvec splint::Interpolator::evaluate(const array::Array & points, std::uint64_t n_threads) {
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
    std::shared_future<void> & current_sync = std::get<std::shared_future<void>>(this->synchronizer_.synchronizer);
    std::shared_future<void> new_sync = std::async(std::launch::async, splint::eval_intpl_cpu, current_sync,
                                                   this->p_coeff_->data(), this->p_grid_, this->p_method_,
                                                   points.data(), evaluated_values.size(), evaluated_values.data(),
                                                   n_threads).share();
    this->synchronizer_ = Synchronizer(std::move(new_sync));
    return evaluated_values;
}

// String representation
std::string splint::Interpolator::str(void) const {
    std::ostringstream os;
    os << "<Interpolator of grid at " << this->p_grid_
       << ", coefficients at " << this->p_coeff_
       << ", method vector at " << this->p_method_
       << " and executed on " << ((this->gpu_id() == static_cast<unsigned int>(-1)) ? "CPU" : "GPU")
       << ">";
    return os.str();
}

// Destructor
splint::Interpolator::~Interpolator(void) {
    // deallocate coeff memory
    if (this->p_coeff_ != nullptr) {
        delete this->p_coeff_;
    }
    // delete grid and method memory
    if (!(this->on_gpu())) {
        // delete memory of each member on CPU
        if (this->p_grid_ != nullptr) {
            delete this->p_grid_;
        }
        if (this->p_method_ != nullptr) {
            delete this->p_method_;
        }
    } else {
        // delete joint memory of both on GPU
        if (this->p_grid_ != nullptr) {
            push_gpu(cuda::Device(this->gpu_id()));
            cuda_mem_free(this->p_grid_, std::get<cuda::Stream>(this->synchronizer_.synchronizer).get_stream_ptr());
            pop_gpu();
        }
    }
}

}  // namespace merlin

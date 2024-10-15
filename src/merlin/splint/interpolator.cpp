// Copyright 2022 quocdang1998
#include "merlin/splint/interpolator.hpp"

#include <future>   // std::async, std::future
#include <sstream>  // std::ostringstream
#include <utility>  // std::move

#include "merlin/array/array.hpp"        // merlin::array::Array
#include "merlin/array/parcel.hpp"       // merlin::array::Parcel
#include "merlin/cuda/copy_helpers.hpp"  // merlin::cuda::Dispatcher
#include "merlin/cuda/device.hpp"        // merlin::cuda::CtxGuard
#include "merlin/env.hpp"                // merlin::Environment
#include "merlin/logger.hpp"             // merlin::Fatal, merlin::cuda_compile_error
#include "merlin/memory.hpp"             // merlin::mem_alloc_device, merlin::memcpy_gpu_to_cpu,
                                         // merlin::mem_free_device, merlin::mem_free_device_noexcept
#include "merlin/splint/tools.hpp"       // merlin::splint::construct_coeff_cpu, merlin::splint::construct_coeff_gpu

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Interpolator
// ---------------------------------------------------------------------------------------------------------------------

// Construct from a CPU array
splint::Interpolator::Interpolator(const grid::CartesianGrid & grid, const array::Array & values,
                                   const splint::Method * p_method, Synchronizer & synchronizer) :
ndim_(grid.ndim()), shared_mem_size_(grid.sharedmem_size()), p_synch_(&synchronizer) {
    // check shape
    if (grid.shape() != values.shape()) {
        Fatal<std::invalid_argument>("Grid and data have different shape.\n");
    }
    // initialize pointers
    if (this->on_gpu()) {
        // initialize GPU context
        this->p_coeff_ = new array::Parcel(values.shape());
        cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
        cuda::CtxGuard guard(stream.get_gpu());
        // copy grid and method to GPU
        Index methods(this->ndim_);
        for (std::uint64_t i_dim = 0; i_dim < this->ndim_; i_dim++) {
            methods[i_dim] = static_cast<std::uint64_t>(p_method[i_dim]);
        }
        cuda::Dispatcher dispatcher(stream.get_stream_ptr(), grid, methods);
        this->p_grid_ = dispatcher.get<0>();
        this->p_method_ = dispatcher.get<1>();
        dispatcher.disown();
        // transfer parcel data to GPU
        static_cast<array::Parcel *>(this->p_coeff_)->transfer_data_to_gpu(values, stream);
    } else {
        // copy interpolation data
        this->p_grid_ = new grid::CartesianGrid(grid);
        this->p_coeff_ = new array::Array(values);
        this->p_method_ = new Index(this->ndim_);
        for (std::uint64_t i_dim = 0; i_dim < this->ndim_; i_dim++) {
            (*(this->p_method_))[i_dim] = static_cast<std::uint64_t>(p_method[i_dim]);
        }
    }
}

// Calculate interpolation coefficients based on provided method
void splint::Interpolator::build_coefficients(std::uint64_t n_threads) {
    if (!(this->on_gpu())) {
        std::future<void> & current_sync = std::get<std::future<void>>(this->p_synch_->core);
        std::future<void> new_sync = std::async(std::launch::async, splint::construct_coeff_cpu,
                                                std::move(current_sync), this->p_coeff_->data(), this->p_grid_,
                                                this->p_method_, n_threads);
        *(this->p_synch_) = Synchronizer(std::move(new_sync));
    } else {
        cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
        cuda::CtxGuard guard(stream.get_gpu());
        splint::construct_coeff_gpu(this->p_coeff_->data(), this->p_grid_, this->p_method_, n_threads,
                                    this->shared_mem_size_, &stream);
    }
}

// Interpolation by CPU.
void splint::Interpolator::evaluate(const array::Array & points, DoubleVec & result, std::uint64_t n_threads) {
    // check if interpolator is on CPU
    if (this->on_gpu()) {
        Fatal<std::invalid_argument>("Interpolator is initialized on GPU.\n");
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
    // evaluate interpolation
    std::future<void> & current_sync = std::get<std::future<void>>(this->p_synch_->core);
    std::future<void> new_sync = std::async(std::launch::async, splint::eval_intpl_cpu, std::move(current_sync),
                                            this->p_coeff_->data(), this->p_grid_, this->p_method_, points.data(),
                                            result.size(), result.data(), n_threads);
    *(this->p_synch_) = Synchronizer(std::move(new_sync));
}

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
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    std::uintptr_t stream_ptr = stream.get_stream_ptr();
    double * result_gpu;
    mem_alloc_device(reinterpret_cast<void **>(&result_gpu), result.size() * sizeof(double), stream_ptr);
    splint::eval_intpl_gpu(this->p_coeff_->data(), this->p_grid_, this->p_method_, points.data(), result.size(),
                           result_gpu, n_threads, this->ndim_, this->shared_mem_size_, &stream);
    stream.synchronize();
    memcpy_gpu_to_cpu(result.data(), result_gpu, result.size() * sizeof(double), stream_ptr);
    mem_free_device(result_gpu, stream_ptr);
}

// String representation
std::string splint::Interpolator::str(void) const {
    std::ostringstream os;
    os << "<Interpolator of grid at " << this->p_grid_ << ", coefficients at " << this->p_coeff_
       << ", method vector at " << this->p_method_ << " and executed on "
       << ((this->on_gpu()) ? "CPU" : "GPU") << ">";
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
            cuda::CtxGuard guard(cuda::Device(this->gpu_id()));
            mem_free_device_noexcept(this->p_grid_, std::get<cuda::Stream>(this->p_synch_->core).get_stream_ptr());
        }
    }
}

}  // namespace merlin

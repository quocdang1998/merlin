// Copyright 2023 quocdang1998
#include "merlin/candy/trainer.hpp"

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/cuda/stream.hpp"  // merlin::cuda::Stream
#include "merlin/cuda_interface.hpp"         // merlin::cuda_mem_free
#include "merlin/env.hpp"                    // merlin::Environment
#include "merlin/logger.hpp"  // FAILURE, merlin::cuda_compile_error

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

#ifndef __MERLIN_CUDA__

/** @brief Allocate memory on GPU for the trainer.*/
void candy::create_trainer_gpu_ptr(const candy::Model & cpu_model, const array::Array & cpu_data,
                                   const candy::Optimizer & cpu_optimizer, candy::Model *& gpu_model,
                                   array::NdData *& gpu_data, candy::Optimizer *& gpu_optimizer,
                                   array::Parcel *& parcel_data, cuda::Stream & stream) {
    FAILURE(cuda_compile_error, "Cannot invoke GPU function since merlin is not compiled with CUDA option.\n");
}

#endif  // __MERLIN_CUDA__

// ---------------------------------------------------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------------------------------------------------

// Constructor a trainer on CPU
candy::Trainer::Trainer(const candy::Model & model, array::Array && data, const candy::Optimizer & optimizer,
                        ProcessorType processor) : ndim_(model.ndim()) {
    // check arguments
    if (model.ndim() != data.ndim()) {
        FAILURE(std::invalid_argument, "Model and train data have inconsistent ndim.\n");
    }
    for (std::uint64_t i_dim = 0; i_dim < model.ndim(); i_dim++) {
        if (model.rshape()[i_dim] != data.shape()[i_dim] * model.rank()) {
            FAILURE(std::invalid_argument, "Model and train data have inconsistent shape.\n");
        }
    }
    if (!(optimizer.is_compatible(model))) {
        FAILURE(std::invalid_argument, "Model and Optimizer are incompatible.\n");
    }
    // copy and allocate data on CPU
    if (processor == ProcessorType::Cpu) {
        this->p_model_ = new candy::Model(model);
        this->p_data_ = new array::Array(std::forward<array::Array>(data));
        this->p_optmz_ = new candy::Optimizer(optimizer);
        this->synch_ = Synchronizer(std::shared_future<void>());
        this->cpu_grad_mem_ = new double[model.num_params()];
    } else {
        this->synch_ = Synchronizer(cuda::Stream(cuda::StreamSetting::NonBlocking));
        cuda::Stream & stream = std::get<cuda::Stream>(this->synch_.synchronizer);
        candy::create_trainer_gpu_ptr(model, std::forward<array::Array>(data), optimizer, this->p_model_, this->p_data_,
                                      this->p_optmz_, this->p_parcel_, stream);
        this->shared_mem_size_ = model.sharedmem_size() + this->p_parcel_->sharedmem_size() + optimizer.sharedmem_size();
        this->shared_mem_size_ += sizeof(double) * model.num_params();
    }
}

// Destructor
candy::Trainer::~Trainer(void) {
    if (!(this->on_gpu())) {
        if (this->p_model_ != nullptr) {
            delete this->p_model_;
        }
        if (this->p_optmz_ != nullptr) {
            delete this->p_optmz_;
        }
        if (this->cpu_grad_mem_ != nullptr) {
            delete[] this->cpu_grad_mem_;
        }
    } else {
        if (this->p_parcel_ != nullptr) {
            delete this->p_parcel_;
        }
        push_gpu(cuda::Device(this->gpu_id()));
        if (this->p_model_ != nullptr) {
            cuda_mem_free(this->p_model_, std::get<cuda::Stream>(this->synch_.synchronizer).get_stream_ptr());
        }
        pop_gpu();
    }
}

}  // namespace merlin

// Copyright 2023 quocdang1998
#include "merlin/candy/trainer.hpp"

#include <future>   // std::async, std::shared_future
#include <utility>  // std::move

#include <omp.h>  // #pragma omp

#include "merlin/array/array.hpp"      // merlin::array::Array
#include "merlin/array/parcel.hpp"     // merlin::array::Parcel
#include "merlin/candy/gradient.hpp"   // merlin::candy::Gradient
#include "merlin/candy/loss.hpp"       // merlin::candy::rmse_cpu
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/cuda/stream.hpp"      // merlin::cuda::Stream
#include "merlin/cuda_interface.hpp"   // merlin::cuda_mem_free
#include "merlin/env.hpp"              // merlin::Environment
#include "merlin/logger.hpp"           // FAILURE, merlin::cuda_compile_error

#define push_gpu(gpu)                                                                                                  \
    Environment::mutex.lock();                                                                                         \
    std::uintptr_t current_ctx = gpu.push_context()
#define pop_gpu()                                                                                                      \
    cuda::Device::pop_context(current_ctx);                                                                            \
    Environment::mutex.unlock()

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__

// Allocate memory on GPU for the trainer
void candy::create_trainer_gpu_ptr(const candy::Model & cpu_model, const array::Array & cpu_data,
                                   const candy::Optimizer & cpu_optimizer, candy::Model *& gpu_model,
                                   array::NdData *& gpu_data, candy::Optimizer *& gpu_optimizer,
                                   array::Parcel *& parcel_data, cuda::Stream & stream) {
    FAILURE(cuda_compile_error, "Cannot invoke GPU function since merlin is not compiled with CUDA option.\n");
}

#endif  // __MERLIN_CUDA__

// Train a model using CPU parallelism
void candy::train_by_cpu(std::shared_future<void> synch, candy::Model * p_model, array::Array * p_data,
                         candy::Optimizer * p_optimizer, double * cpu_grad_mem, candy::TrainMetric metric,
                         std::uint64_t rep, double threshold, std::uint64_t n_threads, intvec * p_cache_mem) {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    // create gradient object
    candy::Gradient gradient(cpu_grad_mem, p_model->num_params(), metric);
    // allocate data for cache memory
    if (p_cache_mem->size() != p_model->ndim() * n_threads) {
        *p_cache_mem = intvec(p_model->ndim() * n_threads, 0);
    }
    // calculate based on error
    double priori_error = 0.0;
    double posteriori_error = candy::rmse_cpu(p_model, p_data, p_cache_mem->data(), n_threads);
    bool go_on = true;
    do {
        priori_error = posteriori_error;
        for (std::uint64_t i = 0; i < rep; i++) {
            #pragma omp parallel num_threads(n_threads)
            {
                gradient.calc_by_cpu(*p_model, *p_data, ::omp_get_thread_num(), n_threads, p_cache_mem->data());
                p_optimizer->update_cpu(*p_model, gradient, ::omp_get_thread_num(), n_threads);
            }
        }
        posteriori_error = candy::rmse_cpu(p_model, p_data, p_cache_mem->data(), n_threads);
        if (std::isnormal(posteriori_error)) {
            go_on = (std::abs(priori_error - posteriori_error) / posteriori_error > threshold);
        } else {
            go_on = false;
        }
    } while (go_on);
}

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
        this->shared_mem_size_ = model.sharedmem_size() + optimizer.sharedmem_size();
        this->shared_mem_size_ += this->p_parcel_->sharedmem_size();
        this->shared_mem_size_ += sizeof(double) * model.num_params();
    }
}

#ifndef __MERLIN_CUDA__

// Get a copy to the current CP model
candy::Model candy::Trainer::get_model(void) const {
    if (this->on_gpu()) {
        FAILURE(cuda_compile_error, "Cannot invoke GPU function since merlin is not compiled with CUDA option.\n");
    }
    return candy::Model(*(this->p_model_));
}

#endif  // __MERLIN_CUDA__

// Update CP model according to gradient
void candy::Trainer::update(std::uint64_t rep, double threshold, std::uint64_t n_threads, candy::TrainMetric metric) {
    if (!(this->on_gpu())) {
        // launch asynchronously the update on CPU
        array::Array * p_train_data = static_cast<array::Array *>(this->p_data_);
        std::shared_future<void> & current_sync = std::get<std::shared_future<void>>(this->synch_.synchronizer);
        std::shared_future<void> new_sync = std::async(std::launch::async, candy::train_by_cpu, current_sync,
                                                       this->p_model_, p_train_data, this->p_optmz_,
                                                       this->cpu_grad_mem_, metric, rep, threshold, n_threads,
                                                       &(this->cpu_cache_mem_)).share();
        this->synch_ = Synchronizer(std::move(new_sync));
    } else {
        // launch asynchronously the update on GPU
        cuda::Stream & stream = std::get<cuda::Stream>(this->synch_.synchronizer);
        push_gpu(stream.get_gpu());
        candy::train_by_gpu(this->p_model_, static_cast<array::Parcel *>(this->p_data_), this->p_optmz_, metric, rep,
                            n_threads, this->ndim_, threshold, this->shared_mem_size_, stream);
        pop_gpu();
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

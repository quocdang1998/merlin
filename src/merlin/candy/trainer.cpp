// Copyright 2023 quocdang1998
#include "merlin/candy/trainer.hpp"

#include <cstddef>  // std::size_t
#include <future>   // std::async, std::future
#include <utility>  // std::move

#include <omp.h>  // #pragma omp

#include "merlin/array/array.hpp"     // merlin::array::Array
#include "merlin/array/parcel.hpp"    // merlin::array::Parcel
#include "merlin/candy/gradient.hpp"  // merlin::candy::Gradient
#include "merlin/candy/loss.hpp"      // merlin::candy::rmse_cpu
#include "merlin/env.hpp"             // merlin::Environment
#include "merlin/logger.hpp"          // merlin::Fatal, merlin::cuda_compile_error
#include "merlin/utils.hpp"           // merlin::is_normal

#define push_gpu(gpu) std::uintptr_t current_ctx = gpu.push_context()
#define pop_gpu() cuda::Device::pop_context(current_ctx)

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Train a model using CPU parallelism
void candy::train_by_cpu(std::future<void> && synch, candy::Model * p_model, const array::Array * p_data,
                         candy::Optimizer * p_optimizer, double * cpu_grad_mem, candy::TrainMetric metric,
                         std::uint64_t rep, double threshold, std::uint64_t n_threads) {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    // create gradient object
    candy::Gradient gradient(cpu_grad_mem, p_model->num_params(), metric);
    // calculate based on error
    double priori_error = 0.0;
    double posteriori_error;
    std::uint64_t normal_count;
    bool go_on = true;
    // calculate error before training
    _Pragma("omp parallel num_threads(n_threads)") {
        Index index_mem;
        index_mem.fill(0);
        std::uint64_t thread_idx = ::omp_get_thread_num();
        candy::rmse_cpu(p_model, p_data, posteriori_error, normal_count, thread_idx, n_threads, index_mem);
    }
    // training loop
    do {
        priori_error = posteriori_error;
        _Pragma("omp parallel num_threads(n_threads)") {
            Index index_mem;
            index_mem.fill(0);
            std::uint64_t thread_idx = ::omp_get_thread_num();
            // gradient descent loop
            for (std::uint64_t i = 0; i < rep; i++) {
                gradient.calc_by_cpu(*p_model, *p_data, thread_idx, n_threads, index_mem);
                p_optimizer->update_cpu(*p_model, gradient, thread_idx, n_threads);
            }
            _Pragma("omp barrier");
            candy::rmse_cpu(p_model, p_data, posteriori_error, normal_count, thread_idx, n_threads, index_mem);
        }
        double rel_err = std::abs(priori_error - posteriori_error) / posteriori_error;
        go_on = (is_normal(posteriori_error)) ? (rel_err > threshold) : false;
    } while (go_on);
}

// Calculate error using CPU parallelism
void candy::error_by_cpu(std::future<void> && synch, candy::Model * p_model, const array::Array * p_data,
                         double * p_rmse, double * p_rmae, std::uint64_t n_threads) {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    // calculate error
    std::uint64_t normal_count;
    _Pragma("omp parallel num_threads(n_threads)") {
        Index index_mem;
        index_mem.fill(0);
        std::uint64_t thread_idx = ::omp_get_thread_num();
        candy::rmse_cpu(p_model, p_data, *p_rmse, normal_count, thread_idx, n_threads, index_mem);
        candy::rmae_cpu(p_model, p_data, *p_rmae, normal_count, thread_idx, n_threads, index_mem);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------------------------------------------------

// Constructor a trainer
candy::Trainer::Trainer(const candy::Model & model, const candy::Optimizer & optimizer, ProcessorType processor) :
model_(model), optmz_(optimizer) {
    // check argument
    if (!(optimizer.is_compatible(model))) {
        Fatal<std::invalid_argument>("Model and Optimizer are incompatible.\n");
    }
    // initialize synchronizer and allocate memory for gradient
    if (processor == ProcessorType::Cpu) {
        this->synch_ = Synchronizer(std::future<void>());
        std::size_t mem_size = sizeof(double) * this->model_.num_params();
        this->cpu_grad_mem_ = static_cast<double *>(::operator new[](mem_size));
    } else {
        this->synch_ = Synchronizer(cuda::Stream(cuda::StreamSetting::NonBlocking));
    }
}

// Update CP model according to gradient on CPU
void candy::Trainer::update_cpu(const array::Array & data, std::uint64_t rep, double threshold, std::uint64_t n_threads,
                                candy::TrainMetric metric) {
    // check if trainer is on CPU
    if (this->on_gpu()) {
        Fatal<std::invalid_argument>("The current object is allocated on GPU.\n");
    }
    // asynchronous launch
    std::future<void> & current_sync = std::get<std::future<void>>(this->synch_.synchronizer);
    std::future<void> new_sync = std::async(std::launch::async, candy::train_by_cpu, std::move(current_sync),
                                            &(this->model_), &data, &(this->optmz_), this->cpu_grad_mem_, metric, rep,
                                            threshold, n_threads);
    this->synch_ = Synchronizer(std::move(new_sync));
}

// Get the RMSE and RMAE error with respect to a given dataset by CPU
void candy::Trainer::error_cpu(const array::Array & data, double & rmse, double & rmae, std::uint64_t n_threads) {
    // check if trainer is on CPU
    if (this->on_gpu()) {
        Fatal<std::invalid_argument>("The current object is allocated on GPU.\n");
    }
    // asynchronous launch
    std::future<void> & current_sync = std::get<std::future<void>>(this->synch_.synchronizer);
    std::future<void> new_sync = std::async(std::launch::async, candy::error_by_cpu, std::move(current_sync),
                                            &(this->model_), &data, &rmse, &rmae, n_threads);
    this->synch_ = Synchronizer(std::move(new_sync));
}

#ifndef __MERLIN_CUDA__

// Update CP model according to gradient on GPU
void candy::Trainer::update_gpu(const array::Parcel & data, std::uint64_t rep, double threshold,
                                std::uint64_t n_threads, candy::TrainMetric metric) {
    Fatal<cuda_compile_error>("Cannot invoke GPU function since merlin is not compiled with CUDA option.\n");
}

// Get the RMSE and RMAE error with respect to a given dataset by GPU
void candy::Trainer::error_gpu(const array::Parcel & data, double & rmse, double & rmae, std::uint64_t n_threads) {
    Fatal<cuda_compile_error>("Cannot invoke GPU function since merlin is not compiled with CUDA option.\n");
}

#endif  // __MERLIN__CUDA__

// Destructor
candy::Trainer::~Trainer(void) {
    if (this->cpu_grad_mem_ != nullptr) {
        std::size_t mem_size = sizeof(double) * this->model_.num_params();
        ::operator delete[](this->cpu_grad_mem_, mem_size);
    }
}

}  // namespace merlin

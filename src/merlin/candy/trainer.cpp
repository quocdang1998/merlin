// Copyright 2023 quocdang1998
#include "merlin/candy/trainer.hpp"

#include <cstddef>  // std::size_t
#include <future>   // std::async, std::future
#include <utility>  // std::move

#include <omp.h>  // ::omp_get_thread_num

#include "merlin/array/array.hpp"     // merlin::array::Array
#include "merlin/array/parcel.hpp"    // merlin::array::Parcel
#include "merlin/candy/gradient.hpp"  // merlin::candy::Gradient
#include "merlin/candy/loss.hpp"      // merlin::candy::rmse_cpu
#include "merlin/env.hpp"             // merlin::Environment
#include "merlin/logger.hpp"          // merlin::Fatal, merlin::cuda_compile_error
#include "merlin/utils.hpp"           // merlin::is_normal, merlin::contiguous_to_ndim_idx

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Train a model using CPU parallelism
void candy::train_by_cpu(std::future<void> && synch, candy::Model * p_model, const array::Array * p_data,
                         candy::Optimizer * p_optimizer, double * cpu_grad_mem, candy::TrainMetric metric,
                         std::uint64_t rep, double threshold, std::uint64_t n_threads, std::string * p_name,
                         std::string * p_fname) {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    candy::start_message(*p_name);
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
    std::uint64_t step = 1;
    do {
        priori_error = posteriori_error;
        _Pragma("omp parallel num_threads(n_threads)") {
            Index index_mem;
            index_mem.fill(0);
            std::uint64_t thread_idx = ::omp_get_thread_num();
            // gradient descent loop
            for (std::uint64_t i = 0; i < rep; i++) {
                gradient.calc_by_cpu(*p_model, *p_data, thread_idx, n_threads, index_mem);
                p_optimizer->update_cpu(*p_model, gradient, step + i, thread_idx, n_threads);
            }
            _Pragma("omp barrier");
            candy::rmse_cpu(p_model, p_data, posteriori_error, normal_count, thread_idx, n_threads, index_mem);
        }
        double rel_err = std::abs(priori_error - posteriori_error) / posteriori_error;
        go_on = (is_normal(posteriori_error)) ? (rel_err > threshold) : false;
        step += rep;
    } while (go_on);
    // save model to file
    candy::end_message(*p_name);
    candy::save_model(p_model, p_fname);
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

// Dry-run the gradient update algorithm using CPU parallelism
void candy::dryrun_by_cpu(std::future<void> && synch, candy::Model * p_model, const array::Array * p_data,
                          candy::Optimizer * p_optimizer, double * cpu_grad_mem, candy::TrainMetric metric,
                          std::uint64_t n_threads, double * error, std::uint64_t * count, std::uint64_t max_iter) {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    // create gradient object
    candy::Gradient gradient(cpu_grad_mem, p_model->num_params(), metric);
    // calculate initial error
    std::uint64_t normal_count;
    _Pragma("omp parallel num_threads(n_threads)") {
        Index index_mem;
        index_mem.fill(0);
        std::uint64_t thread_idx = ::omp_get_thread_num();
        candy::rmse_cpu(p_model, p_data, error[0], normal_count, thread_idx, n_threads, index_mem);
    }
    *count = 1;
    // repeatedly iterate until a surge in the error detected
    _Pragma("omp parallel num_threads(n_threads)") {
        Index index_mem;
        index_mem.fill(0);
        std::uint64_t thread_idx = ::omp_get_thread_num();
        // gradient descent loop
        for (std::uint64_t iter = 1; iter < max_iter; iter++) {
            gradient.calc_by_cpu(*p_model, *p_data, thread_idx, n_threads, index_mem);
            p_optimizer->update_cpu(*p_model, gradient, iter, thread_idx, n_threads);
            candy::rmse_cpu(p_model, p_data, error[iter], normal_count, thread_idx, n_threads, index_mem);
            bool break_condition = !is_normal(error[iter]) || (error[iter] / error[iter - 1] >= 1.0 + 1e-10);
            if (break_condition) {
                break;
            }
            _Pragma("omp single") { *count = iter + 1; }
            _Pragma("omp barrier");
        }
    }
    // release memory
    delete p_model;
    delete p_optimizer;
}

// Reconstruct CP-model using CPU parallelism
void candy::reconstruct_by_cpu(std::future<void> && synch, candy::Model * p_model, array::Array * p_data,
                               std::uint64_t n_threads) {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    _Pragma("omp parallel num_threads(n_threads)") {
        Index index;
        index.fill(0);
        std::uint64_t thread_idx = ::omp_get_thread_num();
        for (std::uint64_t c_index = thread_idx; c_index < p_data->size(); c_index += n_threads) {
            contiguous_to_ndim_idx(c_index, p_data->shape().data(), p_data->ndim(), index.data());
            p_data->operator[](index) = p_model->eval(index);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------------------------------------------------

// Constructor a trainer
candy::Trainer::Trainer(const std::string & name, const candy::Model & model, const candy::Optimizer & optimizer,
                        Synchronizer & synchronizer) :
name_(name), model_(model), optmz_(optimizer), p_synch_(&synchronizer) {
    // check argument
    if (!(optimizer.is_compatible(model))) {
        Fatal<std::invalid_argument>("Model and Optimizer are incompatible.\n");
    }
    // initialize synchronizer and allocate memory for gradient
    if (!this->on_gpu()) {
        std::size_t mem_size = sizeof(double) * this->model_.num_params();
        this->cpu_grad_mem_ = static_cast<double *>(::operator new[](mem_size));
    }
}

// Update CP model according to gradient on CPU
void candy::Trainer::update(const array::Array & data, std::uint64_t rep, double threshold, std::uint64_t n_threads,
                            candy::TrainMetric metric, const std::string & export_file) {
    // check if trainer is on CPU
    if (this->on_gpu()) {
        Fatal<std::invalid_argument>("The current object is allocated on GPU.\n");
    }
    // check shape
    if (!this->model_.check_compatible_shape(data.shape())) {
        Fatal<std::invalid_argument>("Incompatible shape between data and model.\n");
    }
    // asynchronous launch
    std::string * p_fname = new std::string(export_file);
    std::future<void> & current_sync = std::get<std::future<void>>(this->p_synch_->core);
    std::future<void> new_sync = std::async(std::launch::async, candy::train_by_cpu, std::move(current_sync),
                                            &(this->model_), &data, &(this->optmz_), this->cpu_grad_mem_, metric, rep,
                                            threshold, n_threads, &(this->name_), p_fname);
    *(this->p_synch_) = Synchronizer(std::move(new_sync));
}

// Get the RMSE and RMAE error with respect to a given dataset by CPU
void candy::Trainer::get_error(const array::Array & data, double & rmse, double & rmae, std::uint64_t n_threads) {
    // check if trainer is on CPU
    if (this->on_gpu()) {
        Fatal<std::invalid_argument>("The current object is allocated on GPU.\n");
    }
    // check shape
    if (!this->model_.check_compatible_shape(data.shape())) {
        Fatal<std::invalid_argument>("Incompatible shape between data and model.\n");
    }
    // asynchronous launch
    std::future<void> & current_sync = std::get<std::future<void>>(this->p_synch_->core);
    std::future<void> new_sync = std::async(std::launch::async, candy::error_by_cpu, std::move(current_sync),
                                            &(this->model_), &data, &rmse, &rmae, n_threads);
    *(this->p_synch_) = Synchronizer(std::move(new_sync));
}

// Dry-run
void candy::Trainer::dry_run(const array::Array & data, DoubleVec & error, std::uint64_t & actual_iter,
                             std::uint64_t max_iter, std::uint64_t n_threads, candy::TrainMetric metric) {
    // check if trainer is on CPU
    if (this->on_gpu()) {
        Fatal<std::invalid_argument>("The current object is allocated on GPU.\n");
    }
    // check shape
    if (!this->model_.check_compatible_shape(data.shape())) {
        Fatal<std::invalid_argument>("Incompatible shape between data and model.\n");
    }
    // check error length
    if (error.size() < max_iter) {
        Fatal<std::invalid_argument>("Size of error must be greater or equal to max_iter.\n");
    }
    // copy current model and optimizer
    candy::Model * model_dry = new candy::Model(this->model_);
    candy::Optimizer * optmz_dry = new candy::Optimizer(this->optmz_);
    // asynchronous launch
    std::future<void> & current_sync = std::get<std::future<void>>(this->p_synch_->core);
    std::future<void> new_sync = std::async(std::launch::async, candy::dryrun_by_cpu, std::move(current_sync),
                                            model_dry, &data, optmz_dry, this->cpu_grad_mem_, metric, n_threads,
                                            error.data(), &actual_iter, max_iter);
    *(this->p_synch_) = Synchronizer(std::move(new_sync));
}

// Reconstruct a whole multi-dimensional data from the model
void candy::Trainer::reconstruct(array::Array & destination, std::uint64_t n_threads) {
    // check if trainer is on CPU
    if (this->on_gpu()) {
        Fatal<std::invalid_argument>("The current object is allocated on GPU.\n");
    }
    // check shape
    if (!this->model_.check_compatible_shape(destination.shape())) {
        Fatal<std::invalid_argument>("Incompatible shape between data and model.\n");
    }
    // asynchronous launch
    std::future<void> & current_sync = std::get<std::future<void>>(this->p_synch_->core);
    std::future<void> new_sync = std::async(std::launch::async, candy::reconstruct_by_cpu, std::move(current_sync),
                                            &this->model_, &destination, n_threads);
    *(this->p_synch_) = Synchronizer(std::move(new_sync));
}

#ifndef __MERLIN_CUDA__

// Update CP model according to gradient on GPU
void candy::Trainer::update(const array::Parcel & data, std::uint64_t rep, double threshold, std::uint64_t n_threads,
                            candy::TrainMetric metric, const std::string & export_file) {
    Fatal<cuda_compile_error>("Cannot invoke GPU function since merlin is not compiled with CUDA option.\n");
}

// Get the RMSE and RMAE error with respect to a given dataset by GPU
void candy::Trainer::get_error(const array::Parcel & data, double & rmse, double & rmae, std::uint64_t n_threads) {
    Fatal<cuda_compile_error>("Cannot invoke GPU function since merlin is not compiled with CUDA option.\n");
}

// Dry-run using GPU
void candy::Trainer::dry_run(const array::Parcel & data, DoubleVec & error, std::uint64_t & actual_iter,
                             std::uint64_t max_iter, std::uint64_t n_threads, candy::TrainMetric metric) {
    Fatal<cuda_compile_error>("Cannot invoke GPU function since merlin is not compiled with CUDA option.\n");
}

// Reconstruct a whole multi-dimensional data from the model using GPU parallelism
void candy::Trainer::reconstruct(array::Parcel & destination, std::uint64_t n_threads) {
    Fatal<cuda_compile_error>("Cannot invoke GPU function since merlin is not compiled with CUDA option.\n");
}

#endif  // __MERLIN__CUDA__

// Change optimizer
void candy::Trainer::change_optmz(candy::Optimizer && new_optimizer) {
    if (!new_optimizer.is_compatible(this->model_)) {
        Fatal<std::invalid_argument>("New optimizer is not compatible with the current model.\n");
    }
    this->optmz_ = std::forward<candy::Optimizer>(new_optimizer);
}

// Destructor
candy::Trainer::~Trainer(void) {
    if (this->cpu_grad_mem_ != nullptr) {
        std::size_t mem_size = sizeof(double) * this->model_.num_params();
        ::operator delete[](this->cpu_grad_mem_, mem_size);
    }
}

}  // namespace merlin

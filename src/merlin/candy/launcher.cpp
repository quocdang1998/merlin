// Copyright 2023 quocdang1998
#include "merlin/candy/launcher.hpp"

#include <utility>  // std::move

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/logger.hpp"  // FAILURE, merlin::cuda_compile_error
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx, merlin::contiguous_to_model_idx

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// CPU asynchronous launch
// --------------------------------------------------------------------------------------------------------------------

static void update_gradient(candy::Model * p_model, const array::Array * p_train_data, candy::Optimizer * p_optimizer,
                            std::uint64_t model_size, std::uint64_t n_thread, std::uint64_t rep) {
    // get number of points and data shape
    intvec model_shape = p_model->get_model_shape();
    std::uint64_t n_point = p_train_data->size(), n_dim = p_train_data->ndim();
    const intvec & data_shape = p_train_data->shape();
    // repeat for rep time
    for (std::uint64_t time = 0; time < rep; time++) {
        // loop over each parameter
        #pragma omp parallel for num_threads(n_thread)
        for (std::int64_t i_param = 0; i_param < model_size; i_param++) {
            // initialize gradient
            double gradient = 0.0;
            // get parameter index
            auto [param_dim, param_index, param_rank] = contiguous_to_model_idx(i_param, p_model->rank(), model_shape);
            // loop over each point in the dataset to calculate the gradient
            std::uint64_t n_subset = n_point / data_shape[param_dim];
            for (std::uint64_t i_point = 0; i_point < n_subset; i_point++) {
                intvec index_data = candy::contiguous_to_ndim_idx_1(i_point, data_shape, param_dim);
                index_data[param_dim] = param_index;
                double data = p_train_data->get(index_data);
                if (data == 0) {
                    continue;
                }
                double point_gradient = 1.0;
                // divide by 1/data^2
                point_gradient /= data * data;
                // multiply by coefficient of the same rank from other dimension
                for (std::uint64_t i_dim = 0; i_dim < n_dim; i_dim++) {
                    if (i_dim == param_dim) {
                        continue;
                    }
                    point_gradient *= p_model->get(i_dim, index_data[i_dim], param_rank);
                }
                // multiply by value evaluation
                double point_eval = p_model->eval(index_data);
                point_gradient *= point_eval - data;
                // add gradient on a point to gradient of parameter
                gradient += point_gradient;
            }
            // update model parameter bu gradient method
            p_optimizer->update_cpu(*p_model, gradient, i_param, param_dim, param_index, param_rank);
        }
    }
}

// Launch asynchroniously model fitting algorithm on CPU
std::future<void> * candy::cpu_async_launch(candy::Model * p_model, const array::Array * p_train_data,
                                            candy::Optimizer * p_optimizer, std::uint64_t model_size,
                                            std::uint64_t n_thread, std::uint64_t rep) {
    std::future<void> result = std::async(std::launch::async, update_gradient, p_model, p_train_data, p_optimizer,
                                          model_size, n_thread, rep);
    return new std::future<void>(std::move(result));
}

// --------------------------------------------------------------------------------------------------------------------
// Launcher
// --------------------------------------------------------------------------------------------------------------------

// Constructor from a model and CPU array
candy::Launcher::Launcher(candy::Model & model, const array::Array & train_data, candy::Optimizer & optimizer,
                          std::uint64_t n_thread) :
p_model_(&model), p_data_(&train_data), p_optimizer_(&optimizer), n_thread_(n_thread), model_size_(model.size()) {
    // check model and data size
    if (model.ndim() != train_data.ndim()) {
        FAILURE(std::invalid_argument, "Model and train data have different ndim.\n");
    }
    intvec model_shape = model.get_model_shape();
    for (std::uint64_t i_dim = 0; i_dim < model.ndim(); i_dim++) {
        if (model_shape[i_dim] != model.rank() * train_data.shape()[i_dim]) {
            FAILURE(std::invalid_argument, "Model and train data have different shape.\n");
        }
    }
}

#ifndef __MERLIN_CUDA__

// Constructor from a model and array on GPU
candy::Launcher::Launcher(candy::Model * p_model, const array::Parcel * p_train_data, candy::Optimizer * p_optimizer,
                          std::uint64_t model_size, std::uint64_t share_mem_size, std::uint64_t block_size) {
    FAILURE(cuda_compile_error, "Cannot initilize launcher without using CUDA option.\n");
}

#endif  // __MERLIN_CUDA__

// Launch asynchronously the gradient update
void candy::Launcher::launch_async(std::uint64_t rep) {
    if (!this->is_gpu()) {
        const array::Array * p_train_data = static_cast<const array::Array *>(this->p_data_);
        std::future<void> * future_ptr = candy::cpu_async_launch(this->p_model_, p_train_data, this->p_optimizer_,
                                                                 this->model_size_, this->n_thread_, rep);
        this->synchronizer_ = reinterpret_cast<void *>(future_ptr);
    } else {
        FAILURE(std::runtime_error, "GPU launch not implemented.\n");
    }
}

// Synchronize the launch and delete pointer to synchronizer
void candy::Launcher::synchronize(void) {
    if (this->synchronizer_ == nullptr) {
        FAILURE(std::runtime_error, "Asynchronous task has not yet been configured.\n");
    }
    if (!this->is_gpu()) {
        std::future<void> * future_ptr = reinterpret_cast<std::future<void> *>(this->synchronizer_);
        future_ptr->wait();
        delete future_ptr;
        this->synchronizer_ = nullptr;
    } else {
        FAILURE(std::runtime_error, "GPU launch not implemented.\n");
    }
}

#ifndef __MERLIN_CUDA__

// Destructor
candy::Launcher::~Launcher(void) {
    // deallocate synchronizer
    if (this->synchronizer_ != nullptr) {
        std::future<void> * p_future = reinterpret_cast<std::future<void> *>(this->synchronizer_);
        delete p_future;
    }
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin

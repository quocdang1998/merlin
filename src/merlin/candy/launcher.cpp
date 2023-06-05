// Copyright 2023 quocdang1998
#include "merlin/candy/launcher.hpp"

#include <functional>  // std::ref
#include <future>  // std::future, std::aync
#include <utility>  // std::move

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/candy/loss.hpp"  // merlin::candy::calc_gradient_vector_cpu
#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// CPU asynchronous launch
// --------------------------------------------------------------------------------------------------------------------

static void update_gradient(candy::Model * p_model, const array::Array * p_train_data, candy::Optimizer * p_optimizer,
                            floatvec & gradient, std::uint64_t rep, std::uint64_t n_thread) {
    // repeat for rep time
    for (std::uint64_t time = 0; time < rep; time++) {
        // update gradient
        candy::calc_gradient_vector_cpu(*p_model, *p_train_data, gradient, n_thread);
        // update model accordingly
        p_optimizer->update_cpu(*p_model, gradient);
    }
}

static std::future<void> * cpu_async_launch(candy::Model * p_model, const array::Array * p_train_data,
                                          candy::Optimizer * p_optimizer, floatvec & gradient, std::uint64_t rep,
                                          std::uint64_t n_thread) {
    std::future<void> result = std::async(std::launch::async, update_gradient, p_model, p_train_data, p_optimizer,
                                          std::ref(gradient), rep, n_thread);
    return new std::future<void>(std::move(result));
}

// --------------------------------------------------------------------------------------------------------------------
// Launcher
// --------------------------------------------------------------------------------------------------------------------

// Constructor from a model and CPU array
candy::Launcher::Launcher(candy::Model & model, const array::Array & train_data, candy::Optimizer & optimizer,
                          std::uint64_t n_thread) :
p_model_(&model), p_data_(&train_data), p_optimizer_(&optimizer), n_thread_(n_thread), gradient_(model.size()) {
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

// Launch asynchronously the gradient update
void candy::Launcher::launch_async(std::uint64_t rep) {
    if (!this->is_gpu()) {
        const array::Array * p_train_data = static_cast<const array::Array *>(this->p_data_);
        std::future<void> * future_ptr = cpu_async_launch(this->p_model_, p_train_data, this->p_optimizer_,
                                                          this->gradient_, rep, this->n_thread_);
        this->synchronizer_ = reinterpret_cast<void *>(future_ptr);
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
    }
}

// Destructor
candy::Launcher::~Launcher(void) {
    if (this->synchronizer_ != nullptr) {
        if (!this->is_gpu()) {
            std::future<void> * p_future = reinterpret_cast<std::future<void> *>(this->synchronizer_);
            delete p_future;
        }
    }
}

}  // namespace merlin

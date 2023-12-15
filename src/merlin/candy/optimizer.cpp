// Copyright 2023 quocdang1998
#include "merlin/candy/optimizer.hpp"

#include <array>        // std::array
#include <cstring>      // std::memcpy
#include <type_traits>  // std::add_pointer
#include <utility>      // std::exchange

#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/logger.hpp"       // FAILURE, cuda_compile_error

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------------------------------------------------

// Copy constructor
candy::Optimizer::Optimizer(const candy::Optimizer & src) :
static_data(src.static_data), dynamic_size(src.dynamic_size) {
    // copy dynamic data
    if (src.dynamic_size != 0) {
        this->dynamic_data = new char[src.dynamic_size];
        std::memcpy(this->dynamic_data, src.dynamic_data, src.dynamic_size);
    }
    // assign data
    switch (this->static_data.index()) {
        case 0 : {  // gradient descent
            break;
        }
        case 1 : {  // adagrad
            candy::optmz::AdaGrad & opt_algor = std::get<candy::optmz::AdaGrad>(this->static_data);
            opt_algor.grad_history = reinterpret_cast<double *>(this->dynamic_data);
            break;
        }
    }
}

// Copy assignment
candy::Optimizer & candy::Optimizer::operator=(const candy::Optimizer & src) {
    this->static_data = src.static_data;
    this->dynamic_size = src.dynamic_size;
    // release old dynamic data
    if (this->dynamic_data != nullptr) {
        delete[] this->dynamic_data;
    }
    // copy dynamic data
    if (src.dynamic_size != 0) {
        this->dynamic_data = new char[src.dynamic_size];
        std::memcpy(this->dynamic_data, src.dynamic_data, src.dynamic_size);
    }
    // assign data
    switch (this->static_data.index()) {
        case 0 : {  // gradient descent
            break;
        }
        case 1 : {  // adagrad
            candy::optmz::AdaGrad & opt_algor = std::get<candy::optmz::AdaGrad>(this->static_data);
            opt_algor.grad_history = reinterpret_cast<double *>(this->dynamic_data);
            break;
        }
    }
    return *this;
}

// Move constructor
candy::Optimizer::Optimizer(candy::Optimizer && src) : static_data(src.static_data), dynamic_size(src.dynamic_size) {
    // exchange pointer
    this->dynamic_data = std::exchange(src.dynamic_data, nullptr);
}

// Move assignment
candy::Optimizer & candy::Optimizer::operator=(candy::Optimizer && src) {
    this->static_data = src.static_data;
    this->dynamic_size = src.dynamic_size;
    // exchange pointer
    this->dynamic_data = std::exchange(src.dynamic_data, nullptr);
    return *this;
}

// Update model inside a CPU parallel region
void candy::Optimizer::update_cpu(candy::Model & model, const candy::Gradient & grad, std::uint64_t thread_idx,
                                  std::uint64_t n_threads) noexcept {
    using UpdaterByCpu = std::add_pointer<void(void *, candy::Model &, const candy::Gradient &, std::uint64_t,
                                               std::uint64_t) noexcept>::type;
    static std::array<UpdaterByCpu, 2> cpu_updater_func = {candy::optmz::GradDescent::update_cpu,
                                                           candy::optmz::AdaGrad::update_cpu};
    void * optimizer_algor = reinterpret_cast<void *>(&this->static_data);
    cpu_updater_func[this->static_data.index()](optimizer_algor, model, grad, thread_idx, n_threads);
}

#ifndef __MERLIN_CUDA__

// Copy the optimizer from CPU to a pre-allocated memory on GPU
void * candy::Optimizer::copy_to_gpu(candy::Optimizer * gpu_ptr, void * dynamic_data_ptr,
                                     std::uintptr_t stream_ptr) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

#endif  // __MERLIN_CUDA__

// Destructor
candy::Optimizer::~Optimizer(void) {
    if (this->dynamic_data != nullptr) {
        delete[] this->dynamic_data;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------------------------------------------------

// Create an optimizer with gradient descent algorithm
candy::Optimizer candy::create_grad_descent(double learning_rate) {
    candy::Optimizer opt;
    opt.static_data =
        candy::OptmzStatic(std::in_place_type<candy::optmz::GradDescent>, candy::optmz::GradDescent(learning_rate));
    return opt;
}

// Create an optimizer with adagrad algorithm
candy::Optimizer candy::create_adagrad(double learning_rate, const candy::Model & model, double bias) {
    candy::Optimizer opt;
    std::uint64_t num_params = model.num_params();
    opt.dynamic_size = sizeof(double) * num_params;
    opt.dynamic_data = new char[opt.dynamic_size];
    opt.static_data = candy::OptmzStatic(std::in_place_type<candy::optmz::AdaGrad>,
                                         candy::optmz::AdaGrad(learning_rate, opt.dynamic_data, bias));
    return opt;
}

}  // namespace merlin

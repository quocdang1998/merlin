// Copyright 2023 quocdang1998
#include "merlin/candy/optimizer.hpp"

#include <array>    // std::array
#include <cstring>  // std::memcpy, std::memset
#include <sstream>  // std::ostringstream

#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/logger.hpp"       // merlin::Fatal, merlin::cuda_compile_error

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

static void print_dynamic_data(std::ostringstream & os, const double * dynamic_data, std::uint64_t dynamic_size) {
    os << "dynamic_data=<";
    for (std::uint64_t i = 0; i < dynamic_size; i++) {
        if (i != 0) {
            os << " ";
        }
        os << dynamic_data[i];
    }
    os << ">";
}

// ---------------------------------------------------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------------------------------------------------

// Copy constructor
candy::Optimizer::Optimizer(const candy::Optimizer & src) :
static_data(src.static_data), dynamic_size(src.dynamic_size) {
    // copy dynamic data
    if (src.dynamic_size != 0) {
        this->dynamic_data = new double[src.dynamic_size];
        std::memcpy(this->dynamic_data, src.dynamic_data, sizeof(double) * src.dynamic_size);
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
        this->dynamic_data = new double[src.dynamic_size];
        std::memcpy(this->dynamic_data, src.dynamic_data, sizeof(double) * src.dynamic_size);
    }
    return *this;
}

// Check compatibility with a model
bool candy::Optimizer::is_compatible(const candy::Model & model) const {
    switch (this->static_data.index()) {
        case 0 : {  // gradient descent
            break;
        }
        case 1 :
        case 4 : {  // adagrad, rmsprop
            if (this->dynamic_size != model.num_params()) {
                return false;
            }
            break;
        }
        case 2 :
        case 3 : {  // adam, adadelta
            if (this->dynamic_size != 2 * model.num_params()) {
                return false;
            }
            break;
        }
    }
    return true;
}

// Update model inside a CPU parallel region
void candy::Optimizer::update_cpu(candy::Model & model, const candy::Gradient & grad, std::uint64_t time_step,
                                  std::uint64_t thread_idx, std::uint64_t n_threads) noexcept {
    static std::array<candy::OptmzUpdater, 5> cpu_updater_func = {
        candy::optmz::GradDescent::update_cpu,  // grad descent
        candy::optmz::AdaGrad::update_cpu,      // adagrad
        candy::optmz::Adam::update_cpu,         // adam
        candy::optmz::AdaDelta::update_cpu,     // adadelta
        candy::optmz::RmsProp::update_cpu,      // rmsprop
    };
    void * optimizer_algor = reinterpret_cast<void *>(&this->static_data);
    cpu_updater_func[this->static_data.index()](optimizer_algor, this->dynamic_data, model, grad, time_step, thread_idx,
                                                n_threads);
    _Pragma("omp barrier");
}

#ifndef __MERLIN_CUDA__

// Copy the optimizer from CPU to a pre-allocated memory on GPU
void * candy::Optimizer::copy_to_gpu(candy::Optimizer * gpu_ptr, void * dynamic_data_ptr,
                                     std::uintptr_t stream_ptr) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

// Copy data from GPU to CPU
void * candy::Optimizer::copy_from_gpu(double * data_from_gpu, std::uintptr_t stream_ptr) noexcept {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

#endif  // __MERLIN_CUDA__

// String representation
std::string candy::Optimizer::str(void) const {
    std::ostringstream os;
    os << "<Optimizer(";
    switch (this->static_data.index()) {
        case 0 : {  // gradient descent
            os << "type=GradDescent, ";
            const candy::optmz::GradDescent & algor = std::get<0>(this->static_data);
            os << "eta=" << algor.learning_rate;
            break;
        }
        case 1 : {  // adagrad
            os << "type=AdaGrad, ";
            const candy::optmz::AdaGrad & algor = std::get<1>(this->static_data);
            os << "eta=" << algor.learning_rate << ", ";
            os << "bias=" << algor.bias << ", ";
            print_dynamic_data(os, this->dynamic_data, this->dynamic_size);
            break;
        }
        case 2 : {  // adam
            os << "type=Adam, ";
            const candy::optmz::Adam & algor = std::get<2>(this->static_data);
            os << "eta=" << algor.learning_rate << ", ";
            os << "beta_m=" << algor.beta_m << ", ";
            os << "beta_v=" << algor.beta_v << ", ";
            os << "bias=" << algor.bias << ", ";
            print_dynamic_data(os, this->dynamic_data, this->dynamic_size);
            break;
        }
        case 3 : {  // adadelta
            os << "type=AdaDelta, ";
            const candy::optmz::AdaDelta & algor = std::get<3>(this->static_data);
            os << "eta=" << algor.learning_rate << ", ";
            os << "rho=" << algor.rho << ", ";
            os << "bias=" << algor.bias << ", ";
            print_dynamic_data(os, this->dynamic_data, this->dynamic_size);
            break;
        }
        case 4 : {  // rmsprop
            os << "type=RmsProp, ";
            const candy::optmz::RmsProp & algor = std::get<4>(this->static_data);
            os << "eta=" << algor.learning_rate << ", ";
            os << "beta=" << algor.beta << ", ";
            os << "bias=" << algor.bias << ", ";
            print_dynamic_data(os, this->dynamic_data, this->dynamic_size);
            break;
        }
    }
    os << ")>";
    return os.str();
}

// Destructor
candy::Optimizer::~Optimizer(void) {
    if (this->dynamic_data != nullptr) {
        delete[] this->dynamic_data;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------------------------------------------------

// Check for positive learning rate
static inline void check_learning_rate(double learning_rate) {
    if (learning_rate <= 0) {
        Fatal<std::invalid_argument>("Learning rate must be positive.\n");
    }
}

// Check for positive bias
static inline void check_bias(double bias) {
    if (bias < 0) {
        Fatal<std::invalid_argument>("Bias must be positive.\n");
    }
}

// Check for weight
static inline void check_weight(double weight) {
    if (weight * (weight - 1.0) > 0) {
        Fatal<std::invalid_argument>("Weight value must be in range [0.0, 1.0].\n");
    }
}

// Create an optimizer with gradient descent algorithm
candy::Optimizer candy::create_grad_descent(double learning_rate) {
    // check argument
    check_learning_rate(learning_rate);
    // construct optimizer
    candy::Optimizer opt;
    opt.static_data = candy::OptmzStatic(std::in_place_type<candy::optmz::GradDescent>,
                                         candy::optmz::GradDescent(learning_rate));
    return opt;
}

// Create an optimizer with adagrad algorithm
candy::Optimizer candy::create_adagrad(double learning_rate, const candy::Model & model, double bias) {
    // check argument
    check_learning_rate(learning_rate);
    check_bias(bias);
    // construct optimizer
    candy::Optimizer opt;
    opt.dynamic_size = model.num_params();
    opt.dynamic_data = new double[opt.dynamic_size];
    std::memset(opt.dynamic_data, 0, sizeof(double) * opt.dynamic_size);
    opt.static_data = candy::OptmzStatic(std::in_place_type<candy::optmz::AdaGrad>,
                                         candy::optmz::AdaGrad(learning_rate, bias));
    return opt;
}

// Create an optimizer with adam algorithm
candy::Optimizer candy::create_adam(double learning_rate, double beta_m, double beta_v, const candy::Model & model,
                                    double bias) {
    // check argument
    check_learning_rate(learning_rate);
    check_bias(bias);
    check_weight(beta_m);
    check_weight(beta_v);
    // construct optimizer
    candy::Optimizer opt;
    opt.dynamic_size = 2 * model.num_params();
    opt.dynamic_data = new double[opt.dynamic_size];
    std::memset(opt.dynamic_data, 0, sizeof(double) * opt.dynamic_size);
    opt.static_data = candy::OptmzStatic(std::in_place_type<candy::optmz::Adam>,
                                         candy::optmz::Adam(learning_rate, beta_m, beta_v, bias));
    return opt;
}

// Create an optimizer with adadelta algorithm
candy::Optimizer candy::create_adadelta(double learning_rate, double rho, const candy::Model & model, double bias) {
    // check argument
    check_learning_rate(learning_rate);
    check_bias(bias);
    check_weight(rho);
    // construct optimizer
    candy::Optimizer opt;
    opt.dynamic_size = 2 * model.num_params();
    opt.dynamic_data = new double[opt.dynamic_size];
    std::memset(opt.dynamic_data, 0, sizeof(double) * opt.dynamic_size);
    opt.static_data = candy::OptmzStatic(std::in_place_type<candy::optmz::AdaDelta>,
                                         candy::optmz::AdaDelta(learning_rate, rho, bias));
    return opt;
}

// Create an optimizer with rmsprop algorithm
candy::Optimizer candy::create_rmsprop(double learning_rate, double beta, const candy::Model & model, double bias) {
    // check argument
    check_learning_rate(learning_rate);
    check_bias(bias);
    check_weight(beta);
    // construct optimizer
    candy::Optimizer opt;
    opt.dynamic_size = model.num_params();
    opt.dynamic_data = new double[opt.dynamic_size];
    std::memset(opt.dynamic_data, 0, sizeof(double) * opt.dynamic_size);
    opt.static_data = candy::OptmzStatic(std::in_place_type<candy::optmz::RmsProp>,
                                         candy::optmz::RmsProp(learning_rate, beta, bias));
    return opt;
}

}  // namespace merlin

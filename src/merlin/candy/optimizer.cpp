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
static_data_(src.static_data_), dynamic_size_(src.dynamic_size_) {
    // copy dynamic data
    if (src.dynamic_size_ != 0) {
        this->dynamic_data_ = new double[src.dynamic_size_];
        std::memcpy(this->dynamic_data_, src.dynamic_data_, sizeof(double) * src.dynamic_size_);
    }
}

// Copy assignment
candy::Optimizer & candy::Optimizer::operator=(const candy::Optimizer & src) {
    this->static_data_ = src.static_data_;
    this->dynamic_size_ = src.dynamic_size_;
    // release old dynamic data
    if (this->dynamic_data_ != nullptr) {
        delete[] this->dynamic_data_;
    }
    // copy dynamic data
    if (src.dynamic_size_ != 0) {
        this->dynamic_data_ = new double[src.dynamic_size_];
        std::memcpy(this->dynamic_data_, src.dynamic_data_, sizeof(double) * src.dynamic_size_);
    }
    return *this;
}

// Allocate dynamic data
void candy::Optimizer::allocate_data(std::uint64_t size) {
    if (this->dynamic_data_ != nullptr) {
        delete[] this->dynamic_data_;
    }
    this->dynamic_size_ = size;
    this->dynamic_data_ = new double[size];
    std::memset(this->dynamic_data_, 0, sizeof(double) * size);
}

// Check compatibility with a model
bool candy::Optimizer::is_compatible(std::uint64_t num_params) const {
    switch (this->static_data_.index()) {
        case 0 : {  // gradient descent
            break;
        }
        case 1 :
        case 4 : {  // adagrad, rmsprop
            if (this->dynamic_size_ != num_params) {
                return false;
            }
            break;
        }
        case 2 :
        case 3 : {  // adam, adadelta
            if (this->dynamic_size_ != 2 * num_params) {
                return false;
            }
            break;
        }
    }
    return true;
}

// Update model inside a CPU parallel region
void candy::Optimizer::update_cpu(candy::Model & model, const candy::Gradient & grad,
                                  std::uint64_t time_step) noexcept {
    static std::array<candy::OptmzUpdaterCpu, 5> cpu_updater_func = {
        candy::optmz::GradDescent::update_cpu,  // grad descent
        candy::optmz::AdaGrad::update_cpu,      // adagrad
        candy::optmz::Adam::update_cpu,         // adam
        candy::optmz::AdaDelta::update_cpu,     // adadelta
        candy::optmz::RmsProp::update_cpu,      // rmsprop
    };
    void * optimizer_algor = reinterpret_cast<void *>(&this->static_data_);
    cpu_updater_func[this->static_data_.index()](optimizer_algor, this->dynamic_data_, model, grad, time_step);
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
    switch (this->static_data_.index()) {
        case 0 : {  // gradient descent
            os << "type=GradDescent, ";
            const candy::optmz::GradDescent & algor = std::get<0>(this->static_data_);
            os << "eta=" << algor.learning_rate;
            break;
        }
        case 1 : {  // adagrad
            os << "type=AdaGrad, ";
            const candy::optmz::AdaGrad & algor = std::get<1>(this->static_data_);
            os << "eta=" << algor.learning_rate << ", ";
            os << "bias=" << algor.bias << ", ";
            print_dynamic_data(os, this->dynamic_data_, this->dynamic_size_);
            break;
        }
        case 2 : {  // adam
            os << "type=Adam, ";
            const candy::optmz::Adam & algor = std::get<2>(this->static_data_);
            os << "eta=" << algor.learning_rate << ", ";
            os << "beta_m=" << algor.beta_m << ", ";
            os << "beta_v=" << algor.beta_v << ", ";
            os << "bias=" << algor.bias << ", ";
            print_dynamic_data(os, this->dynamic_data_, this->dynamic_size_);
            break;
        }
        case 3 : {  // adadelta
            os << "type=AdaDelta, ";
            const candy::optmz::AdaDelta & algor = std::get<3>(this->static_data_);
            os << "eta=" << algor.learning_rate << ", ";
            os << "rho=" << algor.rho << ", ";
            os << "bias=" << algor.bias << ", ";
            print_dynamic_data(os, this->dynamic_data_, this->dynamic_size_);
            break;
        }
        case 4 : {  // rmsprop
            os << "type=RmsProp, ";
            const candy::optmz::RmsProp & algor = std::get<4>(this->static_data_);
            os << "eta=" << algor.learning_rate << ", ";
            os << "beta=" << algor.beta << ", ";
            os << "bias=" << algor.bias << ", ";
            print_dynamic_data(os, this->dynamic_data_, this->dynamic_size_);
            break;
        }
    }
    os << ")>";
    return os.str();
}

// Destructor
candy::Optimizer::~Optimizer(void) {
    if (this->dynamic_data_ != nullptr) {
        delete[] this->dynamic_data_;
    }
}

}  // namespace merlin

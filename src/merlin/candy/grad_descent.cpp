// Copyright 2023 quocdang1998
#include "merlin/candy/grad_descent.hpp"

#include <cmath>  // std::sqrt, std::pow

#include <omp.h>  // #pragma omp

#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// GradDescent
// --------------------------------------------------------------------------------------------------------------------

// Update model by gradient
void candy::GradDescent::update_cpu(candy::Model & model, const floatvec & gradient) {
    // check size
    std::uint64_t size = model.size();
    if (size != gradient.size()) {
        FAILURE(std::invalid_argument, "Size of model and gradient vector are not equal.\n");
    }
    // update
    #pragma omp parallel for
    for (std::int64_t i_param = 0; i_param < size; i_param++) {
        auto [param_dim, param_index] = model.convert_contiguous(i_param);
        double & param_value = model.parameters()[param_dim][param_index];
        param_value -= this->learning_rate_ * gradient[i_param];
    }
}

#ifndef __MERLIN_CUDA__

// Create an object on GPU by the GPU
candy::GradDescent * candy::GradDescent::create_object_on_gpu(double learning_rate, std::uintptr_t stream_ptr) {
    FAILURE(cuda_compile_error, "Enable CUDA option to use this function.\n");
}

#endif  // __MERLIN_CUDA__

// Destructor
// candy::GradDescent::~GradDescent(void) {}

#ifdef __comment

// --------------------------------------------------------------------------------------------------------------------
// AdaGrad
// --------------------------------------------------------------------------------------------------------------------

// Update model by gradient
void candy::AdaGrad::update_cpu(candy::Model & model, const floatvec & gradient) {
    // check size
    std::uint64_t size = model.size();
    if (size != gradient.size()) {
        FAILURE(std::invalid_argument, "Size of model and gradient vector are not equal.\n");
    }
    // initialize gradient norm
    if (this->cumulative_gradient_norm_.size() == 0) {
        this->cumulative_gradient_norm_ = floatvec(size, 0.0);
    } else if (this->cumulative_gradient_norm_.size() != size) {
        FAILURE(std::invalid_argument, "A model other than the first one has been provided.\n");
    }
    // update
    #pragma omp parallel for
    for (std::int64_t i_param = 0; i_param < size; i_param++) {
        auto [param_dim, param_index] = model.convert_contiguous(i_param);
        double & param_value = model.parameters()[param_dim][param_index];
        this->cumulative_gradient_norm_[i_param] += gradient[i_param] * gradient[i_param];
        double correction = this->learning_rate_ * gradient[i_param];
        correction /= std::sqrt(this->cumulative_gradient_norm_[i_param] + this->bias_);
        param_value -= correction;
    }
}

// Destructor
candy::AdaGrad::~AdaGrad(void) {}

// --------------------------------------------------------------------------------------------------------------------
// Adam
// --------------------------------------------------------------------------------------------------------------------

// Constructor from value
candy::Adam::Adam(double learning_rate, double beta_m, double beta_v, double bias) : learning_rate_(learning_rate),
beta_m_(beta_m), beta_v_(beta_v), bias_(bias) {
    if ((beta_m < 0.0) || (beta_m >= 1.0)) {
        FAILURE(std::invalid_argument, "Expected first moment decay constant to be positive and smaller than 1.0\n");
    }
    if ((beta_v < 0.0) || (beta_v >= 1.0)) {
        FAILURE(std::invalid_argument, "Expected first moment decay constant to be positive and smaller than 1.0\n");
    }
}

// Update model by gradient
void candy::Adam::update_cpu(candy::Model & model, const floatvec & gradient) {
    // check size
    std::uint64_t size = model.size();
    if (size != gradient.size()) {
        FAILURE(std::invalid_argument, "Size of model and gradient vector are not equal.\n");
    }
    // initialize first and second moment vector
    if (this->register_moments_.size() == 0) {
        this->register_moments_ = floatvec(2*size, 0.0);
    } else if (this->register_moments_.size() != 2*size) {
        FAILURE(std::invalid_argument, "A model other than the first one has been provided.\n");
    }
    // update
    this->time_step_ += 1;
    #pragma omp parallel for
    for (std::int64_t i_param = 0; i_param < size; i_param++) {
        // calculate first and second moment
        this->register_moments_[2*i_param] *= this->beta_m_;
        this->register_moments_[2*i_param] += (1.0 - this->beta_m_) *  gradient[i_param];
        this->register_moments_[2*i_param+1] *= this->beta_v_;
        this->register_moments_[2*i_param+1] += (1.0 - this->beta_v_) * gradient[i_param] * gradient[i_param];
        // update parameters
        auto [param_dim, param_index] = model.convert_contiguous(i_param);
        double & param_value = model.parameters()[param_dim][param_index];
        double corrected_first_moment = this->register_moments_[2*i_param];
        corrected_first_moment /= 1.0 - std::pow(this->beta_m_, this->time_step_);
        double corrected_second_moment = this->register_moments_[2*i_param+1];
        corrected_second_moment  /= 1.0 - std::pow(this->beta_v_, this->time_step_);
        double correction = this->learning_rate_ * corrected_first_moment;
        correction /= std::sqrt(corrected_second_moment) + this->bias_;
        param_value -= correction;
    }
}

// Destructor
candy::Adam::~Adam(void) {}

#endif // __comment

}  // namespace merlin

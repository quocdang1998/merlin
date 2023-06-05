// Copyright 2023 quocdang1998
#include "merlin/candy/adagrad.hpp"

#include <cmath>  // std::sqrt, std::pow

#include <omp.h>  // #pragma omp

#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

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
// candy::AdaGrad::~AdaGrad(void) {}

}  // namespace merlin

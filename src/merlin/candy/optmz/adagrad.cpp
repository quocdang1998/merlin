// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/adagrad.hpp"

#include <cmath>    // std::sqrt
#include <cstring>  // std::memset

#include <omp.h>  // #pragma omp

#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/logger.hpp"       // FAILURE

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// AdaGrad
// ---------------------------------------------------------------------------------------------------------------------

// Erase train history
void candy::optmz::AdaGrad::erase_history(void) noexcept {
    std::memset(this->grad_history_.data(), 0, sizeof(double) * this->grad_history_.size());
}

// Update model by gradient
void candy::optmz::AdaGrad::update_cpu(candy::Model & model, floatvec & gradient, std::uint64_t n_thread) noexcept {
    // update
    #pragma omp parallel for num_threads(n_thread)
    for (std::int64_t i_param = 0; i_param < this->grad_history_.size(); i_param++) {
        double & param = model[i_param];
        this->grad_history_[i_param] += gradient[i_param] * gradient[i_param];
        double correction = this->learning_rate_ * gradient[i_param];
        correction /= std::sqrt(this->grad_history_[i_param] + this->bias_);
        param -= correction;
    }
}

}  // namespace merlin

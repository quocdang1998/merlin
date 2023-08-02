// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/adam.hpp"

#include <cmath>    // std::pow
#include <cstring>  // std::memset

#include <omp.h>  // #pragma omp

#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/logger.hpp"       // FAILURE

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------------------------------------------------

// Erase train history
void candy::optmz::Adam::erase_history(void) noexcept {
    std::memset(this->register_moments_.data(), 0, sizeof(double) * this->register_moments_.size());
}

// Update model by gradient
void candy::optmz::Adam::update_cpu(candy::Model & model, floatvec & gradient, std::uint64_t n_thread) noexcept {
    // update time step
    this->time_step_ += 1;
    #pragma omp parallel for num_threads(n_thread)
    for (std::int64_t i_param = 0; i_param < gradient.size(); i_param++) {
        // calculate first and second moment
        this->register_moments_[2 * i_param] *= this->beta_m_;
        this->register_moments_[2 * i_param] += (1.0 - this->beta_m_) * gradient[i_param];
        this->register_moments_[2 * i_param + 1] *= this->beta_v_;
        this->register_moments_[2 * i_param + 1] += (1.0 - this->beta_v_) * gradient[i_param] * gradient[i_param];
        // update parameters
        double & param = model[i_param];
        double corrected_first_moment = this->register_moments_[2 * i_param];
        corrected_first_moment /= 1.0 - std::pow(this->beta_m_, this->time_step_);
        double corrected_second_moment = this->register_moments_[2 * i_param + 1];
        corrected_second_moment /= 1.0 - std::pow(this->beta_v_, this->time_step_);
        double correction = this->learning_rate_ * corrected_first_moment;
        correction /= std::sqrt(corrected_second_moment) + this->bias_;
        param -= correction;
    }
}

}  // namespace merlin

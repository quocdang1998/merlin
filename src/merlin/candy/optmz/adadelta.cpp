// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/adadelta.hpp"

#include <cmath>  // std::sqrt

#include <omp.h>  // #pragma omp

#include "merlin/candy/gradient.hpp"   // merlin::candy::Gradient
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::OptmzStatic

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// AdaDelta
// ---------------------------------------------------------------------------------------------------------------------

// Update model inside a CPU parallel region
void candy::optmz::AdaDelta::update_cpu(void * optimizer_algor, candy::Model & model, const candy::Gradient & grad,
                                        std::uint64_t thread_idx, std::uint64_t n_threads) noexcept {
    candy::OptmzStatic & union_algor = *(reinterpret_cast<candy::OptmzStatic *>(optimizer_algor));
    candy::optmz::AdaDelta & algor = std::get<candy::optmz::AdaDelta>(union_algor);
    for (std::uint64_t i_param = thread_idx; i_param < model.num_params(); i_param += n_threads) {
        // calculate rms
        double rms = algor.rms_delta[2 * i_param];
        double delta = algor.rms_delta[2 * i_param + 1];
        rms = rms * algor.rho + (1.0 - algor.rho) * grad.value()[i_param] * grad.value()[i_param];
        // calculate rescaled gradient
        double rescaled_grad = (std::sqrt(delta + algor.bias) * grad.value()[i_param]) / std::sqrt(rms + algor.bias);
        // update delta
        delta = delta * algor.rho + (1.0 - algor.rho) * rescaled_grad * rescaled_grad;
        model[i_param] -= algor.learning_rate * rescaled_grad;
        algor.rms_delta[2 * i_param] = rms;
        algor.rms_delta[2 * i_param + 1] = delta;
    }
    _Pragma("omp barrier")
}

}  // namespace merlin

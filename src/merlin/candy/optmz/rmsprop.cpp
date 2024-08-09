// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/rmsprop.hpp"

#include <cmath>  // std::sqrt

#include "merlin/candy/gradient.hpp"   // merlin::candy::Gradient
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::OptmzStatic

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// RmsProp
// ---------------------------------------------------------------------------------------------------------------------

// Update model inside a CPU parallel region
void candy::optmz::RmsProp::update_cpu(void * optimizer_algor, double * history, candy::Model & model,
                                       const candy::Gradient & grad, std::uint64_t time_step, std::uint64_t thread_idx,
                                       std::uint64_t n_threads) noexcept {
    candy::OptmzStatic & union_algor = *(reinterpret_cast<candy::OptmzStatic *>(optimizer_algor));
    candy::optmz::RmsProp & algor = std::get<candy::optmz::RmsProp>(union_algor);
    for (std::uint64_t i_param = thread_idx; i_param < model.num_params(); i_param += n_threads) {
        // update root mean square
        double rms = history[i_param];
        rms = algor.beta * rms + grad.value()[i_param] * grad.value()[i_param];
        history[i_param] = rms;
        // update parameter
        double correction = algor.learning_rate * grad.value()[i_param];
        correction /= std::sqrt(algor.bias + rms);
        model[i_param] -= correction;
    }
}

}  // namespace merlin

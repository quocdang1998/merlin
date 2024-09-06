// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/rmsprop.hpp"

#include <cmath>  // std::sqrt

#include "merlin/candy/gradient.hpp"   // merlin::candy::Gradient
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::OptmzStatic
#include "merlin/logger.hpp"           // merlin::Fatal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// RmsProp
// ---------------------------------------------------------------------------------------------------------------------

// Update model inside a CPU parallel region
void candy::optmz::RmsProp::update_cpu(void * optimizer_algor, double * history, candy::Model & model,
                                       const candy::Gradient & grad, std::uint64_t time_step) noexcept {
    candy::OptmzStatic & union_algor = *(reinterpret_cast<candy::OptmzStatic *>(optimizer_algor));
    candy::optmz::RmsProp & algor = std::get<candy::optmz::RmsProp>(union_algor);
    for (std::uint64_t i_param = 0; i_param < model.num_params(); i_param++) {
        // update root mean square
        double rms = history[i_param];
        rms = algor.beta * rms + (1.0 - algor.beta) * grad.value()[i_param] * grad.value()[i_param];
        history[i_param] = rms;
        // update parameter
        double correction = algor.learning_rate * grad.value()[i_param];
        correction /= std::sqrt(algor.bias + rms);
        model[i_param] -= correction;
    }
}

// Create an optimizer with rmsprop algorithm
candy::Optimizer candy::optmz::create_rmsprop(double learning_rate, double beta, std::uint64_t num_params,
                                              double bias) {
    // check argument
    if (learning_rate <= 0) {
        Fatal<std::invalid_argument>("Learning rate must be positive.\n");
    }
    if (bias < 0) {
        Fatal<std::invalid_argument>("Bias must be positive.\n");
    }
    if (beta * (beta - 1.0) > 0) {
        Fatal<std::invalid_argument>("Weight value must be in range [0.0, 1.0].\n");
    }
    // construct optimizer
    candy::Optimizer opt;
    opt.allocate_data(num_params);
    opt.static_data() = candy::OptmzStatic(std::in_place_type<candy::optmz::RmsProp>,
                                           candy::optmz::RmsProp(learning_rate, beta, bias));
    return opt;
}

}  // namespace merlin

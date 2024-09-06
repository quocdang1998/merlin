// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/adadelta.hpp"

#include <cmath>  // std::sqrt

#include "merlin/candy/gradient.hpp"   // merlin::candy::Gradient
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::OptmzStatic
#include "merlin/logger.hpp"           // merlin::Fatal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// AdaDelta
// ---------------------------------------------------------------------------------------------------------------------

// Update model inside a CPU parallel region
void candy::optmz::AdaDelta::update_cpu(void * optimizer_algor, double * history, candy::Model & model,
                                        const candy::Gradient & grad, std::uint64_t time_step) noexcept {
    candy::OptmzStatic & union_algor = *(reinterpret_cast<candy::OptmzStatic *>(optimizer_algor));
    candy::optmz::AdaDelta & algor = std::get<candy::optmz::AdaDelta>(union_algor);
    for (std::uint64_t i_param = 0; i_param < model.num_params(); i_param++) {
        // calculate rms
        double rms = history[2 * i_param];
        double delta = history[2 * i_param + 1];
        rms = rms * algor.rho + (1.0 - algor.rho) * grad.value()[i_param] * grad.value()[i_param];
        // calculate rescaled gradient
        double rescaled_grad = (std::sqrt(delta + algor.bias) * grad.value()[i_param]) / std::sqrt(rms + algor.bias);
        // update delta
        delta = delta * algor.rho + (1.0 - algor.rho) * rescaled_grad * rescaled_grad;
        model[i_param] -= algor.learning_rate * rescaled_grad;
        history[2 * i_param] = rms;
        history[2 * i_param + 1] = delta;
    }
}

// Create an optimizer with adadelta algorithm
candy::Optimizer candy::optmz::create_adadelta(double learning_rate, double rho, std::uint64_t num_params,
                                               double bias) {
    // check argument
    if (learning_rate <= 0) {
        Fatal<std::invalid_argument>("Learning rate must be positive.\n");
    }
    if (bias < 0) {
        Fatal<std::invalid_argument>("Bias must be positive.\n");
    }
    if (rho * (rho - 1.0) > 0) {
        Fatal<std::invalid_argument>("Weight value must be in range [0.0, 1.0].\n");
    }
    // construct optimizer
    candy::Optimizer opt;
    opt.allocate_data(2 * num_params);
    opt.static_data() = candy::OptmzStatic(std::in_place_type<candy::optmz::AdaDelta>,
                                           candy::optmz::AdaDelta(learning_rate, rho, bias));
    return opt;
}

}  // namespace merlin

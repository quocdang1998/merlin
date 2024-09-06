// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/adagrad.hpp"

#include <cmath>  // std::sqrt

#include "merlin/candy/gradient.hpp"   // merlin::candy::Gradient
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::OptmzStatic
#include "merlin/logger.hpp"           // merlin::Fatal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// AdaGrad
// ---------------------------------------------------------------------------------------------------------------------

// Update model inside a CPU parallel region
void candy::optmz::AdaGrad::update_cpu(void * optimizer_algor, double * history, candy::Model & model,
                                       const candy::Gradient & grad, std::uint64_t time_step) noexcept {
    candy::OptmzStatic & union_algor = *(reinterpret_cast<candy::OptmzStatic *>(optimizer_algor));
    candy::optmz::AdaGrad & algor = std::get<candy::optmz::AdaGrad>(union_algor);
    for (std::uint64_t i_param = 0; i_param < model.num_params(); i_param++) {
        // copy gradient history to thread register and copy it back
        double grad_history = history[i_param];
        grad_history += grad.value()[i_param] * grad.value()[i_param];
        history[i_param] = grad_history;
        // update parameter
        double correction = algor.learning_rate * grad.value()[i_param];
        correction /= std::sqrt(grad_history + algor.bias);
        model[i_param] -= correction;
    }
}

// Create an optimizer with adagrad algorithm
candy::Optimizer candy::optmz::create_adagrad(double learning_rate, std::uint64_t num_params, double bias) {
    // check argument
    if (learning_rate <= 0) {
        Fatal<std::invalid_argument>("Learning rate must be positive.\n");
    }
    if (bias < 0) {
        Fatal<std::invalid_argument>("Bias must be positive.\n");
    }
    // construct optimizer
    candy::Optimizer opt;
    opt.allocate_data(num_params);
    opt.static_data() = candy::OptmzStatic(std::in_place_type<candy::optmz::AdaGrad>,
                                           candy::optmz::AdaGrad(learning_rate, bias));
    return opt;
}

}  // namespace merlin

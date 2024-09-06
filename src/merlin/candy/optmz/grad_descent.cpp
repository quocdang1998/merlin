// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/grad_descent.hpp"

#include "merlin/candy/gradient.hpp"   // merlin::candy::Gradient
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::OptmzStatic
#include "merlin/logger.hpp"           // merlin::Fatal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// GradDescent
// ---------------------------------------------------------------------------------------------------------------------

// Update model inside a CPU parallel region
void candy::optmz::GradDescent::update_cpu(void * optimizer_algor, double * history, candy::Model & model,
                                           const candy::Gradient & grad, std::uint64_t time_step) noexcept {
    candy::OptmzStatic & union_algor = *(reinterpret_cast<candy::OptmzStatic *>(optimizer_algor));
    candy::optmz::GradDescent & algor = std::get<candy::optmz::GradDescent>(union_algor);
    for (std::uint64_t i_param = 0; i_param < model.num_params(); i_param++) {
        model[i_param] -= algor.learning_rate * grad.value()[i_param];
    }
}

// Create an optimizer with gradient descent algorithm
candy::Optimizer candy::optmz::create_grad_descent(double learning_rate) {
    // check argument
    if (learning_rate <= 0) {
        Fatal<std::invalid_argument>("Learning rate must be positive.\n");
    }
    // construct optimizer
    candy::Optimizer opt;
    opt.static_data() = candy::OptmzStatic(std::in_place_type<candy::optmz::GradDescent>,
                                           candy::optmz::GradDescent(learning_rate));
    return opt;
}

}  // namespace merlin

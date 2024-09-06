// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/adam.hpp"

#include <cmath>  // std::pow, std::sqrt

#include "merlin/candy/gradient.hpp"   // merlin::candy::Gradient
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::OptmzStatic
#include "merlin/logger.hpp"           // merlin::Fatal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------------------------------------------------

// Update model inside a CPU parallel region
void candy::optmz::Adam::update_cpu(void * optimizer_algor, double * history, candy::Model & model,
                                    const candy::Gradient & grad, std::uint64_t time_step) noexcept {
    candy::OptmzStatic & union_algor = *(reinterpret_cast<candy::OptmzStatic *>(optimizer_algor));
    candy::optmz::Adam & algor = std::get<candy::optmz::Adam>(union_algor);
    for (std::uint64_t i_param = 0; i_param < model.num_params(); i_param++) {
        // calculate first and second moment and save it back
        double first_moment = history[2 * i_param];
        double second_moment = history[2 * i_param + 1];
        first_moment *= algor.beta_m;
        first_moment += (1.0 - algor.beta_m) * grad.value()[i_param];
        history[2 * i_param] = first_moment;
        second_moment *= algor.beta_v;
        second_moment += (1.0 - algor.beta_v) * grad.value()[i_param] * grad.value()[i_param];
        history[2 * i_param + 1] = second_moment;
        // update parameters
        first_moment /= 1.0 - std::pow(algor.beta_m, time_step);
        second_moment /= 1.0 - std::pow(algor.beta_v, time_step);
        double correction = algor.learning_rate * first_moment;
        correction /= std::sqrt(second_moment) + algor.bias;
        model[i_param] -= correction;
    }
}

// Create an optimizer with adam algorithm
candy::Optimizer candy::optmz::create_adam(double learning_rate, double beta_m, double beta_v, std::uint64_t num_params,
                                           double bias) {
    // check argument
    if (learning_rate <= 0) {
        Fatal<std::invalid_argument>("Learning rate must be positive.\n");
    }
    if (bias < 0) {
        Fatal<std::invalid_argument>("Bias must be positive.\n");
    }
    if (beta_m * (beta_m - 1.0) > 0) {
        Fatal<std::invalid_argument>("Weight value must be in range [0.0, 1.0].\n");
    }
    if (beta_v * (beta_v - 1.0) > 0) {
        Fatal<std::invalid_argument>("Weight value must be in range [0.0, 1.0].\n");
    }
    // construct optimizer
    candy::Optimizer opt;
    opt.allocate_data(2 * num_params);
    opt.static_data() = candy::OptmzStatic(std::in_place_type<candy::optmz::Adam>,
                                           candy::optmz::Adam(learning_rate, beta_m, beta_v, bias));
    return opt;
}

}  // namespace merlin

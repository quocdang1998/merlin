// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/adam.hpp"

#include <cmath>  // std::pow, std::sqrt

#include <omp.h>  // #pragma omp

#include "merlin/candy/gradient.hpp"   // merlin::candy::Gradient
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::OptmzStatic

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------------------------------------------------

// Update model inside a CPU parallel region
void candy::optmz::Adam::update_cpu(void * optimizer_algor, candy::Model & model, const candy::Gradient & grad,
                                    std::uint64_t thread_idx, std::uint64_t n_threads) noexcept {
    candy::OptmzStatic & union_algor = *(reinterpret_cast<candy::OptmzStatic *>(optimizer_algor));
    candy::optmz::Adam & algor = std::get<candy::optmz::Adam>(union_algor);
    if (thread_idx == 0) {
        algor.time_step += 1;
    }
    _Pragma("omp barrier");
    for (std::uint64_t i_param = thread_idx; i_param < model.num_params(); i_param += n_threads) {
        // calculate first and second moment and save it back
        double first_moment = algor.moments[2 * i_param];
        double second_moment = algor.moments[2 * i_param + 1];
        first_moment *= algor.beta_m;
        first_moment += (1.0 - algor.beta_m) * grad.value()[i_param];
        algor.moments[2 * i_param] = first_moment;
        second_moment *= algor.beta_v;
        second_moment += (1.0 - algor.beta_v) * grad.value()[i_param] * grad.value()[i_param];
        algor.moments[2 * i_param + 1] = second_moment;
        // update parameters
        first_moment /= 1.0 - std::pow(algor.beta_m, algor.time_step);
        second_moment /= 1.0 - std::pow(algor.beta_v, algor.time_step);
        double correction = algor.learning_rate * first_moment;
        correction /= std::sqrt(second_moment) + algor.bias;
        model[i_param] -= correction;
    }
    _Pragma("omp barrier");
}

}  // namespace merlin

// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/adagrad.hpp"

#include <cmath>  // std::sqrt

#include <omp.h>  // #pragma omp

#include "merlin/candy/gradient.hpp"   // merlin::candy::Gradient
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::OptmzStatic

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// AdaGrad
// ---------------------------------------------------------------------------------------------------------------------

// Update model inside a CPU parallel region
void candy::optmz::AdaGrad::update_cpu(void * optimizer_algor, candy::Model & model, const candy::Gradient & grad,
                                       std::uint64_t thread_idx, std::uint64_t n_threads) noexcept {
    candy::OptmzStatic & union_algor = *(reinterpret_cast<candy::OptmzStatic *>(optimizer_algor));
    candy::optmz::AdaGrad & algor = std::get<candy::optmz::AdaGrad>(union_algor);
    for (std::uint64_t i_param = thread_idx; i_param < model.num_params(); i_param += n_threads) {
        // copy gradient history to thread register and copy it back
        double grad_history = algor.grad_history[i_param];
        grad_history += grad.value()[i_param] * grad.value()[i_param];
        algor.grad_history[i_param] = grad_history;
        // update parameter
        double correction = algor.learning_rate * grad.value()[i_param];
        correction /= std::sqrt(grad_history + algor.bias);
        model[i_param] -= correction;
    }
    _Pragma("omp barrier")
}

}  // namespace merlin

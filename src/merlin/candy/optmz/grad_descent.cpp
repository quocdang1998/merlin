// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/grad_descent.hpp"

#include <omp.h>  // #pragma omp

#include "merlin/candy/gradient.hpp"   // merlin::candy::Gradient
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::OptmzStatic

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// GradDescent
// ---------------------------------------------------------------------------------------------------------------------

// Update model inside a CPU parallel region
void candy::optmz::GradDescent::update_cpu(void * optimizer_algor, candy::Model & model, const candy::Gradient & grad,
                                           std::uint64_t thread_idx, std::uint64_t n_threads) noexcept {
    candy::OptmzStatic & union_algor = *(reinterpret_cast<candy::OptmzStatic *>(optimizer_algor));
    candy::optmz::GradDescent & algor = std::get<candy::optmz::GradDescent>(union_algor);
    for (std::uint64_t i_param = thread_idx; i_param < model.num_params(); i_param += n_threads) {
        model[i_param] -= algor.learning_rate * grad.value()[i_param];
    }
    #pragma omp barrier
}

}  // namespace merlin

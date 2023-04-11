// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/grad_descent.hpp"

#include <omp.h>  // #pragma omp

#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/logger.hpp"       // merlin::cuda_compile_error, FAILURE

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// GradDescent
// ---------------------------------------------------------------------------------------------------------------------

// Update model by gradient
void candy::optmz::GradDescent::update_cpu(candy::Model & model, floatvec & gradient, std::uint64_t n_thread) noexcept {
    #pragma omp parallel for num_threads(n_thread)
    for (std::int64_t i_param = 0; i_param < gradient.size(); i_param++) {
        double & param = model[i_param];
        param -= this->learning_rate_ * gradient[i_param];
    }
}

#ifndef __MERLIN_CUDA__

// Create an object on GPU by the GPU
candy::Optimizer * candy::optmz::GradDescent::new_gpu(void) const {
    FAILURE(cuda_compile_error, "Compile with CUDA option enabled to access GPU features.\n");
    return nullptr;
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin

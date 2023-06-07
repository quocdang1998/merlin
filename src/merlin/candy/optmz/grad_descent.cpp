// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/grad_descent.hpp"

#include <omp.h>  // #pragma omp

#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// GradDescent
// --------------------------------------------------------------------------------------------------------------------

// Update model by gradient
void candy::optmz::GradDescent::update_cpu(candy::Model & model, double gradient, std::uint64_t i_param,
                                           std::uint64_t param_dim, std::uint64_t param_index,
                                           std::uint64_t param_rank) {
    double & param_value = model.parameters()[param_dim][param_index*model.rank() + param_rank];
    param_value -= this->learning_rate_ * gradient;
}

#ifndef __MERLIN_CUDA__

// Create an object on GPU by the GPU
candy::optmz::GradDescent * candy::optmz::GradDescent::create_object_on_gpu(double learning_rate,
                                                                            std::uintptr_t stream_ptr) {
    FAILURE(cuda_compile_error, "Enable CUDA option to use this function.\n");
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin

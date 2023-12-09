// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/adagrad.hpp"

#include <cmath>  // std::sqrt

#include <omp.h>  // #pragma omp

#include "merlin/candy/model.hpp"     // merlin::candy::Model
#include "merlin/candy/gradient.hpp"  // merlin::candy::Gradient
#include "merlin/logger.hpp"          // FAILURE, cuda_compile_error

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// AdaGrad
// ---------------------------------------------------------------------------------------------------------------------

// Update model inside a CPU parallel region
void candy::optmz::AdaGrad::update_cpu(candy::Model & model, const candy::Gradient & grad, std::uint64_t thread_idx,
                                       std::uint64_t n_threads) noexcept {
    for (std::uint64_t i_param = thread_idx; i_param < this->grad_history.size(); i_param += n_threads) {
        double & param = model[i_param];
        this->grad_history[i_param] += grad.value()[i_param] * grad.value()[i_param];
        double correction = this->learning_rate * grad.value()[i_param];
        correction /= std::sqrt(this->grad_history[i_param] + this->bias);
        param -= correction;
    }
    #pragma omp barrier
}

#ifndef __MERLIN_CUDA__

// Copy the optimizer from CPU to a pre-allocated memory on GPU
void * candy::optmz::AdaGrad::copy_to_gpu(candy::optmz::AdaGrad * gpu_ptr, void * dynamic_data_ptr,
                                          std::uintptr_t stream_ptr) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin

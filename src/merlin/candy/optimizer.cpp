// Copyright 2023 quocdang1998
#include "merlin/candy/optimizer.hpp"

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------------------------------------------------

// Update model inside a CPU parallel region
void candy::Optimizer::update_cpu(candy::Model & model, const candy::Gradient & grad, std::uint64_t thread_idx,
                                  std::uint64_t n_threads) noexcept {
    switch (this->algorithm) {
        case candy::OptAlgorithm::GdAlgo : {  // gradient descent
            candy::optmz::GradDescent & optimizer = std::get<candy::optmz::GradDescent>(this->static_data);
            optimizer.update_cpu(model, grad, thread_idx, n_threads);
            break;
        }
    }
}

// Calculate additional number of bytes to allocate for dynamic data
std::uint64_t candy::Optimizer::cumalloc_size(void) const noexcept {
    std::uint64_t static_size = sizeof(candy::Optimizer);
    std::uint64_t dynamic_size = 0;
    switch (this->algorithm) {
        case candy::OptAlgorithm::GdAlgo : {  // gradient descent
            const candy::optmz::GradDescent & optimizer = std::get<candy::optmz::GradDescent>(this->static_data);
            dynamic_size = optimizer.additional_cumalloc();
            break;
        }
    }
    return static_size + dynamic_size;
}

#ifndef __MERLIN_CUDA__

// Copy the optimizer from CPU to a pre-allocated memory on GPU
void * candy::Optimizer::copy_to_gpu(candy::Optimizer * gpu_ptr, void * dynamic_data_ptr,
                                     std::uintptr_t stream_ptr) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

#endif  // __MERLIN_CUDA__

// Calculate additional number of bytes to allocate in CUDA shared memory for dynamic data
std::uint64_t candy::Optimizer::sharedmem_size(void) const noexcept {
    std::uint64_t static_size = sizeof(candy::Optimizer);
    std::uint64_t dynamic_size = 0;
    switch (this->algorithm) {
        case candy::OptAlgorithm::GdAlgo : {  // gradient descent
            const candy::optmz::GradDescent & optimizer = std::get<candy::optmz::GradDescent>(this->static_data);
            dynamic_size = optimizer.additional_sharedmem();
            break;
        }
    }
    return static_size + dynamic_size;
}

// Destructor
candy::Optimizer::~Optimizer(void) {
    if (this->dynamic_data == nullptr) {
        delete[] this->dynamic_data;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------------------------------------------------

// Create an optimizer with gradient descent algorithm
candy::Optimizer candy::create_grad_descent(double learning_rate) {
    candy::Optimizer opt;
    opt.algorithm = candy::OptAlgorithm::GdAlgo;
    candy::optmz::GradDescent optimizer(learning_rate);
    opt.static_data = candy::OptmzStatic(std::in_place_type<candy::optmz::GradDescent>, optimizer);
    return opt;
}

}  // namespace merlin

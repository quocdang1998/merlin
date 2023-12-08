// Copyright 2023 quocdang1998
#include "merlin/candy/optimizer.hpp"

#include <cstddef>  // offsetof

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------------------------------------------------

// Copy the optimizer from CPU to a pre-allocated memory on GPU
void * candy::Optimizer::copy_to_gpu(candy::Optimizer * gpu_ptr, void * dynamic_data_ptr,
                                     std::uintptr_t stream_ptr) const {
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    ::cudaMemcpyAsync(gpu_ptr, this, sizeof(candy::Optimizer), ::cudaMemcpyHostToDevice, stream);
    std::uintptr_t static_offset = offsetof(candy::Optimizer, static_data);
    void * return_ptr;
    switch (this->algorithm) {
        case candy::OptAlgorithm::GdAlgo : {  // gradient descent
            const candy::optmz::GradDescent & optimizer = std::get<candy::optmz::GradDescent>(this->static_data);
            std::uintptr_t gpu_optimizer_ptr = reinterpret_cast<std::uintptr_t>(gpu_ptr) + static_offset;
            return_ptr = optimizer.copy_to_gpu(reinterpret_cast<candy::optmz::GradDescent *>(gpu_optimizer_ptr),
                                               dynamic_data_ptr, stream_ptr);
            break;
        }
    }
    return return_ptr;
}

}  // namespace merlin

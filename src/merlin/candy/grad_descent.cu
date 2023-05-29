// Copyright 2023 quocdang1998
#include "merlin/candy/grad_descent.hpp"

namespace merlin {

// Copy data from CPU to a pre-allocated memory on GPU
void * candy::GradDescent::copy_to_gpu(candy::Optimizer * gpu_ptr, void * next_ptr, std::uintptr_t stream_ptr) const {
    ::cudaMemcpyAsync(gpu_ptr, this, sizeof(candy::GradDescent), ::cudaMemcpyHostToDevice,
                      reinterpret_cast<::cudaStream_t>(stream_ptr));
    return next_ptr;
}

}  // namespace merlin

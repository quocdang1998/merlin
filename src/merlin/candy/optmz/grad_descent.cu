// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/grad_descent.hpp"

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// GradDescent
// ---------------------------------------------------------------------------------------------------------------------

// Copy the optimizer from CPU to a pre-allocated memory on GPU
void * candy::optmz::GradDescent::copy_to_gpu(candy::optmz::GradDescent * gpu_ptr, void * dynamic_data_ptr,
                                              std::uintptr_t stream_ptr) const {
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    ::cudaMemcpyAsync(gpu_ptr, this, sizeof(candy::optmz::GradDescent), ::cudaMemcpyHostToDevice, stream);
    return dynamic_data_ptr;
}

}  // namespace merlin

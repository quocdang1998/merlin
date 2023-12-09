// Copyright 2023 quocdang1998
#include "merlin/candy/optmz/adagrad.hpp"

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// AdaGrad
// ---------------------------------------------------------------------------------------------------------------------

// Copy the optimizer from CPU to a pre-allocated memory on GPU
void * candy::optmz::AdaGrad::copy_to_gpu(candy::optmz::AdaGrad * gpu_ptr, void * dynamic_data_ptr,
                                          std::uintptr_t stream_ptr) const {
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    // create a clone on GPU
    candy::optmz::AdaGrad clone_on_gpu;
    clone_on_gpu.learning_rate = this->learning_rate;
    clone_on_gpu.bias = this->bias;
    clone_on_gpu.grad_history.data() = reinterpret_cast<double *>(dynamic_data_ptr);
    clone_on_gpu.grad_history.size() = this->grad_history.size();
    ::cudaMemcpyAsync(gpu_ptr, &clone_on_gpu, sizeof(candy::optmz::AdaGrad), ::cudaMemcpyHostToDevice, stream);
    // copy grad history to GPU
    ::cudaMemcpyAsync(dynamic_data_ptr, this->grad_history.data(), sizeof(double) * this->grad_history.size(),
                      ::cudaMemcpyHostToDevice, stream);
    // nullify to avoid seg fault
    clone_on_gpu.grad_history.data() = nullptr;
    return reinterpret_cast<double*>(dynamic_data_ptr) + this->grad_history.size();
}

}  // namespace merlin

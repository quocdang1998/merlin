// Copyright 2023 quocdang1998
#include "merlin/candy/optimizer.hpp"

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------------------------------------------------

// Copy the optimizer from CPU to a pre-allocated memory on GPU
void * candy::Optimizer::copy_to_gpu(candy::Optimizer * gpu_ptr, void * dynamic_data_ptr,
                                     std::uintptr_t stream_ptr) const {
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    // create an instant similar to the copy on GPU
    candy::Optimizer copy_on_gpu;
    copy_on_gpu.static_data = this->static_data;
    copy_on_gpu.dynamic_data = reinterpret_cast<double *>(dynamic_data_ptr);
    copy_on_gpu.dynamic_size = this->dynamic_size;
    // copy the clone and dynamic data to GPU
    ::cudaMemcpyAsync(gpu_ptr, &copy_on_gpu, sizeof(candy::Optimizer), ::cudaMemcpyHostToDevice, stream);
    ::cudaMemcpyAsync(dynamic_data_ptr, this->dynamic_data, sizeof(double) * this->dynamic_size,
                      ::cudaMemcpyHostToDevice, stream);
    // nullify pointer on the clone
    double * returned_ptr = copy_on_gpu.dynamic_data;
    copy_on_gpu.dynamic_data = nullptr;
    return returned_ptr + this->dynamic_size;
}

// Copy data from GPU to CPU
void * candy::Optimizer::copy_from_gpu(double * data_from_gpu, std::uintptr_t stream_ptr) noexcept {
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    ::cudaMemcpyAsync(this->dynamic_data, data_from_gpu, sizeof(double) * this->dynamic_size, ::cudaMemcpyDeviceToHost,
                      stream);
    return reinterpret_cast<void *>(data_from_gpu + this->dynamic_size);
}

}  // namespace merlin

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
    copy_on_gpu.dynamic_data = reinterpret_cast<char *>(dynamic_data_ptr);
    copy_on_gpu.dynamic_size = this->dynamic_size;
    switch (this->static_data.index()) {
        case 0 : {  // gradient descent
            break;
        }
        case 1 : {  // adagrad
            candy::optmz::AdaGrad & opt_algor = std::get<candy::optmz::AdaGrad>(copy_on_gpu.static_data);
            opt_algor.grad_history = reinterpret_cast<double *>(dynamic_data_ptr);
            break;
        }
        case 2 : {  // adam
            candy::optmz::Adam & opt_algor = std::get<candy::optmz::Adam>(copy_on_gpu.static_data);
            opt_algor.moments = reinterpret_cast<double *>(dynamic_data_ptr);
            break;
        }
    }
    // copy the clone and dynamic data to GPU
    ::cudaMemcpyAsync(gpu_ptr, &copy_on_gpu, sizeof(candy::Optimizer), ::cudaMemcpyHostToDevice, stream);
    ::cudaMemcpyAsync(dynamic_data_ptr, this->dynamic_data, this->dynamic_size, ::cudaMemcpyHostToDevice, stream);
    // nullify pointer on the clone
    char * returned_ptr = copy_on_gpu.dynamic_data;
    copy_on_gpu.dynamic_data = nullptr;
    return returned_ptr + this->dynamic_size;
}

}  // namespace merlin

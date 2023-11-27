// Copyright 2023 quocdang1998
#include "merlin/candy/model.hpp"

#include <cstdint>  // std::uintptr_t

namespace merlin {

// Copy data to a pre-allocated memory
void * candy::Model::copy_to_gpu(candy::Model * gpu_ptr, void * parameters_data_ptr, std::uintptr_t stream_ptr) const {
    // initialize buffer to store data of the copy before cloning it to GPU
    candy::Model copy_on_gpu;
    copy_on_gpu.rank_ = this->rank_;
    // shallow copy of parameters
    copy_on_gpu.parameters_.data() = reinterpret_cast<floatvec *>(parameters_data_ptr);
    copy_on_gpu.parameters_.size() = this->ndim();
    // copy data of each parameter vector
    std::uintptr_t dptr = reinterpret_cast<std::uintptr_t>(parameters_data_ptr) + this->ndim() * sizeof(floatvec);
    void * data_ptr = reinterpret_cast<void *>(dptr);
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        data_ptr = this->parameters_[i_dim].copy_to_gpu(&(copy_on_gpu.parameters_[i_dim]), data_ptr, stream_ptr);
    }
    // copy temporary object to GPU
    ::cudaMemcpyAsync(gpu_ptr, &copy_on_gpu, sizeof(candy::Model), ::cudaMemcpyHostToDevice,
                      reinterpret_cast<::cudaStream_t>(stream_ptr));
    // nullify data pointer to avoid free data
    copy_on_gpu.parameters_.data() = nullptr;
    return data_ptr;
}

// Copy data from GPU to CPU
void * candy::Model::copy_from_gpu(candy::Model * gpu_ptr, std::uintptr_t stream_ptr) {
    // create a temporary object to get pointer to floatvec data
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    candy::Model gpu_object;
    ::cudaMemcpyAsync(&gpu_object, gpu_ptr, sizeof(candy::Model), ::cudaMemcpyDeviceToHost, stream);
    floatvec * gpu_parameter_vector_data = gpu_object.parameters_.data();
    // copy data from each parameter vector
    void * data_ptr;
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        data_ptr = this->parameters_[i_dim].copy_from_gpu(gpu_parameter_vector_data + i_dim, stream_ptr);
    }
    // avoid seg.fault due to freeing data of temporary object
    gpu_object.parameters_.assign(nullptr, 1UL);
    return data_ptr;
}

}  // namespace merlin

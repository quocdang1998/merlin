// Copyright 2023 quocdang1998
#include "merlin/candy/model.hpp"

#include <cstdint>  // std::uintptr_t

#include "merlin/utils.hpp"  // merlin::ptr_to_subsequence

namespace merlin {

// Copy data to a pre-allocated memory
void * candy::Model::copy_to_gpu(candy::Model * gpu_ptr, void * parameters_data_ptr, std::uintptr_t stream_ptr) const {
    // initialize buffer to store data of the copy before cloning it to GPU
    candy::Model copy_on_gpu;
    // shallow copy of parameters, rshape and param vectors
    double * parameters_data = reinterpret_cast<double *>(parameters_data_ptr);
    copy_on_gpu.parameters_.data() = parameters_data;
    copy_on_gpu.parameters_.size() = this->num_params();
    copy_on_gpu.rshape_ = this->rshape_;
    copy_on_gpu.ndim_ = this->ndim_;
    copy_on_gpu.rank_ = this->rank_;
    copy_on_gpu.param_vectors_.fill(nullptr);
    ptr_to_subsequence(parameters_data, this->rshape_.data(), this->ndim_, copy_on_gpu.param_vectors_.data());
    // copy data of parameters
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    ::cudaMemcpyAsync(parameters_data, this->parameters_.data(), this->num_params() * sizeof(double),
                      ::cudaMemcpyHostToDevice, stream);
    // copy temporary object to GPU
    ::cudaMemcpyAsync(gpu_ptr, &copy_on_gpu, sizeof(candy::Model), ::cudaMemcpyHostToDevice, stream);
    // nullify pointer of temporary object to avoid de-allocate GPU pointer
    copy_on_gpu.parameters_.data() = nullptr;
    return reinterpret_cast<void *>(parameters_data + this->num_params());
}

// Copy data from GPU to CPU
void * candy::Model::copy_from_gpu(double * data_from_gpu, std::uintptr_t stream_ptr) noexcept {
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    ::cudaMemcpyAsync(this->parameters_.data(), data_from_gpu, this->num_params() * sizeof(double),
                      ::cudaMemcpyDeviceToHost, stream);
    return reinterpret_cast<void *>(data_from_gpu + this->num_params());
}

}  // namespace merlin

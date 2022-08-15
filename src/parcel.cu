// Copyright 2022 quocdang1998
#include "merlin/parcel.hpp"

#include <functional>  // std::bind, std::placeholders

#include "merlin/tensor.hpp"  // merlin::Tensor
#include "merlin/utils.hpp"  // merlin::contiguous_strides, merlin::get_current_device
#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// Default constructor
Parcel::Parcel(void) {}

// Constructor from CPU array
Parcel::Parcel(const Tensor & cpu_array, uintptr_t stream) : Array(cpu_array) {
    // get device id
    cudaGetDevice(&(this->device_id_));
    // allocate data
    cudaError_t err_ = cudaMalloc(&(this->data_), sizeof(float) * this->size());
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "Memory allocation failed with message \"%s\".\n", cudaGetErrorString(err_));
    }
    // cast stream
    cudaStream_t copy_stream = reinterpret_cast<cudaStream_t>(stream);
    // reset strides vector
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    // create copy function
    auto copy_func = std::bind(cudaMemcpyAsync, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                               cudaMemcpyHostToDevice, copy_stream);
    // copy data to GPU
    array_copy(dynamic_cast<Array *>(this), dynamic_cast<const Array *>(&cpu_array), copy_func);
    // copy metadata
    this->copy_metadata();
}

// Check if current device holds data pointed by object
int Parcel::check_device(void) const {
    return (this->device_id_ - get_current_device());
}

// Copy constructor
Parcel::Parcel(const Parcel & src) : Array(src) {
    // get device id
    cudaGetDevice(&(this->device_id_));
    // reform strides vector
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    // allocate data
    cudaError_t err_ = cudaMalloc(&(this->data_), sizeof(float) * this->size());
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "Memory allocation failed with message \"%s\".\n", cudaGetErrorString(err_));
    }
    // create copy function
    auto copy_func = std::bind(cudaMemcpyPeer, std::placeholders::_1, this->device_id_,
                               std::placeholders::_2, src.device_id_, std::placeholders::_3);
    // copy data to GPU
    array_copy(dynamic_cast<Array *>(this), dynamic_cast<const Array *>(&src), copy_func);
    // copy metadata
    this->copy_metadata();
}

// Copy assignement
Parcel & Parcel::operator=(const Parcel & src) {
    // free old data
    this->free_current_data();
    // reform strides vector
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    // allocate data
    cudaError_t err_ = cudaMalloc(&(this->data_), sizeof(float) * this->size());
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "Memory allocation failed with message \"%s\".\n", cudaGetErrorString(err_));
    }
    // create copy function
    auto copy_func = std::bind(cudaMemcpyPeer, std::placeholders::_1, this->device_id_,
                               std::placeholders::_2, src.device_id_, std::placeholders::_3);
    // copy data to GPU
    array_copy(dynamic_cast<Array *>(this), dynamic_cast<const Array *>(&src), copy_func);
    // copy metadata
    this->copy_metadata();
    return *this;
}

// Move constructor
Parcel::Parcel(Parcel && src) : Array(src) {
    // move device id
    this->device_id_ = src.device_id_;
    // take over pointer to source
    src.data_ = NULL;
    // take over pointer to metadata
    this->dshape_ = src.dshape_;
    src.dshape_ = NULL;
    this->dstrides_ = src.dstrides_;
    src.dstrides_ = NULL;
}

// Move assignment
Parcel & Parcel::operator=(Parcel && src) {
    // free old data
    this->free_current_data();
    // move device id
    this->device_id_ = src.device_id_;
    // take over pointer to source
    src.data_ = NULL;
    // take over pointer to metadata
    this->dshape_ = src.dshape_;
    src.dshape_ = NULL;
    this->dstrides_ = src.dstrides_;
    src.dstrides_ = NULL;
    return *this;
}

// Slicing operator (get element at a given C contiguous index)
__device__ float & Parcel::operator[](unsigned int index) {
    if (this->check_device() != 0) {
        CUDAOUT("Expected device %d, got %d.\n", this->device_id_, get_current_device());
        return this->data_[0];
    }
    return this->data_[index];  // to be implemented !
}

// Free old data
void Parcel::free_current_data(void) {
    // save current device and set device to the corresponding GPU
    int current_device = get_current_device();
    cudaSetDevice(this->device_id_);
    // free data
    if (this->data_ != NULL) {
        cudaFree(this->data_);
        this->data_ = NULL;
    }
    // free shape vector
    if (this->dshape_ != NULL) {
        cudaFree(this->dshape_);
        this->dshape_ = NULL;
    }
    // free stride vector
    if (this->dstrides_ != NULL) {
        cudaFree(this->dstrides_);
        this->dstrides_ = NULL;
    }
    // finalize: set back the original GPU and unlock the mutex
    cudaSetDevice(current_device);
}

// Update the shape vector and strides vector on GPU memory
void Parcel::copy_metadata(void) {
    // save current device and set device to the corresponding GPU
    int current_device = get_current_device();
    cudaSetDevice(this->device_id_);
    // if not initialized, the allocate array
    if (this->dshape_ == NULL) {
        cudaMalloc(&(this->dshape_), sizeof(unsigned int)*this->ndim_);
    }
    if (this->dstrides_ == NULL) {
        cudaMalloc(&(this->dstrides_), sizeof(unsigned int)*this->ndim_);
    }
    // copy data
    cudaMemcpy(this->dshape_, this->shape_.data(), sizeof(unsigned int)*this->ndim_, cudaMemcpyHostToDevice);
    cudaMemcpy(this->dstrides_, this->strides_.data(), sizeof(unsigned int)*this->ndim_, cudaMemcpyHostToDevice);
    // set back original GPU
    cudaSetDevice(current_device);
}

// Destructor
Parcel::~Parcel(void) {
    this->free_current_data();
}

}  // namespace merlin

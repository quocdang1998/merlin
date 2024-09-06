// Copyright 2022 quocdang1998
#include "merlin/array/parcel.hpp"

#include <cinttypes>   // PRIu64
#include <functional>  // std::bind, std::placeholders
#include <utility>     // std::forward, std::move

#include "merlin/array/array.hpp"      // merlin::array::Array
#include "merlin/array/operation.hpp"  // merlin::array::contiguous_strides, merlin::array::get_leap,
                                       // merlin::array::copy, merlin::array::fill
#include "merlin/env.hpp"              // merlin::Environment
#include "merlin/logger.hpp"           // merlin::Fatal
#include "merlin/utils.hpp"            // merlin::inner_prod

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Parcel
// ---------------------------------------------------------------------------------------------------------------------

// Free old data
void array::Parcel::free_current_data(const cuda::Stream & stream) {
    // free data
    if ((this->data_ != nullptr) && this->release) {
        std::uintptr_t current_ctx = this->device().push_context();
        ::cudaFreeAsync(this->data_, reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr()));
        this->data_ = nullptr;
        cuda::Device::pop_context(current_ctx);
    }
}

// Constructor from shape vector
array::Parcel::Parcel(const Index & shape, const cuda::Stream & stream) : array::NdData(shape) {
    // allocate data
    this->release = true;
    ::cudaError_t err_ = ::cudaMallocAsync(&(this->data_), sizeof(double) * this->size(),
                                           reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr()));
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Memory allocation failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    // set device and context
    this->device_ = cuda::Device::get_current_gpu();
}

// Copy constructor
array::Parcel::Parcel(const array::Parcel & src) : array::NdData(src) {
    // get device id and context
    this->device_ = cuda::Device::get_current_gpu();
    // reform strides vector
    this->strides_ = array::contiguous_strides(this->shape_, this->ndim_, sizeof(double));
    // allocate data
    this->release = true;
    ::cudaError_t err_ = ::cudaMalloc(&(this->data_), sizeof(double) * this->size());
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Memory allocation failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    // create copy function and copy data to GPU
    if (this->device_ != src.device_) {
        auto copy_func = std::bind(::cudaMemcpyPeer, std::placeholders::_1, this->device_.id(), std::placeholders::_2,
                                   src.device_.id(), std::placeholders::_3);
        array::copy(this, &src, copy_func);
    } else {
        auto copy_func = std::bind(::cudaMemcpy, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                   ::cudaMemcpyDeviceToDevice);
        array::copy(this, &src, copy_func);
    }
}

// Copy assignement
array::Parcel & array::Parcel::operator=(const array::Parcel & src) {
    // free old data
    this->free_current_data();
    // copy metadata and reform strides vector
    this->array::NdData::operator=(src);
    this->strides_ = array::contiguous_strides(this->shape_, this->ndim_, sizeof(double));
    // get device and context
    this->device_ = cuda::Device::get_current_gpu();
    // allocate data
    this->release = true;
    cudaError_t err_ = ::cudaMalloc(&(this->data_), sizeof(double) * this->size());
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Memory allocation failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    // create copy function and copy data to GPU
    if (this->device_ != src.device_) {
        auto copy_func = std::bind(::cudaMemcpyPeer, std::placeholders::_1, this->device_.id(), std::placeholders::_2,
                                   src.device_.id(), std::placeholders::_3);
        array::copy(this, &src, copy_func);
    } else {
        auto copy_func = std::bind(::cudaMemcpy, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                   ::cudaMemcpyDeviceToDevice);
        array::copy(this, &src, copy_func);
    }
    return *this;
}

// Move constructor
array::Parcel::Parcel(array::Parcel && src) : array::NdData(std::move(src)) {
    // move device id and context
    this->device_ = src.device_;
    // take over pointer to source
    this->release = src.release;
    src.data_ = nullptr;
    src.release = false;
}

// Move assignment
array::Parcel & array::Parcel::operator=(array::Parcel && src) {
    // free old data
    this->free_current_data();
    // move device id and context
    this->device_ = src.device_;
    // copy metadata
    this->array::NdData::operator=(std::forward<array::Parcel>(src));
    // take over pointer to source
    src.data_ = nullptr;
    this->release = src.release;
    src.release = false;
    return *this;
}

// Get value of element at a n-dim index
double array::Parcel::get(const Index & index) const {
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    double result;
    std::uintptr_t current_ctx = this->device().push_context();
    ::cudaMemcpy(&result, reinterpret_cast<double *>(data_ptr), sizeof(double), ::cudaMemcpyDeviceToHost);
    cuda::Device::pop_context(current_ctx);
    return result;
}

// Get value of element at a C-contiguous index
double array::Parcel::get(std::uint64_t index) const {
    std::uint64_t leap = array::get_leap(index, this->shape_, this->strides_, this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    double result;
    std::uintptr_t current_ctx = this->device().push_context();
    ::cudaMemcpy(&result, reinterpret_cast<double *>(data_ptr), sizeof(double), ::cudaMemcpyDeviceToHost);
    cuda::Device::pop_context(current_ctx);
    return result;
}

// Set value of element at a n-dim index
void array::Parcel::set(const Index & index, double value) {
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    std::uintptr_t current_ctx = this->device().push_context();
    ::cudaMemcpy(reinterpret_cast<double *>(data_ptr), &value, sizeof(double), ::cudaMemcpyHostToDevice);
    cuda::Device::pop_context(current_ctx);
}

// Set value of element at a C-contiguous index
void array::Parcel::set(std::uint64_t index, double value) {
    std::uint64_t leap = array::get_leap(index, this->shape_, this->strides_, this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    std::uintptr_t current_ctx = this->device().push_context();
    ::cudaMemcpy(reinterpret_cast<double *>(data_ptr), &value, sizeof(double), ::cudaMemcpyHostToDevice);
    cuda::Device::pop_context(current_ctx);
}

// Set value of all elements
void array::Parcel::fill(double value) {
    auto copy_func = std::bind(::cudaMemcpy, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                               ::cudaMemcpyHostToDevice);
    array::fill(this, value, copy_func);
}

// Calculate mean and variance of all non-zero and finite elements
std::array<double, 2> array::Parcel::get_mean_variance(void) const {
    auto copy_func = std::bind(::cudaMemcpy, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                               ::cudaMemcpyDeviceToHost);
    return array::stat(this, copy_func);
}

// Transfer data to GPU
void array::Parcel::transfer_data_to_gpu(const array::Array & cpu_array, const cuda::Stream & stream) {
    // get device id
    stream.check_cuda_context();
    // cast stream
    ::cudaStream_t copy_stream = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    // create copy function
    auto copy_func = std::bind(::cudaMemcpyAsync, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                               ::cudaMemcpyHostToDevice, copy_stream);
    // copy data to GPU
    array::copy(this, &cpu_array, copy_func);
}

// Copy data to a pre-allocated memory
void * array::Parcel::copy_to_gpu(array::Parcel * gpu_ptr, void * data_ptr, std::uintptr_t stream_ptr) const {
    // directly copy data to GPU
    ::cudaMemcpyAsync(gpu_ptr, this, sizeof(array::Parcel), ::cudaMemcpyHostToDevice,
                      reinterpret_cast<::cudaStream_t>(stream_ptr));
    return data_ptr;
}

// Destructor
array::Parcel::~Parcel(void) { this->free_current_data(); }

}  // namespace merlin

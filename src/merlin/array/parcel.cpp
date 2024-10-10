// Copyright 2022 quocdang1998
#include "merlin/array/parcel.hpp"

#include <functional>  // std::bind, std::placeholders
#include <utility>     // std::forward, std::move

#include "merlin/array/array.hpp"      // merlin::array::Array
#include "merlin/array/operation.hpp"  // merlin::array::contiguous_strides, merlin::array::get_leap,
                                       // merlin::array::copy, merlin::array::fill, merlin::array::print
#include "merlin/cuda/device.hpp"      // merlin::cuda::CtxGuard
#include "merlin/logger.hpp"           // merlin::Fatal
#include "merlin/memory.hpp"           // merlin::mem_alloc_device, merlin::memcpy_gpu, merlin::memcpy_peer_gpu,
                                       // merlin::memcpy_gpu_to_cpu,  merlin::memcpy_cpu_to_gpu,
                                       // merlin::mem_free_device_noexcept
#include "merlin/utils.hpp"            // merlin::inner_prod

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Parcel
// ---------------------------------------------------------------------------------------------------------------------

// String representation
std::string array::Parcel::str(bool first_call) const { return array::print(this, "Parcel", first_call); }

// Constructor from shape vector
array::Parcel::Parcel(const Index & shape, const cuda::Stream & stream) : array::NdData(shape) {
    // set device and context
    cuda::Device gpu = stream.get_gpu();
    cuda::CtxGuard guard(gpu);
    this->device_ = gpu;
    this->release = true;
    // allocate data
    // this->data_ = reinterpret_cast<double *>(mem_alloc_device(sizeof(double) * this->size(), stream.get_stream_ptr()));
    mem_alloc_device(reinterpret_cast<void **>(&(this->data_)), sizeof(double) * this->size(), stream.get_stream_ptr());
}

// Copy constructor
array::Parcel::Parcel(const array::Parcel & src) : array::NdData(src) {
    // set device and context
    this->device_ = cuda::Device::get_current_gpu();
    cuda::CtxGuard guard(this->device_);
    // calculate stride and initialize attributes
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(double));
    this->release = true;
    // allocate data
    // this->data_ = reinterpret_cast<double *>(mem_alloc_device(sizeof(double) * this->size(), 0));
    mem_alloc_device(reinterpret_cast<void **>(&(this->data_)), sizeof(double) * this->size(), 0);
    // create copy function and copy data to GPU
    if (this->device_ != src.device_) {
        auto copy_func = std::bind(memcpy_peer_gpu, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                   this->device_.id(), src.device_.id(), 0);
        array::copy(this, &src, copy_func);
    } else {
        auto copy_func = std::bind(memcpy_gpu, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, 0);
        array::copy(this, &src, copy_func);
    }
}

// Copy assignement
array::Parcel & array::Parcel::operator=(const array::Parcel & src) {
    // free old data
    this->free_current_data();
    // set device and context
    this->device_ = cuda::Device::get_current_gpu();
    cuda::CtxGuard guard(this->device_);
    // copy metadata and calculate strides vector
    this->array::NdData::operator=(src);
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(double));
    this->release = true;
    // allocate data
    // this->data_ = reinterpret_cast<double *>(mem_alloc_device(sizeof(double) * this->size(), 0));
    mem_alloc_device(reinterpret_cast<void **>(&(this->data_)), sizeof(double) * this->size(), 0);
    // create copy function and copy data to GPU
    if (this->device_ != src.device_) {
        auto copy_func = std::bind(memcpy_peer_gpu, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                   this->device_.id(), src.device_.id(), 0);
        array::copy(this, &src, copy_func);
    } else {
        auto copy_func = std::bind(memcpy_gpu, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, 0);
        array::copy(this, &src, copy_func);
    }
    return *this;
}

// Move constructor
array::Parcel::Parcel(array::Parcel && src) : array::NdData(std::move(src)) {
    // move metadata
    this->device_ = src.device_;
    this->release = src.release;
    // take over source data
    src.data_ = nullptr;
    src.release = false;
}

// Move assignment
array::Parcel & array::Parcel::operator=(array::Parcel && src) {
    // free old data
    this->free_current_data();
    // move metadata
    this->device_ = src.device_;
    this->array::NdData::operator=(std::forward<array::Parcel>(src));
    this->release = src.release;
    // take over source data
    src.data_ = nullptr;
    src.release = false;
    return *this;
}

// Get value of element at a n-dim index
double array::Parcel::get(const Index & index) const {
    cuda::CtxGuard guard(this->device_);
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    double result;
    memcpy_gpu_to_cpu(&result, reinterpret_cast<double *>(data_ptr), sizeof(double));
    return result;
}

// Get value of element at a C-contiguous index
double array::Parcel::get(std::uint64_t index) const {
    cuda::CtxGuard guard(this->device_);
    std::uint64_t leap = array::get_leap(index, this->shape_.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    double result;
    memcpy_gpu_to_cpu(&result, reinterpret_cast<double *>(data_ptr), sizeof(double));
    return result;
}

// Set value of element at a n-dim index
void array::Parcel::set(const Index & index, double value) {
    cuda::CtxGuard guard(this->device_);
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    memcpy_cpu_to_gpu(reinterpret_cast<double *>(data_ptr), &value, sizeof(double));
}

// Set value of element at a C-contiguous index
void array::Parcel::set(std::uint64_t index, double value) {
    cuda::CtxGuard guard(this->device_);
    std::uint64_t leap = array::get_leap(index, this->shape_.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    memcpy_cpu_to_gpu(reinterpret_cast<double *>(data_ptr), &value, sizeof(double));
}

// Set value of all elements
void array::Parcel::fill(double value) {
    auto copy_func = std::bind(memcpy_cpu_to_gpu, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                               0);
    array::fill(this, value, copy_func);
}

// Calculate mean and variance of all non-zero and finite elements
std::array<double, 2> array::Parcel::get_mean_variance(void) const {
    auto copy_func = std::bind(memcpy_gpu_to_cpu, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                               0);
    return array::stat(this, copy_func);
}

// Transfer data to GPU
void array::Parcel::transfer_data_to_gpu(const array::Array & cpu_array, const cuda::Stream & stream) {
    // check GPU
    if (this->device_ != stream.get_gpu()) {
        Fatal<cuda_runtime_error>("GPU of the stream and the destination is not the same.\n");
    }
    // copy data to GPU
    auto copy_func = std::bind(memcpy_cpu_to_gpu, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                               stream.get_stream_ptr());
    array::copy(this, &cpu_array, copy_func);
}

// Copy data to a pre-allocated memory
void * array::Parcel::copy_to_gpu(array::Parcel * gpu_ptr, void * data_ptr, std::uintptr_t stream_ptr) const {
    memcpy_cpu_to_gpu(gpu_ptr, this, sizeof(array::Parcel), stream_ptr);
    return data_ptr;
}

// Free data
void array::Parcel::free_current_data(const cuda::Stream & stream) noexcept {
    if ((this->data_ != nullptr) && this->release) {
        cuda::CtxGuard guard(this->device_);
        mem_free_device_noexcept(this->data_, stream.get_stream_ptr());
        this->data_ = nullptr;
    }
}

// Destructor
array::Parcel::~Parcel(void) { this->free_current_data(); }

}  // namespace merlin

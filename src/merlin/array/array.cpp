// Copyright 2022 quocdang1998
#include "merlin/array/array.hpp"

#include <cstring>       // std::memcpy
#include <functional>    // std::bind, std::placeholders
#include <shared_mutex>  // std::shared_lock
#include <utility>       // std::forward, std::move

#include "merlin/array/operation.hpp"  // merlin::array::contiguous_strides, merlin::array::get_leap,
                                       // merlin::array::copy, merlin::array::fill, merlin::array::print
#include "merlin/array/parcel.hpp"     // merlin::array::Parcel
#include "merlin/array/stock.hpp"      // merlin::array::Stock
#include "merlin/cuda/device.hpp"      // merlin::cuda::CtxGuard
#include "merlin/io/io_engine.hpp"     // merlin::io::ReadEngine
#include "merlin/logger.hpp"           // merlin::Fatal, merlin::cuda_runtime_error
#include "merlin/memory.hpp"           // merlin::mem_alloc_host, merlin::mem_free_host, merlin::mem_register_host,
                                       // merlin::mem_unregister_host, merlin::memcpy_gpu_to_cpu
#include "merlin/utils.hpp"            // merlin::inner_prod

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Array
// ---------------------------------------------------------------------------------------------------------------------

// Construct Array from Numpy array
array::Array::Array(double * data, const Index & shape, const Index & strides, bool copy, bool pin_memory) :
array::NdData(data, shape, strides) {
    // copy or assign data
    this->release = copy;
    if (copy) {
        // allocate a new tensor
        this->data_ = reinterpret_cast<double *>(mem_alloc_host(this->size() * sizeof(double)));
        this->is_pinned = false;
        // reform the stride tensor (force into C shape)
        this->strides_ = array::contiguous_strides(this->shape_, sizeof(double));
        // copy data from old tensor to new tensor (optimized with memcpy)
        array::NdData src(data, shape, strides);
        array::copy(this, &src, std::memcpy);
    } else {
        // assign strides and data pointer
        std::copy(strides.begin(), strides.end(), this->strides_.begin());
        this->data_ = data;
        // pin memory
        std::uint64_t last_elem = array::get_leap(this->size_ - 1, this->shape_.data(), this->strides_.data(),
                                                  this->shape_.size());
        if (pin_memory) {
            this->is_pinned = mem_register_host(reinterpret_cast<void *>(this->data_), last_elem + sizeof(double));
        }
    }
}

// Constructor from shape vector
array::Array::Array(const Index & shape) : array::NdData(shape) {
    // initialize data
    this->data_ = reinterpret_cast<double *>(mem_alloc_host(this->size() * sizeof(double)));
    // other meta data
    this->release = true;
}

// Copy constructor
array::Array::Array(const array::Array & src) : array::NdData(src) {
    // copy / initialize meta data
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(double));
    this->release = true;
    // copy data
    this->data_ = reinterpret_cast<double *>(mem_alloc_host(this->size() * sizeof(double)));
    array::copy(this, &src, std::memcpy);
}

// Copy assignment
array::Array & array::Array::operator=(const array::Array & src) {
    // check for self assignment
    if (this == &src) {
        return *this;
    }
    // copy / initialize meta data
    this->array::NdData::operator=(src);
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(double));
    // free current data
    if (this->data_ != nullptr) {
        if (this->is_pinned) {
            mem_unregister_host(reinterpret_cast<void *>(this->data_));
        }
        if (this->release) {
            mem_free_host(reinterpret_cast<void *>(this->data_));
        }
    }
    this->release = true;
    // copy data
    this->data_ = reinterpret_cast<double *>(mem_alloc_host(this->size() * sizeof(double)));
    array::copy(this, &src, std::memcpy);
    return *this;
}

// Move constructor
array::Array::Array(array::Array && src) : array::NdData(std::move(src)) {
    // disable release of the source
    this->release = src.release;
    src.release = false;
    // nullify source data
    src.data_ = nullptr;
}

// Move assignment
array::Array & array::Array::operator=(array::Array && src) {
    // free current data
    if (this->data_ != nullptr) {
        if (this->is_pinned) {
            mem_unregister_host(reinterpret_cast<void *>(this->data_));
        }
        if (this->release) {
            mem_free_host(reinterpret_cast<void *>(this->data_));
        }
    }
    // copy meta data
    this->array::NdData::operator=(std::forward<array::Array>(src));
    this->release = src.release;
    src.release = false;
    // move data
    src.data_ = nullptr;
    return *this;
}

// Get reference to element at a given ndim index
double & array::Array::operator[](const Index & index) {
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<double *>(data_ptr));
}

// Get reference to element at a given C-contiguous index
double & array::Array::operator[](std::uint64_t index) {
    std::uint64_t leap = array::get_leap(index, this->shape_.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<double *>(data_ptr));
}

// Get constant reference to element at a given ndim index
const double & array::Array::operator[](const Index & index) const {
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<const double *>(data_ptr));
}

// Get const reference to element at a given C-contiguous index
const double & array::Array::operator[](std::uint64_t index) const {
    std::uint64_t leap = array::get_leap(index, this->shape_.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<const double *>(data_ptr));
}

// Get value of element at a n-dim index
double array::Array::get(const Index & index) const {
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<double *>(data_ptr));
}

// Get value of element at a C-contiguous index
double array::Array::get(std::uint64_t index) const {
    std::uint64_t leap = array::get_leap(index, this->shape_.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<double *>(data_ptr));
}

// Set value of element at a n-dim index
void array::Array::set(const Index & index, double value) {
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    *(reinterpret_cast<double *>(data_ptr)) = value;
}

// Set value of element at a C-contiguous index
void array::Array::set(std::uint64_t index, double value) {
    std::uint64_t leap = array::get_leap(index, this->shape_.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    *(reinterpret_cast<double *>(data_ptr)) = value;
}

// Set value of all elements
void array::Array::fill(double value) { array::fill(this, value, std::memcpy); }

// Calculate mean and variance of all non-zero and finite elements
std::array<double, 2> array::Array::get_mean_variance(void) const { return array::stat(this, std::memcpy); }

// Copy data from GPU array
void array::Array::clone_data_from_gpu(const array::Parcel & src, const cuda::Stream & stream) {
    // check GPU
    if (src.get_gpu() != stream.get_gpu()) {
        Fatal<cuda_runtime_error>("GPU of the stream and the source data is not the same.\n");
    }
    // copy data from GPU to CPU
    cuda::CtxGuard guard(src.get_gpu());
    auto copy_func = std::bind(memcpy_gpu_to_cpu, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                               stream.get_stream_ptr());
    array::copy(this, &src, copy_func);
}

// Export data to a file
void array::Array::extract_data_from_file(const array::Stock & src) {
    std::shared_lock<io::FileLock> lock = ((src.is_thread_safe()) ? std::shared_lock<io::FileLock>(src.get_file_lock())
                                                                  : std::shared_lock<io::FileLock>());
    io::ReadEngine<double> reader(src.get_file_ptr());
    array::copy(this, &src, reader);
}

// String representation
std::string array::Array::str(bool first_call) const { return array::print(this, "Array", first_call); }

// Destructor
array::Array::~Array(void) {
    if (this->data_ != nullptr) {
        if (this->is_pinned) {
            mem_unregister_host(reinterpret_cast<void *>(this->data_));
        }
        if (this->release) {
            mem_free_host(reinterpret_cast<void *>(this->data_));
        }
    }
}

}  // namespace merlin

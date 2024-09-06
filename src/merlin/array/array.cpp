// Copyright 2022 quocdang1998
#include "merlin/array/array.hpp"

#include <cinttypes>     // PRIu64
#include <cstdio>        // std::fread, std::fseek
#include <cstring>       // std::memcpy
#include <functional>    // std::bind, std::placeholders
#include <ios>           // std::ios_base::failure
#include <shared_mutex>  // std::shared_lock
#include <utility>       // std::forward, std::move

#include "merlin/array/operation.hpp"  // merlin::array::contiguous_strides, merlin::array::get_leap,
                                       // merlin::array::copy, merlin::array::fill, merlin::array::print
#include "merlin/array/parcel.hpp"     // merlin::array::Parcel
#include "merlin/array/stock.hpp"      // merlin::array::Stock
#include "merlin/logger.hpp"           // merlin::Fatal
#include "merlin/utils.hpp"            // merlin::inner_prod

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Memory lock (allocated array always stays in the RAM)
// ---------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__

// Allocate non pageable memory
double * array::allocate_memory(std::uint64_t size) {
    double * result = new double[size];
    if (result == nullptr) {
        Fatal<std::runtime_error>("Cannot allocate memory.\n");
    }
    return result;
}

// Pin memory to RAM
void array::cuda_pin_memory(double * ptr, std::uint64_t mem_size) {}

// Free non pageable memory
void array::free_memory(double * ptr) { delete[] ptr; }

#endif  // __MERLIN_CUDA__

// ---------------------------------------------------------------------------------------------------------------------
// Read data
// ---------------------------------------------------------------------------------------------------------------------

// Read an array from file
static inline void read_from_file(void * dest, std::FILE * file, const void * src, std::uint64_t bytes,
                                  bool same_endianess) {
    std::fseek(file, reinterpret_cast<std::uintptr_t>(src), SEEK_SET);
    std::uint64_t count = bytes / sizeof(double);
    if (std::fread(dest, sizeof(double), count, file) != count) {
        Fatal<std::ios_base::failure>("Read file error.\n");
    }
    if (!same_endianess) {
        flip_range(reinterpret_cast<std::uint64_t *>(dest), count);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Array
// ---------------------------------------------------------------------------------------------------------------------

// Construct Array from Numpy array
array::Array::Array(double * data, const UIntVec & shape, const UIntVec & strides, bool copy, bool pin_memory) :
array::NdData(data, shape, strides) {
    // copy or assign data
    this->release = copy;
    if (copy) {
        // allocate a new tensor
        this->data_ = array::allocate_memory(this->size());
        // reform the stride tensor (force into C shape)
        this->strides_ = array::contiguous_strides(this->shape_, this->ndim_, sizeof(double));
        // copy data from old tensor to new tensor (optimized with memcpy)
        array::NdData src(data, shape, strides);
        array::copy(this, &src, std::memcpy);
    } else {
        // assign strides and data pointer
        std::copy(strides.begin(), strides.end(), this->strides_.begin());
        this->data_ = data;
        // pin memory
        std::uint64_t last_elem = array::get_leap(this->size_ - 1, this->shape_, this->strides_, this->ndim_);
        if (pin_memory) {
            array::cuda_pin_memory(this->data_, last_elem + sizeof(double));
        }
    }
}

// Constructor from shape vector
array::Array::Array(const Index & shape) : array::NdData(shape) {
    // initialize data
    this->data_ = array::allocate_memory(this->size());
    // other meta data
    this->release = true;
}

// Copy constructor
array::Array::Array(const array::Array & src) : array::NdData(src) {
    // copy / initialize meta data
    this->strides_ = array::contiguous_strides(this->shape_, this->ndim_, sizeof(double));
    this->release = true;
    // copy data
    this->data_ = array::allocate_memory(this->size());
    array::copy(this, &src, std::memcpy);
}

// Copy assignment
array::Array & array::Array::operator=(const array::Array & src) {
    // copy / initialize meta data
    this->array::NdData::operator=(src);
    this->strides_ = array::contiguous_strides(this->shape_, this->ndim_, sizeof(double));
    // free current data
    if (this->release) {
        array::free_memory(this->data_);
    }
    this->release = true;
    // copy data
    this->data_ = array::allocate_memory(this->size());
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
    // disable release of the source and free current data
    if (this->release) {
        array::free_memory(this->data_);
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
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<double *>(data_ptr));
}

// Get reference to element at a given C-contiguous index
double & array::Array::operator[](std::uint64_t index) {
    std::uint64_t leap = array::get_leap(index, this->shape_, this->strides_, this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<double *>(data_ptr));
}

// Get constant reference to element at a given ndim index
const double & array::Array::operator[](const Index & index) const {
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<const double *>(data_ptr));
}

// Get const reference to element at a given C-contiguous index
const double & array::Array::operator[](std::uint64_t index) const {
    std::uint64_t leap = array::get_leap(index, this->shape_, this->strides_, this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<const double *>(data_ptr));
}

// Get value of element at a n-dim index
double array::Array::get(const Index & index) const {
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<double *>(data_ptr));
}

// Get value of element at a C-contiguous index
double array::Array::get(std::uint64_t index) const {
    std::uint64_t leap = array::get_leap(index, this->shape_, this->strides_, this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<double *>(data_ptr));
}

// Set value of element at a n-dim index
void array::Array::set(const Index & index, double value) {
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    *(reinterpret_cast<double *>(data_ptr)) = value;
}

// Set value of element at a C-contiguous index
void array::Array::set(std::uint64_t index, double value) {
    std::uint64_t leap = array::get_leap(index, this->shape_, this->strides_, this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    *(reinterpret_cast<double *>(data_ptr)) = value;
}

// Set value of all elements
void array::Array::fill(double value) { array::fill(this, value, std::memcpy); }

// Calculate mean and variance of all non-zero and finite elements
std::array<double, 2> array::Array::get_mean_variance(void) const { return array::stat(this, std::memcpy); }

// Copy data from GPU array
#ifndef __MERLIN_CUDA__
void array::Array::clone_data_from_gpu(const array::Parcel & src, const cuda::Stream & stream) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}
#endif  // __MERLIN_CUDA__

// Export data to a file
void array::Array::extract_data_from_file(const array::Stock & src) {
    auto read_func = std::bind(read_from_file, std::placeholders::_1, src.get_file_ptr(), std::placeholders::_2,
                               std::placeholders::_3, src.is_same_endianess());
    std::shared_lock<FileLock> lock = ((src.is_thread_safe()) ? std::shared_lock<FileLock>(src.get_file_lock())
                                                              : std::shared_lock<FileLock>());
    array::copy(this, &src, read_func);
}

// String representation
std::string array::Array::str(bool first_call) const { return array::print(this, "Array", first_call); }

// Destructor
array::Array::~Array(void) {
    if (this->release && (this->data_ != nullptr)) {
        array::free_memory(this->data_);
    }
}

}  // namespace merlin

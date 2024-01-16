// Copyright 2022 quocdang1998
#include "merlin/array/array.hpp"

#include <cstdio>      // std::fread, std::fseek
#include <cstring>     // std::memcpy
#include <functional>  // std::bind, std::placeholders
#include <ios>         // std::ios_base::failure
#include <mutex>       // std::mutex

#include "merlin/array/operation.hpp"  // merlin::array::contiguous_strides, merlin::array::get_leap,
                                       // merlin::array::copy, merlin::array::fill, merlin::array::print
#include "merlin/array/parcel.hpp"     // merlin::array::Parcel
#include "merlin/array/stock.hpp"      // merlin::array::Stock
#include "merlin/logger.hpp"           // FAILURE
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
        FAILURE(std::runtime_error, "Cannot allocate memory.\n");
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
static inline void read_from_file(double * dest, std::FILE * file, double * src, std::uint64_t bytes) {
    std::fseek(file, reinterpret_cast<std::uintptr_t>(src), SEEK_SET);
    std::uint64_t count = bytes / sizeof(double);
    if (std::fread(dest, sizeof(double), count, file) != count) {
        FAILURE(std::ios_base::failure, "Read file error.\n");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Array
// ---------------------------------------------------------------------------------------------------------------------

// Constructor Array of one element
array::Array::Array(double value) {
    // allocate data
    this->data_ = array::allocate_memory(1);
    this->data_[0] = value;

    // set metadata
    this->size_ = 1;
    this->strides_ = intvec({sizeof(double)});
    this->shape_ = intvec({1});
    this->release = true;
}

// Construct empty Array from shape vector
array::Array::Array(const intvec & shape) : array::NdData(shape) {
    // initialize data
    this->data_ = array::allocate_memory(this->size());
    // other meta data
    this->release = true;
}

// Construct Array from Numpy array
array::Array::Array(double * data, const intvec & shape, const intvec & strides, bool copy) {
    this->shape_ = shape;
    this->calc_array_size();
    this->release = copy;
    // copy / assign data
    if (copy) {  // copy data
        // allocate a new tensor
        this->data_ = array::allocate_memory(this->size());
        // reform the stride tensor (force into C shape)
        this->strides_ = array::contiguous_strides(this->shape_, sizeof(double));
        // copy data from old tensor to new tensor (optimized with memcpy)
        array::NdData src(data, shape, strides);
        array::copy(dynamic_cast<array::NdData *>(this), &src, std::memcpy);
    } else {
        // assign strides and data pointer
        this->strides_ = strides;
        this->data_ = data;
        // pin memory
        std::uint64_t last_elem = array::get_leap(this->size_ - 1, shape, strides);
        array::cuda_pin_memory(this->data_, last_elem + sizeof(double));
    }
}

// Constructor from a slice
array::Array::Array(const array::Array & whole, const slicevec & slices) : array::NdData(whole, slices) {
    this->release = false;
}

// Copy constructor
array::Array::Array(const array::Array & src) : array::NdData(src) {
    // copy / initialize meta data
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(double));
    this->release = true;
    // copy data
    this->data_ = array::allocate_memory(this->size());
    array::copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&src), std::memcpy);
}

// Copy assignment
array::Array & array::Array::operator=(const array::Array & src) {
    // copy / initialize meta data
    this->array::NdData::operator=(src);
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(double));
    // free current data
    if (this->release) {
        delete[] this->data_;
    }
    this->release = true;
    // copy data
    this->data_ = array::allocate_memory(this->size());
    array::copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&src), std::memcpy);
    return *this;
}

// Move constructor
array::Array::Array(array::Array && src) : array::NdData(src) {
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
    this->array::NdData::operator=(src);
    this->release = src.release;
    src.release = false;
    // move data
    src.data_ = nullptr;
    return *this;
}

// Get reference to element at a given ndim index
double & array::Array::operator[](const intvec & index) {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<double *>(data_ptr));
}

// Get reference to element at a given C-contiguous index
double & array::Array::operator[](std::uint64_t index) {
    std::uint64_t leap = array::get_leap(index, this->shape_, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<double *>(data_ptr));
}

// Get constant reference to element at a given ndim index
const double & array::Array::operator[](const intvec & index) const {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<const double *>(data_ptr));
}

// Get const reference to element at a given C-contiguous index
const double & array::Array::operator[](std::uint64_t index) const {
    std::uint64_t leap = array::get_leap(index, this->shape_, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<const double *>(data_ptr));
}

// Get value of element at a n-dim index
double array::Array::get(const intvec & index) const {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<double *>(data_ptr));
}

// Get value of element at a C-contiguous index
double array::Array::get(std::uint64_t index) const {
    std::uint64_t leap = array::get_leap(index, this->shape_, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<double *>(data_ptr));
}

// Set value of element at a n-dim index
void array::Array::set(const intvec index, double value) {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    *(reinterpret_cast<double *>(data_ptr)) = value;
}

// Set value of element at a C-contiguous index
void array::Array::set(std::uint64_t index, double value) {
    std::uint64_t leap = array::get_leap(index, this->shape_, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    *(reinterpret_cast<double *>(data_ptr)) = value;
}

// Set value of all elements
void array::Array::fill(double value) {
    array::fill(this, value, std::memcpy);
}

// Copy data from GPU array
#ifndef __MERLIN_CUDA__
void array::Array::clone_data_from_gpu(const array::Parcel & src, const cuda::Stream & stream) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}
#endif  // __MERLIN_CUDA__

// Export data to a file
void array::Array::extract_data_from_file(const array::Stock & src) {
    auto read_func = std::bind(read_from_file, std::placeholders::_1, src.get_file_ptr(), std::placeholders::_2,
                               std::placeholders::_3);
    if (src.is_thread_safe()) {
        src.get_file_lock().lock();
    }
    array::copy(this, &src, read_func);
    if (src.is_thread_safe()) {
        src.get_file_lock().unlock();
    }
}

// String representation
std::string array::Array::str(bool first_call) const {
    return array::print(this, "Array", first_call);
}

// Destructor
array::Array::~Array(void) {
    if (this->release && (this->data_ != nullptr)) {
        array::free_memory(this->data_);
    }
}

}  // namespace merlin
